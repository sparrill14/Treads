"""
Hybrid PPO Training for Treads.

Uses TypeScript rollout worker for fast in-process simulation + NN inference,
with Python handling only PPO gradient updates. This eliminates per-tick IPC.

Protocol:
  1. Extract weights from SB3 PPO model → send to Node.js worker
  2. Worker runs simulation with TS MLP inference → collects n_steps transitions
  3. Worker returns bulk rollout data (~5MB JSON)
  4. Python populates SB3 RolloutBuffer → runs PPO.train()
  5. Repeat
"""

import os
import sys
import csv
import json
import time
import socket
import signal
import subprocess
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, cast
import grpc
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(__file__))

from sb3_compat import ensure_pickle_compat
from export_onnx import export_to_onnx
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from runtime_codec import decode_multi_discrete_action
from treads_env import OBS_SIZE, TreadsEnv
from model_contract import (
    ACTION_HEAD_SIZES,
    MODEL_CONTRACT,
    assert_contract_compatible,
    write_contract_for_model,
)

ensure_pickle_compat()


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance (reward normalization)."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: NDArray[np.float32]) -> None:
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.mean = new_mean
        self.count = tot_count

    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "RunningMeanStd":
        rms = cls()
        rms.mean = float(d.get("mean", 0.0))
        rms.var = float(d.get("var", 1.0))
        rms.count = float(d.get("count", 1e-4))
        return rms


def linear_schedule_between(start: float, end: float) -> Callable[[float], float]:
    """Schedule where progress=1 -> start and progress=0 -> end."""
    def _schedule(progress_remaining: float) -> float:
        p = min(1.0, max(0.0, float(progress_remaining)))
        return end + (start - end) * p
    return _schedule

# Path to compiled rollout worker
ROLLOUT_WORKER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "rollout-worker.js"
)
PROTO_PATH = os.path.join(os.path.dirname(__file__), "proto", "treads.proto")
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "_generated")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_proto_stubs() -> Tuple[Any, Any]:
    os.makedirs(GENERATED_DIR, exist_ok=True)
    if GENERATED_DIR not in sys.path:
        sys.path.insert(0, GENERATED_DIR)

    pb2_path = os.path.join(GENERATED_DIR, "treads_pb2.py")
    pb2_grpc_path = os.path.join(GENERATED_DIR, "treads_pb2_grpc.py")
    proto_mtime = os.path.getmtime(PROTO_PATH)
    stubs_stale = not (os.path.exists(pb2_path) and os.path.exists(pb2_grpc_path)) or any(
        os.path.getmtime(path) < proto_mtime for path in (pb2_path, pb2_grpc_path) if os.path.exists(path)
    )
    if stubs_stale:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{os.path.dirname(PROTO_PATH)}",
                f"--python_out={GENERATED_DIR}",
                f"--grpc_python_out={GENERATED_DIR}",
                PROTO_PATH,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to generate protobuf stubs. "
                f"stdout={result.stdout[-1000:]} stderr={result.stderr[-1000:]}"
            )

    import treads_pb2  # type: ignore[import-not-found]
    import treads_pb2_grpc  # type: ignore[import-not-found]

    return treads_pb2, treads_pb2_grpc


TREADS_PB2, TREADS_PB2_GRPC = _ensure_proto_stubs()

# ---- Curriculum configuration ----
DEFAULT_CURRICULUM: List[Dict[str, Any]] = [
    # ── Phase 1-3: Fundamentals (mastered quickly) ──
    {"name": "Phase 1 (aim + track)", "scenario_ids": [111, 112, 114], "min_phase_episodes": 1200, "force_phase_episodes": 4800, "required_win_rate": 0.50, "player_max_ammo": 2},
    {"name": "Phase 2 (dodge + fight)", "scenario_ids": [121, 122, 123], "min_phase_episodes": 1600, "force_phase_episodes": 5600, "required_win_rate": 0.45, "player_max_ammo": 2},
    {"name": "Phase 2.5 (bridge to full turret)", "scenario_ids": [122, 123, 131, 132], "min_phase_episodes": 1400, "force_phase_episodes": 5600, "required_win_rate": 0.45, "player_max_ammo": 2},
    {"name": "Phase 3 (full turret)", "scenario_ids": [131, 132, 133], "min_phase_episodes": 1800, "force_phase_episodes": 6400, "required_win_rate": 0.45, "player_max_ammo": 3},
    # ── Phase 3→4: Obstacle introduction (overlap both neighbors) ──
    {"name": "Phase 3.5 (bridge to obstacles)", "scenario_ids": [132, 133, 141], "min_phase_episodes": 1600, "force_phase_episodes": 6400, "required_win_rate": 0.45, "player_max_ammo": 3},
    {"name": "Phase 4 (obstacles)", "scenario_ids": [141, 142, 143], "min_phase_episodes": 2000, "force_phase_episodes": 8000, "required_win_rate": 0.42, "player_max_ammo": 3},
    # ── Phase 4→5: Multi-enemy ramp (gentle bridge via 145-150, then full) ──
    # Every phase includes at least 1 obstacle scenario to prevent forgetting
    {"name": "Phase 4.5a (unarmed 2nd target)", "scenario_ids": [142, 143, 145, 146], "min_phase_episodes": 3000, "force_phase_episodes": 13000, "required_win_rate": 0.42, "player_max_ammo": 4},
    {"name": "Phase 4.5b (armed multi open)", "scenario_ids": [143, 146, 147, 148, 149], "min_phase_episodes": 3600, "force_phase_episodes": 16000, "required_win_rate": 0.40, "player_max_ammo": 4},
    {"name": "Phase 4.5c (multi + obstacles)", "scenario_ids": [148, 149, 150, 151, 152], "min_phase_episodes": 4200, "force_phase_episodes": 18000, "required_win_rate": 0.38, "player_max_ammo": 4},
    {"name": "Phase 5 (multi-enemy full)", "scenario_ids": [150, 151, 152, 153], "min_phase_episodes": 4500, "force_phase_episodes": 20000, "required_win_rate": 0.36, "player_max_ammo": 4},
    # ── Phase 5→6: Bouncing shots introduction ──
    {"name": "Phase 5.5 (bridge to bouncing)", "scenario_ids": [152, 153, 161], "min_phase_episodes": 2800, "force_phase_episodes": 12000, "required_win_rate": 0.36, "player_max_ammo": 4},
    {"name": "Phase 6 (bouncing shots)", "scenario_ids": [161, 162, 163], "min_phase_episodes": 3200, "force_phase_episodes": 13000, "required_win_rate": 0.36, "player_max_ammo": 4},
    # ── Phase 6→7: Bomber introduction ──
    {"name": "Phase 6.5 (bridge to bombers)", "scenario_ids": [162, 163, 171], "min_phase_episodes": 2800, "force_phase_episodes": 12000, "required_win_rate": 0.35, "player_max_ammo": 5},
    {"name": "Phase 7 (bombers)", "scenario_ids": [171, 172, 173], "min_phase_episodes": 3200, "force_phase_episodes": 13000, "required_win_rate": 0.35, "player_max_ammo": 5},
    # ── Phase 7→8: Full mix ──
    {"name": "Phase 7.5 (bridge to full mix)", "scenario_ids": [172, 173, 181], "min_phase_episodes": 3000, "force_phase_episodes": 13000, "required_win_rate": 0.34, "player_max_ammo": 5},
    {"name": "Phase 8 (full mix)", "scenario_ids": [181, 182, 183], "min_phase_episodes": 3600, "force_phase_episodes": 14000, "required_win_rate": 0.34, "player_max_ammo": 5},
    # ── Real game levels ──
    {"name": "Phase 9 (easy levels)", "scenario_ids": [1, 2, 3, 4], "min_phase_episodes": 2800, "force_phase_episodes": 11000, "required_win_rate": 0.33, "player_max_ammo": 5},
    {"name": "Phase 10 (mid levels)", "scenario_ids": [1, 2, 3, 4, 5, 6], "min_phase_episodes": 3200, "force_phase_episodes": 12000, "required_win_rate": 0.32, "player_max_ammo": 5},
    {"name": "Phase 11 (all levels)", "scenario_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9], "min_phase_episodes": None, "required_win_rate": 0.30, "player_max_ammo": 5},
]


class HybridTrainer:
    """Orchestrates hybrid TS rollout collection + Python PPO training."""

    def __init__(
        self,
        levels: Optional[List[int]] = None,
        curriculum: Optional[List[Dict[str, Any]]] = None,
        n_steps: int = 4096,
        batch_size: int = 512,
        n_epochs: int = 10,
        gamma: float = 0.997,
        gae_lambda: float = 0.95,
        learning_rate: float = 1e-4,
        clip_range: float = 0.15,
        clip_range_final: float = 0.08,
        ent_coef: float = 0.015,
        ent_coef_final: float = 0.003,
        target_kl: float = 0.02,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        max_episode_steps: int = 720,
        output_dir: Optional[str] = None,
        load_model_path: Optional[str] = None,
        checkpoint_episode_interval: int = 500,
        replay_episode_interval: int = 250,
        eval_interval_episodes: int = 500,
        eval_episodes: int = 12,
        eval_levels: Optional[List[int]] = None,
        eval_seed_start: int = 25000,
        seed: int = 42,
        num_workers: int = 4,
        phase_pass_evals_required: int = 2,
    ) -> None:
        self._explicit_levels = levels is not None
        self.curriculum: List[Dict[str, Any]] = curriculum or DEFAULT_CURRICULUM
        self.current_phase_index = 0
        self.phase_episode_wins: List[int] = []
        self.phase_episode_rewards: List[float] = []
        self.phase_start_episode = 0
        self.rehearsal_ids: List[int] = []
        if self._explicit_levels:
            self.levels: List[int] = list(levels)  # type: ignore[arg-type]
        else:
            self.levels = list(self.curriculum[0]["scenario_ids"])
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.seed_counter = seed
        self.load_model_path = load_model_path or None
        self._ret_rms_path = os.path.join(
            output_dir or os.path.join(os.path.dirname(__file__), "output"),
            "ret_rms_state.json"
        )
        self.checkpoint_episode_interval = checkpoint_episode_interval
        self.replay_episode_interval = replay_episode_interval
        self.num_workers = max(1, num_workers)
        self.target_kl = target_kl
        self.ent_coef_start = ent_coef
        self.ent_coef_final = ent_coef_final
        self.eval_interval_episodes = max(1, eval_interval_episodes)
        self.eval_episodes = max(1, eval_episodes)
        self.eval_levels = eval_levels or [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.eval_seed_start = eval_seed_start
        self.evaluation_round = 0
        self.procedural_levels = not self._explicit_levels
        self.curriculum_difficulty_band = 0.0 if not self._explicit_levels else 0.5
        self.phase_pass_evals_required = max(1, int(phase_pass_evals_required))
        self.phase_eval_pass_streak = 0
        self.last_phase_eval_win_rate = 0.0
        self.last_eval_lower_bound = 0.0
        self.last_eval_details: Dict[int, Dict[str, float]] = {}
        self.last_global_eval_lower_bound = 0.0
        self.last_global_eval_details: Dict[int, Dict[str, float]] = {}
        self.last_phase_eval_lower_bound = 0.0
        self.last_phase_eval_details: Dict[int, Dict[str, float]] = {}
        self.stable_checkpoints: Dict[int, str] = {}
        self.peak_phase_index: int = 0
        self.ret_rms = RunningMeanStd()
        self.discounted_returns: Dict[int, float] = {}
        self.scenario_competence: Dict[int, Dict[str, float]] = {}

        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.replay_dir = os.path.join(self.output_dir, "replays")
        os.makedirs(self.replay_dir, exist_ok=True)
        self.training_log_path = os.path.join(self.output_dir, "training_log.csv")
        self.run_manifest_path = os.path.join(self.output_dir, "run_manifest.json")
        self.live_metrics_path = os.path.join(self.output_dir, "live_metrics.json")
        self.metrics_history_path = os.path.join(self.output_dir, "metrics_history.jsonl")
        self.best_model_path = os.path.join(self.output_dir, "treads_ppo_best")
        self.best_train_model_path = os.path.join(self.output_dir, "treads_ppo_best_train")
        self.final_model_path = os.path.join(self.output_dir, "treads_ppo_final")
        self.current_onnx_path = os.path.join(self.output_dir, "treads_policy.onnx")
        self._weights_cache_dir = os.path.join(self.output_dir, ".weights_cache")
        os.makedirs(self._weights_cache_dir, exist_ok=True)
        self._weights_file_path = os.path.join(self._weights_cache_dir, "policy_state_dict.json")
        self.run_started_at = datetime.now(timezone.utc)
        self.run_finished_at: Optional[datetime] = None

        class _DummyMultiDiscreteEnv(gym.Env[NDArray[np.float32], NDArray[np.integer[Any]]]):
            metadata = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
                self.action_space = spaces.MultiDiscrete(ACTION_HEAD_SIZES)

            def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[Dict[str, Any]] = None,
            ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
                super().reset(seed=seed)
                return np.zeros((OBS_SIZE,), dtype=np.float32), {}

            def step(self, action: NDArray[np.integer[Any]]) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
                _ = action
                return np.zeros((OBS_SIZE,), dtype=np.float32), 0.0, True, False, {}

        # Create a dummy MultiDiscrete env so SB3 policy/distribution match worker sampling.
        dummy_env = _DummyMultiDiscreteEnv()
        clip_schedule = linear_schedule_between(clip_range, clip_range_final)
        lr_schedule = linear_schedule_between(learning_rate, 0.0)
        if load_model_path and os.path.exists(load_model_path):
            print(f"Loading PPO model from: {load_model_path}")
            assert_contract_compatible(load_model_path)
            self.model = cast(Any, PPO.load(  # pyright: ignore[reportUnknownMemberType]
                load_model_path,
                env=dummy_env,
                device="cpu",
                learning_rate=lr_schedule,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_schedule,
                clip_range_vf=clip_schedule,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                ent_coef=ent_coef,
                target_kl=target_kl,
            ))
            assert_contract_compatible(load_model_path, self.model.action_space)
            print(
                "  Resume settings: "
                f"n_steps={cast(int, self.model.n_steps)} "
                f"batch_size={cast(int, self.model.batch_size)} "
                f"n_epochs={cast(int, self.model.n_epochs)} "
                f"gamma={cast(float, self.model.gamma)} "
                f"gae_lambda={cast(float, self.model.gae_lambda)} "
                f"ent_coef={cast(float, self.model.ent_coef)}"
            )
        else:
            self.model = PPO(  # pyright: ignore[reportCallIssue]
                "MlpPolicy",
                dummy_env,
                verbose=0,
                learning_rate=lr_schedule,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_schedule,
                clip_range_vf=clip_schedule,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                ent_coef=ent_coef,
                target_kl=target_kl,
                device="cpu",
                policy_kwargs=dict(net_arch=[512, 256], optimizer_kwargs=dict(eps=1e-5)),
                seed=seed,
            )
            # The semantic zero-residual aim action is useful immediately and
            # still leaves broad categorical exploration available.
            import torch
            with torch.no_grad():
                aim_offset = ACTION_HEAD_SIZES[0]
                neutral_bin = int(MODEL_CONTRACT["action"]["neutralAimBin"])
                cast(Any, self.model).policy.action_net.bias[aim_offset + neutral_bin] = 2.0
        if cast(int, self.model.n_steps) != self.n_steps or cast(int, self.model.rollout_buffer.buffer_size) != self.n_steps:
            raise RuntimeError(
                "Model rollout settings do not match trainer settings: "
                f"trainer n_steps={self.n_steps}, "
                f"model n_steps={cast(int, self.model.n_steps)}, "
                f"buffer_size={cast(int, self.model.rollout_buffer.buffer_size)}"
            )
        dummy_env.close()

        # Set up SB3 logger (required for model.train())
        self.model.set_logger(configure(self.output_dir, ["stdout"]))

        # Logging state
        self.total_timesteps = 0
        self.total_episodes = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_wins: List[int] = []
        self.episode_levels: List[int] = []
        self.episode_reward_breakdowns: List[Dict[str, Any]] = []
        self.best_win_rate = 0.0
        self.best_train_win_rate = 0.0
        self.best_eval_win_rate = 0.0
        self.best_eval_lower_bound = 0.0
        self.start_time = time.time()
        if self.load_model_path:
            self._restore_trainer_state(self.load_model_path)
        if not self._explicit_levels:
            self.levels = self._build_mixed_levels()
        self.workers: List[Dict[str, Any]] = []
        self._start_workers()
        self._write_run_manifest(status="initializing")
        if not self.load_model_path:
            with open(self.metrics_history_path, "w", encoding="utf-8"):
                pass

    def _evaluate_policy(self) -> float:
        """Deterministic actions on held-out randomized level variants."""
        seed_start = self.eval_seed_start + self.evaluation_round * 100_000
        self.evaluation_round += 1
        win_rate = self._evaluate_levels(self.eval_levels, self.eval_episodes, seed_start)
        self.last_global_eval_lower_bound = self.last_eval_lower_bound
        self.last_global_eval_details = dict(self.last_eval_details)
        return win_rate

    def _evaluate_levels(self, levels: List[int], episodes_per_level: int, seed_start: int) -> float:
        """Evaluate deterministic actions on independent held-out variants."""
        wins = 0
        episodes = 0
        seed = seed_start
        details: Dict[int, Dict[str, float]] = {}

        for level in levels:
            level_wins = 0
            env: Optional[TreadsEnv] = None

            def create_env(episode_seed: int) -> TreadsEnv:
                return TreadsEnv(
                    level=level,
                    seed_start=episode_seed,
                    max_episode_steps=self.max_episode_steps,
                    spawn_jitter=True,
                    procedural_levels=True,
                    difficulty_band=max(0.15, self.curriculum_difficulty_band),
                    player_max_ammo=self._get_player_max_ammo(),
                    player_max_bombs=self._get_player_max_bombs(),
                )

            try:
                for _ in range(episodes_per_level):
                    result: Dict[str, Any] = {}
                    for attempt in range(3):
                        if env is None:
                            env = create_env(seed)
                        try:
                            obs, _ = env.reset(seed=seed)
                            done = False
                            info: Dict[str, Any] = {}
                            while not done:
                                action, _ = cast(Any, self.model).predict(obs, deterministic=True)
                                obs_raw = cast(Optional[Dict[str, Any]], getattr(env, "_last_obs_raw", None))
                                decoded = decode_multi_discrete_action(action, obs_raw)
                                obs, _reward, terminated, truncated, info = env.step(decoded)
                                done = terminated or truncated
                            result = cast(Dict[str, Any], info.get("result", {}))
                            break
                        except Exception as exc:
                            env.close()
                            env = None
                            if attempt == 2:
                                raise RuntimeError(
                                    f"Evaluation failed after 3 attempts for level={level}, seed={seed}"
                                ) from exc
                            print(
                                f"  WARNING: retrying eval level={level} seed={seed} "
                                f"after attempt {attempt + 1}: {exc}"
                            )
                    outcome = int(bool(result.get("win", False)))
                    wins += outcome
                    level_wins += outcome
                    episodes += 1
                    seed += 1
            finally:
                if env is not None:
                    env.close()
            details[level] = {
                "wins": float(level_wins),
                "episodes": float(episodes_per_level),
                "win_rate": float(level_wins / max(episodes_per_level, 1)),
                "wilson_lower": self._wilson_lower_bound(level_wins, episodes_per_level),
            }

        self.last_eval_lower_bound = self._wilson_lower_bound(wins, episodes)
        self.last_eval_details = details
        return float(wins / max(episodes, 1))

    @staticmethod
    def _wilson_lower_bound(wins: int, episodes: int, z: float = 1.96) -> float:
        if episodes <= 0:
            return 0.0
        rate = wins / episodes
        denominator = 1.0 + z * z / episodes
        center = rate + z * z / (2.0 * episodes)
        margin = z * np.sqrt((rate * (1.0 - rate) + z * z / (4.0 * episodes)) / episodes)
        return float(max(0.0, (center - margin) / denominator))

    def _save_ret_rms(self) -> None:
        """Persist running-mean-std reward normalizer state next to the model."""
        try:
            self._json_dump(self._ret_rms_path, self.ret_rms.to_dict())
        except Exception as exc:
            print(f"  WARNING: failed to persist ret_rms state to {self._ret_rms_path}: {exc}")

    @staticmethod
    def _trainer_state_path(model_path: str) -> str:
        root, extension = os.path.splitext(model_path)
        if extension.lower() != ".zip":
            root = model_path
        return root + ".trainer.json"

    def _trainer_state_payload(self) -> Dict[str, Any]:
        return {
            "schemaVersion": 3,
            "modelContract": MODEL_CONTRACT,
            "totalTimesteps": self.total_timesteps,
            "totalEpisodes": self.total_episodes,
            "seedCounter": self.seed_counter,
            "currentPhaseIndex": self.current_phase_index,
            "peakPhaseIndex": self.peak_phase_index,
            "phaseStartEpisode": self.phase_start_episode,
            "phaseEpisodeWins": self.phase_episode_wins,
            "phaseEpisodeRewards": self.phase_episode_rewards,
            "phaseEvalPassStreak": self.phase_eval_pass_streak,
            "lastPhaseEvalWinRate": self.last_phase_eval_win_rate,
            "lastGlobalEvalLowerBound": self.last_global_eval_lower_bound,
            "lastGlobalEvalDetails": self.last_global_eval_details,
            "lastPhaseEvalLowerBound": self.last_phase_eval_lower_bound,
            "lastPhaseEvalDetails": self.last_phase_eval_details,
            "curriculumDifficultyBand": self.curriculum_difficulty_band,
            "scenarioCompetence": self.scenario_competence,
            "evaluationRound": self.evaluation_round,
            "bestWinRate": self.best_win_rate,
            "bestTrainWinRate": self.best_train_win_rate,
            "bestEvalWinRate": self.best_eval_win_rate,
            "bestEvalLowerBound": self.best_eval_lower_bound,
            "returnNormalizer": self.ret_rms.to_dict(),
            "discountedReturnsByWorker": self.discounted_returns,
            "stableCheckpoints": self.stable_checkpoints,
        }

    def _save_model_checkpoint(self, model_base_path: str) -> str:
        model_path = model_base_path if model_base_path.endswith(".zip") else model_base_path + ".zip"
        temp_model_path = f"{model_path}.{os.getpid()}.{threading.get_ident()}.tmp.zip"
        try:
            cast(Any, self.model).save(temp_model_path)
            os.replace(temp_model_path, model_path)
        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
        write_contract_for_model(model_path)
        self._json_dump(self._trainer_state_path(model_path), self._trainer_state_payload())
        self._save_ret_rms()
        return model_path

    def _restore_trainer_state(self, model_path: str) -> None:
        state_path = self._trainer_state_path(model_path)
        if not os.path.isfile(state_path):
            raise ValueError(
                f"Checkpoint is missing trainer state: {state_path}. "
                "A model-only load would reset curriculum and schedules, so resume is refused."
            )
        with open(state_path, "r", encoding="utf-8") as handle:
            state = cast(Dict[str, Any], json.load(handle))
        if int(state.get("schemaVersion", 0)) != 3:
            raise ValueError(f"Unsupported trainer state schema in {state_path}.")

        self.total_timesteps = int(state.get("totalTimesteps", 0))
        self.total_episodes = int(state.get("totalEpisodes", 0))
        self.seed_counter = int(state.get("seedCounter", self.seed))
        self.current_phase_index = int(state.get("currentPhaseIndex", 0))
        if not 0 <= self.current_phase_index < len(self.curriculum):
            raise ValueError(f"Invalid curriculum phase in {state_path}: {self.current_phase_index}")
        self.peak_phase_index = int(state.get("peakPhaseIndex", self.current_phase_index))
        self.phase_start_episode = int(state.get("phaseStartEpisode", self.total_episodes))
        self.phase_episode_wins = [int(value) for value in state.get("phaseEpisodeWins", [])]
        self.phase_episode_rewards = [float(value) for value in state.get("phaseEpisodeRewards", [])]
        self.phase_eval_pass_streak = int(state.get("phaseEvalPassStreak", 0))
        self.last_phase_eval_win_rate = float(state.get("lastPhaseEvalWinRate", 0.0))
        self.last_global_eval_lower_bound = float(state.get("lastGlobalEvalLowerBound", 0.0))
        self.last_global_eval_details = {
            int(key): {name: float(value) for name, value in cast(Dict[str, Any], details).items()}
            for key, details in cast(Dict[str, Any], state.get("lastGlobalEvalDetails", {})).items()
        }
        self.last_phase_eval_lower_bound = float(state.get("lastPhaseEvalLowerBound", 0.0))
        self.last_phase_eval_details = {
            int(key): {name: float(value) for name, value in cast(Dict[str, Any], details).items()}
            for key, details in cast(Dict[str, Any], state.get("lastPhaseEvalDetails", {})).items()
        }
        self.curriculum_difficulty_band = float(state.get("curriculumDifficultyBand", 0.0))
        self.scenario_competence = {
            int(key): {name: float(value) for name, value in cast(Dict[str, Any], stats).items()}
            for key, stats in cast(Dict[str, Any], state.get("scenarioCompetence", {})).items()
        }
        self.evaluation_round = int(state.get("evaluationRound", 0))
        self.best_win_rate = float(state.get("bestWinRate", 0.0))
        self.best_train_win_rate = float(state.get("bestTrainWinRate", 0.0))
        self.best_eval_win_rate = float(state.get("bestEvalWinRate", 0.0))
        self.best_eval_lower_bound = float(state.get("bestEvalLowerBound", 0.0))
        self.ret_rms = RunningMeanStd.from_dict(cast(Dict[str, float], state.get("returnNormalizer", {})))
        self.discounted_returns = {
            int(key): float(value)
            for key, value in cast(Dict[str, Any], state.get("discountedReturnsByWorker", {})).items()
        }
        self.stable_checkpoints = {
            int(key): str(value)
            for key, value in cast(Dict[str, Any], state.get("stableCheckpoints", {})).items()
        }
        cast(Any, self.model).num_timesteps = self.total_timesteps
        print(
            f"Restored trainer state: episodes={self.total_episodes}, timesteps={self.total_timesteps}, "
            f"phase={self.current_phase_index + 1}, ret_var={self.ret_rms.var:.4f}"
        )

    def _save_stable_checkpoint(self, phase_index: int, eval_win_rate: float) -> None:
        path = os.path.join(self.output_dir, f"treads_ppo_phase{phase_index}_stable")
        self.stable_checkpoints[phase_index] = path + ".zip"
        self._save_model_checkpoint(path)
        print(
            f"  Stable checkpoint saved for phase {phase_index + 1} "
            f"(eval wr={eval_win_rate:.3f}): {self.stable_checkpoints[phase_index]}"
        )

    def _advance_curriculum_from_eval(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if self._explicit_levels or self.current_phase_index >= len(self.curriculum) - 1:
            return None
        previous = self._get_curriculum_phase()
        self.current_phase_index += 1
        self.peak_phase_index = max(self.peak_phase_index, self.current_phase_index)
        self.rehearsal_ids = self._build_rehearsal_ids()
        self.levels = self._build_mixed_levels()
        self.phase_episode_wins = []
        self.phase_episode_rewards = []
        self.phase_start_episode = self.total_episodes
        self.phase_eval_pass_streak = 0
        self.last_phase_eval_win_rate = 0.0
        return previous, self._get_curriculum_phase()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _json_dump(self, path: str, payload: Dict[str, Any]) -> None:
        temp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    def _count_replays(self) -> int:
        if not os.path.isdir(self.replay_dir):
            return 0
        return sum(1 for name in os.listdir(self.replay_dir) if name.endswith(".json"))

    def _write_run_manifest(self, status: str, error: str = "") -> None:
        learning_rate = cast(float, self.model.learning_rate(1.0) if callable(self.model.learning_rate) else self.model.learning_rate)
        clip_range = cast(float, self.model.clip_range(1.0) if callable(self.model.clip_range) else self.model.clip_range)
        payload: Dict[str, Any] = {
            "runId": os.path.basename(os.path.abspath(self.output_dir)),
            "outputDir": os.path.abspath(self.output_dir),
            "status": status,
            "processId": os.getpid(),
            "startedAt": self.run_started_at.isoformat(),
            "finishedAt": self.run_finished_at.isoformat() if self.run_finished_at is not None else None,
            "updatedAt": self._now_iso(),
            "loadModelPath": self.load_model_path,
            "seed": self.seed,
            "numWorkers": self.num_workers,
            "bestWinRate": self.best_win_rate,
            "totals": {
                "timesteps": self.total_timesteps,
                "episodes": self.total_episodes,
            },
            "curriculum": {
                "enabled": not self._explicit_levels,
                "currentPhaseIndex": self.current_phase_index if not self._explicit_levels else None,
                "currentPhaseName": self._get_curriculum_phase()["name"] if not self._explicit_levels else "Explicit levels",
                "activeScenarios": sorted(set(self.levels)),
                "samplingWeights": {
                    str(scenario_id): self.levels.count(scenario_id)
                    for scenario_id in sorted(set(self.levels))
                },
                "rehearsalScenarios": self.rehearsal_ids,
                "difficultyBand": self.curriculum_difficulty_band,
                "playerMaxAmmo": self._get_player_max_ammo(),
                "playerMaxBombs": self._get_player_max_bombs(),
                "scenarioCompetence": self.scenario_competence,
            },
            "hyperparameters": {
                "nSteps": self.n_steps,
                "batchSize": cast(int, self.model.batch_size),
                "nEpochs": cast(int, self.model.n_epochs),
                "gamma": self.gamma,
                "gaeLambda": self.gae_lambda,
                "learningRate": learning_rate,
                "clipRange": clip_range,
                "entCoef": cast(float, self.model.ent_coef),
                "targetKl": self.target_kl,
                "maxEpisodeSteps": self.max_episode_steps,
                "checkpointEpisodeInterval": self.checkpoint_episode_interval,
                "replayEpisodeInterval": self.replay_episode_interval,
                "evalIntervalEpisodes": self.eval_interval_episodes,
                "evalEpisodesPerLevel": self.eval_episodes,
                "evalLevels": self.eval_levels,
                "proceduralLevels": self.procedural_levels,
            },
            "paths": {
                "trainingLog": os.path.abspath(self.training_log_path),
                "liveMetrics": os.path.abspath(self.live_metrics_path),
                "metricsHistory": os.path.abspath(self.metrics_history_path),
                "replays": os.path.abspath(self.replay_dir),
                "bestModel": os.path.abspath(self.best_model_path + ".zip"),
                "bestTrainModel": os.path.abspath(self.best_train_model_path + ".zip"),
                "finalModel": os.path.abspath(self.final_model_path + ".zip"),
                "onnxModel": os.path.abspath(self.current_onnx_path),
            },
            "artifacts": {
                "replayCount": self._count_replays(),
                "onnxAvailable": os.path.exists(self.current_onnx_path),
                "bestEvalWinRate": self.best_eval_win_rate,
            },
            "evaluation": {
                "lastGlobalWilsonLower": self.last_global_eval_lower_bound,
                "lastGlobalDetails": self.last_global_eval_details,
                "lastPhaseWilsonLower": self.last_phase_eval_lower_bound,
                "lastPhaseDetails": self.last_phase_eval_details,
            },
            "error": error or None,
        }
        self._json_dump(self.run_manifest_path, payload)

    def _build_iteration_snapshot(
        self,
        iteration: int,
        avg_reward: float,
        avg_winrate: float,
        avg_tick: float,
        avg_hit: float,
        avg_hurt: float,
        avg_kill: float,
        avg_death: float,
        avg_terminal_win: float,
        avg_terminal_loss: float,
        avg_timeout: float,
        avg_approach: float,
        avg_dodge: float,
        steps_per_sec: float,
        elapsed: float,
    ) -> Dict[str, Any]:
        current_phase_name = self._get_curriculum_phase()["name"] if not self._explicit_levels else "Explicit levels"
        return {
            "timestamp": self._now_iso(),
            "iteration": iteration,
            "timesteps": self.total_timesteps,
            "episodes": self.total_episodes,
            "curriculumPhase": current_phase_name,
            "currentPhaseIndex": self.current_phase_index if not self._explicit_levels else None,
            "activeScenarios": list(self.levels),
            "rehearsalScenarios": list(self.rehearsal_ids),
            "difficultyBand": self.curriculum_difficulty_band,
            "phaseRecentWinrate500": self._phase_recent_win_rate() if not self._explicit_levels else None,
            "phaseEvalPassStreak": self.phase_eval_pass_streak if not self._explicit_levels else None,
            "lastPhaseEvalWinRate": self.last_phase_eval_win_rate if not self._explicit_levels else None,
            "lastGlobalEvalWilsonLower": self.last_global_eval_lower_bound,
            "lastGlobalEvalDetails": self.last_global_eval_details,
            "lastPhaseEvalWilsonLower": self.last_phase_eval_lower_bound,
            "lastPhaseEvalDetails": self.last_phase_eval_details,
            "avgReward50": avg_reward,
            "avgWinrate50": avg_winrate,
            "avgTick50": avg_tick,
            "avgHit50": avg_hit,
            "avgHurt50": avg_hurt,
            "avgKill50": avg_kill,
            "avgDeath50": avg_death,
            "avgTerminalWin50": avg_terminal_win,
            "avgTerminalLoss50": avg_terminal_loss,
            "avgTimeout50": avg_timeout,
            "avgApproach50": avg_approach,
            "avgDodge50": avg_dodge,
            "bestWinRate": self.best_win_rate,
            "stepsPerSec": steps_per_sec,
            "elapsedSec": elapsed,
            "replayCount": self._count_replays(),
            "onnxAvailable": os.path.exists(self.current_onnx_path),
        }

    def _write_iteration_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._json_dump(self.live_metrics_path, snapshot)
        with open(self.metrics_history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot))
            handle.write("\n")

    def _export_policy_onnx(self, model_base_path: str) -> None:
        model_path = model_base_path if model_base_path.endswith(".zip") else f"{model_base_path}.zip"
        if not os.path.exists(model_path):
            return
        try:
            export_to_onnx(model_path, self.current_onnx_path)
        except Exception as exc:
            print(f"  WARNING: failed to export ONNX model: {exc}")

    def _start_workers(self) -> None:
        """Start Node.js rollout-worker gRPC processes."""
        if not os.path.exists(ROLLOUT_WORKER_PATH):
            raise FileNotFoundError(
                "Rollout worker entrypoint not found. Run `npm run build:training` first. "
                f"Expected: {ROLLOUT_WORKER_PATH}"
            )
        self.workers = []
        for _ in range(self.num_workers):
            self.workers.append(self._start_worker_handle())

        for i, w in enumerate(self.workers):
            ready = self._read_msg_from(w)
            assert ready["type"] == "ready", f"Worker {i}: expected ready, got {ready}"

    @staticmethod
    def _start_worker_handle() -> Dict[str, Any]:
        port = _find_free_port()
        proc = subprocess.Popen(
            [
                "node",
                ROLLOUT_WORKER_PATH,
                "--port",
                str(port),
                "--parent-pid",
                str(os.getpid()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        channel = grpc.insecure_channel(
            f"127.0.0.1:{port}",
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ],
        )
        stub = TREADS_PB2_GRPC.RolloutWorkerServiceStub(channel)
        deadline = time.time() + 10.0
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                stub.Health(TREADS_PB2.HealthRequest(), timeout=1.0)
                return {
                    "process": proc,
                    "channel": channel,
                    "stub": stub,
                    "pending": None,
                }
            except Exception as exc:  # pragma: no cover - transient process startup
                last_error = exc
                time.sleep(0.1)
        try:
            channel.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        raise RuntimeError(f"Failed to start rollout worker gRPC process: {last_error}")

    @staticmethod
    def _close_worker_handle(worker: Dict[str, Any]) -> None:
        try:
            stub = worker.get("stub")
            if stub is not None:
                stub.Close(TREADS_PB2.CloseRequest(), timeout=1.0)
        except Exception:
            pass

        channel = worker.get("channel")
        if channel is not None:
            try:
                channel.close()
            except Exception:
                pass

        proc = cast(Optional[subprocess.Popen[str]], worker.get("process"))
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass

    def _restart_worker(self, index: int) -> None:
        """Restart a single rollout worker after timeout/crash."""
        old = self.workers[index]
        self._close_worker_handle(old)

        self.workers[index] = self._start_worker_handle()
        ready = self._read_msg_from(self.workers[index])
        if ready.get("type") != "ready":
            raise RuntimeError(f"Worker {index}: restart failed, expected ready but got {ready}")

    def _get_curriculum_phase(self) -> Dict[str, Any]:
        return self.curriculum[self.current_phase_index]

    def _get_player_max_ammo(self) -> int:
        return int(self._get_curriculum_phase().get("player_max_ammo", 5))

    def _get_player_max_bombs(self) -> int:
        if self._explicit_levels:
            return 0
        return int(self._get_curriculum_phase().get(
            "player_max_bombs",
            0 if self.current_phase_index < 12 else 1,
        ))

    def _get_curriculum_levels(self) -> List[int]:
        return list(self._get_curriculum_phase()["scenario_ids"])

    def _phase_episode_count(self) -> int:
        return self.total_episodes - self.phase_start_episode

    def _build_rehearsal_ids(self) -> List[int]:
        """Build rehearsal scenario list from ALL completed phases, not just recent ones.

        Foundational skills (obstacle nav, 1v1 combat) decay when the curriculum
        moves to multi-enemy open-field phases. This method ensures every mastered
        skill category stays represented in the rehearsal pool.

        Strategy: pick 1 representative scenario from each completed phase,
        prioritizing obstacle / combat scenarios that test distinct skills.
        """
        completed_phases = self.curriculum[: self.current_phase_index]
        if not completed_phases:
            return []

        # One representative per completed phase (last scenario_id = hardest in that phase)
        foundational: List[int] = []
        seen: set[int] = set()
        for phase in completed_phases:
            sids = phase["scenario_ids"]
            # Pick the last (hardest) scenario that we haven't already included
            for sid in reversed(sids):
                if sid not in seen:
                    foundational.append(sid)
                    seen.add(sid)
                    break

        return foundational

    def _build_mixed_levels(self) -> List[int]:
        """Build a competence-weighted pool with permanent rehearsal.

        Every previously introduced scenario remains eligible. Under-mastered
        scenarios receive more samples, while all nine real levels are present
        from the first update so the policy never faces a late distribution jump.
        """
        if self._explicit_levels:
            return list(self.levels)

        introduced: List[int] = []
        for phase in self.curriculum[: self.current_phase_index + 1]:
            for scenario_id in cast(List[int], phase["scenario_ids"]):
                if scenario_id not in introduced:
                    introduced.append(scenario_id)

        weighted: List[int] = []
        current_ids = set(self._get_curriculum_levels())
        for scenario_id in introduced:
            stats = self.scenario_competence.get(scenario_id, {})
            competence = float(stats.get("ema", 0.0)) if stats.get("episodes", 0.0) >= 20 else 0.0
            repetitions = 1 + round(6.0 * (1.0 - competence))
            if scenario_id in current_ids:
                repetitions += 2
            weighted.extend([scenario_id] * max(1, repetitions))

        for real_level in range(1, 10):
            stats = self.scenario_competence.get(real_level, {})
            episodes = float(stats.get("episodes", 0.0))
            competence = float(stats.get("ema", 0.0)) if episodes >= 20 else 0.0
            repetitions = 1 if episodes < 20 else 1 + round(2.0 * (1.0 - competence))
            weighted.extend([real_level] * repetitions)

        return weighted

    def _target_difficulty_band(self) -> float:
        """Map phase progress + competence into a smooth [0,1] difficulty target."""
        if self._explicit_levels:
            return 0.5

        phase_count = max(1, len(self.curriculum) - 1)
        phase_position = self.current_phase_index / phase_count
        phase = self._get_curriculum_phase()
        min_phase_episodes = int(phase.get("min_phase_episodes") or 2000)
        in_phase_progress = min(1.0, max(0.0, self._phase_episode_count() / max(min_phase_episodes, 1)))
        base = 0.80 * phase_position + 0.20 * in_phase_progress

        wr = self._phase_recent_win_rate()
        if wr < 0.15:
            base -= 0.15
        elif wr < 0.25:
            base -= 0.08
        elif wr > 0.80:
            base += 0.08
        elif wr > 0.65:
            base += 0.05

        return min(1.0, max(0.0, base))

    def _update_difficulty_band(self) -> None:
        """EMA update to create many micro-levels instead of abrupt jumps."""
        target = self._target_difficulty_band()
        self.curriculum_difficulty_band = 0.90 * self.curriculum_difficulty_band + 0.10 * target

    def _phase_recent_win_rate(self) -> float:
        if not self.phase_episode_wins:
            return 0.0
        window = self.phase_episode_wins[-500:]
        return float(np.mean(window))

    def _phase_recent_reward(self, window: int = 200) -> float:
        if not self.phase_episode_rewards:
            return 0.0
        return float(np.mean(self.phase_episode_rewards[-window:]))

    def _phase_reward_trend(self, window: int = 200) -> float:
        """Positive means recent rewards improved vs the previous window."""
        if len(self.phase_episode_rewards) < window * 2:
            return 0.0
        recent = float(np.mean(self.phase_episode_rewards[-window:]))
        previous = float(np.mean(self.phase_episode_rewards[-window * 2 : -window]))
        return recent - previous

    def _phase_is_stable(self) -> bool:
        """Require non-collapsing reward dynamics before curriculum promotion."""
        if len(self.phase_episode_rewards) < 300:
            return False  # insufficient data to judge stability
        # Disallow large drops in the recent trend.
        return self._phase_reward_trend(window=150) >= -0.35

    @staticmethod
    def _send_msg_to(worker: Dict[str, Any], msg: Dict[str, Any]) -> None:
        """Queue or execute a command for a specific rollout worker."""
        msg_type = str(msg.get("type", ""))
        if msg_type == "collect":
            worker["pending"] = msg
            return

        stub = worker["stub"]
        if msg_type == "set_weights_from_file":
            path_value = str(msg.get("path", ""))
            if not path_value:
                raise RuntimeError("Missing path for set_weights_from_file")
            stub.SetWeightsFromFile(
                TREADS_PB2.SetWeightsFromFileRequest(path=path_value),
                timeout=30.0,
            )
            worker["pending"] = {"type": "weights_set"}
            return

        if msg_type == "exit":
            try:
                stub.Close(TREADS_PB2.CloseRequest(), timeout=2.0)
            except Exception:
                pass
            worker["pending"] = {"type": "closed"}
            return

        raise RuntimeError(f"Unsupported worker command type: {msg_type}")

    @staticmethod
    def _read_msg_from(worker: Dict[str, Any], timeout_sec: float = 180.0) -> Dict[str, Any]:
        """Read command result from a specific rollout worker."""
        pending = worker.get("pending")
        if pending is not None and pending.get("type") in ("weights_set", "closed"):
            worker["pending"] = None
            return cast(Dict[str, Any], pending)

        stub = worker["stub"]
        if pending is not None and pending.get("type") == "collect":
            req = pending
            response = stub.Collect(
                TREADS_PB2.CollectRequest(
                    n_steps=int(req.get("n_steps", 4096)),
                    levels=[int(v) for v in cast(List[Any], req.get("levels", [1]))],
                    max_ticks=int(req.get("maxTicks", 1800)),
                    tick_norm_ticks=int(req.get("tickNormTicks", 1800)),
                    seed_start=int(req.get("seedStart", 0)),
                    replay_every_episodes=int(req.get("replayEveryEpisodes", 0)),
                    replay_dir=str(req.get("replayDir", "")),
                    worker_id=int(req.get("workerId", 0)),
                    target_episodes=int(req.get("targetEpisodes", 0)),
                    shaping_scale=float(req.get("shapingScale", 1.0)),
                    procedural_levels=bool(req.get("proceduralLevels", True)),
                    difficulty_band=float(req.get("difficultyBand", 0.0)),
                    player_max_ammo=int(req.get("playerMaxAmmo", 0)),
                    player_max_bombs=int(req.get("playerMaxBombs", 0)),
                    global_episode_offset=int(req.get("globalEpisodeOffset", 0)),
                    gamma=float(req.get("gamma", 0.99)),
                ),
                timeout=timeout_sec,
            )
            worker["pending"] = None
            payload = cast(Dict[str, Any], json.loads(str(response.rollout_json)))
            if payload.get("type") != "rollout":
                raise RuntimeError(f"Worker returned unexpected payload type: {payload.get('type')}")
            return payload

        health = stub.Health(TREADS_PB2.HealthRequest(), timeout=min(timeout_sec, 5.0))
        if not bool(health.ok):
            raise RuntimeError(f"Worker health check failed: {health.message}")
        return {"type": "ready"}

    def _send_msg(self, msg: Dict[str, Any]) -> None:
        """Send JSON message to the first worker (legacy helper)."""
        self._send_msg_to(self.workers[0], msg)

    def _read_msg(self) -> Dict[str, Any]:
        """Read JSON message from the first worker (legacy helper)."""
        return self._read_msg_from(self.workers[0])

    def _extract_weights(self) -> Dict[str, Any]:
        """Extract model weights as nested lists for JSON transfer."""
        state_dict: Dict[str, Any] = {}
        for key, tensor in cast(Any, self.model).policy.state_dict().items():
            state_dict[key] = tensor.cpu().numpy().tolist()
        return state_dict

    def _send_weights(self) -> None:
        """Send current model weights to all rollout workers."""
        weights = self._extract_weights()
        tmp_file = self._weights_file_path + f".{os.getpid()}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as handle:
            json.dump(weights, handle, separators=(",", ":"))
        os.replace(tmp_file, self._weights_file_path)
        msg: Dict[str, Any] = {"type": "set_weights_from_file", "path": self._weights_file_path}
        for w in self.workers:
            self._send_msg_to(w, msg)
        for i, w in enumerate(self.workers):
            ack = self._read_msg_from(w)
            assert ack["type"] == "weights_set", f"Worker {i}: expected weights_set, got {ack}"

    def _collect_rollout(self, target_episodes: int) -> Dict[str, Any]:
        """Request rollout collection from workers (parallel when num_workers > 1)."""
        num_w = len(self.workers)
        # Divide steps as evenly as possible.
        base_steps = self.n_steps // num_w
        remainder = self.n_steps % num_w
        worker_steps = [base_steps + (1 if i < remainder else 0) for i in range(num_w)]

        # Keep deterministic seeds while avoiding overlap.
        worker_seeds: List[int] = []
        running_seed = self.seed_counter
        for steps in worker_steps:
            worker_seeds.append(running_seed)
            running_seed += steps

        for attempt in range(2):
            # Send collect commands to all workers (non-blocking writes)
            for i, w in enumerate(self.workers):
                self._send_msg_to(w, {
                    "type": "collect",
                    "n_steps": worker_steps[i],
                    "levels": self.levels,
                    "maxTicks": self.max_episode_steps,
                    "tickNormTicks": self.max_episode_steps,
                    "seedStart": worker_seeds[i],
                    "replayEveryEpisodes": self.replay_episode_interval,
                    "replayDir": self.replay_dir,
                    "workerId": i,
                    "targetEpisodes": target_episodes,
                    "shapingScale": max(0.3, 1.0 - self.current_phase_index * 0.06),
                    "proceduralLevels": self.procedural_levels,
                    "difficultyBand": self.curriculum_difficulty_band,
                "playerMaxAmmo": self._get_player_max_ammo(),
                "playerMaxBombs": self._get_player_max_bombs(),
                    "globalEpisodeOffset": self.total_episodes,
                    "gamma": self.gamma,
                })

            # Read results from all workers in parallel using threads
            results: List[Optional[Dict[str, Any]]] = [None] * num_w
            errors: List[Optional[Exception]] = [None] * num_w

            def read_worker(idx: int) -> None:
                try:
                    results[idx] = self._read_msg_from(self.workers[idx])
                except Exception as exc:
                    errors[idx] = exc

            if num_w == 1:
                read_worker(0)
            else:
                threads = [threading.Thread(target=read_worker, args=(i,)) for i in range(num_w)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            failed = [i for i, err in enumerate(errors) if err is not None]
            if failed:
                if attempt == 0:
                    print(f"  Worker failure detected ({failed}), restarting and retrying collect once...")
                    for idx in failed:
                        self._restart_worker(idx)
                    self._send_weights()
                    continue
                detail = ", ".join(f"worker {i}: {errors[i]}" for i in failed)
                raise RuntimeError(f"Rollout collection failed after retry: {detail}")

            # Merge rollout data and preserve per-worker chunks for correct GAE.
            merged: Dict[str, Any] = {
                "type": "rollout",
                "n_steps": 0,
                "obs": [],
                "actions": [],
                "rewards": [],
                "episode_starts": [],
                "values": [],
                "log_probs": [],
                "episode_rewards": [],
                "episode_lengths": [],
                "episode_wins": [],
                "episode_levels": [],
                "episode_reward_breakdowns": [],
                "worker_rollouts": [],
            }

            for i, r in enumerate(results):
                assert r is not None and r.get("type") == "rollout", f"Worker {i}: expected rollout, got {r}"
                merged["obs"].extend(r["obs"])
                merged["actions"].extend(r["actions"])
                merged["rewards"].extend(r["rewards"])
                merged["episode_starts"].extend(r["episode_starts"])
                merged["values"].extend(r["values"])
                merged["log_probs"].extend(r["log_probs"])
                merged["episode_rewards"].extend(r.get("episode_rewards", []))
                merged["episode_lengths"].extend(r.get("episode_lengths", []))
                merged["episode_wins"].extend(r.get("episode_wins", []))
                merged["episode_levels"].extend(r.get("episode_levels", []))
                merged["episode_reward_breakdowns"].extend(r.get("episode_reward_breakdowns", []))
                merged["worker_rollouts"].append({
                    "n_steps": r["n_steps"],
                    "obs": r["obs"],
                    "actions": r["actions"],
                    "rewards": r["rewards"],
                    "episode_starts": r["episode_starts"],
                    "values": r["values"],
                    "log_probs": r["log_probs"],
                    "truncation_values": r.get("truncation_values", [0.0] * len(r["rewards"])),
                    "last_done": r["last_done"],
                    "last_value": r["last_value"],
                })

            merged["n_steps"] = len(cast(List[Any], merged["obs"]))
            if merged["n_steps"] != self.n_steps:
                raise RuntimeError(
                    f"Collected rollout size mismatch: expected {self.n_steps}, got {merged['n_steps']}"
                )
            self.seed_counter = running_seed
            return merged

        raise RuntimeError("Unreachable collect retry state")

    def _compute_chunk_gae(
        self,
        rewards: NDArray[np.float32],
        episode_starts: NDArray[np.float32],
        values: NDArray[np.float32],
        last_value: float,
        last_done: bool,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute GAE/returns for an independent rollout chunk."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0.0
        n = rewards.shape[0]
        for step in range(n - 1, -1, -1):
            if step == n - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = float(last_value)
            else:
                next_non_terminal = 1.0 - float(episode_starts[step + 1])
                next_values = float(values[step + 1])

            delta = float(rewards[step]) + self.gamma * next_values * next_non_terminal - float(values[step])
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + values
        return advantages, returns

    def _log_per_head_entropy(self) -> None:
        """Print per-head entropy of the current policy on the buffered observations.
        Helps detect collapse of one head (most often `aim` collapsing to one bin).
        """
        try:
            import torch
            buf = self.model.rollout_buffer
            buffered_obs = np.asarray(buf.observations, dtype=np.float32)
            obs_t = torch.as_tensor(buffered_obs.reshape(-1, OBS_SIZE), dtype=torch.float32)
            policy = cast(Any, self.model).policy
            with torch.no_grad():
                dist = policy.get_distribution(obs_t)
                # MultiCategoricalDistribution stores per-head torch.distributions.Categorical
                # in `distribution` (a list).
                inner = getattr(dist, "distribution", None)
                if inner is None or not isinstance(inner, list):
                    return
                inner_list = cast(List[Any], inner)
                head_entropies = [
                    float(d.entropy().mean().item()) for d in inner_list
                ]
            head_names = ["move", "aim", "fire", "bomb"]
            head_max_ent = [float(np.log(s)) for s in ACTION_HEAD_SIZES]
            parts = [
                f"{name}={ent:.3f}/{maxent:.3f}"
                for name, ent, maxent in zip(head_names, head_entropies, head_max_ent)
            ]
            print(f"  PPO head-ent: {' '.join(parts)}")
        except Exception as exc:  # pragma: no cover - logging only
            print(f"  (per-head entropy unavailable: {exc})")

    @staticmethod
    def _reward_source_percentages(breakdowns: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        if not breakdowns:
            return 0.0, 0.0, 0.0

        combat_keys = ["hit", "hurt", "kill", "death", "terminalWin", "terminalLoss", "timeout"]
        combat_abs = 0.0
        shaping_abs = 0.0
        for b in breakdowns:
            for key, value in b.items():
                val = abs(float(value))
                if key in combat_keys:
                    combat_abs += val
                else:
                    shaping_abs += val

        total_abs = combat_abs + shaping_abs
        if total_abs <= 1e-8:
            return 0.0, 0.0, 0.0
        return combat_abs * 100.0 / total_abs, shaping_abs * 100.0 / total_abs, total_abs

    def _populate_buffer(self, rollout: Dict[str, Any]) -> None:
        """Populate SB3's RolloutBuffer from worker rollout data."""
        buf = self.model.rollout_buffer
        buf.reset()

        worker_rollouts = cast(List[Dict[str, Any]], rollout.get("worker_rollouts", []))
        if not worker_rollouts:
            worker_rollouts = [rollout]

        obs_parts: List[NDArray[np.float32]] = []
        actions_parts: List[NDArray[np.float32]] = []
        rewards_parts: List[NDArray[np.float32]] = []
        starts_parts: List[NDArray[np.float32]] = []
        values_parts: List[NDArray[np.float32]] = []
        log_probs_parts: List[NDArray[np.float32]] = []
        adv_parts: List[NDArray[np.float32]] = []
        returns_parts: List[NDArray[np.float32]] = []

        # VecNormalize-style statistics track discounted environmental returns,
        # independently of critic predictions and GAE lambda. Each worker keeps
        # its own accumulator across rollout chunks.
        discounted_samples: List[float] = []
        for worker_index, chunk in enumerate(worker_rollouts):
            raw_rewards = np.asarray(chunk["rewards"], dtype=np.float32)
            starts = np.asarray(chunk["episode_starts"], dtype=np.float32)
            running_return = float(self.discounted_returns.get(worker_index, 0.0))
            for reward, episode_start in zip(raw_rewards, starts):
                if episode_start > 0.5:
                    running_return = 0.0
                running_return = self.gamma * running_return + float(reward)
                discounted_samples.append(running_return)
            if bool(chunk.get("last_done", False)):
                running_return = 0.0
            self.discounted_returns[worker_index] = running_return

        if discounted_samples:
            self.ret_rms.update(np.asarray(discounted_samples, dtype=np.float32))
        ret_std = float(np.sqrt(self.ret_rms.var + 1e-8))

        for chunk in worker_rollouts:
            rewards_arr = np.array(chunk["rewards"], dtype=np.float32)
            starts_arr = np.array(chunk["episode_starts"], dtype=np.float32)
            values_arr = np.array(chunk["values"], dtype=np.float32)
            # Truncation bootstrap (per-step gamma * V(s_post) at timeout indices,
            # 0 elsewhere). Applied AFTER reward normalization below so the
            # bootstrap shares the value head's (normalized) scale.
            trunc_values_arr = np.array(
                chunk.get("truncation_values", [0.0] * len(rewards_arr)),
                dtype=np.float32,
            )

            if ret_std > 1e-6:
                rewards_arr = rewards_arr / ret_std
            # Add the unnormalized bootstrap AFTER scaling raw rewards. The
            # value head is regressed against normalized returns, so V(s_post)
            # is already on the normalized scale.
            if np.any(trunc_values_arr != 0.0):
                rewards_arr = rewards_arr + self.gamma * trunc_values_arr

            advantages_arr, returns_arr = self._compute_chunk_gae(
                rewards=rewards_arr,
                episode_starts=starts_arr,
                values=values_arr,
                last_value=float(chunk["last_value"]),
                last_done=bool(chunk["last_done"]),
            )

            obs_parts.append(np.array(chunk["obs"], dtype=np.float32))
            actions_parts.append(np.array(chunk["actions"], dtype=np.float32))
            rewards_parts.append(rewards_arr)
            starts_parts.append(starts_arr)
            values_parts.append(values_arr)
            log_probs_parts.append(np.array(chunk["log_probs"], dtype=np.float32))
            adv_parts.append(advantages_arr)
            returns_parts.append(returns_arr)

        obs_arr = np.concatenate(obs_parts, axis=0)
        actions_arr = np.concatenate(actions_parts, axis=0)
        rewards_arr = np.concatenate(rewards_parts, axis=0)
        episode_starts_arr = np.concatenate(starts_parts, axis=0)
        values_arr = np.concatenate(values_parts, axis=0)
        log_probs_arr = np.concatenate(log_probs_parts, axis=0)
        advantages_arr = np.concatenate(adv_parts, axis=0)
        returns_arr = np.concatenate(returns_parts, axis=0)

        if obs_arr.shape[0] != buf.buffer_size:
            raise RuntimeError(
                f"RolloutBuffer size mismatch: expected {buf.buffer_size}, got {obs_arr.shape[0]}"
            )

        # Populate buffer directly; training uses these tensors via RolloutBuffer.get().
        buf.observations[:, 0, :] = obs_arr
        buf.actions[:, 0, :] = actions_arr
        buf.rewards[:, 0] = rewards_arr
        buf.episode_starts[:, 0] = episode_starts_arr
        buf.values[:, 0] = values_arr
        buf.log_probs[:, 0] = log_probs_arr
        buf.advantages[:, 0] = advantages_arr
        buf.returns[:, 0] = returns_arr
        buf.pos = buf.buffer_size
        buf.full = True

    def _log_episodes(self, rollout: Dict[str, Any], iteration: int) -> Tuple[float, float]:
        """Log episode statistics from the rollout."""
        ep_rewards = rollout.get("episode_rewards", [])
        ep_lengths = rollout.get("episode_lengths", [])
        ep_wins = rollout.get("episode_wins", [])
        ep_levels = rollout.get("episode_levels", [])
        ep_breakdowns = rollout.get("episode_reward_breakdowns", [])

        self.episode_rewards.extend(ep_rewards)
        self.episode_lengths.extend(ep_lengths)
        self.episode_wins.extend(ep_wins)
        self.episode_levels.extend(ep_levels)
        self.episode_reward_breakdowns.extend(ep_breakdowns)
        for scenario_id, win in zip(ep_levels, ep_wins):
            sid = int(scenario_id)
            stats = self.scenario_competence.setdefault(
                sid,
                {"episodes": 0.0, "wins": 0.0, "ema": 0.0},
            )
            stats["episodes"] += 1.0
            stats["wins"] += float(win)
            alpha = 0.05
            stats["ema"] = (1.0 - alpha) * stats["ema"] + alpha * float(win)
        if not self._explicit_levels:
            self.phase_episode_wins.extend(ep_wins)
            self.phase_episode_rewards.extend([float(r) for r in ep_rewards])
            # Cap phase history to avoid unbounded memory growth in long runs.
            # 5000 entries comfortably exceeds min_phase_episodes window sizes.
            _MAX_PHASE_HISTORY = 5000
            if len(self.phase_episode_wins) > _MAX_PHASE_HISTORY:
                self.phase_episode_wins = self.phase_episode_wins[-_MAX_PHASE_HISTORY:]
            if len(self.phase_episode_rewards) > _MAX_PHASE_HISTORY:
                self.phase_episode_rewards = self.phase_episode_rewards[-_MAX_PHASE_HISTORY:]

        for offset, ep_reward in enumerate(ep_rewards):
            absolute_episode = self.total_episodes + offset + 1
            breakdown = cast(Dict[str, float], ep_breakdowns[offset] if offset < len(ep_breakdowns) else {})
            scenario_id = ep_levels[offset] if offset < len(ep_levels) else -1
            win = ep_wins[offset] if offset < len(ep_wins) else 0
            print(
                f"  Episode {absolute_episode} | Scenario {scenario_id} | Win={win} | Reward={float(ep_reward):.3f} | "
                f"tick={float(breakdown.get('tick', 0.0)):.3f} hit={float(breakdown.get('hit', 0.0)):.3f} "
                f"hurt={float(breakdown.get('hurt', 0.0)):.3f} kill={float(breakdown.get('kill', 0.0)):.3f} "
                f"death={float(breakdown.get('death', 0.0)):.3f} winR={float(breakdown.get('terminalWin', 0.0)):.3f} "
                f"lossR={float(breakdown.get('terminalLoss', 0.0)):.3f} timeout={float(breakdown.get('timeout', 0.0)):.3f}"
            )
        self.total_episodes += len(ep_rewards)

        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-50:]
            recent_wins = self.episode_wins[-50:]
            recent_levels = self.episode_levels[-50:]
            avg_reward = np.mean(recent_rewards)
            avg_winrate = np.mean(recent_wins)
            elapsed = time.time() - self.start_time
            steps_per_sec = self.total_timesteps / max(elapsed, 1)

            level_stats = ""
            if len(self.levels) > 1:
                parts: List[str] = []
                for lvl in sorted(set(recent_levels)):
                    lvl_wins = [w for w, l in zip(recent_wins, recent_levels) if l == lvl]
                    if lvl_wins:
                        parts.append(f"S{lvl}:{np.mean(lvl_wins):.2f}")
                level_stats = " | " + " ".join(parts)

            print(
                f"Iter {iteration} | "
                f"Episodes: {self.total_episodes} | "
                f"Timesteps: {self.total_timesteps} | "
                f"Avg Reward (50): {avg_reward:.2f} | "
                f"Win Rate (50): {avg_winrate:.3f}{level_stats} | "
                f"Steps/s: {steps_per_sec:.0f} | "
                f"Time: {elapsed:.0f}s"
            )

            if self.episode_reward_breakdowns:
                recent_breakdowns = self.episode_reward_breakdowns[-50:]
                keys = ["tick", "hit", "hurt", "kill", "death", "terminalWin", "terminalLoss", "timeout", "approach", "dodge"]
                summary: List[str] = []
                for key in keys:
                    vals = [float(b.get(key, 0.0)) for b in recent_breakdowns]
                    summary.append(f"{key}={np.mean(vals):.3f}")
                print("  RewardBreakdown(50): " + " ".join(summary))

            return float(avg_reward), float(avg_winrate)
        return 0.0, 0.0

    def train(self, target_episodes: int = 100_000, max_timesteps: int = 80_000_000) -> None:
        """Main training loop."""
        append_log = self.load_model_path is not None and os.path.exists(self.training_log_path)
        csv_file: TextIO = open(self.training_log_path, "a" if append_log else "w", newline="")
        csv_writer = csv.writer(csv_file)
        if not append_log:
            csv_writer.writerow([
                "iteration", "timesteps", "episodes", "curriculum_phase", "active_scenarios", "phase_recent_winrate_500",
                "difficulty_band",
                "avg_reward_50", "avg_winrate_50",
                "avg_tick_50", "avg_hit_50", "avg_hurt_50", "avg_kill_50", "avg_death_50", "avg_terminal_win_50", "avg_terminal_loss_50", "avg_timeout_50", "avg_approach_50", "avg_dodge_50",
                "steps_per_sec", "elapsed_sec"
            ])
        csv_file.flush()
        self._write_run_manifest(status="running")

        iteration = 0
        next_checkpoint_episode = (
            self.total_episodes // self.checkpoint_episode_interval + 1
        ) * self.checkpoint_episode_interval
        next_eval_episode = (
            self.total_episodes // self.eval_interval_episodes + 1
        ) * self.eval_interval_episodes
        interrupt_reason = ""
        failure_reason = ""
        try:
            while self.total_episodes < target_episodes and self.total_timesteps < max_timesteps:
                iteration += 1

                curr_phase_name = self._get_curriculum_phase()["name"] if not self._explicit_levels else "Explicit levels"

                # Resample continuously from per-scenario competence. No policy
                # rollback is used, so learning progress is never discarded.
                if not self._explicit_levels:
                    self.levels = self._build_mixed_levels()
                self._update_difficulty_band()

                # 1. Send current weights to worker
                t0 = time.perf_counter()
                self._send_weights()
                t_weights = time.perf_counter() - t0

                # 2. Collect rollout (fast: all in TypeScript)
                t0 = time.perf_counter()
                rollout = self._collect_rollout(target_episodes)
                t_rollout = time.perf_counter() - t0

                self.total_timesteps += self.n_steps

                # 3. Populate SB3 buffer
                t0 = time.perf_counter()
                self._populate_buffer(rollout)
                t_buffer = time.perf_counter() - t0

                # 4. Update progress for learning rate schedules
                timestep_progress = min(1.0, max(0.0, self.total_timesteps / max(max_timesteps, 1)))
                cast(Any, self.model)._current_progress_remaining = 1.0 - timestep_progress
                current_ent_coef = self.ent_coef_final + (self.ent_coef_start - self.ent_coef_final) * (1.0 - timestep_progress)
                cast(Any, self.model).ent_coef = float(current_ent_coef)
                cast(Any, self.model).num_timesteps = self.total_timesteps

                # 5. PPO gradient updates
                t0 = time.perf_counter()
                cast(Any, self.model).train()
                t_train = time.perf_counter() - t0

                # 5b. Log SB3 training metrics periodically
                if iteration <= 5 or iteration % 50 == 0:
                    logger = cast(Any, self.model).logger
                    nv = logger.name_to_value
                    pg = nv.get("train/policy_gradient_loss", 0)
                    vl = nv.get("train/value_loss", 0)
                    ent = nv.get("train/entropy_loss", 0)
                    clip_frac = nv.get("train/clip_fraction", 0)
                    approx_kl = nv.get("train/approx_kl", 0)
                    expl_var = nv.get("train/explained_variance", 0)
                    print(
                        f"  PPO: pg_loss={pg:.4f} val_loss={vl:.4f} ent={ent:.4f} "
                        f"clip_frac={clip_frac:.3f} kl={approx_kl:.4f} expl_var={expl_var:.3f}"
                    )
                    self._log_per_head_entropy()
                    if not np.isfinite(float(vl)):
                        raise RuntimeError("PPO value_loss became non-finite; aborting to prevent corrupted checkpoints.")
                    if self.target_kl > 0 and float(approx_kl) > self.target_kl * 4.0:
                        print(
                            f"  WARNING: PPO approx_kl spike ({float(approx_kl):.4f}) exceeded "
                            f"4x target_kl ({self.target_kl:.4f}). SB3's early-stop should have "
                            f"halted further epochs; continuing to next iteration."
                        )
                    if float(clip_frac) > 0.8:
                        print("  WARNING: clip_fraction > 0.8; updates may be too aggressive.")

                # 6. Log
                avg_reward, avg_winrate = self._log_episodes(rollout, iteration)

                elapsed = time.time() - self.start_time
                steps_per_sec = self.total_timesteps / max(elapsed, 1)

                if iteration <= 3 or iteration % 10 == 0:
                    print(
                        f"  Timing: weights={t_weights:.2f}s rollout={t_rollout:.2f}s "
                        f"buffer={t_buffer:.2f}s train={t_train:.2f}s"
                    )

                if self.episode_reward_breakdowns:
                    recent_breakdowns = self.episode_reward_breakdowns[-50:]
                    avg_tick = float(np.mean([float(b.get("tick", 0.0)) for b in recent_breakdowns]))
                    avg_hit = float(np.mean([float(b.get("hit", 0.0)) for b in recent_breakdowns]))
                    avg_hurt = float(np.mean([float(b.get("hurt", 0.0)) for b in recent_breakdowns]))
                    avg_kill = float(np.mean([float(b.get("kill", 0.0)) for b in recent_breakdowns]))
                    avg_death = float(np.mean([float(b.get("death", 0.0)) for b in recent_breakdowns]))
                    avg_terminal_win = float(np.mean([float(b.get("terminalWin", 0.0)) for b in recent_breakdowns]))
                    avg_terminal_loss = float(np.mean([float(b.get("terminalLoss", 0.0)) for b in recent_breakdowns]))
                    avg_timeout = float(np.mean([float(b.get("timeout", 0.0)) for b in recent_breakdowns]))
                    avg_approach = float(np.mean([float(b.get("approach", 0.0)) for b in recent_breakdowns]))
                    avg_dodge = float(np.mean([float(b.get("dodge", 0.0)) for b in recent_breakdowns]))
                else:
                    avg_tick = avg_hit = avg_hurt = avg_kill = avg_death = avg_terminal_win = avg_terminal_loss = avg_timeout = avg_approach = avg_dodge = 0.0

                csv_writer.writerow([
                    iteration,
                    self.total_timesteps,
                    self.total_episodes,
                    curr_phase_name,
                    ",".join(str(l) for l in self.levels),
                    f"{self._phase_recent_win_rate():.3f}" if not self._explicit_levels else "",
                    f"{self.curriculum_difficulty_band:.3f}",
                    f"{avg_reward:.3f}" if self.episode_rewards else "0",
                    f"{avg_winrate:.3f}" if self.episode_wins else "0",
                    f"{avg_tick:.3f}",
                    f"{avg_hit:.3f}",
                    f"{avg_hurt:.3f}",
                    f"{avg_kill:.3f}",
                    f"{avg_death:.3f}",
                    f"{avg_terminal_win:.3f}",
                    f"{avg_terminal_loss:.3f}",
                    f"{avg_timeout:.3f}",
                    f"{avg_approach:.3f}",
                    f"{avg_dodge:.3f}",
                    f"{steps_per_sec:.0f}",
                    f"{elapsed:.1f}",
                ])
                csv_file.flush()
                snapshot = self._build_iteration_snapshot(
                    iteration,
                    avg_reward,
                    avg_winrate,
                    avg_tick,
                    avg_hit,
                    avg_hurt,
                    avg_kill,
                    avg_death,
                    avg_terminal_win,
                    avg_terminal_loss,
                    avg_timeout,
                    avg_approach,
                    avg_dodge,
                    steps_per_sec,
                    elapsed,
                )
                self._write_iteration_snapshot(snapshot)
                self._write_run_manifest(status="running")

                # 7. Track best training-window model separately from eval best.
                if len(self.episode_rewards) >= 20 and avg_winrate > self.best_train_win_rate:
                    self.best_train_win_rate = avg_winrate
                    self._save_model_checkpoint(self.best_train_model_path)
                    print(f"  ** New best training-window model: {avg_winrate:.3f} **")

                # 7b. Periodic deterministic eval for robust best checkpoint selection.
                while self.total_episodes >= next_eval_episode:
                    try:
                        eval_win_rate = self._evaluate_policy()
                        print(
                            f"  Eval | Episodes={self.total_episodes} | "
                            f"WinRate={eval_win_rate:.3f} | Wilson95Lower={self.last_global_eval_lower_bound:.3f} "
                            f"on levels={self.eval_levels}"
                        )
                        print(
                            "  EvalByLevel | "
                            + " ".join(
                                f"L{level}={int(details['wins'])}/{int(details['episodes'])}"
                                for level, details in sorted(self.last_global_eval_details.items())
                            )
                        )
                        if self.last_global_eval_lower_bound > self.best_eval_lower_bound:
                            self.best_eval_win_rate = eval_win_rate
                            self.best_eval_lower_bound = self.last_global_eval_lower_bound
                            self.best_win_rate = eval_win_rate
                            self._save_model_checkpoint(self.best_model_path)
                            self._export_policy_onnx(self.best_model_path)
                            print(f"  ** New best eval model! Win rate: {eval_win_rate:.3f} **")
                    except Exception as eval_exc:
                        print(f"  WARNING: _evaluate_policy failed ({eval_exc}) - continuing with phase gate checks.")

                    if not self._explicit_levels:
                        phase_levels = self._get_curriculum_levels()
                        phase_eval_source = "held-out-randomized"
                        try:
                            phase_eval = self._evaluate_levels(
                                phase_levels,
                                self.eval_episodes,
                                self.eval_seed_start + 50_000_000 + self.total_episodes * 10,
                            )
                            phase_lower_bound = self.last_eval_lower_bound
                            self.last_phase_eval_lower_bound = phase_lower_bound
                            self.last_phase_eval_details = dict(self.last_eval_details)
                        except Exception as phase_eval_exc:
                            phase_eval_source = "training-window-fallback"
                            phase_eval = self._phase_recent_win_rate()
                            phase_window = self.phase_episode_wins[-500:]
                            phase_lower_bound = self._wilson_lower_bound(
                                int(sum(phase_window)),
                                len(phase_window),
                            )
                            self.last_phase_eval_lower_bound = phase_lower_bound
                            self.last_phase_eval_details = {}
                            print(
                                f"  WARNING: phase randomized eval failed ({phase_eval_exc}) - "
                                f"using phase_recent_winrate_500 fallback={phase_eval:.3f}"
                            )

                        self.last_phase_eval_win_rate = phase_eval
                        current = self._get_curriculum_phase()
                        required_wr = float(current.get("required_win_rate", 0.40))
                        if phase_lower_bound >= required_wr:
                            self.phase_eval_pass_streak += 1
                        else:
                            self.phase_eval_pass_streak = 0

                        print(
                            f"  PhaseEval[{phase_eval_source}] | Phase={current['name']} | WR={phase_eval:.3f} | "
                            f"Wilson95Lower={phase_lower_bound:.3f} | required={required_wr:.3f} | "
                            f"pass_streak={self.phase_eval_pass_streak}"
                        )
                        if self.last_phase_eval_details:
                            print(
                                "  PhaseEvalByLevel | "
                                + " ".join(
                                    f"L{level}={int(details['wins'])}/{int(details['episodes'])}"
                                    for level, details in sorted(self.last_phase_eval_details.items())
                                )
                            )

                        phase_episodes = self._phase_episode_count()
                        min_phase_episodes = int(current.get("min_phase_episodes") or 0)
                        can_promote = (
                            self.current_phase_index < len(self.curriculum) - 1
                            and phase_episodes >= min_phase_episodes
                            and self.phase_eval_pass_streak >= self.phase_pass_evals_required
                            and self._phase_is_stable()
                        )

                        if can_promote:
                            self._save_stable_checkpoint(self.current_phase_index, phase_eval)
                            transition = self._advance_curriculum_from_eval()
                            if transition is not None:
                                previous_phase, next_phase = transition
                                print(
                                    f"\n*** Competence-gated phase transition at episode {self.total_episodes}: "
                                    f"{previous_phase['name']} -> {next_phase['name']} | "
                                    f"phase_eval={phase_eval:.3f}, lower={phase_lower_bound:.3f} ***\n"
                                )
                    next_eval_episode += self.eval_interval_episodes
                    self._write_run_manifest(status="running")

                # 8. Episode-based periodic checkpoints
                while self.total_episodes >= next_checkpoint_episode:
                    ckpt_path = os.path.join(
                        self.output_dir, f"treads_ppo_ep{next_checkpoint_episode}"
                    )
                    self._save_model_checkpoint(ckpt_path)
                    next_checkpoint_episode += self.checkpoint_episode_interval

        except KeyboardInterrupt:
            interrupt_reason = "Training interrupted by user."
            print("\nTraining interrupted by user.")
        except BaseException as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            print(f"\nTraining failed: {failure_reason}")
            traceback.print_exc()
            raise
        finally:
            self.run_finished_at = datetime.now(timezone.utc)
            if self.episode_reward_breakdowns:
                all_terminal_pct, all_shaping_pct, _ = self._reward_source_percentages(self.episode_reward_breakdowns)
                late_breakdowns = self.episode_reward_breakdowns[-500:]
                late_terminal_pct, late_shaping_pct, _ = self._reward_source_percentages(late_breakdowns)
                all_tick_abs = sum(abs(float(b.get("tick", 0.0))) for b in self.episode_reward_breakdowns)
                all_abs = sum(sum(abs(float(v)) for v in b.values()) for b in self.episode_reward_breakdowns)
                tick_share = (all_tick_abs * 100.0 / all_abs) if all_abs > 1e-8 else 0.0

                print(
                    "RewardSource(All): "
                    f"combat+terminal={all_terminal_pct:.1f}% "
                    f"other={all_shaping_pct:.1f}%"
                )
                print(
                    "RewardSource(Late500): "
                    f"combat+terminal={late_terminal_pct:.1f}% "
                    f"other={late_shaping_pct:.1f}%"
                )
                print(f"TickPenaltyShare(All): {tick_share:.1f}%")
                if tick_share > 10.0:
                    print(
                        "WARNING: Tick penalty exceeds 10% of total absolute reward; "
                        "consider reducing the per-tick penalty further."
                    )

            csv_file.close()
            # Save final model
            self._save_model_checkpoint(self.final_model_path)
            self._export_policy_onnx(self.final_model_path)
            final_breakdowns = self.episode_reward_breakdowns[-50:] if self.episode_reward_breakdowns else []
            final_snapshot = self._build_iteration_snapshot(
                iteration,
                float(np.mean(self.episode_rewards[-50:])) if self.episode_rewards else 0.0,
                float(np.mean(self.episode_wins[-50:])) if self.episode_wins else 0.0,
                float(np.mean([float(b.get("tick", 0.0)) for b in final_breakdowns])) if final_breakdowns else 0.0,
                float(np.mean([float(b.get("hit", 0.0)) for b in final_breakdowns])) if final_breakdowns else 0.0,
                float(np.mean([float(b.get("hurt", 0.0)) for b in final_breakdowns])) if final_breakdowns else 0.0,
                float(np.mean([float(b.get("kill", 0.0)) for b in final_breakdowns])) if final_breakdowns else 0.0,
                float(np.mean([float(b.get("death", 0.0)) for b in final_breakdowns])) if final_breakdowns else 0.0,
                float(np.mean([float(b.get("terminalWin", 0.0)) for b in final_breakdowns])) if final_breakdowns else 0.0,
                float(np.mean([float(b.get("terminalLoss", 0.0)) for b in final_breakdowns])) if final_breakdowns else 0.0,
                float(np.mean([float(b.get("timeout", 0.0)) for b in final_breakdowns])) if final_breakdowns else 0.0,
                float(np.mean([float(b.get("approach", 0.0)) for b in final_breakdowns])) if final_breakdowns else 0.0,
                float(np.mean([float(b.get("dodge", 0.0)) for b in final_breakdowns])) if final_breakdowns else 0.0,
                self.total_timesteps / max(time.time() - self.start_time, 1),
                time.time() - self.start_time,
            )
            self._write_iteration_snapshot(final_snapshot)
            if failure_reason:
                final_status = "failed"
                final_error = failure_reason
            elif interrupt_reason:
                final_status = "interrupted"
                final_error = interrupt_reason
            elif self.total_episodes >= target_episodes:
                final_status = "completed"
                final_error = ""
            else:
                final_status = "budget_exhausted"
                final_error = (
                    f"Transition budget exhausted at {self.total_episodes}/{target_episodes} episodes."
                )
            self._write_run_manifest(status=final_status, error=final_error)
            print(f"Final model saved to {self.final_model_path}")
            self.close()

    def close(self) -> None:
        """Clean up all worker processes."""
        for w in self.workers:
            try:
                self._close_worker_handle(w)
            except Exception:
                pass
        self.workers = []

    def send_weights(self) -> None:
        """Public wrapper used by diagnostics/scripts."""
        self._send_weights()

    def collect_rollout(self, target_episodes: int) -> Dict[str, Any]:
        """Public wrapper used by diagnostics/scripts."""
        return self._collect_rollout(target_episodes)

    def populate_buffer(self, rollout: Dict[str, Any]) -> None:
        """Public wrapper used by diagnostics/scripts."""
        self._populate_buffer(rollout)


def train() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=str, default="", help="Optional comma-separated level numbers; omit to use curriculum")
    parser.add_argument("--timesteps", type=int, default=0, help="Max timesteps (default: auto-computed from target-episodes × max-ticks)")
    parser.add_argument("--target-episodes", type=int, default=100_000, help="Stop when this many episodes are collected")
    parser.add_argument("--load-model", type=str, default="", help="Optional .zip model path to resume from")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output directory override")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Checkpoint interval in episodes")
    parser.add_argument("--replay-interval", type=int, default=250, help="Replay save interval in episodes")
    parser.add_argument("--gamma", type=float, default=0.997, help="Discount factor (default: 0.997)")
    parser.add_argument("--ent-coef", type=float, default=0.015, help="Entropy coefficient (default: 0.015)")
    parser.add_argument("--ent-coef-final", type=float, default=0.003, help="Final entropy coefficient at end of training (default: 0.003)")
    parser.add_argument("--max-ticks", type=int, default=720, help="Max ticks per episode (default: 720)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--n-steps", type=int, default=8192, help="Rollout steps per iteration (default: 8192)")
    parser.add_argument("--clip-range", type=float, default=0.15, help="PPO clip range (default: 0.15)")
    parser.add_argument("--clip-range-final", type=float, default=0.08, help="Final PPO clip range at end of training (default: 0.08)")
    parser.add_argument("--target-kl", type=float, default=0.02, help="PPO target KL early-stop threshold (default: 0.02)")
    parser.add_argument("--phase-pass-evals", type=int, default=2, help="Consecutive confidence-bound phase eval passes required before promotion")
    parser.add_argument("--eval-interval", type=int, default=500, help="Run deterministic eval every N collected episodes (default: 500)")
    parser.add_argument("--eval-episodes", type=int, default=12, help="Held-out randomized eval episodes per level (default: 12)")
    parser.add_argument("--eval-levels", type=str, default="1,2,3,4,5,6,7,8,9", help="Comma-separated levels for deterministic eval")
    parser.add_argument("--max-update-steps", type=int, default=16384, help="Cap total rollout steps per PPO update (default: 16384)")
    cpu_count = os.cpu_count() or 1
    default_workers = max(1, min(cpu_count - 4, 12))  # leave cores for OS + Python; cap at 12
    parser.add_argument("--num-workers", type=int, default=default_workers, help=f"Number of parallel rollout workers (default: {default_workers}, detected {cpu_count} cores)")
    args = parser.parse_args()

    # Keep updates fresh: enforce worker minimum while capping total steps per PPO update.
    min_steps_per_worker = 1024
    effective_n_steps = max(args.n_steps, args.num_workers * min_steps_per_worker)
    effective_n_steps = min(effective_n_steps, max(args.max_update_steps, args.num_workers))
    remainder = effective_n_steps % args.num_workers
    if remainder != 0:
        effective_n_steps -= remainder
        effective_n_steps = max(args.num_workers, effective_n_steps)
    if effective_n_steps != args.n_steps:
        print(
            "Auto-scaling n_steps: "
            f"{args.n_steps} -> {effective_n_steps} "
            f"({args.num_workers} workers, min {min_steps_per_worker}/worker, max update {args.max_update_steps}, evenly divisible)"
        )

    levels = [int(x) for x in args.levels.split(",") if x.strip()] if args.levels else None
    eval_levels = [int(x) for x in args.eval_levels.split(",") if x.strip()]

    trainer = HybridTrainer(
        levels=levels,
        n_steps=effective_n_steps,
        batch_size=512,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        learning_rate=args.lr,
        clip_range=args.clip_range,
        clip_range_final=args.clip_range_final,
        ent_coef=args.ent_coef,
        ent_coef_final=args.ent_coef_final,
        target_kl=args.target_kl,
        max_episode_steps=args.max_ticks,
        output_dir=args.output_dir or None,
        load_model_path=args.load_model or None,
        checkpoint_episode_interval=args.checkpoint_interval,
        replay_episode_interval=args.replay_interval,
        eval_interval_episodes=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_levels=eval_levels,
        num_workers=args.num_workers,
        phase_pass_evals_required=args.phase_pass_evals,
    )

    def request_graceful_shutdown(signum: int, _frame: Any) -> None:
        print(f"\nReceived signal {signum}; finishing the current cleanup path.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_graceful_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_graceful_shutdown)

    print(f"Starting hybrid training (rollout in TypeScript, PPO in Python, {args.num_workers} worker(s))")
    print(f"n_steps: {effective_n_steps} ({effective_n_steps // args.num_workers} per worker)")
    if levels is None:
        print(f"Initial curriculum: {trainer.levels} ({trainer.curriculum[0]['name']})")
    else:
        print(f"Explicit scenarios: {trainer.levels} (curriculum disabled)")
    print(
        f"Gamma: {args.gamma} | EntCoef: {args.ent_coef}->{args.ent_coef_final} | "
        f"MaxTicks: {args.max_ticks} | ClipRange: {args.clip_range}->{args.clip_range_final} | "
        f"TargetKL: {args.target_kl}"
    )
    print(f"Rollout worker: {ROLLOUT_WORKER_PATH}")
    max_ts = args.timesteps
    if max_ts <= 0:
        # Auto-compute: target_episodes × max_ticks gives a reasonable estimate
        # of total timesteps, with 1.5× headroom for partial episodes in rollout chunks
        max_ts = int(args.target_episodes * args.max_ticks * 1.5)
        print(f"Auto-computed max_timesteps: {max_ts:,} ({args.target_episodes} episodes × {args.max_ticks} ticks × 1.5)")
    trainer.train(target_episodes=args.target_episodes, max_timesteps=max_ts)


if __name__ == "__main__":
    train()
