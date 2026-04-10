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
import subprocess
import threading
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
from runtime_codec import decode_continuous_action
from treads_env import OBS_SIZE, TreadsEnv

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
    if not (os.path.exists(pb2_path) and os.path.exists(pb2_grpc_path)):
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
    # ── Phase 4→5: Multi-enemy ramp (gentle bridge via 145-148, then full) ──
    {"name": "Phase 4.5a (unarmed 2nd target)", "scenario_ids": [143, 145, 146], "min_phase_episodes": 2400, "force_phase_episodes": 10000, "required_win_rate": 0.42, "player_max_ammo": 4},
    {"name": "Phase 4.5b (armed multi open)", "scenario_ids": [146, 147, 148], "min_phase_episodes": 2800, "force_phase_episodes": 12000, "required_win_rate": 0.40, "player_max_ammo": 4},
    {"name": "Phase 4.5c (multi + obstacles)", "scenario_ids": [148, 151, 152], "min_phase_episodes": 3200, "force_phase_episodes": 14000, "required_win_rate": 0.38, "player_max_ammo": 4},
    {"name": "Phase 5 (multi-enemy full)", "scenario_ids": [151, 152, 153], "min_phase_episodes": 3600, "force_phase_episodes": 15000, "required_win_rate": 0.36, "player_max_ammo": 4},
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
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 1e-4,
        clip_range: float = 0.15,
        clip_range_final: float = 0.08,
        ent_coef: float = 0.015,
        ent_coef_final: float = 0.003,
        target_kl: float = 0.02,
        max_episode_steps: int = 720,
        output_dir: Optional[str] = None,
        load_model_path: Optional[str] = None,
        checkpoint_episode_interval: int = 500,
        replay_episode_interval: int = 250,
        eval_interval_episodes: int = 500,
        eval_episodes: int = 24,
        eval_levels: Optional[List[int]] = None,
        eval_seed_start: int = 25000,
        seed: int = 42,
        num_workers: int = 1,
        stability_mode: bool = True,
        phase_pass_evals_required: int = 3,
        phase_fail_evals_before_rollback: int = 3,
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
        self.procedural_levels = not self._explicit_levels
        self.curriculum_difficulty_band = 0.0 if not self._explicit_levels else 0.5
        self.stability_mode = bool(stability_mode)
        self.phase_pass_evals_required = max(1, int(phase_pass_evals_required))
        self.phase_fail_evals_before_rollback = max(1, int(phase_fail_evals_before_rollback))
        self.phase_eval_pass_streak = 0
        self.phase_eval_fail_streak = 0
        self.last_phase_eval_win_rate = 0.0
        self.stable_checkpoints: Dict[int, str] = {}
        self.peak_phase_index: int = 0
        self.last_rollback_episode: int = 0
        self.ret_rms = RunningMeanStd()
        if load_model_path:
            rms_path = os.path.join(os.path.dirname(load_model_path), "ret_rms_state.json")
            if os.path.exists(rms_path):
                try:
                    with open(rms_path, "r") as f:
                        self.ret_rms = RunningMeanStd.from_dict(json.load(f))
                    print(f"Loaded reward normalizer state from: {rms_path} (var={self.ret_rms.var:.4f}, count={self.ret_rms.count:.0f})")
                except Exception as exc:
                    print(f"WARNING: failed to load reward normalizer state ({exc}), starting fresh")
        self._ent_coef_boost: float = 0.0
        self._stale_eval_count: int = 0
        self._last_phase_eval_value: Optional[float] = None

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

        class _DummyContinuousActionEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
            metadata = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
                # [move_x, move_y, aim_signal, fire_signal, bomb_signal] in [-1, 1]
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

            def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[Dict[str, Any]] = None,
            ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
                super().reset(seed=seed)
                return np.zeros((OBS_SIZE,), dtype=np.float32), {}

            def step(self, action: NDArray[np.float32]) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
                _ = action
                return np.zeros((OBS_SIZE,), dtype=np.float32), 0.0, True, False, {}

        # Create a dummy continuous-action env so SB3 policy/distribution match worker sampling.
        dummy_env = _DummyContinuousActionEnv()
        clip_schedule = linear_schedule_between(clip_range, clip_range_final)
        lr_schedule = linear_schedule_between(learning_rate, 0.0)
        if load_model_path and os.path.exists(load_model_path):
            print(f"Loading PPO model from: {load_model_path}")
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
                clip_range_vf=None,
                ent_coef=ent_coef,
                target_kl=target_kl,
                policy_kwargs=dict(optimizer_kwargs=dict(eps=1e-5)),
            ))
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
                clip_range_vf=None,
                ent_coef=ent_coef,
                target_kl=target_kl,
                device="cpu",
                policy_kwargs=dict(net_arch=[256, 256], optimizer_kwargs=dict(eps=1e-5)),
                seed=seed,
            )
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

        # Start rollout workers
        self.workers: List[Dict[str, Any]] = []
        self._start_workers()

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
        self.start_time = time.time()
        self._write_run_manifest(status="initializing")
        with open(self.metrics_history_path, "w", encoding="utf-8"):
            pass

    def _evaluate_policy(self) -> float:
        """Deterministic eval on fixed levels/seeds for stable model selection."""
        return self._evaluate_levels(self.eval_levels, self.eval_episodes, self.eval_seed_start)

    def _evaluate_levels(self, levels: List[int], episodes_per_level: int, seed_start: int) -> float:
        """Deterministic eval on an explicit level set."""
        wins = 0
        episodes = 0
        seed = seed_start

        for level in levels:
            for _ in range(episodes_per_level):
                env = TreadsEnv(level=level, seed_start=seed, max_episode_steps=self.max_episode_steps)
                try:
                    obs, _ = env.reset()
                    done = False
                    info: Dict[str, Any] = {}
                    while not done:
                        action, _ = cast(Any, self.model).predict(obs, deterministic=True)
                        obs_raw = cast(Optional[Dict[str, Any]], getattr(env, "_last_obs_raw", None))
                        decoded = decode_continuous_action(action, obs_raw)
                        obs, _reward, terminated, truncated, info = env.step(decoded)
                        done = terminated or truncated
                    result = cast(Dict[str, Any], info.get("result", {}))
                    wins += int(bool(result.get("win", False)))
                    episodes += 1
                finally:
                    env.close()
                    seed += 1

        return float(wins / max(episodes, 1))

    def _save_ret_rms(self) -> None:
        """Persist reward normalizer state alongside model checkpoints."""
        try:
            with open(self._ret_rms_path, "w") as f:
                json.dump(self.ret_rms.to_dict(), f)
        except Exception as exc:
            print(f"WARNING: failed to save reward normalizer state ({exc})")

    def _save_stable_checkpoint(self, phase_index: int, eval_win_rate: float) -> None:
        path = os.path.join(self.output_dir, f"treads_ppo_phase{phase_index}_stable")
        cast(Any, self.model).save(path)
        self._save_ret_rms()
        self.stable_checkpoints[phase_index] = path + ".zip"
        print(
            f"  Stable checkpoint saved for phase {phase_index + 1} "
            f"(eval wr={eval_win_rate:.3f}): {self.stable_checkpoints[phase_index]}"
        )

    def _restore_stable_checkpoint(self) -> bool:
        # Find the best available checkpoint at or below the current phase
        ckpt_path: Optional[str] = None
        for idx in range(self.current_phase_index, -1, -1):
            candidate = self.stable_checkpoints.get(idx)
            if candidate and os.path.exists(candidate):
                ckpt_path = candidate
                break
        if not ckpt_path:
            return False
        try:
            cast(Any, self.model).set_parameters(ckpt_path, exact_match=False, device="cpu")
            print(f"  Restored stable checkpoint: {ckpt_path}")
            return True
        except Exception as exc:
            print(f"  WARNING: failed to restore stable checkpoint ({exc})")
            return False

    def _advance_curriculum_from_eval(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if self._explicit_levels or self.current_phase_index >= len(self.curriculum) - 1:
            return None
        previous = self._get_curriculum_phase()
        completed_phases = self.curriculum[: self.current_phase_index + 1]
        recent_phases = completed_phases[-3:] if len(completed_phases) > 3 else completed_phases
        self.rehearsal_ids = [sid for phase in recent_phases for sid in phase["scenario_ids"]]
        self.current_phase_index += 1
        self.peak_phase_index = max(self.peak_phase_index, self.current_phase_index)
        self.levels = self._build_mixed_levels()
        self.phase_episode_wins = []
        self.phase_episode_rewards = []
        self.phase_start_episode = self.total_episodes
        self.phase_eval_pass_streak = 0
        self.phase_eval_fail_streak = 0
        self.last_phase_eval_win_rate = 0.0
        self._stale_eval_count = 0
        self._last_phase_eval_value = None
        return previous, self._get_curriculum_phase()

    def _rollback_curriculum_from_eval(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if self._explicit_levels or self.current_phase_index <= 0:
            return None
        # Enforce depth limit with relaxation after sustained failure
        rollback_floor = self._rollback_floor()
        if self.current_phase_index - 1 < rollback_floor:
            print(f"  Rollback blocked: would go below floor phase {rollback_floor + 1} (peak={self.peak_phase_index + 1})")
            return None
        # Enforce cooldown: require 2500+ episodes between rollbacks
        if self.total_episodes - self.last_rollback_episode < 2500:
            print(f"  Rollback blocked: cooldown ({self.total_episodes - self.last_rollback_episode}/2500 episodes)")
            return None
        previous = self._get_curriculum_phase()
        self.current_phase_index -= 1
        completed_phases = self.curriculum[: self.current_phase_index + 1]
        recent_phases = completed_phases[-3:] if len(completed_phases) > 3 else completed_phases
        self.rehearsal_ids = [sid for phase in recent_phases for sid in phase["scenario_ids"]]
        self.levels = self._build_mixed_levels()
        # Seed phase history with neutral data instead of wiping
        target_wr = float(self._get_curriculum_phase().get("required_win_rate", 0.40))
        seed_count = 100
        self.phase_episode_wins = [1 if i < int(seed_count * target_wr) else 0 for i in range(seed_count)]
        self.phase_episode_rewards = [0.0] * seed_count
        self.phase_start_episode = self.total_episodes
        self.phase_eval_pass_streak = 0
        self.phase_eval_fail_streak = 0
        self.last_phase_eval_win_rate = 0.0
        self._stale_eval_count = 0
        self._last_phase_eval_value = None
        self.last_rollback_episode = self.total_episodes
        self._ent_coef_boost = self.ent_coef_start * 0.5
        return previous, self._get_curriculum_phase()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _json_dump(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

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
                "activeScenarios": self.levels,
                "rehearsalScenarios": self.rehearsal_ids,
                "difficultyBand": self.curriculum_difficulty_band,
                "playerMaxAmmo": self._get_player_max_ammo(),
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
            "phaseEvalFailStreak": self.phase_eval_fail_streak if not self._explicit_levels else None,
            "lastPhaseEvalWinRate": self.last_phase_eval_win_rate if not self._explicit_levels else None,
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
            ["node", ROLLOUT_WORKER_PATH, "--port", str(port)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
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

    def _get_curriculum_levels(self) -> List[int]:
        return list(self._get_curriculum_phase()["scenario_ids"])

    def _phase_episode_count(self) -> int:
        return self.total_episodes - self.phase_start_episode

    def _target_current_share(self) -> float:
        """Adaptive current-phase share with stronger rehearsal during collapse."""
        if self._explicit_levels:
            return 1.0
        anneal_episodes = 2000.0
        progress = min(1.0, max(0.0, self._phase_episode_count() / anneal_episodes))
        base = 0.60 + 0.20 * progress
        phase_wr = self._phase_recent_win_rate()
        # When current phase is unstable, increase rehearsal to prevent forgetting.
        if phase_wr < 0.12:
            return min(base, 0.45)
        if phase_wr < 0.20:
            return min(base, 0.55)
        if phase_wr < 0.30:
            return min(base, 0.65)
        return base

    def _build_mixed_levels(self) -> List[int]:
        """Build level list with adaptive rehearsal mix (starts ~60/40, anneals to ~80/20)."""
        current = self._get_curriculum_levels()
        if not self.rehearsal_ids:
            return current
        rehearsal_unique = list(set(self.rehearsal_ids))
        n_current = len(current)
        n_rehearsal = len(rehearsal_unique)
        target_current = self._target_current_share()
        target_rehearsal = max(1e-6, 1.0 - target_current)
        # Scale current repetitions to approximate target mix while keeping scenario diversity.
        current_reps = max(1, round((target_current / target_rehearsal) * (n_rehearsal / max(n_current, 1))))
        return current * current_reps + rehearsal_unique

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

    def _maybe_advance_curriculum(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        if self._explicit_levels or self.current_phase_index >= len(self.curriculum) - 1:
            return None
        current = self._get_curriculum_phase()
        phase_episodes = self._phase_episode_count()
        min_phase_episodes = current.get("min_phase_episodes")
        if min_phase_episodes is None:
            return None
        if phase_episodes < int(min_phase_episodes):
            return None

        win_rate = self._phase_recent_win_rate()
        required_wr = float(current.get("required_win_rate", 0.40))
        stable = self._phase_is_stable()

        meets_regular_criteria = win_rate >= required_wr and stable
        if not meets_regular_criteria:
            # Forced advance is only allowed much later and only if reward trend is not degrading.
            force_phase_episodes = current.get("force_phase_episodes")
            if force_phase_episodes is None or phase_episodes < int(force_phase_episodes):
                return None

            reward_trend = self._phase_reward_trend(window=200)
            minimally_competent = win_rate >= (required_wr * 0.95)
            if reward_trend <= 0.0 or not minimally_competent or not stable:
                return None

            print(
                f"  *** FORCED ADVANCE at episode {self.total_episodes} "
                f"(phase_episodes={phase_episodes}, win rate {win_rate:.3f} < {required_wr:.2f}, "
                f"reward trend={reward_trend:.3f}) ***"
            )

        previous = current
        # Keep only scenario_ids from the most recent 3 completed phases,
        # including the phase we are transitioning out of right now.
        completed_phases = self.curriculum[: self.current_phase_index + 1]
        recent_phases = completed_phases[-3:] if len(completed_phases) > 3 else completed_phases
        self.rehearsal_ids = [sid for phase in recent_phases for sid in phase["scenario_ids"]]
        self.current_phase_index += 1
        self.peak_phase_index = max(self.peak_phase_index, self.current_phase_index)
        self.levels = self._build_mixed_levels()
        self.phase_episode_wins = []
        self.phase_episode_rewards = []
        self.phase_start_episode = self.total_episodes
        self._stale_eval_count = 0
        self._last_phase_eval_value = None
        return previous, self._get_curriculum_phase(), win_rate

    def _rollback_floor(self) -> int:
        """Compute rollback floor. Relaxes after sustained failure (5000+ episodes stuck)."""
        base_floor = max(0, self.peak_phase_index - 2)
        stuck_episodes = self._phase_episode_count()
        if stuck_episodes >= 10000 and self._phase_recent_win_rate() < 0.05:
            # Deeply stuck — allow going back up to 4 phases below peak
            return max(0, self.peak_phase_index - 4)
        if stuck_episodes >= 5000 and self._phase_recent_win_rate() < 0.10:
            # Moderately stuck — allow going back up to 3 phases below peak
            return max(0, self.peak_phase_index - 3)
        return base_floor

    def _maybe_rollback_curriculum(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """Rollback one phase when sustained collapse is detected in current phase."""
        if self._explicit_levels or self.current_phase_index <= 0:
            return None

        phase_episodes = self._phase_episode_count()
        if phase_episodes < 1200:
            return None

        win_rate = self._phase_recent_win_rate()
        reward_trend = self._phase_reward_trend(window=200)
        if win_rate >= 0.25 or reward_trend > 0.0:
            return None

        # Enforce depth limit with relaxation after sustained failure
        rollback_floor = self._rollback_floor()
        if self.current_phase_index - 1 < rollback_floor:
            print(f"  Rollback blocked: would go below floor phase {rollback_floor + 1} (peak={self.peak_phase_index + 1})")
            return None
        # Enforce cooldown: require 2500+ episodes between rollbacks
        if self.total_episodes - self.last_rollback_episode < 2500:
            print(f"  Rollback blocked: cooldown ({self.total_episodes - self.last_rollback_episode}/2500 episodes)")
            return None

        previous = self._get_curriculum_phase()
        self.current_phase_index -= 1
        completed_phases = self.curriculum[: self.current_phase_index + 1]
        recent_phases = completed_phases[-3:] if len(completed_phases) > 3 else completed_phases
        self.rehearsal_ids = [sid for phase in recent_phases for sid in phase["scenario_ids"]]
        self.levels = self._build_mixed_levels()
        # Seed phase history with neutral data instead of wiping
        target_wr = float(self._get_curriculum_phase().get("required_win_rate", 0.40))
        seed_count = 100
        self.phase_episode_wins = [1 if i < int(seed_count * target_wr) else 0 for i in range(seed_count)]
        self.phase_episode_rewards = [0.0] * seed_count
        self.phase_start_episode = self.total_episodes
        self.last_rollback_episode = self.total_episodes
        self._ent_coef_boost = self.ent_coef_start * 0.5
        self.phase_eval_pass_streak = 0
        self.phase_eval_fail_streak = 0
        self.last_phase_eval_win_rate = 0.0
        self._stale_eval_count = 0
        self._last_phase_eval_value = None
        return previous, self._get_curriculum_phase(), win_rate

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

        for chunk in worker_rollouts:
            rewards_arr = np.array(chunk["rewards"], dtype=np.float32)
            starts_arr = np.array(chunk["episode_starts"], dtype=np.float32)
            values_arr = np.array(chunk["values"], dtype=np.float32)

            # Reward normalization: track discounted returns, normalize by running std
            n = rewards_arr.shape[0]
            rets = np.zeros(n, dtype=np.float32)
            ret = 0.0
            for i in range(n):
                if starts_arr[i] > 0.5:
                    ret = 0.0
                ret = ret * self.gamma + float(rewards_arr[i])
                rets[i] = ret
            self.ret_rms.update(rets)
            rewards_arr = np.clip(
                rewards_arr / np.sqrt(max(self.ret_rms.var, 1e-8)),
                -10.0, 10.0,
            ).astype(np.float32)

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
        if not self._explicit_levels:
            self.phase_episode_wins.extend(ep_wins)
            self.phase_episode_rewards.extend([float(r) for r in ep_rewards])

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

    def train(self, target_episodes: int = 10_000, max_timesteps: int = 20_000_000) -> None:
        """Main training loop."""
        csv_file: TextIO = open(self.training_log_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
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
        next_checkpoint_episode = self.checkpoint_episode_interval
        next_eval_episode = self.eval_interval_episodes
        interrupt_reason = ""
        try:
            while self.total_episodes < target_episodes and self.total_timesteps < max_timesteps:
                iteration += 1

                curr_phase_name = self._get_curriculum_phase()["name"] if not self._explicit_levels else "Explicit levels"

                # Update adaptive rehearsal mix continuously within a phase.
                if self.stability_mode and not self._explicit_levels and self.phase_eval_fail_streak > 0:
                    # During instability, reduce distribution drift.
                    self.levels = self._get_curriculum_levels()
                    self.curriculum_difficulty_band = max(0.0, self.curriculum_difficulty_band - 0.02)
                else:
                    if not self._explicit_levels and self.rehearsal_ids:
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
                # Apply entropy boost from rollback recovery (decays each iteration)
                current_ent_coef = min(self.ent_coef_start, current_ent_coef + self._ent_coef_boost)
                self._ent_coef_boost = max(0.0, self._ent_coef_boost - 0.0005)
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
                    if not np.isfinite(float(vl)):
                        raise RuntimeError("PPO value_loss became non-finite; aborting to prevent corrupted checkpoints.")
                    if self.target_kl > 0 and float(approx_kl) > self.target_kl * 4.0:
                        raise RuntimeError(
                            f"PPO approx_kl spike ({float(approx_kl):.4f}) exceeded 4x target_kl ({self.target_kl:.4f})."
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

                if not self.stability_mode:
                    phase_transition = self._maybe_advance_curriculum()
                    if phase_transition is not None:
                        previous_phase, next_phase, trigger_win_rate = phase_transition
                        recent_breakdowns = self.episode_reward_breakdowns[-50:]
                        summary_parts: List[str] = []
                        for key in ["tick", "hit", "hurt", "kill", "death", "terminalWin", "terminalLoss", "timeout", "approach", "dodge"]:
                            vals = [float(b.get(key, 0.0)) for b in recent_breakdowns]
                            summary_parts.append(f"{key}={np.mean(vals):.3f}")
                        print(
                            f"\n*** Curriculum phase transition at episode {self.total_episodes}: "
                            f"{previous_phase['name']} -> {next_phase['name']} | trigger win rate={trigger_win_rate:.3f} ***"
                        )
                        print("  RewardBreakdown(trigger window): " + " ".join(summary_parts) + "\n")

                # Safety-net: episode-based rollback as fallback in ALL modes
                # (catches cases where eval dead-zones prevent eval-based rollback)
                if not self._explicit_levels and self.current_phase_index > 0:
                    phase_rollback = self._maybe_rollback_curriculum()
                    if phase_rollback is not None:
                        previous_phase, rollback_phase, collapse_win_rate = phase_rollback
                        restored = self._restore_stable_checkpoint()
                        print(
                            f"\n*** Safety-net rollback at episode {self.total_episodes}: "
                            f"{previous_phase['name']} -> {rollback_phase['name']} | "
                            f"phase_recent_winrate_500={collapse_win_rate:.3f} | restored={restored} ***\n"
                        )

                # 7. Track best training-window model separately from eval best.
                if len(self.episode_rewards) >= 20 and avg_winrate > self.best_train_win_rate:
                    self.best_train_win_rate = avg_winrate
                    cast(Any, self.model).save(self.best_train_model_path)
                    self._save_ret_rms()
                    print(f"  ** New best training-window model: {avg_winrate:.3f} **")

                # 7b. Periodic deterministic eval for robust best checkpoint selection.
                while self.total_episodes >= next_eval_episode:
                    try:
                        eval_win_rate = self._evaluate_policy()
                        print(
                            f"  Eval | Episodes={self.total_episodes} | "
                            f"WinRate={eval_win_rate:.3f} on levels={self.eval_levels}"
                        )
                        if eval_win_rate > self.best_eval_win_rate:
                            self.best_eval_win_rate = eval_win_rate
                            self.best_win_rate = eval_win_rate
                            cast(Any, self.model).save(self.best_model_path)
                            self._save_ret_rms()
                            self._export_policy_onnx(self.best_model_path)
                            print(f"  ** New best eval model! Win rate: {eval_win_rate:.3f} **")
                    except Exception as eval_exc:
                        print(f"  WARNING: _evaluate_policy failed ({eval_exc}) - continuing with phase gate checks.")

                    if self.stability_mode and not self._explicit_levels:
                        phase_levels = self._get_curriculum_levels()
                        phase_eval_source = "deterministic"
                        try:
                            phase_eval = self._evaluate_levels(
                                phase_levels,
                                self.eval_episodes,
                                self.eval_seed_start + 100000 + self.current_phase_index * 1000,
                            )
                        except Exception as phase_eval_exc:
                            phase_eval_source = "training-window-fallback"
                            phase_eval = self._phase_recent_win_rate()
                            print(
                                f"  WARNING: phase deterministic eval failed ({phase_eval_exc}) - "
                                f"using phase_recent_winrate_500 fallback={phase_eval:.3f}"
                            )

                        self.last_phase_eval_win_rate = phase_eval
                        current = self._get_curriculum_phase()
                        required_wr = float(current.get("required_win_rate", 0.40))
                        promote_threshold = required_wr if phase_eval_source == "deterministic" else max(required_wr, 0.65)
                        collapse_threshold = max(0.15, required_wr * 0.75)

                        # Detect stale eval: same value repeated means deterministic
                        # eval is stuck on fixed seeds, not reflecting real performance.
                        if self._last_phase_eval_value is not None and abs(phase_eval - self._last_phase_eval_value) < 1e-6:
                            self._stale_eval_count += 1
                        else:
                            self._stale_eval_count = 0
                        self._last_phase_eval_value = phase_eval

                        # Cross-check: if training WR is near zero but eval says
                        # otherwise, trust training WR — eval is stale/degenerate.
                        training_wr = self._phase_recent_win_rate()
                        stale_and_collapsed = (
                            self._stale_eval_count >= 4
                            and training_wr < 0.05
                            and self._phase_episode_count() >= 3000
                        )

                        if stale_and_collapsed:
                            # Override: treat stale eval as failure
                            self.phase_eval_fail_streak += 1
                            self.phase_eval_pass_streak = 0
                            print(
                                f"  ** Stale eval override: eval={phase_eval:.3f} unchanged {self._stale_eval_count}x "
                                f"but training_wr={training_wr:.3f} — counting as fail (streak={self.phase_eval_fail_streak}) **"
                            )
                        elif phase_eval >= promote_threshold:
                            self.phase_eval_pass_streak += 1
                            self.phase_eval_fail_streak = 0
                        elif phase_eval <= collapse_threshold:
                            self.phase_eval_fail_streak += 1
                            self.phase_eval_pass_streak = 0
                        else:
                            self.phase_eval_pass_streak = 0
                            self.phase_eval_fail_streak = 0

                        print(
                            f"  PhaseEval[{phase_eval_source}] | Phase={current['name']} | WR={phase_eval:.3f} | "
                            f"required={required_wr:.3f} | pass_streak={self.phase_eval_pass_streak} | "
                            f"fail_streak={self.phase_eval_fail_streak}"
                        )

                        phase_episodes = self._phase_episode_count()
                        min_phase_episodes = int(current.get("min_phase_episodes") or 0)
                        force_phase_episodes = int(current.get("force_phase_episodes") or 0)
                        force_eligible = (
                            force_phase_episodes > 0
                            and phase_episodes >= force_phase_episodes
                            and phase_eval >= required_wr
                        )
                        can_promote = (
                            self.current_phase_index < len(self.curriculum) - 1
                            and phase_episodes >= min_phase_episodes
                            and (self.phase_eval_pass_streak >= self.phase_pass_evals_required or force_eligible)
                            and self._phase_is_stable()
                        )

                        if can_promote:
                            trigger = "force-advance" if force_eligible else "eval-gated"
                            self._save_stable_checkpoint(self.current_phase_index, phase_eval)
                            transition = self._advance_curriculum_from_eval()
                            if transition is not None:
                                previous_phase, next_phase = transition
                                print(
                                    f"\n*** {trigger} phase transition at episode {self.total_episodes}: "
                                    f"{previous_phase['name']} -> {next_phase['name']} | "
                                    f"phase_eval={phase_eval:.3f} ({phase_eval_source}) ***\n"
                                )

                        rollback_floor = self._rollback_floor()
                        can_rollback = (
                            self.current_phase_index > 0
                            and self.current_phase_index - 1 >= rollback_floor
                            and phase_episodes >= 700
                            and self.phase_eval_fail_streak >= self.phase_fail_evals_before_rollback
                            and self.total_episodes - self.last_rollback_episode >= 1500
                        )
                        if can_rollback:
                            rollback = self._rollback_curriculum_from_eval()
                            restored = self._restore_stable_checkpoint()
                            if rollback is not None:
                                previous_phase, rollback_phase = rollback
                                print(
                                    f"\n*** Eval-gated rollback at episode {self.total_episodes}: "
                                    f"{previous_phase['name']} -> {rollback_phase['name']} | "
                                    f"phase_eval={phase_eval:.3f} ({phase_eval_source}) | restored={restored} ***\n"
                                )
                    next_eval_episode += self.eval_interval_episodes
                    self._write_run_manifest(status="running")

                # 8. Episode-based periodic checkpoints
                while self.total_episodes >= next_checkpoint_episode:
                    ckpt_path = os.path.join(
                        self.output_dir, f"treads_ppo_ep{next_checkpoint_episode}"
                    )
                    cast(Any, self.model).save(ckpt_path)
                    self._save_ret_rms()
                    next_checkpoint_episode += self.checkpoint_episode_interval

        except KeyboardInterrupt:
            interrupt_reason = "Training interrupted by user."
            print("\nTraining interrupted by user.")
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
            cast(Any, self.model).save(self.final_model_path)
            self._save_ret_rms()
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
            self._write_run_manifest(status="completed" if not interrupt_reason else "interrupted", error=interrupt_reason)
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
    parser.add_argument("--target-episodes", type=int, default=10_000, help="Stop when this many episodes are collected")
    parser.add_argument("--load-model", type=str, default="", help="Optional .zip model path to resume from")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output directory override")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Checkpoint interval in episodes")
    parser.add_argument("--replay-interval", type=int, default=250, help="Replay save interval in episodes")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--ent-coef", type=float, default=0.015, help="Entropy coefficient (default: 0.015)")
    parser.add_argument("--ent-coef-final", type=float, default=0.003, help="Final entropy coefficient at end of training (default: 0.003)")
    parser.add_argument("--max-ticks", type=int, default=720, help="Max ticks per episode (default: 720)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--n-steps", type=int, default=8192, help="Rollout steps per iteration (default: 8192)")
    parser.add_argument("--clip-range", type=float, default=0.15, help="PPO clip range (default: 0.15)")
    parser.add_argument("--clip-range-final", type=float, default=0.08, help="Final PPO clip range at end of training (default: 0.08)")
    parser.add_argument("--target-kl", type=float, default=0.02, help="PPO target KL early-stop threshold (default: 0.02)")
    parser.add_argument("--stability-mode", action="store_true", default=True, help="Enable eval-gated curriculum stabilization mode (default: enabled)")
    parser.add_argument("--no-stability-mode", action="store_false", dest="stability_mode", help="Disable eval-gated curriculum stabilization mode")
    parser.add_argument("--phase-pass-evals", type=int, default=3, help="Consecutive phase eval passes required before promotion (default: 3)")
    parser.add_argument("--phase-fail-evals", type=int, default=2, help="Consecutive phase eval failures before rollback (default: 2)")
    parser.add_argument("--eval-interval", type=int, default=500, help="Run deterministic eval every N collected episodes (default: 500)")
    parser.add_argument("--eval-episodes", type=int, default=4, help="Deterministic eval episodes per level (default: 4)")
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
        stability_mode=args.stability_mode,
        phase_pass_evals_required=args.phase_pass_evals,
        phase_fail_evals_before_rollback=args.phase_fail_evals,
    )

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
