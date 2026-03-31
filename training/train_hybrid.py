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
import subprocess
from typing import Any, Dict, List, Optional, TextIO, Tuple, cast
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from treads_env import OBS_SIZE

# Path to compiled rollout worker
ROLLOUT_WORKER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "rollout-worker.js"
)

# ---- Curriculum configuration ----
DEFAULT_CURRICULUM: List[Dict[str, Any]] = [
    {"name": "Phase 1 (target practice)", "scenario_ids": [111, 112, 113], "advance_after_episodes": 1500, "required_win_rate": 0.40},
    {"name": "Phase 2 (live fire fundamentals)", "scenario_ids": [121, 122, 123], "advance_after_episodes": 4000, "required_win_rate": 0.40},
    {"name": "Phase 3 (bounce and bombs)", "scenario_ids": [301, 302, 303], "advance_after_episodes": 9000, "required_win_rate": 0.40},
    {"name": "Phase 4 (full game)", "scenario_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9], "advance_after_episodes": None, "required_win_rate": 0.40},
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
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        learning_rate: float = 1e-4,
        clip_range: float = 0.15,
        ent_coef: float = 0.08,
        max_episode_steps: int = 720,
        output_dir: Optional[str] = None,
        load_model_path: Optional[str] = None,
        checkpoint_episode_interval: int = 500,
        replay_episode_interval: int = 250,
        seed: int = 42,
    ) -> None:
        self._explicit_levels = levels is not None
        self.curriculum: List[Dict[str, Any]] = curriculum or DEFAULT_CURRICULUM
        self.current_phase_index = 0
        self.phase_episode_wins: List[int] = []
        self.phase_start_episode = 0
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
        self.checkpoint_episode_interval = checkpoint_episode_interval
        self.replay_episode_interval = replay_episode_interval

        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.replay_dir = os.path.join(self.output_dir, "replays")
        os.makedirs(self.replay_dir, exist_ok=True)

        class _DummyContinuousActionEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
            metadata = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
                # [move_signal, aim_signal, fire_signal, bomb_signal] in [-1, 1]
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

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
        if load_model_path and os.path.exists(load_model_path):
            print(f"Loading PPO model from: {load_model_path}")
            # Override key hyperparameters so new training settings take effect
            self.model = cast(Any, PPO.load(  # pyright: ignore[reportUnknownMemberType]
                load_model_path, env=dummy_env, device="cpu",
                custom_objects={
                    "gamma": gamma,
                    "ent_coef": ent_coef,
                    "learning_rate": learning_rate,
                    "clip_range": clip_range,
                }
            ))
            print(f"  Overriding gamma={gamma}, ent_coef={ent_coef}, lr={learning_rate}")
        else:
            self.model = PPO(
                "MlpPolicy",
                dummy_env,
                verbose=0,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                device="cpu",
                policy_kwargs=dict(net_arch=[256, 256]),
                seed=seed,
            )
        dummy_env.close()

        # Set up SB3 logger (required for model.train())
        self.model.set_logger(configure(self.output_dir, ["stdout"]))

        # Start rollout worker
        self.worker: Optional[subprocess.Popen[str]] = None
        self._start_worker()

        # Logging state
        self.total_timesteps = 0
        self.total_episodes = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_wins: List[int] = []
        self.episode_levels: List[int] = []
        self.episode_reward_breakdowns: List[Dict[str, Any]] = []
        self.best_win_rate = 0.0
        self.start_time = time.time()

    def _start_worker(self) -> None:
        """Start the Node.js rollout worker process."""
        self.worker = subprocess.Popen(
            ["node", ROLLOUT_WORKER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Wait for ready
        ready = self._read_msg()
        assert ready["type"] == "ready", f"Expected ready, got {ready}"

    def _get_curriculum_phase(self) -> Dict[str, Any]:
        return self.curriculum[self.current_phase_index]

    def _get_curriculum_levels(self) -> List[int]:
        return list(self._get_curriculum_phase()["scenario_ids"])

    def _phase_recent_win_rate(self) -> float:
        if not self.phase_episode_wins:
            return 0.0
        window = self.phase_episode_wins[-500:]
        return float(np.mean(window))

    def _maybe_advance_curriculum(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        if self._explicit_levels or self.current_phase_index >= len(self.curriculum) - 1:
            return None
        current = self._get_curriculum_phase()
        advance_after = current.get("advance_after_episodes")
        if advance_after is None or self.total_episodes < int(advance_after):
            return None
        win_rate = self._phase_recent_win_rate()
        if win_rate < float(current.get("required_win_rate", 0.40)):
            return None
        previous = current
        self.current_phase_index += 1
        self.levels = self._get_curriculum_levels()
        self.phase_episode_wins = []
        self.phase_start_episode = self.total_episodes
        return previous, self._get_curriculum_phase(), win_rate

    def _send_msg(self, msg: Dict[str, Any]) -> None:
        """Send JSON message to worker."""
        assert self.worker is not None and self.worker.stdin is not None
        self.worker.stdin.write(json.dumps(msg) + "\n")
        self.worker.stdin.flush()

    def _read_msg(self) -> Dict[str, Any]:
        """Read JSON message from worker."""
        assert self.worker is not None and self.worker.stdout is not None
        line = self.worker.stdout.readline()
        if not line:
            stderr = ""
            if self.worker.stderr is not None:
                stderr = self.worker.stderr.read()
            raise RuntimeError(f"Worker process died. stderr: {stderr}")
        return cast(Dict[str, Any], json.loads(line.strip()))

    def _extract_weights(self) -> Dict[str, Any]:
        """Extract model weights as nested lists for JSON transfer."""
        state_dict: Dict[str, Any] = {}
        for key, tensor in cast(Any, self.model).policy.state_dict().items():
            state_dict[key] = tensor.cpu().numpy().tolist()
        return state_dict

    def _send_weights(self) -> None:
        """Send current model weights to the rollout worker."""
        weights = self._extract_weights()
        self._send_msg({"type": "set_weights", "state_dict": weights})
        ack = self._read_msg()
        assert ack["type"] == "weights_set", f"Expected weights_set, got {ack}"

    def _collect_rollout(self, target_episodes: int) -> Dict[str, Any]:
        """Request rollout collection from the worker."""
        self.seed_counter += self.n_steps  # advance seed to avoid repeats
        self._send_msg({
            "type": "collect",
            "n_steps": self.n_steps,
            "levels": self.levels,
            "maxTicks": self.max_episode_steps,
            "seedStart": self.seed_counter,
            "replayEveryEpisodes": self.replay_episode_interval,
            "replayDir": self.replay_dir,
            "episodeOffset": self.total_episodes,
            "targetEpisodes": target_episodes,
        })
        rollout = self._read_msg()
        assert rollout["type"] == "rollout", f"Expected rollout, got {rollout.get('type')}"
        return rollout

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

        obs_arr = np.array(rollout["obs"], dtype=np.float32)
        actions_arr = np.array(rollout["actions"], dtype=np.float32)
        rewards_arr = np.array(rollout["rewards"], dtype=np.float32)
        episode_starts_arr = np.array(rollout["episode_starts"], dtype=np.float32)
        values_arr = np.array(rollout["values"], dtype=np.float32)
        log_probs_arr = np.array(rollout["log_probs"], dtype=np.float32)

        n_steps = len(obs_arr)
        for i in range(n_steps):
            buf.add(
                obs=obs_arr[i:i+1],  # (1, obs_dim) for n_envs=1
                action=actions_arr[i:i+1],
                reward=np.array([rewards_arr[i]]),
                episode_start=np.array([episode_starts_arr[i]]),
                value=torch.tensor([values_arr[i]]),
                log_prob=torch.tensor([log_probs_arr[i]]),
            )

        # Compute returns and advantages (GAE)
        last_values = torch.tensor([rollout["last_value"]])
        last_dones = np.array([1.0 if rollout["last_done"] else 0.0])
        buf.compute_returns_and_advantage(last_values=last_values, dones=last_dones)

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
                keys = ["tick", "hit", "hurt", "kill", "death", "terminalWin", "terminalLoss", "timeout", "wastedShot", "aimJitter", "approach", "wastedBomb"]
                summary: List[str] = []
                for key in keys:
                    vals = [float(b.get(key, 0.0)) for b in recent_breakdowns]
                    summary.append(f"{key}={np.mean(vals):.3f}")
                print("  RewardBreakdown(50): " + " ".join(summary))

            return float(avg_reward), float(avg_winrate)
        return 0.0, 0.0

    def train(self, target_episodes: int = 10_000, max_timesteps: int = 20_000_000) -> None:
        """Main training loop."""
        log_path = os.path.join(self.output_dir, "training_log.csv")
        best_model_path = os.path.join(self.output_dir, "treads_ppo_best")

        csv_file: TextIO = open(log_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "iteration", "timesteps", "episodes", "curriculum_phase", "active_scenarios", "phase_recent_winrate_500",
            "avg_reward_50", "avg_winrate_50",
            "avg_tick_50", "avg_hit_50", "avg_hurt_50", "avg_kill_50", "avg_death_50", "avg_terminal_win_50", "avg_terminal_loss_50", "avg_timeout_50", "avg_wasted_shot_50", "avg_aim_jitter_50", "avg_approach_50", "avg_wasted_bomb_50",
            "steps_per_sec", "elapsed_sec"
        ])

        iteration = 0
        next_checkpoint_episode = self.checkpoint_episode_interval
        try:
            while self.total_episodes < target_episodes and self.total_timesteps < max_timesteps:
                iteration += 1

                curr_phase_name = self._get_curriculum_phase()["name"] if not self._explicit_levels else "Explicit levels"

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
                cast(Any, self.model)._current_progress_remaining = 1.0 - self.total_episodes / max(target_episodes, 1)
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
                    avg_tick = np.mean([float(b.get("tick", 0.0)) for b in recent_breakdowns])
                    avg_hit = np.mean([float(b.get("hit", 0.0)) for b in recent_breakdowns])
                    avg_hurt = np.mean([float(b.get("hurt", 0.0)) for b in recent_breakdowns])
                    avg_kill = np.mean([float(b.get("kill", 0.0)) for b in recent_breakdowns])
                    avg_death = np.mean([float(b.get("death", 0.0)) for b in recent_breakdowns])
                    avg_terminal_win = np.mean([float(b.get("terminalWin", 0.0)) for b in recent_breakdowns])
                    avg_terminal_loss = np.mean([float(b.get("terminalLoss", 0.0)) for b in recent_breakdowns])
                    avg_timeout = np.mean([float(b.get("timeout", 0.0)) for b in recent_breakdowns])
                    avg_wasted_shot = np.mean([float(b.get("wastedShot", 0.0)) for b in recent_breakdowns])
                    avg_aim_jitter = np.mean([float(b.get("aimJitter", 0.0)) for b in recent_breakdowns])
                    avg_approach = np.mean([float(b.get("approach", 0.0)) for b in recent_breakdowns])
                    avg_wasted_bomb = np.mean([float(b.get("wastedBomb", 0.0)) for b in recent_breakdowns])
                else:
                    avg_tick = avg_hit = avg_hurt = avg_kill = avg_death = avg_terminal_win = avg_terminal_loss = avg_timeout = avg_wasted_shot = avg_aim_jitter = avg_approach = avg_wasted_bomb = 0.0

                csv_writer.writerow([
                    iteration,
                    self.total_timesteps,
                    self.total_episodes,
                    curr_phase_name,
                    ",".join(str(l) for l in self.levels),
                    f"{self._phase_recent_win_rate():.3f}" if not self._explicit_levels else "",
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
                    f"{avg_wasted_shot:.3f}",
                    f"{avg_aim_jitter:.3f}",
                    f"{avg_approach:.3f}",
                    f"{avg_wasted_bomb:.3f}",
                    f"{steps_per_sec:.0f}",
                    f"{elapsed:.1f}",
                ])
                csv_file.flush()

                phase_transition = self._maybe_advance_curriculum()
                if phase_transition is not None:
                    previous_phase, next_phase, trigger_win_rate = phase_transition
                    recent_breakdowns = self.episode_reward_breakdowns[-50:]
                    summary_parts: List[str] = []
                    for key in ["tick", "hit", "hurt", "kill", "death", "terminalWin", "terminalLoss", "timeout", "wastedShot", "aimJitter", "approach", "wastedBomb"]:
                        vals = [float(b.get(key, 0.0)) for b in recent_breakdowns]
                        summary_parts.append(f"{key}={np.mean(vals):.3f}")
                    print(
                        f"\n*** Curriculum phase transition at episode {self.total_episodes}: "
                        f"{previous_phase['name']} -> {next_phase['name']} | trigger win rate={trigger_win_rate:.3f} ***"
                    )
                    print("  RewardBreakdown(trigger window): " + " ".join(summary_parts) + "\n")

                # 7. Save best model
                if len(self.episode_rewards) >= 20 and avg_winrate > self.best_win_rate:
                    self.best_win_rate = avg_winrate
                    cast(Any, self.model).save(best_model_path)
                    print(f"  ** New best model! Win rate: {avg_winrate:.3f} **")

                # 8. Episode-based periodic checkpoints
                while self.total_episodes >= next_checkpoint_episode:
                    ckpt_path = os.path.join(
                        self.output_dir, f"treads_ppo_ep{next_checkpoint_episode}"
                    )
                    cast(Any, self.model).save(ckpt_path)
                    next_checkpoint_episode += self.checkpoint_episode_interval

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
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
            final_path = os.path.join(self.output_dir, "treads_ppo_final")
            cast(Any, self.model).save(final_path)
            print(f"Final model saved to {final_path}")
            self.close()

    def close(self) -> None:
        """Clean up worker process."""
        if self.worker:
            try:
                self._send_msg({"type": "exit"})
                self.worker.wait(timeout=5)
            except Exception:
                self.worker.kill()
            self.worker = None

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
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--target-episodes", type=int, default=10_000, help="Stop when this many episodes are collected")
    parser.add_argument("--load-model", type=str, default="", help="Optional .zip model path to resume from")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output directory override")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Checkpoint interval in episodes")
    parser.add_argument("--replay-interval", type=int, default=250, help="Replay save interval in episodes")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor (default: 0.999)")
    parser.add_argument("--ent-coef", type=float, default=0.03, help="Entropy coefficient (default: 0.03)")
    parser.add_argument("--max-ticks", type=int, default=720, help="Max ticks per episode (default: 720)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--n-steps", type=int, default=4096, help="Rollout steps per iteration (default: 4096)")
    parser.add_argument("--clip-range", type=float, default=0.10, help="PPO clip range (default: 0.10)")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()] if args.levels else None

    trainer = HybridTrainer(
        levels=levels,
        n_steps=args.n_steps,
        batch_size=512,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        learning_rate=args.lr,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        max_episode_steps=args.max_ticks,
        output_dir=args.output_dir or None,
        load_model_path=args.load_model or None,
        checkpoint_episode_interval=args.checkpoint_interval,
        replay_episode_interval=args.replay_interval,
    )

    print(f"Starting hybrid training (rollout in TypeScript, PPO in Python)")
    if levels is None:
        print(f"Initial curriculum: {trainer.levels} ({trainer.curriculum[0]['name']})")
    else:
        print(f"Explicit scenarios: {trainer.levels} (curriculum disabled)")
    print(f"Gamma: {args.gamma} | EntCoef: {args.ent_coef} | MaxTicks: {args.max_ticks} | ClipRange: {args.clip_range}")
    print(f"Rollout worker: {ROLLOUT_WORKER_PATH}")
    trainer.train(target_episodes=args.target_episodes, max_timesteps=args.timesteps)


if __name__ == "__main__":
    train()
