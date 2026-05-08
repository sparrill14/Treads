"""
PPO Training script for Treads using Stable Baselines3.
Trains against a scripted sparring partner on Level 1 (simple: 1 stationary enemy).
"""

import os
import sys
import csv
import time
from typing import Any, Callable, Dict, List, Optional, TextIO, cast
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from treads_env import TreadsEnvDiscrete


class TrainingLogger(BaseCallback):
    """Logs episode stats to CSV and console. Saves best model by win rate."""

    def __init__(
        self,
        log_path: str,
        best_model_path: str,
        replay_interval: int = 50_000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.log_path = log_path
        self.best_model_path = best_model_path
        self.replay_interval = replay_interval
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_wins: List[int] = []
        self.csv_file: Optional[TextIO] = None
        self.csv_writer = None
        self.total_episodes = 0
        self.start_time = time.time()
        self._last_replay_timestep = 0
        self.best_win_rate = 0.0

    def _on_training_start(self) -> None:
        self.csv_file = open(self.log_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["episode", "timestep", "reward", "length", "win", "avg_reward_50", "avg_winrate_50", "elapsed_sec"]
        )

    def _on_step(self) -> bool:
        locals_dict = self.locals

        # Check if episode ended
        infos = cast(List[Dict[str, Any]], locals_dict.get("infos", []))
        for info in infos:
            if "result" in info:
                result = info["result"]
                _ = locals_dict.get("rewards", [0])
                _ = locals_dict.get("n_steps", 0)

                # SB3 provides episode info in monitor wrapper
                # but we track our own from the result messages
                win = bool(cast(Dict[str, Any], result).get("win", False))
                self.episode_wins.append(1 if win else 0)
                self.total_episodes += 1

        # Also check the "episode" key that SB3 Monitor adds
        dones = cast(List[bool], locals_dict.get("dones", []))
        if any(dones):
            for i, done in enumerate(dones):
                infos_for_ep = cast(List[Dict[str, Any]], locals_dict.get("infos", []))
                if done and i < len(infos_for_ep) and "episode" in infos_for_ep[i]:
                    ep_info = cast(Dict[str, Any], infos_for_ep[i]["episode"])
                    self.episode_rewards.append(float(ep_info["r"]))
                    self.episode_lengths.append(int(ep_info["l"]))

        # Periodically enable replay saving for the next episode
        training_env_any = cast(Any, self.training_env)
        env = training_env_any.envs[0]
        inner = env.env if hasattr(env, "env") else env
        if self.num_timesteps - self._last_replay_timestep >= self.replay_interval:
            if hasattr(inner, "_env"):
                inner._env._save_replay = True
            elif hasattr(inner, "_save_replay"):
                inner._save_replay = True
            self._last_replay_timestep = self.num_timesteps
        else:
            if hasattr(inner, "_env"):
                inner._env._save_replay = False
            elif hasattr(inner, "_save_replay"):
                inner._save_replay = False

        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-50:]
            recent_wins = self.episode_wins[-50:] if self.episode_wins else [0]
            avg_reward = np.mean(recent_rewards)
            avg_winrate = np.mean(recent_wins) if recent_wins else 0

            elapsed = time.time() - self.start_time
            print(
                f"Episodes: {len(self.episode_rewards)} | "
                f"Avg Reward (50): {avg_reward:.3f} | "
                f"Avg WinRate (50): {avg_winrate:.3f} | "
                f"Time: {elapsed:.0f}s"
            )

            # Save best model by win rate (need at least 20 episodes for stable estimate)
            if len(self.episode_rewards) >= 20 and avg_winrate > self.best_win_rate:
                self.best_win_rate = avg_winrate
                cast(Any, self.model).save(self.best_model_path)
                print(f"  ** New best model saved! Win rate: {avg_winrate:.3f} **")

            if self.csv_writer:
                self.csv_writer.writerow([
                    len(self.episode_rewards),
                    self.num_timesteps,
                    self.episode_rewards[-1] if self.episode_rewards else 0,
                    self.episode_lengths[-1] if self.episode_lengths else 0,
                    self.episode_wins[-1] if self.episode_wins else 0,
                    avg_reward,
                    avg_winrate,
                    elapsed,
                ])
                if self.csv_file is not None:
                    self.csv_file.flush()

    def _on_training_end(self) -> None:
        if self.csv_file:
            self.csv_file.close()


def linear_schedule_between(initial_value: float, final_value: float) -> Callable[[float], float]:
    """Linear schedule where progress=1 -> initial_value and progress=0 -> final_value."""
    def func(progress_remaining: float) -> float:
        p = min(1.0, max(0.0, float(progress_remaining)))
        return final_value + (initial_value - final_value) * p
    return func


def make_env() -> Any:
    """Create the training environment with SB3 Monitor wrapper."""
    from stable_baselines3.common.monitor import Monitor
    env = TreadsEnvDiscrete(level=1, seed_start=0, max_episode_steps=720)
    return cast(Any, Monitor(env))


def train() -> None:
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    device = "cpu"
    print(f"Training on: {device}")

    print("Creating environment...")
    env = make_env()

    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.997,
        gae_lambda=0.95,
        clip_range=linear_schedule_between(0.15, 0.08),
        ent_coef=0.02,
        target_kl=0.03,
        device=device,
        policy_kwargs=dict(
            net_arch=[256, 256],
        ),
        seed=42,
    )

    log_path = os.path.join(output_dir, "training_log.csv")
    best_model_path = os.path.join(output_dir, "treads_ppo_best")
    callback = TrainingLogger(log_path, best_model_path)

    total_timesteps = 5_000_000
    print(f"Training for {total_timesteps} timesteps...")

    cast(Any, model).learn(total_timesteps=total_timesteps, callback=callback)

    # Save final model as well
    final_model_path = os.path.join(output_dir, "treads_ppo_final")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Copy the best model to the canonical path used by export/validate
    model_path = os.path.join(output_dir, "treads_ppo")
    best_zip = best_model_path + ".zip"
    if os.path.exists(best_zip):
        import shutil
        shutil.copy2(best_zip, model_path + ".zip")
        print(f"Best model (win rate {callback.best_win_rate:.3f}) copied to {model_path}")
    else:
        model.save(model_path)
        print(f"No best model found, using final model at {model_path}")

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    train()
