"""
PPO Training script for Treads using Stable Baselines3.
Trains against a scripted sparring partner on Level 1 (simple: 1 stationary enemy).
"""

import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from treads_env import TreadsEnvDiscrete


class TrainingLogger(BaseCallback):
    """Logs episode stats to CSV and console."""

    def __init__(self, log_path, verbose=1):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        self.csv_file = None
        self.csv_writer = None
        self.total_episodes = 0
        self.start_time = time.time()

    def _on_training_start(self):
        self.csv_file = open(self.log_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["episode", "timestep", "reward", "length", "win", "avg_reward_50", "avg_winrate_50", "elapsed_sec"]
        )

    def _on_step(self):
        # Check if episode ended
        infos = self.locals.get("infos", [])
        for info in infos:
            if "result" in info:
                result = info["result"]
                ep_reward = sum(self.locals.get("rewards", [0]))
                ep_len = self.locals.get("n_steps", 0)

                # SB3 provides episode info in monitor wrapper
                # but we track our own from the result messages
                win = result.get("win", False)
                self.episode_wins.append(1 if win else 0)
                self.total_episodes += 1

        # Also check the "episode" key that SB3 Monitor adds
        dones = self.locals.get("dones", [])
        if any(dones):
            for i, done in enumerate(dones):
                if done and "episode" in (self.locals.get("infos", [{}])[i] if i < len(self.locals.get("infos", [])) else {}):
                    ep_info = self.locals["infos"][i]["episode"]
                    self.episode_rewards.append(ep_info["r"])
                    self.episode_lengths.append(ep_info["l"])

        return True

    def _on_rollout_end(self):
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
                self.csv_file.flush()

    def _on_training_end(self):
        if self.csv_file:
            self.csv_file.close()


def make_env():
    """Create the training environment with SB3 Monitor wrapper."""
    from stable_baselines3.common.monitor import Monitor
    env = TreadsEnvDiscrete(level=1, seed_start=0, max_episode_steps=600)
    return Monitor(env)


def train():
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Creating environment...")
    env = make_env()

    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        device="cpu",
        policy_kwargs=dict(
            net_arch=[256, 256],
        ),
        seed=42,
    )

    log_path = os.path.join(output_dir, "training_log.csv")
    callback = TrainingLogger(log_path)

    total_timesteps = 200_000
    print(f"Training for {total_timesteps} timesteps...")

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model_path = os.path.join(output_dir, "treads_ppo")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    train()
