"""
Test the TreadsEnv Gymnasium environment by running 5 episodes with random actions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from treads_env import TreadsEnvFlat
import numpy as np


def test_env():
    env = TreadsEnvFlat(level=1, seed_start=100, max_episode_steps=600)
    num_episodes = 5

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        result = info.get("result", {})
        status = result.get("status", "unknown")
        print(
            f"Episode {ep + 1}: steps={steps}, reward={total_reward:.3f}, "
            f"status={status}, win={result.get('win', False)}"
        )

    env.close()
    print("\nAll 5 episodes completed successfully!")


if __name__ == "__main__":
    test_env()
