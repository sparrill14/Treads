"""
Validate the trained model by running it in the Gym environment and reporting actions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from treads_env import TreadsEnvDiscrete, MOVE_INTENTS, NUM_AIM_BINS
from stable_baselines3 import PPO
import numpy as np


def validate():
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    model_path = os.path.join(output_dir, "treads_ppo.zip")

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device="cpu")

    env = TreadsEnvDiscrete(level=1, seed_start=5000, max_episode_steps=1800)

    num_episodes = 10
    wins = 0
    total_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        actions_taken = {"moves": {}, "fires": 0, "aims": {}, "total_steps": 0}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            move_name = MOVE_INTENTS[int(action[0])]
            actions_taken["moves"][move_name] = actions_taken["moves"].get(move_name, 0) + 1
            aim_bin = int(action[1])
            actions_taken["aims"][aim_bin] = actions_taken["aims"].get(aim_bin, 0) + 1
            if int(action[2]) == 1:
                actions_taken["fires"] += 1
            actions_taken["total_steps"] = steps

        result = info.get("result", {})
        is_win = result.get("win", False)
        wins += int(is_win)
        total_rewards.append(total_reward)

        print(f"Episode {ep+1}: steps={steps}, reward={total_reward:.3f}, "
              f"status={result.get('status', '?')}, win={is_win}")

        top_moves = sorted(actions_taken["moves"].items(), key=lambda x: -x[1])[:5]
        move_str = ", ".join(f"{m}={c}" for m, c in top_moves)
        print(f"  Moves: {move_str}")
        print(f"  Fires: {actions_taken['fires']}/{steps} ticks")
        aim_spread = len(actions_taken["aims"])
        print(f"  Aim spread: {aim_spread}/{NUM_AIM_BINS} bins used")

    print(f"\nSummary: {wins}/{num_episodes} wins, "
          f"avg reward: {np.mean(total_rewards):.3f}")

    env.close()


if __name__ == "__main__":
    validate()
