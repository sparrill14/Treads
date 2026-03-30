"""Benchmark persistent vs legacy vs hybrid modes."""
import time
import os
import sys
import json
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from treads_env import TreadsEnv, TreadsEnvDiscrete

NUM_EPISODES = 20
ROLLOUT_WORKER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "rollout-worker.js"
)


def benchmark_env_mode(persistent: bool) -> dict:
    env = TreadsEnvDiscrete(level=1, seed_start=0, max_episode_steps=1800)
    env._env._persistent = persistent
    mode_name = "persistent" if persistent else "legacy"

    start = time.perf_counter()
    total_steps = 0
    wins = 0

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
        if info.get("result", {}).get("win", False):
            wins += 1

    elapsed = time.perf_counter() - start
    env.close()

    return {
        "mode": mode_name,
        "total_steps": total_steps,
        "elapsed_s": round(elapsed, 2),
        "steps_per_sec": round(total_steps / elapsed, 1),
    }


def benchmark_hybrid(n_steps=4096) -> dict:
    """Benchmark the hybrid rollout worker (rollout collection only)."""
    from train_hybrid import HybridTrainer

    trainer = HybridTrainer(level=1, n_steps=n_steps, max_episode_steps=1800)

    start = time.perf_counter()
    trainer._send_weights()
    rollout = trainer._collect_rollout()
    elapsed = time.perf_counter() - start

    total_steps = rollout["n_steps"]
    trainer.close()

    return {
        "mode": "hybrid",
        "total_steps": total_steps,
        "elapsed_s": round(elapsed, 2),
        "steps_per_sec": round(total_steps / elapsed, 1),
    }


if __name__ == "__main__":
    print(f"Benchmarking {NUM_EPISODES} episodes (random actions)...\n")

    legacy = benchmark_env_mode(persistent=False)
    print(f"Legacy:     {legacy['elapsed_s']}s | {legacy['steps_per_sec']} steps/s")

    persistent = benchmark_env_mode(persistent=True)
    print(f"Persistent: {persistent['elapsed_s']}s | {persistent['steps_per_sec']} steps/s")

    hybrid = benchmark_hybrid(n_steps=4096)
    print(f"Hybrid:     {hybrid['elapsed_s']}s | {hybrid['steps_per_sec']} steps/s")

    print(f"\nSpeedup vs legacy:")
    print(f"  Persistent: {legacy['elapsed_s'] / persistent['elapsed_s']:.2f}x")
    print(f"  Hybrid:     {legacy['steps_per_sec'] / 1.0:.0f} → {hybrid['steps_per_sec']:.0f} steps/s ({hybrid['steps_per_sec'] / legacy['steps_per_sec']:.1f}x)")

