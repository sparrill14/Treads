"""Diagnostic test: Level 1 only with PPO loss logging."""
import os, sys
from typing import Any, List, cast
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))

from train_hybrid import HybridTrainer

# Level 1 only — should match old training performance
trainer = HybridTrainer(levels=[1], n_steps=4096, max_episode_steps=720)
print("Level 1 only diagnostic training (20 iterations)...")

for iteration in range(1, 21):
    trainer.send_weights()
    rollout = trainer.collect_rollout(trainer.total_episodes + 1)
    trainer.total_timesteps += trainer.n_steps
    trainer.populate_buffer(rollout)

    cast(Any, trainer.model)._current_progress_remaining = 1.0 - trainer.total_timesteps / 5_000_000
    cast(Any, trainer.model).num_timesteps = trainer.total_timesteps

    # Before training: check ratio
    buf = trainer.model.rollout_buffer
    sample = list(buf.get(min(512, trainer.n_steps)))[0]
    with torch.no_grad():
        values, log_prob, entropy = trainer.model.policy.evaluate_actions(
            sample.observations, sample.actions
        )
    ratio = torch.exp(log_prob - sample.old_log_prob)
    print(f"  Pre-train ratio: mean={ratio.mean().item():.4f} min={ratio.min().item():.4f} max={ratio.max().item():.4f}")
    print(f"  Old log_probs: mean={sample.old_log_prob.mean().item():.4f} std={sample.old_log_prob.std().item():.4f}")
    print(f"  New log_probs: mean={log_prob.mean().item():.4f} std={log_prob.std().item():.4f}")

    cast(Any, trainer.model).train()

    # Log episode stats
    rollout_dict = rollout
    ep_wins = cast(List[int], rollout_dict.get("episode_wins", []))
    ep_rewards = cast(List[float], rollout_dict.get("episode_rewards", []))
    n_eps = len(ep_rewards)
    wins = sum(ep_wins)
    avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    print(f"Iter {iteration} | Episodes: {n_eps} | Wins: {wins}/{n_eps} | Avg Reward: {avg_r:.2f} | Steps: {trainer.total_timesteps}")

trainer.close()
print("Done")
