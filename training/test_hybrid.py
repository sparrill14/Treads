"""Quick end-to-end test of the hybrid rollout pipeline with curriculum."""
import os, sys, time
from collections import Counter
from typing import Any, Dict, List, cast
sys.path.insert(0, os.path.dirname(__file__))

from train_hybrid import HybridTrainer

trainer = HybridTrainer(levels=[1, 2, 3, 4], n_steps=2048, max_episode_steps=720)
print("Trainer created. Testing pipeline with curriculum [1,2,3,4]...")

t0 = time.perf_counter()
trainer.send_weights()
t1 = time.perf_counter()
print(f"Weights sent in {t1-t0:.3f}s")

rollout = trainer.collect_rollout(trainer.total_episodes + 1)
rollout_dict = cast(Dict[str, Any], rollout)
t2 = time.perf_counter()
print(f"Rollout collected in {t2-t1:.3f}s")
print(f"  n_steps={rollout_dict['n_steps']}, episodes={len(rollout_dict['episode_rewards'])}")
print(f"  wins={sum(cast(List[int], rollout_dict['episode_wins']))}/{len(cast(List[int], rollout_dict['episode_wins']))}")

# Per-level breakdown
level_counts = Counter(cast(List[int], rollout_dict['episode_levels']))
for lvl in sorted(level_counts.keys()):
    lvl_wins = sum(
        1
        for w, l in zip(cast(List[int], rollout_dict['episode_wins']), cast(List[int], rollout_dict['episode_levels']))
        if l == lvl and w
    )
    lvl_total = level_counts[lvl]
    print(f"  Level {lvl}: {lvl_wins}/{lvl_total} wins")

if cast(List[float], rollout_dict['episode_rewards']):
    rewards = cast(List[float], rollout_dict['episode_rewards'])
    avg_r = sum(rewards) / len(rewards)
    print(f"  avg_reward={avg_r:.2f}")

trainer.populate_buffer(rollout_dict)
t3 = time.perf_counter()
print(f"Buffer populated in {t3-t2:.3f}s")

cast(Any, trainer.model)._current_progress_remaining = 1.0
cast(Any, trainer.model).num_timesteps = 2048
cast(Any, trainer.model).train()
t4 = time.perf_counter()
print(f"PPO train in {t4-t3:.3f}s")

rollout_sps = 2048 / (t2 - t1)
total_sps = 2048 / (t4 - t0)
print(f"\nRollout throughput: {rollout_sps:.0f} steps/s")
print(f"Total throughput: {total_sps:.0f} steps/s")

trainer.close()
print("Test complete!")
