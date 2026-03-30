"""Quick debug test: run a single rollout manually to verify the env works with SB3."""
import sys
import os
from typing import Any, cast
sys.path.insert(0, os.path.dirname(__file__))

from treads_env import TreadsEnvDiscrete
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import time

print("Creating env...")
env: Any = cast(Any, Monitor(TreadsEnvDiscrete(level=1, seed_start=0, max_episode_steps=600)))

print("Creating PPO model with small rollout...")
model: Any = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=128,
    batch_size=32,
    n_epochs=4,
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=[128, 128]),
)

print("Starting learn (512 timesteps = ~1 episode)...")
start = time.time()
model.learn(total_timesteps=512)
elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s")

env.close()
print("Success!")
