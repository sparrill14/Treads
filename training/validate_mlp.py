"""Validate that TS MLP inference matches PyTorch for same weights+input."""
import json
import os
import subprocess
import sys
from typing import Any, cast

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from treads_env import OBS_SIZE, TreadsEnvDiscrete

ROLLOUT_WORKER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "rollout-worker.js"
)

# Create model
env = cast(Any, Monitor(TreadsEnvDiscrete(level=1, max_episode_steps=720)))
model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]), seed=42, device="cpu")
env.close()

# Extract weights
state_dict = {}
for key, tensor in model.policy.state_dict().items():
    state_dict[key] = tensor.cpu().numpy().tolist()

# Create test observations
np.random.seed(42)
test_obs = np.random.rand(5, OBS_SIZE).astype(np.float32)

# PyTorch forward pass
print("=== PyTorch forward pass ===")
for i, obs in enumerate(test_obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        # Get features
        features = model.policy.extract_features(obs_tensor, model.policy.features_extractor)
        features_tensor = cast(torch.Tensor, features)
        pi_features = model.policy.mlp_extractor.forward_actor(features_tensor)
        vf_features = model.policy.mlp_extractor.forward_critic(features_tensor)
        action_logits = model.policy.action_net(pi_features)
        value = model.policy.value_net(vf_features)
    print(f"Obs {i}: logits[:5]={action_logits[0][:5].numpy()}, value={value[0][0].item():.6f}")

# TypeScript forward pass via worker
print("\n=== TypeScript forward pass ===")
worker = subprocess.Popen(
    ["node", ROLLOUT_WORKER_PATH],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1,
)
assert worker.stdin is not None and worker.stdout is not None
# Read ready
ready = json.loads(worker.stdout.readline())
assert ready["type"] == "ready"

# Send weights
worker.stdin.write(json.dumps({"type": "set_weights", "state_dict": state_dict}) + "\n")
worker.stdin.flush()
ack = json.loads(worker.stdout.readline())
assert ack["type"] == "weights_set"

# Send test observations for inference
# We'll use a special test command
worker.stdin.write(json.dumps({
    "type": "test_forward",
    "observations": test_obs.tolist(),
}) + "\n")
worker.stdin.flush()
result = json.loads(worker.stdout.readline())

if result.get("type") == "test_result":
    for i in range(len(test_obs)):
        logits = result["logits"][i]
        value = result["values"][i]
        print(f"Obs {i}: logits[:5]={logits[:5]}, value={value:.6f}")
else:
    print(f"Worker returned: {result.get('type', 'unknown')}")
    print("Expected test_result from rollout worker")

# Cleanup
worker.stdin.write(json.dumps({"type": "exit"}) + "\n")
worker.stdin.flush()
worker.wait(timeout=5)
