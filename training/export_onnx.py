"""
Export the trained SB3 PPO model to ONNX format for use in the browser.
Handles MultiDiscrete action space: outputs logits for [9, 16, 2, 2].
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from treads_env import OBS_SIZE, NUM_AIM_BINS

# Action dimensions for MultiDiscrete
ACTION_DIMS = [9, NUM_AIM_BINS, 2, 2]  # move, aim, fire, bomb
TOTAL_LOGITS = sum(ACTION_DIMS)


def export_to_onnx(model_path, onnx_path):
    """Export an SB3 PPO model to ONNX."""
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device="cpu")

    policy = model.policy

    class PolicyWrapper(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            features = self.policy.extract_features(obs, self.policy.features_extractor)
            latent_pi, _ = self.policy.mlp_extractor(features)
            # For MultiDiscrete, action_net outputs logits for all categories
            logits = self.policy.action_net(latent_pi)
            return logits

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    dummy_input = torch.zeros(1, OBS_SIZE, dtype=torch.float32)

    print("Exporting to ONNX...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["logits"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to {onnx_path}")

    # Verify with onnxruntime
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    test_obs = np.zeros((1, OBS_SIZE), dtype=np.float32)
    result = session.run(None, {"observation": test_obs})
    logits = result[0][0]
    print(f"ONNX verification - output shape: {result[0].shape}, total logits: {len(logits)}")

    # Decode to actions via argmax per group
    offset = 0
    action_names = ["move", "aim", "fire", "bomb"]
    for name, dim in zip(action_names, ACTION_DIMS):
        group = logits[offset : offset + dim]
        chosen = np.argmax(group)
        print(f"  {name}: logits={group}, chosen={chosen}")
        offset += dim


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    model_path = os.path.join(output_dir, "treads_ppo.zip")
    onnx_path = os.path.join(output_dir, "treads_policy.onnx")
    export_to_onnx(model_path, onnx_path)
