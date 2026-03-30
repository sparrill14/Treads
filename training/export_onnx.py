"""
Export the trained SB3 PPO model to ONNX format for use in the browser.
Handles continuous control action space: outputs action means for
[move_signal, aim_signal, fire_signal, bomb_signal] in [-1, 1].
"""

import os
import sys
from typing import Any, cast
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from treads_env import OBS_SIZE


def export_to_onnx(model_path: str, onnx_path: str) -> None:
    """Export an SB3 PPO model to ONNX."""
    print(f"Loading model from {model_path}...")
    model = cast(Any, PPO.load(model_path, device="cpu"))  # pyright: ignore[reportUnknownMemberType]

    policy = model.policy

    class PolicyWrapper(nn.Module):
        def __init__(self, policy_obj: Any) -> None:
            super().__init__()
            self.policy: Any = policy_obj

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            features = self.policy.extract_features(obs, self.policy.features_extractor)
            latent_pi, _ = self.policy.mlp_extractor(features)
            action_mean = self.policy.action_net(latent_pi)
            assert isinstance(action_mean, torch.Tensor)
            return action_mean

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    dummy_input = torch.zeros(1, OBS_SIZE, dtype=torch.float32)

    print("Exporting to ONNX...")
    onnx_export = cast(Any, torch.onnx.export)  # pyright: ignore[reportUnknownMemberType]
    onnx_export(
        wrapper,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action_mean"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action_mean": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to {onnx_path}")

    # Verify with onnxruntime
    import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]

    session = ort.InferenceSession(onnx_path)
    test_obs = np.zeros((1, OBS_SIZE), dtype=np.float32)
    result = cast(Any, session).run(None, {"observation": test_obs})
    output_arr = np.asarray(result[0])
    action_mean: NDArray[np.float32] = output_arr[0].astype(np.float32)
    print(f"ONNX verification - output shape: {output_arr.shape}, action dims: {len(action_mean)}")

    move_signal = float(np.clip(action_mean[0], -1.0, 1.0))
    aim_signal = float(np.clip(action_mean[1], -1.0, 1.0))
    fire_signal = float(np.clip(action_mean[2], -1.0, 1.0))
    bomb_signal = float(np.clip(action_mean[3], -1.0, 1.0))
    move_idx = int(round(((move_signal + 1.0) * 0.5) * 8.0))
    move_idx = max(0, min(8, move_idx))
    aim_angle = ((aim_signal + 1.0) * 0.5) * 2.0 * np.pi
    fire = fire_signal > 0
    bomb = bomb_signal > 0
    print(f"  decoded move_idx={move_idx} aim_angle={aim_angle:.3f} fire={fire} bomb={bomb}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    model_path = os.path.join(output_dir, "treads_ppo.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(output_dir, "treads_ppo_best.zip")
    onnx_path = os.path.join(output_dir, "treads_policy.onnx")
    export_to_onnx(model_path, onnx_path)
