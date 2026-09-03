"""
Export the trained SB3 PPO model to ONNX format for use in the browser.
The exported network outputs raw logits matching the versioned action contract.
Browser-side code argmax-decodes each
contiguous slice into [move_idx, aim_bin, fire, bomb].
"""

import os
import sys
from typing import Any, cast
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(__file__))

from sb3_compat import ensure_pickle_compat
from stable_baselines3 import PPO
from treads_env import OBS_SIZE
from model_contract import (
    ACTION_HEAD_SIZES,
    ACTION_VERSION,
    NEUTRAL_AIM_BIN,
    assert_contract_compatible,
    write_contract_for_model,
)

ensure_pickle_compat()

ACTION_LOGITS_DIM = sum(ACTION_HEAD_SIZES)
MOVE_INTENTS = ["none", "n", "s", "e", "w", "ne", "nw", "se", "sw"]


def export_to_onnx(model_path: str, onnx_path: str) -> None:
    """Export an SB3 PPO model to ONNX."""
    print(f"Loading model from {model_path}...")
    model = cast(Any, PPO.load(model_path, device="cpu"))  # pyright: ignore[reportUnknownMemberType]
    assert_contract_compatible(model_path, model.action_space)

    policy = model.policy

    class PolicyWrapper(nn.Module):
        def __init__(self, policy_obj: Any) -> None:
            super().__init__()
            self.policy: Any = policy_obj

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            features = self.policy.extract_features(obs, self.policy.features_extractor)
            latent_pi, _ = self.policy.mlp_extractor(features)
            logits = self.policy.action_net(latent_pi)
            assert isinstance(logits, torch.Tensor)
            return logits

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
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["logits"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"ONNX model saved to {onnx_path}")
    write_contract_for_model(onnx_path)

    # Verify with onnxruntime
    import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]

    session = ort.InferenceSession(onnx_path)
    test_obs = np.zeros((1, OBS_SIZE), dtype=np.float32)
    result = cast(Any, session).run(None, {"observation": test_obs})
    output_arr = np.asarray(result[0])
    logits: NDArray[np.float32] = output_arr[0].astype(np.float32)
    print(f"ONNX verification - output shape: {output_arr.shape}, logits dim: {len(logits)}")
    if len(logits) != ACTION_LOGITS_DIM:
        raise RuntimeError(
            f"Expected {ACTION_LOGITS_DIM} logits, got {len(logits)}"
        )

    # Argmax-decode each head
    offset = 0
    decoded: list[int] = []
    for size in ACTION_HEAD_SIZES:
        slice_ = logits[offset:offset + size]
        decoded.append(int(np.argmax(slice_)))
        offset += size
    move_idx, aim_bin, fire, bomb = decoded
    move_dir = MOVE_INTENTS[move_idx]
    print(
        f"  decoded move={move_dir} aim_bin={aim_bin} "
        f"(neutral={NEUTRAL_AIM_BIN}, mode={ACTION_VERSION}) "
        f"fire={bool(fire)} bomb={bool(bomb)}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default=None)
    parser.add_argument("onnx_path", nargs="?", default=None)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    model_path = args.model_path
    if not model_path:
        model_path = os.path.join(output_dir, "treads_ppo_best.zip")
    onnx_path = args.onnx_path
    if not onnx_path:
        onnx_path = os.path.join(output_dir, "treads_policy.onnx")
    export_to_onnx(model_path, onnx_path)
