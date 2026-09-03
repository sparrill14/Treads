"""Canonical observation/action contract shared by training and browser inference."""

import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple, cast

CONTRACT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "game",
    "controllers",
    "neural-model-contract.json",
)

with open(CONTRACT_PATH, "r", encoding="utf-8") as _handle:
    MODEL_CONTRACT = cast(Dict[str, Any], json.load(_handle))

CONTRACT_VERSION = int(MODEL_CONTRACT["contractVersion"])
OBSERVATION_VERSION = str(MODEL_CONTRACT["observationVersion"])
ACTION_VERSION = str(MODEL_CONTRACT["actionVersion"])
OBS_SIZE = int(MODEL_CONTRACT["observation"]["size"])
SELF_DIM = int(MODEL_CONTRACT["observation"]["selfDim"])
ENEMY_DIM = int(MODEL_CONTRACT["observation"]["enemyDim"])
ACTION_HEAD_SIZES: Tuple[int, ...] = tuple(
    int(value) for value in cast(Iterable[int], MODEL_CONTRACT["action"]["headSizes"])
)
NUM_AIM_BINS = ACTION_HEAD_SIZES[1]
NEUTRAL_AIM_BIN = int(MODEL_CONTRACT["action"]["neutralAimBin"])
AIM_RESIDUAL_MIN = float(MODEL_CONTRACT["action"]["aimResidualMinRadians"])
AIM_RESIDUAL_MAX = float(MODEL_CONTRACT["action"]["aimResidualMaxRadians"])


def checkpoint_contract_path(model_path: str) -> str:
    root, extension = os.path.splitext(model_path)
    if extension.lower() not in {".zip", ".onnx"}:
        root = model_path
    return root + ".contract.json"


def write_contract_for_model(model_path: str) -> str:
    destination = checkpoint_contract_path(model_path)
    temp_path = f"{destination}.{os.getpid()}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(MODEL_CONTRACT, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        return destination
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def assert_contract_compatible(model_path: str, action_space: Optional[Any] = None) -> None:
    sidecar = checkpoint_contract_path(model_path)
    if not os.path.isfile(sidecar):
        raise ValueError(
            f"Checkpoint is missing its model contract sidecar: {sidecar}. "
            "Legacy checkpoints cannot be loaded by the current runtime."
        )
    with open(sidecar, "r", encoding="utf-8") as handle:
        actual = cast(Dict[str, Any], json.load(handle))
    expected_keys = (
        "contractVersion",
        "observationVersion",
        "actionVersion",
        "observation",
        "action",
    )
    mismatches = [
        key for key in expected_keys if actual.get(key) != MODEL_CONTRACT.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint model contract is incompatible with this runtime: "
            + ", ".join(
                f"{key}={actual.get(key)!r}, expected={MODEL_CONTRACT.get(key)!r}"
                for key in mismatches
            )
        )
    if action_space is not None:
        nvec = tuple(int(value) for value in getattr(action_space, "nvec", ()))
        if nvec != ACTION_HEAD_SIZES:
            raise ValueError(
                f"Checkpoint action space {nvec} does not match contract {ACTION_HEAD_SIZES}."
            )
