"""Regression test for large rollout transport over gRPC."""

import os
import sys
import tempfile
from typing import Any, Dict, cast

sys.path.insert(0, os.path.dirname(__file__))

from model_contract import OBS_SIZE
from train_hybrid import HybridTrainer


def main() -> None:
    steps = 8192
    output = tempfile.TemporaryDirectory()
    trainer = HybridTrainer(
        levels=[111, 112, 114],
        n_steps=steps,
        batch_size=512,
        n_epochs=1,
        num_workers=1,
        max_episode_steps=720,
        output_dir=output.name,
    )
    try:
        trainer.send_weights()
        rollout = cast(Dict[str, Any], trainer.collect_rollout(100_000))
        assert rollout["n_steps"] == steps
        assert len(rollout["obs"]) == steps
        assert len(rollout["obs"][0]) == OBS_SIZE
        assert len(rollout["actions"][0]) == 4
        assert len(rollout["worker_rollouts"][0]["obs"][0]) == OBS_SIZE
        print(f"Large gRPC rollout passed: {steps} transitions")
    finally:
        trainer.close()
        output.cleanup()


if __name__ == "__main__":
    main()
