"""Fast end-to-end smoke test for the current hybrid PPO contract."""

import os
import sys
import tempfile
from typing import Any, Dict, cast

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model_contract import ACTION_HEAD_SIZES, OBS_SIZE
from train_hybrid import HybridTrainer


def main() -> None:
    output = tempfile.TemporaryDirectory()
    trainer = HybridTrainer(
        levels=[111, 112, 114],
        n_steps=1024,
        batch_size=64,
        n_epochs=1,
        num_workers=1,
        max_episode_steps=720,
        output_dir=output.name,
    )
    try:
        trainer.send_weights()
        rollout = cast(Dict[str, Any], trainer.collect_rollout(10_000))
        observations = np.asarray(rollout["obs"], dtype=np.float32)
        actions = np.asarray(rollout["actions"], dtype=np.int64)
        assert observations.shape == (1024, OBS_SIZE), observations.shape
        assert actions.shape == (1024, len(ACTION_HEAD_SIZES)), actions.shape
        for head, size in enumerate(ACTION_HEAD_SIZES):
            assert np.all(actions[:, head] >= 0)
            assert np.all(actions[:, head] < size)
        trainer.populate_buffer(rollout)
        assert len(rollout["episode_rewards"]) >= 1
        print(
            f"Hybrid smoke test passed: observations={observations.shape}, "
            f"actions={actions.shape}, episodes={len(rollout['episode_rewards'])}"
        )
    finally:
        trainer.close()
        output.cleanup()


if __name__ == "__main__":
    main()
