"""Verify TypeScript sampling values and log probabilities against SB3."""

import os
import sys
import tempfile
from typing import Any, Dict, cast

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from train_hybrid import HybridTrainer


def main() -> None:
    output = tempfile.TemporaryDirectory()
    trainer = HybridTrainer(
        levels=[111, 112, 114],
        n_steps=256,
        batch_size=64,
        n_epochs=1,
        num_workers=1,
        output_dir=output.name,
    )
    try:
        trainer.send_weights()
        rollout = cast(Dict[str, Any], trainer.collect_rollout(10_000))
        observations = torch.as_tensor(np.asarray(rollout["obs"], dtype=np.float32))
        actions = torch.as_tensor(np.asarray(rollout["actions"], dtype=np.int64))
        with torch.no_grad():
            values, log_probs, _ = cast(Any, trainer.model).policy.evaluate_actions(
                observations,
                actions,
            )
        expected_values = np.asarray(rollout["values"], dtype=np.float32)
        expected_log_probs = np.asarray(rollout["log_probs"], dtype=np.float32)
        value_error = float(np.max(np.abs(values.cpu().numpy().reshape(-1) - expected_values)))
        log_prob_error = float(np.max(np.abs(log_probs.cpu().numpy().reshape(-1) - expected_log_probs)))
        assert value_error < 1e-5, value_error
        assert log_prob_error < 1e-5, log_prob_error
        print(f"Sampler parity passed: value_error={value_error:.2e}, log_prob_error={log_prob_error:.2e}")
    finally:
        trainer.close()
        output.cleanup()


if __name__ == "__main__":
    main()
