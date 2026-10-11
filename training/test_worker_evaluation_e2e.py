"""Exercise deterministic held-out evaluation inside the rollout workers."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))

from train_hybrid import HybridTrainer


def main() -> None:
    with tempfile.TemporaryDirectory() as output_dir:
        trainer = HybridTrainer(
            levels=[111],
            n_steps=256,
            batch_size=64,
            n_epochs=1,
            max_episode_steps=720,
            output_dir=output_dir,
            eval_episodes=4,
            num_workers=4,
        )
        try:
            started_at = time.perf_counter()
            win_rate = trainer._evaluate_levels([111], 4, 9100)
            elapsed = time.perf_counter() - started_at
            details = trainer.last_eval_details[111]
            assert 0.0 <= win_rate <= 1.0
            assert details["episodes"] == 4.0
            assert details["wins"] == win_rate * 4.0
            assert trainer.evaluation_progress is not None
            assert trainer.evaluation_progress["status"] == "completed"
            assert trainer.evaluation_progress["levelsCompleted"] == 1
            assert elapsed < 30.0, f"Worker evaluation took too long: {elapsed:.1f}s"
        finally:
            trainer.close()

    print(f"Worker-native evaluation passed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
