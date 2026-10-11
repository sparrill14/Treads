"""Verify that a current-contract checkpoint round-trips all state required for resume."""

import glob
import os
import sys
import tempfile
import threading
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from train_hybrid import HybridTrainer  # noqa: E402


def make_trainer(output_dir: str, load_model_path: Optional[str] = None) -> HybridTrainer:
    return HybridTrainer(
        n_steps=256,
        batch_size=64,
        n_epochs=1,
        num_workers=1,
        output_dir=output_dir,
        load_model_path=load_model_path,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as output_dir:
        first = make_trainer(output_dir)
        checkpoint = os.path.join(output_dir, "round_trip")
        try:
            first.total_episodes = 321
            first.total_timesteps = 654
            first.current_phase_index = 2
            first.peak_phase_index = 3
            first.evaluation_round = 7
            first.last_global_eval_lower_bound = 0.42
            first.last_global_eval_details = {141: {"wins": 8.0, "episodes": 12.0}}
            first.ret_rms.mean = 1.25
            first.ret_rms.var = 2.5
            model_path = first._save_model_checkpoint(checkpoint)
            if os.name == "nt":
                # An antivirus/indexer-style reader prevents deleting the old ZIP
                # until it releases its Windows handle.
                reader = open(model_path, "rb")
                release = threading.Timer(2.0, reader.close)
                release.start()
                try:
                    started = time.monotonic()
                    first._save_model_checkpoint(checkpoint)
                    assert time.monotonic() - started >= 1.5
                finally:
                    release.join()
                    reader.close()
        finally:
            first.close()

        assert not glob.glob(os.path.join(output_dir, "*.tmp*"))
        resumed = make_trainer(output_dir, model_path)
        try:
            assert resumed.total_episodes == 321
            assert resumed.total_timesteps == 654
            assert resumed.current_phase_index == 2
            assert resumed.peak_phase_index == 3
            assert resumed.evaluation_round == 7
            assert resumed.last_global_eval_lower_bound == 0.42
            assert resumed.last_global_eval_details[141]["wins"] == 8.0
            assert resumed.ret_rms.mean == 1.25
            assert resumed.ret_rms.var == 2.5
        finally:
            resumed.close()
    print("Atomic checkpoint and trainer-state resume passed")


if __name__ == "__main__":
    main()
