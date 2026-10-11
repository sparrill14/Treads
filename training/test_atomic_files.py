"""Verify file-lock retries preserve existing artifacts and remain bounded."""

import os
import tempfile
import time
from pathlib import Path

from atomic_files import atomic_replace


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "checkpoint"
        source = Path(directory) / "checkpoint.tmp"
        destination.write_bytes(b"old checkpoint")
        source.write_bytes(b"new checkpoint")
        if os.name == "nt":
            with destination.open("rb"):
                started = time.monotonic()
                try:
                    atomic_replace(str(source), str(destination), retry_seconds=0.1)
                except PermissionError:
                    assert time.monotonic() - started < 1.0
                else:
                    raise AssertionError("A persistently locked artifact must fail after the retry budget")
            assert destination.read_bytes() == b"old checkpoint"
            assert source.read_bytes() == b"new checkpoint"
        atomic_replace(str(source), str(destination))
        assert destination.read_bytes() == b"new checkpoint"
        assert not source.exists()
        started = time.monotonic()
        try:
            atomic_replace(str(source), str(destination))
        except FileNotFoundError:
            assert time.monotonic() - started < 1.0
        else:
            raise AssertionError("Non-lock errors must propagate")
        assert destination.read_bytes() == b"new checkpoint"
    print("Atomic replacement preserves artifacts, bounds lock retries, and propagates other errors")


if __name__ == "__main__":
    main()
