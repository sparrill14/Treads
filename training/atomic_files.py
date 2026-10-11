"""Atomic replacement with bounded recovery from transient Windows file locks."""

import os
import time


def atomic_replace(source: str, destination: str, retry_seconds: float = 5.0) -> None:
    """Keep the existing file intact until a single atomic replacement succeeds."""
    deadline = time.monotonic() + retry_seconds
    delay = 0.05
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            remaining = deadline - time.monotonic()
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33) or remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.5)
