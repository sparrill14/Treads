"""Run pipeline integration tests with the same interpreter on every platform."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    directory = Path(__file__).resolve().parent
    # Legacy manual scripts lack assertions or write into the default training run.
    excluded = {'test_env.py', 'test_hybrid.py'}
    tests = [test for test in sorted(directory.glob('test_*.py')) if test.name not in excluded]
    tests.append(directory / 'smoke_test.py')
    for test in tests:
        print(f'Running {test.name}', flush=True)
        subprocess.run([sys.executable, str(test)], check=True, cwd=directory.parent)


if __name__ == '__main__':
    main()
