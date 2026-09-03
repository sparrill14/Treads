"""Run the hybrid trainer with checkpoint-based abnormal-exit recovery."""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, cast

sys.path.insert(0, os.path.dirname(__file__))

from model_contract import assert_contract_compatible  # noqa: E402


def _atomic_json_dump(path: str, payload: Dict[str, Any]) -> None:
    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def find_latest_checkpoint(output_dir: str) -> Optional[Tuple[str, int]]:
    candidates: List[Tuple[int, str]] = []
    for state_path in glob.glob(os.path.join(output_dir, "*.trainer.json")):
        model_path = state_path[: -len(".trainer.json")] + ".zip"
        if not os.path.isfile(model_path):
            continue
        try:
            with open(state_path, "r", encoding="utf-8") as handle:
                state = cast(Dict[str, Any], json.load(handle))
            if int(state.get("schemaVersion", 0)) != 3:
                continue
            assert_contract_compatible(model_path)
            candidates.append((int(state.get("totalEpisodes", -1)), model_path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    episodes, path = max(candidates, key=lambda item: (item[0], os.path.getmtime(item[1])))
    return path, episodes


def _option_value(arguments: List[str], option: str) -> Optional[str]:
    for index, argument in enumerate(arguments):
        if argument == option:
            return arguments[index + 1] if index + 1 < len(arguments) else None
        if argument.startswith(option + "="):
            return argument[len(option) + 1 :]
    return None


def _without_option(arguments: List[str], option: str) -> List[str]:
    cleaned: List[str] = []
    index = 0
    while index < len(arguments):
        if arguments[index] == option:
            index += 2
            continue
        if arguments[index].startswith(option + "="):
            index += 1
            continue
        cleaned.append(arguments[index])
        index += 1
    return cleaned


def _read_trainer_status(output_dir: str) -> Optional[str]:
    manifest_path = os.path.join(output_dir, "run_manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = cast(Dict[str, Any], json.load(handle))
        status = str(payload.get("status", ""))
        return status or None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--restart-delay", type=float, default=5.0)
    args, trainer_args = parser.parse_known_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    status_path = os.path.join(output_dir, "supervisor.json")
    trainer_path = os.path.join(os.path.dirname(__file__), "train_hybrid.py")
    trainer_args = list(trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    trainer_args = _without_option(trainer_args, "--output-dir")
    base_command = [sys.executable, trainer_path, "--output-dir", output_dir, *trainer_args]
    initial_checkpoint = _option_value(trainer_args, "--load-model")
    initial_checkpoint_episodes = 0
    if initial_checkpoint:
        try:
            assert_contract_compatible(initial_checkpoint)
            state_path = initial_checkpoint.removesuffix(".zip") + ".trainer.json"
            with open(state_path, "r", encoding="utf-8") as handle:
                initial_state = cast(Dict[str, Any], json.load(handle))
            if int(initial_state.get("schemaVersion", 0)) != 3:
                raise ValueError(f"Unsupported trainer state schema in {state_path}")
            initial_checkpoint_episodes = int(initial_state.get("totalEpisodes", 0))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _atomic_json_dump(
                status_path,
                {
                    "status": "failed",
                    "supervisorPid": os.getpid(),
                    "error": f"Invalid initial checkpoint: {exc}",
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"Invalid initial checkpoint: {exc}", file=sys.stderr)
            return 2

    child: Optional[subprocess.Popen[bytes]] = None
    shutdown_requested = False

    def request_shutdown(signum: int, _frame: Any) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
        _atomic_json_dump(
            status_path,
            {
                "status": "stopping",
                "supervisorPid": os.getpid(),
                "childPid": child.pid if child is not None else None,
                "signal": signum,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
        )

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_shutdown)

    restart_count = 0
    resume_checkpoint: Optional[str] = None
    while True:
        command = list(base_command) if restart_count == 0 else _without_option(base_command, "--load-model")
        if restart_count > 0 and resume_checkpoint:
            command.extend(["--load-model", resume_checkpoint])
        child = subprocess.Popen(command)
        _atomic_json_dump(
            status_path,
            {
                "status": "running",
                "supervisorPid": os.getpid(),
                "childPid": child.pid,
                "restartCount": restart_count,
                "resumeCheckpoint": resume_checkpoint,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
        )
        exit_code = child.wait()
        if shutdown_requested:
            _atomic_json_dump(
                status_path,
                {
                    "status": "interrupted",
                    "supervisorPid": os.getpid(),
                    "childPid": child.pid,
                    "restartCount": restart_count,
                    "lastExitCode": exit_code,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 130
        if exit_code == 0:
            trainer_status = _read_trainer_status(output_dir)
            terminal_status = (
                trainer_status
                if trainer_status in {"completed", "budget_exhausted", "interrupted", "failed"}
                else "completed"
            )
            _atomic_json_dump(
                status_path,
                {
                    "status": terminal_status,
                    "supervisorPid": os.getpid(),
                    "childPid": child.pid,
                    "restartCount": restart_count,
                    "lastExitCode": exit_code,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 1 if terminal_status == "failed" else (130 if terminal_status == "interrupted" else 0)

        latest = find_latest_checkpoint(output_dir)
        if restart_count >= max(0, args.max_restarts):
            _atomic_json_dump(
                status_path,
                {
                    "status": "failed",
                    "supervisorPid": os.getpid(),
                    "childPid": child.pid,
                    "restartCount": restart_count,
                    "lastExitCode": exit_code,
                    "error": "Restart limit reached",
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                },
            )
            return exit_code or 1

        if latest is None:
            resume_checkpoint = initial_checkpoint
            episodes = initial_checkpoint_episodes
        else:
            resume_checkpoint, episodes = latest
        restart_count += 1
        _atomic_json_dump(
            status_path,
            {
                "status": "restarting",
                "supervisorPid": os.getpid(),
                "childPid": child.pid,
                "restartCount": restart_count,
                "lastExitCode": exit_code,
                "resumeCheckpoint": resume_checkpoint,
                "resumeEpisodes": episodes,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
        )
        time.sleep(max(0.0, args.restart_delay))


if __name__ == "__main__":
    raise SystemExit(main())
