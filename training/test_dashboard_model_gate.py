"""Verify that the dashboard exposes only contract-compatible ONNX models."""

import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from typing import cast


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_PATH = os.path.join(
    REPO_ROOT,
    ".training-dist",
    "training",
    "dashboard-server.js",
)
CONTRACT_PATH = os.path.join(
    REPO_ROOT,
    "src",
    "game",
    "controllers",
    "neural-model-contract.json",
)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fetch_json(port: int, path: str) -> dict[str, object]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2.0) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_model_status(port: int, run_id: str = "") -> dict[str, object]:
    query = "?" + urllib.parse.urlencode({"run": run_id}) if run_id else ""
    return fetch_json(port, "/api/model" + query)


def main() -> None:
    with tempfile.TemporaryDirectory() as output_dir:
        model_path = os.path.join(output_dir, "treads_policy.onnx")
        with open(model_path, "wb") as handle:
            handle.write(b"fixture")
        stale_dir = os.path.join(output_dir, "stale-run")
        os.makedirs(stale_dir)
        stale_manifest = os.path.join(stale_dir, "run_manifest.json")
        with open(stale_manifest, "w", encoding="utf-8") as handle:
            json.dump({"status": "running"}, handle)
        old_timestamp = time.time() - 120.0
        os.utime(stale_manifest, (old_timestamp, old_timestamp))

        port = free_port()
        environment = os.environ.copy()
        environment["TREADS_OUTPUT_ROOT"] = output_dir
        environment["TREADS_DASHBOARD_PORT"] = str(port)
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            ["node", DASHBOARD_PATH],
            cwd=REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        try:
            status = None
            deadline = time.time() + 10.0
            while time.time() < deadline and process.poll() is None:
                try:
                    status = fetch_model_status(port)
                    break
                except OSError:
                    time.sleep(0.1)
            assert status is not None, f"Dashboard failed to start, exit={process.poll()}"
            assert status["available"] is False
            runs = cast(list[dict[str, object]], fetch_json(port, "/api/runs")["runs"])
            stale = next(run for run in runs if run["id"] == "stale-run")
            assert stale["status"] == "stale"
            assert stale["isRunning"] is False

            with open(CONTRACT_PATH, "r", encoding="utf-8") as source:
                contract = json.load(source)
            contract_path = os.path.join(output_dir, "treads_policy.contract.json")
            incompatible = dict(contract)
            incompatible["observation"] = dict(contract["observation"])
            incompatible["observation"]["size"] -= 1
            with open(contract_path, "w", encoding="utf-8") as destination:
                json.dump(incompatible, destination)
            assert fetch_model_status(port)["available"] is False
            with open(
                contract_path,
                "w",
                encoding="utf-8",
            ) as destination:
                json.dump(contract, destination)
            compatible = fetch_model_status(port)
            assert compatible["available"] is True
            assert fetch_model_status(port, "missing-run")["available"] is False
        finally:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
    print("Dashboard model-contract gate passed")


if __name__ == "__main__":
    main()
