"""
Integration test for both gRPC services:
  1. TreadsEnvService (grpc-env-server.ts) — Reset/Step/Close lifecycle
  2. RolloutWorkerService (rollout-worker.ts) — Health/SetWeightsFromFile/Collect/Close

Usage:
  python training/test_grpc.py
"""

import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple, cast

import grpc
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from runtime_codec import OBS_SIZE

# ---- Paths ----
GRPC_ENV_SERVER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "grpc-env-server.js"
)
ROLLOUT_WORKER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "rollout-worker.js"
)
PROTO_PATH = os.path.join(os.path.dirname(__file__), "proto", "treads.proto")
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "_generated")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_stubs() -> Tuple[Any, Any]:
    """Generate and import protobuf stubs for treads.proto."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    if GENERATED_DIR not in sys.path:
        sys.path.insert(0, GENERATED_DIR)

    pb2_path = os.path.join(GENERATED_DIR, "treads_pb2.py")
    pb2_grpc_path = os.path.join(GENERATED_DIR, "treads_pb2_grpc.py")
    if not (os.path.exists(pb2_path) and os.path.exists(pb2_grpc_path)):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{os.path.dirname(PROTO_PATH)}",
                f"--python_out={GENERATED_DIR}",
                f"--grpc_python_out={GENERATED_DIR}",
                PROTO_PATH,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to generate protobuf stubs. "
                f"stderr={result.stderr[-500:]}"
            )

    pb2 = __import__("treads_pb2")
    pb2_grpc = __import__("treads_pb2_grpc")
    return pb2, pb2_grpc


def _start_server(js_path: str, port: int) -> subprocess.Popen[str]:
    """Start a Node.js gRPC server process."""
    if not os.path.exists(js_path):
        raise FileNotFoundError(f"Server JS not found: {js_path}")

    proc = subprocess.Popen(
        ["node", js_path, "--port", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def _wait_for_health(stub: Any, pb2: Any, timeout: float = 10.0) -> None:
    """Wait until the server responds to Health RPC."""
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            resp = stub.Health(pb2.HealthRequest(), timeout=2.0)
            if resp.ok:
                return
        except Exception as exc:
            last_err = exc
            time.sleep(0.15)
    raise RuntimeError(f"Server did not become healthy within {timeout}s: {last_err}")


def _kill(proc: subprocess.Popen[str]) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


# ============================================================
# Test 1: TreadsEnvService (grpc-env-server)
# ============================================================
def test_treads_env_service() -> bool:
    print("\n=== Test 1: TreadsEnvService (grpc-env-server) ===")
    pb2, pb2_grpc = _ensure_stubs()

    port = _find_free_port()
    proc = _start_server(GRPC_ENV_SERVER_PATH, port)
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = pb2_grpc.TreadsEnvServiceStub(channel)

    try:
        # Health check
        _wait_for_health(stub, pb2)
        print("  [OK] Health check passed")

        # Reset
        reset_resp = stub.Reset(
            pb2.ResetRequest(
                session_id="test-session",
                level=1,
                seed=42,
                max_ticks=200,
                save_replay=False,
            ),
            timeout=5.0,
        )
        assert reset_resp.session_id, "Reset should return a session ID"
        assert reset_resp.init_json, "Reset should return init JSON"
        assert reset_resp.observation_json, "Reset should return observation JSON"
        init_data = json.loads(reset_resp.init_json)
        obs_data = json.loads(reset_resp.observation_json)
        assert "arena" in init_data, "Init should contain arena"
        assert "self" in obs_data, "Observation should contain self"
        assert "enemies" in obs_data, "Observation should contain enemies"
        print(f"  [OK] Reset: session={reset_resp.session_id}, tanks={len(init_data.get('tanks', []))}")

        # Step a few times
        steps_done = 0
        done = False
        for _ in range(50):
            step_resp = stub.Step(
                pb2.StepRequest(
                    session_id=reset_resp.session_id,
                    move="e",
                    aim_angle=0.5,
                    fire=True,
                    plant_bomb=False,
                ),
                timeout=5.0,
            )
            steps_done += 1
            if step_resp.done:
                done = True
                break

        if done:
            result = json.loads(step_resp.result_json) if step_resp.result_json else {}
            print(f"  [OK] Episode ended after {steps_done} steps. Status={result.get('status', '?')}")
        else:
            print(f"  [OK] Stepped {steps_done} times without episode end")

        # Close session
        close_resp = stub.Close(
            pb2.CloseRequest(session_id=reset_resp.session_id),
            timeout=5.0,
        )
        assert close_resp.closed, "Close response should indicate closed"
        print("  [OK] Session closed")

        # Try stepping with invalid session → should get NOT_FOUND
        try:
            stub.Step(
                pb2.StepRequest(session_id="nonexistent", move="n"),
                timeout=5.0,
            )
            print("  [WARN] Expected NOT_FOUND error for invalid session")
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.NOT_FOUND:
                print("  [OK] Invalid session correctly returns NOT_FOUND")
            else:
                print(f"  [WARN] Got unexpected error code: {exc.code()}")

        print("  PASSED: TreadsEnvService")
        return True

    except Exception as exc:
        print(f"  FAILED: {exc}")
        stderr = proc.stderr.read() if proc.stderr else ""
        if stderr:
            print(f"  Server stderr: {stderr[:500]}")
        return False
    finally:
        channel.close()
        _kill(proc)


# ============================================================
# Test 2: RolloutWorkerService (rollout-worker)
# ============================================================
def test_rollout_worker_service() -> bool:
    print("\n=== Test 2: RolloutWorkerService (rollout-worker) ===")
    pb2, pb2_grpc = _ensure_stubs()

    port = _find_free_port()
    proc = _start_server(ROLLOUT_WORKER_PATH, port)
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = pb2_grpc.RolloutWorkerServiceStub(channel)

    try:
        # Health check (should say weights_not_set)
        _wait_for_health(stub, pb2)
        health = stub.Health(pb2.HealthRequest(), timeout=5.0)
        assert health.ok, "Health should be ok"
        print(f"  [OK] Health check: message='{health.message}'")

        # Collect without weights → should fail with FAILED_PRECONDITION
        try:
            stub.Collect(
                pb2.CollectRequest(n_steps=64, levels=[1]),
                timeout=10.0,
            )
            print("  [WARN] Expected FAILED_PRECONDITION when collecting without weights")
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.FAILED_PRECONDITION:
                print("  [OK] Collect without weights correctly returns FAILED_PRECONDITION")
            else:
                print(f"  [WARN] Got unexpected error: {exc.code()} {exc.details()}")

        # Create dummy SB3 weights and send via file
        weights = _create_dummy_weights()
        weights_dir = os.path.join(os.path.dirname(__file__), "smoke_test_output")
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, "test_weights.json")
        with open(weights_path, "w") as f:
            json.dump(weights, f, separators=(",", ":"))

        set_resp = stub.SetWeightsFromFile(
            pb2.SetWeightsFromFileRequest(path=weights_path),
            timeout=10.0,
        )
        assert set_resp.ok, "SetWeightsFromFile should succeed"
        print(f"  [OK] Weights loaded from file")

        # Health check should now be 'ready'
        health2 = stub.Health(pb2.HealthRequest(), timeout=5.0)
        assert health2.message == "ready", f"Expected 'ready', got '{health2.message}'"
        print(f"  [OK] Health after weights: '{health2.message}'")

        # Collect a small rollout
        collect_resp = stub.Collect(
            pb2.CollectRequest(
                n_steps=128,
                levels=[1],
                max_ticks=200,
                tick_norm_ticks=200,
                seed_start=42,
                replay_every_episodes=0,
                worker_id=0,
                shaping_scale=1.0,
                procedural_levels=False,
                difficulty_band=0.0,
            ),
            timeout=30.0,
        )
        assert collect_resp.rollout_json, "Collect should return rollout JSON"
        rollout = json.loads(collect_resp.rollout_json)
        assert rollout["type"] == "rollout", f"Expected type='rollout', got {rollout['type']}"
        assert rollout["n_steps"] == 128, f"Expected n_steps=128, got {rollout['n_steps']}"
        assert len(rollout["obs"]) == 128, f"Expected 128 observations, got {len(rollout['obs'])}"
        assert len(rollout["actions"]) == 128, f"Expected 128 actions, got {len(rollout['actions'])}"
        assert len(rollout["rewards"]) == 128, f"Expected 128 rewards, got {len(rollout['rewards'])}"
        assert len(rollout["values"]) == 128, f"Expected 128 values, got {len(rollout['values'])}"
        assert len(rollout["log_probs"]) == 128, f"Expected 128 log_probs, got {len(rollout['log_probs'])}"

        # Verify observation dimensions
        for i, obs in enumerate(rollout["obs"]):
            assert len(obs) == OBS_SIZE, f"Obs[{i}] dim={len(obs)}, expected {OBS_SIZE}"

        # Action dimensions (5: move_x, move_y, aim, fire, bomb)
        for i, act in enumerate(rollout["actions"]):
            assert len(act) == 5, f"Action[{i}] dim={len(act)}, expected 5"

        n_episodes = len(rollout["episode_rewards"])
        n_wins = sum(rollout["episode_wins"])
        print(
            f"  [OK] Collect: n_steps={rollout['n_steps']}, "
            f"episodes={n_episodes}, wins={n_wins}, "
            f"last_done={rollout['last_done']}"
        )

        # Verify last_obs and last_value
        assert len(rollout["last_obs"]) == OBS_SIZE, f"last_obs dim mismatch"
        assert isinstance(rollout["last_value"], (int, float)), "last_value should be numeric"
        print(f"  [OK] last_obs dim={len(rollout['last_obs'])}, last_value={rollout['last_value']:.4f}")

        # Close
        close_resp = stub.Close(pb2.CloseRequest(), timeout=5.0)
        assert close_resp.closed, "Close should return closed=true"
        print("  [OK] Worker closed")

        print("  PASSED: RolloutWorkerService")
        return True

    except Exception as exc:
        print(f"  FAILED: {exc}")
        import traceback
        traceback.print_exc()
        stderr = proc.stderr.read() if proc.stderr else ""
        if stderr:
            print(f"  Server stderr: {stderr[:500]}")
        return False
    finally:
        channel.close()
        _kill(proc)


def _create_dummy_weights() -> Dict[str, Any]:
    """Create random SB3-compatible MLP weights for testing."""
    np.random.seed(0)
    obs_dim = OBS_SIZE
    hidden = 256
    action_dim = 5  # continuous: move_x, move_y, aim, fire, bomb

    def rand(shape: Tuple[int, ...]) -> Any:
        return (np.random.randn(*shape) * 0.1).tolist()

    return {
        "mlp_extractor.policy_net.0.weight": rand((hidden, obs_dim)),
        "mlp_extractor.policy_net.0.bias": rand((hidden,)),
        "mlp_extractor.policy_net.2.weight": rand((hidden, hidden)),
        "mlp_extractor.policy_net.2.bias": rand((hidden,)),
        "mlp_extractor.value_net.0.weight": rand((hidden, obs_dim)),
        "mlp_extractor.value_net.0.bias": rand((hidden,)),
        "mlp_extractor.value_net.2.weight": rand((hidden, hidden)),
        "mlp_extractor.value_net.2.bias": rand((hidden,)),
        "action_net.weight": rand((action_dim, hidden)),
        "action_net.bias": rand((action_dim,)),
        "value_net.weight": rand((1, hidden)),
        "value_net.bias": rand((1,)),
        "log_std": rand((action_dim,)),
    }


if __name__ == "__main__":
    results = []
    results.append(("TreadsEnvService", test_treads_env_service()))
    results.append(("RolloutWorkerService", test_rollout_worker_service()))

    print("\n" + "=" * 50)
    print("SUMMARY:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll gRPC tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED.")
        sys.exit(1)
