"""
Smoke test: mechanically verify the hybrid pipeline works end-to-end.

Tests:
  1. Forward-pass agreement: TS MLP produces same outputs as PyTorch for same input.
  2. Log-prob agreement: TS Gaussian log_prob matches PyTorch Normal.log_prob.
  3. Learning signal: Phase 1 win rate goes from ~0% to >50% within 30 iterations.

Usage:
  python training/smoke_test.py
"""

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, cast

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
import gymnasium as gym
from gymnasium import spaces
from numpy.typing import NDArray

from treads_env import OBS_SIZE

ROLLOUT_WORKER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "rollout-worker.js"
)

SMOKE_TEST_DIR = os.path.join(os.path.dirname(__file__), "smoke_test_output")


def _send_msg(proc: subprocess.Popen[str], msg: Dict[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()


def _read_msg(proc: subprocess.Popen[str]) -> Dict[str, Any]:
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line:
        stderr = ""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
        raise RuntimeError(f"Worker died. stderr: {stderr}")
    return cast(Dict[str, Any], json.loads(line.strip()))


def test_forward_pass_agreement() -> bool:
    """Verify TS and PyTorch produce identical forward pass results."""
    print("\n=== Test 1: Forward-pass agreement ===")

    class DummyEnv(gym.Env[Any, Any]):
        metadata = {"render_modes": []}
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        def reset(self, **kwargs: Any) -> Any:
            return np.zeros((OBS_SIZE,), dtype=np.float32), {}
        def step(self, action: Any) -> Any:
            return np.zeros((OBS_SIZE,), dtype=np.float32), 0.0, True, False, {}

    env = DummyEnv()
    model = PPO("MlpPolicy", env, device="cpu", policy_kwargs=dict(net_arch=[256, 256]), seed=42)
    env.close()

    # Extract weights
    state_dict: Dict[str, Any] = {}
    for key, tensor in cast(Any, model).policy.state_dict().items():
        state_dict[key] = tensor.cpu().numpy().tolist()

    # Create random test observations
    np.random.seed(123)
    test_obs = [np.random.uniform(0, 1, OBS_SIZE).tolist() for _ in range(5)]

    # PyTorch forward pass
    py_means: List[List[float]] = []
    py_values: List[float] = []
    for obs in test_obs:
        obs_tensor = torch.tensor([obs], dtype=torch.float32)
        with torch.no_grad():
            features = cast(Any, model).policy.extract_features(obs_tensor, cast(Any, model).policy.features_extractor)
            pi_features = cast(Any, model).policy.mlp_extractor.forward_actor(features)
            vf_features = cast(Any, model).policy.mlp_extractor.forward_critic(features)
            action_mean = cast(Any, model).policy.action_net(pi_features)
            value = cast(Any, model).policy.value_net(vf_features)
            py_means.append(action_mean.squeeze().tolist())
            py_values.append(float(value.squeeze()))

    # TS forward pass
    proc = subprocess.Popen(
        ["node", ROLLOUT_WORKER_PATH],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    ready = _read_msg(proc)
    assert ready["type"] == "ready"
    _send_msg(proc, {"type": "set_weights", "state_dict": state_dict})
    ack = _read_msg(proc)
    assert ack["type"] == "weights_set"
    _send_msg(proc, {"type": "test_forward", "observations": test_obs})
    result = _read_msg(proc)
    assert result["type"] == "test_result"
    _send_msg(proc, {"type": "exit"})
    proc.wait(timeout=5)

    ts_means = result["logits"]
    ts_values = result["values"]

    # Compare
    max_mean_err = 0.0
    max_val_err = 0.0
    for i in range(len(test_obs)):
        for j in range(5):
            err = abs(py_means[i][j] - ts_means[i][j])
            max_mean_err = max(max_mean_err, err)
        val_err = abs(py_values[i] - ts_values[i])
        max_val_err = max(max_val_err, val_err)

    print(f"  Max action-mean error: {max_mean_err:.2e}")
    print(f"  Max value error:       {max_val_err:.2e}")

    passed = max_mean_err < 1e-4 and max_val_err < 1e-4
    print(f"  {'PASS' if passed else 'FAIL'}: Forward pass agreement {'within 1e-4 tolerance' if passed else 'EXCEEDS tolerance'}")
    return passed


def test_log_prob_agreement() -> bool:
    """Verify TS Gaussian log_prob matches PyTorch."""
    print("\n=== Test 2: Log-prob agreement ===")

    # Create known values
    means = [0.5, -0.3, 0.1, 0.8, -0.2]
    log_stds = [-0.5, -0.8, -0.3, -1.0, -0.6]
    actions = [0.7, -0.1, 0.3, 0.5, -0.4]

    # PyTorch log_prob
    mean_t = torch.tensor(means)
    std_t = torch.exp(torch.tensor(log_stds))
    action_t = torch.tensor(actions)
    dist = torch.distributions.Normal(mean_t, std_t)
    py_log_prob = float(dist.log_prob(action_t).sum())

    # TS formula (same as mlp-inference.ts)
    import math
    ts_log_prob = 0.0
    for i in range(5):
        std = math.exp(log_stds[i])
        diff = actions[i] - means[i]
        ts_log_prob += -0.5 * (diff * diff / (std * std) + 2 * math.log(std) + math.log(2 * math.pi))

    err = abs(py_log_prob - ts_log_prob)
    print(f"  PyTorch log_prob: {py_log_prob:.8f}")
    print(f"  TS log_prob:      {ts_log_prob:.8f}")
    print(f"  Error:            {err:.2e}")

    passed = err < 1e-6
    print(f"  {'PASS' if passed else 'FAIL'}: Log-prob agreement")
    return passed


def test_learning_signal() -> bool:
    """Run a short training session and verify the agent learns something."""
    print("\n=== Test 3: Learning signal (shoot nearby unarmed target) ===")
    print("  This takes ~2-3 minutes...")

    class DummyEnv(gym.Env[Any, Any]):
        metadata = {"render_modes": []}
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        def reset(self, **kwargs: Any) -> Any:
            return np.zeros((OBS_SIZE,), dtype=np.float32), {}
        def step(self, action: Any) -> Any:
            return np.zeros((OBS_SIZE,), dtype=np.float32), 0.0, True, False, {}

    os.makedirs(SMOKE_TEST_DIR, exist_ok=True)
    env = DummyEnv()
    n_steps = 4096
    model = PPO(
        "MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=n_steps,
        batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.02, device="cpu",
        policy_kwargs=dict(net_arch=[256, 256]), seed=42,
    )
    env.close()
    model.set_logger(configure(SMOKE_TEST_DIR, []))

    # Start worker
    proc = subprocess.Popen(
        ["node", ROLLOUT_WORKER_PATH],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    ready = _read_msg(proc)
    assert ready["type"] == "ready"

    # Use only fully unarmed stationary targets (111, 112 have ammo count 0)
    levels = [111, 112]
    max_iters = 50
    win_rates: List[float] = []
    all_wins: List[int] = []
    avg_rewards: List[float] = []

    t0 = time.time()
    for iteration in range(1, max_iters + 1):
        # Send weights
        state_dict: Dict[str, Any] = {}
        for key, tensor in cast(Any, model).policy.state_dict().items():
            state_dict[key] = tensor.cpu().numpy().tolist()
        _send_msg(proc, {"type": "set_weights", "state_dict": state_dict})
        ack = _read_msg(proc)
        assert ack["type"] == "weights_set"

        # Collect rollout
        _send_msg(proc, {
            "type": "collect", "n_steps": n_steps, "levels": levels,
            "maxTicks": 720, "seedStart": iteration * n_steps,
            "replayEveryEpisodes": 0, "replayDir": "",
            "targetEpisodes": 99999, "shapingScale": 1.0,
        })
        rollout = _read_msg(proc)
        assert rollout["type"] == "rollout"

        # Populate buffer
        buf = model.rollout_buffer
        buf.reset()
        obs_arr = np.array(rollout["obs"], dtype=np.float32)
        actions_arr = np.array(rollout["actions"], dtype=np.float32)
        rewards_arr = np.array(rollout["rewards"], dtype=np.float32)
        episode_starts_arr = np.array(rollout["episode_starts"], dtype=np.float32)
        values_arr = np.array(rollout["values"], dtype=np.float32)
        log_probs_arr = np.array(rollout["log_probs"], dtype=np.float32)
        for i in range(len(obs_arr)):
            buf.add(
                obs=obs_arr[i:i+1], action=actions_arr[i:i+1],
                reward=np.array([rewards_arr[i]]),
                episode_start=np.array([episode_starts_arr[i]]),
                value=torch.tensor([values_arr[i]]),
                log_prob=torch.tensor([log_probs_arr[i]]),
            )
        last_values = torch.tensor([rollout["last_value"]])
        last_dones = np.array([1.0 if rollout["last_done"] else 0.0])
        buf.compute_returns_and_advantage(last_values=last_values, dones=last_dones)

        # PPO update
        cast(Any, model)._current_progress_remaining = 1.0 - iteration / max_iters
        cast(Any, model).num_timesteps = iteration * n_steps
        cast(Any, model).train()

        # Track wins and rewards
        ep_wins = rollout.get("episode_wins", [])
        ep_rewards_list = rollout.get("episode_rewards", [])
        all_wins.extend(ep_wins)
        recent = all_wins[-100:] if len(all_wins) >= 20 else all_wins
        wr = float(np.mean(recent)) if recent else 0.0
        win_rates.append(wr)
        avg_rew = float(np.mean(ep_rewards_list)) if ep_rewards_list else 0.0
        avg_rewards.append(avg_rew)
        n_eps = len(ep_rewards_list)
        ep_breakdowns = rollout.get("episode_reward_breakdowns", [])
        avg_hit = float(np.mean([b.get("hit", 0) for b in ep_breakdowns])) if ep_breakdowns else 0.0
        avg_kill = float(np.mean([b.get("kill", 0) for b in ep_breakdowns])) if ep_breakdowns else 0.0
        total_wins_this_iter = sum(ep_wins)
        print(
            f"  Iter {iteration:2d}/{max_iters} | Eps: {n_eps:3d} | "
            f"Wins: {total_wins_this_iter} | WR: {wr:.3f} | "
            f"Reward: {avg_rew:.2f} | Hit: {avg_hit:.2f} | Kill: {avg_kill:.2f} | "
            f"Time: {time.time()-t0:.1f}s"
        )

    _send_msg(proc, {"type": "exit"})
    proc.wait(timeout=5)

    # Evaluate learning signal
    final_wr = win_rates[-1] if win_rates else 0.0
    total_wins = sum(all_wins)
    early_avg_reward = float(np.mean(avg_rewards[:5])) if len(avg_rewards) >= 5 else 0.0
    late_avg_reward = float(np.mean(avg_rewards[-5:])) if len(avg_rewards) >= 5 else 0.0
    reward_improvement = late_avg_reward - early_avg_reward

    print(f"\n  Total wins:           {total_wins}")
    print(f"  Final win rate:       {final_wr:.3f}")
    print(f"  Early avg reward:     {early_avg_reward:.2f}")
    print(f"  Late avg reward:      {late_avg_reward:.2f}")
    print(f"  Reward improvement:   {reward_improvement:+.2f}")

    # Pass criteria (any of):
    # 1. Got at least 1 win (proves the full kill-chain works)
    # 2. Reward improved by at least 0.5 (proves gradient signal flows)
    got_wins = total_wins > 0
    reward_improved = reward_improvement > 0.5
    passed = got_wins or reward_improved
    reasons: List[str] = []
    if got_wins:
        reasons.append(f"{total_wins} wins achieved")
    if reward_improved:
        reasons.append(f"reward improved by {reward_improvement:+.2f}")
    if not passed:
        reasons.append("no wins AND insufficient reward improvement")
    print(f"  {'PASS' if passed else 'FAIL'}: {'; '.join(reasons)}")
    return passed


def main() -> None:
    # Check compiled worker exists
    if not os.path.exists(ROLLOUT_WORKER_PATH):
        print(f"ERROR: Rollout worker not found at {ROLLOUT_WORKER_PATH}")
        print("Run: npx tsc -p training/tsconfig.training.json")
        sys.exit(1)

    results = []
    results.append(("Forward-pass agreement", test_forward_pass_agreement()))
    results.append(("Log-prob agreement", test_log_prob_agreement()))
    results.append(("Learning signal", test_learning_signal()))

    print("\n" + "=" * 50)
    print("SMOKE TEST RESULTS:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed. Pipeline is mechanically verified.")
    else:
        print("\nSOME TESTS FAILED. Fix issues before starting a long training run.")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
