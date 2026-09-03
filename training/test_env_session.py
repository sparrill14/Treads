"""End-to-end regression test for terminal gRPC session idempotence."""

import os
import sys
from typing import Any, Dict, cast

sys.path.insert(0, os.path.dirname(__file__))

from model_contract import NEUTRAL_AIM_BIN  # noqa: E402
from runtime_codec import decode_multi_discrete_action  # noqa: E402
from treads_env import TreadsEnv  # noqa: E402


def main() -> None:
    env = TreadsEnv(level=111, seed_start=4242, max_episode_steps=720)
    try:
        for episode in range(5):
            _obs, _ = env.reset(seed=4242 + episode)
            done = False
            decoded: Dict[str, Any] = {}
            terminal_result: Dict[str, Any] = {}
            while not done:
                decoded = decode_multi_discrete_action(
                    [0, NEUTRAL_AIM_BIN, 1, 0], env._last_obs_raw
                )
                _obs, _reward, terminated, truncated, info = env.step(decoded)
                done = terminated or truncated
                terminal_result = cast(Dict[str, Any], info.get("result", {}))
            repeated = env._rpc_step(
                {
                    "move": "none",
                    "aimAngle": float(decoded["aim_angle"][0]),
                    "fire": False,
                    "plantBomb": False,
                }
            )
            assert repeated["done"] is True
            assert cast(Dict[str, Any], repeated["result"]).get("status") == terminal_result.get("status")
        print("Terminal gRPC session remained idempotent across 5 reset cycles")
    finally:
        env.close()


if __name__ == "__main__":
    main()
