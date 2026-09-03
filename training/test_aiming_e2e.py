"""End-to-end regression for the long-range aiming scenario that failed with coarse bins."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model_contract import NEUTRAL_AIM_BIN
from runtime_codec import decode_multi_discrete_action
from treads_env import TreadsEnv


def main() -> None:
    env = TreadsEnv(
        level=132,
        max_episode_steps=720,
        player_max_bombs=0,
    )
    try:
        env.reset(seed=77)
        done = False
        steps = 0
        info: dict[str, object] = {}
        while not done:
            action = decode_multi_discrete_action(
                np.asarray([0, NEUTRAL_AIM_BIN, 1, 0], dtype=np.int64),
                env._last_obs_raw,
            )
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        result = info.get("result", {})
        assert isinstance(result, dict) and bool(result.get("win")), result
        assert steps < 240, steps
        print(f"Long-range target-relative aiming passed in {steps} ticks")
    finally:
        env.close()


if __name__ == "__main__":
    main()
