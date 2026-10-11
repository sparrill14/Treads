"""End-to-end test that the canonical navigation cue solves the obstacle bridge."""

import math
import os
import sys
from typing import Any, Dict, cast

sys.path.insert(0, os.path.dirname(__file__))

from model_contract import NEUTRAL_AIM_BIN  # noqa: E402
from runtime_codec import MOVE_DIR_MAP, MOVE_INTENTS, decode_multi_discrete_action, normalize_observation  # noqa: E402
from treads_env import TreadsEnv  # noqa: E402


def navigation_move_index(observation: Any) -> int:
    nav_x = float(observation[19]) * 2.0 - 1.0
    nav_y = float(observation[18]) * 2.0 - 1.0
    length = math.sqrt(nav_x * nav_x + nav_y * nav_y)
    assert length > 0.5
    nav_x /= length
    nav_y /= length
    candidates = [intent for intent in MOVE_INTENTS if intent != "none"]
    best = max(candidates, key=lambda intent: MOVE_DIR_MAP[intent][0] * nav_x + MOVE_DIR_MAP[intent][1] * nav_y)
    return MOVE_INTENTS.index(best)


def solve_with_navigation(seed: int, randomized: bool) -> None:
    env = TreadsEnv(
        level=141,
        seed_start=seed,
        max_episode_steps=720,
        spawn_jitter=randomized,
        procedural_levels=randomized,
        difficulty_band=0.38 if randomized else 0.0,
        player_max_ammo=3,
        player_max_bombs=0,
    )
    try:
        _obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        result: Dict[str, Any] = {}
        while not done:
            assert env._last_obs_raw is not None
            encoded = normalize_observation(env._last_obs_raw)
            move_index = navigation_move_index(encoded)
            decoded = decode_multi_discrete_action(
                [move_index, NEUTRAL_AIM_BIN, 1, 0], env._last_obs_raw
            )
            _obs, _reward, terminated, truncated, info = env.step(decoded)
            done = terminated or truncated
            steps += 1
            result = cast(Dict[str, Any], info.get("result", {}))
        assert result.get("win") is True, (seed, randomized, steps, result)
        assert steps < 720
    finally:
        env.close()


def main() -> None:
    for seed in (14100, 14101, 14102):
        solve_with_navigation(seed, randomized=False)

    for seed in (24100, 24101, 24102, 24103, 24104):
        solve_with_navigation(seed, randomized=True)

    print("Navigation guidance solved scenario 141 on deterministic and randomized variants")


if __name__ == "__main__":
    main()
