"""Compare Python evaluation observations with the TypeScript rollout encoder."""

import json
import copy
import os
import subprocess
import sys
from typing import Any, Dict, List, cast

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model_contract import NEUTRAL_AIM_BIN  # noqa: E402
from runtime_codec import decode_multi_discrete_action, normalize_observation  # noqa: E402
from treads_env import TreadsEnv  # noqa: E402


CLI_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    ".training-dist",
    "training",
    "observation-parity-cli.js",
)


def main() -> None:
    samples: List[Dict[str, Any]] = []
    env = TreadsEnv(level=111, seed_start=9100, max_episode_steps=720)
    try:
        for offset, level in enumerate((111, 141, 171, 7)):
            env.level = level
            _observation, _info = env.reset(seed=9100 + offset)
            assert env._last_obs_raw is not None
            samples.append(cast(Dict[str, Any], env._last_obs_raw))
            for _step in range(12):
                decoded = decode_multi_discrete_action(
                    [3, NEUTRAL_AIM_BIN, 1, 0],
                    env._last_obs_raw,
                )
                _obs, _reward, terminated, truncated, _info = env.step(decoded)
                assert env._last_obs_raw is not None
                samples.append(cast(Dict[str, Any], env._last_obs_raw))
                if terminated or truncated:
                    break
    finally:
        env.close()

    busy = copy.deepcopy(samples[-1])
    busy["tick"] = 1500
    busy["rules"] = {"projectileBounces": False, "turretSpeedMultiplier": 2.5,
                     "projectileDamage": 3, "bombDamage": 5, "invulnerabilityTicks": 16}
    busy["self"].update(team="alpha", maxAmmo=7, maxBombs=4, bombType="love",
                        bombCooldownTicks=30, bombCooldownTicksOnPlant=60)
    for enemy in busy["enemies"]:
        enemy.update(team="beta", lastMoveIntent="ne", wasLastMoveBlocked=False,
                     shotCooldownTicks=3, bombCooldownTicks=15, bombCooldownTicksOnPlant=30)
    busy["projectiles"] = [{"id": f"p{index}", "ownerTankId": busy["self"]["id"],
        "team": "alpha" if index % 2 else "beta", "kind": "super", "x": 50 + index * 20,
        "y": 200, "vx": -270, "vy": 0, "speed": 270, "radius": 4,
        "bounces": index % 3, "maxBounces": 2} for index in reversed(range(40))]
    busy["bombs"] = [{"id": f"b{index}", "ownerTankId": busy["self"]["id"], "team": "alpha",
        "kind": "love", "x": 50 + index * 40, "y": 200, "radius": 5, "blastRadius": 70,
        "fuseTicksRemaining": 100 + index} for index in reversed(range(18))]
    samples.append(busy)

    result = subprocess.run(
        ["node", CLI_PATH],
        input=json.dumps(samples),
        capture_output=True,
        text=True,
        check=True,
    )
    typescript = np.asarray(json.loads(result.stdout), dtype=np.float32)
    original = json.dumps(samples)
    python = np.asarray([normalize_observation(sample) for sample in samples], dtype=np.float32)
    assert json.dumps(samples) == original, "Encoding must not mutate observations"
    assert np.isfinite(python).all() and ((python >= 0) & (python <= 1)).all()
    max_error = float(np.max(np.abs(typescript - python)))
    assert max_error < 1e-6, max_error
    print(f"Observation parity passed across {len(samples)} live states: max_error={max_error:.2e}")


if __name__ == "__main__":
    main()
