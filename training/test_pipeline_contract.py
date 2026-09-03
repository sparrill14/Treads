"""Focused regressions for model encoding, action decoding, and confidence gates."""

import math
import os
import sys
import json
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model_contract import ACTION_HEAD_SIZES, NEUTRAL_AIM_BIN, OBS_SIZE, write_contract_for_model
from runtime_codec import decode_multi_discrete_action, normalize_observation
from supervise_training import _option_value, _read_trainer_status, _without_option, find_latest_checkpoint
from train_hybrid import DEFAULT_CURRICULUM, HybridTrainer


def observation(aim_angle: float = 0.0, max_bombs: int = 0) -> dict[str, object]:
    tank_common = {
        "size": 30,
        "speed": 50,
        "health": 3,
        "maxHealth": 3,
        "destroyed": False,
        "activeAmmo": 0,
        "maxAmmo": 2,
        "shotCooldownTicks": 0,
        "shotCooldownTicksOnFire": 10,
        "activeBombs": 0,
        "maxBombs": max_bombs,
        "bombType": "basic" if max_bombs else None,
        "wasLastMoveBlocked": False,
        "invulnerabilityTicksRemaining": 0,
        "lastMoveIntent": "none",
        "ammoType": "basic",
    }
    return {
        "tick": 0,
        "self": {**tank_common, "id": "player-0", "team": "player", "x": 200, "y": 180, "aimAngle": aim_angle},
        "enemies": [{**tank_common, "id": "enemy-0", "team": "enemy", "x": 700, "y": 320, "aimAngle": 0.0}],
        "projectiles": [],
        "bombs": [],
        "obstacles": [],
    }


def main() -> None:
    upper = normalize_observation(observation(math.pi / 2.0))
    lower = normalize_observation(observation(3.0 * math.pi / 2.0))
    assert upper.shape == (OBS_SIZE,)
    assert not np.allclose(upper[2:4], lower[2:4])

    action = decode_multi_discrete_action(
        np.asarray([0, NEUTRAL_AIM_BIN, 1, 1]),
        observation(max_bombs=0),
    )
    expected = math.atan2(140.0, 500.0)
    assert abs(float(action["aim_angle"][0]) - expected) < 1e-6
    assert int(action["plant_bomb"]) == 0
    assert ACTION_HEAD_SIZES == (9, 65, 2, 2)

    assert HybridTrainer._wilson_lower_bound(12, 12) > 0.70
    assert HybridTrainer._wilson_lower_bound(2, 4) < 0.20

    curriculum_probe = HybridTrainer.__new__(HybridTrainer)
    curriculum_probe._explicit_levels = False
    curriculum_probe.curriculum = DEFAULT_CURRICULUM
    curriculum_probe.current_phase_index = 0
    curriculum_probe.scenario_competence = {
        111: {"episodes": 100.0, "wins": 90.0, "ema": 0.9},
        112: {"episodes": 100.0, "wins": 10.0, "ema": 0.1},
    }
    pool = curriculum_probe._build_mixed_levels()
    assert all(level in pool for level in range(1, 10))
    assert pool.count(112) > pool.count(111)

    assert _without_option(
        ["trainer.py", "--load-model=old.zip", "--target-episodes", "10"],
        "--load-model",
    ) == ["trainer.py", "--target-episodes", "10"]
    assert _option_value(["--load-model=old.zip"], "--load-model") == "old.zip"
    with tempfile.TemporaryDirectory() as output_dir:
        for episodes in (100, 250):
            model_path = os.path.join(output_dir, f"treads_ppo_ep{episodes}.zip")
            with open(model_path, "wb") as handle:
                handle.write(b"checkpoint")
            write_contract_for_model(model_path)
            with open(model_path.removesuffix(".zip") + ".trainer.json", "w", encoding="utf-8") as handle:
                json.dump({"schemaVersion": 3, "totalEpisodes": episodes}, handle)
        latest = find_latest_checkpoint(output_dir)
        assert latest is not None
        assert latest[0].endswith("treads_ppo_ep250.zip")
        assert latest[1] == 250
        incompatible_path = os.path.join(output_dir, "treads_ppo_ep500.zip")
        with open(incompatible_path, "wb") as handle:
            handle.write(b"checkpoint")
        incompatible_contract_path = write_contract_for_model(incompatible_path)
        with open(incompatible_contract_path, "r", encoding="utf-8") as handle:
            incompatible_contract = json.load(handle)
        incompatible_contract["observation"]["size"] -= 1
        with open(incompatible_contract_path, "w", encoding="utf-8") as handle:
            json.dump(incompatible_contract, handle)
        with open(
            incompatible_path.removesuffix(".zip") + ".trainer.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump({"schemaVersion": 3, "totalEpisodes": 500}, handle)
        assert find_latest_checkpoint(output_dir) == latest
        with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"status": "budget_exhausted"}, handle)
        assert _read_trainer_status(output_dir) == "budget_exhausted"
    print("Pipeline contract tests passed")


if __name__ == "__main__":
    main()
