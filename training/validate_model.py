"""
Validate a hybrid-trained PPO checkpoint on the game runtime.

Behavior:
- Auto-detects the most recent run directory under training/output by default.
- Uses treads_ppo_best.zip when available, otherwise treads_ppo_final.zip.
- Evaluates across requested levels and reports overall + per-opponent stats.
- Prints firing/bomb usage rates to catch "never fires" regressions.
"""

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, cast
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(__file__))

from treads_env import MOVE_INTENTS, TreadsEnv  # noqa: E402


# Keep this mapping in sync with src/game/LevelConfig.ts.
LEVEL_TO_OPPONENT_KIND = {
    1: "stationary",
    2: "stationary",
    3: "stationary-random-aim",
    4: "simple-moving",
    5: "stationary-random-aim",
    6: "bomber",
    7: "stationary-random-aim",
    8: "super-bomber",
    9: "super-bomber",
}

MOVE_COUNT = len(MOVE_INTENTS)
FIRE_THRESHOLD = 0.0
BOMB_THRESHOLD = 0.5
MOVE_DEAD_ZONE = 0.33


@dataclass
class EpisodeStats:
    win: int
    reward: float
    steps: int
    fires: int
    bombs: int
    level: int
    opponent_kind: str


ObsDict = Dict[str, Any]
DecodedAction = Dict[str, Any]


def _decode_action(continuous_action: Any, obs_raw: Optional[ObsDict]) -> DecodedAction:
    """Decode 5D continuous action using 2D movement and enemy-relative aim encoding.

    Action layout: [move_x, move_y, aim_signal, fire_signal, bomb_signal]
    Movement: 2D (move_x, move_y) mapped to 9 discrete intents via dead-zone thresholds.
    Aim: aim_signal = 0 → pointed at nearest enemy, ±1 → ±10° offset.
    """
    a = np.asarray(continuous_action, dtype=np.float32).reshape(-1)
    if a.shape[0] < 5:
        aim = float(obs_raw["self"]["aimAngle"]) if obs_raw is not None else 0.0
        return {
            "move": 0,
            "aim_angle": np.array([aim], dtype=np.float32),
            "fire": 0,
            "plant_bomb": 0,
        }

    mx = float(np.clip(a[0], -1.0, 1.0))
    my = float(np.clip(a[1], -1.0, 1.0))
    aim_signal = float(np.clip(a[2], -1.0, 1.0))
    fire_signal = float(np.clip(a[3], -1.0, 1.0))
    bomb_signal = float(np.clip(a[4], -1.0, 1.0))

    # 2D movement decode
    go_e = mx > MOVE_DEAD_ZONE
    go_w = mx < -MOVE_DEAD_ZONE
    go_s = my > MOVE_DEAD_ZONE
    go_n = my < -MOVE_DEAD_ZONE
    if go_n and go_e:
        move_idx = MOVE_INTENTS.index("ne")
    elif go_n and go_w:
        move_idx = MOVE_INTENTS.index("nw")
    elif go_s and go_e:
        move_idx = MOVE_INTENTS.index("se")
    elif go_s and go_w:
        move_idx = MOVE_INTENTS.index("sw")
    elif go_n:
        move_idx = MOVE_INTENTS.index("n")
    elif go_s:
        move_idx = MOVE_INTENTS.index("s")
    elif go_e:
        move_idx = MOVE_INTENTS.index("e")
    elif go_w:
        move_idx = MOVE_INTENTS.index("w")
    else:
        move_idx = 0  # none

    # Enemy-relative aim encoding
    raw_obs: ObsDict = obs_raw or {}
    alive_enemies = [
        e
        for e in cast(List[ObsDict], raw_obs.get("enemies", []))
        if not bool(e.get("destroyed", False))
    ]
    if alive_enemies and obs_raw is not None:
        sx = float(raw_obs["self"]["x"]) + float(raw_obs["self"]["size"]) / 2
        sy = float(raw_obs["self"]["y"]) + float(raw_obs["self"]["size"]) / 2
        nearest = min(
            alive_enemies,
            key=lambda e: (float(e["x"]) + float(e["size"]) / 2 - sx) ** 2
            + (float(e["y"]) + float(e["size"]) / 2 - sy) ** 2,
        )
        ex = float(nearest["x"]) + float(nearest["size"]) / 2
        ey = float(nearest["y"]) + float(nearest["size"]) / 2
        angle_to_enemy = math.atan2(ey - sy, ex - sx)
        aim_angle = angle_to_enemy + aim_signal * (math.pi / 18)
    else:
        aim_angle = float(raw_obs["self"]["aimAngle"]) if obs_raw is not None else 0.0

    return {
        "move": move_idx,
        "aim_angle": np.array([aim_angle], dtype=np.float32),
        "fire": int(fire_signal > FIRE_THRESHOLD),
        "plant_bomb": int(bomb_signal > BOMB_THRESHOLD),
    }


def _find_latest_run_dir(output_dir: str) -> str:
    candidates: List[Tuple[float, str]] = []
    for entry in os.scandir(output_dir):
        if not entry.is_dir():
            continue
        log_path = os.path.join(entry.path, "training_log.csv")
        best_zip = os.path.join(entry.path, "treads_ppo_best.zip")
        final_zip = os.path.join(entry.path, "treads_ppo_final.zip")
        if os.path.isfile(log_path) and (os.path.isfile(best_zip) or os.path.isfile(final_zip)):
            candidates.append((os.path.getmtime(log_path), entry.path))

    if not candidates:
        raise FileNotFoundError(
            "No hybrid run directories with training_log.csv and checkpoint zip found under training/output."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _resolve_model_path(model_path_arg: str) -> str:
    if model_path_arg:
        if not os.path.isfile(model_path_arg):
            raise FileNotFoundError(f"Specified model path does not exist: {model_path_arg}")
        return model_path_arg

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    run_dir = _find_latest_run_dir(output_dir)
    best_zip = os.path.join(run_dir, "treads_ppo_best.zip")
    final_zip = os.path.join(run_dir, "treads_ppo_final.zip")

    if os.path.isfile(best_zip):
        return best_zip
    if os.path.isfile(final_zip):
        return final_zip
    raise FileNotFoundError(f"No best/final checkpoint found in detected run dir: {run_dir}")


def _print_summary(stats: List[EpisodeStats]) -> None:
    if not stats:
        print("No episodes were evaluated.")
        return

    wins = sum(s.win for s in stats)
    total_eps = len(stats)
    total_steps = sum(s.steps for s in stats)
    total_fires = sum(s.fires for s in stats)
    total_bombs = sum(s.bombs for s in stats)
    avg_reward = float(np.mean([s.reward for s in stats]))

    print("\nOverall Summary")
    print(f"  Win rate: {wins}/{total_eps} = {wins / total_eps:.3f}")
    print(f"  Avg reward: {avg_reward:.3f}")
    print(f"  Fire rate: {total_fires}/{total_steps} ticks = {total_fires / max(total_steps, 1):.3f}")
    print(f"  Bomb rate: {total_bombs}/{total_steps} ticks = {total_bombs / max(total_steps, 1):.3f}")

    print("\nPer Opponent Kind")
    by_kind: Dict[str, List[EpisodeStats]] = {}
    for s in stats:
        by_kind.setdefault(s.opponent_kind, []).append(s)
    for kind in sorted(by_kind.keys()):
        group = by_kind[kind]
        kind_wins = sum(g.win for g in group)
        kind_eps = len(group)
        kind_steps = sum(g.steps for g in group)
        kind_fires = sum(g.fires for g in group)
        print(
            f"  {kind}: win={kind_wins}/{kind_eps} ({kind_wins / kind_eps:.3f}), "
            f"fire_rate={kind_fires / max(kind_steps, 1):.3f}"
        )

    print("\nPer Level")
    by_level: Dict[int, List[EpisodeStats]] = {}
    for s in stats:
        by_level.setdefault(s.level, []).append(s)
    for lvl in sorted(by_level.keys()):
        group = by_level[lvl]
        lvl_wins = sum(g.win for g in group)
        lvl_eps = len(group)
        print(f"  L{lvl}: {lvl_wins}/{lvl_eps} ({lvl_wins / lvl_eps:.3f})")


def validate(
    model_path: str,
    levels: List[int],
    episodes_per_level: int,
    max_ticks: int,
    seed_start: int,
    deterministic: bool,
) -> None:
    model = cast(Any, PPO.load(model_path, device="cpu"))  # pyright: ignore[reportUnknownMemberType]
    print(f"Loaded checkpoint: {model_path}")
    print(
        f"Validation config: levels={levels}, episodes_per_level={episodes_per_level}, "
        f"max_ticks={max_ticks}, deterministic={deterministic}"
    )

    stats: List[EpisodeStats] = []
    seed = seed_start

    for level in levels:
        for _ in range(episodes_per_level):
            env = TreadsEnv(level=level, seed_start=seed, max_episode_steps=max_ticks)
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            fires = 0
            bombs = 0
            info: dict[str, object] = {}

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs_raw = cast(Optional[ObsDict], getattr(env, "_last_obs_raw", None))
                decoded = _decode_action(action, obs_raw)

                fires += int(decoded["fire"])
                bombs += int(decoded["plant_bomb"])

                obs, reward, terminated, truncated, info = env.step(decoded)
                total_reward += float(reward)
                steps += 1
                done = terminated or truncated

            result = cast(dict[str, Any], info.get("result", {}))
            win = int(bool(result.get("win", False)))
            kind = LEVEL_TO_OPPONENT_KIND.get(level, "unknown")
            stats.append(
                EpisodeStats(
                    win=win,
                    reward=total_reward,
                    steps=steps,
                    fires=fires,
                    bombs=bombs,
                    level=level,
                    opponent_kind=kind,
                )
            )

            print(
                f"Episode L{level} seed={seed}: steps={steps} win={bool(win)} "
                f"reward={total_reward:.3f} fires={fires} bombs={bombs}"
            )

            env.close()
            seed += 1

    _print_summary(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional explicit checkpoint path (.zip). Defaults to latest run's best/final.",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="1,2,3,4,5,6,7,8,9",
        help="Comma-separated levels to validate.",
    )
    parser.add_argument("--episodes-per-level", type=int, default=20)
    parser.add_argument("--max-ticks", type=int, default=720)
    parser.add_argument("--seed-start", type=int, default=14000)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic policy mean.",
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_path)
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    validate(
        model_path=model_path,
        levels=levels,
        episodes_per_level=args.episodes_per_level,
        max_ticks=args.max_ticks,
        seed_start=args.seed_start,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
