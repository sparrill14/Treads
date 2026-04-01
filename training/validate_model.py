"""
Validate a hybrid-trained PPO checkpoint on the game runtime.

Behavior:
- Auto-detects the most recent run directory under training/output by default.
- Uses treads_ppo_best.zip when available, otherwise treads_ppo_final.zip.
- Evaluates across requested levels and reports overall + per-opponent stats.
- Prints firing/bomb usage rates to catch "never fires" regressions.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, cast
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(__file__))

from sb3_compat import ensure_pickle_compat  # noqa: E402
from runtime_codec import OBS_SIZE, decode_continuous_action  # noqa: E402
from treads_env import TreadsEnv  # noqa: E402

ensure_pickle_compat()


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
    model_obs_shape = tuple(model.observation_space.shape or ())
    if model_obs_shape != (OBS_SIZE,):
        raise ValueError(
            "Checkpoint observation shape does not match the current runtime contract: "
            f"model={model_obs_shape}, expected={(OBS_SIZE,)}. "
            "Export or validate a checkpoint trained with the current 197-feature observation layout."
        )
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
            try:
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
                    decoded = decode_continuous_action(action, obs_raw)

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
            finally:
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
