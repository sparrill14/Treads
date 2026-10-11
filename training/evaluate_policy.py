"""Benchmark a checkpoint and heuristic reference on paired, held-out game episodes."""

import argparse
import json
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from model_contract import assert_contract_compatible
from train_hybrid import HybridTrainer, TREADS_PB2


def summarize(episodes: list[dict]) -> dict:
    wins = sum(bool(episode["win"]) for episode in episodes)
    return {
        "episodes": len(episodes),
        "wins": wins,
        "winRate": wins / len(episodes),
        "wilson95Lower": HybridTrainer._wilson_lower_bound(wins, len(episodes)),
        "outcomes": dict(Counter(episode["outcome"] for episode in episodes)),
        "meanTicks": float(np.mean([episode["ticks"] for episode in episodes])),
        "meanShots": float(np.mean([episode["shots"] for episode in episodes])),
        "meanBombs": float(np.mean([episode["bombs"] for episode in episodes])),
        "blockedFraction": sum(episode["blockedSteps"] for episode in episodes) /
                           max(1, sum(episode["ticks"] for episode in episodes)),
        "meanDisplacement": float(np.mean([episode["displacement"] for episode in episodes])),
        "meanPlayerHealth": float(np.mean([episode["playerHealth"] for episode in episodes])),
        "meanEnemiesDestroyed": float(np.mean([episode["enemiesDestroyed"] for episode in episodes])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per level and map variant")
    parser.add_argument("--max-ticks", type=int, default=1800)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2_000_000_000, help="Unseen test seeds, distinct from training validation")
    parser.add_argument("--output", default="")
    parser.add_argument("--gate", action="store_true", help="Exit nonzero when competence thresholds fail")
    parser.add_argument("--min-overall", type=float, default=0.70)
    parser.add_argument("--min-level", type=float, default=0.40)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_ticks < 1 or args.workers < 1:
        parser.error("episodes, max-ticks, and workers must be positive")
    if not 0 <= args.min_overall <= 1 or not 0 <= args.min_level <= 1:
        parser.error("competence thresholds must be between zero and one")
    if not 0 <= args.seed <= 2_147_483_647 - 100_000 - args.episodes:
        parser.error("seed range must fit the RPC signed 32-bit integer")
    args.workers = min(args.workers, 10 * args.episodes)
    assert_contract_compatible(args.model)
    report = {"model": str(Path(args.model).resolve()), "seed": args.seed,
              "maxTicks": args.max_ticks, "episodesPerLevel": args.episodes, "results": {}}

    with tempfile.TemporaryDirectory() as output_dir:
        trainer = HybridTrainer(levels=[1], n_steps=max(64, args.workers), batch_size=32,
                                n_epochs=1, num_workers=args.workers, output_dir=output_dir)
        try:
            trainer.model = PPO.load(args.model, device="cpu")
            assert_contract_compatible(args.model, trainer.model.action_space)
            trainer.send_weights()
            for controller in ("neural", "heuristic"):
                controller_results = {}
                for variant in ("canonical", "randomized"):
                    jobs = [(level, args.seed + level * 10_000 + index)
                            for level in range(1, 11) for index in range(args.episodes)]
                    shards = [jobs[index::args.workers] for index in range(args.workers)]

                    def evaluate(shard_index: int) -> list[dict]:
                        response = trainer.workers[shard_index]["stub"].Evaluate(
                            TREADS_PB2.EvaluateRequest(
                                episodes=[TREADS_PB2.EvaluationEpisode(level=level, seed=seed)
                                          for level, seed in shards[shard_index]],
                                max_ticks=args.max_ticks, tick_norm_ticks=720,
                                procedural_levels=variant == "randomized", difficulty_band=0.25,
                                player_max_ammo=-1, player_max_bombs=-1,
                                canonical_spawns=variant == "canonical",
                                heuristic_player=controller == "heuristic",
                            ), timeout=600,
                        )
                        return json.loads(response.evaluation_json)["episodes"]

                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        episodes = [episode for shard in executor.map(evaluate, range(args.workers)) for episode in shard]
                    assert len(episodes) == len(jobs)
                    expected = Counter(jobs)
                    assert Counter((episode["level"], episode["seed"]) for episode in episodes) == expected
                    summary = summarize(episodes)
                    summary["byLevel"] = {str(level): summarize([episode for episode in episodes if episode["level"] == level])
                                          for level in range(1, 11)}
                    summary["episodeDetails"] = sorted(episodes, key=lambda episode: (episode["level"], episode["seed"]))
                    controller_results[variant] = summary
                    print(f"{controller} / {variant}: {summary['wins']}/{len(episodes)} wins "
                          f"({summary['winRate']:.1%}), Wilson lower={summary['wilson95Lower']:.3f}", flush=True)
                report["results"][controller] = controller_results
        finally:
            trainer.close()

    candidate = report["results"]["neural"]
    report["ready"] = all(result["wilson95Lower"] >= args.min_overall and
                          all(level["winRate"] >= args.min_level for level in result["byLevel"].values())
                          for result in candidate.values())
    report["thresholds"] = {"overallWilson95Lower": args.min_overall, "perLevelWinRate": args.min_level}
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Competence gate: {'PASS' if report['ready'] else 'FAIL'}", flush=True)
    return 1 if args.gate and not report["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
