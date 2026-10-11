"""Run bounded PPO updates and measure worker configurations before a long experiment."""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from model_contract import MODEL_CONTRACT
from train_hybrid import HybridTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', default='4,8')
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    workers = [int(value) for value in args.workers.split(',')]
    if not workers or min(workers) < 1 or min(args.updates, args.steps, args.epochs, args.torch_threads) < 1:
        parser.error('all sizes and counts must be positive')
    if args.steps < max(workers) or args.steps % 512:
        parser.error('steps must cover every worker and be divisible by the 512-sample batch size')
    report = {'contract': MODEL_CONTRACT, 'levels': [111, 121, 141, 171, 1, 7, 10], 'results': []}
    for count in workers:
        with tempfile.TemporaryDirectory() as directory:
            trainer = HybridTrainer(levels=report['levels'], n_steps=args.steps, batch_size=512,
                                    n_epochs=args.epochs, num_workers=count, output_dir=directory,
                                    torch_threads=args.torch_threads, replay_episode_interval=0)
            totals = dict(weights=0.0, rollout=0.0, buffer=0.0, ppo=0.0)
            completed = 0
            try:
                for _update in range(args.updates):
                    started = time.perf_counter()
                    trainer.send_weights()
                    totals['weights'] += time.perf_counter() - started
                    started = time.perf_counter()
                    rollout = trainer.collect_rollout(target_episodes=100_000)
                    totals['rollout'] += time.perf_counter() - started
                    completed += len(rollout['episode_rewards'])
                    started = time.perf_counter()
                    trainer._populate_buffer(rollout)
                    totals['buffer'] += time.perf_counter() - started
                    if not np.isfinite(trainer.model.rollout_buffer.returns).all():
                        raise RuntimeError('Non-finite PPO returns')
                    started = time.perf_counter()
                    trainer.model.train()
                    totals['ppo'] += time.perf_counter() - started
                    if not all(torch.isfinite(parameter).all() for parameter in trainer.model.policy.parameters()):
                        raise RuntimeError('Non-finite PPO parameters')
                steps = args.steps * args.updates
                result = {'workers': count, 'torchThreads': trainer.torch_threads, 'steps': steps,
                          'updates': args.updates, 'epochs': args.epochs, 'completedEpisodes': completed,
                          'seconds': totals, 'stepsPerSecond': steps / sum(totals.values()),
                          'rolloutStepsPerSecond': steps / totals['rollout'],
                          'ppo': {key: float(value) for key, value in trainer.model.logger.name_to_value.items()
                                  if key in ('train/approx_kl', 'train/clip_fraction', 'train/value_loss',
                                             'train/entropy_loss', 'train/explained_variance')}}
                report['results'].append(result)
                print(json.dumps(result), flush=True)
            finally:
                trainer.close()
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
