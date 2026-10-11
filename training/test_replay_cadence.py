"""Parallel workers must not skip replay intervals when global episode counts jump."""

import json
import tempfile
from pathlib import Path

from train_hybrid import HybridTrainer


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        trainer = HybridTrainer(levels=[111], n_steps=2, batch_size=2, n_epochs=1,
                                num_workers=1, max_episode_steps=1,
                                replay_episode_interval=5, output_dir=directory)
        try:
            trainer.send_weights()
            # Other workers finish episodes between requests to this worker.
            for global_offset in (0, 10, 20):
                trainer.total_episodes = global_offset
                rollout = trainer.collect_rollout(target_episodes=1000)
                assert rollout['episode_lengths'] == [1, 1]
            replays = list(Path(directory, 'replays').glob('*.json'))
            assert len(replays) == 1, 'Six local episodes must cross the five-episode replay interval'
            replay = json.loads(replays[0].read_text())
            assert len(replay['ticks']) == 1
        finally:
            trainer.close()
    print('Replay cadence survives parallel episode-count jumps')


if __name__ == '__main__':
    main()
