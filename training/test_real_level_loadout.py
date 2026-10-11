"""Real maps in the early curriculum must use the same player loadout as gameplay."""

import json
import tempfile
from pathlib import Path

from train_hybrid import HybridTrainer
from treads_env import TreadsEnv


def main() -> None:
    with tempfile.TemporaryDirectory() as output:
        trainer = HybridTrainer(n_steps=64, batch_size=32, n_epochs=1, num_workers=1,
                                max_episode_steps=64, replay_episode_interval=1, output_dir=output)
        env = TreadsEnv(level=1)
        try:
            assert trainer._get_player_max_ammo() == 2
            assert trainer._get_player_max_bombs() == 0
            trainer.levels = [1]
            trainer.send_weights()
            trainer.collect_rollout(10)
            replay = json.loads(next(Path(output, 'replays').glob('*.json')).read_text())
            player = replay['levelConfig']['player']
            env.reset(seed=42)
            live = env._last_obs_raw['self']
            assert live['maxAmmo'] == player.get('ammo', {}).get('count', 5) == 5
            assert live['maxBombs'] == player.get('bombs', {}).get('count', 2) == 2
        finally:
            trainer.close()
            env.close()
    print('Early-curriculum real maps preserve gameplay ammunition and bombs')


if __name__ == '__main__':
    main()
