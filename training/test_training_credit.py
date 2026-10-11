"""Verify delayed combat credit and finite PPO optimization with the current defaults."""

import tempfile

import numpy as np
import torch

from train_hybrid import HybridTrainer
from model_contract import NEUTRAL_AIM_BIN
from runtime_codec import ARENA_DIAGONAL, decode_multi_discrete_action
from treads_env import APPROACH_SCALE, STEP_PENALTY, TreadsEnv


def check_live_shaping(gamma: float, override: bool = False) -> None:
    env = TreadsEnv(level=111, reward_gamma=gamma) if override else TreadsEnv(level=111)
    try:
        env.reset(seed=42)
        previous = 1.0 - env._prev_enemy_dist / ARENA_DIAGONAL
        action = decode_multi_discrete_action([3, NEUTRAL_AIM_BIN, 0, 0], env._last_obs_raw)
        _, reward, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
        current = 1.0 - env._prev_enemy_dist / ARENA_DIAGONAL
        expected = STEP_PENALTY + APPROACH_SCALE * (gamma * current - previous)
        assert np.isclose(reward, expected), "Live environment shaping must use the optimizer's discount"
    finally:
        env.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        trainer = HybridTrainer(levels=[111], n_steps=256, batch_size=64, n_epochs=2,
                                num_workers=1, output_dir=directory)
        try:
            assert trainer.model.clip_range_vf is None
            check_live_shaping(trainer.gamma)
            check_live_shaping(0.997, override=True)
            rewards = np.zeros(181, dtype=np.float32)
            rewards[-1] = 1.0
            advantages, returns = trainer._compute_chunk_gae(
                rewards, np.zeros_like(rewards), np.zeros_like(rewards), 0.0, True)
            expected = (trainer.gamma * trainer.gae_lambda) ** 180
            assert np.isclose(advantages[0], expected)
            assert advantages[0] > 0.1, "A three-second delayed hit must retain meaningful trace credit"
            assert (0.997 * 0.95) ** 180 < 0.0001
            assert trainer.gamma ** 1800 > 0.4, "Thirty-second outcomes need usable discount weight"
            assert np.isfinite(returns).all()

            trainer.send_weights()
            rollout = trainer.collect_rollout(target_episodes=10)
            trainer._populate_buffer(rollout)
            policy = trainer.model.policy
            before = [parameter.detach().clone() for parameter in policy.parameters()]
            trainer.model.train()
            assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
            assert any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))
            assert np.isfinite(trainer.model.rollout_buffer.advantages).all()
            print("Delayed combat credit, unclipped critic, and finite PPO update passed")
        finally:
            trainer.close()


if __name__ == '__main__':
    main()
