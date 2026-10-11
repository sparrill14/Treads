"""Exercise real RPC chunks, weight updates, evaluations, and full-episode replays."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import grpc

from train_hybrid import HybridTrainer, TREADS_PB2


def main() -> None:
    with tempfile.TemporaryDirectory() as output_dir:
        trainer = HybridTrainer(levels=[111], n_steps=64, batch_size=32, n_epochs=1,
                                max_episode_steps=192, num_workers=1, output_dir=output_dir,
                                replay_episode_interval=1)
        try:
            # A stationary, unarmed actor keeps the match running until its time limit.
            with torch.no_grad():
                trainer.model.policy.action_net.weight.zero_()
                trainer.model.policy.action_net.bias.fill_(-100)
                for index in (0, 9 + 32, 9 + 65, 9 + 65 + 2):
                    trainer.model.policy.action_net.bias[index] = 100
            trainer.send_weights()
            first = trainer.collect_rollout(10)
            assert first["episode_starts"][0] == 1
            assert not first["episode_rewards"]
            assert round(first["obs"][-1][8] * 720) == 63

            # Evaluation and publishing updated weights must not reset the live match.
            trainer._evaluate_levels([111], 1, 1_500_000_000)
            try:
                trainer.workers[0]['stub'].Evaluate(TREADS_PB2.EvaluateRequest(
                    episodes=[TREADS_PB2.EvaluationEpisode(level=111, seed=42)],
                    tick_norm_ticks=192), timeout=5)
            except grpc.RpcError as error:
                assert error.code() == grpc.StatusCode.INVALID_ARGUMENT
            else:
                raise AssertionError('An evaluation must not change the trained observation scale')
            trainer.send_weights()
            second = trainer.collect_rollout(10)
            assert second["episode_starts"][0] == 0
            assert round(second["obs"][0][8] * 720) == 64
            trainer.populate_buffer(second)
            assert np.isfinite(trainer.model.rollout_buffer.advantages).all()

            third = trainer.collect_rollout(10)
            assert round(third["obs"][0][8] * 720) == 128
            assert third["episode_lengths"] == [192]
            assert third["worker_rollouts"][0]["last_done"] is True
            fourth = trainer.collect_rollout(10)
            assert fourth["episode_starts"][0] == 1
            assert fourth["obs"][0][8] == 0
            replays = list(Path(output_dir, "replays").glob("*.json"))
            assert len(replays) == 1
            replay = json.loads(replays[0].read_text())
            assert len(replay["ticks"]) == 192
            assert [tick["tick"] for tick in replay["ticks"]] == list(range(192))
        finally:
            trainer.close()
    print("Rollout continuity passed across updates, evaluation, timeout, reset, and replay")


if __name__ == "__main__":
    main()
