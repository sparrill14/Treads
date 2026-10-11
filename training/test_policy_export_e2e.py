"""Export SB3, serve real artifacts, and compare browser ticks and logits with Python."""

import functools
import json
import shutil
import subprocess
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from export_onnx import export_to_onnx
from model_contract import write_contract_for_model
from train_hybrid import HybridTrainer


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        trainer = HybridTrainer(levels=[111], n_steps=128, batch_size=32, n_epochs=1,
                                num_workers=1, output_dir=directory, replay_episode_interval=0)
        try:
            # Exercise a real optimizer update before checkpoint serialization and export.
            initial_weights = trainer.model.policy.action_net.weight.detach().clone()
            trainer.send_weights()
            rollout = trainer.collect_rollout(10_000)
            trainer.populate_buffer(rollout)
            trainer.model.num_timesteps = 128
            trainer.model._current_progress_remaining = 1.0
            trainer.model.train()
            assert not torch.equal(initial_weights, trainer.model.policy.action_net.weight)
            assert all(torch.isfinite(parameter).all() for parameter in trainer.model.policy.parameters())
            checkpoint = str(output / 'policy.zip')
            trainer.model.save(checkpoint)
            write_contract_for_model(checkpoint)
            export_to_onnx(checkpoint, str(output / 'policy.onnx'))
            shutil.copyfile(output / 'policy.contract.json', output / 'invalid.contract.json')
            (output / 'invalid.weights.json').write_text('{}', encoding='utf-8')
            server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                subprocess.run(['node', '.training-dist/training/browser-policy-parity-cli.js',
                                f'http://127.0.0.1:{server.server_port}/policy.onnx',
                                str(output / 'policy.weights.json'), str(output / 'trace.json')], check=True)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
            trace = json.loads((output / 'trace.json').read_text())
            assert trace['steps'] >= 100
            samples = trace['samples']
            observations = torch.tensor([sample['obs'] for sample in samples], dtype=torch.float32)
            with torch.no_grad():
                distribution = trainer.model.policy.get_distribution(observations)
                expected_actions = distribution.get_actions(deterministic=True).numpy()
                features = trainer.model.policy.extract_features(observations)
                latent, _ = trainer.model.policy.mlp_extractor(features)
                logits = trainer.model.policy.action_net(latent).numpy()
            np.testing.assert_allclose([sample['logits'] for sample in samples], logits, rtol=1e-4, atol=1e-5)
            np.testing.assert_array_equal([sample['indices'] for sample in samples], expected_actions)
            print(f'PPO update, checkpoint export, ONNX, and browser action parity passed for {len(samples)} real observations')
        finally:
            trainer.close()


if __name__ == '__main__':
    main()
