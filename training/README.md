# PPO training

The supported pipeline is `train_hybrid.py`, which runs simulation and inference in TypeScript and PPO updates in Python.

Create the environment and install dependencies from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r training\requirements.txt
npm install
```

Run every preflight gate before starting a long job.

```powershell
npm test
npm run lint
npm run build:prod
npm run verify:training
npm run test:training
.venv\Scripts\python.exe training\test_grpc.py
```

Start a new curriculum run from scratch through the supervisor.
The supervisor records the child process, forwards shutdown signals, and resumes from the latest compatible checkpoint after an unexpected trainer exit.

```powershell
npm run train:supervised -- --output-dir training\output\run_name
```

The default run targets 100,000 completed episodes and samples all real game levels from the beginning.

Every checkpoint has a `.contract.json` model-contract sidecar and a `.trainer.json` state sidecar.
Resume requires both sidecars so curriculum, schedules, competence estimates, and normalization state cannot silently reset.

```powershell
npm run train:supervised -- `
  --load-model training\output\run_name\treads_ppo_ep10000.zip `
  --output-dir training\output\run_name
```

The current v3 observation contract adds obstacle-aware path direction, path distance, and reachability features.
Legacy checkpoints use incompatible observations and actions and are intentionally rejected by the current runtime, so the first v3 run must start from scratch.
