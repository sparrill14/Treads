# PPO training

The supported pipeline is `train_hybrid.py`, which runs simulation and inference in TypeScript and PPO updates in Python.

Create the environment and install dependencies from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r training\requirements.txt
npm ci
```

Run every preflight gate before starting a long job.

```powershell
npm test
npm run lint
npm run build:prod
npm run verify:training
npm run test:training
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

The current v4 observation contract includes obstacle-aware navigation, combat resources, cooldowns, rule settings, enemy movement state, and projectile bounce budgets.
Legacy checkpoints, including v3 checkpoints, use incompatible observations and are intentionally rejected by the current runtime, so the first v4 run must start from scratch.

## Review and corrected behavior, October 2026

The deterministic TypeScript simulation is a sound foundation and remains the single game engine used by gameplay, rollouts, evaluation, and replays.
The review found concrete errors in how controllers and training used that engine.
Those errors needed correction before spending more compute on PPO.

| Area | Reproduced failure | Correction |
| --- | --- | --- |
| Rollout boundaries | Four short collection requests restarted at tick zero and discarded unfinished matches. | Keep the simulation, reward trackers, sampling RNG, and replay alive across weight updates and evaluation. |
| Browser inference | Asynchronous ONNX requests applied delayed actions to newer observations. | Browser and worker share the synchronous `PolicyMLP` actor and TypeScript observation encoder. |
| Advanced navigation | Level 10 repeatedly selected movement into a wall despite an available A* route. | Follow obstacle-aware guidance before scoring local combat movement. |
| Bomb tactics | An advanced bomber rejected every bomb because the proposed blast overlapped its owner. | Check a feasible escape route before planting and evade blasts from every team, including its own. |
| Shooting | Predicted hits used the requested aim instead of the turret angle reachable that tick. | Validate the actual firing angle, lead moving targets, and honor the match's bounce rules. |
| Ricochets | A super round fired beside a wall bounced twice in one tick and disappeared. | Apply changed velocity to remaining projectile substeps. |
| Draws and time limits | A mutual kill could continue or be treated as a timeout. | End draws immediately as true terminals; only external time limits bootstrap the critic. |
| Approach reward | Switching targets after a kill silently restarted the distance potential. | Compute one nearest-live-target state potential across every transition, including kills and terminal states. |
| Scenario coverage | Sampling and evaluation omitted level 10. | Cover all ten built-in maps. |
| Episode horizon | Twelve seconds was too short for slow, three-hit opponents and delayed bombs. | Default to 1,800 ticks, or 30 seconds, while preserving the v3 observation tick scale of 720. |
| Evaluation fairness | Validation changed seeds and difficulty and used curriculum ammunition limits. | Use fixed validation seeds, fixed real-map difficulty, and the actual level's player resources. |
| Real-map training resources | Early curriculum sampling gave real levels two ammunition slots and no bombs instead of the playable loadout. | Preserve actual resources for every real-map rollout; keep lesson resource limits on synthetic scenarios. |
| Human controls | Typing in another panel could move the tank and arrow keys could scroll the page. | Require arena focus, prevent movement-key scrolling, clear held input on blur, and provide controls and restart. |
| Build and verification | Browser builds copied about 82 MB of unused ONNX WASM and Python commands assumed Windows. | Remove the unused runtime, add portable commands, and run game and training checks in CI. |
| Local dashboard | A `run=../<directory>` HTTP request could read metrics outside training output. | Reject path components in run IDs and bind the dashboard and development server to loopback. |
| Dashboard scaling | Live polls reparsed every saved run's full history and delayed model loading as the archive grew. | Read only the latest metric for run summaries and resolve explicitly selected exports without scanning other runs. |

`npm run test:training` builds the worker and runs the supported integration suite, including large gRPC payloads, observation and sampler parity, rollout continuity, mutual kills, evaluation, and checkpoint resume.
Historical trainers and manual diagnostics remain in `training/`, but `train_hybrid.py` and `evaluate_policy.py` are the supported training and readiness commands.
The export test performs a real PPO optimizer update, saves the checkpoint, exports its artifacts, serves them over HTTP, and compares browser-controller actions and complete simulation states across levels 1, 7, and 10.
It also compares real gameplay logits and deterministic actions with Python and rejects malformed weights.
The game tests require advanced tanks to defeat an idle player on level 10 for three seeds and require a bomb-only advanced tank to kill a nearby opponent while retaining full health.

Start a fresh run after these changes.
The follow-up optimization introduces the v4 observation layout, so old v3 checkpoints cannot be resumed or loaded in the browser.
Re-export any checkpoint before trying it in the browser.
An export now produces `.onnx`, `.contract.json`, and `.weights.json` companions.
The dashboard requires all companions; a static deployment loads validated companions from `models/treads_policy.*` when they are included in the build.
An available export establishes that a model can be loaded, not that it is competent.

## Measure competence before promoting a policy

Use separate training seeds, fixed validation seeds for checkpoint selection, and unseen test seeds for the final readiness decision.
These practices follow the independent evaluation and multi-run guidance in [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
The terminal and timeout behavior follows [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

```powershell
node training/run-python.mjs training/evaluate_policy.py `
  training/output/run_name/treads_ppo_best.zip `
  --episodes 30 --workers 4 --gate `
  --output training/output/run_name/readiness.json
```

The evaluator uses the actual player loadout and runs the candidate and advanced heuristic reference on paired seeds across all ten levels.
It tests both exact game spawns and randomized spawns with procedural difficulty fixed at 0.25.
The JSON report includes per-episode evidence and each level's wins, outcomes, shots, bombs, blocked movement, remaining health, enemies destroyed, and maximum displacement from spawn.
By default, `--gate` requires a 95% Wilson lower bound of at least 70% overall and at least 40% wins on every level, separately for both map variants.
These are explicit initial acceptance targets and can be tightened with `--min-overall` and `--min-level`.
The report-only mode prints failures but exits successfully; use `--gate` when a promotion process must fail on weak policies.
The gate does not automatically promote an export or establish that combat feels fair to a human.

The existing `v21_navigation_v3_20260903/treads_ppo_best_train.zip` checkpoint failed the October review benchmark with zero wins across 80 held-out episodes.
The sample used four episodes per level per variant and a 30-second horizon.
The advanced heuristic player reference won 39 of the same 80 matches, including 20 of 40 on exact game maps and 19 of 40 on randomized variants.
On exact game maps, the neural checkpoint averaged only 0.45 shots per match and was blocked on 36.5% of simulated ticks, compared with 17.0 shots and 3.2% blocked ticks for the heuristic reference.
This exposes concrete combat and navigation weaknesses despite successful model loading and inference parity.
That is a diagnostic sample, not a precise estimate of a strong policy's win rate.
No current neural policy should be described as formidable on that evidence.

## Follow-up optimization and the next run

Advanced tanks now predict the first tank that absorbs a projectile, rather than independently tracing hypothetical hits on every tank.
They can fire when an opponent intercepts a round before an ally, while still rejecting shots intercepted by allies.
Their dodge search advances both the tank and ricocheting projectiles over up to 120 ticks, checks bomb fuse timing, and preserves combat progress among safe choices.
Projectile forecasts are shared across dodge and positioning candidates within each decision.
Gameplay regressions reproduce and verify both an ally behind a target and an avoidable incoming round.

On 80 paired diagnostic episodes with seeds beginning at 1,100,000,000, the advanced player reference improved from 37 to 53 wins.
Exact-map wins improved from 19/40 to 26/40, and randomized-map wins improved from 18/40 to 27/40.
These seeds are for tactical development; the final policy test namespace beginning at 2,000,000,000 remains separate.
This sample establishes useful improvements in those matches, not universal superiority or human difficulty balance.

The v4 encoder has 462 features, with capacity for six enemies, 32 projectiles, 16 bombs, and five obstacles.
It includes bomb cooldown, absolute player ammunition and bomb capacities, weapon types, bounce and damage rules, enemy movement components and cooldowns, projectile ownership and remaining bounces, and overflow fractions.
Team features compare with the observing tank's team, so custom team names work correctly.
All bombs contribute to the nearest-hazard summary, including the player's own bombs.
Python encoding no longer sorts the source observation in place.
The tick scale is recorded in the contract at 1,800 ticks and shared by browser, worker, evaluation, and Python encoding.

Fresh PPO defaults use `gamma=0.9995` and `gae_lambda=0.99` to retain credit for delayed combat effects.
A reward delayed by 180 ticks retains about 15% of its GAE trace weight, compared with about 0.006% under the previous defaults.
A 30-second outcome retains about 41% of its discount weight rather than about 0.45%.
These are mathematical credit weights, not measured learning gains.
Use `--gamma` and `--gae-lambda` for controlled comparisons.
The standalone Gym environments default to the same shaping discount; pass `reward_gamma` equal to a custom optimizer discount when using them directly.
Critic clipping is disabled by default and can be enabled explicitly with `--value-clip`; it is independent of policy clipping because its effect depends on reward scale, as described in [SB3's PPO parameters](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
PPO uses at most four CPU threads by default, adjustable with `--torch-threads`, and records this setting with critic clipping in the run manifest.
Batch size and PPO epochs are configurable with `--batch-size` and `--epochs`.

Use the bounded preflight to compare worker counts, verify finite returns and optimization, and measure complete update throughput on a representative scenario mix.
It uses temporary models and does not promote its policies.
On this machine, a clean three-update comparison measured about 918 complete training steps per second with eight workers, versus 717 with four, using four PPO threads and 8,192-step updates.
Both configurations produced finite returns and parameters, and the eight-worker result was about 28% faster.
This is a short throughput measurement on a representative mix, not a competence result or a guarantee of full-curriculum speed.

```powershell
node training/run-python.mjs training/preflight_training.py `
  --workers 4,8 --updates 3 `
  --output training/output/preflight_v4.json
```

The proposed first experiment is a fresh seed-42 pilot capped at two million transitions, with eight rollout workers and four PPO threads.
Leave the full curriculum enabled, evaluate all ten levels every 500 completed episodes, and save replays frequently enough to inspect movement, firing, self-damage, and bomb escape.
Replay intervals use each worker's continuous episode count, so parallel jumps in the global episode total cannot skip saves.
Treat it as a learning check rather than a promotion attempt.
Budget roughly 45 to 60 minutes for the pilot, including validation and checkpoint overhead, based on the local preflight.
Require increasing held-out lesson competence and inspect per-level combat behavior before extending it.

```powershell
npm run train:supervised -- `
  --output-dir training/output/v4_pilot_seed42 `
  --seed 42 --timesteps 2000000 --target-episodes 100000 `
  --num-workers 8 --torch-threads 4 --n-steps 8192 `
  --eval-interval 500 --eval-episodes 12 --replay-interval 100
```

If the pilot learns reliably, compare seeds 43 and 44 and a matched run using the previous discount and trace settings before selecting the longer experiment.
Continue a selected compatible v4 checkpoint through `--load-model`, preserving its trainer sidecar and output directory.
Select checkpoints with validation data, then apply the final unseen-seed readiness gate to the selected candidates.

## Remaining design limits and next experiments

The corrected pipeline is suitable for controlled training experiments, but a formidable neural tank remains an empirical result that must be demonstrated.
Run independent jobs with `--seed 42`, `--seed 43`, and `--seed 44`, track per-level validation, and inspect replays of both successes and failures.
Use the final readiness evaluator only after selecting a checkpoint with validation data, so repeated tuning does not consume the test set.
Keep the weak current checkpoint as a diagnostic baseline rather than continuing a long run blindly.

The v4 feedforward observation exposes current movement and combat state but remains partially observable with respect to controller memory and future changes in intent.
Movement components derive from the previous movement intent and blocked flag rather than a history of measured positions.
It also truncates entity lists beyond its caps and cannot represent arbitrary large editor maps completely.
Overflow features expose truncation, but do not reconstruct omitted hazards or teammates.
Before broadening training beyond the ten-level benchmark, assess temporal context or a recurrent policy and richer teammate or variable-size entity representations.
Do not append features silently to a compatible checkpoint.

Tanks currently pass through each other, and friendly projectiles and bombs can damage their owner and teammates.
Those rules are shared between gameplay and training and are not parity bugs.
Changing body collision, blast obstruction, or friendly fire is a game design change that should precede a new long training run and include map reachability and replay checks.
The advanced heuristics account for friendly hazards and first projectile interception.
Their movement forecasts still assume constant current intent and cannot anticipate every future turn or opponent decision.

Checkpoint resume restores the policy, optimizer, curriculum, metrics, and reward normalization, but does not serialize live worker simulations or every RNG state.
It is recovery from an episode boundary, not bit-for-bit continuation.
Worker failures restart all workers together and discard the failed collection, preventing mixed episode histories from entering one PPO batch.

The dependency review found no production npm vulnerabilities.
Development dependencies still inherit the unpatched `braces` stack-exhaustion advisory through the development server's proxy dependency.
The [upstream advisory](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm) lists no patched version; avoid the destructive downgrade suggested by `npm audit fix --force` and revisit when upstream publishes a fix.
