# Treads - Architectural TODO

## 1. Separate Update from Render ✅

- [x] Extract all game logic (position updates, aiming, shooting, bomb planting) out of `GameRenderer.render()`
- [x] Create a distinct `update(deltaTime)` phase in the game loop
- [x] Game loop becomes: `gameLoop() → update(deltaTime) → render()`
- **Why:** Enables fixed timestep physics, pause without freezing render, slow-mo effects, replay systems, and debug visualization layers

## 2. Delta-Time Based Movement ✅

- [x] Change all movement from pixels-per-frame to `position += speed * deltaTime`
- [x] Update tank, ammunition, and bomb movement to use delta time
- [x] Remove dependency on frame rate for gameplay feel
- **Why:** Framerate independence — game plays the same regardless of monitor refresh rate or performance

## 3. Centralized Input Manager

- [x] Create an `InputManager` class that owns all DOM event listeners
- [x] Remove listener registration from `PlayerTank` constructor
- [x] Add proper cleanup on level teardown (`stop()` removes listeners)
- **Why:** Prevents memory leaks and ghost input bugs from stacking listeners on level switch

## 4. Data-Driven Levels ✅

- [x] Define a `LevelConfig` interface for obstacle, enemy, and player placement
- [x] Replace individual `Level1`–`Level9` classes with a single `Level` class that reads config
- [x] Store level configs as data (JSON or typed objects)
- **Why:** Scalability — adding levels requires zero new classes; opens door to level editor and procedural generation

## 5. Centralized Collision System ✅

- [x] Create a collision manager that runs once per frame
- [x] Consolidate scattered collision checks (ammunition, bombs, obstacles) into one system
- [x] Make it easy to add new collidable entity types (power-ups, hazards)
- **Why:** Eliminates duplicate checks, single place to optimize (spatial partitioning) as entity counts grow

## 6. Neural Net Controller ✅

- [x] Audit observation pipeline (Python ↔ TypeScript normalization verified identical)
- [x] Audit action pipeline (ONNX logit decoding verified identical)
- [x] Build diagnostic overlay (F2 toggle: move arrows, aim lines, fire/bomb indicators)
- [x] Increase aim bins from 16 → 32 (11.25° precision, enables hitting from ~300px)
- [x] Improve reward shaping (bullet-approach reward, boosted fire/kill/win rewards)
- [x] Train PPO agent (500K timesteps, MlpPolicy [256,256])

## 7. Replay & Training Tools ✅

- [x] Training replay capture (--save-replay flag, periodic saving every 50K steps)
- [x] Replay viewer page (file picker, play/pause, speed control, scrub bar, seek)
- [x] Training progress dashboard (d3 charts: reward, win rate, episode length)

## 8. Tank Unification ✅

- [x] Unified TankStateView interface — no separate PlayerTank/EnemyTank classes
- [x] All controllers use TankObservation/TankAction system
- [x] Legacy tank files removed
