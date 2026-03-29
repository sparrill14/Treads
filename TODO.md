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

- [ ] Create an `InputManager` class that owns all DOM event listeners
- [ ] Remove listener registration from `PlayerTank` constructor
- [ ] Add proper cleanup on level teardown (`stop()` removes listeners)
- **Why:** Prevents memory leaks and ghost input bugs from stacking listeners on level switch

## 4. Data-Driven Levels

- [ ] Define a `LevelConfig` interface for obstacle, enemy, and player placement
- [ ] Replace individual `Level1`–`Level9` classes with a single `Level` class that reads config
- [ ] Store level configs as data (JSON or typed objects)
- **Why:** Scalability — adding levels requires zero new classes; opens door to level editor and procedural generation

## 5. Centralized Collision System

- [ ] Create a collision manager that runs once per frame
- [ ] Consolidate scattered collision checks (ammunition, bombs, obstacles) into one system
- [ ] Make it easy to add new collidable entity types (power-ups, hazards)
- **Why:** Eliminates duplicate checks, single place to optimize (spatial partitioning) as entity counts grow
