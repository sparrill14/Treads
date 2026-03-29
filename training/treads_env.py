"""
Treads Gymnasium Environment.
Wraps the Node.js CLI runner as an OpenAI Gymnasium environment.
"""

import os
import json
import math
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Path to the compiled CLI runner
CLI_RUNNER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "cli-runner.js"
)

# Arena dimensions (fixed in the game)
ARENA_WIDTH = 1000.0
ARENA_HEIGHT = 500.0

# Observation dimensions:
# self: x, y, aimAngle, speed, destroyed, shotCooldown, activeAmmo, maxAmmo (8)
# up to 3 enemies: x, y, aimAngle, speed, destroyed (5 each = 15)
# up to 5 projectiles: x, y, vx, vy, team_is_enemy (5 each = 25)
# up to 3 obstacles: x, y, w, h (4 each = 12)
# Total: 8 + 15 + 25 + 12 = 60
MAX_ENEMIES = 3
MAX_PROJECTILES = 5
MAX_OBSTACLES = 3
SELF_DIM = 8
ENEMY_DIM = 5
PROJ_DIM = 5
OBS_DIM = 4
OBS_SIZE = SELF_DIM + MAX_ENEMIES * ENEMY_DIM + MAX_PROJECTILES * PROJ_DIM + MAX_OBSTACLES * OBS_DIM

# Move intents mapping
MOVE_INTENTS = ["none", "n", "s", "e", "w", "ne", "nw", "se", "sw"]


class TreadsEnv(gym.Env):
    """Gymnasium environment for the Treads tank game."""

    metadata = {"render_modes": []}

    def __init__(self, level=1, seed_start=0, max_episode_steps=1800):
        super().__init__()
        self.level = level
        self.seed_counter = seed_start
        self.max_episode_steps = max_episode_steps
        self.process = None
        self._buffer = ""
        self._init_data = None
        self._last_obs_raw = None
        self._prev_enemy_health_count = 0
        self._prev_distance_to_enemy = float("inf")
        self._step_count = 0

        # Action space: MultiDiscrete([9, 2, 2]) + Box for aim angle
        # 9 moves, fire (0/1), plantBomb (0/1), aim angle [0, 2pi]
        self.action_space = spaces.Dict(
            {
                "move": spaces.Discrete(9),
                "aim_angle": spaces.Box(
                    low=0.0, high=2 * np.pi, shape=(1,), dtype=np.float32
                ),
                "fire": spaces.Discrete(2),
                "plant_bomb": spaces.Discrete(2),
            }
        )

        # Observation space: flat float32 array
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

    def _start_process(self, seed):
        """Spawn the Node.js CLI runner subprocess."""
        if self.process is not None:
            self._kill_process()

        self.process = subprocess.Popen(
            [
                "node",
                CLI_RUNNER_PATH,
                "--level",
                str(self.level),
                "--seed",
                str(seed),
                "--max-ticks",
                str(self.max_episode_steps),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._buffer = ""

    def _kill_process(self):
        """Kill the subprocess."""
        if self.process is not None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception:
                pass
            self.process = None

    def _read_message(self):
        """Read a single JSON line from the subprocess stdout."""
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("CLI runner process terminated unexpectedly")
        return json.loads(line.strip())

    def _send_action(self, action_dict):
        """Send an action as JSON to the subprocess stdin."""
        line = json.dumps(action_dict) + "\n"
        self.process.stdin.write(line)
        self.process.stdin.flush()

    def _normalize_obs(self, obs_raw):
        """Convert raw TankObservation to a flat normalized numpy array."""
        result = np.zeros(OBS_SIZE, dtype=np.float32)
        idx = 0

        # Self state
        self_data = obs_raw["self"]
        result[idx] = self_data["x"] / ARENA_WIDTH
        result[idx + 1] = self_data["y"] / ARENA_HEIGHT
        result[idx + 2] = self_data["aimAngle"] / (2 * math.pi)
        result[idx + 3] = self_data["speed"] / 100.0  # normalize speed
        result[idx + 4] = 1.0 if self_data["destroyed"] else 0.0
        result[idx + 5] = min(self_data["shotCooldownTicks"] / 300.0, 1.0)
        result[idx + 6] = self_data["activeAmmo"] / max(self_data["maxAmmo"], 1)
        result[idx + 7] = self_data["maxAmmo"] / 5.0
        idx += SELF_DIM

        # Enemies (up to MAX_ENEMIES)
        enemies = [e for e in obs_raw["enemies"] if not e["destroyed"]]
        # sort by distance to self
        sx = self_data["x"] + self_data["size"] / 2
        sy = self_data["y"] + self_data["size"] / 2
        enemies.sort(
            key=lambda e: (e["x"] + e["size"] / 2 - sx) ** 2
            + (e["y"] + e["size"] / 2 - sy) ** 2
        )
        for i in range(MAX_ENEMIES):
            if i < len(enemies):
                e = enemies[i]
                result[idx] = e["x"] / ARENA_WIDTH
                result[idx + 1] = e["y"] / ARENA_HEIGHT
                result[idx + 2] = e["aimAngle"] / (2 * math.pi)
                result[idx + 3] = e["speed"] / 100.0
                result[idx + 4] = 0.0  # alive
            # else zeros (no enemy)
            idx += ENEMY_DIM

        # Projectiles (up to MAX_PROJECTILES, sorted by distance to self)
        projectiles = obs_raw.get("projectiles", [])
        projectiles.sort(
            key=lambda p: (p["x"] - sx) ** 2 + (p["y"] - sy) ** 2
        )
        for i in range(MAX_PROJECTILES):
            if i < len(projectiles):
                p = projectiles[i]
                result[idx] = p["x"] / ARENA_WIDTH
                result[idx + 1] = p["y"] / ARENA_HEIGHT
                result[idx + 2] = p["vx"] / 300.0 * 0.5 + 0.5  # normalize to [0,1]
                result[idx + 3] = p["vy"] / 300.0 * 0.5 + 0.5
                result[idx + 4] = 1.0 if p["team"] == "enemy" else 0.0
            idx += PROJ_DIM

        # Obstacles (up to MAX_OBSTACLES, sorted by distance to self)
        obstacles = obs_raw.get("obstacles", [])
        obstacles.sort(
            key=lambda o: (o["x"] + o["width"] / 2 - sx) ** 2
            + (o["y"] + o["height"] / 2 - sy) ** 2
        )
        for i in range(MAX_OBSTACLES):
            if i < len(obstacles):
                o = obstacles[i]
                result[idx] = o["x"] / ARENA_WIDTH
                result[idx + 1] = o["y"] / ARENA_HEIGHT
                result[idx + 2] = o["width"] / ARENA_WIDTH
                result[idx + 3] = o["height"] / ARENA_HEIGHT
            idx += OBS_DIM

        return np.clip(result, 0.0, 1.0)

    def _compute_reward(self, obs_raw, action_dict=None, done_msg=None):
        """Compute step reward."""
        reward = 0.0

        # Small per-tick penalty
        reward -= 0.002

        # Check enemy kills - big reward
        alive_enemies = sum(
            1 for e in obs_raw["enemies"] if not e["destroyed"]
        )
        enemies_killed = self._prev_enemy_health_count - alive_enemies
        if enemies_killed > 0:
            reward += 20.0 * enemies_killed
        self._prev_enemy_health_count = alive_enemies

        # Distance bonus: reward for getting closer to nearest enemy
        enemies = [e for e in obs_raw["enemies"] if not e["destroyed"]]
        if enemies:
            sx = obs_raw["self"]["x"] + obs_raw["self"]["size"] / 2
            sy = obs_raw["self"]["y"] + obs_raw["self"]["size"] / 2
            min_dist = min(
                math.sqrt(
                    (e["x"] + e["size"] / 2 - sx) ** 2
                    + (e["y"] + e["size"] / 2 - sy) ** 2
                )
                for e in enemies
            )
            arena_diag = math.sqrt(ARENA_WIDTH**2 + ARENA_HEIGHT**2)
            proximity_reward = 1.0 - (min_dist / arena_diag)
            reward += 0.03 * proximity_reward

            # Reward for moving closer
            dist_improvement = self._prev_distance_to_enemy - min_dist
            reward += 0.08 * (dist_improvement / arena_diag * 10)
            self._prev_distance_to_enemy = min_dist

            # Aim alignment reward
            nearest = min(
                enemies,
                key=lambda e: (e["x"] + e["size"] / 2 - sx) ** 2
                + (e["y"] + e["size"] / 2 - sy) ** 2,
            )
            ex = nearest["x"] + nearest["size"] / 2
            ey = nearest["y"] + nearest["size"] / 2
            to_enemy_angle = math.atan2(ey - sy, ex - sx)
            aim_angle = obs_raw["self"]["aimAngle"]
            angle_diff = abs(
                math.atan2(
                    math.sin(aim_angle - to_enemy_angle),
                    math.cos(aim_angle - to_enemy_angle),
                )
            )
            # Reward for good aim (0 when perfectly aimed, -pi when opposite)
            aim_reward = 1.0 - (angle_diff / math.pi)
            reward += 0.02 * aim_reward

            # Reward for firing when well-aimed
            if action_dict and action_dict.get("fire"):
                if angle_diff < math.pi / 6:  # within 30°
                    reward += 0.1  # good shot attempt
                else:
                    reward -= 0.02  # wasting ammo

            # Reward for moving (not standing still)
            if action_dict and action_dict.get("move") != "none":
                reward += 0.005

        return reward

    def reset(self, seed=None, options=None):
        """Reset the environment with a new seed."""
        super().reset(seed=seed)
        self.seed_counter += 1
        game_seed = self.seed_counter

        self._start_process(game_seed)
        self._step_count = 0

        # Read init message
        self._init_data = self._read_message()
        assert self._init_data["type"] == "init", f"Expected init, got {self._init_data['type']}"

        # Read first observation
        obs_msg = self._read_message()
        assert obs_msg["type"] == "observation", f"Expected observation, got {obs_msg['type']}"

        self._last_obs_raw = obs_msg["observation"]

        # Init tracking
        enemies = [
            e for e in self._last_obs_raw["enemies"] if not e["destroyed"]
        ]
        self._prev_enemy_health_count = len(enemies)
        if enemies:
            sx = self._last_obs_raw["self"]["x"] + self._last_obs_raw["self"]["size"] / 2
            sy = self._last_obs_raw["self"]["y"] + self._last_obs_raw["self"]["size"] / 2
            self._prev_distance_to_enemy = min(
                math.sqrt(
                    (e["x"] + e["size"] / 2 - sx) ** 2
                    + (e["y"] + e["size"] / 2 - sy) ** 2
                )
                for e in enemies
            )
        else:
            self._prev_distance_to_enemy = 0.0

        return self._normalize_obs(self._last_obs_raw), {}

    def step(self, action):
        """Take a step in the environment."""
        # Convert action to JSON
        if isinstance(action, dict):
            move_idx = int(action["move"])
            aim_angle = float(action["aim_angle"][0])
            fire = bool(action["fire"])
            plant_bomb = bool(action["plant_bomb"])
        else:
            # Handle flat numpy array from SB3 (we'll use a wrapper)
            raise ValueError("Action must be a dict")

        action_dict = {
            "move": MOVE_INTENTS[move_idx],
            "aimAngle": aim_angle,
            "fire": fire,
            "plantBomb": plant_bomb,
        }

        self._send_action(action_dict)
        self._step_count += 1

        # Read next message
        msg = self._read_message()

        if msg["type"] == "result":
            # Match ended
            reward = 0.0
            if msg["win"]:
                reward += 1.0  # win bonus
            elif msg["loss"]:
                reward -= 1.0  # death penalty
            # Use last obs
            obs = self._normalize_obs(self._last_obs_raw)
            self._kill_process()
            return obs, reward, True, False, {"result": msg}

        assert msg["type"] == "observation", f"Expected observation, got {msg['type']}"
        self._last_obs_raw = msg["observation"]

        # Check if player is destroyed
        done = self._last_obs_raw["self"]["destroyed"]

        reward = self._compute_reward(self._last_obs_raw, action_dict=action_dict)
        if done:
            reward -= 1.0  # death penalty

        obs = self._normalize_obs(self._last_obs_raw)

        # Check if we need to read a result message when done
        truncated = False
        if done:
            # Read the result message that follows
            result_msg = self._read_message()
            self._kill_process()
            return obs, reward, True, False, {"result": result_msg}

        return obs, reward, False, truncated, {}

    def close(self):
        """Clean up the subprocess."""
        self._kill_process()


# Number of discrete aim bins
NUM_AIM_BINS = 16


class TreadsEnvDiscrete(gym.Env):
    """
    MultiDiscrete action space wrapper for TreadsEnv to work with SB3.
    Action: MultiDiscrete([9, 16, 2, 2])
      - move: 0-8 (9 directions)
      - aim: 0-15 (16 bins of 22.5° each)
      - fire: 0/1
      - bomb: 0/1
    """

    metadata = {"render_modes": []}

    def __init__(self, level=1, seed_start=0, max_episode_steps=1800):
        super().__init__()
        self._env = TreadsEnv(
            level=level, seed_start=seed_start, max_episode_steps=max_episode_steps
        )

        self.action_space = spaces.MultiDiscrete([9, NUM_AIM_BINS, 2, 2])
        self.observation_space = self._env.observation_space

    def _convert_action(self, action):
        """Convert MultiDiscrete action to dict action."""
        move_idx = int(action[0])
        aim_bin = int(action[1])
        aim_angle = float(aim_bin) / NUM_AIM_BINS * 2 * np.pi
        fire = bool(action[2])
        plant_bomb = bool(action[3])

        return {
            "move": move_idx,
            "aim_angle": np.array([aim_angle], dtype=np.float32),
            "fire": int(fire),
            "plant_bomb": int(plant_bomb),
        }

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        dict_action = self._convert_action(action)
        return self._env.step(dict_action)

    def close(self):
        self._env.close()
