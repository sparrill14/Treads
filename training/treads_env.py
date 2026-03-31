"""
Treads Gymnasium Environment.
Wraps the Node.js CLI runner as an OpenAI Gymnasium environment.
"""

import os
import json
import math
import subprocess
from typing import Any, Dict, List, Optional, Tuple, cast
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces

# Path to the compiled CLI runner
CLI_RUNNER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "cli-runner.js"
)

# Arena dimensions (fixed in the game)
ARENA_WIDTH = 1000.0
ARENA_HEIGHT = 500.0

# Observation dimensions (Fix 2+3: expanded caps, bombs, summary features):
# self: x, y, aimAngle, speed, destroyed, wasLastMoveBlocked, invulnerability,
#       tickProgress, health_norm, angleToEnemy, distToEnemy, aimError (12)
# up to 6 enemies: x, y, aimAngle, speed, hasBombs, health_norm (6 each = 36)
# up to 10 projectiles: x, y, vx, vy, team_is_enemy (5 each = 50)
# up to 5 obstacles: x, y, w, h (4 each = 20)
# up to 6 bombs: x, y, fuse_norm, blast_norm, team_is_enemy (5 each = 30)
# 6 summary: enemy_count, enemy_farthest_dist, proj_count, bomb_count,
#            closest_enemy_bomb_dist, farthest_proj_dist
# Total: 12 + 36 + 50 + 20 + 30 + 6 = 154
MAX_ENEMIES = 6       # Level 7 has 5 enemies; +1 buffer
MAX_PROJECTILES = 10  # Level 8: 3×3=9 super shots; +1 buffer
MAX_OBSTACLES = 5     # Level 7 has 4 obstacles; +1 buffer
MAX_BOMBS = 6         # Level 6: 9 theoretical; cap at 6 live
SELF_DIM = 12
ENEMY_DIM = 6
PROJ_DIM = 5
OBS_DIM = 4
BOMB_DIM = 5          # x, y, fuse_norm, blast_norm, team_is_enemy
SUMMARY_DIM = 6       # entity count + farthest-dist summaries
OBS_SIZE = (
    SELF_DIM + MAX_ENEMIES * ENEMY_DIM + MAX_PROJECTILES * PROJ_DIM
    + MAX_OBSTACLES * OBS_DIM + MAX_BOMBS * BOMB_DIM + SUMMARY_DIM
)
MAX_FUSE_TICKS = 360.0    # max fuse ticks for any bomb type
MAX_BLAST_RADIUS = 100.0  # normalize blast radius by this value
PROJECTILE_SPEED_NORM = 300.0  # normalizer for projectile velocity (max super=270)

# Move intents mapping
MOVE_INTENTS = ["none", "n", "s", "e", "w", "ne", "nw", "se", "sw"]


def _projectile_distance_sq(p: Dict[str, Any], sx: float, sy: float) -> float:
    return (float(p["x"]) - sx) ** 2 + (float(p["y"]) - sy) ** 2


def _obstacle_center_distance_sq(o: Dict[str, Any], sx: float, sy: float) -> float:
    return (float(o["x"]) + float(o["width"]) / 2 - sx) ** 2 + (
        float(o["y"]) + float(o["height"]) / 2 - sy
    ) ** 2


def _bomb_distance_sq(b: Dict[str, Any], sx: float, sy: float) -> float:
    return (float(b["x"]) - sx) ** 2 + (float(b["y"]) - sy) ** 2


class TreadsEnv(gym.Env[NDArray[np.float32], Dict[str, Any]]):
    """Gymnasium environment for the Treads tank game."""

    metadata = {"render_modes": []}

    def __init__(self, level: int = 1, seed_start: int = 0, max_episode_steps: int = 1800) -> None:
        super().__init__()
        self.level = level
        self.seed_counter = seed_start
        self.max_episode_steps = max_episode_steps
        self.process: Optional[subprocess.Popen[str]] = None
        self._buffer = ""
        self._init_data = None
        self._last_obs_raw = None
        self._prev_enemy_alive_count = 0
        self._prev_enemy_health_total = 0
        self._prev_self_health = 0
        self._step_count = 0
        self._save_replay = False
        self._persistent = True  # use persistent mode by default

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

    def _start_process(self, seed: int) -> None:
        """Spawn the Node.js CLI runner subprocess (persistent or legacy)."""
        if self._persistent and self.process is not None:
            # Persistent mode: process already running, just send reset command
            return

        if self.process is not None:
            self._kill_process()

        cmd = [
            "node",
            CLI_RUNNER_PATH,
        ]
        if self._persistent:
            cmd.append("--persistent")
        else:
            cmd.extend([
                "--level", str(self.level),
                "--seed", str(seed),
                "--max-ticks", str(self.max_episode_steps),
            ])
            if self._save_replay:
                cmd.append("--save-replay")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._buffer = ""

        if self._persistent:
            # Wait for {"type":"ready"} from the persistent process
            ready_msg = self._read_message()
            assert ready_msg["type"] == "ready", f"Expected ready, got {ready_msg['type']}"

    def _kill_process(self) -> None:
        """Kill the subprocess, after giving it a moment to finish cleanup (e.g. writing replay files)."""
        if self.process is not None:
            try:
                # In persistent mode, send exit command first
                if self._persistent:
                    try:
                        if self.process.stdin is not None:
                            self.process.stdin.write(json.dumps({"type": "exit"}) + "\n")
                            self.process.stdin.flush()
                    except Exception:
                        pass
                if self.process.stdin is not None:
                    self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                    self.process.wait(timeout=5)
                except Exception:
                    pass
            self.process = None

    def _read_message(self) -> Dict[str, Any]:
        """Read a single JSON line from the subprocess stdout."""
        assert self.process is not None and self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("CLI runner process terminated unexpectedly")
        return json.loads(line.strip())  # type: ignore[no-any-return]

    def _send_action(self, action_dict: Dict[str, Any]) -> None:
        """Send an action as JSON to the subprocess stdin."""
        assert self.process is not None and self.process.stdin is not None
        line = json.dumps(action_dict) + "\n"
        self.process.stdin.write(line)
        self.process.stdin.flush()

    def _normalize_obs(self, obs_raw: Dict[str, Any]) -> NDArray[np.float32]:
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
        result[idx + 5] = 1.0 if self_data.get("wasLastMoveBlocked", False) else 0.0
        result[idx + 6] = min(float(self_data.get("invulnerabilityTicksRemaining", 0)) / 8.0, 1.0)
        result[idx + 7] = min(float(obs_raw.get("tick", 0)) / 1080.0, 1.0)
        result[idx + 8] = self_data["health"] / max(self_data["maxHealth"], 1)

        # Derived aim features (critical for learning)
        sx = self_data["x"] + self_data["size"] / 2
        sy = self_data["y"] + self_data["size"] / 2
        enemies_alive = [e for e in obs_raw["enemies"] if not e["destroyed"]]
        if enemies_alive:
            nearest = min(
                enemies_alive,
                key=lambda e: (e["x"] + e["size"] / 2 - sx) ** 2
                + (e["y"] + e["size"] / 2 - sy) ** 2,
            )
            ex = nearest["x"] + nearest["size"] / 2
            ey = nearest["y"] + nearest["size"] / 2
            angle_to_enemy = math.atan2(ey - sy, ex - sx)
            dist_to_enemy = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
            aim_angle = self_data["aimAngle"]
            aim_error = math.atan2(
                math.sin(aim_angle - angle_to_enemy),
                math.cos(aim_angle - angle_to_enemy),
            )
            result[idx + 9] = angle_to_enemy / (2 * math.pi) + 0.5  # normalize to [0,1]
            arena_diag = math.sqrt(ARENA_WIDTH**2 + ARENA_HEIGHT**2)
            result[idx + 10] = min(dist_to_enemy / arena_diag, 1.0)
            result[idx + 11] = aim_error / math.pi * 0.5 + 0.5  # normalize to [0,1]
        else:
            result[idx + 9] = 0.5
            result[idx + 10] = 0.0
            result[idx + 11] = 0.5
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
                result[idx + 4] = 1.0 if e.get("bombType") else 0.0
                result[idx + 5] = e["health"] / max(e["maxHealth"], 1)
            # else zeros (no enemy)
            idx += ENEMY_DIM

        # Projectiles (up to MAX_PROJECTILES, sorted by distance to self)
        projectiles = cast(List[Dict[str, Any]], obs_raw.get("projectiles", []))
        projectiles.sort(key=lambda p: _projectile_distance_sq(p, sx, sy))
        for i in range(MAX_PROJECTILES):
            if i < len(projectiles):
                p = projectiles[i]
                result[idx] = p["x"] / ARENA_WIDTH
                result[idx + 1] = p["y"] / ARENA_HEIGHT
                result[idx + 2] = p["vx"] / PROJECTILE_SPEED_NORM * 0.5 + 0.5  # normalize to [0,1]
                result[idx + 3] = p["vy"] / PROJECTILE_SPEED_NORM * 0.5 + 0.5
                result[idx + 4] = 1.0 if p["team"] == "enemy" else 0.0
            idx += PROJ_DIM

        # Obstacles (up to MAX_OBSTACLES, sorted by distance to self)
        obstacles = cast(List[Dict[str, Any]], obs_raw.get("obstacles", []))
        obstacles.sort(key=lambda o: _obstacle_center_distance_sq(o, sx, sy))
        for i in range(MAX_OBSTACLES):
            if i < len(obstacles):
                o = obstacles[i]
                result[idx] = o["x"] / ARENA_WIDTH
                result[idx + 1] = o["y"] / ARENA_HEIGHT
                result[idx + 2] = o["width"] / ARENA_WIDTH
                result[idx + 3] = o["height"] / ARENA_HEIGHT
            idx += OBS_DIM

        # Fix 2: Bombs (up to MAX_BOMBS, sorted by distance to self)
        arena_diag = math.sqrt(ARENA_WIDTH ** 2 + ARENA_HEIGHT ** 2)
        bombs = cast(List[Dict[str, Any]], obs_raw.get("bombs", []))
        bombs.sort(key=lambda b: _bomb_distance_sq(b, sx, sy))
        for i in range(MAX_BOMBS):
            if i < len(bombs):
                b = bombs[i]
                result[idx] = b["x"] / ARENA_WIDTH
                result[idx + 1] = b["y"] / ARENA_HEIGHT
                result[idx + 2] = min(float(b["fuseTicksRemaining"]) / MAX_FUSE_TICKS, 1.0)
                result[idx + 3] = min(float(b["blastRadius"]) / MAX_BLAST_RADIUS, 1.0)
                result[idx + 4] = 1.0 if b["team"] == "enemy" else 0.0
            idx += BOMB_DIM

        # Fix 3: Summary features — entity counts + farthest-distance cues
        alive_enemies = [e for e in obs_raw["enemies"] if not e["destroyed"]]
        # [0] alive enemy count (normalized)
        result[idx] = min(len(alive_enemies) / max(MAX_ENEMIES, 1), 1.0)
        # [1] distance to farthest alive enemy (normalized)
        if alive_enemies:
            farthest_sq = max(
                (e["x"] + e["size"] / 2 - sx) ** 2 + (e["y"] + e["size"] / 2 - sy) ** 2
                for e in alive_enemies
            )
            result[idx + 1] = min(math.sqrt(farthest_sq) / arena_diag, 1.0)
        else:
            result[idx + 1] = 0.0
        # [2] projectile count (normalized)
        all_projs = cast(List[Dict[str, Any]], obs_raw.get("projectiles", []))
        result[idx + 2] = min(len(all_projs) / max(MAX_PROJECTILES, 1), 1.0)
        # [3] bomb count (normalized)
        result[idx + 3] = min(len(bombs) / max(MAX_BOMBS, 1), 1.0)
        # [4] closest enemy bomb distance (threat indicator; 1.0 = no threat)
        enemy_bombs = [b for b in bombs if b.get("team") == "enemy"]
        if enemy_bombs:
            closest_bomb_sq = min(_bomb_distance_sq(b, sx, sy) for b in enemy_bombs)
            result[idx + 4] = min(math.sqrt(closest_bomb_sq) / arena_diag, 1.0)
        else:
            result[idx + 4] = 1.0
        # [5] farthest projectile distance (spread indicator)
        if all_projs:
            farthest_proj_sq = max(
                (p["x"] - sx) ** 2 + (p["y"] - sy) ** 2 for p in all_projs
            )
            result[idx + 5] = min(math.sqrt(farthest_proj_sq) / arena_diag, 1.0)
        else:
            result[idx + 5] = 0.0
        idx += SUMMARY_DIM

        return np.clip(result, 0.0, 1.0)

    def _compute_reward(self, obs_raw: Dict[str, Any], action_dict: Optional[Dict[str, Any]] = None, done_msg: Optional[Dict[str, Any]] = None) -> float:
        """Compute HP-based combat reward used by the modern training pipeline."""
        reward = -0.001
        alive_enemies = sum(
            1 for e in obs_raw["enemies"] if not e["destroyed"]
        )
        enemy_health_total = sum(
            float(e["health"]) for e in obs_raw["enemies"] if not e["destroyed"]
        )
        self_health = float(obs_raw["self"]["health"])

        damage_dealt = self._prev_enemy_health_total - enemy_health_total
        if damage_dealt > 0:
            reward += 0.3 * damage_dealt

        damage_taken = self._prev_self_health - self_health
        if damage_taken > 0:
            reward -= 0.3 * damage_taken

        enemies_killed = self._prev_enemy_alive_count - alive_enemies
        if enemies_killed > 0:
            reward += 2.0 * enemies_killed

        self._prev_enemy_alive_count = alive_enemies
        self._prev_enemy_health_total = enemy_health_total
        self._prev_self_health = self_health

        if done_msg is not None:
            if done_msg.get("win"):
                reward += 5.0
            elif done_msg.get("loss"):
                reward -= 3.0
            elif done_msg.get("draw"):
                reward -= 1.0

        if obs_raw["self"].get("destroyed"):
            reward -= 2.0

        return reward

    @staticmethod
    def _has_clear_los(obs_raw: Dict[str, Any], sx: float, sy: float, ex: float, ey: float) -> bool:
        """Check clear line-of-sight from (sx,sy) to (ex,ey) past obstacles."""
        for o in obs_raw.get("obstacles", []):
            ox, oy = o["x"], o["y"]
            ow, oh = o["width"], o["height"]
            if TreadsEnv._line_intersects_rect(sx, sy, ex, ey, ox, oy, ow, oh):
                return False
        return True

    @staticmethod
    def _line_intersects_rect(x1: float, y1: float, x2: float, y2: float, rx: float, ry: float, rw: float, rh: float) -> bool:
        """Check if line segment (x1,y1)-(x2,y2) intersects axis-aligned rect."""
        dx = x2 - x1
        dy = y2 - y1
        # Check intersections with vertical edges
        for edge_x in (rx, rx + rw):
            if dx != 0:
                t = (edge_x - x1) / dx
                if 0 <= t <= 1:
                    y_at_t = y1 + t * dy
                    if ry <= y_at_t <= ry + rh:
                        return True
        # Check intersections with horizontal edges
        for edge_y in (ry, ry + rh):
            if dy != 0:
                t = (edge_y - y1) / dy
                if 0 <= t <= 1:
                    x_at_t = x1 + t * dx
                    if rx <= x_at_t <= rx + rw:
                        return True
        return False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[NDArray[np.float32], Dict[str, Any]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Reset the environment with a new seed."""
        super().reset(seed=seed)
        self.seed_counter += 1
        game_seed = self.seed_counter

        self._start_process(game_seed)
        self._step_count = 0

        if self._persistent:
            # Send reset command to the persistent process
            reset_cmd = cast(
                Dict[str, Any],
                {
                "type": "reset",
                "level": self.level,
                "seed": game_seed,
                "maxTicks": self.max_episode_steps,
                "saveReplay": self._save_replay,
                },
            )
            self._send_action(reset_cmd)

        # Read init message
        self._init_data = self._read_message()
        assert self._init_data["type"] == "init", f"Expected init, got {self._init_data['type']}"

        # Read first observation
        obs_msg = self._read_message()
        assert obs_msg["type"] == "observation", f"Expected observation, got {obs_msg['type']}"

        self._last_obs_raw = obs_msg["observation"]

        # Init tracking
        enemies = [e for e in self._last_obs_raw["enemies"] if not e["destroyed"]]
        self._prev_enemy_alive_count = len(enemies)
        self._prev_enemy_health_total = sum(float(e["health"]) for e in enemies)
        self._prev_self_health = float(self._last_obs_raw["self"]["health"])

        return self._normalize_obs(self._last_obs_raw), {}

    def step(self, action: Any) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Convert action to JSON
        if not isinstance(action, dict):
            raise ValueError("Action must be a dict")

        action_map = cast(Dict[str, Any], action)
        move_idx = int(cast(int, action_map["move"]))
        aim_arr = cast(NDArray[np.float32], action_map["aim_angle"])
        aim_angle = float(aim_arr[0])
        fire = bool(cast(int, action_map["fire"]))
        plant_bomb = bool(cast(int, action_map["plant_bomb"]))

        action_dict = cast(
            Dict[str, Any],
            {
                "move": MOVE_INTENTS[move_idx],
                "aimAngle": aim_angle,
                "fire": fire,
                "plantBomb": plant_bomb,
            },
        )

        self._send_action(action_dict)
        self._step_count += 1

        # Read next message
        msg = self._read_message()

        if msg["type"] == "result":
            reward = self._compute_reward(self._last_obs_raw, done_msg=msg) if self._last_obs_raw is not None else 0.0
            # Use last obs
            assert self._last_obs_raw is not None
            obs = self._normalize_obs(self._last_obs_raw)
            if not self._persistent:
                self._kill_process()
            return obs, reward, True, False, {"result": msg}

        assert msg["type"] == "observation", f"Expected observation, got {msg['type']}"
        self._last_obs_raw = msg["observation"]

        # Check if player is destroyed
        done = self._last_obs_raw["self"]["destroyed"]

        assert self._last_obs_raw is not None
        reward = self._compute_reward(self._last_obs_raw, action_dict=action_dict)
        if done:
            reward -= 2.0

        obs = self._normalize_obs(self._last_obs_raw)

        # Check if we need to read a result message when done
        truncated = False
        if done:
            # Read the result message that follows
            result_msg = self._read_message()
            if not self._persistent:
                self._kill_process()
            return obs, reward, True, False, {"result": result_msg}

        return obs, reward, False, truncated, {}

    def close(self) -> None:
        """Clean up the subprocess."""
        self._kill_process()


# Number of discrete aim bins (evenly spaced across 0–2π)
NUM_AIM_BINS = 16


class TreadsEnvDiscrete(gym.Env[NDArray[np.float32], Any]):
    """
    MultiDiscrete action space wrapper for TreadsEnv to work with SB3.
    Action: MultiDiscrete([9, NUM_AIM_BINS, 2, 2])
      - move: 0-8 (9 directions)
      - aim:  0-(NUM_AIM_BINS-1), mapped to [0, 2π)
      - fire: 0/1
      - bomb: 0/1
    """

    metadata = {"render_modes": []}

    def __init__(self, level: int = 1, seed_start: int = 0, max_episode_steps: int = 1800) -> None:
        super().__init__()
        self._env = TreadsEnv(
            level=level, seed_start=seed_start, max_episode_steps=max_episode_steps
        )

        self.action_space = spaces.MultiDiscrete([9, NUM_AIM_BINS, 2, 2])
        self.observation_space = self._env.observation_space

    def _convert_action(self, action: NDArray[np.int64]) -> Dict[str, Any]:
        """Convert MultiDiscrete action to dict action."""
        move_idx = int(action[0])
        aim_bin = int(action[1])
        fire = bool(action[2])
        plant_bomb = bool(action[3])

        # Convert aim bin to radians: bin i → i * (2π / NUM_AIM_BINS)
        aim_angle = aim_bin * (2 * math.pi / NUM_AIM_BINS)

        return {
            "move": move_idx,
            "aim_angle": np.array([aim_angle], dtype=np.float32),
            "fire": int(fire),
            "plant_bomb": int(plant_bomb),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[NDArray[np.float32], Dict[str, Any]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._env.reset(seed=seed, options=options)

    def step(self, action: NDArray[np.int64]) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        dict_action = self._convert_action(action)
        return self._env.step(dict_action)

    def close(self) -> None:
        self._env.close()
