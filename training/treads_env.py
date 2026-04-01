"""
Treads Gymnasium Environment.
Wraps the Node.js CLI runner as an OpenAI Gymnasium environment.
"""

import os
import json
import math
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces

from runtime_codec import (
    ARENA_DIAGONAL,
    MOVE_INTENTS,
    OBS_SIZE,
    normalize_observation,
    set_tick_norm_ticks,
)

# Path to the compiled CLI runner
CLI_RUNNER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "cli-runner.js"
)

STEP_PENALTY = -0.001
HIT_REWARD = 0.3
TOOK_DAMAGE_PENALTY = -0.3
KILL_REWARD = 2.0
DEATH_REWARD = -2.0
TERMINAL_WIN_REWARD = 5.0
TERMINAL_LOSS_REWARD = -3.0
TIMEOUT_REWARD = -1.0
APPROACH_SCALE = 0.5
DODGE_SCALE = 0.15


class TreadsEnv(gym.Env[NDArray[np.float32], Dict[str, Any]]):
    """Gymnasium environment for the Treads tank game."""

    metadata = {"render_modes": []}

    def __init__(self, level: int = 1, seed_start: int = 0, max_episode_steps: int = 720) -> None:
        super().__init__()
        self.level = level
        self.seed_counter = seed_start
        self.max_episode_steps = max_episode_steps
        set_tick_norm_ticks(max_episode_steps)
        self.process: Optional[subprocess.Popen[str]] = None
        self._buffer = ""
        self._init_data = None
        self._last_obs_raw = None
        self._prev_enemy_alive_count = 0
        self._prev_enemy_health_total = 0
        self._prev_self_health = 0
        self._prev_enemy_dist = 0.0
        self._prev_enemy_proj_dist = ARENA_DIAGONAL
        self._approach_target_id = ""
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

        if not os.path.exists(CLI_RUNNER_PATH):
            raise FileNotFoundError(
                "Compiled CLI runner not found. Run `npm run build:training` first. "
                f"Expected: {CLI_RUNNER_PATH}"
            )

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
                    except Exception as exc:
                        print(f"WARNING: failed to send exit command to cli-runner: {exc}")
                if self.process.stdin is not None:
                    self.process.stdin.close()
            except Exception as exc:
                print(f"WARNING: failed while closing cli-runner stdin: {exc}")
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                    self.process.wait(timeout=5)
                except Exception as exc:
                    print(f"WARNING: failed to terminate cli-runner process cleanly: {exc}")
            self.process = None

    def _read_message(self) -> Dict[str, Any]:
        """Read a single JSON line from the subprocess stdout."""
        assert self.process is not None and self.process.stdout is not None
        stdout = self.process.stdout
        holder: Dict[str, str] = {"line": ""}

        def _reader() -> None:
            holder["line"] = stdout.readline()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout=60.0)
        if t.is_alive():
            # Pipe deadlock — kill the subprocess so the reader thread unblocks eventually.
            self._kill_process()
            raise TimeoutError("TreadsEnv._read_message timed out after 60s (pipe deadlock suspected)")

        line = holder["line"]
        if not line:
            stderr_tail = ""
            try:
                # Best-effort: the process may have exited already.
                if self.process.stderr is not None:  # type: ignore[union-attr]
                    stderr_tail = self.process.stderr.read().strip()  # type: ignore[union-attr]
            except Exception:
                pass
            raise RuntimeError(
                "CLI runner process terminated unexpectedly. "
                f"stderr: {stderr_tail[-2000:] if stderr_tail else '<empty>'}"
            )

        msg = cast(Dict[str, Any], json.loads(line.strip()))
        if msg.get("type") == "error":
            raise RuntimeError(f"CLI runner reported error: {msg}")
        return msg

    def _send_action(self, action_dict: Dict[str, Any]) -> None:
        """Send an action as JSON to the subprocess stdin."""
        assert self.process is not None and self.process.stdin is not None
        line = json.dumps(action_dict) + "\n"
        self.process.stdin.write(line)
        self.process.stdin.flush()

    def _normalize_obs(self, obs_raw: Dict[str, Any]) -> NDArray[np.float32]:
        return normalize_observation(obs_raw)

    @staticmethod
    def _nearest_enemy_distance(obs_raw: Dict[str, Any]) -> Tuple[float, str]:
        self_data = cast(Dict[str, Any], obs_raw["self"])
        sx = float(self_data["x"]) + float(self_data["size"]) / 2.0
        sy = float(self_data["y"]) + float(self_data["size"]) / 2.0
        alive_enemies = [
            e for e in cast(List[Dict[str, Any]], obs_raw["enemies"]) if not bool(e["destroyed"])
        ]
        if not alive_enemies:
            return 0.0, ""

        nearest = min(
            alive_enemies,
            key=lambda enemy: (float(enemy["x"]) + float(enemy["size"]) / 2.0 - sx) ** 2
            + (float(enemy["y"]) + float(enemy["size"]) / 2.0 - sy) ** 2,
        )
        dx = float(nearest["x"]) + float(nearest["size"]) / 2.0 - sx
        dy = float(nearest["y"]) + float(nearest["size"]) / 2.0 - sy
        return math.sqrt(dx * dx + dy * dy), str(nearest["id"])

    @staticmethod
    def _nearest_enemy_projectile_distance(obs_raw: Dict[str, Any]) -> float:
        self_data = cast(Dict[str, Any], obs_raw["self"])
        sx = float(self_data["x"]) + float(self_data["size"]) / 2.0
        sy = float(self_data["y"]) + float(self_data["size"]) / 2.0
        enemy_projectiles = [
            projectile
            for projectile in cast(List[Dict[str, Any]], obs_raw.get("projectiles", []))
            if projectile.get("team") == "enemy"
        ]
        if not enemy_projectiles:
            return ARENA_DIAGONAL
        return min(
            math.sqrt(
                (float(projectile["x"]) - sx) ** 2 + (float(projectile["y"]) - sy) ** 2
            )
            for projectile in enemy_projectiles
        )

    def _reset_reward_trackers(self, obs_raw: Dict[str, Any]) -> None:
        enemies = [e for e in cast(List[Dict[str, Any]], obs_raw["enemies"]) if not bool(e["destroyed"])]
        self._prev_enemy_alive_count = len(enemies)
        self._prev_enemy_health_total = sum(float(enemy["health"]) for enemy in enemies)
        self._prev_self_health = float(obs_raw["self"]["health"])
        self._prev_enemy_dist, self._approach_target_id = self._nearest_enemy_distance(obs_raw)
        self._prev_enemy_proj_dist = self._nearest_enemy_projectile_distance(obs_raw)

    def _compute_reward(
        self,
        obs_raw: Dict[str, Any],
        done_msg: Optional[Dict[str, Any]] = None,
    ) -> float:
        reward = STEP_PENALTY
        alive_enemies = [
            e for e in cast(List[Dict[str, Any]], obs_raw["enemies"]) if not bool(e["destroyed"])
        ]
        enemy_health_total = sum(float(enemy["health"]) for enemy in alive_enemies)
        self_health = float(obs_raw["self"]["health"])

        damage_dealt = self._prev_enemy_health_total - enemy_health_total
        if damage_dealt > 0:
            reward += HIT_REWARD * damage_dealt

        damage_taken = self._prev_self_health - self_health
        if damage_taken > 0:
            reward += TOOK_DAMAGE_PENALTY * damage_taken

        enemies_killed = self._prev_enemy_alive_count - len(alive_enemies)
        if enemies_killed > 0:
            reward += KILL_REWARD * enemies_killed
            self._prev_enemy_dist = -1.0
            self._approach_target_id = ""

        current_enemy_dist, current_target_id = self._nearest_enemy_distance(obs_raw)
        if self._prev_enemy_dist >= 0.0 and current_target_id and current_target_id == self._approach_target_id:
            reward += (APPROACH_SCALE * (self._prev_enemy_dist - current_enemy_dist)) / ARENA_DIAGONAL
        self._prev_enemy_dist = current_enemy_dist
        self._approach_target_id = current_target_id

        current_projectile_dist = self._nearest_enemy_projectile_distance(obs_raw)
        reward += (DODGE_SCALE * (current_projectile_dist - self._prev_enemy_proj_dist)) / ARENA_DIAGONAL
        self._prev_enemy_proj_dist = current_projectile_dist

        self._prev_enemy_alive_count = len(alive_enemies)
        self._prev_enemy_health_total = enemy_health_total
        self._prev_self_health = self_health

        if done_msg is not None:
            if done_msg.get("win"):
                reward += TERMINAL_WIN_REWARD
            elif done_msg.get("loss"):
                reward += DEATH_REWARD + TERMINAL_LOSS_REWARD
            elif done_msg.get("draw") or done_msg.get("timeout"):
                reward += TIMEOUT_REWARD

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[NDArray[np.float32], Dict[str, Any]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Reset the environment with a new seed."""
        super().reset(seed=seed)
        if seed is not None:
            game_seed = int(seed)
            self.seed_counter = game_seed + 1
        else:
            game_seed = self.seed_counter
            self.seed_counter += 1

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

        self._reset_reward_trackers(self._last_obs_raw)

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
            final_obs = cast(Optional[Dict[str, Any]], msg.get("observation"))
            if final_obs is not None:
                self._last_obs_raw = final_obs
            assert self._last_obs_raw is not None
            reward = self._compute_reward(self._last_obs_raw, done_msg=msg)
            obs = self._normalize_obs(self._last_obs_raw)
            terminated = bool(msg.get("win") or msg.get("loss"))
            truncated = bool(msg.get("draw") or msg.get("timeout"))
            if not self._persistent:
                self._kill_process()
            return obs, reward, terminated, truncated, {"result": msg}

        assert msg["type"] == "observation", f"Expected observation, got {msg['type']}"
        self._last_obs_raw = msg["observation"]

        assert self._last_obs_raw is not None
        reward = self._compute_reward(self._last_obs_raw)
        obs = self._normalize_obs(self._last_obs_raw)
        return obs, reward, False, False, {}

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

    def __init__(self, level: int = 1, seed_start: int = 0, max_episode_steps: int = 720) -> None:
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


TreadsEnvFlat = TreadsEnvDiscrete
