"""
Treads Gymnasium Environment.
Wraps the Node.js CLI runner as an OpenAI Gymnasium environment.
"""

import os
import json
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import grpc
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces

from runtime_codec import (
    ACTION_HEAD_SIZES,
    ARENA_DIAGONAL,
    MOVE_INTENTS,
    OBS_SIZE,
    decode_multi_discrete_action,
    navigation_guidance,
    normalize_observation,
    set_tick_norm_ticks,
)

GRPC_SERVER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".training-dist", "training", "grpc-env-server.js"
)
PROTO_PATH = os.path.join(os.path.dirname(__file__), "proto", "treads.proto")
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "_generated")


def _ensure_proto_stubs() -> Tuple[Any, Any]:
    os.makedirs(GENERATED_DIR, exist_ok=True)
    if GENERATED_DIR not in sys.path:
        sys.path.insert(0, GENERATED_DIR)

    pb2_path = os.path.join(GENERATED_DIR, "treads_pb2.py")
    pb2_grpc_path = os.path.join(GENERATED_DIR, "treads_pb2_grpc.py")
    proto_mtime = os.path.getmtime(PROTO_PATH)
    stubs_stale = not (os.path.exists(pb2_path) and os.path.exists(pb2_grpc_path)) or any(
        os.path.getmtime(path) < proto_mtime for path in (pb2_path, pb2_grpc_path) if os.path.exists(path)
    )
    if stubs_stale:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{os.path.dirname(PROTO_PATH)}",
                f"--python_out={GENERATED_DIR}",
                f"--grpc_python_out={GENERATED_DIR}",
                PROTO_PATH,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to generate protobuf stubs. "
                f"stdout={result.stdout[-1000:]} stderr={result.stderr[-1000:]}"
            )

    import treads_pb2  # type: ignore[import-not-found]
    import treads_pb2_grpc  # type: ignore[import-not-found]

    return treads_pb2, treads_pb2_grpc


TREADS_PB2, TREADS_PB2_GRPC = _ensure_proto_stubs()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

STEP_PENALTY = -0.001
HIT_REWARD = 0.3
TOOK_DAMAGE_PENALTY = -0.3
KILL_REWARD = 2.0
DEATH_REWARD = -2.0
TERMINAL_WIN_REWARD = 5.0
TERMINAL_LOSS_REWARD = -3.0
TIMEOUT_REWARD = -1.0
# 5x bump from prior (0.5/0.15) so per-step shaping is meaningful vs the
# per-step penalty after ARENA_DIAGONAL (~1118) normalization.
APPROACH_SCALE = 2.5
REWARD_GAMMA = 0.997


class TreadsEnv(gym.Env[NDArray[np.float32], Dict[str, Any]]):
    """Gymnasium environment for the Treads tank game."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        level: int = 1,
        seed_start: int = 0,
        max_episode_steps: int = 720,
        spawn_jitter: bool = False,
        procedural_levels: bool = False,
        difficulty_band: float = 0.0,
        player_max_ammo: int = -1,
        player_max_bombs: int = -1,
    ) -> None:
        super().__init__()
        self.level = level
        self.seed_counter = seed_start
        self.max_episode_steps = max_episode_steps
        self.spawn_jitter = bool(spawn_jitter)
        self.procedural_levels = bool(procedural_levels)
        self.difficulty_band = float(np.clip(difficulty_band, 0.0, 1.0))
        self.player_max_ammo = int(player_max_ammo)
        self.player_max_bombs = int(player_max_bombs)
        set_tick_norm_ticks(max_episode_steps)
        self.process: Optional[subprocess.Popen[str]] = None
        self._grpc_port: Optional[int] = None
        self._grpc_channel: Optional[grpc.Channel] = None
        self._grpc_stub: Optional[Any] = None
        self._session_id: str = ""
        self._init_data = None
        self._last_obs_raw = None
        self._prev_enemy_alive_count = 0
        self._prev_enemy_health_total = 0
        self._prev_self_health = 0
        self._prev_enemy_dist = 0.0
        self._approach_target_id = ""
        self._step_count = 0
        self._save_replay = False
        self._rpc_timeout_sec = 30.0

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

    def _start_process(self) -> None:
        """Start the gRPC env server process and initialize a client channel."""
        if self.process is not None and self._grpc_stub is not None and self._grpc_channel is not None:
            return

        if not os.path.exists(GRPC_SERVER_PATH):
            raise FileNotFoundError(
                "gRPC env server not found. "
                f"Expected: {GRPC_SERVER_PATH}"
            )

        if self.process is not None:
            self._kill_process()

        port = _find_free_port()
        self._grpc_port = port
        cmd = [
            "node",
            GRPC_SERVER_PATH,
            "--port",
            str(port),
            "--parent-pid",
            str(os.getpid()),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        endpoint = f"127.0.0.1:{port}"
        self._grpc_channel = grpc.insecure_channel(
            endpoint,
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ],
        )
        self._grpc_stub = TREADS_PB2_GRPC.TreadsEnvServiceStub(self._grpc_channel)

        # Wait for server readiness.
        deadline = time.time() + 10.0
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                assert self._grpc_stub is not None
                self._grpc_stub.Health(TREADS_PB2.HealthRequest(), timeout=1.0)
                return
            except Exception as exc:  # pragma: no cover - transient process spin-up
                last_error = exc
                time.sleep(0.1)

        self._kill_process()
        raise RuntimeError(f"Failed to start gRPC env server: {last_error}")

    def _kill_process(self) -> None:
        if self._grpc_channel is not None:
            try:
                self._grpc_channel.close()
            except Exception:
                pass
        self._grpc_channel = None
        self._grpc_stub = None
        self._grpc_port = None
        self._session_id = ""

        if self.process is not None:
            try:
                self.process.terminate()
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

    def _rpc_reset(self, game_seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self._grpc_stub is not None
        request = TREADS_PB2.ResetRequest(
            session_id=self._session_id,
            level=int(self.level),
            seed=int(game_seed),
            max_ticks=int(self.max_episode_steps),
            save_replay=bool(self._save_replay),
            spawn_jitter=self.spawn_jitter,
            procedural_levels=self.procedural_levels,
            difficulty_band=self.difficulty_band,
            player_max_ammo=self.player_max_ammo,
            player_max_bombs=self.player_max_bombs,
        )
        response = self._grpc_stub.Reset(request, timeout=self._rpc_timeout_sec)
        self._session_id = str(response.session_id)
        init_data = cast(Dict[str, Any], json.loads(str(response.init_json)))
        obs_data = cast(Dict[str, Any], json.loads(str(response.observation_json)))
        return init_data, obs_data

    def _rpc_step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        assert self._grpc_stub is not None
        request = TREADS_PB2.StepRequest(
            session_id=self._session_id,
            move=str(action.get("move", "none")),
            aim_angle=float(action.get("aimAngle", 0.0)),
            fire=bool(action.get("fire", False)),
            plant_bomb=bool(action.get("plantBomb", False)),
        )
        response = self._grpc_stub.Step(request, timeout=self._rpc_timeout_sec)
        payload: Dict[str, Any] = {
            "done": bool(response.done),
            "observation": None,
            "result": None,
        }
        if response.observation_json:
            payload["observation"] = cast(Dict[str, Any], json.loads(str(response.observation_json)))
        if response.result_json and response.result_json != "null":
            payload["result"] = cast(Dict[str, Any], json.loads(str(response.result_json)))
        return payload

    def _normalize_obs(self, obs_raw: Dict[str, Any]) -> NDArray[np.float32]:
        return normalize_observation(obs_raw)

    @staticmethod
    def _enemy_path_distance(obs_raw: Dict[str, Any], enemy: Dict[str, Any]) -> float:
        self_data = cast(Dict[str, Any], obs_raw["self"])
        sx = float(self_data["x"]) + float(self_data["size"]) / 2.0
        sy = float(self_data["y"]) + float(self_data["size"]) / 2.0
        ex = float(enemy["x"]) + float(enemy["size"]) / 2.0
        ey = float(enemy["y"]) + float(enemy["size"]) / 2.0
        _angle, distance, _reachable = navigation_guidance(
            obs_raw, sx, sy, ex, ey, float(self_data["size"])
        )
        return distance

    @classmethod
    def _nearest_enemy_distance(cls, obs_raw: Dict[str, Any]) -> Tuple[float, str]:
        alive_enemies = [
            e for e in cast(List[Dict[str, Any]], obs_raw["enemies"]) if not bool(e["destroyed"])
        ]
        if not alive_enemies:
            return 0.0, ""

        distances = [(cls._enemy_path_distance(obs_raw, enemy), enemy) for enemy in alive_enemies]
        distance, nearest = min(distances, key=lambda item: item[0])
        return distance, str(nearest["id"])

    def _reset_reward_trackers(self, obs_raw: Dict[str, Any]) -> None:
        enemies = [e for e in cast(List[Dict[str, Any]], obs_raw["enemies"]) if not bool(e["destroyed"])]
        self._prev_enemy_alive_count = len(enemies)
        self._prev_enemy_health_total = sum(float(enemy["health"]) for enemy in enemies)
        self._prev_self_health = float(obs_raw["self"]["health"])
        self._prev_enemy_dist, self._approach_target_id = self._nearest_enemy_distance(obs_raw)

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

        terminal = bool(done_msg and (done_msg.get("win") or done_msg.get("loss")))
        tracked_enemy = next(
            (enemy for enemy in alive_enemies if str(enemy["id"]) == self._approach_target_id),
            None,
        )
        if self._prev_enemy_dist >= 0.0 and self._approach_target_id:
            close_potential = terminal or tracked_enemy is None
            current_enemy_dist = (
                0.0
                if close_potential
                else self._enemy_path_distance(obs_raw, cast(Dict[str, Any], tracked_enemy))
            )
            previous_potential = 1.0 - min(max(self._prev_enemy_dist / ARENA_DIAGONAL, 0.0), 1.0)
            current_potential = (
                0.0
                if close_potential
                else 1.0 - min(max(current_enemy_dist / ARENA_DIAGONAL, 0.0), 1.0)
            )
            reward += APPROACH_SCALE * (
                REWARD_GAMMA * current_potential - previous_potential
            )
            self._prev_enemy_dist = -1.0 if close_potential else current_enemy_dist
            if close_potential:
                self._approach_target_id = ""

        if not terminal and not self._approach_target_id and alive_enemies:
            self._prev_enemy_dist, self._approach_target_id = self._nearest_enemy_distance(obs_raw)

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

        self._start_process()
        self._step_count = 0

        try:
            self._init_data, self._last_obs_raw = self._rpc_reset(game_seed)
        except grpc.RpcError as exc:
            self._kill_process()
            raise TimeoutError(f"TreadsEnv.reset RPC failed: {exc}") from exc

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

        self._step_count += 1

        try:
            msg = self._rpc_step(action_dict)
        except grpc.RpcError as exc:
            self._kill_process()
            raise TimeoutError(f"TreadsEnv.step RPC failed: {exc}") from exc

        if msg["done"]:
            result = cast(Optional[Dict[str, Any]], msg.get("result"))
            final_obs = cast(Optional[Dict[str, Any]], msg.get("observation"))
            if final_obs is not None:
                self._last_obs_raw = final_obs
            assert self._last_obs_raw is not None
            reward = self._compute_reward(self._last_obs_raw, done_msg=result)
            obs = self._normalize_obs(self._last_obs_raw)
            terminated = bool(result and (result.get("win") or result.get("loss")))
            truncated = bool(result and (result.get("draw") or result.get("timeout")))
            return obs, reward, terminated, truncated, {"result": result}

        observation = cast(Optional[Dict[str, Any]], msg.get("observation"))
        if observation is None:
            raise RuntimeError("TreadsEnv.step RPC returned no observation for non-terminal step")
        self._last_obs_raw = observation

        assert self._last_obs_raw is not None
        reward = self._compute_reward(self._last_obs_raw)
        obs = self._normalize_obs(self._last_obs_raw)
        return obs, reward, False, False, {}

    def close(self) -> None:
        """Clean up the subprocess."""
        self._kill_process()


class TreadsEnvDiscrete(gym.Env[NDArray[np.float32], Any]):
    """
    MultiDiscrete action wrapper using the versioned neural model contract.
    """

    metadata = {"render_modes": []}

    def __init__(self, level: int = 1, seed_start: int = 0, max_episode_steps: int = 720) -> None:
        super().__init__()
        self._env = TreadsEnv(
            level=level, seed_start=seed_start, max_episode_steps=max_episode_steps
        )

        self.action_space = spaces.MultiDiscrete(ACTION_HEAD_SIZES)
        self.observation_space = self._env.observation_space

    def _convert_action(self, action: NDArray[np.int64]) -> Dict[str, Any]:
        """Convert MultiDiscrete action to dict action."""
        return decode_multi_discrete_action(action, self._env._last_obs_raw)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[NDArray[np.float32], Dict[str, Any]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._env.reset(seed=seed, options=options)

    def step(self, action: NDArray[np.int64]) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        dict_action = self._convert_action(action)
        return self._env.step(dict_action)

    def close(self) -> None:
        self._env.close()


TreadsEnvFlat = TreadsEnvDiscrete
