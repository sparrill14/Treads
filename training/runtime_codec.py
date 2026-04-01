import math
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

ARENA_WIDTH = 1000.0
ARENA_HEIGHT = 500.0
MAX_ENEMIES = 6
MAX_PROJECTILES = 15
MAX_OBSTACLES = 5
MAX_BOMBS = 6
SELF_DIM = 12
ENEMY_DIM = 9
PROJ_DIM = 5
OBS_DIM = 4
BOMB_DIM = 5
SUMMARY_DIM = 6
OBS_SIZE = (
    SELF_DIM
    + MAX_ENEMIES * ENEMY_DIM
    + MAX_PROJECTILES * PROJ_DIM
    + MAX_OBSTACLES * OBS_DIM
    + MAX_BOMBS * BOMB_DIM
    + SUMMARY_DIM
)

MAX_FUSE_TICKS = 360.0
MAX_BLAST_RADIUS = 100.0
PROJECTILE_SPEED_NORM = 300.0
MOVE_INTENTS = ["none", "n", "s", "e", "w", "ne", "nw", "se", "sw"]
FIRE_THRESHOLD = 0.0
BOMB_THRESHOLD = 0.5
MOVE_DEAD_ZONE = 0.33
AIM_OFFSET_LIMIT = math.pi
ARENA_DIAGONAL = math.sqrt(ARENA_WIDTH * ARENA_WIDTH + ARENA_HEIGHT * ARENA_HEIGHT)
tick_norm_ticks = 720.0

ObsDict = Dict[str, Any]
DecodedAction = Dict[str, Any]

SQRT2_2 = math.sqrt(2.0) / 2.0
MOVE_DIR_MAP = {
    "none": (0.0, 0.0),
    "n": (0.0, -1.0),
    "s": (0.0, 1.0),
    "e": (1.0, 0.0),
    "w": (-1.0, 0.0),
    "ne": (SQRT2_2, -SQRT2_2),
    "nw": (-SQRT2_2, -SQRT2_2),
    "se": (SQRT2_2, SQRT2_2),
    "sw": (-SQRT2_2, SQRT2_2),
}


def _projectile_distance_sq(p: ObsDict, sx: float, sy: float) -> float:
    return (float(p["x"]) - sx) ** 2 + (float(p["y"]) - sy) ** 2


def _obstacle_center_distance_sq(o: ObsDict, sx: float, sy: float) -> float:
    return (float(o["x"]) + float(o["width"]) / 2.0 - sx) ** 2 + (
        float(o["y"]) + float(o["height"]) / 2.0 - sy
    ) ** 2


def _bomb_distance_sq(b: ObsDict, sx: float, sy: float) -> float:
    return (float(b["x"]) - sx) ** 2 + (float(b["y"]) - sy) ** 2


def move_intent_to_dir(intent: str) -> Tuple[float, float]:
    return MOVE_DIR_MAP.get(intent, (0.0, 0.0))


def segment_intersects_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    rx: float,
    ry: float,
    rw: float,
    rh: float,
) -> bool:
    dx = x2 - x1
    dy = y2 - y1
    t_min = 0.0
    t_max = 1.0

    if abs(dx) < 1e-10:
        if x1 < rx or x1 > rx + rw:
            return False
    else:
        t1 = (rx - x1) / dx
        t2 = (rx + rw - x1) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False

    if abs(dy) < 1e-10:
        if y1 < ry or y1 > ry + rh:
            return False
    else:
        t1 = (ry - y1) / dy
        t2 = (ry + rh - y1) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False

    return True


def has_line_of_sight(
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    obstacles: List[ObsDict],
) -> bool:
    for obstacle in obstacles:
        if segment_intersects_rect(
            sx,
            sy,
            tx,
            ty,
            float(obstacle["x"]),
            float(obstacle["y"]),
            float(obstacle["width"]),
            float(obstacle["height"]),
        ):
            return False
    return True


def set_tick_norm_ticks(ticks: int) -> None:
    global tick_norm_ticks
    tick_norm_ticks = max(1.0, float(ticks))


def normalize_observation(obs_raw: ObsDict) -> NDArray[np.float32]:
    result = np.zeros(OBS_SIZE, dtype=np.float32)
    idx = 0

    self_data = cast(ObsDict, obs_raw["self"])
    sx = float(self_data["x"]) + float(self_data["size"]) / 2.0
    sy = float(self_data["y"]) + float(self_data["size"]) / 2.0
    living_enemies = [
        e
        for e in cast(List[ObsDict], obs_raw.get("enemies", []))
        if not bool(e.get("destroyed", False))
    ]
    living_enemies.sort(
        key=lambda e: (float(e["x"]) + float(e["size"]) / 2.0 - sx) ** 2
        + (float(e["y"]) + float(e["size"]) / 2.0 - sy) ** 2
    )

    result[idx] = float(self_data["x"]) / ARENA_WIDTH
    result[idx + 1] = float(self_data["y"]) / ARENA_HEIGHT
    result[idx + 2] = float(self_data["aimAngle"]) / (2.0 * math.pi) + 0.5
    result[idx + 3] = float(self_data["speed"]) / 100.0

    has_los = 0.0
    if living_enemies:
        nearest = living_enemies[0]
        has_los = (
            1.0
            if has_line_of_sight(
                sx,
                sy,
                float(nearest["x"]) + float(nearest["size"]) / 2.0,
                float(nearest["y"]) + float(nearest["size"]) / 2.0,
                cast(List[ObsDict], obs_raw.get("obstacles", [])),
            )
            else 0.0
        )
    result[idx + 4] = has_los
    result[idx + 5] = 1.0 if bool(self_data.get("wasLastMoveBlocked", False)) else 0.0
    result[idx + 6] = min(float(self_data.get("invulnerabilityTicksRemaining", 0.0)) / 8.0, 1.0)
    result[idx + 7] = min(float(obs_raw.get("tick", 0.0)) / tick_norm_ticks, 1.0)
    result[idx + 8] = float(self_data["health"]) / max(float(self_data["maxHealth"]), 1.0)

    if living_enemies:
        nearest = living_enemies[0]
        ex = float(nearest["x"]) + float(nearest["size"]) / 2.0
        ey = float(nearest["y"]) + float(nearest["size"]) / 2.0
        angle_to_enemy = math.atan2(ey - sy, ex - sx)
        dist_to_enemy = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
        aim_angle = float(self_data["aimAngle"])
        aim_error = math.atan2(
            math.sin(aim_angle - angle_to_enemy),
            math.cos(aim_angle - angle_to_enemy),
        )
        result[idx + 9] = angle_to_enemy / (2.0 * math.pi) + 0.5
        result[idx + 10] = min(dist_to_enemy / ARENA_DIAGONAL, 1.0)
        result[idx + 11] = (aim_error / math.pi) * 0.5 + 0.5
    else:
        result[idx + 9] = 0.5
        result[idx + 10] = 0.0
        result[idx + 11] = 0.5
    idx += SELF_DIM

    for i in range(MAX_ENEMIES):
        if i < len(living_enemies):
            enemy = living_enemies[i]
            ecx = float(enemy["x"]) + float(enemy["size"]) / 2.0
            ecy = float(enemy["y"]) + float(enemy["size"]) / 2.0
            result[idx] = ((ecx - sx) / ARENA_DIAGONAL) * 0.5 + 0.5
            result[idx + 1] = ((ecy - sy) / ARENA_DIAGONAL) * 0.5 + 0.5
            result[idx + 2] = float(enemy["aimAngle"]) / (2.0 * math.pi) + 0.5
            result[idx + 3] = float(enemy["speed"]) / 100.0
            result[idx + 4] = 1.0 if enemy.get("bombType") else 0.0
            result[idx + 5] = float(enemy["health"]) / max(float(enemy["maxHealth"]), 1.0)

            angle_from_enemy_to_player = math.atan2(sy - ecy, sx - ecx)
            enemy_aim_error = math.atan2(
                math.sin(float(enemy["aimAngle"]) - angle_from_enemy_to_player),
                math.cos(float(enemy["aimAngle"]) - angle_from_enemy_to_player),
            )
            result[idx + 6] = 1.0 - abs(enemy_aim_error) / math.pi
            has_ammo = float(enemy.get("maxAmmo", 0.0)) > 0.0
            result[idx + 7] = (
                1.0 if enemy.get("ammoType") == "super" else 0.5
            ) if has_ammo else 0.0

            move_dir = move_intent_to_dir(str(enemy.get("lastMoveIntent", "none")))
            if move_dir[0] != 0.0 or move_dir[1] != 0.0:
                to_player_dx = sx - ecx
                to_player_dy = sy - ecy
                to_player_dist = math.sqrt(to_player_dx * to_player_dx + to_player_dy * to_player_dy)
                if to_player_dist > 1e-6:
                    dot = (move_dir[0] * to_player_dx + move_dir[1] * to_player_dy) / to_player_dist
                    result[idx + 8] = dot * 0.5 + 0.5
                else:
                    result[idx + 8] = 0.5
            else:
                result[idx + 8] = 0.5
        idx += ENEMY_DIM

    projectiles = cast(List[ObsDict], obs_raw.get("projectiles", []))
    projectiles.sort(key=lambda p: _projectile_distance_sq(p, sx, sy))
    for i in range(MAX_PROJECTILES):
        if i < len(projectiles):
            projectile = projectiles[i]
            result[idx] = ((float(projectile["x"]) - sx) / ARENA_DIAGONAL) * 0.5 + 0.5
            result[idx + 1] = ((float(projectile["y"]) - sy) / ARENA_DIAGONAL) * 0.5 + 0.5
            result[idx + 2] = (float(projectile["vx"]) / PROJECTILE_SPEED_NORM) * 0.5 + 0.5
            result[idx + 3] = (float(projectile["vy"]) / PROJECTILE_SPEED_NORM) * 0.5 + 0.5
            result[idx + 4] = 1.0 if projectile.get("team") == "enemy" else 0.0
        idx += PROJ_DIM

    obstacles = cast(List[ObsDict], obs_raw.get("obstacles", []))
    obstacles.sort(key=lambda o: _obstacle_center_distance_sq(o, sx, sy))
    for i in range(MAX_OBSTACLES):
        if i < len(obstacles):
            obstacle = obstacles[i]
            result[idx] = (
                (float(obstacle["x"]) + float(obstacle["width"]) / 2.0 - sx)
                / ARENA_DIAGONAL
            ) * 0.5 + 0.5
            result[idx + 1] = (
                (float(obstacle["y"]) + float(obstacle["height"]) / 2.0 - sy)
                / ARENA_DIAGONAL
            ) * 0.5 + 0.5
            result[idx + 2] = float(obstacle["width"]) / ARENA_WIDTH
            result[idx + 3] = float(obstacle["height"]) / ARENA_HEIGHT
        idx += OBS_DIM

    bombs = cast(List[ObsDict], obs_raw.get("bombs", []))
    bombs.sort(key=lambda b: _bomb_distance_sq(b, sx, sy))
    for i in range(MAX_BOMBS):
        if i < len(bombs):
            bomb = bombs[i]
            result[idx] = ((float(bomb["x"]) - sx) / ARENA_DIAGONAL) * 0.5 + 0.5
            result[idx + 1] = ((float(bomb["y"]) - sy) / ARENA_DIAGONAL) * 0.5 + 0.5
            result[idx + 2] = min(float(bomb["fuseTicksRemaining"]) / MAX_FUSE_TICKS, 1.0)
            result[idx + 3] = min(float(bomb["blastRadius"]) / MAX_BLAST_RADIUS, 1.0)
            result[idx + 4] = 1.0 if bomb.get("team") == "enemy" else 0.0
        idx += BOMB_DIM

    result[idx] = min(len(living_enemies) / MAX_ENEMIES, 1.0)
    if living_enemies:
        farthest_enemy_sq = max(
            (float(enemy["x"]) + float(enemy["size"]) / 2.0 - sx) ** 2
            + (float(enemy["y"]) + float(enemy["size"]) / 2.0 - sy) ** 2
            for enemy in living_enemies
        )
        result[idx + 1] = min(math.sqrt(farthest_enemy_sq) / ARENA_DIAGONAL, 1.0)
    else:
        result[idx + 1] = 0.0

    result[idx + 2] = min(len(projectiles) / MAX_PROJECTILES, 1.0)
    result[idx + 3] = min(len(bombs) / MAX_BOMBS, 1.0)

    enemy_bombs = [bomb for bomb in bombs if bomb.get("team") == "enemy"]
    if enemy_bombs:
        closest_bomb_sq = min(_bomb_distance_sq(bomb, sx, sy) for bomb in enemy_bombs)
        result[idx + 4] = min(math.sqrt(closest_bomb_sq) / ARENA_DIAGONAL, 1.0)
    else:
        result[idx + 4] = 1.0

    if projectiles:
        farthest_projectile_sq = max(_projectile_distance_sq(projectile, sx, sy) for projectile in projectiles)
        result[idx + 5] = min(math.sqrt(farthest_projectile_sq) / ARENA_DIAGONAL, 1.0)
    else:
        result[idx + 5] = 0.0

    return np.clip(result, 0.0, 1.0)


def decode_continuous_action(continuous_action: Any, obs_raw: Optional[ObsDict]) -> DecodedAction:
    action = np.asarray(continuous_action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 5:
        aim = float(obs_raw["self"]["aimAngle"]) if obs_raw is not None else 0.0
        return {
            "move": 0,
            "aim_angle": np.array([aim], dtype=np.float32),
            "fire": 0,
            "plant_bomb": 0,
        }

    mx = float(np.clip(action[0], -1.0, 1.0))
    my = float(np.clip(action[1], -1.0, 1.0))
    aim_signal = float(np.clip(action[2], -1.0, 1.0))
    fire_signal = float(np.clip(action[3], -1.0, 1.0))
    bomb_signal = float(np.clip(action[4], -1.0, 1.0))

    go_e = mx > MOVE_DEAD_ZONE
    go_w = mx < -MOVE_DEAD_ZONE
    go_s = my > MOVE_DEAD_ZONE
    go_n = my < -MOVE_DEAD_ZONE
    if go_n and go_e:
        move_idx = MOVE_INTENTS.index("ne")
    elif go_n and go_w:
        move_idx = MOVE_INTENTS.index("nw")
    elif go_s and go_e:
        move_idx = MOVE_INTENTS.index("se")
    elif go_s and go_w:
        move_idx = MOVE_INTENTS.index("sw")
    elif go_n:
        move_idx = MOVE_INTENTS.index("n")
    elif go_s:
        move_idx = MOVE_INTENTS.index("s")
    elif go_e:
        move_idx = MOVE_INTENTS.index("e")
    elif go_w:
        move_idx = MOVE_INTENTS.index("w")
    else:
        move_idx = 0

    raw_obs = obs_raw or {}
    alive_enemies = [
        enemy
        for enemy in cast(List[ObsDict], raw_obs.get("enemies", []))
        if not bool(enemy.get("destroyed", False))
    ]
    if alive_enemies and obs_raw is not None:
        sx = float(raw_obs["self"]["x"]) + float(raw_obs["self"]["size"]) / 2.0
        sy = float(raw_obs["self"]["y"]) + float(raw_obs["self"]["size"]) / 2.0
        nearest = min(
            alive_enemies,
            key=lambda enemy: (float(enemy["x"]) + float(enemy["size"]) / 2.0 - sx) ** 2
            + (float(enemy["y"]) + float(enemy["size"]) / 2.0 - sy) ** 2,
        )
        ex = float(nearest["x"]) + float(nearest["size"]) / 2.0
        ey = float(nearest["y"]) + float(nearest["size"]) / 2.0
        aim_angle = math.atan2(ey - sy, ex - sx) + aim_signal * AIM_OFFSET_LIMIT
    else:
        aim_angle = float(raw_obs["self"]["aimAngle"]) if obs_raw is not None else 0.0

    return {
        "move": move_idx,
        "aim_angle": np.array([aim_angle], dtype=np.float32),
        "fire": int(fire_signal > FIRE_THRESHOLD),
        "plant_bomb": int(bomb_signal > BOMB_THRESHOLD),
    }
