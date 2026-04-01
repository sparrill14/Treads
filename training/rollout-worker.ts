/**
 * Hybrid Rollout Worker
 *
 * Runs the game simulation and neural network inference entirely in TypeScript.
 * Collects full rollout buffers (obs, action, reward, done, value, log_prob)
 * and bulk-transfers them to Python for PPO gradient updates.
 *
 * Protocol (JSON over stdin/stdout):
 *   Worker starts → {"type":"ready"}
 *   Python → {"type":"set_weights","state_dict":{...}}
 *   Worker → {"type":"weights_set"}
 *   Python → {"type":"collect","n_steps":4096,"level":1,"maxTicks":1800,"seedStart":N}
 *   Worker → {"type":"rollout",...}  (bulk transfer ~5MB)
 *   ... repeat from set_weights ...
 *   Python → {"type":"exit"}
 */

import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';

import { createDefaultControllers, createInitialGameState } from '../src/game/core/MatchFactory';
import { SeededRandom } from '../src/game/core/prng';
import { ReplayRecorder } from '../src/game/core/Replay';
import { Simulation } from '../src/game/core/Simulation';
import type { DeepReadonly } from '../src/game/core/stateUtils';
import type {
	GameState,
	MatchInit,
	MoveIntent,
	TankAction,
	TankController,
	TankObservation,
} from '../src/game/core/types';
import { LEVEL_CONFIGS, type LevelConfig } from '../src/game/LevelConfig';
import { NavigationPlanner } from '../src/game/navigation/NavigationPlanner';
import { PolicyMLP, parseWeightsFromStateDict } from './mlp-inference';

// ---- Constants matching treads_env.py ----
const ARENA_WIDTH = 1000.0;
const ARENA_HEIGHT = 500.0;
// Fix 3: caps raised to cover all 9 levels (max 5 enemies/4 obstacles in Level 7)
const MAX_ENEMIES = 6; // Level 7 has 5 enemies; +1 buffer
const MAX_PROJECTILES = 15; // Level 8: 3×3=9 super shots; generous buffer
const MAX_OBSTACLES = 5; // Level 7 has 4 obstacles; +1 buffer
const MAX_BOMBS = 6; // Level 6: 9 theoretical; cap at 6 live
const SELF_DIM = 12;
const ENEMY_DIM = 9; // dx, dy (relative), aimAngle, speed, hasBombs, health, aimedAtMe, ammoThreat, isApproaching
const PROJ_DIM = 5;
const OBS_DIM = 4;
const BOMB_DIM = 5; // x, y, fuse_norm, blast_norm, team_is_enemy  (Fix 2)
const SUMMARY_DIM = 6; // entity count + farthest-dist summaries        (Fix 3)
const OBS_SIZE =
	SELF_DIM +
	MAX_ENEMIES * ENEMY_DIM +
	MAX_PROJECTILES * PROJ_DIM +
	MAX_OBSTACLES * OBS_DIM +
	MAX_BOMBS * BOMB_DIM +
	SUMMARY_DIM;
const PLAYER_TANK_ID = 'player-0';
const FIRE_THRESHOLD = 0.0; // Fire only when signal > 0.
const BOMB_THRESHOLD = 0.5;
const AIM_OFFSET_LIMIT = Math.PI / 18; // Keep exploration near target bearing (~10deg).
const STEP_PENALTY = -0.001;
const HIT_REWARD = 0.3;
const TOOK_DAMAGE_PENALTY = -0.3;
const KILL_REWARD = 2.0;
const DEATH_REWARD = -2.0;
const TERMINAL_WIN_REWARD = 5.0;
const TERMINAL_LOSS_REWARD = -3.0;
const TIMEOUT_REWARD = -1.0;
const APPROACH_SCALE = 0.5; // potential-based: total ~0.5 for full diagonal approach (~10% of win reward)
const DODGE_SCALE = 0.15; // potential-based: reward for staying away from enemy projectiles
const ARENA_DIAGONAL = Math.sqrt(ARENA_WIDTH * ARENA_WIDTH + ARENA_HEIGHT * ARENA_HEIGHT);
const MAX_FUSE_TICKS = 360.0; // Max fuse ticks for any bomb type
const MAX_BLAST_RADIUS = 100.0; // Normalize blast radius by this value
const PROJECTILE_SPEED_NORM = 300.0; // Normalizer for projectile velocity (max super=270)
let TICK_NORM_TICKS = 720.0;

// ---- MoveIntent → unit direction vector mapping ----
const SQRT2_2 = Math.SQRT2 / 2;
const MOVE_DIR_MAP: Record<MoveIntent, [number, number]> = {
	none: [0, 0],
	n: [0, -1],
	s: [0, 1],
	e: [1, 0],
	w: [-1, 0],
	ne: [SQRT2_2, -SQRT2_2],
	nw: [-SQRT2_2, -SQRT2_2],
	se: [SQRT2_2, SQRT2_2],
	sw: [-SQRT2_2, SQRT2_2],
};
function moveIntentToDir(intent: MoveIntent): [number, number] {
	return MOVE_DIR_MAP[intent] ?? [0, 0];
}

// ---- Line-of-sight utility (segment-AABB intersection) ----
function segmentIntersectsRect(
	x1: number,
	y1: number,
	x2: number,
	y2: number,
	rx: number,
	ry: number,
	rw: number,
	rh: number
): boolean {
	const dx = x2 - x1;
	const dy = y2 - y1;
	let tMin = 0;
	let tMax = 1;
	if (Math.abs(dx) < 1e-10) {
		if (x1 < rx || x1 > rx + rw) return false;
	} else {
		let t1 = (rx - x1) / dx;
		let t2 = (rx + rw - x1) / dx;
		if (t1 > t2) {
			const tmp = t1;
			t1 = t2;
			t2 = tmp;
		}
		tMin = Math.max(tMin, t1);
		tMax = Math.min(tMax, t2);
		if (tMin > tMax) return false;
	}
	if (Math.abs(dy) < 1e-10) {
		if (y1 < ry || y1 > ry + rh) return false;
	} else {
		let t1 = (ry - y1) / dy;
		let t2 = (ry + rh - y1) / dy;
		if (t1 > t2) {
			const tmp = t1;
			t1 = t2;
			t2 = tmp;
		}
		tMin = Math.max(tMin, t1);
		tMax = Math.min(tMax, t2);
		if (tMin > tMax) return false;
	}
	return true;
}

function hasLineOfSight(
	sx: number,
	sy: number,
	tx: number,
	ty: number,
	obstacles: readonly { x: number; y: number; width: number; height: number }[]
): boolean {
	for (const o of obstacles) {
		if (segmentIntersectsRect(sx, sy, tx, ty, o.x, o.y, o.width, o.height)) {
			return false;
		}
	}
	return true;
}

const TRAINING_SCENARIOS: Record<number, LevelConfig> = {
	111: {
		player: { x: 220, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 520, y: 250, ammo: { type: 'basic', count: 0 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2.5 },
	},
	112: {
		player: { x: 220, y: 180 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 520, y: 320, ammo: { type: 'basic', count: 0 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2.5 },
	},
	113: {
		player: { x: 220, y: 250 },
		obstacles: [],
		enemies: [
			{ type: 'simple-moving', x: 620, y: 250, ammo: { type: 'basic', count: 0 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false, turretSpeedMultiplier: 3 },
	},
	121: {
		player: { x: 220, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 560, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 1.5 },
	},
	122: {
		player: { x: 220, y: 180 },
		obstacles: [],
		enemies: [{ type: 'stationary-random-aim', x: 560, y: 320, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 1.5 },
	},
	123: {
		player: { x: 220, y: 250 },
		obstacles: [],
		enemies: [
			{ type: 'simple-moving', x: 620, y: 250, ammo: { type: 'basic', count: 0 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false, turretSpeedMultiplier: 1.5 },
	},
	101: {
		player: { x: 120, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 840, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2 },
	},
	102: {
		player: { x: 120, y: 180 },
		obstacles: [],
		enemies: [{ type: 'stationary-random-aim', x: 820, y: 320, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2 },
	},
	103: {
		player: { x: 120, y: 320 },
		obstacles: [],
		enemies: [
			{ type: 'simple-moving', x: 860, y: 180, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2 },
	},
	201: {
		player: { x: 100, y: 250 },
		obstacles: [{ x: 430, y: 160, width: 40, height: 180 }],
		enemies: [{ type: 'stationary', x: 860, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	},
	202: {
		player: { x: 120, y: 250 },
		obstacles: [{ x: 360, y: 120, width: 30, height: 260 }],
		enemies: [
			{ type: 'stationary', x: 780, y: 140, ammo: { type: 'basic', count: 1 } },
			{ type: 'stationary', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
		rules: { projectileBounces: false },
	},
	203: {
		player: { x: 100, y: 400 },
		obstacles: [
			{ x: 280, y: 110, width: 220, height: 30 },
			{ x: 420, y: 260, width: 220, height: 30 },
		],
		enemies: [
			{ type: 'simple-moving', x: 850, y: 100, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	301: {
		player: { x: 140, y: 250 },
		obstacles: [{ x: 520, y: 120, width: 35, height: 260 }],
		enemies: [
			{
				type: 'bomber',
				x: 780,
				y: 250,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
		],
	},
	302: {
		player: { x: 130, y: 210 },
		obstacles: [
			{ x: 300, y: 120, width: 30, height: 260 },
			{ x: 640, y: 150, width: 30, height: 200 },
		],
		enemies: [
			{
				type: 'bomber',
				x: 820,
				y: 150,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
			{ type: 'stationary-random-aim', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
	},
	303: {
		player: { x: 150, y: 260 },
		obstacles: [
			{ x: 340, y: 110, width: 280, height: 30 },
			{ x: 340, y: 360, width: 280, height: 30 },
		],
		enemies: [
			{
				type: 'super-bomber',
				x: 800,
				y: 140,
				ammo: { type: 'super', count: 0 },
				bombs: { type: 'love', count: 1 },
				navigator: { type: 'astar', aggressionFactor: 4 },
			},
			{ type: 'stationary-random-aim', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
	},
	// ---- Phase 1 supplement: armed target practice (turret 2.5x) ----
	114: {
		player: { x: 220, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 620, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2.5 },
	},
	// ---- Phase 3: full turret, no obstacles ----
	131: {
		player: { x: 200, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 700, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	},
	132: {
		player: { x: 200, y: 180 },
		obstacles: [],
		enemies: [{ type: 'stationary-random-aim', x: 700, y: 320, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	},
	133: {
		player: { x: 200, y: 250 },
		obstacles: [],
		enemies: [
			{ type: 'simple-moving', x: 700, y: 250, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	// ---- Phase 4: obstacles intro, single enemy ----
	141: {
		player: { x: 150, y: 250 },
		obstacles: [{ x: 450, y: 160, width: 40, height: 180 }],
		enemies: [{ type: 'stationary', x: 800, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	},
	142: {
		player: { x: 150, y: 350 },
		obstacles: [{ x: 400, y: 120, width: 30, height: 200 }],
		enemies: [{ type: 'stationary-random-aim', x: 800, y: 150, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	},
	143: {
		player: { x: 150, y: 250 },
		obstacles: [{ x: 380, y: 200, width: 30, height: 150 }],
		enemies: [
			{ type: 'simple-moving', x: 800, y: 250, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	// ---- Phase 5: multi-enemy + obstacles ----
	151: {
		player: { x: 120, y: 250 },
		obstacles: [{ x: 430, y: 160, width: 40, height: 180 }],
		enemies: [
			{ type: 'stationary', x: 780, y: 140, ammo: { type: 'basic', count: 1 } },
			{ type: 'stationary', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
		rules: { projectileBounces: false },
	},
	152: {
		player: { x: 120, y: 250 },
		obstacles: [
			{ x: 360, y: 120, width: 30, height: 260 },
			{ x: 640, y: 200, width: 30, height: 150 },
		],
		enemies: [
			{ type: 'stationary', x: 800, y: 150, ammo: { type: 'basic', count: 1 } },
			{ type: 'simple-moving', x: 800, y: 350, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	153: {
		player: { x: 100, y: 400 },
		obstacles: [{ x: 350, y: 200, width: 30, height: 200 }],
		enemies: [
			{ type: 'stationary-random-aim', x: 800, y: 120, ammo: { type: 'basic', count: 1 } },
			{ type: 'simple-moving', x: 850, y: 350, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	// ---- Phase 6: bouncing shots enabled ----
	161: {
		player: { x: 150, y: 250 },
		obstacles: [{ x: 450, y: 120, width: 35, height: 260 }],
		enemies: [{ type: 'stationary', x: 800, y: 250, ammo: { type: 'basic', count: 1 } }],
	},
	162: {
		player: { x: 130, y: 210 },
		obstacles: [{ x: 500, y: 150, width: 30, height: 200 }],
		enemies: [
			{ type: 'stationary', x: 780, y: 140, ammo: { type: 'basic', count: 1 } },
			{ type: 'stationary', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
	},
	163: {
		player: { x: 150, y: 300 },
		obstacles: [
			{ x: 350, y: 110, width: 200, height: 30 },
			{ x: 350, y: 360, width: 200, height: 30 },
		],
		enemies: [
			{ type: 'simple-moving', x: 800, y: 200, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
	},
	// ---- Phase 7: bomber introduction ----
	171: {
		player: { x: 140, y: 250 },
		obstacles: [{ x: 520, y: 120, width: 35, height: 260 }],
		enemies: [
			{
				type: 'bomber',
				x: 780,
				y: 250,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
		],
	},
	172: {
		player: { x: 140, y: 250 },
		obstacles: [{ x: 500, y: 150, width: 30, height: 200 }],
		enemies: [
			{
				type: 'bomber',
				x: 780,
				y: 350,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
			{ type: 'stationary', x: 820, y: 120, ammo: { type: 'basic', count: 1 } },
		],
	},
	173: {
		player: { x: 150, y: 260 },
		obstacles: [
			{ x: 320, y: 110, width: 250, height: 30 },
			{ x: 320, y: 360, width: 250, height: 30 },
		],
		enemies: [
			{
				type: 'bomber',
				x: 800,
				y: 200,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
		],
	},
	// ---- Phase 8: full mix (bombers + shooters + super-bombers) ----
	181: {
		player: { x: 140, y: 250 },
		obstacles: [{ x: 500, y: 150, width: 30, height: 250 }],
		enemies: [
			{
				type: 'bomber',
				x: 800,
				y: 150,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
			{ type: 'stationary-random-aim', x: 860, y: 350, ammo: { type: 'basic', count: 1 } },
		],
	},
	182: {
		player: { x: 130, y: 210 },
		obstacles: [
			{ x: 300, y: 120, width: 30, height: 260 },
			{ x: 640, y: 150, width: 30, height: 200 },
		],
		enemies: [
			{
				type: 'super-bomber',
				x: 820,
				y: 150,
				ammo: { type: 'super', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
			{ type: 'stationary', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
	},
	183: {
		player: { x: 150, y: 260 },
		obstacles: [
			{ x: 340, y: 110, width: 280, height: 30 },
			{ x: 340, y: 360, width: 280, height: 30 },
		],
		enemies: [
			{
				type: 'super-bomber',
				x: 800,
				y: 140,
				ammo: { type: 'super', count: 0 },
				bombs: { type: 'love', count: 1 },
				navigator: { type: 'astar', aggressionFactor: 4 },
			},
			{ type: 'stationary-random-aim', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
	},
};

function resolveScenarioConfig(scenarioId: number): LevelConfig {
	if (scenarioId in TRAINING_SCENARIOS) {
		return TRAINING_SCENARIOS[scenarioId];
	}
	const builtIn = LEVEL_CONFIGS[scenarioId - 1];
	if (!builtIn) {
		throw new Error(`Unknown scenario id: ${scenarioId}`);
	}
	return builtIn;
}

// ---- Observation normalization (port of treads_env.py _normalize_obs) ----
function normalizeObs(obs: TankObservation): number[] {
	const result = new Array<number>(OBS_SIZE).fill(0);
	let idx = 0;

	const s = obs.self;
	const sx = s.x + s.size / 2;
	const sy = s.y + s.size / 2;
	const arenaDiag = ARENA_DIAGONAL;
	const aliveEnemies = obs.enemies.filter((e) => !e.destroyed);

	// ── Self features (SELF_DIM = 12) ──
	result[idx] = s.x / ARENA_WIDTH;
	result[idx + 1] = s.y / ARENA_HEIGHT;
	result[idx + 2] = s.aimAngle / (2 * Math.PI) + 0.5; // Fix 1: map [-π,π] → [0,1]
	result[idx + 3] = s.speed / 100.0;
	// LOS to nearest alive enemy (1.0 = clear shot, 0.0 = blocked by obstacle)
	let hasLOS = 0.0;
	if (aliveEnemies.length > 0) {
		let losTarget = aliveEnemies[0];
		let losDistSq = Infinity;
		for (const e of aliveEnemies) {
			const dx = e.x + e.size / 2 - sx;
			const dy = e.y + e.size / 2 - sy;
			const dSq = dx * dx + dy * dy;
			if (dSq < losDistSq) {
				losDistSq = dSq;
				losTarget = e;
			}
		}
		hasLOS = hasLineOfSight(sx, sy, losTarget.x + losTarget.size / 2, losTarget.y + losTarget.size / 2, obs.obstacles)
			? 1.0
			: 0.0;
	}
	result[idx + 4] = hasLOS;
	result[idx + 5] = s.wasLastMoveBlocked ? 1.0 : 0.0;
	result[idx + 6] = Math.min(s.invulnerabilityTicksRemaining / 8.0, 1.0);
	result[idx + 7] = Math.min(obs.tick / TICK_NORM_TICKS, 1.0);
	result[idx + 8] = s.health / Math.max(s.maxHealth, 1);

	// Derived aim features (relative to nearest enemy)
	if (aliveEnemies.length > 0) {
		let nearest = aliveEnemies[0];
		let nearestDistSq = Infinity;
		for (const e of aliveEnemies) {
			const dx = e.x + e.size / 2 - sx;
			const dy = e.y + e.size / 2 - sy;
			const dSq = dx * dx + dy * dy;
			if (dSq < nearestDistSq) {
				nearestDistSq = dSq;
				nearest = e;
			}
		}
		const ex = nearest.x + nearest.size / 2;
		const ey = nearest.y + nearest.size / 2;
		const angleToEnemy = Math.atan2(ey - sy, ex - sx);
		const distToEnemy = Math.sqrt(nearestDistSq);
		const aimAngle = s.aimAngle;
		const aimError = Math.atan2(Math.sin(aimAngle - angleToEnemy), Math.cos(aimAngle - angleToEnemy));
		result[idx + 9] = angleToEnemy / (2 * Math.PI) + 0.5;
		result[idx + 10] = Math.min(distToEnemy / arenaDiag, 1.0);
		result[idx + 11] = (aimError / Math.PI) * 0.5 + 0.5;
	} else {
		result[idx + 9] = 0.5;
		result[idx + 10] = 0.0;
		result[idx + 11] = 0.5;
	}
	idx += SELF_DIM;

	// ── Enemies (ENEMY_DIM = 9, player-relative positions) ──
	const enemies = aliveEnemies.slice().sort((a, b) => {
		const dxA = a.x + a.size / 2 - sx;
		const dyA = a.y + a.size / 2 - sy;
		const dxB = b.x + b.size / 2 - sx;
		const dyB = b.y + b.size / 2 - sy;
		return dxA * dxA + dyA * dyA - (dxB * dxB + dyB * dyB);
	});
	for (let i = 0; i < MAX_ENEMIES; i++) {
		if (i < enemies.length) {
			const e = enemies[i];
			const ecx = e.x + e.size / 2;
			const ecy = e.y + e.size / 2;
			result[idx] = ((ecx - sx) / arenaDiag) * 0.5 + 0.5; // player-relative dx
			result[idx + 1] = ((ecy - sy) / arenaDiag) * 0.5 + 0.5; // player-relative dy
			result[idx + 2] = e.aimAngle / (2 * Math.PI) + 0.5; // angle normalization
			result[idx + 3] = e.speed / 100.0;
			result[idx + 4] = e.bombType ? 1.0 : 0.0;
			result[idx + 5] = e.health / Math.max(e.maxHealth, 1);
			// aimed_at_me — how much enemy turret points toward player (1=direct, 0=away)
			const angleFromEnemyToPlayer = Math.atan2(sy - ecy, sx - ecx);
			const enemyAimError = Math.atan2(
				Math.sin(e.aimAngle - angleFromEnemyToPlayer),
				Math.cos(e.aimAngle - angleFromEnemyToPlayer)
			);
			result[idx + 6] = 1.0 - Math.abs(enemyAimError) / Math.PI;
			// ammoThreat — 0=none, 0.5=basic, 1.0=super
			result[idx + 7] = e.maxAmmo > 0 ? (e.ammoType === 'super' ? 1.0 : 0.5) : 0.0;
			// isApproaching — dot(moveDir, enemyToPlayer) mapped to [0,1]
			const moveDir = moveIntentToDir(e.lastMoveIntent);
			if (moveDir[0] !== 0 || moveDir[1] !== 0) {
				const toPlayerDx = sx - ecx;
				const toPlayerDy = sy - ecy;
				const toPlayerDist = Math.sqrt(toPlayerDx * toPlayerDx + toPlayerDy * toPlayerDy);
				if (toPlayerDist > 1e-6) {
					const dot = (moveDir[0] * toPlayerDx + moveDir[1] * toPlayerDy) / toPlayerDist;
					result[idx + 8] = dot * 0.5 + 0.5; // [-1,1] → [0,1]
				} else {
					result[idx + 8] = 0.5;
				}
			} else {
				result[idx + 8] = 0.5; // stationary → neutral
			}
		}
		idx += ENEMY_DIM;
	}

	// ── Projectiles (PROJ_DIM = 5, player-relative positions) ──
	const projectiles = obs.projectiles.slice().sort((a, b) => {
		const dxA = a.x - sx;
		const dyA = a.y - sy;
		const dxB = b.x - sx;
		const dyB = b.y - sy;
		return dxA * dxA + dyA * dyA - (dxB * dxB + dyB * dyB);
	});
	for (let i = 0; i < MAX_PROJECTILES; i++) {
		if (i < projectiles.length) {
			const p = projectiles[i];
			result[idx] = ((p.x - sx) / arenaDiag) * 0.5 + 0.5; // Fix 3: player-relative
			result[idx + 1] = ((p.y - sy) / arenaDiag) * 0.5 + 0.5; // Fix 3: player-relative
			result[idx + 2] = (p.vx / PROJECTILE_SPEED_NORM) * 0.5 + 0.5;
			result[idx + 3] = (p.vy / PROJECTILE_SPEED_NORM) * 0.5 + 0.5;
			result[idx + 4] = p.team === 'enemy' ? 1.0 : 0.0;
		}
		idx += PROJ_DIM;
	}

	// ── Obstacles (OBS_DIM = 4, player-relative center positions) ──
	const obstacles = obs.obstacles.slice().sort((a, b) => {
		const dxA = a.x + a.width / 2 - sx;
		const dyA = a.y + a.height / 2 - sy;
		const dxB = b.x + b.width / 2 - sx;
		const dyB = b.y + b.height / 2 - sy;
		return dxA * dxA + dyA * dyA - (dxB * dxB + dyB * dyB);
	});
	for (let i = 0; i < MAX_OBSTACLES; i++) {
		if (i < obstacles.length) {
			const o = obstacles[i];
			result[idx] = ((o.x + o.width / 2 - sx) / arenaDiag) * 0.5 + 0.5; // Fix 3: player-relative center
			result[idx + 1] = ((o.y + o.height / 2 - sy) / arenaDiag) * 0.5 + 0.5;
			result[idx + 2] = o.width / ARENA_WIDTH;
			result[idx + 3] = o.height / ARENA_HEIGHT;
		}
		idx += OBS_DIM;
	}

	// ── Bombs (BOMB_DIM = 5, player-relative positions) ──
	const bombs = obs.bombs.slice().sort((a, b) => {
		const dxA = a.x - sx;
		const dyA = a.y - sy;
		const dxB = b.x - sx;
		const dyB = b.y - sy;
		return dxA * dxA + dyA * dyA - (dxB * dxB + dyB * dyB);
	});
	for (let i = 0; i < MAX_BOMBS; i++) {
		if (i < bombs.length) {
			const b = bombs[i];
			result[idx] = ((b.x - sx) / arenaDiag) * 0.5 + 0.5; // Fix 3: player-relative
			result[idx + 1] = ((b.y - sy) / arenaDiag) * 0.5 + 0.5; // Fix 3: player-relative
			result[idx + 2] = Math.min(b.fuseTicksRemaining / MAX_FUSE_TICKS, 1.0);
			result[idx + 3] = Math.min(b.blastRadius / MAX_BLAST_RADIUS, 1.0);
			result[idx + 4] = b.team === 'enemy' ? 1.0 : 0.0;
		}
		idx += BOMB_DIM;
	}

	// ── Summary features (SUMMARY_DIM = 6) — entity counts + distance cues ──
	result[idx] = Math.min(aliveEnemies.length / MAX_ENEMIES, 1.0);
	if (aliveEnemies.length > 0) {
		let farthestEnemyDistSq = 0;
		for (const e of aliveEnemies) {
			const dx = e.x + e.size / 2 - sx;
			const dy = e.y + e.size / 2 - sy;
			const dSq = dx * dx + dy * dy;
			if (dSq > farthestEnemyDistSq) farthestEnemyDistSq = dSq;
		}
		result[idx + 1] = Math.min(Math.sqrt(farthestEnemyDistSq) / arenaDiag, 1.0);
	} else {
		result[idx + 1] = 0.0;
	}
	result[idx + 2] = Math.min(obs.projectiles.length / MAX_PROJECTILES, 1.0);
	result[idx + 3] = Math.min(obs.bombs.length / MAX_BOMBS, 1.0);
	const enemyBombs = obs.bombs.filter((b) => b.team === 'enemy');
	if (enemyBombs.length > 0) {
		let closestBombDistSq = Infinity;
		for (const b of enemyBombs) {
			const dx = b.x - sx;
			const dy = b.y - sy;
			const dSq = dx * dx + dy * dy;
			if (dSq < closestBombDistSq) closestBombDistSq = dSq;
		}
		result[idx + 4] = Math.min(Math.sqrt(closestBombDistSq) / arenaDiag, 1.0);
	} else {
		result[idx + 4] = 1.0;
	}
	if (obs.projectiles.length > 0) {
		let farthestProjDistSq = 0;
		for (const p of obs.projectiles) {
			const dx = p.x - sx;
			const dy = p.y - sy;
			const dSq = dx * dx + dy * dy;
			if (dSq > farthestProjDistSq) farthestProjDistSq = dSq;
		}
		result[idx + 5] = Math.min(Math.sqrt(farthestProjDistSq) / arenaDiag, 1.0);
	} else {
		result[idx + 5] = 0.0;
	}

	// Clip to [0, 1]
	for (let i = 0; i < OBS_SIZE; i++) {
		result[i] = Math.max(0, Math.min(1, result[i]));
	}
	return result;
}

// ---- Reward computation (port of treads_env.py _compute_reward) ----
interface RewardTracker {
	prevEnemyAliveCount: number;
	prevEnemyHealthTotal: number;
	prevSelfHealth: number;
	prevEnemyPathDist: number;
	prevEnemyProjDist: number;
	approachTargetId: string;
}

interface RewardBreakdown {
	tick: number;
	hit: number;
	hurt: number;
	kill: number;
	death: number;
	terminalWin: number;
	terminalLoss: number;
	timeout: number;
	approach: number;
	dodge: number;
}

function createRewardBreakdown(): RewardBreakdown {
	return {
		tick: 0,
		hit: 0,
		hurt: 0,
		kill: 0,
		death: 0,
		terminalWin: 0,
		terminalLoss: 0,
		timeout: 0,
		approach: 0,
		dodge: 0,
	};
}

function mergeRewardBreakdown(target: RewardBreakdown, add: RewardBreakdown): void {
	target.tick += add.tick;
	target.hit += add.hit;
	target.hurt += add.hurt;
	target.kill += add.kill;
	target.death += add.death;
	target.terminalWin += add.terminalWin;
	target.terminalLoss += add.terminalLoss;
	target.timeout += add.timeout;
	target.approach += add.approach;
	target.dodge += add.dodge;
}

/**
 * Decodes NN action signals into a TankAction.
 * Action layout: [move_x, move_y, aim_signal, fire_signal, bomb_signal]
 *
 * Movement: 2D continuous (move_x, move_y) mapped to 9 discrete intents.
 *   Geometrically meaningful: nearby values → nearby directions.
 *   Center (0,0) → 'none'; thresholds at ±0.33 for cardinal/diagonal.
 *
 * Aim: aim_signal ∈ [-1, 1] is an offset from angle-to-nearest-enemy.
 *   aim_signal = 0  → aimed directly at enemy
 *   aim_signal = ±1 → aimed opposite the nearest-enemy angle (±pi)
 */
function decodeActionSignal(signal: number[], rawObs: TankObservation): { decoded: TankAction; clamped: number[] } {
	const clamped = signal.map((v) => Math.max(-1, Math.min(1, v)));

	// 2D movement decode: (move_x, move_y) → MoveIntent
	const mx = clamped[0];
	const my = clamped[1];
	const MOVE_DEAD_ZONE = 0.33;
	const goE = mx > MOVE_DEAD_ZONE;
	const goW = mx < -MOVE_DEAD_ZONE;
	const goS = my > MOVE_DEAD_ZONE;
	const goN = my < -MOVE_DEAD_ZONE;
	let moveIntent: MoveIntent;
	if (goN && goE) moveIntent = 'ne';
	else if (goN && goW) moveIntent = 'nw';
	else if (goS && goE) moveIntent = 'se';
	else if (goS && goW) moveIntent = 'sw';
	else if (goN) moveIntent = 'n';
	else if (goS) moveIntent = 's';
	else if (goE) moveIntent = 'e';
	else if (goW) moveIntent = 'w';
	else moveIntent = 'none';

	// Enemy-relative aim encoding
	const s = rawObs.self;
	const sx = s.x + s.size / 2;
	const sy = s.y + s.size / 2;
	const aliveEnemies = rawObs.enemies.filter((e) => !e.destroyed);
	let aimAngle: number;
	if (aliveEnemies.length > 0) {
		let nearest = aliveEnemies[0];
		let nearestDistSq = Infinity;
		for (const e of aliveEnemies) {
			const dx = e.x + e.size / 2 - sx;
			const dy = e.y + e.size / 2 - sy;
			const dSq = dx * dx + dy * dy;
			if (dSq < nearestDistSq) {
				nearestDistSq = dSq;
				nearest = e;
			}
		}
		const ex = nearest.x + nearest.size / 2;
		const ey = nearest.y + nearest.size / 2;
		const angleToEnemy = Math.atan2(ey - sy, ex - sx);
		aimAngle = angleToEnemy + clamped[2] * AIM_OFFSET_LIMIT;
	} else {
		// No living enemy: hold current aim
		aimAngle = s.aimAngle;
	}

	return {
		clamped,
		decoded: {
			move: moveIntent,
			aimAngle,
			fire: clamped[3] > FIRE_THRESHOLD,
			plantBomb: clamped[4] > BOMB_THRESHOLD,
		},
	};
}

function initRewardTracker(obs: TankObservation, navPlanner: NavigationPlanner | null): RewardTracker {
	const alive = obs.enemies.filter((e) => !e.destroyed);
	const sx = obs.self.x + obs.self.size / 2;
	const sy = obs.self.y + obs.self.size / 2;
	let initialDist = 0;
	let nearestId = '';
	if (alive.length > 0) {
		let nearestEnemy = alive[0];
		let nearestDistSq = Infinity;
		for (const e of alive) {
			const dx = e.x + e.size / 2 - sx;
			const dy = e.y + e.size / 2 - sy;
			const dSq = dx * dx + dy * dy;
			if (dSq < nearestDistSq) {
				nearestDistSq = dSq;
				nearestEnemy = e;
			}
		}
		nearestId = nearestEnemy.id;
		if (navPlanner) {
			initialDist = navPlanner.getPathDistance(
				sx,
				sy,
				nearestEnemy.x + nearestEnemy.size / 2,
				nearestEnemy.y + nearestEnemy.size / 2
			);
		} else {
			initialDist = Math.sqrt(nearestDistSq);
		}
	}
	// Initial closest enemy projectile distance for dodge shaping
	const enemyProjs = obs.projectiles.filter((p) => p.team === 'enemy');
	let initialProjDist = ARENA_DIAGONAL;
	if (enemyProjs.length > 0) {
		let closestProjDistSq = Infinity;
		for (const p of enemyProjs) {
			const dx = p.x - sx;
			const dy = p.y - sy;
			const dSq = dx * dx + dy * dy;
			if (dSq < closestProjDistSq) closestProjDistSq = dSq;
		}
		initialProjDist = Math.sqrt(closestProjDistSq);
	}

	return {
		prevEnemyAliveCount: alive.length,
		prevEnemyHealthTotal: alive.reduce((total, enemy) => total + enemy.health, 0),
		prevSelfHealth: obs.self.health,
		prevEnemyPathDist: initialDist,
		prevEnemyProjDist: initialProjDist,
		approachTargetId: nearestId,
	};
}

function computeSteppingReward(
	state: DeepReadonly<GameState>,
	tracker: RewardTracker,
	decodedAction: TankAction,
	_rawObs: TankObservation,
	shapingScale: number,
	navPlanner: NavigationPlanner | null
): { reward: number; breakdown: RewardBreakdown } {
	let reward = STEP_PENALTY;
	const breakdown = createRewardBreakdown();
	breakdown.tick += STEP_PENALTY;

	const player = state.tanks.find((t) => t.id === PLAYER_TANK_ID);
	if (!player) return { reward: 0, breakdown };
	const enemies = state.tanks.filter((t) => t.team === 'enemy');
	const aliveEnemies = enemies.filter((e) => !e.destroyed);
	const enemyHealthTotal = aliveEnemies.reduce((total, enemy) => total + enemy.health, 0);
	const damageDealt = tracker.prevEnemyHealthTotal - enemyHealthTotal;
	if (damageDealt > 0) {
		const value = HIT_REWARD * damageDealt;
		reward += value;
		breakdown.hit += value;
	}
	tracker.prevEnemyHealthTotal = enemyHealthTotal;

	const selfDamageTaken = tracker.prevSelfHealth - player.health;
	if (selfDamageTaken > 0) {
		const value = TOOK_DAMAGE_PENALTY * selfDamageTaken;
		reward += value;
		breakdown.hurt += value;
	}
	tracker.prevSelfHealth = player.health;

	// Kill event
	const enemiesKilled = tracker.prevEnemyAliveCount - aliveEnemies.length;
	if (enemiesKilled > 0) {
		const value = KILL_REWARD * enemiesKilled;
		reward += value;
		breakdown.kill += value;
		// Reset approach tracking: the nearest enemy changed, so prevDist
		// was computed against a now-dead enemy. Skip approach delta this tick.
		tracker.prevEnemyPathDist = -1;
		tracker.approachTargetId = '';
	}
	tracker.prevEnemyAliveCount = aliveEnemies.length;

	// ── Shaping: potential-based approach reward (Ng et al. 1999) ──
	// Φ(s) = -dist/ARENA_DIAGONAL, reward = γ·Φ(s') - Φ(s) ≈ Φ(s') - Φ(s) since γ≈1
	// Uses A* pathfinding distance so flanking around obstacles is rewarded correctly.
	if (aliveEnemies.length > 0) {
		const px = player.x + player.size / 2;
		const py = player.y + player.size / 2;
		let nearestEnemy = aliveEnemies[0];
		let nearestDistSq = Infinity;
		for (const e of aliveEnemies) {
			const dx = e.x + e.size / 2 - px;
			const dy = e.y + e.size / 2 - py;
			const dSq = dx * dx + dy * dy;
			if (dSq < nearestDistSq) {
				nearestDistSq = dSq;
				nearestEnemy = e;
			}
		}
		let currDist: number;
		if (navPlanner) {
			currDist = navPlanner.getPathDistance(
				px,
				py,
				nearestEnemy.x + nearestEnemy.size / 2,
				nearestEnemy.y + nearestEnemy.size / 2
			);
		} else {
			currDist = Math.sqrt(nearestDistSq);
		}
		const prevDist = tracker.prevEnemyPathDist;
		// Skip approach delta when target switched (kill or natural retarget)
		const targetSwitched = nearestEnemy.id !== tracker.approachTargetId;
		if (prevDist >= 0 && !targetSwitched) {
			const approachReward = (APPROACH_SCALE * (prevDist - currDist) * shapingScale) / ARENA_DIAGONAL;
			if (Math.abs(approachReward) > 1e-8) {
				reward += approachReward;
				breakdown.approach += approachReward;
			}
		}
		tracker.prevEnemyPathDist = currDist;
		tracker.approachTargetId = nearestEnemy.id;
	}

	// ── Shaping: dodge reward — potential-based on distance to nearest enemy projectile ──
	// Φ(s) = dist/ARENA_DIAGONAL, reward = Φ(s') - Φ(s) — moving away from incoming fire is positive
	const enemyProjectiles = state.projectiles.filter((p) => p.team === 'enemy');
	let currProjDist = ARENA_DIAGONAL;
	if (enemyProjectiles.length > 0) {
		const px = player.x + player.size / 2;
		const py2 = player.y + player.size / 2;
		let closestProjDistSq = Infinity;
		for (const p of enemyProjectiles) {
			const dx = p.x - px;
			const dy = p.y - py2;
			const dSq = dx * dx + dy * dy;
			if (dSq < closestProjDistSq) closestProjDistSq = dSq;
		}
		currProjDist = Math.sqrt(closestProjDistSq);
	}
	const dodgeReward = (DODGE_SCALE * (currProjDist - tracker.prevEnemyProjDist) * shapingScale) / ARENA_DIAGONAL;
	if (Math.abs(dodgeReward) > 1e-8) {
		reward += dodgeReward;
		breakdown.dodge += dodgeReward;
	}
	tracker.prevEnemyProjDist = currProjDist;

	return { reward, breakdown };
}

// ---- RL Controller: captures obs/action/value/logprob during act() ----
class RLController implements TankController {
	private mlp: PolicyMLP;
	public lastNormalizedObs: number[] = [];
	public lastAction: number[] = [];
	public lastDecodedAction: TankAction = { move: 'none', aimAngle: 0, fire: false, plantBomb: false };
	public lastValue = 0;
	public lastLogProb = 0;
	public lastRawObs: TankObservation | null = null;
	// Fix 1: pendingDecodedAction removed — act() now executes the sampled action immediately,
	// so the obs/action/logprob stored in the rollout buffer are always aligned.

	constructor(mlp: PolicyMLP) {
		this.mlp = mlp;
	}

	reset(_initial: MatchInit): void {
		this.lastDecodedAction = { move: 'none', aimAngle: 0, fire: false, plantBomb: false };
	}

	act(obs: TankObservation): TankAction {
		this.lastRawObs = obs;
		const normalized = normalizeObs(obs);
		this.lastNormalizedObs = normalized;

		const { actionMean, value } = this.mlp.forward(normalized);
		const { action, logProb } = this.mlp.sampleGaussianAction(actionMean);

		this.lastAction = action;
		this.lastValue = value;
		this.lastLogProb = logProb;

		// Fix 1: decode and execute the SAME action that gets stored in the rollout buffer.
		// Previously a one-tick pending mechanism caused obs→action mislabeling.
		const { decoded } = decodeActionSignal(action, obs);
		this.lastDecodedAction = decoded;
		return decoded;
	}

	setMLP(mlp: PolicyMLP): void {
		this.mlp = mlp;
	}
}

// ---- Rollout collection ----
interface RolloutData {
	type: 'rollout';
	n_steps: number;
	obs: number[][];
	actions: number[][];
	rewards: number[];
	episode_starts: number[];
	values: number[];
	log_probs: number[];
	last_obs: number[];
	last_done: boolean;
	last_value: number;
	episode_rewards: number[];
	episode_lengths: number[];
	episode_wins: (0 | 1)[];
	episode_levels: number[];
	episode_reward_breakdowns: RewardBreakdown[];
}

function applySpawnJitter(config: LevelConfig, seed: number): LevelConfig {
	const JITTER = 40;
	const TANK_SIZE = 30;
	const rng = new SeededRandom(seed * 7919 + 13);
	const jitter = () => rng.nextRange(-JITTER, JITTER);
	const clampX = (x: number) => Math.max(0, Math.min(ARENA_WIDTH - TANK_SIZE, x));
	const clampY = (y: number) => Math.max(0, Math.min(ARENA_HEIGHT - TANK_SIZE, y));
	const jittered: LevelConfig = {
		...config,
	};

	if (config.player) {
		jittered.player = {
			x: clampX(config.player.x + jitter()),
			y: clampY(config.player.y + jitter()),
		};
	}

	if (config.enemies) {
		jittered.enemies = config.enemies.map((enemy) => ({
			...enemy,
			x: clampX(enemy.x + jitter()),
			y: clampY(enemy.y + jitter()),
		}));
	}

	if (config.tanks) {
		jittered.tanks = config.tanks.map((tank) => ({
			...tank,
			x: clampX(tank.x + jitter()),
			y: clampY(tank.y + jitter()),
		}));
	}

	return jittered;
}

function collectRollout(
	mlp: PolicyMLP,
	nSteps: number,
	levels: number[],
	maxTicks: number,
	seedStart: number,
	replayEveryEpisodes: number,
	replayDir: string,
	workerId: number,
	shapingScale: number,
	startEpisode: number
): RolloutData {
	const obs: number[][] = [];
	const actions: number[][] = [];
	const rewards: number[] = [];
	const episodeStarts: number[] = [];
	const values: number[] = [];
	const logProbs: number[] = [];
	const episodeRewards: number[] = [];
	const episodeLengths: number[] = [];
	const episodeWins: (0 | 1)[] = [];
	const episodeLevels: number[] = [];
	const episodeRewardBreakdowns: RewardBreakdown[] = [];

	const rlController = new RLController(mlp);
	const levelRng = new SeededRandom(seedStart * 31337 + 7);
	let seed = seedStart;
	let tracker: RewardTracker | null = null;
	let episodeTick = 0;
	let episodeReward = 0;
	let isNewEpisode = true;
	let currentLevel = levels[0];
	let episodeBreakdown = createRewardBreakdown();
	let replayRecorder: ReplayRecorder | null = null;
	let completedEpisodes = 0;
	let navPlanner: NavigationPlanner | null = null;

	function resetEpisode(): Simulation {
		// Pick scenario uniformly at random from the active curriculum pool
		currentLevel = levels[levelRng.nextInt(0, levels.length - 1)];
		const baseConfig = resolveScenarioConfig(currentLevel);
		const levelConfig = applySpawnJitter(baseConfig, seed);
		const initialState = createInitialGameState(levelConfig, seed);
		const controllers = createDefaultControllers(levelConfig);
		controllers[PLAYER_TANK_ID] = rlController;
		replayRecorder = new ReplayRecorder(levelConfig, seed);
		// Create nav planner for A*-based approach distance (handles obstacles)
		navPlanner = new NavigationPlanner(initialState.arena, initialState.obstacles);
		tracker = null;
		episodeTick = 0;
		episodeReward = 0;
		episodeBreakdown = createRewardBreakdown();
		isNewEpisode = true;
		seed++;
		return new Simulation(initialState, controllers, { debugFreeze: false });
	}

	let sim = resetEpisode();

	for (let step = 0; step < nSteps; step++) {
		// Record episode_start for this step
		episodeStarts.push(isNewEpisode ? 1 : 0);
		isNewEpisode = false;

		// Step the simulation. This calls RLController.act() for the player,
		// which captures obs, action, value, log_prob.
		const stepResult = sim.step();
		if (replayRecorder !== null) {
			(replayRecorder as ReplayRecorder).record(stepResult);
		}

		// Read captured data from controller
		obs.push(rlController.lastNormalizedObs);
		actions.push(rlController.lastAction);
		values.push(rlController.lastValue);
		logProbs.push(rlController.lastLogProb);

		// Initialize reward tracker from first observation
		if (tracker === null) {
			if (!rlController.lastRawObs) throw new Error('lastRawObs is null after step');
			tracker = initRewardTracker(rlController.lastRawObs, navPlanner);
		}

		// Check post-step state for reward computation
		const state = sim.getState();
		const player = state.tanks.find((t) => t.id === PLAYER_TANK_ID);
		const lastDecodedAction = rlController.lastDecodedAction;
		if (!rlController.lastRawObs) throw new Error('lastRawObs is null during reward computation');
		const lastRawObs = rlController.lastRawObs;
		episodeTick++;

		let done = false;
		let stepReward: number;

		if (state.status === 'player_win') {
			const stepping = computeSteppingReward(state, tracker, lastDecodedAction, lastRawObs, shapingScale, navPlanner);
			stepReward = stepping.reward + TERMINAL_WIN_REWARD;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
			episodeBreakdown.terminalWin += TERMINAL_WIN_REWARD;
			done = true;
		} else if (state.status === 'enemy_win' || player?.destroyed) {
			const stepping = computeSteppingReward(state, tracker, lastDecodedAction, lastRawObs, shapingScale, navPlanner);
			stepReward = stepping.reward + DEATH_REWARD + TERMINAL_LOSS_REWARD;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
			episodeBreakdown.death += DEATH_REWARD;
			episodeBreakdown.terminalLoss += TERMINAL_LOSS_REWARD;
			done = true;
		} else if (episodeTick >= maxTicks) {
			const stepping = computeSteppingReward(state, tracker, lastDecodedAction, lastRawObs, shapingScale, navPlanner);
			stepReward = stepping.reward + TIMEOUT_REWARD;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
			episodeBreakdown.timeout += TIMEOUT_REWARD;
			done = true;
		} else {
			const stepping = computeSteppingReward(state, tracker, lastDecodedAction, lastRawObs, shapingScale, navPlanner);
			stepReward = stepping.reward;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
		}

		rewards.push(stepReward);
		episodeReward += stepReward;

		if (done) {
			completedEpisodes += 1;
			episodeRewards.push(episodeReward);
			episodeLengths.push(episodeTick);
			episodeWins.push(state.status === 'player_win' ? 1 : 0);
			episodeLevels.push(currentLevel);
			episodeRewardBreakdowns.push(episodeBreakdown);
			const absoluteEpisode = startEpisode + completedEpisodes;
			if (replayEveryEpisodes > 0 && absoluteEpisode % replayEveryEpisodes === 0 && replayRecorder !== null) {
				fs.mkdirSync(replayDir, { recursive: true });
				const replayPath = path.join(
					replayDir,
					`episode_${absoluteEpisode}_W${workerId}_L${currentLevel}_S${seed - 1}_${state.status}.json`
				);
				fs.writeFileSync(replayPath, JSON.stringify((replayRecorder as ReplayRecorder).toJSON()));
			}
			sim = resetEpisode();
		}
	}

	// Compute last_value for bootstrapping (value of the last observation)
	// If the last step was terminal, last_value = 0. Otherwise, get current obs value.
	const lastDone = episodeStarts.length > 0 && isNewEpisode;
	let lastValue: number;
	let lastObs: number[];

	if (lastDone) {
		// Last step ended an episode and we just reset — value is 0
		lastValue = 0;
		// Get obs from the new episode for last_obs
		const state = sim.getState();
		const playerTank = state.tanks.find((t) => t.id === PLAYER_TANK_ID);
		if (!playerTank) throw new Error('Player tank not found in reset state');
		const dummyObs: TankObservation = {
			tick: state.tick,
			self: JSON.parse(JSON.stringify(playerTank)),
			allies: state.tanks
				.filter((t) => t.team === playerTank.team && t.id !== playerTank.id)
				.map((t) => JSON.parse(JSON.stringify(t))),
			enemies: state.tanks.filter((t) => t.team !== playerTank.team).map((t) => JSON.parse(JSON.stringify(t))),
			projectiles: state.projectiles.map((p) => JSON.parse(JSON.stringify(p))),
			bombs: state.bombs.map((b) => JSON.parse(JSON.stringify(b))),
			obstacles: state.obstacles.map((o) => JSON.parse(JSON.stringify(o))),
			arena: { ...state.arena },
		};
		lastObs = normalizeObs(dummyObs);
	} else {
		// Mid-episode: use the last controller obs and value
		lastObs = rlController.lastNormalizedObs;
		const { value } = mlp.forward(lastObs);
		lastValue = value;
	}

	return {
		type: 'rollout',
		n_steps: nSteps,
		obs,
		actions,
		rewards,
		episode_starts: episodeStarts,
		values,
		log_probs: logProbs,
		last_obs: lastObs,
		last_done: lastDone,
		last_value: lastValue,
		episode_rewards: episodeRewards,
		episode_lengths: episodeLengths,
		episode_wins: episodeWins,
		episode_levels: episodeLevels,
		episode_reward_breakdowns: episodeRewardBreakdowns,
	};
}

// ---- stdin/stdout communication ----
function writeLine(obj: unknown): void {
	process.stdout.write(JSON.stringify(obj) + '\n');
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
const lineQueue: string[] = [];
let lineResolve: (() => void) | null = null;

rl.on('line', (line: string) => {
	lineQueue.push(line);
	if (lineResolve) {
		const resolve = lineResolve;
		lineResolve = null;
		resolve();
	}
});

rl.on('close', () => {
	process.exit(0);
});

function readLine(): Promise<string> {
	if (lineQueue.length > 0) {
		return Promise.resolve(lineQueue.shift() ?? '');
	}
	return new Promise<string>((resolve) => {
		lineResolve = () => resolve(lineQueue.shift() ?? '');
	});
}

// ---- Main loop ----
async function main(): Promise<void> {
	let mlp: PolicyMLP | null = null;
	let workerTotalEpisodes = 0;

	writeLine({ type: 'ready' });

	while (true) {
		const line = await readLine();
		let cmd: Record<string, unknown>;
		try {
			cmd = JSON.parse(line);
		} catch {
			writeLine({ type: 'error', message: 'Invalid JSON command received by rollout worker' });
			continue;
		}

		if (cmd.type === 'set_weights') {
			const stateDict = cmd.state_dict as Record<string, number[][] | number[]>;
			const weights = parseWeightsFromStateDict(stateDict);
			mlp = new PolicyMLP(weights);
			writeLine({ type: 'weights_set' });
		} else if (cmd.type === 'set_weights_from_file') {
			const filePath = String(cmd.path ?? '');
			if (!filePath) {
				writeLine({ type: 'error', message: 'Missing path for set_weights_from_file' });
				continue;
			}
			const raw = fs.readFileSync(filePath, 'utf8');
			const stateDict = JSON.parse(raw) as Record<string, number[][] | number[]>;
			const weights = parseWeightsFromStateDict(stateDict);
			mlp = new PolicyMLP(weights);
			writeLine({ type: 'weights_set' });
		} else if (cmd.type === 'collect') {
			if (!mlp) {
				writeLine({ type: 'error', message: 'Weights not set' });
				continue;
			}
			const nSteps = (cmd.n_steps as number) ?? 4096;
			const levels = (cmd.levels as number[]) ?? [(cmd.level as number) ?? 1];
			const maxTicks = (cmd.maxTicks as number) ?? 1800;
			TICK_NORM_TICKS = Math.max(1, Number((cmd.tickNormTicks as number) ?? maxTicks));
			const seedStart = (cmd.seedStart as number) ?? 0;
			const replayEveryEpisodes = (cmd.replayEveryEpisodes as number) ?? 0;
			const replayDir = (cmd.replayDir as string) ?? path.join(__dirname, '..', '..', 'training', 'output', 'replays');
			const workerId = (cmd.workerId as number) ?? 0;
			const shapingScale = Math.max(0, Math.min(1, (cmd.shapingScale as number) ?? 1.0));
			const rollout = collectRollout(
				mlp,
				nSteps,
				levels,
				maxTicks,
				seedStart,
				replayEveryEpisodes,
				replayDir,
				workerId,
				shapingScale,
				workerTotalEpisodes
			);
			workerTotalEpisodes += (rollout.episode_rewards as number[]).length;
			writeLine(rollout);
		} else if (cmd.type === 'test_forward') {
			if (!mlp) {
				writeLine({ type: 'error', message: 'Weights not set' });
				continue;
			}
			const observations = cmd.observations as number[][];
			const logits: number[][] = [];
			const vals: number[] = [];
			for (const obs of observations) {
				const result = mlp.forward(obs);
				logits.push(result.actionMean);
				vals.push(result.value);
			}
			writeLine({ type: 'test_result', logits, values: vals });
		} else if (cmd.type === 'exit') {
			process.exit(0);
		} else {
			writeLine({ type: 'error', message: `Unknown command type: ${String(cmd.type ?? 'undefined')}` });
		}
	}
}

main().catch((err) => {
	process.stderr.write(`Rollout worker error: ${err}\n`);
	process.exit(1);
});
