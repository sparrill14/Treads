/**
 * Hybrid Rollout Worker
 *
 * Runs the game simulation and neural network inference entirely in TypeScript.
 * Collects full rollout buffers (obs, action, reward, done, value, log_prob)
 * and bulk-transfers them to Python for PPO gradient updates.
 *
 * Transport:
 *   gRPC server with unary RPC methods for health, weight loading, rollout collection,
 *   and graceful shutdown.
 */

import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import * as fs from 'fs';
import * as path from 'path';

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
import { type LevelConfig } from '../src/game/LevelConfig';
import { NavigationPlanner } from '../src/game/navigation/NavigationPlanner';
import { ACTION_DIM, ACTION_HEAD_SIZES, PolicyMLP, parseWeightsFromStateDict } from './mlp-inference';
import { resolveScenarioConfig } from './training-scenarios';

const MOVE_INTENTS_BY_INDEX: MoveIntent[] = ['none', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];
const NUM_AIM_BINS = ACTION_HEAD_SIZES[1];
const AIM_BIN_RADIANS = (2 * Math.PI) / NUM_AIM_BINS;

type HealthRequest = Record<string, never>;
interface HealthResponse {
	ok: boolean;
	message: string;
}

interface SetWeightsFromFileRequest {
	path?: string;
}

interface SetWeightsFromFileResponse {
	ok: boolean;
}

interface CollectRequest {
	nSteps?: number;
	levels?: number[];
	maxTicks?: number;
	tickNormTicks?: number;
	seedStart?: number;
	replayEveryEpisodes?: number;
	replayDir?: string;
	workerId?: number;
	targetEpisodes?: number;
	shapingScale?: number;
	proceduralLevels?: boolean;
	difficultyBand?: number;
	playerMaxAmmo?: number;
	globalEpisodeOffset?: number;
	gamma?: number;
}

interface CollectResponse {
	rolloutJson: string;
}

type CloseRequest = Record<string, never>;
interface CloseResponse {
	closed: boolean;
}

const ROLLOUT_PROTO_PATH = path.join(__dirname, '..', '..', 'training', 'proto', 'treads.proto');
const rolloutPackageDef = protoLoader.loadSync(ROLLOUT_PROTO_PATH, {
	keepCase: false,
	longs: String,
	enums: String,
	defaults: true,
	oneofs: true,
});
const rolloutGrpcObj = grpc.loadPackageDefinition(rolloutPackageDef) as grpc.GrpcObject;
const rolloutPkg = rolloutGrpcObj.treads as grpc.GrpcObject;
const rolloutServiceDef = rolloutPkg.RolloutWorkerService as grpc.ServiceClientConstructor & {
	service: grpc.ServiceDefinition;
};

// ---- Constants matching treads_env.py ----
const ARENA_WIDTH = 1000.0;
const ARENA_HEIGHT = 500.0;
// Fix 3: caps raised to cover all 9 levels (max 5 enemies/4 obstacles in Level 7)
const MAX_ENEMIES = 6; // Level 7 has 5 enemies; +1 buffer
const MAX_PROJECTILES = 15; // Level 8: 3×3=9 super shots; generous buffer
const MAX_OBSTACLES = 5; // Level 7 has 4 obstacles; +1 buffer
const MAX_BOMBS = 6; // Level 6: 9 theoretical; cap at 6 live
const SELF_DIM = 15; // pos(2), aim, speed, LOS, blocked, invuln, tick, hp, ammoRatio, shotCD, bombRatio, aimFeatures(3)
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
const STEP_PENALTY = -0.001;
const HIT_REWARD = 0.3;
const TOOK_DAMAGE_PENALTY = -0.3;
const KILL_REWARD = 2.0;
const DEATH_REWARD = -2.0;
const TERMINAL_WIN_REWARD = 5.0;
const TERMINAL_LOSS_REWARD = -3.0;
const TIMEOUT_REWARD = -1.0;
// GAMMA is no longer needed in the worker (truncation bootstrap is now applied
// in Python AFTER reward normalization), but the request still carries it for
// forward-compat / debugging.
// Potential-based shaping scales. Increased 5x (0.15->0.75, 0.5->2.5) so the
// per-step shaping signal is meaningful relative to STEP_PENALTY and survives
// reward normalization (was effectively ~4e-4/step, now ~2e-3/step before
// normalization which is on the same order as sparse-event rewards).
const DODGE_SCALE = 0.75; // potential-based: reward for staying away from enemy projectiles
const APPROACH_SCALE = 2.5; // potential-based: reward for closing distance to nearest enemy
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

// ---- Observation normalization (port of treads_env.py _normalize_obs) ----
function normalizeObs(obs: TankObservation): number[] {
	const result = new Array<number>(OBS_SIZE).fill(0);
	let idx = 0;

	const s = obs.self;
	const sx = s.x + s.size / 2;
	const sy = s.y + s.size / 2;
	const arenaDiag = ARENA_DIAGONAL;
	const aliveEnemies = obs.enemies.filter((e) => !e.destroyed);

	// ── Self features (SELF_DIM = 15) ──
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

	// Resource features — ammo, cooldown, bombs
	result[idx + 9] = s.activeAmmo / Math.max(s.maxAmmo, 1); // ammo ratio (1=full, 0=empty)
	result[idx + 10] = s.shotCooldownTicks / Math.max(s.shotCooldownTicksOnFire, 1); // shot cooldown (0=ready, 1=just fired)
	result[idx + 11] = s.activeBombs / Math.max(s.maxBombs, 1); // bomb ratio (0=none/empty)

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
		result[idx + 12] = angleToEnemy / (2 * Math.PI) + 0.5;
		result[idx + 13] = Math.min(distToEnemy / arenaDiag, 1.0);
		result[idx + 14] = (aimError / Math.PI) * 0.5 + 0.5;
	} else {
		result[idx + 12] = 0.5;
		result[idx + 13] = 0.0;
		result[idx + 14] = 0.5;
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
	prevEnemyProjDist: number;
	prevEnemyDist: number;
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
 * Decode a MultiDiscrete([9, 16, 2, 2]) action vector into a TankAction.
 *   action[0] ∈ [0, 9)  → MoveIntent index
 *   action[1] ∈ [0, 16) → absolute aim bin (each = 22.5°)
 *   action[2] ∈ {0, 1}  → fire
 *   action[3] ∈ {0, 1}  → plant bomb
 */
function decodeMultiDiscreteAction(action: number[], rawObs: TankObservation): TankAction {
	if (action.length !== ACTION_DIM) {
		throw new Error(`Expected ${ACTION_DIM} action heads, got ${action.length}`);
	}
	const moveIdx = Math.max(0, Math.min(MOVE_INTENTS_BY_INDEX.length - 1, action[0] | 0));
	const aimBin = (((action[1] | 0) % NUM_AIM_BINS) + NUM_AIM_BINS) % NUM_AIM_BINS;
	const fire = (action[2] | 0) !== 0;
	const plantBomb = (action[3] | 0) !== 0;
	const aimAngle = aimBin * AIM_BIN_RADIANS;
	// rawObs is unused here but kept in signature for parity with prior decoder.
	void rawObs;
	return {
		move: MOVE_INTENTS_BY_INDEX[moveIdx],
		aimAngle,
		fire,
		plantBomb,
	};
}

function nearestEnemyDistance(obs: TankObservation, sx: number, sy: number): { dist: number; id: string } {
	const alive = obs.enemies.filter((e) => !e.destroyed);
	if (alive.length === 0) return { dist: 0, id: '' };
	let best = alive[0];
	let bestSq = Infinity;
	for (const e of alive) {
		const dx = e.x + e.size / 2 - sx;
		const dy = e.y + e.size / 2 - sy;
		const dSq = dx * dx + dy * dy;
		if (dSq < bestSq) {
			bestSq = dSq;
			best = e;
		}
	}
	return { dist: Math.sqrt(bestSq), id: String(best.id ?? '') };
}

function initRewardTracker(obs: TankObservation, _navPlanner: NavigationPlanner | null): RewardTracker {
	const alive = obs.enemies.filter((e) => !e.destroyed);
	const sx = obs.self.x + obs.self.size / 2;
	const sy = obs.self.y + obs.self.size / 2;

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

	const { dist: initialEnemyDist, id: initialEnemyId } = nearestEnemyDistance(obs, sx, sy);

	return {
		prevEnemyAliveCount: alive.length,
		prevEnemyHealthTotal: alive.reduce((total, enemy) => total + enemy.health, 0),
		prevSelfHealth: obs.self.health,
		prevEnemyProjDist: initialProjDist,
		prevEnemyDist: initialEnemyDist,
		approachTargetId: initialEnemyId,
	};
}

function computeSteppingReward(
	state: DeepReadonly<GameState>,
	tracker: RewardTracker,
	decodedAction: TankAction,
	_rawObs: TankObservation,
	shapingScale: number,
	_navPlanner: NavigationPlanner | null
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
		// Reset approach baseline so next-tick distance change isn't a huge jump
		tracker.prevEnemyDist = -1.0;
		tracker.approachTargetId = '';
	}
	tracker.prevEnemyAliveCount = aliveEnemies.length;

	// ── Shaping: approach reward — potential-based on Euclidean distance to nearest enemy ──
	// Φ(s) = -dist/ARENA_DIAGONAL, reward = γΦ(s') - Φ(s); approximated via
	//   APPROACH_SCALE * (prevDist - currDist) / ARENA_DIAGONAL
	// Only credited when the same enemy stays nearest across ticks (avoids reward jumps
	// when the nearest target changes due to kills or movement).
	const px = player.x + player.size / 2;
	const py = player.y + player.size / 2;
	const playerObs: TankObservation = {
		tick: 0,
		self: player as TankObservation['self'],
		allies: [],
		enemies: aliveEnemies as unknown as TankObservation['enemies'],
		projectiles: [],
		bombs: [],
		obstacles: [],
		arena: state.arena,
	};
	const { dist: currEnemyDist, id: currTargetId } = nearestEnemyDistance(playerObs, px, py);
	if (tracker.prevEnemyDist >= 0.0 && currTargetId !== '' && currTargetId === tracker.approachTargetId) {
		const approachReward = (APPROACH_SCALE * (tracker.prevEnemyDist - currEnemyDist) * shapingScale) / ARENA_DIAGONAL;
		if (Math.abs(approachReward) > 1e-8) {
			reward += approachReward;
			breakdown.approach += approachReward;
		}
	}
	tracker.prevEnemyDist = currEnemyDist;
	tracker.approachTargetId = currTargetId;

	// ── Shaping: dodge reward — potential-based on distance to nearest enemy projectile ──
	// Φ(s) = dist/ARENA_DIAGONAL, reward = Φ(s') - Φ(s) — moving away from incoming fire is positive
	const enemyProjectiles = state.projectiles.filter((p) => p.team === 'enemy');
	let currProjDist = ARENA_DIAGONAL;
	if (enemyProjectiles.length > 0) {
		let closestProjDistSq = Infinity;
		for (const p of enemyProjectiles) {
			const dx = p.x - px;
			const dy = p.y - py;
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
	private rng: () => number;
	public lastNormalizedObs: number[] = [];
	public lastAction: number[] = []; // MultiDiscrete: [moveIdx, aimBin, fire, bomb] (ints)
	public lastDecodedAction: TankAction = { move: 'none', aimAngle: 0, fire: false, plantBomb: false };
	public lastValue = 0;
	public lastLogProb = 0;
	public lastRawObs: TankObservation | null = null;

	constructor(mlp: PolicyMLP, rng: () => number = Math.random) {
		this.mlp = mlp;
		this.rng = rng;
	}

	reset(_initial: MatchInit): void {
		this.lastDecodedAction = { move: 'none', aimAngle: 0, fire: false, plantBomb: false };
	}

	act(obs: TankObservation): TankAction {
		this.lastRawObs = obs;
		const normalized = normalizeObs(obs);
		this.lastNormalizedObs = normalized;

		const { logits, value } = this.mlp.forward(normalized);
		const { action, logProb } = this.mlp.sampleMultiCategorical(logits, this.rng);

		this.lastAction = action;
		this.lastValue = value;
		this.lastLogProb = logProb;

		const decoded = decodeMultiDiscreteAction(action, obs);
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
	// Per-step continuation-value contribution at truncated (timeout) steps.
	// Equals V(s_post) at the timeout step, 0 elsewhere. Python multiplies
	// by gamma and adds AFTER reward normalization so the bootstrap stays on
	// the same scale as the value head (which regresses against normalized
	// returns). See train_hybrid.py::_populate_buffer.
	truncation_values: number[];
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
			...config.player,
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

function clamp(value: number, minValue: number, maxValue: number): number {
	return Math.max(minValue, Math.min(maxValue, value));
}

function deepCloneLevel(config: LevelConfig): LevelConfig {
	return JSON.parse(JSON.stringify(config)) as LevelConfig;
}

function applyProceduralDifficulty(config: LevelConfig, seed: number, difficultyBand: number): LevelConfig {
	const d = clamp(difficultyBand, 0, 1);
	if (d <= 0) {
		return config;
	}

	const rng = new SeededRandom(seed * 3571 + 97);
	const varied = deepCloneLevel(config);
	const tankSize = 30;
	const clampX = (x: number) => clamp(x, 0, ARENA_WIDTH - tankSize);
	const clampY = (y: number) => clamp(y, 0, ARENA_HEIGHT - tankSize);

	if (varied.player) {
		const playerJitter = 8 + 24 * d;
		varied.player.x = clampX(varied.player.x + rng.nextRange(-playerJitter, playerJitter));
		varied.player.y = clampY(varied.player.y + rng.nextRange(-playerJitter, playerJitter));
	}

	if (varied.enemies) {
		const enemyJitter = 12 + 36 * d;
		varied.enemies = varied.enemies.map((enemy) => {
			const nextEnemy = { ...enemy };
			nextEnemy.x = clampX(nextEnemy.x + rng.nextRange(-enemyJitter, enemyJitter));
			nextEnemy.y = clampY(nextEnemy.y + rng.nextRange(-enemyJitter, enemyJitter));

			if (nextEnemy.type === 'stationary' && rng.nextFloat() < 0.2 * d) {
				nextEnemy.type = 'stationary-random-aim';
			}
			if (nextEnemy.type === 'stationary-random-aim' && rng.nextFloat() < 0.15 * Math.max(0, d - 0.3)) {
				nextEnemy.type = 'simple-moving';
				nextEnemy.navigator = nextEnemy.navigator ?? { type: 'simple' };
			}
			if (nextEnemy.type === 'simple-moving' && rng.nextFloat() < 0.12 * Math.max(0, d - 0.55)) {
				nextEnemy.type = 'bomber';
				nextEnemy.navigator = { type: 'astar' };
				nextEnemy.bombs = nextEnemy.bombs ?? { type: 'basic', count: 1 };
			}

			if (rng.nextFloat() < 0.45 * d) {
				const ammoType = nextEnemy.ammo?.type ?? 'basic';
				const ammoCount = Math.min(3, (nextEnemy.ammo?.count ?? 1) + 1);
				nextEnemy.ammo = { type: ammoType, count: ammoCount };
			}

			if ((nextEnemy.type === 'bomber' || nextEnemy.type === 'super-bomber') && !nextEnemy.bombs) {
				nextEnemy.bombs = { type: 'basic', count: 1 };
			}

			return nextEnemy;
		});

		if (d > 0.45 && varied.enemies.length < 4 && rng.nextFloat() < 0.35 * (d - 0.45)) {
			const template = varied.enemies[rng.nextInt(0, varied.enemies.length - 1)] ?? varied.enemies[0];
			if (template) {
				const clone = deepCloneLevel({ enemies: [template], obstacles: [] }).enemies?.[0];
				if (clone) {
					clone.x = clampX(rng.nextRange(560, 900));
					clone.y = clampY(rng.nextRange(80, 420));
					varied.enemies.push(clone);
				}
			}
		}
	}

	if (varied.obstacles) {
		const obsJitter = 6 + 18 * d;
		varied.obstacles = varied.obstacles.map((obs) => {
			const widthScale = 1 + rng.nextRange(-0.1 * d, 0.14 * d);
			const heightScale = 1 + rng.nextRange(-0.1 * d, 0.14 * d);
			const width = clamp(obs.width * widthScale, 18, 260);
			const height = clamp(obs.height * heightScale, 18, 280);
			const x = clamp(obs.x + rng.nextRange(-obsJitter, obsJitter), 0, ARENA_WIDTH - width);
			const y = clamp(obs.y + rng.nextRange(-obsJitter, obsJitter), 0, ARENA_HEIGHT - height);
			return { x, y, width, height };
		});

		if (d > 0.55 && varied.obstacles.length < 5 && rng.nextFloat() < 0.22 * (d - 0.55) * 2.0) {
			const width = rng.nextRange(24, 80);
			const height = rng.nextRange(24, 140);
			varied.obstacles.push({
				x: clamp(rng.nextRange(240, 820), 0, ARENA_WIDTH - width),
				y: clamp(rng.nextRange(60, 420), 0, ARENA_HEIGHT - height),
				width,
				height,
			});
		}
	}

	const rules = { ...(varied.rules ?? {}) };
	const baseTurret = rules.turretSpeedMultiplier ?? 1.0;
	rules.turretSpeedMultiplier = clamp(baseTurret * (1 + 0.35 * d + rng.nextRange(-0.08, 0.08)), 0.8, 3.5);
	if (d > 0.7 && rng.nextFloat() < (d - 0.7) * 1.8) {
		rules.projectileBounces = true;
	}
	varied.rules = rules;

	return varied;
}

/**
 * Build a TankObservation from the current simulation state for the player tank.
 * Used to compute V(s_{T+1}) for truncation bootstrapping and end-of-rollout
 * value bootstrap.
 */
function buildPlayerObservation(state: DeepReadonly<GameState>): TankObservation | null {
	const playerTank = state.tanks.find((t) => t.id === PLAYER_TANK_ID);
	if (!playerTank) return null;
	return {
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
	startEpisode: number,
	proceduralLevels: boolean,
	difficultyBand: number,
	playerMaxAmmo: number,
	actionRng: () => number
): RolloutData {
	const obs: number[][] = [];
	const actions: number[][] = [];
	const rewards: number[] = [];
	const episodeStarts: number[] = [];
	const values: number[] = [];
	const logProbs: number[] = [];
	const truncationValues: number[] = [];
	const episodeRewards: number[] = [];
	const episodeLengths: number[] = [];
	const episodeWins: (0 | 1)[] = [];
	const episodeLevels: number[] = [];
	const episodeRewardBreakdowns: RewardBreakdown[] = [];

	const rlController = new RLController(mlp, actionRng);
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
		const levelConfig = proceduralLevels
			? applyProceduralDifficulty(applySpawnJitter(baseConfig, seed), seed, difficultyBand)
			: applySpawnJitter(baseConfig, seed);
		if (playerMaxAmmo > 0 && levelConfig.player) {
			levelConfig.player.ammo = { type: levelConfig.player.ammo?.type ?? 'basic', count: playerMaxAmmo };
		}
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
		let stepTruncValue = 0;

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
			// Truncation: emit raw reward only and report V(s_post) separately so
			// Python can apply the gamma * V_post bootstrap AFTER reward
			// normalization (the value head is trained against normalized returns,
			// so the bootstrap must not be divided by ret_std).
			const postObs = buildPlayerObservation(state);
			stepTruncValue = postObs ? mlp.forward(normalizeObs(postObs)).value : 0;
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
		truncationValues.push(stepTruncValue);
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
		const dummyObs = buildPlayerObservation(state);
		if (!dummyObs) throw new Error('Player tank not found in reset state');
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
		truncation_values: truncationValues,
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

let mlp: PolicyMLP | null = null;
let workerTotalEpisodes = 0;

function getPortArg(): number {
	const args = process.argv.slice(2);
	for (let i = 0; i < args.length; i += 1) {
		if (args[i] === '--port' && args[i + 1]) {
			return Number(args[i + 1]);
		}
	}
	return 50061;
}

function health(
	_call: grpc.ServerUnaryCall<HealthRequest, HealthResponse>,
	callback: grpc.sendUnaryData<HealthResponse>
): void {
	callback(null, { ok: true, message: mlp ? 'ready' : 'weights_not_set' });
}

function setWeightsFromFile(
	call: grpc.ServerUnaryCall<SetWeightsFromFileRequest, SetWeightsFromFileResponse>,
	callback: grpc.sendUnaryData<SetWeightsFromFileResponse>
): void {
	try {
		const filePath = String(call.request.path ?? '');
		if (!filePath) {
			callback({ code: grpc.status.INVALID_ARGUMENT, message: 'Missing path for set_weights_from_file' }, null);
			return;
		}
		const raw = fs.readFileSync(filePath, 'utf8');
		const stateDict = JSON.parse(raw) as Record<string, number[][] | number[]>;
		const weights = parseWeightsFromStateDict(stateDict);
		mlp = new PolicyMLP(weights);
		callback(null, { ok: true });
	} catch (err) {
		callback({ code: grpc.status.INTERNAL, message: String(err) }, null);
	}
}

function collect(
	call: grpc.ServerUnaryCall<CollectRequest, CollectResponse>,
	callback: grpc.sendUnaryData<CollectResponse>
): void {
	try {
		if (!mlp) {
			callback({ code: grpc.status.FAILED_PRECONDITION, message: 'Weights not set' }, null);
			return;
		}
		const req = call.request;
		const nSteps = Number(req.nSteps ?? 4096);
		const levels = (req.levels && req.levels.length > 0 ? req.levels : [1]).map((v) => Number(v));
		const maxTicks = Number(req.maxTicks ?? 1800);
		TICK_NORM_TICKS = Math.max(1, Number(req.tickNormTicks ?? maxTicks));
		const seedStart = Number(req.seedStart ?? 0);
		const replayEveryEpisodes = Number(req.replayEveryEpisodes ?? 0);
		const replayDir = String(req.replayDir ?? path.join(__dirname, '..', '..', 'training', 'output', 'replays'));
		const workerId = Number(req.workerId ?? 0);
		const shapingScale = Math.max(0, Math.min(1, Number(req.shapingScale ?? 1.0)));
		const proceduralLevels = Boolean(req.proceduralLevels ?? true);
		const difficultyBand = clamp(Number(req.difficultyBand ?? 0), 0, 1);
		const playerMaxAmmo = Number(req.playerMaxAmmo ?? 0);
		const globalEpisodeOffset = Number(req.globalEpisodeOffset ?? 0);
		const gamma = Number(req.gamma ?? 0);
		void gamma; // accepted for protocol back-compat; bootstrap is now Python-side

		// Deterministic action-sampling RNG keyed on (seedStart, workerId, episodeOffset)
		// so that, given identical weights and seeds, rollouts are reproducible.
		const actionRngState = new SeededRandom(
			(seedStart || 1) * 0x9e3779b1 + workerId * 0x85ebca77 + (globalEpisodeOffset + 1)
		);
		const actionRng: () => number = () => actionRngState.nextFloat();

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
			globalEpisodeOffset || workerTotalEpisodes,
			proceduralLevels,
			difficultyBand,
			playerMaxAmmo,
			actionRng
		);
		workerTotalEpisodes += (rollout.episode_rewards as number[]).length;
		callback(null, { rolloutJson: JSON.stringify(rollout) });
	} catch (err) {
		callback({ code: grpc.status.INTERNAL, message: String(err) }, null);
	}
}

function close(
	_call: grpc.ServerUnaryCall<CloseRequest, CloseResponse>,
	callback: grpc.sendUnaryData<CloseResponse>
): void {
	callback(null, { closed: true });
	setTimeout(() => process.exit(0), 0);
}

function main(): void {
	const server = new grpc.Server();
	server.addService(rolloutServiceDef.service, {
		Health: health,
		SetWeightsFromFile: setWeightsFromFile,
		Collect: collect,
		Close: close,
	});

	const port = getPortArg();
	const bindAddress = `127.0.0.1:${port}`;
	server.bindAsync(bindAddress, grpc.ServerCredentials.createInsecure(), (err) => {
		if (err) {
			process.stderr.write(`rollout-worker gRPC bind error: ${String(err)}\n`);
			process.exit(1);
		}
	});
}

main();
