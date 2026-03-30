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
import { LEVEL_CONFIGS } from '../src/game/LevelConfig';
import { PolicyMLP, parseWeightsFromStateDict } from './mlp-inference';

// ---- Constants matching treads_env.py ----
const ARENA_WIDTH = 1000.0;
const ARENA_HEIGHT = 500.0;
const MAX_ENEMIES = 3;
const MAX_PROJECTILES = 5;
const MAX_OBSTACLES = 3;
const SELF_DIM = 11;
const ENEMY_DIM = 5;
const PROJ_DIM = 5;
const OBS_DIM = 4;
const OBS_SIZE = SELF_DIM + MAX_ENEMIES * ENEMY_DIM + MAX_PROJECTILES * PROJ_DIM + MAX_OBSTACLES * OBS_DIM;
const MOVE_INTENTS: MoveIntent[] = ['none', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];
const PLAYER_TANK_ID = 'player-0';
const TRACK_AIM_THRESHOLD_RAD = (15 * Math.PI) / 180;
const FIRE_THRESHOLD = -0.2;
const BOMB_THRESHOLD = 0.8;
const TERMINAL_WIN_REWARD = 500.0;
const TERMINAL_LOSS_REWARD = -60.0;
const KILL_REWARD = 180.0;
const SHAPING_MIN_SCALE = 0.2;

// ---- Observation normalization (port of treads_env.py _normalize_obs) ----
function normalizeObs(obs: TankObservation): number[] {
	const result = new Array<number>(OBS_SIZE).fill(0);
	let idx = 0;

	const s = obs.self;
	result[idx] = s.x / ARENA_WIDTH;
	result[idx + 1] = s.y / ARENA_HEIGHT;
	result[idx + 2] = s.aimAngle / (2 * Math.PI);
	result[idx + 3] = s.speed / 100.0;
	result[idx + 4] = s.destroyed ? 1.0 : 0.0;
	result[idx + 5] = Math.min(s.shotCooldownTicks / 300.0, 1.0);
	result[idx + 6] = s.activeAmmo / Math.max(s.maxAmmo, 1);
	result[idx + 7] = s.maxAmmo / 5.0;

	// Derived aim features
	const sx = s.x + s.size / 2;
	const sy = s.y + s.size / 2;
	const aliveEnemies = obs.enemies.filter((e) => !e.destroyed);

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
		const arenaDiag = Math.sqrt(ARENA_WIDTH * ARENA_WIDTH + ARENA_HEIGHT * ARENA_HEIGHT);
		result[idx + 8] = angleToEnemy / (2 * Math.PI) + 0.5;
		result[idx + 9] = Math.min(distToEnemy / arenaDiag, 1.0);
		result[idx + 10] = (aimError / Math.PI) * 0.5 + 0.5;
	} else {
		result[idx + 8] = 0.5;
		result[idx + 9] = 0.0;
		result[idx + 10] = 0.5;
	}
	idx += SELF_DIM;

	// Enemies (up to MAX_ENEMIES, sorted by distance)
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
			result[idx] = e.x / ARENA_WIDTH;
			result[idx + 1] = e.y / ARENA_HEIGHT;
			result[idx + 2] = e.aimAngle / (2 * Math.PI);
			result[idx + 3] = e.speed / 100.0;
			result[idx + 4] = 0.0; // alive
		}
		idx += ENEMY_DIM;
	}

	// Projectiles (up to MAX_PROJECTILES, sorted by distance)
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
			result[idx] = p.x / ARENA_WIDTH;
			result[idx + 1] = p.y / ARENA_HEIGHT;
			result[idx + 2] = (p.vx / 300.0) * 0.5 + 0.5;
			result[idx + 3] = (p.vy / 300.0) * 0.5 + 0.5;
			result[idx + 4] = p.team === 'enemy' ? 1.0 : 0.0;
		}
		idx += PROJ_DIM;
	}

	// Obstacles (up to MAX_OBSTACLES, sorted by distance)
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
			result[idx] = o.x / ARENA_WIDTH;
			result[idx + 1] = o.y / ARENA_HEIGHT;
			result[idx + 2] = o.width / ARENA_WIDTH;
			result[idx + 3] = o.height / ARENA_HEIGHT;
		}
		idx += OBS_DIM;
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
	prevDistToEnemy: number;
	prevAimError: number;
	prevHasLOS: boolean;
	prevMoveIntent: MoveIntent;
}

interface RewardBreakdown {
	kill: number;
	distance: number;
	aim: number;
	trackAim: number;
	pursuit: number;
	retreat: number;
	jitter: number;
	bulletProximity: number;
	losGain: number;
	losLoss: number;
	losMaintain: number;
	badBomb: number;
	terminalWin: number;
	terminalLoss: number;
}

function createRewardBreakdown(): RewardBreakdown {
	return {
		kill: 0,
		distance: 0,
		aim: 0,
		trackAim: 0,
		pursuit: 0,
		retreat: 0,
		jitter: 0,
		bulletProximity: 0,
		losGain: 0,
		losLoss: 0,
		losMaintain: 0,
		badBomb: 0,
		terminalWin: 0,
		terminalLoss: 0,
	};
}

function mergeRewardBreakdown(target: RewardBreakdown, add: RewardBreakdown): void {
	target.kill += add.kill;
	target.distance += add.distance;
	target.aim += add.aim;
	target.trackAim += add.trackAim;
	target.pursuit += add.pursuit;
	target.retreat += add.retreat;
	target.jitter += add.jitter;
	target.bulletProximity += add.bulletProximity;
	target.losGain += add.losGain;
	target.losLoss += add.losLoss;
	target.losMaintain += add.losMaintain;
	target.badBomb += add.badBomb;
	target.terminalWin += add.terminalWin;
	target.terminalLoss += add.terminalLoss;
}

function getBombBlastRadius(kind: string | null): number {
	if (kind === 'love') return 80;
	if (kind === 'basic') return 50;
	return 50;
}

/**
 * Decodes NN action signals into a TankAction.
 * aim_signal ∈ [-1, 1] is interpreted as an offset from angle-to-nearest-enemy:
 *   aim_signal = 0  → aimed directly at enemy
 *   aim_signal = ±1 → aimed 180° away from enemy
 * This makes the optimal aim policy trivially learnable: output 0.
 * When no living enemy exists, falls back to current aim angle.
 */
function decodeActionSignal(signal: number[], rawObs: TankObservation): { decoded: TankAction; clamped: number[] } {
	const clamped = signal.map((v) => Math.max(-1, Math.min(1, v)));
	const moveIdx = Math.max(0, Math.min(8, Math.round((clamped[0] + 1) * 0.5 * 8)));

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
		aimAngle = angleToEnemy + clamped[1] * Math.PI;
	} else {
		// No living enemy: hold current aim
		aimAngle = s.aimAngle;
	}

	return {
		clamped,
		decoded: {
			move: MOVE_INTENTS[moveIdx],
			aimAngle,
			fire: clamped[2] > FIRE_THRESHOLD,
			plantBomb: clamped[3] > BOMB_THRESHOLD,
		},
	};
}

function getShapingScale(episodeIndex: number, targetEpisodes: number): number {
	if (targetEpisodes <= 0) return SHAPING_MIN_SCALE;
	const progress = Math.max(0, Math.min(1, episodeIndex / targetEpisodes));
	return Math.max(SHAPING_MIN_SCALE, 1.0 - progress);
}

function lineIntersectsRect(
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
	for (const edgeX of [rx, rx + rw]) {
		if (dx !== 0) {
			const t = (edgeX - x1) / dx;
			if (t >= 0 && t <= 1) {
				const yAtT = y1 + t * dy;
				if (yAtT >= ry && yAtT <= ry + rh) return true;
			}
		}
	}
	for (const edgeY of [ry, ry + rh]) {
		if (dy !== 0) {
			const t = (edgeY - y1) / dy;
			if (t >= 0 && t <= 1) {
				const xAtT = x1 + t * dx;
				if (xAtT >= rx && xAtT <= rx + rw) return true;
			}
		}
	}
	return false;
}

function hasClearLOS(state: DeepReadonly<GameState>, sx: number, sy: number, ex: number, ey: number): boolean {
	for (const o of state.obstacles) {
		if (lineIntersectsRect(sx, sy, ex, ey, o.x, o.y, o.width, o.height)) {
			return false;
		}
	}
	return true;
}

function initRewardTracker(obs: TankObservation): RewardTracker {
	const s = obs.self;
	const sx = s.x + s.size / 2;
	const sy = s.y + s.size / 2;
	const alive = obs.enemies.filter((e) => !e.destroyed);

	if (alive.length === 0) {
		return { prevEnemyAliveCount: 0, prevDistToEnemy: 0, prevAimError: 0, prevHasLOS: false, prevMoveIntent: 'none' };
	}

	let nearest = alive[0];
	let nearestDistSq = Infinity;
	for (const e of alive) {
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
	const toEnemy = Math.atan2(ey - sy, ex - sx);
	const aimError = Math.abs(Math.atan2(Math.sin(s.aimAngle - toEnemy), Math.cos(s.aimAngle - toEnemy)));

	// Compute initial LOS (matching Python's _prev_has_los initialization)
	let hasLOS = true;
	for (const o of obs.obstacles) {
		if (lineIntersectsRect(sx, sy, ex, ey, o.x, o.y, o.width, o.height)) {
			hasLOS = false;
			break;
		}
	}

	return {
		prevEnemyAliveCount: alive.length,
		prevDistToEnemy: Math.sqrt(nearestDistSq),
		prevAimError: aimError,
		prevHasLOS: hasLOS,
		prevMoveIntent: 'none',
	};
}

function computeSteppingReward(
	state: DeepReadonly<GameState>,
	tracker: RewardTracker,
	action: TankAction,
	shapingScale: number
): { reward: number; breakdown: RewardBreakdown } {
	let reward = 0;
	const breakdown = createRewardBreakdown();

	const player = state.tanks.find((t) => t.id === PLAYER_TANK_ID);
	if (!player) return { reward: 0, breakdown };
	const sx = player.x + player.size / 2;
	const sy = player.y + player.size / 2;
	const enemies = state.tanks.filter((t) => t.team === 'enemy');
	const aliveEnemies = enemies.filter((e) => !e.destroyed);

	// Kill event
	const enemiesKilled = tracker.prevEnemyAliveCount - aliveEnemies.length;
	if (enemiesKilled > 0) {
		const value = KILL_REWARD * enemiesKilled;
		reward += value;
		breakdown.kill += value;
	}
	tracker.prevEnemyAliveCount = aliveEnemies.length;

	if (action.move !== 'none' && tracker.prevMoveIntent !== 'none' && action.move !== tracker.prevMoveIntent) {
		const jitterPenalty = -0.01 * shapingScale;
		reward += jitterPenalty;
		breakdown.jitter += jitterPenalty;
	}
	tracker.prevMoveIntent = action.move;

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
		const dist = Math.sqrt(nearestDistSq);
		const arenaDiag = Math.sqrt(ARENA_WIDTH * ARENA_WIDTH + ARENA_HEIGHT * ARENA_HEIGHT);

		// LOS: compute first so we can weight other rewards
		const hasLOS = hasClearLOS(state, sx, sy, ex, ey);

		// Scale distance/aim rewards by LOS quality — no LOS means these potentials are misleading
		const losMultiplier = hasLOS ? 1.0 : 0.1;

		// Distance potential (only strongly rewarded when we have clear sight)
		const distDelta = (tracker.prevDistToEnemy - dist) / arenaDiag;
		const distanceRaw = distDelta * 1.2 * losMultiplier * shapingScale;
		const distanceValue = Math.max(-0.08 * shapingScale, Math.min(0.08 * shapingScale, distanceRaw));
		reward += distanceValue;
		breakdown.distance += distanceValue;
		tracker.prevDistToEnemy = dist;

		// Aim potential (only meaningful with LOS)
		const aimAngle = player.aimAngle;
		const toEnemy = Math.atan2(ey - sy, ex - sx);
		const aimError = Math.abs(Math.atan2(Math.sin(aimAngle - toEnemy), Math.cos(aimAngle - toEnemy)));
		const aimDelta = (tracker.prevAimError - aimError) / Math.PI;
		const aimRaw = aimDelta * 0.5 * losMultiplier * shapingScale;
		const aimValue = Math.max(-0.08 * shapingScale, Math.min(0.08 * shapingScale, aimRaw));
		reward += aimValue;
		breakdown.aim += aimValue;
		tracker.prevAimError = aimError;

		// Continuous alignment reward: stronger the closer to perfect aim
		// aimError=0 → full bonus, aimError=π → 0 bonus
		const alignmentQuality = Math.max(0, 1.0 - aimError / Math.PI);
		const trackValue = alignmentQuality * 0.008 * shapingScale;
		reward += trackValue;
		breakdown.trackAim += trackValue;

		const threatDistSq = 140 * 140;
		const underThreat = state.projectiles.some((p) => {
			if (p.team !== 'enemy') return false;
			const dx = p.x - sx;
			const dy = p.y - sy;
			return dx * dx + dy * dy <= threatDistSq;
		});
		if (!underThreat && distDelta > 0) {
			const pursuitValue = distDelta * 0.4 * shapingScale;
			reward += pursuitValue;
			breakdown.pursuit += pursuitValue;
		} else if (underThreat && distDelta < 0) {
			const retreatValue = -distDelta * 0.4 * shapingScale;
			reward += retreatValue;
			breakdown.retreat += retreatValue;
		}

		// Bullet proximity — friendly bullet near enemy is good
		const friendlyProj = state.projectiles.filter((p) => p.team !== 'enemy');
		for (const p of friendlyProj) {
			const pDist = Math.sqrt((p.x - ex) * (p.x - ex) + (p.y - ey) * (p.y - ey));
			if (pDist < 200) {
				const bulletValue = 0.05 * (1.0 - pDist / 200.0) * shapingScale;
				reward += bulletValue;
				breakdown.bulletProximity += bulletValue;
			}
		}

		// LOS: strong one-time reward for gaining/losing sight, plus per-tick bonus for maintaining it
		if (hasLOS && !tracker.prevHasLOS) {
			const losGainValue = 0.8 * shapingScale;
			reward += losGainValue;
			breakdown.losGain += losGainValue;
		} else if (!hasLOS && tracker.prevHasLOS) {
			const losLossValue = -0.3 * shapingScale;
			reward += losLossValue;
			breakdown.losLoss += losLossValue;
		}
		if (hasLOS) {
			const losMaintainValue = 0.01 * shapingScale;
			reward += losMaintainValue;
			breakdown.losMaintain += losMaintainValue;
		}
		tracker.prevHasLOS = hasLOS;

		// Fire incentive: proportional to alignment quality; penalize blind fire
		// When LOS exists, reward firing when aligned, penalize not firing
		if (hasLOS) {
			// Continuous fire incentive based on alignment quality
			const fireAlignBonus = alignmentQuality * 0.25 * shapingScale;
			if (action.fire) {
				reward += fireAlignBonus;
				breakdown.trackAim += fireAlignBonus;
			} else if (aimError <= TRACK_AIM_THRESHOLD_RAD) {
				// Penalize not firing when closely aimed
				const noFirePenalty = -0.15 * shapingScale;
				reward += noFirePenalty;
				breakdown.trackAim += noFirePenalty;
			}
		} else if (action.fire) {
			// Blind fire penalty
			const blindFirePenalty = -0.01 * shapingScale;
			reward += blindFirePenalty;
			breakdown.aim += blindFirePenalty;
		}

		if (action.plantBomb) {
			const blastRadius = getBombBlastRadius(player.bombType);
			const usefulRadius = blastRadius * 2;
			if (dist > usefulRadius) {
				const badBombPenalty = -0.08 * shapingScale;
				reward += badBombPenalty;
				breakdown.badBomb += badBombPenalty;
			}
		}
	}

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
	private pendingDecodedAction: TankAction | null = null;

	constructor(mlp: PolicyMLP) {
		this.mlp = mlp;
	}

	reset(_initial: MatchInit): void {
		this.pendingDecodedAction = null;
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

		const decoded = decodeActionSignal(action, obs).decoded;
		const actionForThisTick = this.pendingDecodedAction ?? {
			move: 'none',
			aimAngle: obs.self.aimAngle,
			fire: false,
			plantBomb: false,
		};
		this.pendingDecodedAction = decoded;
		this.lastDecodedAction = actionForThisTick;
		return actionForThisTick;
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

function collectRollout(
	mlp: PolicyMLP,
	nSteps: number,
	levels: number[],
	maxTicks: number,
	seedStart: number,
	replayEveryEpisodes: number,
	replayDir: string,
	episodeOffset: number,
	targetEpisodes: number
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
	let seed = seedStart;
	let tracker: RewardTracker | null = null;
	let episodeTick = 0;
	let episodeReward = 0;
	let isNewEpisode = true;
	let currentLevel = levels[0];
	let episodeBreakdown = createRewardBreakdown();
	let replayRecorder: ReplayRecorder | null = null;
	let completedEpisodes = 0;

	function resetEpisode(): Simulation {
		// Pick level from curriculum based on seed for diversity
		currentLevel = levels[seed % levels.length];
		const levelConfig = LEVEL_CONFIGS[currentLevel - 1];
		const initialState = createInitialGameState(levelConfig, seed);
		const controllers = createDefaultControllers(levelConfig);
		controllers[PLAYER_TANK_ID] = rlController;
		replayRecorder = new ReplayRecorder(levelConfig, seed);
		tracker = null;
		episodeTick = 0;
		episodeReward = 0;
		episodeBreakdown = createRewardBreakdown();
		isNewEpisode = true;
		seed++;
		return new Simulation(initialState, controllers);
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
			tracker = initRewardTracker(rlController.lastRawObs);
		}

		// Check post-step state for reward computation
		const state = sim.getState();
		const player = state.tanks.find((t) => t.id === PLAYER_TANK_ID);
		episodeTick++;

		let done = false;
		let stepReward: number;
		const absoluteEpisode = episodeOffset + completedEpisodes + 1;
		const shapingScale = getShapingScale(absoluteEpisode, targetEpisodes);

		if (state.status === 'player_win') {
			// Terminal: win
			const stepping = computeSteppingReward(state, tracker, rlController.lastDecodedAction, shapingScale);
			stepReward = stepping.reward + TERMINAL_WIN_REWARD;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
			episodeBreakdown.terminalWin += TERMINAL_WIN_REWARD;
			done = true;
		} else if (state.status === 'enemy_win' || player?.destroyed) {
			// Terminal: loss
			const stepping = computeSteppingReward(state, tracker, rlController.lastDecodedAction, shapingScale);
			stepReward = stepping.reward + TERMINAL_LOSS_REWARD;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
			episodeBreakdown.terminalLoss += TERMINAL_LOSS_REWARD;
			done = true;
		} else if (episodeTick >= maxTicks) {
			// Truncation: unresolved episode gets a moderate penalty to encourage decisive play.
			const stepping = computeSteppingReward(state, tracker, rlController.lastDecodedAction, shapingScale);
			stepReward = stepping.reward - 20.0;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
			episodeBreakdown.terminalLoss -= 20.0;
			done = true;
		} else {
			// Normal step
			const stepping = computeSteppingReward(state, tracker, rlController.lastDecodedAction, shapingScale);
			stepReward = stepping.reward;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
		}

		rewards.push(stepReward);
		episodeReward += stepReward;

		if (done) {
			completedEpisodes += 1;
			const absoluteEpisode = episodeOffset + completedEpisodes;
			episodeRewards.push(episodeReward);
			episodeLengths.push(episodeTick);
			episodeWins.push(state.status === 'player_win' ? 1 : 0);
			episodeLevels.push(currentLevel);
			episodeRewardBreakdowns.push(episodeBreakdown);
			if (replayEveryEpisodes > 0 && absoluteEpisode % replayEveryEpisodes === 0 && replayRecorder !== null) {
				fs.mkdirSync(replayDir, { recursive: true });
				const replayPath = path.join(
					replayDir,
					`episode_${absoluteEpisode}_L${currentLevel}_S${seed - 1}_${state.status}.json`
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

	writeLine({ type: 'ready' });

	while (true) {
		const line = await readLine();
		let cmd: Record<string, unknown>;
		try {
			cmd = JSON.parse(line);
		} catch {
			continue;
		}

		if (cmd.type === 'set_weights') {
			const stateDict = cmd.state_dict as Record<string, number[][] | number[]>;
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
			const seedStart = (cmd.seedStart as number) ?? 0;
			const replayEveryEpisodes = (cmd.replayEveryEpisodes as number) ?? 0;
			const replayDir = (cmd.replayDir as string) ?? path.join(__dirname, '..', '..', 'training', 'output', 'replays');
			const episodeOffset = (cmd.episodeOffset as number) ?? 0;
			const targetEpisodes = (cmd.targetEpisodes as number) ?? 10_000;
			const rollout = collectRollout(
				mlp,
				nSteps,
				levels,
				maxTicks,
				seedStart,
				replayEveryEpisodes,
				replayDir,
				episodeOffset,
				targetEpisodes
			);
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
		}
	}
}

main().catch((err) => {
	process.stderr.write(`Rollout worker error: ${err}\n`);
	process.exit(1);
});
