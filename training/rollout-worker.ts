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
	TankAction,
	TankController,
	TankObservation,
} from '../src/game/core/types';
import { NavigationPlanner } from '../src/game/navigation/NavigationPlanner';
import { PolicyMLP, parseWeightsFromStateDict } from './mlp-inference';
import {
	decodePolicyAction,
} from '../src/game/controllers/NeuralModelContract';
import { resolveScenarioConfig } from './training-scenarios';
import {
	applyProceduralDifficulty as varyDifficulty,
	applySpawnJitter as varySpawn,
	configurePlayerResources,
} from './scenario-variation';
import { potentialBasedApproachReward } from './reward-shaping';
import { normalizePolicyObservation, POLICY_TICK_NORM_TICKS } from '../src/game/controllers/NeuralObservation';
import { ScriptedEnemyController } from '../src/game/controllers/ScriptedEnemyController';
import { LEVEL_CONFIGS } from '../src/game/LevelConfig';

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
	playerMaxBombs?: number;
	globalEpisodeOffset?: number;
	gamma?: number;
}

interface CollectResponse {
	rolloutJson: string;
}

interface EvaluationEpisodeRequest {
	level?: number;
	seed?: number;
}

interface EvaluateRequest {
	episodes?: EvaluationEpisodeRequest[];
	maxTicks?: number;
	tickNormTicks?: number;
	proceduralLevels?: boolean;
	difficultyBand?: number;
	playerMaxAmmo?: number;
	playerMaxBombs?: number;
	canonicalSpawns?: boolean;
	heuristicPlayer?: boolean;
}

interface EvaluateResponse {
	evaluationJson: string;
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

const PLAYER_TANK_ID = 'player-0';
const STEP_PENALTY = -0.001;
const HIT_REWARD = 0.3;
const TOOK_DAMAGE_PENALTY = -0.3;
const KILL_REWARD = 2.0;
const DEATH_REWARD = -2.0;
const TERMINAL_WIN_REWARD = 5.0;
const TERMINAL_LOSS_REWARD = -3.0;
const TIMEOUT_REWARD = -1.0;
const BOMB_PLANT_PENALTY = -0.03;
const APPROACH_SCALE = 2.5;
const ARENA_DIAGONAL = Math.hypot(1000, 500);
const TICK_NORM_TICKS = POLICY_TICK_NORM_TICKS;

// One encoder for both rollout collection and the deployed policy.
export function normalizeWorkerObservation(obs: TankObservation, planner?: NavigationPlanner | null): number[] {
	return normalizePolicyObservation(obs, TICK_NORM_TICKS, planner);
}

// ---- Reward computation (port of treads_env.py _compute_reward) ----
interface RewardTracker {
	prevEnemyAliveCount: number;
	prevEnemyHealthTotal: number;
	prevSelfHealth: number;
	prevEnemyDist: number;
	approachTargetId: string;
	prevOwnBombCount: number;
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
	bombCost: number;
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
		bombCost: 0,
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
	target.bombCost += add.bombCost;
}

/**
 * Decode the versioned policy action into a TankAction.
 *   action[0] ∈ [0, 9)  → MoveIntent index
 *   action[1] selects a nearest-enemy-relative full-circle residual
 *   action[2] ∈ {0, 1}  → fire
 *   action[3] ∈ {0, 1}  → plant bomb
 */
function decodeMultiDiscreteAction(action: number[], rawObs: TankObservation): TankAction {
	return decodePolicyAction(action, rawObs);
}

function nearestEnemyDistance(
	obs: TankObservation,
	sx: number,
	sy: number,
	navPlanner: NavigationPlanner | null = null
): { dist: number; id: string } {
	const alive = obs.enemies.filter((e) => !e.destroyed);
	if (alive.length === 0) return { dist: 0, id: '' };
	let best = alive[0];
	let bestSq = Infinity;
	for (const e of alive) {
		const dx = e.x + e.size / 2 - sx;
		const dy = e.y + e.size / 2 - sy;
		const distance = navPlanner
			? navPlanner.getPathDistance(sx, sy, e.x + e.size / 2, e.y + e.size / 2)
			: Math.sqrt(dx * dx + dy * dy);
		const score = distance * distance;
		if (score < bestSq) {
			bestSq = score;
			best = e;
		}
	}
	return { dist: Math.sqrt(bestSq), id: String(best.id ?? '') };
}

function nearestEnemyPathDistance(
	state: DeepReadonly<GameState>,
	navPlanner: NavigationPlanner | null
): { dist: number; id: string } {
	const player = state.tanks.find((tank) => tank.id === PLAYER_TANK_ID);
	if (!player) return { dist: 0, id: '' };
	const px = player.x + player.size / 2;
	const py = player.y + player.size / 2;
	let bestDist = Number.POSITIVE_INFINITY;
	let bestId = '';
	for (const enemy of state.tanks) {
		if (enemy.team !== 'enemy' || enemy.destroyed) continue;
		const ex = enemy.x + enemy.size / 2;
		const ey = enemy.y + enemy.size / 2;
		const dx = ex - px;
		const dy = ey - py;
		const distance = navPlanner ? navPlanner.getPathDistance(px, py, ex, ey) : Math.sqrt(dx * dx + dy * dy);
		if (distance < bestDist) {
			bestDist = distance;
			bestId = enemy.id;
		}
	}
	return { dist: Number.isFinite(bestDist) ? bestDist : 0, id: bestId };
}

function initRewardTracker(obs: TankObservation, navPlanner: NavigationPlanner | null): RewardTracker {
	const alive = obs.enemies.filter((e) => !e.destroyed);
	const sx = obs.self.x + obs.self.size / 2;
	const sy = obs.self.y + obs.self.size / 2;

	const { dist: initialEnemyDist, id: initialEnemyId } = nearestEnemyDistance(obs, sx, sy, navPlanner);

	return {
		prevEnemyAliveCount: alive.length,
		prevEnemyHealthTotal: alive.reduce((total, enemy) => total + enemy.health, 0),
		prevSelfHealth: obs.self.health,
		prevEnemyDist: initialEnemyDist,
		approachTargetId: initialEnemyId,
		prevOwnBombCount: obs.bombs.filter((bomb) => bomb.ownerTankId === PLAYER_TANK_ID).length,
	};
}

function computeSteppingReward(
	state: DeepReadonly<GameState>,
	tracker: RewardTracker,
	decodedAction: TankAction,
	_rawObs: TankObservation,
	shapingScale: number,
	navPlanner: NavigationPlanner | null,
	gamma: number
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

	const ownBombCount = state.bombs.filter((bomb) => bomb.ownerTankId === PLAYER_TANK_ID).length;
	if (ownBombCount > tracker.prevOwnBombCount) {
		const value = BOMB_PLANT_PENALTY * (ownBombCount - tracker.prevOwnBombCount);
		reward += value;
		breakdown.bombCost += value;
	}
	tracker.prevOwnBombCount = ownBombCount;

	// Kill event
	const enemiesKilled = tracker.prevEnemyAliveCount - aliveEnemies.length;
	if (enemiesKilled > 0) {
		const value = KILL_REWARD * enemiesKilled;
		reward += value;
		breakdown.kill += value;
	}
	tracker.prevEnemyAliveCount = aliveEnemies.length;

	// A state potential must also include target changes on kills. Closing one
	// target and silently initializing another breaks the telescoping guarantee.
	const episodeTerminal = state.status !== 'running';
	const nextTarget = nearestEnemyPathDistance(state, navPlanner);
	const currentDistance = nextTarget.id ? nextTarget.dist : Number.POSITIVE_INFINITY;
	const approachReward = potentialBasedApproachReward(
		tracker.prevEnemyDist, currentDistance, ARENA_DIAGONAL, gamma,
		APPROACH_SCALE * shapingScale, episodeTerminal
	);
	reward += approachReward;
	breakdown.approach += approachReward;
	tracker.prevEnemyDist = currentDistance;
	tracker.approachTargetId = nextTarget.id;

	// Projectile-nearest shaping was removed because entity appearance and
	// disappearance made its potential discontinuous. Damage avoidance remains
	// directly represented by the hurt and death penalties.

	return { reward, breakdown };
}

// ---- RL Controller: captures obs/action/value/logprob during act() ----
class RLController implements TankController {
	private mlp: PolicyMLP;
	private rng: () => number;
	private deterministic: boolean;
	private navigationPlanner: NavigationPlanner | null = null;
	public lastNormalizedObs: number[] = [];
	public lastAction: number[] = []; // MultiDiscrete: [moveIdx, aimBin, fire, bomb] (ints)
	public lastDecodedAction: TankAction = { move: 'none', aimAngle: 0, fire: false, plantBomb: false };
	public lastValue = 0;
	public lastLogProb = 0;
	public lastRawObs: TankObservation | null = null;

	constructor(mlp: PolicyMLP, rng: () => number = Math.random, deterministic = false) {
		this.mlp = mlp;
		this.rng = rng;
		this.deterministic = deterministic;
	}

	reset(initial: MatchInit): void {
		this.lastDecodedAction = { move: 'none', aimAngle: 0, fire: false, plantBomb: false };
		const self = initial.tanks.find((tank) => tank.id === initial.selfId);
		this.navigationPlanner = new NavigationPlanner(initial.arena, initial.obstacles, self?.size ?? 30);
	}

	act(obs: TankObservation): TankAction {
		this.lastRawObs = obs;
		const normalized = normalizeWorkerObservation(obs, this.navigationPlanner);
		this.lastNormalizedObs = normalized;

		const { logits, value } = this.mlp.forward(normalized);
		let action: number[];
		let logProb = 0;
		if (this.deterministic) {
			action = this.mlp.argmaxMultiCategorical(logits);
		} else {
			({ action, logProb } = this.mlp.sampleMultiCategorical(logits, this.rng));
		}

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

interface EvaluationEpisodeResult {
	level: number;
	seed: number;
	win: boolean;
	ticks: number;
	outcome: string;
	shots: number;
	bombs: number;
	blockedSteps: number;
	displacement: number;
	playerHealth: number;
	enemiesDestroyed: number;
}

interface EvaluationData {
	type: 'evaluation';
	episodes: EvaluationEpisodeResult[];
}

function clamp(value: number, minValue: number, maxValue: number): number {
	return Math.max(minValue, Math.min(maxValue, value));
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

interface RolloutSession {
	sim: Simulation;
	controller: RLController;
	levelRng: SeededRandom;
	tracker: RewardTracker | null;
	episodeTick: number;
	episodeReward: number;
	currentLevel: number;
	episodeSeed: number;
	nextSeed: number;
	breakdown: RewardBreakdown;
	recorder: ReplayRecorder | null;
	planner: NavigationPlanner | null;
	shapingScale: number;
	maxTicks: number;
}

let rolloutSession: RolloutSession | null = null;

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
	playerMaxBombs: number,
	gamma: number,
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

	const prior = rolloutSession;
	const rlController = prior?.controller ?? new RLController(mlp, actionRng);
	rlController.setMLP(mlp);
	const levelRng = prior?.levelRng ?? new SeededRandom(seedStart * 31337 + 7);
	let seed = Math.max(seedStart, prior?.nextSeed ?? seedStart);
	let episodeSeed = prior?.episodeSeed ?? seedStart;
	let tracker: RewardTracker | null = prior?.tracker ?? null;
	let episodeTick = prior?.episodeTick ?? 0;
	let episodeReward = prior?.episodeReward ?? 0;
	let isNewEpisode = episodeTick === 0;
	let currentLevel = prior?.currentLevel ?? levels[0];
	let episodeBreakdown = prior?.breakdown ?? createRewardBreakdown();
	let replayRecorder: ReplayRecorder | null = prior?.recorder ?? null;
	let completedEpisodes = 0;
	let navPlanner: NavigationPlanner | null = prior?.planner ?? null;
	let episodeShapingScale = prior?.shapingScale ?? shapingScale;
	let episodeMaxTicks = prior?.maxTicks ?? maxTicks;

	function resetEpisode(): Simulation {
		// Pick scenario uniformly at random from the active curriculum pool
		currentLevel = levels[levelRng.nextInt(0, levels.length - 1)];
		const baseConfig = resolveScenarioConfig(currentLevel);
		const variedConfig = proceduralLevels
			? varyDifficulty(varySpawn(baseConfig, seed), seed, difficultyBand)
			: varySpawn(baseConfig, seed);
		const realGameLevel = currentLevel >= 1 && currentLevel <= LEVEL_CONFIGS.length;
		const levelConfig = configurePlayerResources(variedConfig,
			realGameLevel ? -1 : playerMaxAmmo, realGameLevel ? -1 : playerMaxBombs);
		const initialState = createInitialGameState(levelConfig, seed);
		const controllers = createDefaultControllers(levelConfig);
		controllers[PLAYER_TANK_ID] = rlController;
		episodeSeed = seed;
		episodeShapingScale = shapingScale;
		episodeMaxTicks = maxTicks;
		replayRecorder = replayEveryEpisodes > 0 ? new ReplayRecorder(levelConfig, seed) : null;
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

	let sim = prior?.sim ?? resetEpisode();

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
			const stepping = computeSteppingReward(state, tracker, lastDecodedAction, lastRawObs, episodeShapingScale, navPlanner, gamma);
			stepReward = stepping.reward + TERMINAL_WIN_REWARD;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
			episodeBreakdown.terminalWin += TERMINAL_WIN_REWARD;
			done = true;
		} else if (state.status !== 'running' || player?.destroyed) {
			const stepping = computeSteppingReward(state, tracker, lastDecodedAction, lastRawObs, episodeShapingScale, navPlanner, gamma);
			stepReward = stepping.reward + DEATH_REWARD + TERMINAL_LOSS_REWARD;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
			episodeBreakdown.death += DEATH_REWARD;
			episodeBreakdown.terminalLoss += TERMINAL_LOSS_REWARD;
			done = true;
		} else if (episodeTick >= episodeMaxTicks) {
			const stepping = computeSteppingReward(state, tracker, lastDecodedAction, lastRawObs, episodeShapingScale, navPlanner, gamma);
			// Truncation: emit raw reward only and report V(s_post) separately so
			// Python can apply the gamma * V_post bootstrap AFTER reward
			// normalization (the value head is trained against normalized returns,
			// so the bootstrap must not be divided by ret_std).
			const postObs = buildPlayerObservation(state);
			stepTruncValue = postObs ? mlp.forward(normalizeWorkerObservation(postObs, navPlanner)).value : 0;
			stepReward = stepping.reward + TIMEOUT_REWARD;
			mergeRewardBreakdown(episodeBreakdown, stepping.breakdown);
			episodeBreakdown.timeout += TIMEOUT_REWARD;
			done = true;
		} else {
			const stepping = computeSteppingReward(state, tracker, lastDecodedAction, lastRawObs, episodeShapingScale, navPlanner, gamma);
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
					`episode_${absoluteEpisode}_W${workerId}_L${currentLevel}_S${episodeSeed}_${state.status}.json`
				);
				fs.writeFileSync(replayPath, JSON.stringify((replayRecorder as ReplayRecorder).toJSON()));
			}
			sim = resetEpisode();
		}
	}
	rolloutSession = {
		sim, controller: rlController, levelRng, tracker, episodeTick, episodeReward,
		currentLevel, episodeSeed, nextSeed: seed, breakdown: episodeBreakdown,
		recorder: replayRecorder, planner: navPlanner, shapingScale: episodeShapingScale, maxTicks: episodeMaxTicks,
	};

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
		lastObs = normalizeWorkerObservation(dummyObs, navPlanner);
	} else {
		// Mid-episode: bootstrap from the post-step state s_(T+1), not the
		// controller's pre-step observation s_T.
		const postObs = buildPlayerObservation(sim.getState());
		if (!postObs) throw new Error('Player tank not found at rollout boundary');
		lastObs = normalizeWorkerObservation(postObs, navPlanner);
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

/** Run deterministic held-out episodes without per-tick Python RPC. */
export function evaluatePolicy(
	mlp: PolicyMLP,
	episodes: EvaluationEpisodeRequest[],
	maxTicks: number,
	proceduralLevels: boolean,
	difficultyBand: number,
	playerMaxAmmo: number,
	playerMaxBombs: number,
	canonicalSpawns = false,
	heuristicPlayer = false
): EvaluationData {
	const results: EvaluationEpisodeResult[] = [];
	for (const episode of episodes) {
		const level = Number(episode.level ?? 1);
		const seed = Number(episode.seed ?? 0);
		const baseConfig = resolveScenarioConfig(level);
		const spawns = canonicalSpawns ? baseConfig : varySpawn(baseConfig, seed);
		const variedConfig = proceduralLevels ? varyDifficulty(spawns, seed, difficultyBand) : spawns;
		const levelConfig = configurePlayerResources(variedConfig, playerMaxAmmo, playerMaxBombs);
		const initialState = createInitialGameState(levelConfig, seed);
		const controllers = createDefaultControllers(levelConfig);
		controllers[PLAYER_TANK_ID] = heuristicPlayer
			? new ScriptedEnemyController({ navigationMode: 'astar-avoidance', randomAim: false,
				recalculationInterval: 20, aggressionFactor: 7, heuristicProfile: 'advanced' })
			: new RLController(mlp, Math.random, true);
		const simulation = new Simulation(initialState, controllers, { debugFreeze: false });

		let ticks = 0;
		let shots = 0;
		let bombs = 0;
		let blockedSteps = 0;
		let displacement = 0;
		const startPlayer = initialState.tanks.find((tank) => tank.id === PLAYER_TANK_ID);
		if (!startPlayer) throw new Error('Evaluation scenario has no player tank');
		let state = simulation.getState();
		while (state.status === 'running' && ticks < maxTicks) {
			const result = simulation.step();
			shots += result.events.filter((event) => event.type === 'projectile-fired' && event.tankId === PLAYER_TANK_ID).length;
			bombs += result.events.filter((event) => event.type === 'bomb-planted' && event.tankId === PLAYER_TANK_ID).length;
			ticks++;
			state = simulation.getState();
			const player = state.tanks.find((tank) => tank.id === PLAYER_TANK_ID);
			if (!player) throw new Error('Evaluation lost its player tank');
			if (player.wasLastMoveBlocked) blockedSteps++;
			displacement = Math.max(displacement, Math.hypot(player.x - startPlayer.x, player.y - startPlayer.y));
		}

		results.push({
			level,
			seed,
			win: state.status === 'player_win',
			ticks,
			outcome: state.status === 'running' ? 'timeout' : state.status,
			shots, bombs, blockedSteps, displacement,
			playerHealth: state.tanks.find((tank) => tank.id === PLAYER_TANK_ID)?.health ?? 0,
			enemiesDestroyed: state.tanks.filter((tank) => tank.team !== startPlayer.team && tank.destroyed).length,
		});
	}
	return { type: 'evaluation', episodes: results };
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

function getParentPidArg(): number | null {
	const args = process.argv.slice(2);
	for (let i = 0; i < args.length; i += 1) {
		if (args[i] === '--parent-pid' && args[i + 1]) {
			const parentPid = Number(args[i + 1]);
			return Number.isSafeInteger(parentPid) && parentPid > 0 ? parentPid : null;
		}
	}
	return null;
}

function watchParentProcess(parentPid: number | null): void {
	if (parentPid === null) return;
	const timer = setInterval(() => {
		try {
			process.kill(parentPid, 0);
		} catch {
			clearInterval(timer);
			process.exit(0);
		}
	}, 1000);
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
		const nSteps = Number(req.nSteps);
		if (!Number.isInteger(nSteps) || nSteps < 1 ||
			(req.tickNormTicks && req.tickNormTicks !== TICK_NORM_TICKS)) {
			callback({ code: grpc.status.INVALID_ARGUMENT, message: 'Positive n_steps and the v3 tick scale of 720 are required' }, null);
			return;
		}
		const levels = (req.levels && req.levels.length > 0 ? req.levels : [1]).map((v) => Number(v));
		const maxTicks = Number(req.maxTicks || 1800);
		const seedStart = Number(req.seedStart ?? 0);
		const replayEveryEpisodes = Number(req.replayEveryEpisodes ?? 0);
		const replayDir = String(req.replayDir ?? path.join(__dirname, '..', '..', 'training', 'output', 'replays'));
		const workerId = Number(req.workerId ?? 0);
		const shapingScale = Math.max(0, Math.min(1, Number(req.shapingScale ?? 1.0)));
		const proceduralLevels = Boolean(req.proceduralLevels ?? true);
		const difficultyBand = clamp(Number(req.difficultyBand ?? 0), 0, 1);
		const playerMaxAmmo = Number(req.playerMaxAmmo ?? 0);
		const playerMaxBombs = Number(req.playerMaxBombs ?? 0);
		const globalEpisodeOffset = Number(req.globalEpisodeOffset ?? 0);
		const gamma = Math.max(0, Math.min(1, Number(req.gamma ?? 0.997)));

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
			playerMaxBombs,
			gamma,
			actionRng
		);
		workerTotalEpisodes += (rollout.episode_rewards as number[]).length;
		callback(null, { rolloutJson: JSON.stringify(rollout) });
	} catch (err) {
		callback({ code: grpc.status.INTERNAL, message: String(err) }, null);
	}
}

function evaluate(
	call: grpc.ServerUnaryCall<EvaluateRequest, EvaluateResponse>,
	callback: grpc.sendUnaryData<EvaluateResponse>
): void {
	try {
		if (!mlp) {
			callback({ code: grpc.status.FAILED_PRECONDITION, message: 'Weights not set' }, null);
			return;
		}
		const req = call.request;
		const episodes = req.episodes ?? [];
		if (episodes.length === 0) {
			callback({ code: grpc.status.INVALID_ARGUMENT, message: 'No evaluation episodes supplied' }, null);
			return;
		}
		if (req.tickNormTicks && req.tickNormTicks !== TICK_NORM_TICKS) {
			callback({ code: grpc.status.INVALID_ARGUMENT, message: 'The v3 tick scale must be 720' }, null);
			return;
		}
		const maxTicks = Math.max(1, Number(req.maxTicks || 1800));
		const evaluation = evaluatePolicy(
			mlp,
			episodes,
			maxTicks,
			Boolean(req.proceduralLevels ?? true),
			clamp(Number(req.difficultyBand ?? 0), 0, 1),
			Number(req.playerMaxAmmo ?? 0),
			Number(req.playerMaxBombs ?? 0),
			Boolean(req.canonicalSpawns),
			Boolean(req.heuristicPlayer)
		);
		callback(null, { evaluationJson: JSON.stringify(evaluation) });
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
	watchParentProcess(getParentPidArg());
	const server = new grpc.Server();
	server.addService(rolloutServiceDef.service, {
		Health: health,
		SetWeightsFromFile: setWeightsFromFile,
		Collect: collect,
		Evaluate: evaluate,
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

if (require.main === module) {
	main();
}
