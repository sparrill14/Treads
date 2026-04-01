import {
	circlesOverlap,
	computeClearGunBarrelEnd,
	deriveAimTarget,
	getMoveDelta,
	normalizeAngle,
	rotateAngleTowards,
	tankIntersectsBlast,
} from './geometry';
import { projectileHitsTank, stepProjectile } from './physics';
import { SeededRandom } from './prng';
import { getBombSpec, getProjectileSpec } from './specs';
import {
	blankBomb,
	blankObstacle,
	blankProjectile,
	blankTank,
	cloneGameState,
	cloneJson,
	copyArena,
	copyBomb,
	copyObstacle,
	copyProjectile,
	copyTank,
	createReadonlySnapshot,
	type DeepReadonly,
	freezeObservation,
} from './stateUtils';
import type {
	BombStateView,
	ControllerStepRecord,
	GameState,
	MatchInit,
	MoveIntent,
	ObstacleStateView,
	ProjectileStateView,
	SimulationEvent,
	SimulationStepResult,
	TankAction,
	TankController,
	TankObservation,
	TankStateView,
} from './types';
import { MOVE_INTENTS } from './types';

const NO_OP_ACTION: TankAction = {
	move: 'none',
	aimAngle: 0,
	fire: false,
	plantBomb: false,
};

// Prevent degenerate same-origin projectile self-cancellation while preserving intended projectile interactions.
const MIN_SAFE_SHOT_COOLDOWN_TICKS = 3;
const MIN_BOMB_COOLDOWN_TICKS = 15;
const BASE_MAX_TURRET_ROTATION_RADIANS_PER_TICK = 0.3;

interface SharedObservationViews {
	tick: number;
	arena: GameState['arena'];
	obstacles: GameState['obstacles'];
	projectiles: GameState['projectiles'];
	bombs: GameState['bombs'];
	playerTeam: GameState['tanks'];
	enemyTeam: GameState['tanks'];
}

export interface SimulationOptions {
	/** When true, observations are deep-frozen for mutation safety (browser/debug). Default: true. */
	debugFreeze?: boolean;
	/** When true, accumulate per-phase timing stats in step(). Default: false. */
	profiling?: boolean;
}

// Preallocated per-tank observation buffer used when debugFreeze is off.
interface TankObsBuffer {
	obs: TankObservation;
	selfBuf: TankStateView;
	allyBufs: TankStateView[];
	enemyBufs: TankStateView[];
	projBufs: ProjectileStateView[];
	bombBufs: BombStateView[];
	obstacleBufs: ObstacleStateView[];
	arenaBuf: GameState['arena'];
}

// Profiling accumulator
interface ProfilingStats {
	ticks: number;
	observationNs: number;
	controllerNs: number;
	actionNs: number;
	projectileNs: number;
	bombNs: number;
	refreshNs: number;
	statusNs: number;
	recordNs: number;
	totalNs: number;
}

function emptyProfilingStats(): ProfilingStats {
	return {
		ticks: 0,
		observationNs: 0,
		controllerNs: 0,
		actionNs: 0,
		projectileNs: 0,
		bombNs: 0,
		refreshNs: 0,
		statusNs: 0,
		recordNs: 0,
		totalNs: 0,
	};
}

export class Simulation {
	private state: GameState;
	private controllers: Record<string, TankController>;
	private snapshotCache: DeepReadonly<GameState> | null = null;
	private invalidMoveWarnings = new Set<string>();

	// Options
	private readonly debugFreeze: boolean;
	private readonly profiling: boolean;

	// Preallocated observation buffers (Fix 3) — keyed by tank id.
	private obsBuffers = new Map<string, TankObsBuffer>();

	// Shared-views scratch space (reused each step to avoid re-allocation)
	private sharedArena: GameState['arena'] = { width: 0, height: 0 };
	private sharedObstacles: ObstacleStateView[] = [];
	private sharedProjectiles: ProjectileStateView[] = [];
	private sharedBombs: BombStateView[] = [];
	private sharedPlayerTeam: TankStateView[] = [];
	private sharedEnemyTeam: TankStateView[] = [];

	// Profiling
	private profilingStats: ProfilingStats = emptyProfilingStats();

	constructor(initialState: GameState, controllers: Record<string, TankController>, options?: SimulationOptions) {
		this.debugFreeze = options?.debugFreeze ?? true;
		this.profiling = options?.profiling ?? false;
		this.state = cloneGameState(initialState);
		this.controllers = { ...controllers };
		this.refreshDerivedState();
		this.allocateObservationBuffers();
		this.resetControllers();
	}

	public getState(): DeepReadonly<GameState> {
		if (this.snapshotCache === null) {
			this.snapshotCache = createReadonlySnapshot(this.state);
		}
		return this.snapshotCache;
	}

	public getStateSnapshot(): GameState {
		return cloneGameState(this.state);
	}

	public step(): SimulationStepResult {
		if (this.state.status !== 'running') {
			return {
				tick: this.state.tick,
				actions: {},
				events: [],
				records: [],
				replayTankStates: {},
			};
		}

		const profiling = this.profiling;
		let t0 = 0,
			t1 = 0,
			t2 = 0,
			t3 = 0,
			t4 = 0,
			t5 = 0,
			t6 = 0;
		if (profiling) t0 = performance.now();

		// ── observation building ──
		const observations: Record<string, TankObservation> = {};
		const actions: Record<string, TankAction> = {};
		const sharedViews = this.buildSharedObservationViews();
		for (const tank of this.state.tanks) {
			if (tank.destroyed) {
				continue;
			}
			const observation = this.buildObservation(tank.id, sharedViews);
			observations[tank.id] = observation;
		}
		if (profiling) t1 = performance.now();

		// ── controller act() calls ──
		for (const tank of this.state.tanks) {
			if (tank.destroyed) {
				continue;
			}
			const controller = this.controllers[tank.controllerId];
			actions[tank.id] = this.sanitizeAction(
				controller?.act(observations[tank.id]) ?? { ...NO_OP_ACTION, aimAngle: tank.aimAngle },
				tank
			);
		}
		if (profiling) t2 = performance.now();

		// ── action resolution (movement, firing, projectiles, bombs) ──
		const events: SimulationEvent[] = [];
		const rng = new SeededRandom(this.state.rngState);
		this.applyActions(actions, events, rng);
		this.state.rngState = rng.getState();
		if (profiling) t3 = performance.now();

		// ── refresh derived state ──
		this.refreshDerivedState();
		if (profiling) t4 = performance.now();

		// ── update status ──
		this.updateStatus();
		if (profiling) t5 = performance.now();

		this.snapshotCache = null;

		// ── build records ──
		const records = this.buildRecords(observations, actions);
		if (profiling) t6 = performance.now();

		const completedTick = this.state.tick;
		this.state.tick += 1;

		if (profiling) {
			const tEnd = performance.now();
			const s = this.profilingStats;
			s.ticks += 1;
			s.observationNs += t1 - t0;
			s.controllerNs += t2 - t1;
			s.actionNs += t3 - t2;
			s.refreshNs += t4 - t3;
			s.statusNs += t5 - t4;
			s.recordNs += t6 - t5;
			s.totalNs += tEnd - t0;
		}

		return {
			tick: completedTick,
			actions,
			events,
			records,
			replayTankStates: Object.fromEntries(
				this.state.tanks.map((tank) => [
					tank.id,
					{
						health: tank.health,
						maxHealth: tank.maxHealth,
						invulnerabilityTicksRemaining: tank.invulnerabilityTicksRemaining,
						destroyed: tank.destroyed,
					},
				])
			),
		};
	}

	public printProfilingReport(): void {
		const s = this.profilingStats;
		if (s.ticks === 0) {
			console.log('  No profiling data collected.');
			return;
		}
		const pct = (v: number) => ((v / s.totalNs) * 100).toFixed(1);
		const ms = (v: number) => v.toFixed(1);
		console.log(`  Profiling over ${s.ticks} ticks (${ms(s.totalNs)} ms total):`);
		console.log(`    Observation building: ${ms(s.observationNs)} ms (${pct(s.observationNs)}%)`);
		console.log(`    Controller act():     ${ms(s.controllerNs)} ms (${pct(s.controllerNs)}%)`);
		console.log(`    Action resolution:    ${ms(s.actionNs)} ms (${pct(s.actionNs)}%)`);
		console.log(`    Refresh derived:      ${ms(s.refreshNs)} ms (${pct(s.refreshNs)}%)`);
		console.log(`    Status update:        ${ms(s.statusNs)} ms (${pct(s.statusNs)}%)`);
		console.log(`    Record building:      ${ms(s.recordNs)} ms (${pct(s.recordNs)}%)`);
		console.log(`    Per tick avg:         ${(s.totalNs / s.ticks).toFixed(3)} ms`);
	}

	public resetProfiling(): void {
		this.profilingStats = emptyProfilingStats();
	}

	private buildSharedObservationViews(): SharedObservationViews {
		if (this.debugFreeze) {
			// Legacy path: full deep-clone for mutation safety when freeze is on
			return {
				tick: this.state.tick,
				arena: cloneJson(this.state.arena),
				obstacles: cloneJson(this.state.obstacles),
				projectiles: cloneJson(this.state.projectiles),
				bombs: cloneJson(this.state.bombs),
				playerTeam: cloneJson(this.state.tanks.filter((tank) => tank.team === 'player')),
				enemyTeam: cloneJson(this.state.tanks.filter((tank) => tank.team === 'enemy')),
			};
		}

		// Optimized path: typed copy into reusable scratch arrays
		copyArena(this.state.arena, this.sharedArena);

		const obstacles = this.state.obstacles;
		this.ensureArrayCapacity(this.sharedObstacles, obstacles.length, blankObstacle);
		for (let i = 0; i < obstacles.length; i++) {
			copyObstacle(obstacles[i], this.sharedObstacles[i]);
		}

		const projectiles = this.state.projectiles;
		this.ensureArrayCapacity(this.sharedProjectiles, projectiles.length, blankProjectile);
		for (let i = 0; i < projectiles.length; i++) {
			copyProjectile(projectiles[i], this.sharedProjectiles[i]);
		}

		const bombs = this.state.bombs;
		this.ensureArrayCapacity(this.sharedBombs, bombs.length, blankBomb);
		for (let i = 0; i < bombs.length; i++) {
			copyBomb(bombs[i], this.sharedBombs[i]);
		}

		let pCount = 0,
			eCount = 0;
		for (const tank of this.state.tanks) {
			if (tank.team === 'player') {
				this.ensureArrayCapacity(this.sharedPlayerTeam, pCount + 1, blankTank);
				copyTank(tank, this.sharedPlayerTeam[pCount++]);
			} else {
				this.ensureArrayCapacity(this.sharedEnemyTeam, eCount + 1, blankTank);
				copyTank(tank, this.sharedEnemyTeam[eCount++]);
			}
		}

		return {
			tick: this.state.tick,
			arena: this.sharedArena,
			obstacles: this.sharedObstacles.slice(0, obstacles.length),
			projectiles: this.sharedProjectiles.slice(0, projectiles.length),
			bombs: this.sharedBombs.slice(0, bombs.length),
			playerTeam: this.sharedPlayerTeam.slice(0, pCount),
			enemyTeam: this.sharedEnemyTeam.slice(0, eCount),
		};
	}

	/** Grow a preallocated array if needed (never shrink). */
	private ensureArrayCapacity<T>(arr: T[], needed: number, factory: () => T): void {
		while (arr.length < needed) {
			arr.push(factory());
		}
	}

	/** Preallocate one observation buffer per live tank (Fix 3). */
	private allocateObservationBuffers(): void {
		const maxEnemies = this.state.tanks.length;
		const maxProjectiles = 32; // generous cap — grows if needed
		const maxBombs = 16;
		const maxObstacles = this.state.obstacles.length;

		for (const tank of this.state.tanks) {
			const selfBuf = blankTank();
			const enemyBufs: TankStateView[] = [];
			for (let i = 0; i < maxEnemies; i++) enemyBufs.push(blankTank());
			const allyBufs: TankStateView[] = [];
			for (let i = 0; i < maxEnemies; i++) allyBufs.push(blankTank());
			const projBufs: ProjectileStateView[] = [];
			for (let i = 0; i < maxProjectiles; i++) projBufs.push(blankProjectile());
			const bombBufs: BombStateView[] = [];
			for (let i = 0; i < maxBombs; i++) bombBufs.push(blankBomb());
			const obstacleBufs: ObstacleStateView[] = [];
			for (let i = 0; i < maxObstacles; i++) obstacleBufs.push(blankObstacle());
			const arenaBuf = { width: 0, height: 0 };

			this.obsBuffers.set(tank.id, {
				obs: {
					tick: 0,
					self: selfBuf,
					allies: allyBufs,
					enemies: enemyBufs,
					projectiles: projBufs,
					bombs: bombBufs,
					obstacles: obstacleBufs,
					arena: arenaBuf,
				},
				selfBuf,
				allyBufs,
				enemyBufs,
				projBufs,
				bombBufs,
				obstacleBufs,
				arenaBuf,
			});
		}
	}

	private resetControllers(): void {
		for (const tank of this.state.tanks) {
			const controller = this.controllers[tank.controllerId];
			controller?.reset?.(this.createMatchInit(tank.id, tank.controllerId));
		}
	}

	private createMatchInit(selfId: string, controllerId: string): MatchInit {
		return {
			seed: this.state.seed,
			tickRate: this.state.tickRate,
			arena: cloneJson(this.state.arena),
			obstacles: cloneJson(this.state.obstacles),
			tanks: cloneJson(this.state.tanks),
			selfId,
			controllerId,
		};
	}

	private buildObservation(tankId: string, sharedViews: SharedObservationViews): TankObservation {
		const self = this.requireTank(tankId);

		if (this.debugFreeze) {
			// Legacy path: clone self, freeze entire observation for mutation safety.
			const observation: TankObservation = {
				tick: sharedViews.tick,
				self: cloneJson(self),
				allies: cloneJson(
					(self.team === 'player' ? sharedViews.playerTeam : sharedViews.enemyTeam).filter(
						(tank) => tank.id !== self.id
					)
				),
				enemies: self.team === 'player' ? sharedViews.enemyTeam : sharedViews.playerTeam,
				projectiles: sharedViews.projectiles,
				bombs: sharedViews.bombs,
				obstacles: sharedViews.obstacles,
				arena: sharedViews.arena,
			};
			return freezeObservation(observation);
		}

		// Optimized path: copy into preallocated buffer (Fix 3).
		// Controllers MUST treat the returned observation as read-only and
		// MUST NOT hold references to it across ticks.
		const buf = this.obsBuffers.get(tankId);
		if (!buf) throw new Error(`No observation buffer for tank: ${tankId}`);
		buf.obs.tick = sharedViews.tick;
		copyTank(self, buf.selfBuf);

		const enemies = self.team === 'player' ? sharedViews.enemyTeam : sharedViews.playerTeam;
		this.ensureArrayCapacity(buf.enemyBufs, enemies.length, blankTank);
		for (let i = 0; i < enemies.length; i++) {
			copyTank(enemies[i], buf.enemyBufs[i]);
		}
		buf.obs.enemies = buf.enemyBufs.slice(0, enemies.length);

		const sameTeam = self.team === 'player' ? sharedViews.playerTeam : sharedViews.enemyTeam;
		this.ensureArrayCapacity(buf.allyBufs, sameTeam.length, blankTank);
		let allyCount = 0;
		for (const teammate of sameTeam) {
			if (teammate.id === self.id) {
				continue;
			}
			copyTank(teammate, buf.allyBufs[allyCount]);
			allyCount += 1;
		}
		buf.obs.allies = buf.allyBufs.slice(0, allyCount);

		const projs = sharedViews.projectiles;
		this.ensureArrayCapacity(buf.projBufs, projs.length, blankProjectile);
		for (let i = 0; i < projs.length; i++) {
			copyProjectile(projs[i], buf.projBufs[i]);
		}
		buf.obs.projectiles = buf.projBufs.slice(0, projs.length);

		const bombs = sharedViews.bombs;
		this.ensureArrayCapacity(buf.bombBufs, bombs.length, blankBomb);
		for (let i = 0; i < bombs.length; i++) {
			copyBomb(bombs[i], buf.bombBufs[i]);
		}
		buf.obs.bombs = buf.bombBufs.slice(0, bombs.length);

		const obstacles = sharedViews.obstacles;
		this.ensureArrayCapacity(buf.obstacleBufs, obstacles.length, blankObstacle);
		for (let i = 0; i < obstacles.length; i++) {
			copyObstacle(obstacles[i], buf.obstacleBufs[i]);
		}
		buf.obs.obstacles = buf.obstacleBufs.slice(0, obstacles.length);

		copyArena(sharedViews.arena, buf.arenaBuf);

		return buf.obs;
	}

	private sanitizeAction(action: TankAction, tank: TankStateView): TankAction {
		const aimAngle = Number.isFinite(action.aimAngle) ? normalizeAngle(action.aimAngle) : tank.aimAngle;
		const move = MOVE_INTENTS.includes(action.move) ? action.move : this.sanitizeInvalidMove(action.move, tank);
		return {
			move,
			aimAngle,
			fire: Boolean(action.fire),
			plantBomb: Boolean(action.plantBomb),
			aimTarget: action.aimTarget ? { x: action.aimTarget.x, y: action.aimTarget.y } : undefined,
		};
	}

	private sanitizeInvalidMove(move: unknown, tank: TankStateView): MoveIntent {
		const warningKey = `${tank.controllerId}:${String(move)}`;
		if (!this.invalidMoveWarnings.has(warningKey)) {
			this.invalidMoveWarnings.add(warningKey);
			console.warn(`Invalid move intent "${String(move)}" from controller ${tank.controllerId}; clamping to "none".`);
		}
		return 'none';
	}

	private applyActions(actions: Record<string, TankAction>, events: SimulationEvent[], rng: SeededRandom): void {
		for (const tank of this.state.tanks) {
			if (tank.destroyed) {
				continue;
			}
			if (tank.invulnerabilityTicksRemaining > 0) {
				tank.invulnerabilityTicksRemaining -= 1;
			}
			if (tank.shotCooldownTicks > 0) {
				tank.shotCooldownTicks -= 1;
			}
			if (tank.bombCooldownTicks > 0) {
				tank.bombCooldownTicks -= 1;
			}
			const action = actions[tank.id] ?? { ...NO_OP_ACTION, aimAngle: tank.aimAngle };
			const targetAim = normalizeAngle(action.aimAngle);
			tank.aimAngle = rotateAngleTowards(
				tank.aimAngle,
				targetAim,
				BASE_MAX_TURRET_ROTATION_RADIANS_PER_TICK * this.state.rules.turretSpeedMultiplier
			);
			if (action.aimTarget) {
				tank.aimTargetX = action.aimTarget.x;
				tank.aimTargetY = action.aimTarget.y;
			} else {
				const derived = deriveAimTarget(
					targetAim,
					tank.x + tank.size / 2,
					tank.y + tank.size / 2,
					this.state.arena,
					Math.max(this.state.arena.width, this.state.arena.height)
				);
				tank.aimTargetX = derived.x;
				tank.aimTargetY = derived.y;
			}
			tank.wasLastMoveBlocked = false;
			this.moveTank(tank, action.move);
		}

		for (const tank of this.state.tanks) {
			if (tank.destroyed) {
				continue;
			}
			const action = actions[tank.id] ?? NO_OP_ACTION;
			if (action.fire) {
				this.fireProjectile(tank, events);
			}
			if (action.plantBomb) {
				this.plantBomb(tank, events);
			}
		}

		this.resolveProjectiles(events, rng);
		this.resolveBombs(events, rng);
	}

	private fireProjectile(tank: TankStateView, events: SimulationEvent[]): void {
		if (tank.shotCooldownTicks > 0 || tank.activeAmmo >= tank.maxAmmo) {
			return;
		}
		const projectileSpec = getProjectileSpec(tank.ammoType);
		const barrelEnd = computeClearGunBarrelEnd(tank, this.state.obstacles, this.state.arena);
		const projectile: ProjectileStateView = {
			id: `projectile-${this.state.nextEntityId++}`,
			ownerTankId: tank.id,
			team: tank.team,
			kind: tank.ammoType,
			x: barrelEnd.x,
			y: barrelEnd.y,
			vx: Math.cos(tank.aimAngle) * projectileSpec.speed,
			vy: Math.sin(tank.aimAngle) * projectileSpec.speed,
			speed: projectileSpec.speed,
			radius: projectileSpec.radius,
			bounces: 0,
			maxBounces: projectileSpec.maxBounces,
		};
		this.state.projectiles.push(projectile);
		tank.shotCooldownTicks = Math.max(tank.shotCooldownTicksOnFire, MIN_SAFE_SHOT_COOLDOWN_TICKS);
		events.push({
			type: 'projectile-fired',
			tick: this.state.tick,
			tankId: tank.id,
			projectileId: projectile.id,
			x: projectile.x,
			y: projectile.y,
			audioCue: 'tank-fire',
		});
	}

	private plantBomb(tank: TankStateView, events: SimulationEvent[]): void {
		if (!tank.bombType || tank.bombCooldownTicks > 0 || tank.activeBombs >= tank.maxBombs) {
			return;
		}
		const bombSpec = getBombSpec(tank.bombType);
		const bomb: BombStateView = {
			id: `bomb-${this.state.nextEntityId++}`,
			ownerTankId: tank.id,
			team: tank.team,
			kind: tank.bombType,
			x: tank.x + tank.size / 2,
			y: tank.y + tank.size / 2,
			radius: bombSpec.radius,
			blastRadius: bombSpec.blastRadius,
			fuseTicksRemaining: bombSpec.fuseTicks,
		};
		this.state.bombs.push(bomb);
		tank.bombCooldownTicks = Math.max(tank.bombCooldownTicksOnPlant, MIN_BOMB_COOLDOWN_TICKS);
		events.push({
			type: 'bomb-planted',
			tick: this.state.tick,
			bombId: bomb.id,
			tankId: tank.id,
			x: bomb.x,
			y: bomb.y,
		});
	}

	private resolveProjectiles(events: SimulationEvent[], rng: SeededRandom): void {
		const destroyedProjectileIds = new Set<string>();
		const bombIdsToExplode = new Set<string>();
		for (const projectile of this.state.projectiles) {
			if (destroyedProjectileIds.has(projectile.id)) {
				continue;
			}
			stepProjectile(projectile, this.state.arena, this.state.obstacles, this.state.rules.projectileBounces);
			if (projectile.bounces > projectile.maxBounces) {
				this.markProjectileDestroyed(projectile, destroyedProjectileIds, events, rng);
			}
		}

		for (let index = 0; index < this.state.projectiles.length; index++) {
			const projectile = this.state.projectiles[index];
			if (destroyedProjectileIds.has(projectile.id)) {
				continue;
			}
			for (let otherIndex = index + 1; otherIndex < this.state.projectiles.length; otherIndex++) {
				const otherProjectile = this.state.projectiles[otherIndex];
				if (destroyedProjectileIds.has(otherProjectile.id)) {
					continue;
				}
				if (
					circlesOverlap(
						projectile.x,
						projectile.y,
						projectile.radius,
						otherProjectile.x,
						otherProjectile.y,
						otherProjectile.radius
					)
				) {
					this.markProjectileDestroyed(projectile, destroyedProjectileIds, events, rng);
					this.markProjectileDestroyed(otherProjectile, destroyedProjectileIds, events, rng);
					break;
				}
			}
		}

		for (const projectile of this.state.projectiles) {
			if (destroyedProjectileIds.has(projectile.id)) {
				continue;
			}
			for (const bomb of this.state.bombs) {
				if (bombIdsToExplode.has(bomb.id)) {
					continue;
				}
				if (circlesOverlap(projectile.x, projectile.y, projectile.radius, bomb.x, bomb.y, bomb.radius)) {
					this.markProjectileDestroyed(projectile, destroyedProjectileIds, events, rng);
					bombIdsToExplode.add(bomb.id);
					break;
				}
			}
		}

		for (const projectile of this.state.projectiles) {
			if (destroyedProjectileIds.has(projectile.id)) {
				continue;
			}
			for (const tank of this.getProjectileTargets(projectile)) {
				if (!tank.destroyed && projectileHitsTank(projectile, tank)) {
					this.markProjectileDestroyed(projectile, destroyedProjectileIds, events, rng);
					this.applyDamage(tank, this.state.rules.projectileDamage, events, rng);
					break;
				}
			}
		}

		this.state.projectiles = this.state.projectiles.filter((projectile) => !destroyedProjectileIds.has(projectile.id));
		if (bombIdsToExplode.size > 0) {
			for (const bomb of this.state.bombs.filter((candidate) => bombIdsToExplode.has(candidate.id))) {
				this.explodeBomb(bomb, events, rng);
			}
			this.state.bombs = this.state.bombs.filter((bomb) => !bombIdsToExplode.has(bomb.id));
		}
	}

	private resolveBombs(events: SimulationEvent[], rng: SeededRandom): void {
		const explodedBombIds = new Set<string>();
		for (const bomb of this.state.bombs) {
			bomb.fuseTicksRemaining -= 1;
			if (bomb.fuseTicksRemaining <= 0) {
				explodedBombIds.add(bomb.id);
				this.explodeBomb(bomb, events, rng);
			}
		}
		if (explodedBombIds.size > 0) {
			this.state.bombs = this.state.bombs.filter((bomb) => !explodedBombIds.has(bomb.id));
		}
	}

	private explodeBomb(bomb: BombStateView, events: SimulationEvent[], rng: SeededRandom): void {
		events.push({
			type: 'bomb-exploded',
			tick: this.state.tick,
			bombId: bomb.id,
			tankId: bomb.ownerTankId,
			x: bomb.x,
			y: bomb.y,
			blastRadius: bomb.blastRadius,
			visualSeed: rng.nextUint32(),
			audioCue: 'bomb-explode',
		});
		for (const tank of this.getBombTargets(bomb)) {
			if (!tank.destroyed && this.tankInBlast(tank, bomb)) {
				this.applyDamage(tank, this.state.rules.bombDamage, events, rng);
			}
		}
	}

	private applyDamage(tank: TankStateView, amount: number, events: SimulationEvent[], rng: SeededRandom): void {
		if (tank.destroyed || amount <= 0 || tank.invulnerabilityTicksRemaining > 0) {
			return;
		}
		tank.health = Math.max(0, tank.health - amount);
		tank.invulnerabilityTicksRemaining = this.state.rules.invulnerabilityTicks;
		if (tank.health <= 0) {
			this.destroyTank(tank, events, rng);
		}
	}

	private tankInBlast(tank: TankStateView, bomb: BombStateView): boolean {
		return tankIntersectsBlast(tank, bomb.x, bomb.y, bomb.blastRadius);
	}

	private getProjectileTargets(_projectile: ProjectileStateView): TankStateView[] {
		return this.state.tanks;
	}

	private getBombTargets(_bomb: BombStateView): TankStateView[] {
		return this.state.tanks;
	}

	private markProjectileDestroyed(
		projectile: ProjectileStateView,
		destroyedProjectileIds: Set<string>,
		events: SimulationEvent[],
		rng: SeededRandom
	): void {
		if (destroyedProjectileIds.has(projectile.id)) {
			return;
		}
		destroyedProjectileIds.add(projectile.id);
		events.push({
			type: 'projectile-destroyed',
			tick: this.state.tick,
			projectileId: projectile.id,
			x: projectile.x,
			y: projectile.y,
			visualSeed: rng.nextUint32(),
			audioCue: 'ammunition-explode',
		});
	}

	private destroyTank(tank: TankStateView, events: SimulationEvent[], rng: SeededRandom): void {
		if (tank.destroyed) {
			return;
		}
		tank.health = 0;
		tank.invulnerabilityTicksRemaining = 0;
		tank.destroyed = true;
		events.push({
			type: 'tank-destroyed',
			tick: this.state.tick,
			tankId: tank.id,
			x: tank.x + tank.size / 2,
			y: tank.y + tank.size / 2,
			visualSeed: rng.nextUint32(),
			audioCue: 'tank-destroy',
		});
	}

	private buildRecords(
		observations: Record<string, TankObservation>,
		actions: Record<string, TankAction>
	): ControllerStepRecord[] {
		return Object.entries(observations).map(([tankId, observation]) => {
			const postStepTank = this.requireTank(tankId);
			const reward = this.calculateReward(observation, postStepTank);
			return {
				controllerId: observation.self.controllerId,
				tankId,
				observation,
				action: actions[tankId],
				reward,
				done: postStepTank.destroyed || this.state.status !== 'running',
			};
		});
	}

	private calculateReward(observation: TankObservation, postStepTank: TankStateView): number {
		let reward = -0.005;
		const selfDamageTaken = Math.max(0, observation.self.health - postStepTank.health);
		if (selfDamageTaken > 0) {
			reward -= 0.3 * selfDamageTaken;
		}
		if (!observation.self.destroyed && postStepTank.destroyed) {
			reward -= 2;
		}
		const opponentHealthBefore = observation.enemies
			.filter((enemy) => !enemy.destroyed)
			.reduce((total, enemy) => total + enemy.health, 0);
		const livingOpponentsBefore = observation.enemies.filter((enemy) => !enemy.destroyed).length;
		const opponentHealthAfter = this.state.tanks
			.filter((tank) => tank.team !== observation.self.team && !tank.destroyed)
			.reduce((total, tank) => total + tank.health, 0);
		const livingOpponentsAfter = this.state.tanks.filter(
			(tank) => tank.team !== observation.self.team && !tank.destroyed
		).length;
		if (opponentHealthAfter < opponentHealthBefore) {
			reward += 0.3 * (opponentHealthBefore - opponentHealthAfter);
		}
		if (livingOpponentsAfter < livingOpponentsBefore) {
			reward += 2 * (livingOpponentsBefore - livingOpponentsAfter);
		}
		if (this.state.status === 'player_win') {
			reward += observation.self.team === 'player' ? 5 : -3;
		}
		if (this.state.status === 'enemy_win') {
			reward += observation.self.team === 'enemy' ? 5 : -3;
		}
		return reward;
	}

	private updateStatus(): void {
		const playerTank = this.requireTank(this.state.playerTankId);
		if (playerTank.destroyed) {
			this.state.status = 'enemy_win';
			return;
		}
		const allEnemiesDestroyed = this.state.tanks
			.filter((tank) => tank.team === 'enemy')
			.every((tank) => tank.destroyed);
		this.state.status = allEnemiesDestroyed ? 'player_win' : 'running';
	}

	private refreshDerivedState(): void {
		for (const tank of this.state.tanks) {
			tank.activeAmmo = 0;
			tank.activeBombs = 0;
		}
		for (const projectile of this.state.projectiles) {
			const owner = this.state.tanks.find((tank) => tank.id === projectile.ownerTankId);
			if (owner) {
				owner.activeAmmo += 1;
			}
		}
		for (const bomb of this.state.bombs) {
			const owner = this.state.tanks.find((tank) => tank.id === bomb.ownerTankId);
			if (owner) {
				owner.activeBombs += 1;
			}
		}
	}

	private moveTank(tank: TankStateView, moveIntent: MoveIntent): void {
		if (moveIntent === 'none') {
			tank.lastMoveIntent = 'none';
			tank.consecutiveDirectionMoves = 0;
			return;
		}
		if (tank.lastMoveIntent === moveIntent) {
			tank.consecutiveDirectionMoves += 1;
		} else {
			tank.consecutiveDirectionMoves = 0;
		}
		tank.lastMoveIntent = moveIntent;
		const move = getMoveDelta(moveIntent, tank.speed, 1 / this.state.tickRate);
		switch (moveIntent) {
			case 'n':
				this.moveNorth(tank, Math.abs(move.dy));
				break;
			case 's':
				this.moveSouth(tank, Math.abs(move.dy));
				break;
			case 'e':
				this.moveEast(tank, Math.abs(move.dx));
				break;
			case 'w':
				this.moveWest(tank, Math.abs(move.dx));
				break;
			case 'ne':
				this.moveNorthEast(tank, Math.abs(move.dx), Math.abs(move.dy));
				break;
			case 'nw':
				this.moveNorthWest(tank, Math.abs(move.dx), Math.abs(move.dy));
				break;
			case 'se':
				this.moveSouthEast(tank, Math.abs(move.dx), Math.abs(move.dy));
				break;
			case 'sw':
				this.moveSouthWest(tank, Math.abs(move.dx), Math.abs(move.dy));
				break;
		}
	}

	private moveNorth(tank: TankStateView, moveY: number): void {
		let blocked = false;
		for (const obstacle of this.state.obstacles) {
			if (
				tank.y - moveY < obstacle.y + obstacle.height &&
				tank.y > obstacle.y &&
				obstacle.x < tank.x + tank.size &&
				tank.x < obstacle.x + obstacle.width
			) {
				tank.y = obstacle.y + obstacle.height;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			tank.y = Math.max(tank.y - moveY, 0);
		} else {
			tank.wasLastMoveBlocked = true;
		}
	}

	private moveSouth(tank: TankStateView, moveY: number): void {
		let blocked = false;
		for (const obstacle of this.state.obstacles) {
			if (
				tank.y + moveY + tank.size > obstacle.y &&
				tank.y < obstacle.y + obstacle.height &&
				obstacle.x < tank.x + tank.size &&
				tank.x < obstacle.x + obstacle.width
			) {
				tank.y = obstacle.y - tank.size;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			tank.y = Math.min(tank.y + moveY, this.state.arena.height - tank.size);
		} else {
			tank.wasLastMoveBlocked = true;
		}
	}

	private moveEast(tank: TankStateView, moveX: number): void {
		let blocked = false;
		for (const obstacle of this.state.obstacles) {
			if (
				tank.x + moveX + tank.size > obstacle.x &&
				tank.x < obstacle.x + obstacle.width &&
				obstacle.y < tank.y + tank.size &&
				tank.y < obstacle.y + obstacle.height
			) {
				tank.x = obstacle.x - tank.size;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			tank.x = Math.min(tank.x + moveX, this.state.arena.width - tank.size);
		} else {
			tank.wasLastMoveBlocked = true;
		}
	}

	private moveWest(tank: TankStateView, moveX: number): void {
		let blocked = false;
		for (const obstacle of this.state.obstacles) {
			if (
				tank.x - moveX < obstacle.x + obstacle.width &&
				tank.x > obstacle.x &&
				obstacle.y < tank.y + tank.size &&
				tank.y < obstacle.y + obstacle.height
			) {
				tank.x = obstacle.x + obstacle.width;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			tank.x = Math.max(tank.x - moveX, 0);
		} else {
			tank.wasLastMoveBlocked = true;
		}
	}

	private moveNorthEast(tank: TankStateView, moveX: number, moveY: number): void {
		const blockedNorth = this.tryMoveNorthAxis(tank, moveY);
		const blockedEast = this.tryMoveEastAxis(tank, moveX);
		tank.wasLastMoveBlocked = blockedNorth && blockedEast;
	}

	private moveNorthWest(tank: TankStateView, moveX: number, moveY: number): void {
		const blockedNorth = this.tryMoveNorthAxis(tank, moveY);
		const blockedWest = this.tryMoveWestAxis(tank, moveX);
		tank.wasLastMoveBlocked = blockedNorth && blockedWest;
	}

	private moveSouthEast(tank: TankStateView, moveX: number, moveY: number): void {
		const blockedSouth = this.tryMoveSouthAxis(tank, moveY);
		const blockedEast = this.tryMoveEastAxis(tank, moveX);
		tank.wasLastMoveBlocked = blockedSouth && blockedEast;
	}

	private moveSouthWest(tank: TankStateView, moveX: number, moveY: number): void {
		const blockedSouth = this.tryMoveSouthAxis(tank, moveY);
		const blockedWest = this.tryMoveWestAxis(tank, moveX);
		tank.wasLastMoveBlocked = blockedSouth && blockedWest;
	}

	private tryMoveNorthAxis(tank: TankStateView, moveY: number): boolean {
		for (const obstacle of this.state.obstacles) {
			if (
				tank.y - moveY < obstacle.y + obstacle.height &&
				tank.y > obstacle.y &&
				obstacle.x < tank.x + tank.size &&
				tank.x < obstacle.x + obstacle.width
			) {
				tank.y = obstacle.y + obstacle.height;
				return true;
			}
		}
		tank.y = Math.max(tank.y - moveY, 0);
		return false;
	}

	private tryMoveSouthAxis(tank: TankStateView, moveY: number): boolean {
		for (const obstacle of this.state.obstacles) {
			if (
				tank.y + moveY + tank.size > obstacle.y &&
				tank.y < obstacle.y + obstacle.height &&
				obstacle.x < tank.x + tank.size &&
				tank.x < obstacle.x + obstacle.width
			) {
				tank.y = obstacle.y - tank.size;
				return true;
			}
		}
		tank.y = Math.min(tank.y + moveY, this.state.arena.height - tank.size);
		return false;
	}

	private tryMoveEastAxis(tank: TankStateView, moveX: number): boolean {
		for (const obstacle of this.state.obstacles) {
			if (
				tank.x + moveX + tank.size > obstacle.x &&
				tank.x < obstacle.x + obstacle.width &&
				obstacle.y < tank.y + tank.size &&
				tank.y < obstacle.y + obstacle.height
			) {
				tank.x = obstacle.x - tank.size;
				return true;
			}
		}
		tank.x = Math.min(tank.x + moveX, this.state.arena.width - tank.size);
		return false;
	}

	private tryMoveWestAxis(tank: TankStateView, moveX: number): boolean {
		for (const obstacle of this.state.obstacles) {
			if (
				tank.x - moveX < obstacle.x + obstacle.width &&
				tank.x > obstacle.x &&
				obstacle.y < tank.y + tank.size &&
				tank.y < obstacle.y + obstacle.height
			) {
				tank.x = obstacle.x + obstacle.width;
				return true;
			}
		}
		tank.x = Math.max(tank.x - moveX, 0);
		return false;
	}

	private requireTank(tankId: string): TankStateView {
		const tank = this.state.tanks.find((candidate) => candidate.id === tankId);
		if (!tank) {
			throw new Error(`Tank not found: ${tankId}`);
		}
		return tank;
	}
}
