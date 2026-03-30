import {
	circlesOverlap,
	computeGunBarrelEnd,
	deriveAimTarget,
	getMoveDelta,
	normalizeAngle,
	rotateAngleTowards,
	tankIntersectsBlast,
} from './geometry';
import { projectileHitsTank, stepProjectile } from './physics';
import { SeededRandom } from './prng';
import { getBombSpec, getProjectileSpec } from './specs';
import { cloneGameState, cloneJson, createReadonlySnapshot, type DeepReadonly, freezeObservation } from './stateUtils';
import type {
	BombStateView,
	ControllerStepRecord,
	GameState,
	MatchInit,
	MoveIntent,
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
const MAX_TURRET_ROTATION_RADIANS_PER_TICK = 0.3;

export class Simulation {
	private state: GameState;
	private controllers: Record<string, TankController>;
	private snapshotCache: DeepReadonly<GameState> | null = null;
	private invalidMoveWarnings = new Set<string>();

	constructor(initialState: GameState, controllers: Record<string, TankController>) {
		this.state = cloneGameState(initialState);
		this.controllers = { ...controllers };
		this.refreshDerivedState();
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
			};
		}

		const observations: Record<string, TankObservation> = {};
		const actions: Record<string, TankAction> = {};
		for (const tank of this.state.tanks) {
			if (tank.destroyed) {
				continue;
			}
			const observation = this.buildObservation(tank.id);
			observations[tank.id] = observation;
			const controller = this.controllers[tank.controllerId];
			actions[tank.id] = this.sanitizeAction(
				controller?.act(observation) ?? { ...NO_OP_ACTION, aimAngle: tank.aimAngle },
				tank
			);
		}

		const events: SimulationEvent[] = [];
		const rng = new SeededRandom(this.state.rngState);
		this.applyActions(actions, events, rng);
		this.state.rngState = rng.getState();
		this.refreshDerivedState();
		this.updateStatus();
		this.snapshotCache = null;
		const records = this.buildRecords(observations, actions);
		const completedTick = this.state.tick;
		this.state.tick += 1;
		return {
			tick: completedTick,
			actions,
			events,
			records,
		};
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

	private buildObservation(tankId: string): TankObservation {
		const self = this.requireTank(tankId);
		const observation = {
			tick: this.state.tick,
			self: cloneJson(self),
			enemies: cloneJson(this.state.tanks.filter((tank) => tank.team !== self.team)),
			projectiles: cloneJson(this.state.projectiles),
			bombs: cloneJson(this.state.bombs),
			obstacles: cloneJson(this.state.obstacles),
			arena: cloneJson(this.state.arena),
		};
		return freezeObservation(observation);
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
			if (tank.shotCooldownTicks > 0) {
				tank.shotCooldownTicks -= 1;
			}
			if (tank.bombCooldownTicks > 0) {
				tank.bombCooldownTicks -= 1;
			}
			const action = actions[tank.id] ?? { ...NO_OP_ACTION, aimAngle: tank.aimAngle };
			const targetAim = normalizeAngle(action.aimAngle);
			tank.aimAngle = rotateAngleTowards(tank.aimAngle, targetAim, MAX_TURRET_ROTATION_RADIANS_PER_TICK);
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
		const barrelEnd = computeGunBarrelEnd(tank);
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
			stepProjectile(projectile, this.state.arena, this.state.obstacles);
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
					this.destroyTank(tank, events, rng);
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
				this.destroyTank(tank, events, rng);
			}
		}
	}

	private tankInBlast(tank: TankStateView, bomb: BombStateView): boolean {
		return tankIntersectsBlast(tank, bomb.x, bomb.y, bomb.blastRadius);
	}

	private getProjectileTargets(projectile: ProjectileStateView): TankStateView[] {
		if (projectile.team === 'enemy') {
			return this.state.tanks.filter((tank) => tank.team === 'player');
		}
		return this.state.tanks;
	}

	private getBombTargets(bomb: BombStateView): TankStateView[] {
		if (bomb.team === 'enemy') {
			return this.state.tanks.filter((tank) => tank.team === 'player');
		}
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
		let reward = 0;
		if (!observation.self.destroyed && postStepTank.destroyed) {
			reward -= 1;
		}
		const livingOpponentsBefore = observation.enemies.filter((enemy) => !enemy.destroyed).length;
		const livingOpponentsAfter = this.state.tanks.filter(
			(tank) => tank.team !== observation.self.team && !tank.destroyed
		).length;
		if (livingOpponentsAfter < livingOpponentsBefore) {
			reward += livingOpponentsBefore - livingOpponentsAfter;
		}
		if (this.state.status === 'player_win') {
			reward += observation.self.team === 'player' ? 10 : -10;
		}
		if (this.state.status === 'enemy_win') {
			reward += observation.self.team === 'enemy' ? 10 : -10;
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
