import type {
	ArenaState,
	BombStateView,
	GameState,
	ObstacleStateView,
	ProjectileStateView,
	TankObservation,
	TankStateView,
} from './types';

export type DeepReadonly<T> = T extends (...args: never[]) => unknown
	? T
	: T extends readonly (infer U)[]
		? readonly DeepReadonly<U>[]
		: T extends object
			? { readonly [K in keyof T]: DeepReadonly<T[K]> }
			: T;

export function cloneJson<T>(value: T): T {
	return JSON.parse(JSON.stringify(value)) as T;
}

function freezeObject<T>(value: T): DeepReadonly<T> {
	if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
		Object.freeze(value);
		for (const nestedValue of Object.values(value as Record<string, unknown>)) {
			freezeObject(nestedValue);
		}
	}
	return value as DeepReadonly<T>;
}

export function freezeObservation<T extends TankObservation>(observation: T): T {
	return freezeObject(observation) as T;
}

export function createReadonlySnapshot<T>(value: T): DeepReadonly<T> {
	return freezeObject(cloneJson(value));
}

export function cloneGameState(state: GameState | DeepReadonly<GameState>): GameState {
	return cloneJson(state) as GameState;
}

export function serializeGameState(state: GameState | DeepReadonly<GameState>): string {
	return JSON.stringify(state);
}

// ── Typed manual-copy functions (Fix 2) ─────────────────────────────────
// Field-by-field copy with no serialization, no prototype-chain walking.

export function copyArena(src: ArenaState, dst: ArenaState): ArenaState {
	dst.width = src.width;
	dst.height = src.height;
	return dst;
}

export function copyObstacle(src: ObstacleStateView, dst: ObstacleStateView): ObstacleStateView {
	dst.id = src.id;
	dst.x = src.x;
	dst.y = src.y;
	dst.width = src.width;
	dst.height = src.height;
	return dst;
}

export function copyTank(src: TankStateView, dst: TankStateView): TankStateView {
	dst.id = src.id;
	dst.controllerId = src.controllerId;
	dst.team = src.team;
	dst.kind = src.kind;
	dst.x = src.x;
	dst.y = src.y;
	dst.size = src.size;
	dst.speed = src.speed;
	dst.color = src.color;
	dst.aimAngle = src.aimAngle;
	dst.aimTargetX = src.aimTargetX;
	dst.aimTargetY = src.aimTargetY;
	dst.health = src.health;
	dst.maxHealth = src.maxHealth;
	dst.invulnerabilityTicksRemaining = src.invulnerabilityTicksRemaining;
	dst.destroyed = src.destroyed;
	dst.ammoType = src.ammoType;
	dst.maxAmmo = src.maxAmmo;
	dst.activeAmmo = src.activeAmmo;
	dst.bombType = src.bombType;
	dst.maxBombs = src.maxBombs;
	dst.activeBombs = src.activeBombs;
	dst.shotCooldownTicks = src.shotCooldownTicks;
	dst.shotCooldownTicksOnFire = src.shotCooldownTicksOnFire;
	dst.bombCooldownTicks = src.bombCooldownTicks;
	dst.bombCooldownTicksOnPlant = src.bombCooldownTicksOnPlant;
	dst.wasLastMoveBlocked = src.wasLastMoveBlocked;
	dst.lastMoveIntent = src.lastMoveIntent;
	dst.consecutiveDirectionMoves = src.consecutiveDirectionMoves;
	dst.aggressionFactor = src.aggressionFactor;
	return dst;
}

export function copyProjectile(src: ProjectileStateView, dst: ProjectileStateView): ProjectileStateView {
	dst.id = src.id;
	dst.ownerTankId = src.ownerTankId;
	dst.team = src.team;
	dst.kind = src.kind;
	dst.x = src.x;
	dst.y = src.y;
	dst.vx = src.vx;
	dst.vy = src.vy;
	dst.speed = src.speed;
	dst.radius = src.radius;
	dst.bounces = src.bounces;
	dst.maxBounces = src.maxBounces;
	return dst;
}

export function copyBomb(src: BombStateView, dst: BombStateView): BombStateView {
	dst.id = src.id;
	dst.ownerTankId = src.ownerTankId;
	dst.team = src.team;
	dst.kind = src.kind;
	dst.x = src.x;
	dst.y = src.y;
	dst.radius = src.radius;
	dst.blastRadius = src.blastRadius;
	dst.fuseTicksRemaining = src.fuseTicksRemaining;
	return dst;
}

// ── Blank-object factories for preallocated buffers (Fix 3) ─────────────

export function blankTank(): TankStateView {
	return {
		id: '',
		controllerId: '',
		team: 'player',
		kind: 'player',
		x: 0,
		y: 0,
		size: 0,
		speed: 0,
		color: '',
		aimAngle: 0,
		aimTargetX: null,
		aimTargetY: null,
		health: 0,
		maxHealth: 0,
		invulnerabilityTicksRemaining: 0,
		destroyed: false,
		ammoType: 'basic',
		maxAmmo: 0,
		activeAmmo: 0,
		bombType: null,
		maxBombs: 0,
		activeBombs: 0,
		shotCooldownTicks: 0,
		shotCooldownTicksOnFire: 0,
		bombCooldownTicks: 0,
		bombCooldownTicksOnPlant: 0,
		wasLastMoveBlocked: false,
		lastMoveIntent: 'none',
		consecutiveDirectionMoves: 0,
		aggressionFactor: 0,
	};
}

export function blankProjectile(): ProjectileStateView {
	return {
		id: '',
		ownerTankId: '',
		team: 'player',
		kind: 'basic',
		x: 0,
		y: 0,
		vx: 0,
		vy: 0,
		speed: 0,
		radius: 0,
		bounces: 0,
		maxBounces: 0,
	};
}

export function blankBomb(): BombStateView {
	return {
		id: '',
		ownerTankId: '',
		team: 'player',
		kind: 'basic',
		x: 0,
		y: 0,
		radius: 0,
		blastRadius: 0,
		fuseTicksRemaining: 0,
	};
}

export function blankObstacle(): ObstacleStateView {
	return { id: '', x: 0, y: 0, width: 0, height: 0 };
}

// ── Preallocated observation buffer (Fix 3) ─────────────────────────────
// Controllers MUST treat the returned observation as read-only and MUST NOT
// hold references to it across ticks. The buffer is overwritten in-place
// every tick to avoid GC pressure during training.

export interface PreallocatedObservation {
	obs: TankObservation;
	/** How many enemy slots are actually populated this tick. */
	enemyCount: number;
	/** How many projectile slots are actually populated this tick. */
	projectileCount: number;
	/** How many bomb slots are actually populated this tick. */
	bombCount: number;
	/** How many obstacle slots are actually populated this tick. */
	obstacleCount: number;
}

export function createPreallocatedObservation(
	maxEnemies: number,
	maxProjectiles: number,
	maxBombs: number,
	maxObstacles: number
): PreallocatedObservation {
	const enemies: TankStateView[] = [];
	for (let i = 0; i < maxEnemies; i++) enemies.push(blankTank());

	const projectiles: ProjectileStateView[] = [];
	for (let i = 0; i < maxProjectiles; i++) projectiles.push(blankProjectile());

	const bombs: BombStateView[] = [];
	for (let i = 0; i < maxBombs; i++) bombs.push(blankBomb());

	const obstacles: ObstacleStateView[] = [];
	for (let i = 0; i < maxObstacles; i++) obstacles.push(blankObstacle());

	return {
		obs: {
			tick: 0,
			self: blankTank(),
			enemies,
			projectiles,
			bombs,
			obstacles,
			arena: { width: 0, height: 0 },
		},
		enemyCount: 0,
		projectileCount: 0,
		bombCount: 0,
		obstacleCount: 0,
	};
}
