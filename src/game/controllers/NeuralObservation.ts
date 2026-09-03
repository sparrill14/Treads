import type { MoveIntent, TankObservation } from '../core/types';
import { NavigationPlanner } from '../navigation/NavigationPlanner';
import { NEURAL_MODEL_CONTRACT, OBS_SIZE, unitAngleFeature } from './NeuralModelContract';

const ARENA_WIDTH = 1000;
const ARENA_HEIGHT = 500;
const ARENA_DIAGONAL = Math.sqrt(ARENA_WIDTH * ARENA_WIDTH + ARENA_HEIGHT * ARENA_HEIGHT);
const MAX_ENEMIES = NEURAL_MODEL_CONTRACT.observation.maxEnemies;
const MAX_PROJECTILES = NEURAL_MODEL_CONTRACT.observation.maxProjectiles;
const MAX_OBSTACLES = NEURAL_MODEL_CONTRACT.observation.maxObstacles;
const MAX_BOMBS = NEURAL_MODEL_CONTRACT.observation.maxBombs;
const SELF_DIM = NEURAL_MODEL_CONTRACT.observation.selfDim;
const ENEMY_DIM = NEURAL_MODEL_CONTRACT.observation.enemyDim;
const PROJ_DIM = NEURAL_MODEL_CONTRACT.observation.projectileDim;
const OBSTACLE_DIM = NEURAL_MODEL_CONTRACT.observation.obstacleDim;
const BOMB_DIM = NEURAL_MODEL_CONTRACT.observation.bombDim;
const MAX_FUSE_TICKS = 360;
const MAX_BLAST_RADIUS = 100;
const PROJECTILE_SPEED_NORM = 300;

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
		if (t1 > t2) [t1, t2] = [t2, t1];
		tMin = Math.max(tMin, t1);
		tMax = Math.min(tMax, t2);
		if (tMin > tMax) return false;
	}
	if (Math.abs(dy) < 1e-10) {
		if (y1 < ry || y1 > ry + rh) return false;
	} else {
		let t1 = (ry - y1) / dy;
		let t2 = (ry + rh - y1) / dy;
		if (t1 > t2) [t1, t2] = [t2, t1];
		tMin = Math.max(tMin, t1);
		tMax = Math.min(tMax, t2);
		if (tMin > tMax) return false;
	}
	return true;
}

function hasLineOfSight(obs: TankObservation, sx: number, sy: number, tx: number, ty: number): boolean {
	return !obs.obstacles.some((obstacle) =>
		segmentIntersectsRect(
			sx,
			sy,
			tx,
			ty,
			obstacle.x,
			obstacle.y,
			obstacle.width,
			obstacle.height
		)
	);
}

/**
 * Encode the canonical policy observation used by rollout workers and browser inference.
 * Features 18-21 expose the first A* step, remaining path distance, and reachability
 * for the nearest live enemy so obstacle routing is directly observable.
 */
export function normalizePolicyObservation(
	obs: TankObservation,
	maxTicks = 720,
	navigationPlanner?: NavigationPlanner | null
): number[] {
	const result = new Array<number>(OBS_SIZE).fill(0);
	let idx = 0;
	const self = obs.self;
	const sx = self.x + self.size / 2;
	const sy = self.y + self.size / 2;
	const livingEnemies = obs.enemies
		.filter((enemy) => !enemy.destroyed)
		.sort((a, b) => {
			const da = (a.x + a.size / 2 - sx) ** 2 + (a.y + a.size / 2 - sy) ** 2;
			const db = (b.x + b.size / 2 - sx) ** 2 + (b.y + b.size / 2 - sy) ** 2;
			return da - db;
		});

	result[idx] = self.x / ARENA_WIDTH;
	result[idx + 1] = self.y / ARENA_HEIGHT;
	[result[idx + 2], result[idx + 3]] = unitAngleFeature(self.aimAngle);
	result[idx + 4] = self.speed / 100;
	result[idx + 5] =
		livingEnemies.length > 0 &&
		hasLineOfSight(
			obs,
			sx,
			sy,
			livingEnemies[0].x + livingEnemies[0].size / 2,
			livingEnemies[0].y + livingEnemies[0].size / 2
		)
			? 1
			: 0;
	result[idx + 6] = self.wasLastMoveBlocked ? 1 : 0;
	result[idx + 7] = Math.min(self.invulnerabilityTicksRemaining / 8, 1);
	result[idx + 8] = Math.min(obs.tick / Math.max(maxTicks, 1), 1);
	result[idx + 9] = self.health / Math.max(self.maxHealth, 1);
	result[idx + 10] = self.activeAmmo / Math.max(self.maxAmmo, 1);
	result[idx + 11] = self.shotCooldownTicks / Math.max(self.shotCooldownTicksOnFire, 1);
	result[idx + 12] = self.activeBombs / Math.max(self.maxBombs, 1);

	if (livingEnemies.length > 0) {
		const nearest = livingEnemies[0];
		const ex = nearest.x + nearest.size / 2;
		const ey = nearest.y + nearest.size / 2;
		const angleToEnemy = Math.atan2(ey - sy, ex - sx);
		const distanceToEnemy = Math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2);
		const aimError = Math.atan2(Math.sin(self.aimAngle - angleToEnemy), Math.cos(self.aimAngle - angleToEnemy));
		[result[idx + 13], result[idx + 14]] = unitAngleFeature(angleToEnemy);
		result[idx + 15] = Math.min(distanceToEnemy / ARENA_DIAGONAL, 1);
		[result[idx + 16], result[idx + 17]] = unitAngleFeature(aimError);

		const planner = navigationPlanner ?? new NavigationPlanner(obs.arena, obs.obstacles, self.size);
		const guidance = planner.getPathGuidance(sx, sy, ex, ey);
		const navigationAngle = Math.atan2(guidance.nextY - sy, guidance.nextX - sx);
		[result[idx + 18], result[idx + 19]] = unitAngleFeature(navigationAngle);
		result[idx + 20] = Math.min(guidance.distance / ARENA_DIAGONAL, 1);
		result[idx + 21] = guidance.reachable ? 1 : 0;
	} else {
		result[idx + 13] = 0.5;
		result[idx + 14] = 1;
		result[idx + 15] = 0;
		result[idx + 16] = 0.5;
		result[idx + 17] = 1;
		result[idx + 18] = 0.5;
		result[idx + 19] = 1;
		result[idx + 20] = 0;
		result[idx + 21] = 0;
	}
	idx += SELF_DIM;

	for (let i = 0; i < MAX_ENEMIES; i++) {
		if (i < livingEnemies.length) {
			const enemy = livingEnemies[i];
			const ex = enemy.x + enemy.size / 2;
			const ey = enemy.y + enemy.size / 2;
			result[idx] = ((ex - sx) / ARENA_DIAGONAL) * 0.5 + 0.5;
			result[idx + 1] = ((ey - sy) / ARENA_DIAGONAL) * 0.5 + 0.5;
			[result[idx + 2], result[idx + 3]] = unitAngleFeature(enemy.aimAngle);
			result[idx + 4] = enemy.speed / 100;
			result[idx + 5] = enemy.bombType ? 1 : 0;
			result[idx + 6] = enemy.health / Math.max(enemy.maxHealth, 1);
			const angleFromEnemyToPlayer = Math.atan2(sy - ey, sx - ex);
			const enemyAimError = Math.atan2(
				Math.sin(enemy.aimAngle - angleFromEnemyToPlayer),
				Math.cos(enemy.aimAngle - angleFromEnemyToPlayer)
			);
			result[idx + 7] = 1 - Math.abs(enemyAimError) / Math.PI;
			result[idx + 8] = enemy.maxAmmo > 0 ? (enemy.ammoType === 'super' ? 1 : 0.5) : 0;
			const moveDirection = MOVE_DIR_MAP[enemy.lastMoveIntent] ?? [0, 0];
			if (moveDirection[0] !== 0 || moveDirection[1] !== 0) {
				const dx = sx - ex;
				const dy = sy - ey;
				const distance = Math.sqrt(dx * dx + dy * dy);
				result[idx + 9] = distance > 1e-6 ? ((moveDirection[0] * dx + moveDirection[1] * dy) / distance) * 0.5 + 0.5 : 0.5;
			} else {
				result[idx + 9] = 0.5;
			}
		}
		idx += ENEMY_DIM;
	}

	const projectiles = [...obs.projectiles].sort(
		(a, b) => (a.x - sx) ** 2 + (a.y - sy) ** 2 - ((b.x - sx) ** 2 + (b.y - sy) ** 2)
	);
	for (let i = 0; i < MAX_PROJECTILES; i++) {
		if (i < projectiles.length) {
			const projectile = projectiles[i];
			result[idx] = ((projectile.x - sx) / ARENA_DIAGONAL) * 0.5 + 0.5;
			result[idx + 1] = ((projectile.y - sy) / ARENA_DIAGONAL) * 0.5 + 0.5;
			result[idx + 2] = (projectile.vx / PROJECTILE_SPEED_NORM) * 0.5 + 0.5;
			result[idx + 3] = (projectile.vy / PROJECTILE_SPEED_NORM) * 0.5 + 0.5;
			result[idx + 4] = projectile.team === 'enemy' ? 1 : 0;
		}
		idx += PROJ_DIM;
	}

	const obstacles = [...obs.obstacles].sort(
		(a, b) =>
			(a.x + a.width / 2 - sx) ** 2 +
			(a.y + a.height / 2 - sy) ** 2 -
			((b.x + b.width / 2 - sx) ** 2 + (b.y + b.height / 2 - sy) ** 2)
	);
	for (let i = 0; i < MAX_OBSTACLES; i++) {
		if (i < obstacles.length) {
			const obstacle = obstacles[i];
			result[idx] = ((obstacle.x + obstacle.width / 2 - sx) / ARENA_DIAGONAL) * 0.5 + 0.5;
			result[idx + 1] = ((obstacle.y + obstacle.height / 2 - sy) / ARENA_DIAGONAL) * 0.5 + 0.5;
			result[idx + 2] = obstacle.width / ARENA_WIDTH;
			result[idx + 3] = obstacle.height / ARENA_HEIGHT;
		}
		idx += OBSTACLE_DIM;
	}

	const bombs = [...obs.bombs].sort(
		(a, b) => (a.x - sx) ** 2 + (a.y - sy) ** 2 - ((b.x - sx) ** 2 + (b.y - sy) ** 2)
	);
	for (let i = 0; i < MAX_BOMBS; i++) {
		if (i < bombs.length) {
			const bomb = bombs[i];
			result[idx] = ((bomb.x - sx) / ARENA_DIAGONAL) * 0.5 + 0.5;
			result[idx + 1] = ((bomb.y - sy) / ARENA_DIAGONAL) * 0.5 + 0.5;
			result[idx + 2] = Math.min(bomb.fuseTicksRemaining / MAX_FUSE_TICKS, 1);
			result[idx + 3] = Math.min(bomb.blastRadius / MAX_BLAST_RADIUS, 1);
			result[idx + 4] = bomb.team === 'enemy' ? 1 : 0;
		}
		idx += BOMB_DIM;
	}

	result[idx] = Math.min(livingEnemies.length / MAX_ENEMIES, 1);
	result[idx + 1] =
		livingEnemies.length > 0
			? Math.min(
					Math.sqrt(
						Math.max(
							...livingEnemies.map(
								(enemy) => (enemy.x + enemy.size / 2 - sx) ** 2 + (enemy.y + enemy.size / 2 - sy) ** 2
							)
						)
					) / ARENA_DIAGONAL,
					1
			  )
			: 0;
	result[idx + 2] = Math.min(projectiles.length / MAX_PROJECTILES, 1);
	result[idx + 3] = Math.min(bombs.length / MAX_BOMBS, 1);
	const enemyBombs = bombs.filter((bomb) => bomb.team === 'enemy');
	result[idx + 4] =
		enemyBombs.length > 0
			? Math.min(
					Math.sqrt(Math.min(...enemyBombs.map((bomb) => (bomb.x - sx) ** 2 + (bomb.y - sy) ** 2))) /
						ARENA_DIAGONAL,
					1
			  )
			: 1;
	result[idx + 5] =
		projectiles.length > 0
			? Math.min(
					Math.sqrt(Math.max(...projectiles.map((projectile) => (projectile.x - sx) ** 2 + (projectile.y - sy) ** 2))) /
						ARENA_DIAGONAL,
					1
			  )
			: 0;

	for (let feature = 0; feature < OBS_SIZE; feature++) {
		result[feature] = Math.max(0, Math.min(1, result[feature]));
	}
	return result;
}
