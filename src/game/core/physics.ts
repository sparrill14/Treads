import { tankIntersectsBlast } from './geometry';
import {
	SIMULATION_TICK_SECONDS,
	type ArenaState,
	type ObstacleStateView,
	type ProjectileStateView,
	type TankStateView,
} from './types';

const MIN_PROJECTILE_SUBSTEP_DISTANCE = 4;

function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(value, max));
}

export function stepProjectile(
	projectile: ProjectileStateView,
	arena: ArenaState,
	obstacles: ObstacleStateView[],
	allowBounces: boolean
): void {
	const tickDx = projectile.vx * SIMULATION_TICK_SECONDS;
	const tickDy = projectile.vy * SIMULATION_TICK_SECONDS;
	const tickDistance = Math.sqrt(tickDx * tickDx + tickDy * tickDy);
	const substeps = Math.max(1, Math.ceil(tickDistance / MIN_PROJECTILE_SUBSTEP_DISTANCE));

	for (let step = 0; step < substeps; step++) {
		projectile.x += tickDx / substeps;
		projectile.y += tickDy / substeps;

		const minX = projectile.radius;
		const maxX = arena.width - projectile.radius;
		if (projectile.x <= minX || projectile.x >= maxX) {
			if (!allowBounces) {
				projectile.bounces = projectile.maxBounces + 1;
				return;
			}
			projectile.x = clamp(projectile.x, minX, maxX);
			projectile.vx = -projectile.vx;
			projectile.bounces += 1;
		}

		const minY = projectile.radius;
		const maxY = arena.height - projectile.radius;
		if (projectile.y <= minY || projectile.y >= maxY) {
			if (!allowBounces) {
				projectile.bounces = projectile.maxBounces + 1;
				return;
			}
			projectile.y = clamp(projectile.y, minY, maxY);
			projectile.vy = -projectile.vy;
			projectile.bounces += 1;
		}

		for (const obstacle of obstacles) {
			if (
				projectile.x > obstacle.x &&
				projectile.x < obstacle.x + obstacle.width &&
				projectile.y > obstacle.y &&
				projectile.y < obstacle.y + obstacle.height
			) {
				if (!allowBounces) {
					projectile.bounces = projectile.maxBounces + 1;
					return;
				}
				const fromLeft = Math.abs(projectile.x - obstacle.x);
				const fromRight = Math.abs(projectile.x - (obstacle.x + obstacle.width));
				const fromTop = Math.abs(projectile.y - obstacle.y);
				const fromBottom = Math.abs(projectile.y - (obstacle.y + obstacle.height));
				const minDistance = Math.min(fromLeft, fromRight, fromTop, fromBottom);
				if (minDistance === fromTop) {
					projectile.y = obstacle.y - 1;
					projectile.vy = -projectile.vy;
				} else if (minDistance === fromBottom) {
					projectile.y = obstacle.y + obstacle.height + 1;
					projectile.vy = -projectile.vy;
				} else if (minDistance === fromLeft) {
					projectile.x = obstacle.x - 1;
					projectile.vx = -projectile.vx;
				} else {
					projectile.x = obstacle.x + obstacle.width + 1;
					projectile.vx = -projectile.vx;
				}
				projectile.bounces += 1;
			}
		}

		if (projectile.bounces > projectile.maxBounces) {
			return;
		}
	}
}

export function projectileHitsTank(projectile: ProjectileStateView, tank: TankStateView): boolean {
	const closestX = clamp(projectile.x, tank.x, tank.x + tank.size);
	const closestY = clamp(projectile.y, tank.y, tank.y + tank.size);
	const dx = projectile.x - closestX;
	const dy = projectile.y - closestY;
	return dx * dx + dy * dy <= projectile.radius * projectile.radius;
}

const PREDICTION_MAX_TICKS = 600;

export function predictProjectileWillHitTank(
	projectile: ProjectileStateView,
	tank: TankStateView,
	arena: ArenaState,
	obstacles: ObstacleStateView[]
): boolean {
	const predictedProjectile: ProjectileStateView = { ...projectile };
	for (let tick = 0; tick < PREDICTION_MAX_TICKS; tick++) {
		if (predictedProjectile.bounces > predictedProjectile.maxBounces) {
			return false;
		}
		stepProjectile(predictedProjectile, arena, obstacles, true);
		if (projectileHitsTank(predictedProjectile, tank)) {
			return true;
		}
	}
	return false;
}

export function bombWouldHitTank(x: number, y: number, blastRadius: number, tank: TankStateView): boolean {
	return tankIntersectsBlast(tank, x, y, blastRadius);
}
