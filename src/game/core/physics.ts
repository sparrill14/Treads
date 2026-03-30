import { projectileObstacleResponse, tankIntersectsBlast } from './geometry';
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
	const tickDistance = Math.hypot(tickDx, tickDy);
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
			const response = projectileObstacleResponse(projectile, obstacle);
			if (response.hit) {
				if (!allowBounces) {
					projectile.bounces = projectile.maxBounces + 1;
					return;
				}
				projectile.x = response.x;
				projectile.y = response.y;
				projectile.vx = response.vx;
				projectile.vy = response.vy;
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

export function predictProjectileWillHitTank(
	projectile: ProjectileStateView,
	tank: TankStateView,
	arena: ArenaState,
	obstacles: ObstacleStateView[]
): boolean {
	const predictedProjectile: ProjectileStateView = { ...projectile };
	while (predictedProjectile.bounces <= predictedProjectile.maxBounces) {
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
