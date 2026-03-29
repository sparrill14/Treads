import { SIMULATION_TICK_SECONDS, type ArenaState, type ObstacleStateView, type ProjectileStateView, type TankStateView } from './types';
import { projectileObstacleResponse, tankIntersectsBlast } from './geometry';

export function stepProjectile(projectile: ProjectileStateView, arena: ArenaState, obstacles: ObstacleStateView[]): void {
	projectile.x += projectile.vx * SIMULATION_TICK_SECONDS;
	projectile.y += projectile.vy * SIMULATION_TICK_SECONDS;

	if (projectile.x <= 0 || projectile.x > arena.width) {
		projectile.vx = -projectile.vx;
		projectile.bounces += 1;
	}

	if (projectile.y <= 0 || projectile.y > arena.height) {
		projectile.vy = -projectile.vy;
		projectile.bounces += 1;
	}

	for (const obstacle of obstacles) {
		const response = projectileObstacleResponse(projectile, obstacle);
		if (response.hit) {
			projectile.x = response.x;
			projectile.y = response.y;
			projectile.vx = response.vx;
			projectile.vy = response.vy;
			projectile.bounces += 1;
		}
	}
}

export function projectileHitsTank(projectile: ProjectileStateView, tank: TankStateView): boolean {
	return (
		projectile.x > tank.x &&
		projectile.x < tank.x + tank.size &&
		projectile.y > tank.y &&
		projectile.y < tank.y + tank.size
	);
}

export function predictProjectileWillHitTank(
	projectile: ProjectileStateView,
	tank: TankStateView,
	arena: ArenaState,
	obstacles: ObstacleStateView[]
): boolean {
	const predictedProjectile: ProjectileStateView = { ...projectile };
	while (predictedProjectile.bounces <= predictedProjectile.maxBounces) {
		stepProjectile(predictedProjectile, arena, obstacles);
		if (projectileHitsTank(predictedProjectile, tank)) {
			return true;
		}
	}
	return false;
}

export function bombWouldHitTank(x: number, y: number, blastRadius: number, tank: TankStateView): boolean {
	return tankIntersectsBlast(tank, x, y, blastRadius);
}
