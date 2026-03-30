import type { ArenaState, MoveIntent, ObstacleStateView, ProjectileStateView, TankStateView } from './types';

export function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(value, max));
}

export function normalizeAngle(angle: number): number {
	let normalized = angle % (Math.PI * 2);
	if (normalized < 0) {
		normalized += Math.PI * 2;
	}
	return normalized;
}

export function signedAngleDelta(from: number, to: number): number {
	const normalizedFrom = normalizeAngle(from);
	const normalizedTo = normalizeAngle(to);
	const delta = normalizedTo - normalizedFrom;
	if (delta > Math.PI) {
		return delta - Math.PI * 2;
	}
	if (delta < -Math.PI) {
		return delta + Math.PI * 2;
	}
	return delta;
}

export function rotateAngleTowards(current: number, target: number, maxDelta: number): number {
	const delta = signedAngleDelta(current, target);
	if (Math.abs(delta) <= maxDelta) {
		return normalizeAngle(target);
	}
	return normalizeAngle(current + Math.sign(delta) * maxDelta);
}

export function getTankCenter(tank: Pick<TankStateView, 'x' | 'y' | 'size'>): { x: number; y: number } {
	return {
		x: tank.x + tank.size / 2,
		y: tank.y + tank.size / 2,
	};
}

export function computeGunBarrelEnd(tank: Pick<TankStateView, 'x' | 'y' | 'size' | 'aimAngle'>): {
	x: number;
	y: number;
} {
	const center = getTankCenter(tank);
	return {
		x: center.x + Math.cos(tank.aimAngle) * tank.size,
		y: center.y + Math.sin(tank.aimAngle) * tank.size,
	};
}

export function circlesOverlap(x1: number, y1: number, r1: number, x2: number, y2: number, r2: number): boolean {
	const dx = x1 - x2;
	const dy = y1 - y2;
	return dx * dx + dy * dy < (r1 + r2) * (r1 + r2);
}

export function tankIntersectsBlast(
	tank: Pick<TankStateView, 'x' | 'y' | 'size'>,
	x: number,
	y: number,
	blastRadius: number
): boolean {
	const center = getTankCenter(tank);
	const halfSide = tank.size / 2;
	const closestX = Math.max(center.x - halfSide, Math.min(x, center.x + halfSide));
	const closestY = Math.max(center.y - halfSide, Math.min(y, center.y + halfSide));
	const dx = closestX - x;
	const dy = closestY - y;
	return Math.sqrt(dx * dx + dy * dy) <= blastRadius;
}

export function aimAngleAtTarget(
	source: Pick<TankStateView, 'x' | 'y' | 'size'>,
	target: Pick<TankStateView, 'x' | 'y' | 'size'>
): number {
	const sourceCenter = getTankCenter(source);
	const targetCenter = getTankCenter(target);
	return normalizeAngle(Math.atan2(targetCenter.y - sourceCenter.y, targetCenter.x - sourceCenter.x));
}

export function deriveAimTarget(
	angle: number,
	centerX: number,
	centerY: number,
	arena: ArenaState,
	maxDistance: number
): { x: number; y: number } {
	const dx = Math.cos(angle);
	const dy = Math.sin(angle);
	const candidates: number[] = [];
	if (dx !== 0) {
		candidates.push((0 - centerX) / dx, (arena.width - centerX) / dx);
	}
	if (dy !== 0) {
		candidates.push((0 - centerY) / dy, (arena.height - centerY) / dy);
	}
	const positiveDistances = candidates.filter((candidate) => candidate > 0);
	const distance = positiveDistances.length > 0 ? Math.min(...positiveDistances, maxDistance) : maxDistance;
	return {
		x: clamp(centerX + dx * distance, 0, arena.width),
		y: clamp(centerY + dy * distance, 0, arena.height),
	};
}

export function getMoveDelta(move: MoveIntent, speed: number, tickSeconds: number): { dx: number; dy: number } {
	const moveDistance = speed * tickSeconds;
	switch (move) {
		case 'n':
			return { dx: 0, dy: -moveDistance };
		case 's':
			return { dx: 0, dy: moveDistance };
		case 'e':
			return { dx: moveDistance, dy: 0 };
		case 'w':
			return { dx: -moveDistance, dy: 0 };
		case 'ne':
			return { dx: moveDistance, dy: -moveDistance };
		case 'nw':
			return { dx: -moveDistance, dy: -moveDistance };
		case 'se':
			return { dx: moveDistance, dy: moveDistance };
		case 'sw':
			return { dx: -moveDistance, dy: moveDistance };
		default:
			return { dx: 0, dy: 0 };
	}
}

export function projectileObstacleResponse(
	projectile: ProjectileStateView,
	obstacle: ObstacleStateView
): { hit: boolean; x: number; y: number; vx: number; vy: number } {
	if (
		projectile.x > obstacle.x &&
		projectile.x < obstacle.x + obstacle.width &&
		projectile.y > obstacle.y &&
		projectile.y < obstacle.y + obstacle.height
	) {
		const fromLeft = Math.abs(projectile.x - obstacle.x);
		const fromRight = Math.abs(projectile.x - (obstacle.x + obstacle.width));
		const fromTop = Math.abs(projectile.y - obstacle.y);
		const fromBottom = Math.abs(projectile.y - (obstacle.y + obstacle.height));
		const minDistance = Math.min(fromLeft, fromRight, fromTop, fromBottom);
		if (minDistance === fromTop) {
			return { hit: true, x: projectile.x, y: obstacle.y - 1, vx: projectile.vx, vy: -projectile.vy };
		}
		if (minDistance === fromBottom) {
			return { hit: true, x: projectile.x, y: obstacle.y + obstacle.height + 1, vx: projectile.vx, vy: -projectile.vy };
		}
		if (minDistance === fromLeft) {
			return { hit: true, x: obstacle.x - 1, y: projectile.y, vx: -projectile.vx, vy: projectile.vy };
		}
		return { hit: true, x: obstacle.x + obstacle.width + 1, y: projectile.y, vx: -projectile.vx, vy: projectile.vy };
	}
	return {
		hit: false,
		x: projectile.x,
		y: projectile.y,
		vx: projectile.vx,
		vy: projectile.vy,
	};
}

export function obstacleIntersectsTank(x: number, y: number, size: number, obstacle: ObstacleStateView): boolean {
	return (
		x < obstacle.x + obstacle.width &&
		x + size > obstacle.x &&
		y < obstacle.y + obstacle.height &&
		y + size > obstacle.y
	);
}

export function clampTankToArena(tank: TankStateView, arena: ArenaState): void {
	tank.x = clamp(tank.x, 0, arena.width - tank.size);
	tank.y = clamp(tank.y, 0, arena.height - tank.size);
}
