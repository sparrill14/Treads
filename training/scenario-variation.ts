import { SeededRandom } from '../src/game/core/prng';
import type { LevelConfig } from '../src/game/LevelConfig';

interface Positioned {
	x: number;
	y: number;
}

const ARENA_WIDTH = 1000;
const ARENA_HEIGHT = 500;
const TANK_SIZE = 30;

function clamp(value: number, minValue: number, maxValue: number): number {
	return Math.max(minValue, Math.min(maxValue, value));
}

function cloneLevel(config: LevelConfig): LevelConfig {
	return JSON.parse(JSON.stringify(config)) as LevelConfig;
}

function tankOverlapsObstacle(tank: Positioned, obstacle: LevelConfig['obstacles'][number]): boolean {
	return (
		tank.x < obstacle.x + obstacle.width &&
		tank.x + TANK_SIZE > obstacle.x &&
		tank.y < obstacle.y + obstacle.height &&
		tank.y + TANK_SIZE > obstacle.y
	);
}

function tanksOverlap(a: Positioned, b: Positioned): boolean {
	const padding = 4;
	return (
		a.x < b.x + TANK_SIZE + padding &&
		a.x + TANK_SIZE + padding > b.x &&
		a.y < b.y + TANK_SIZE + padding &&
		a.y + TANK_SIZE + padding > b.y
	);
}

function validTankPlacement(candidate: Positioned, placed: Positioned[], obstacles: LevelConfig['obstacles']): boolean {
	return !obstacles.some((obstacle) => tankOverlapsObstacle(candidate, obstacle)) && !placed.some((tank) => tanksOverlap(candidate, tank));
}

function variedTank<T extends Positioned>(
	tank: T,
	jitter: number,
	rng: SeededRandom,
	placed: Positioned[],
	obstacles: LevelConfig['obstacles']
): T {
	for (let attempt = 0; attempt < 24; attempt++) {
		const candidate = {
			...tank,
			x: clamp(tank.x + rng.nextRange(-jitter, jitter), 0, ARENA_WIDTH - TANK_SIZE),
			y: clamp(tank.y + rng.nextRange(-jitter, jitter), 0, ARENA_HEIGHT - TANK_SIZE),
		};
		if (validTankPlacement(candidate, placed, obstacles)) return candidate;
	}
	return { ...tank };
}

export function applySpawnJitter(config: LevelConfig, seed: number): LevelConfig {
	const varied = cloneLevel(config);
	const rng = new SeededRandom(seed * 7919 + 13);
	const placed: Positioned[] = [];
	if (varied.player) {
		varied.player = variedTank(varied.player, 40, rng, placed, varied.obstacles);
		placed.push(varied.player);
	}
	if (varied.enemies) {
		varied.enemies = varied.enemies.map((enemy) => {
			const next = variedTank(enemy, 40, rng, placed, varied.obstacles);
			placed.push(next);
			return next;
		});
	}
	if (varied.tanks) {
		varied.tanks = varied.tanks.map((tank) => {
			const next = variedTank(tank, 40, rng, placed, varied.obstacles);
			placed.push(next);
			return next;
		});
	}
	return varied;
}

export function applyProceduralDifficulty(config: LevelConfig, seed: number, difficultyBand: number): LevelConfig {
	const d = clamp(difficultyBand, 0, 1);
	if (d <= 0) return config;
	const varied = cloneLevel(config);
	const rng = new SeededRandom(seed * 3571 + 97);
	if (varied.enemies) {
		varied.enemies = varied.enemies.map((enemy) => {
			const nextEnemy = { ...enemy };
			if (nextEnemy.type === 'stationary' && rng.nextFloat() < 0.2 * d) nextEnemy.type = 'stationary-random-aim';
			if (nextEnemy.type === 'stationary-random-aim' && rng.nextFloat() < 0.15 * Math.max(0, d - 0.3)) {
				nextEnemy.type = 'simple-moving';
				nextEnemy.navigator = nextEnemy.navigator ?? { type: 'simple' };
			}
			if (nextEnemy.type === 'simple-moving' && rng.nextFloat() < 0.12 * Math.max(0, d - 0.55)) {
				nextEnemy.type = 'bomber';
				nextEnemy.navigator = { type: 'astar' };
				nextEnemy.bombs = nextEnemy.bombs ?? { type: 'basic', count: 1 };
			}
			if (rng.nextFloat() < 0.45 * d) {
				const ammoType = nextEnemy.ammo?.type ?? 'basic';
				nextEnemy.ammo = { type: ammoType, count: Math.min(3, (nextEnemy.ammo?.count ?? 1) + 1) };
			}
			return nextEnemy;
		});
	}
	const tanks: Positioned[] = [
		...(varied.player ? [varied.player] : []),
		...(varied.enemies ?? []),
		...(varied.tanks ?? []),
	];
	varied.obstacles = varied.obstacles.map((obstacle) => {
		const width = clamp(obstacle.width * (1 + rng.nextRange(-0.08 * d, 0.1 * d)), 18, 260);
		const height = clamp(obstacle.height * (1 + rng.nextRange(-0.08 * d, 0.1 * d)), 18, 280);
		const candidate = {
			x: clamp(obstacle.x + rng.nextRange(-16 * d, 16 * d), 0, ARENA_WIDTH - width),
			y: clamp(obstacle.y + rng.nextRange(-16 * d, 16 * d), 0, ARENA_HEIGHT - height),
			width,
			height,
		};
		return tanks.some((tank) => tankOverlapsObstacle(tank, candidate)) ? obstacle : candidate;
	});
	const rules = { ...(varied.rules ?? {}) };
	const baseTurret = rules.turretSpeedMultiplier ?? 1;
	rules.turretSpeedMultiplier = clamp(baseTurret * (1 + 0.35 * d + rng.nextRange(-0.08, 0.08)), 0.8, 3.5);
	if (d > 0.7 && rng.nextFloat() < (d - 0.7) * 1.8) rules.projectileBounces = true;
	varied.rules = rules;
	return varied;
}

export function configurePlayerResources(config: LevelConfig, maxAmmo: number, maxBombs: number): LevelConfig {
	const varied = cloneLevel(config);
	if (varied.player) {
		if (maxAmmo >= 0) varied.player.ammo = { type: varied.player.ammo?.type ?? 'basic', count: maxAmmo };
		if (maxBombs >= 0) varied.player.bombs = { type: varied.player.bombs?.type ?? 'basic', count: maxBombs };
	}
	return varied;
}
