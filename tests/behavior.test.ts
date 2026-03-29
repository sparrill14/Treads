import assert from 'node:assert/strict';
import { createMatchBootstrap } from '../src/game/core/MatchFactory';
import { Simulation } from '../src/game/core/Simulation';
import { PassiveTankController } from '../src/game/controllers/ReplayController';
import type { EnemyConfig, LevelConfig, ObstacleConfig, PlayerConfig } from '../src/game/LevelConfig';

interface EnemyMetrics {
	initialDistance: number;
	finalDistance: number;
	moved: boolean;
	maxDisplacement: number;
	maxAbsDeltaY: number;
	projectileFires: number;
	bombPlants: number;
	aimChanges: number;
	projectileKinds: Set<string>;
	bombKinds: Set<string>;
}

function runEnemyScenario(
	enemy: EnemyConfig,
	player: PlayerConfig,
	obstacles: ObstacleConfig[] = [],
	seed = 1,
	ticks = 500
): EnemyMetrics {
	const levelConfig: LevelConfig = {
		player,
		obstacles,
		enemies: [enemy],
	};
	const bootstrap = createMatchBootstrap(levelConfig, seed, {
		playerController: new PassiveTankController(),
	});
	const simulation = new Simulation(bootstrap.initialState, bootstrap.controllers);
	const startingState = simulation.getStateSnapshot();
	const enemyId = 'enemy-0';
	const startEnemy = startingState.tanks.find((tank) => tank.id === enemyId);
	const startPlayer = startingState.tanks.find((tank) => tank.team === 'player');
	if (!startEnemy || !startPlayer) {
		throw new Error('Scenario bootstrap failed.');
	}

	let projectileFires = 0;
	let bombPlants = 0;
	let aimChanges = 0;
	let maxDisplacement = 0;
	let maxAbsDeltaY = 0;
	const projectileKinds = new Set<string>();
	const bombKinds = new Set<string>();
	let previousAim = startEnemy.aimAngle;

	for (let tick = 0; tick < ticks; tick++) {
		const stepResult = simulation.step();
		const state = simulation.getStateSnapshot();
		const enemyState = state.tanks.find((tank) => tank.id === enemyId);
		const playerState = state.tanks.find((tank) => tank.team === 'player');
		if (!enemyState || !playerState) {
			throw new Error('Scenario state invalid during simulation.');
		}

		const displacement = Math.hypot(enemyState.x - startEnemy.x, enemyState.y - startEnemy.y);
		maxDisplacement = Math.max(maxDisplacement, displacement);
		maxAbsDeltaY = Math.max(maxAbsDeltaY, Math.abs(enemyState.y - startEnemy.y));
		if (Math.abs(enemyState.aimAngle - previousAim) > 1e-9) {
			aimChanges += 1;
		}
		previousAim = enemyState.aimAngle;

		for (const event of stepResult.events) {
			if (event.type === 'projectile-fired' && event.tankId === enemyId) {
				projectileFires += 1;
			}
			if (event.type === 'bomb-planted' && event.tankId === enemyId) {
				bombPlants += 1;
			}
		}

		for (const projectile of state.projectiles) {
			if (projectile.ownerTankId === enemyId) {
				projectileKinds.add(projectile.kind);
			}
		}
		for (const bomb of state.bombs) {
			if (bomb.ownerTankId === enemyId) {
				bombKinds.add(bomb.kind);
			}
		}
	}

	const finalState = simulation.getStateSnapshot();
	const finalEnemy = finalState.tanks.find((tank) => tank.id === enemyId);
	const finalPlayer = finalState.tanks.find((tank) => tank.team === 'player');
	if (!finalEnemy || !finalPlayer) {
		throw new Error('Scenario final state invalid.');
	}

	return {
		initialDistance: Math.hypot(startEnemy.x - startPlayer.x, startEnemy.y - startPlayer.y),
		finalDistance: Math.hypot(finalEnemy.x - finalPlayer.x, finalEnemy.y - finalPlayer.y),
		moved: maxDisplacement > 1e-6,
		maxDisplacement,
		maxAbsDeltaY,
		projectileFires,
		bombPlants,
		aimChanges,
		projectileKinds,
		bombKinds,
	};
}

export function runBehaviorTests(): void {
	const stationary = runEnemyScenario({ type: 'stationary', x: 800, y: 250 }, { x: 100, y: 250 });
	assert.equal(stationary.moved, false);
	assert.ok(stationary.projectileFires >= 1);
	assert.ok(stationary.projectileKinds.has('basic'));

	const randomAim = runEnemyScenario(
		{ type: 'stationary-random-aim', x: 800, y: 250, ammo: { type: 'super', count: 1 } },
		{ x: 100, y: 250 }
	);
	assert.equal(randomAim.moved, false);
	assert.ok(randomAim.aimChanges > 100);
	assert.ok(randomAim.projectileKinds.has('super'));

	const simpleMoving = runEnemyScenario(
		{ type: 'simple-moving', x: 900, y: 50, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		{ x: 100, y: 450 }
	);
	assert.equal(simpleMoving.moved, true);
	assert.ok(simpleMoving.finalDistance < simpleMoving.initialDistance - 100);
	assert.ok(simpleMoving.projectileFires >= 1);

	const astarMoving = runEnemyScenario(
		{ type: 'simple-moving', x: 800, y: 250, ammo: { type: 'basic', count: 1 }, navigator: { type: 'astar' } },
		{ x: 100, y: 250 },
		[{ x: 400, y: 100, width: 30, height: 300 }]
	);
	assert.equal(astarMoving.moved, true);
	assert.ok(astarMoving.finalDistance < astarMoving.initialDistance - 100);
	assert.ok(astarMoving.maxAbsDeltaY > 50);

	const bomber = runEnemyScenario(
		{ type: 'bomber', x: 500, y: 100, ammo: { type: 'basic', count: 1 }, bombs: { type: 'basic', count: 3 }, navigator: { type: 'astar' } },
		{ x: 100, y: 400 },
		[{ x: 220, y: 180, width: 40, height: 140 }]
	);
	assert.equal(bomber.moved, true);
	assert.ok(bomber.bombPlants >= 1);
	assert.ok(bomber.projectileKinds.has('basic'));
	assert.ok(bomber.bombKinds.has('basic'));

	const superBomber = runEnemyScenario(
		{
			type: 'super-bomber',
			x: 320,
			y: 250,
			ammo: { type: 'super', count: 3 },
			bombs: { type: 'love', count: 2 },
			navigator: { type: 'astar', aggressionFactor: 5 },
		},
		{ x: 100, y: 250 }
	);
	assert.equal(superBomber.moved, true);
	assert.ok(superBomber.projectileKinds.has('super'));
	assert.ok(superBomber.bombPlants >= 1);
	assert.ok(superBomber.bombKinds.has('love'));

	const avoidance = runEnemyScenario(
		{
			type: 'super-bomber',
			x: 800,
			y: 100,
			ammo: { type: 'super', count: 3 },
			bombs: { type: 'basic', count: 2 },
			navigator: { type: 'astar-avoidance', aggressionFactor: 5 },
		},
		{ x: 100, y: 250 },
		[
			{ x: 350, y: 0, width: 30, height: 200 },
			{ x: 350, y: 300, width: 30, height: 200 },
			{ x: 700, y: 100, width: 30, height: 300 },
		]
	);
	assert.equal(avoidance.moved, true);
	assert.ok(avoidance.finalDistance < avoidance.initialDistance - 200);
	assert.ok(avoidance.maxAbsDeltaY > 50);
	assert.ok(avoidance.projectileKinds.has('super'));
}
