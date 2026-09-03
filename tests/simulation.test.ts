import assert from 'node:assert/strict';
import type { LevelConfig } from '../src/game/LevelConfig';
import { LEVEL_CONFIGS } from '../src/game/LevelConfig';
import { PassiveTankController } from '../src/game/controllers/ReplayController';
import { exportDatasetFromReplay } from '../src/game/core/Dataset';
import { runHeadlessMatch } from '../src/game/core/HeadlessRunner';
import { createInitialGameState, createMatchBootstrap } from '../src/game/core/MatchFactory';
import { runReplay } from '../src/game/core/Replay';
import { Simulation } from '../src/game/core/Simulation';
import { serializeGameState } from '../src/game/core/stateUtils';
import type { TankAction, TankController } from '../src/game/core/types';

const TEST_LEVEL = LEVEL_CONFIGS[8];
const TEST_TICKS = 300;
const TEST_SEED = 20260329;

function defined<T>(value: T | undefined, message: string): T {
	assert.ok(value, message);
	return value;
}

function createRun() {
	return runHeadlessMatch({
		levelConfig: TEST_LEVEL,
		seed: TEST_SEED,
		ticks: TEST_TICKS,
		playerController: new PassiveTankController(),
		recordReplay: true,
		collectDataset: true,
	});
}

class FixedAimController implements TankController {
	constructor(private readonly action: TankAction) {}
	act(): TankAction {
		return this.action;
	}
}

function createProjectileState(id: string, x: number, y: number) {
	return {
		id,
		ownerTankId: 'player-0',
		team: 'player' as const,
		kind: 'basic' as const,
		x,
		y,
		vx: 0,
		vy: 0,
		speed: 0,
		radius: 4,
		bounces: 0,
		maxBounces: 1,
	};
}

export function runSimulationTests(): void {
	const firstRun = createRun();
	const secondRun = createRun();
	assert.equal(serializeGameState(firstRun.finalState), serializeGameState(secondRun.finalState));
	assert.deepEqual(firstRun.replay, secondRun.replay);
	assert.deepEqual(firstRun.dataset, secondRun.dataset);

	const replay = defined(firstRun.replay, 'Expected recorded replay');
	const replayedSimulation = runReplay(replay, replay.ticks.length);
	assert.equal(serializeGameState(replayedSimulation.getStateSnapshot()), serializeGameState(firstRun.finalState));
	const dataset = exportDatasetFromReplay(replay);
	assert.ok(dataset.length > 0);
	assert.equal(dataset.length, firstRun.dataset?.length ?? 0);
	assert.deepEqual(dataset[0]?.action, firstRun.dataset?.[0]?.action);

	const hpLevel: LevelConfig = {
		player: { x: 100, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 200, y: 250, ammo: { type: 'basic', count: 1 } }],
	};
	const hpState = createInitialGameState(hpLevel, 1);
	const enemy = defined(hpState.tanks.find((tank) => tank.id === 'enemy-0'), 'Expected enemy tank');
	hpState.projectiles.push(createProjectileState('p1', enemy.x + 5, enemy.y + 5));
	hpState.projectiles.push(createProjectileState('p2', enemy.x + enemy.size - 5, enemy.y + enemy.size - 5));
	const hpSim = new Simulation(
		hpState,
		createMatchBootstrap(hpLevel, 1, { playerController: new PassiveTankController() }).controllers
	);
	hpSim.step();
	const hpEnemyAfter = defined(
		hpSim.getStateSnapshot().tanks.find((tank) => tank.id === 'enemy-0'),
		'Expected enemy after projectile hit'
	);
	assert.equal(hpEnemyAfter.health, 2);
	assert.equal(hpEnemyAfter.invulnerabilityTicksRemaining, 8);

	const bombLevel: LevelConfig = {
		player: { x: 100, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 200, y: 250, ammo: { type: 'basic', count: 1 } }],
	};
	const bombState = createInitialGameState(bombLevel, 2);
	bombState.bombs.push({
		id: 'b1',
		ownerTankId: 'player-0',
		team: 'player',
		kind: 'basic',
		x: 215,
		y: 265,
		radius: 15,
		blastRadius: 50,
		fuseTicksRemaining: 1,
	});
	const bombSim = new Simulation(
		bombState,
		createMatchBootstrap(bombLevel, 2, { playerController: new PassiveTankController() }).controllers
	);
	bombSim.step();
	const bombEnemyAfter = defined(
		bombSim.getStateSnapshot().tanks.find((tank) => tank.id === 'enemy-0'),
		'Expected enemy after bomb hit'
	);
	assert.equal(bombEnemyAfter.health, 1);

	const noBounceLevel: LevelConfig = {
		player: { x: 100, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 700, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	};
	const noBounceState = createInitialGameState(noBounceLevel, 3);
	noBounceState.projectiles.push({
		id: 'wall-shot',
		ownerTankId: 'player-0',
		team: 'player',
		kind: 'basic',
		x: 999,
		y: 250,
		vx: 180,
		vy: 0,
		speed: 180,
		radius: 4,
		bounces: 0,
		maxBounces: 1,
	});
	const noBounceSim = new Simulation(
		noBounceState,
		createMatchBootstrap(noBounceLevel, 3, { playerController: new PassiveTankController() }).controllers
	);
	noBounceSim.step();
	assert.equal(noBounceSim.getStateSnapshot().projectiles.length, 0);

	const fastTurretLevel: LevelConfig = {
		player: { x: 100, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 700, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { turretSpeedMultiplier: 2 },
	};
	const fastBootstrap = createMatchBootstrap(fastTurretLevel, 4, {
		playerController: new FixedAimController({ move: 'none', aimAngle: Math.PI, fire: false, plantBomb: false }),
	});
	const fastTurretSim = new Simulation(fastBootstrap.initialState, fastBootstrap.controllers);
	fastTurretSim.step();
	const playerAfterTurn = defined(
		fastTurretSim.getStateSnapshot().tanks.find((tank) => tank.id === 'player-0'),
		'Expected player after turret turn'
	);
	assert.ok(Math.abs(playerAfterTurn.aimAngle - 0.6) < 1e-6);
}
