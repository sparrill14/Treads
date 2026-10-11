import assert from 'node:assert/strict';
import { LEVEL_CONFIGS, type LevelConfig } from '../src/game/LevelConfig';
import { createMatchBootstrap } from '../src/game/core/MatchFactory';
import { Simulation } from '../src/game/core/Simulation';
import { PassiveTankController } from '../src/game/controllers/ReplayController';

function playAgainstIdlePlayer(config: LevelConfig, seed: number, maxTicks: number) {
	const bootstrap = createMatchBootstrap(config, seed, { playerController: new PassiveTankController() });
	const simulation = new Simulation(bootstrap.initialState, bootstrap.controllers);
	let shots = 0;
	let bombs = 0;
	let blockedSteps = 0;
	while (simulation.getState().status === 'running' && simulation.getState().tick < maxTicks) {
		const result = simulation.step();
		shots += result.events.filter((event) => event.type === 'projectile-fired').length;
		bombs += result.events.filter((event) => event.type === 'bomb-planted').length;
		blockedSteps += simulation.getState().tanks.filter((tank) => tank.team === 'enemy' && tank.wasLastMoveBlocked).length;
	}
	return { state: simulation.getStateSnapshot(), shots, bombs, blockedSteps };
}

export function runHeuristicReadinessTests(): void {
	const turning = createMatchBootstrap({
		player: { x: 100, y: 250, ammo: { type: 'basic', count: 0 }, bombs: { type: 'basic', count: 0 } },
		enemies: [{ type: 'simple-moving', x: 700, y: 250, ammo: { type: 'basic', count: 1 },
			navigator: { type: 'astar', heuristicProfile: 'advanced' } }],
		obstacles: [], rules: { projectileBounces: false },
	}, 42, { playerController: new PassiveTankController() });
	const turningSim = new Simulation(turning.initialState, turning.controllers);
	let firstShotTick = 0;
	for (let tick = 0; tick < 30; tick++) {
		const step = turningSim.step();
		if (step.events.some((event) => event.type === 'projectile-fired' && event.tankId === 'enemy-0')) {
			firstShotTick = turningSim.getState().tick;
			break;
		}
	}
	assert.ok(firstShotTick >= 10 && firstShotTick <= 20, `Turret must align before firing: tick ${firstShotTick}`);
	for (const seed of [1, 42, 99]) {
		const result = playAgainstIdlePlayer(LEVEL_CONFIGS[9], seed, 1200);
		assert.equal(result.state.status, 'enemy_win', `Advanced tanks failed to finish level 10, seed ${seed}`);
		assert.ok(result.shots > 0, 'Advanced tanks should shoot after navigating around cover');
		assert.ok(result.blockedSteps < 300, `Advanced tanks remain stuck against cover: ${result.blockedSteps}`);
	}
	const bomber = playAgainstIdlePlayer({
		player: { x: 450, y: 250, ammo: { type: 'basic', count: 0 }, bombs: { type: 'basic', count: 0 } },
		enemies: [{ type: 'super-bomber', x: 500, y: 250, ammo: { type: 'super', count: 0 },
			bombs: { type: 'love', count: 2 }, navigator: { type: 'astar-avoidance', heuristicProfile: 'advanced', tacticalRole: 'pressure' } }],
		obstacles: [],
	}, 42, 1200);
	assert.equal(bomber.state.status, 'enemy_win', 'An armed bomber should defeat a nearby idle opponent');
	assert.ok(bomber.bombs >= 2, 'Three-hit tanks need two bomb hits');
	assert.equal(bomber.state.tanks.find((tank) => tank.id === 'enemy-0')?.health, 3, 'The bomber must escape its own explosions');
}
