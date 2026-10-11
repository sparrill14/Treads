import assert from 'node:assert/strict';
import { ScriptedEnemyController } from '../src/game/controllers/ScriptedEnemyController';
import { PassiveTankController } from '../src/game/controllers/ReplayController';
import { createMatchBootstrap } from '../src/game/core/MatchFactory';
import { Simulation } from '../src/game/core/Simulation';
import type { LevelConfig } from '../src/game/LevelConfig';

function advancedController() {
	return new ScriptedEnemyController({ navigationMode: 'astar', randomAim: false,
		recalculationInterval: 15, aggressionFactor: 3, heuristicProfile: 'advanced', tacticalRole: 'pressure' });
}

export function runAdvancedTacticsTests(): void {
	for (const allyX of [250, 700]) {
		const config: LevelConfig = { obstacles: [], rules: { projectileBounces: false }, tanks: [
			{ id: 'player-0', team: 'alpha', kind: 'player', control: 'human', x: 100, y: 250,
				ammo: { type: 'basic', count: 5 }, bombs: { type: 'basic', count: 0 } },
			{ id: 'target', team: 'beta', kind: 'stationary', x: 450, y: 250, ammo: { type: 'basic', count: 0 } },
			{ id: 'ally', team: 'alpha', kind: 'stationary', x: allyX, y: 250, ammo: { type: 'basic', count: 0 } },
		] };
		const bootstrap = createMatchBootstrap(config, 42, { playerController: advancedController() });
		const simulation = new Simulation(bootstrap.initialState, bootstrap.controllers);
		const first = simulation.step();
		assert.equal(first.actions['player-0'].fire, allyX > 450,
			'Fire only when an enemy intercepts the round before an ally');
		if (allyX > 450) {
			for (let tick = 0; tick < 400 && simulation.getState().status === 'running'; tick++) simulation.step();
			assert.equal(simulation.getState().tanks.find((tank) => tank.id === 'target')?.destroyed, true);
			assert.equal(simulation.getState().tanks.find((tank) => tank.id === 'ally')?.health, 3);
		}
	}

	const bootstrap = createMatchBootstrap({ player: { x: 500, y: 200, ammo: { type: 'basic', count: 0 },
		bombs: { type: 'basic', count: 0 } }, obstacles: [], rules: { projectileBounces: false },
		enemies: [{ type: 'stationary', x: 850, y: 400, ammo: { type: 'basic', count: 0 } }],
	}, 42, { playerController: advancedController() });
	const state = bootstrap.initialState;
	state.projectiles.push({ id: 'incoming', ownerTankId: 'enemy-0', team: 'enemy', kind: 'super',
		x: 510, y: 365, vx: 0, vy: -270, speed: 270, radius: 4, bounces: 0, maxBounces: 2 });
	const simulation = new Simulation(state, { ...bootstrap.controllers, 'enemy-0': new PassiveTankController() });
	for (let tick = 0; tick < 90; tick++) simulation.step();
	assert.equal(simulation.getState().tanks.find((tank) => tank.id === 'player-0')?.health, 3,
		'A tank with room to sidestep must evade an incoming round');
}
