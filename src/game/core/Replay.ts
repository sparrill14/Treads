import { ReplayController } from '../controllers/ReplayController';
import type { LevelConfig } from '../LevelConfig';
import { createInitialGameState } from './MatchFactory';
import { Simulation } from './Simulation';
import { cloneJson } from './stateUtils';
import type { GameState, ReplayData, SimulationStepResult, TankController } from './types';

export class ReplayRecorder {
	private readonly replay: ReplayData;

	constructor(levelConfig: LevelConfig, seed: number) {
		this.replay = {
			seed,
			levelConfig: cloneJson(levelConfig),
			ticks: [],
		};
	}

	public record(stepResult: SimulationStepResult): void {
		this.replay.ticks.push({
			tick: stepResult.tick,
			actions: cloneJson(stepResult.actions),
			tanks: cloneJson(stepResult.replayTankStates),
		});
	}

	public toJSON(): ReplayData {
		return cloneJson(this.replay);
	}
}

export function createReplayControllers(initialState: GameState, replay: ReplayData): Record<string, TankController> {
	const controllers: Record<string, TankController> = {};
	for (const tank of initialState.tanks) {
		controllers[tank.controllerId] = new ReplayController(tank.id, replay);
	}
	return controllers;
}

export function runReplay(replay: ReplayData, tickCount: number = replay.ticks.length): Simulation {
	const initialState = createInitialGameState(replay.levelConfig, replay.seed);
	const simulation = new Simulation(initialState, createReplayControllers(initialState, replay));
	for (let tick = 0; tick < tickCount; tick++) {
		simulation.step();
	}
	return simulation;
}
