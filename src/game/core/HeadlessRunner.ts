import type { LevelConfig } from '../LevelConfig';
import { DatasetCollector } from './Dataset';
import { createMatchBootstrap } from './MatchFactory';
import { ReplayRecorder } from './Replay';
import { Simulation, type SimulationOptions } from './Simulation';
import type { DatasetSample, GameState, ReplayData, TankController } from './types';

export interface HeadlessRunOptions {
	levelConfig: LevelConfig;
	seed: number;
	ticks: number;
	playerController?: TankController;
	controllerOverrides?: Record<string, TankController>;
	recordReplay?: boolean;
	collectDataset?: boolean;
	simulationOptions?: SimulationOptions;
}

export interface HeadlessRunResult {
	finalState: GameState;
	replay?: ReplayData;
	dataset?: DatasetSample[];
	simulation: Simulation;
}

export function runHeadlessMatch(options: HeadlessRunOptions): HeadlessRunResult {
	const bootstrap = createMatchBootstrap(options.levelConfig, options.seed, {
		playerController: options.playerController,
	});
	const controllers = {
		...bootstrap.controllers,
		...(options.controllerOverrides ?? {}),
	};
	const simulation = new Simulation(bootstrap.initialState, controllers, options.simulationOptions);
	const replayRecorder = options.recordReplay ? new ReplayRecorder(options.levelConfig, options.seed) : null;
	const datasetCollector = options.collectDataset ? new DatasetCollector() : null;
	for (let tick = 0; tick < options.ticks; tick++) {
		const stepResult = simulation.step();
		replayRecorder?.record(stepResult);
		datasetCollector?.recordStep(stepResult);
	}
	return {
		finalState: simulation.getStateSnapshot(),
		replay: replayRecorder?.toJSON(),
		dataset: datasetCollector?.toJSON(),
		simulation,
	};
}
