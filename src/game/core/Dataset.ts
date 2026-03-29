import { cloneJson } from './stateUtils';
import { createReplayControllers } from './Replay';
import { createInitialGameState } from './MatchFactory';
import { Simulation } from './Simulation';
import type { DatasetSample, ReplayData, SimulationStepResult } from './types';

export class DatasetCollector {
	private readonly samples: DatasetSample[] = [];

	public recordStep(stepResult: SimulationStepResult): void {
		for (const record of stepResult.records) {
			this.samples.push(cloneJson({
				controllerId: record.controllerId,
				tankId: record.tankId,
				tick: stepResult.tick,
				observation: record.observation,
				action: record.action,
				reward: record.reward,
				done: record.done,
			}));
		}
	}

	public toJSON(): DatasetSample[] {
		return cloneJson(this.samples);
	}
}

export function exportDatasetFromReplay(replay: ReplayData): DatasetSample[] {
	const initialState = createInitialGameState(replay.levelConfig, replay.seed);
	const simulation = new Simulation(initialState, createReplayControllers(initialState, replay));
	const collector = new DatasetCollector();
	for (const _tick of replay.ticks) {
		collector.recordStep(simulation.step());
	}
	return collector.toJSON();
}
