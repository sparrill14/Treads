import assert from 'node:assert/strict';
import { LEVEL_CONFIGS } from '../src/game/LevelConfig';
import { exportDatasetFromReplay } from '../src/game/core/Dataset';
import { runHeadlessMatch } from '../src/game/core/HeadlessRunner';
import { runReplay } from '../src/game/core/Replay';
import { serializeGameState } from '../src/game/core/stateUtils';
import { PassiveTankController } from '../src/game/controllers/ReplayController';

const TEST_LEVEL = LEVEL_CONFIGS[8];
const TEST_TICKS = 300;
const TEST_SEED = 20260329;

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

export function runSimulationTests(): void {
	const firstRun = createRun();
	const secondRun = createRun();
	assert.equal(serializeGameState(firstRun.finalState), serializeGameState(secondRun.finalState));
	assert.deepEqual(firstRun.replay, secondRun.replay);
	assert.deepEqual(firstRun.dataset, secondRun.dataset);

	assert.ok(firstRun.replay);
	const replayedSimulation = runReplay(firstRun.replay, firstRun.replay.ticks.length);
	assert.equal(
		serializeGameState(replayedSimulation.getStateSnapshot()),
		serializeGameState(firstRun.finalState)
	);
	const dataset = exportDatasetFromReplay(firstRun.replay);
	assert.ok(dataset.length > 0);
	assert.equal(dataset.length, firstRun.dataset?.length ?? 0);
	assert.deepEqual(dataset[0]?.action, firstRun.dataset?.[0]?.action);
}
