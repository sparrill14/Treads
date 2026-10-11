import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { NeuralNetController } from '../src/game/controllers/NeuralNetController';
import { decodePolicyAction } from '../src/game/controllers/NeuralModelContract';
import { normalizePolicyObservation } from '../src/game/controllers/NeuralObservation';
import { parseWeightsFromStateDict, PolicyMLP } from '../src/game/controllers/PolicyMLP';
import { LEVEL_CONFIGS } from '../src/game/LevelConfig';
import { createMatchBootstrap } from '../src/game/core/MatchFactory';
import { Simulation } from '../src/game/core/Simulation';
import type { MatchInit, TankAction, TankController, TankObservation } from '../src/game/core/types';
import { NavigationPlanner } from '../src/game/navigation/NavigationPlanner';

async function main(): Promise<void> {
	const [modelUrl, weightsPath, tracePath] = process.argv.slice(2);
	const mlp = new PolicyMLP(parseWeightsFromStateDict(JSON.parse(readFileSync(weightsPath, 'utf8'))));
	const browser = new NeuralNetController(modelUrl);
	await browser.loadModel();
	const samples: { obs: number[]; logits: number[]; indices: number[] }[] = [];
	let steps = 0;
	for (const level of [1, 7, 10]) {
		let planner: NavigationPlanner;
		const reference: TankController = {
			reset(init: MatchInit): void { planner = new NavigationPlanner(init.arena, init.obstacles, 30); },
			act(obs: TankObservation): TankAction {
				const input = normalizePolicyObservation(obs, 720, planner);
				const logits = mlp.policyLogits(input);
				const indices = mlp.argmaxMultiCategorical(logits);
				if (obs.tick % 37 === 0) samples.push({ obs: input, logits, indices });
				return decodePolicyAction(indices, obs);
			},
		};
		const create = (controller: TankController) => {
			const bootstrap = createMatchBootstrap(LEVEL_CONFIGS[level - 1], 42, { playerController: controller });
			return new Simulation(bootstrap.initialState, bootstrap.controllers);
		};
		const actualSim = create(browser);
		const expectedSim = create(reference);
		while (expectedSim.getState().status === 'running' && expectedSim.getState().tick < 600) {
			const expectedStep = expectedSim.step();
			const actualStep = actualSim.step();
			assert.deepEqual(actualStep.actions, expectedStep.actions);
			assert.deepEqual(actualSim.getStateSnapshot(), expectedSim.getStateSnapshot());
			steps++;
		}
	}
	await assert.rejects(new NeuralNetController(modelUrl.replace('policy.onnx', 'invalid.onnx')).loadModel());
	writeFileSync(tracePath, JSON.stringify({ steps, samples }));
	console.log(`Browser actor parity passed for ${steps} ticks on levels 1, 7, and 10; malformed weights rejected`);
}

void main().catch((error) => { console.error(error); process.exitCode = 1; });
