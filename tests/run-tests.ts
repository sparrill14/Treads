import { runBehaviorTests } from './behavior.test';
import { runSimulationTests } from './simulation.test';
import { runNeuralModelContractTests } from './neural-model-contract.test';

try {
	runSimulationTests();
	runBehaviorTests();
	runNeuralModelContractTests();
	console.log('Simulation, behavior, and neural model contract tests passed.');
} catch (error) {
	console.error('Simulation tests failed.');
	console.error(error);
	process.exitCode = 1;
}
