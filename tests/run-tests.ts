import { runBehaviorTests } from './behavior.test';
import { runSimulationTests } from './simulation.test';
import { runNeuralModelContractTests } from './neural-model-contract.test';
import { runHeuristicReadinessTests } from './heuristic-readiness.test';

try {
	runSimulationTests();
	runBehaviorTests();
	runNeuralModelContractTests();
	runHeuristicReadinessTests();
	console.log('Simulation, behavior, and neural model contract tests passed.');
} catch (error) {
	console.error('Simulation tests failed.');
	console.error(error);
	process.exitCode = 1;
}
