import { runBehaviorTests } from './behavior.test';
import { runSimulationTests } from './simulation.test';

try {
	runSimulationTests();
	runBehaviorTests();
	console.log('Simulation and behavior tests passed.');
} catch (error) {
	console.error('Simulation tests failed.');
	console.error(error);
	process.exitCode = 1;
}
