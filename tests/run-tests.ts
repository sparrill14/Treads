import { runSimulationTests } from './simulation.test';

try {
	runSimulationTests();
	console.log('Simulation tests passed.');
} catch (error) {
	console.error('Simulation tests failed.');
	console.error(error);
	process.exitCode = 1;
}
