/**
 * CLI runner for headless Treads simulation.
 * Communicates over stdin/stdout with newline-delimited JSON.
 *
 * Usage: node cli-runner.js [--level <1-9>] [--seed <number>] [--max-ticks <number>]
 *
 * Protocol:
 *   1. Runner writes {"type":"init", ...matchInit} for each AI tank
 *   2. Each tick: runner writes {"type":"observations", "observations": [{tankId, obs}...]}
 *   3. Python responds with {"actions": [{tankId, action}...]}
 *   4. When match ends: runner writes {"type":"result", ...matchResult}
 */

import * as readline from 'readline';
import { createDefaultControllers, createInitialGameState } from '../src/game/core/MatchFactory';
import { ReplayRecorder } from '../src/game/core/Replay';
import { Simulation } from '../src/game/core/Simulation';
import type {
	MatchInit,
	MoveIntent,
	SimulationStepResult,
	TankAction,
	TankController,
	TankObservation,
} from '../src/game/core/types';
import { LEVEL_CONFIGS } from '../src/game/LevelConfig';

// --- Stdin-driven controller: waits for actions from Python ---
class ExternalController implements TankController {
	public pendingAction: TankAction = {
		move: 'none',
		aimAngle: 0,
		fire: false,
		plantBomb: false,
	};

	public reset(_initial: MatchInit): void {
		// no-op
	}

	public act(_obs: TankObservation): TankAction {
		return this.pendingAction;
	}
}

function writeLine(obj: unknown): void {
	process.stdout.write(JSON.stringify(obj) + '\n');
}

function parseArgs(): { level: number; seed: number; maxTicks: number } {
	const args = process.argv.slice(2);
	let level = 1;
	let seed = 42;
	let maxTicks = 3600; // 60 seconds at 60 ticks/sec

	for (let i = 0; i < args.length; i++) {
		if (args[i] === '--level' && args[i + 1]) {
			level = parseInt(args[i + 1], 10);
			i++;
		} else if (args[i] === '--seed' && args[i + 1]) {
			seed = parseInt(args[i + 1], 10);
			i++;
		} else if (args[i] === '--max-ticks' && args[i + 1]) {
			maxTicks = parseInt(args[i + 1], 10);
			i++;
		}
	}

	level = Math.max(1, Math.min(level, LEVEL_CONFIGS.length));
	return { level, seed, maxTicks };
}

async function main(): Promise<void> {
	const { level, seed, maxTicks } = parseArgs();
	const levelConfig = LEVEL_CONFIGS[level - 1];

	// Create initial state and default controllers (scripted enemies)
	const initialState = createInitialGameState(levelConfig, seed);
	const defaultControllers = createDefaultControllers(levelConfig);

	// The player tank will be driven by the external (Python) controller
	const playerTankId = 'player-0';
	const externalControllers: Record<string, ExternalController> = {};
	const playerController = new ExternalController();
	externalControllers[playerTankId] = playerController;

	const controllers: Record<string, TankController> = {
		...defaultControllers,
		[playerTankId]: playerController,
	};

	const simulation = new Simulation(initialState, controllers);
	const replayRecorder = new ReplayRecorder(levelConfig, seed);

	// Setup readline for stdin
	const rl = readline.createInterface({
		input: process.stdin,
		terminal: false,
	});

	const lineQueue: string[] = [];
	let lineResolve: (() => void) | null = null;

	rl.on('line', (line: string) => {
		lineQueue.push(line);
		if (lineResolve) {
			const resolve = lineResolve;
			lineResolve = null;
			resolve();
		}
	});

	rl.on('close', () => {
		// stdin closed, terminate
		process.exit(0);
	});

	async function readLine(): Promise<string> {
		if (lineQueue.length > 0) {
			return lineQueue.shift()!;
		}
		return new Promise<string>((resolve) => {
			lineResolve = () => resolve(lineQueue.shift()!);
		});
	}

	// Send init message with arena info
	writeLine({
		type: 'init',
		seed,
		level,
		maxTicks,
		arena: initialState.arena,
		tanks: initialState.tanks.map((t) => ({
			id: t.id,
			team: t.team,
			kind: t.kind,
			x: t.x,
			y: t.y,
			size: t.size,
			speed: t.speed,
		})),
		obstacles: initialState.obstacles,
		playerTankId,
		aiTankIds: [playerTankId],
	});

	// Main simulation loop
	let tick = 0;
	while (tick < maxTicks) {
		// Build observations for AI tanks
		const state = simulation.getState();
		if (state.status !== 'running') {
			break;
		}

		// Build observation for the player tank
		const playerTank = state.tanks.find((t) => t.id === playerTankId);
		if (!playerTank || playerTank.destroyed) {
			break;
		}

		// Create a minimal observation for the external controller
		const obs: TankObservation = {
			tick: state.tick,
			self: JSON.parse(JSON.stringify(playerTank)),
			enemies: state.tanks.filter((t) => t.team !== playerTank.team).map((t) => JSON.parse(JSON.stringify(t))),
			projectiles: state.projectiles.map((p) => JSON.parse(JSON.stringify(p))),
			bombs: state.bombs.map((b) => JSON.parse(JSON.stringify(b))),
			obstacles: state.obstacles.map((o) => JSON.parse(JSON.stringify(o))),
			arena: { ...state.arena },
		};

		// Send observation
		writeLine({
			type: 'observation',
			tick: state.tick,
			observation: obs,
		});

		// Read action from Python
		const actionLine = await readLine();
		let actionData: {
			move?: MoveIntent;
			aimAngle?: number;
			fire?: boolean;
			plantBomb?: boolean;
		};
		try {
			actionData = JSON.parse(actionLine);
		} catch {
			actionData = { move: 'none', aimAngle: 0, fire: false, plantBomb: false };
		}

		// Set the action on the external controller
		playerController.pendingAction = {
			move: actionData.move ?? 'none',
			aimAngle: actionData.aimAngle ?? 0,
			fire: actionData.fire ?? false,
			plantBomb: actionData.plantBomb ?? false,
		};

		// Step simulation
		const stepResult: SimulationStepResult = simulation.step();
		replayRecorder.record(stepResult);
		tick++;
	}

	// Match ended - compute result
	const finalState = simulation.getStateSnapshot();
	const playerTank = finalState.tanks.find((t) => t.id === playerTankId);
	const enemiesDestroyed = finalState.tanks.filter((t) => t.team === 'enemy' && t.destroyed).length;
	const totalEnemies = finalState.tanks.filter((t) => t.team === 'enemy').length;

	writeLine({
		type: 'result',
		status: finalState.status,
		ticks: finalState.tick,
		playerDestroyed: playerTank?.destroyed ?? true,
		enemiesDestroyed,
		totalEnemies,
		win: finalState.status === 'player_win',
		loss: finalState.status === 'enemy_win',
		draw: finalState.status === 'running', // timed out
	});

	process.exit(0);
}

main().catch((err) => {
	process.stderr.write(`CLI runner error: ${err}\n`);
	process.exit(1);
});
