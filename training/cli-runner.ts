/**
 * CLI runner for headless Treads simulation.
 * Communicates over stdin/stdout with newline-delimited JSON.
 *
 * Supports two modes:
 *   --persistent : stays alive across episodes; Python sends reset commands
 *   (default)    : single-episode mode (legacy)
 *
 * Persistent Protocol:
 *   1. Runner writes {"type":"ready"}
 *   2. Python sends {"type":"reset","level":N,"seed":N,"maxTicks":N,"saveReplay":bool}
 *   3. Runner writes {"type":"init", ...}
 *   4. Each tick: runner writes {"type":"observation",...}, Python responds with action JSON
 *   5. When match ends: runner writes {"type":"result",...}
 *   6. Go to step 2 (next episode)
 *
 * Legacy Protocol (no --persistent):
 *   Same as before: args on command line, single episode, then exit.
 */

import * as fs from 'fs';
import * as path from 'path';
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

// ---- Shared readline setup ----
const rl = readline.createInterface({ input: process.stdin, terminal: false });
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
	process.exit(0);
});

function readLine(): Promise<string> {
	if (lineQueue.length > 0) {
		return Promise.resolve(lineQueue.shift() ?? '');
	}
	return new Promise<string>((resolve) => {
		lineResolve = () => resolve(lineQueue.shift() ?? '');
	});
}

function parseArgs(): { level: number; seed: number; maxTicks: number; saveReplay: boolean; persistent: boolean } {
	const args = process.argv.slice(2);
	let level = 1;
	let seed = 42;
	let maxTicks = 3600;
	let saveReplay = false;
	let persistent = false;

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
		} else if (args[i] === '--save-replay') {
			saveReplay = true;
		} else if (args[i] === '--persistent') {
			persistent = true;
		}
	}

	level = Math.max(1, Math.min(level, LEVEL_CONFIGS.length));
	return { level, seed, maxTicks, saveReplay, persistent };
}

// ---- Run a single episode ----
async function runEpisode(level: number, seed: number, maxTicks: number, saveReplay: boolean): Promise<void> {
	const levelConfig = LEVEL_CONFIGS[level - 1];
	const initialState = createInitialGameState(levelConfig, seed);
	const defaultControllers = createDefaultControllers(levelConfig);

	const playerTankId = 'player-0';
	const playerController = new ExternalController();
	const controllers: Record<string, TankController> = {
		...defaultControllers,
		[playerTankId]: playerController,
	};

	const simulation = new Simulation(initialState, controllers, { debugFreeze: false });
	const replayRecorder = new ReplayRecorder(levelConfig, seed);

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

	const buildObservation = (simulation: Simulation): TankObservation | null => {
		const state = simulation.getState();
		const currentPlayerTank = state.tanks.find((t) => t.id === playerTankId);
		if (!currentPlayerTank) {
			return null;
		}
		return {
			tick: state.tick,
			self: JSON.parse(JSON.stringify(currentPlayerTank)),
			enemies: state.tanks.filter((t) => t.team !== currentPlayerTank.team).map((t) => JSON.parse(JSON.stringify(t))),
			projectiles: state.projectiles.map((p) => JSON.parse(JSON.stringify(p))),
			bombs: state.bombs.map((b) => JSON.parse(JSON.stringify(b))),
			obstacles: state.obstacles.map((o) => JSON.parse(JSON.stringify(o))),
			arena: { ...state.arena },
		};
	};

	let tick = 0;
	while (tick < maxTicks) {
		const state = simulation.getState();
		if (state.status !== 'running') break;

		const playerTank = state.tanks.find((t) => t.id === playerTankId);
		if (!playerTank || playerTank.destroyed) break;

		const obs = buildObservation(simulation);
		if (!obs) {
			break;
		}

		writeLine({ type: 'observation', tick: state.tick, observation: obs });

		const actionLine = await readLine();
		let actionData: { move?: MoveIntent; aimAngle?: number; fire?: boolean; plantBomb?: boolean };
		try {
			actionData = JSON.parse(actionLine);
		} catch {
			actionData = { move: 'none', aimAngle: 0, fire: false, plantBomb: false };
		}

		playerController.pendingAction = {
			move: actionData.move ?? 'none',
			aimAngle: actionData.aimAngle ?? 0,
			fire: actionData.fire ?? false,
			plantBomb: actionData.plantBomb ?? false,
		};

		const stepResult: SimulationStepResult = simulation.step();
		replayRecorder.record(stepResult);
		tick++;
	}

	const finalState = simulation.getStateSnapshot();
	const playerTank = finalState.tanks.find((t) => t.id === playerTankId);
	const enemiesDestroyed = finalState.tanks.filter((t) => t.team === 'enemy' && t.destroyed).length;
	const totalEnemies = finalState.tanks.filter((t) => t.team === 'enemy').length;
	const finalObservation = buildObservation(simulation);

	writeLine({
		type: 'result',
		status: finalState.status,
		ticks: finalState.tick,
		playerDestroyed: playerTank?.destroyed ?? true,
		enemiesDestroyed,
		totalEnemies,
		win: finalState.status === 'player_win',
		loss: finalState.status === 'enemy_win',
		draw: finalState.status === 'running',
		timeout: finalState.status === 'running',
		observation: finalObservation,
	});

	if (saveReplay) {
		const replayDir = path.join(__dirname, '..', '..', 'training', 'output', 'replays');
		fs.mkdirSync(replayDir, { recursive: true });
		const replayPath = path.join(replayDir, `replay_L${level}_S${seed}.json`);
		fs.writeFileSync(replayPath, JSON.stringify(replayRecorder.toJSON()));
	}
}

// ---- Persistent mode: stay alive, receive reset commands ----
async function runPersistent(): Promise<void> {
	writeLine({ type: 'ready' });

	while (true) {
		const line = await readLine();
		let cmd: { type: string; level?: number; seed?: number; maxTicks?: number; saveReplay?: boolean };
		try {
			cmd = JSON.parse(line);
		} catch {
			continue;
		}
		if (cmd.type === 'reset') {
			const level = Math.max(1, Math.min(cmd.level ?? 1, LEVEL_CONFIGS.length));
			const seed = cmd.seed ?? 42;
			const maxTicks = cmd.maxTicks ?? 3600;
			const saveReplay = cmd.saveReplay ?? false;
			await runEpisode(level, seed, maxTicks, saveReplay);
		} else if (cmd.type === 'exit') {
			process.exit(0);
		}
	}
}

// ---- Legacy single-episode mode ----
async function runSingleEpisode(): Promise<void> {
	const { level, seed, maxTicks, saveReplay } = parseArgs();
	await runEpisode(level, seed, maxTicks, saveReplay);
	process.exit(0);
}

// ---- Entry point ----
const { persistent } = parseArgs();
if (persistent) {
	runPersistent().catch((err) => {
		process.stderr.write(`CLI runner error: ${err}\n`);
		process.exit(1);
	});
} else {
	runSingleEpisode().catch((err) => {
		process.stderr.write(`CLI runner error: ${err}\n`);
		process.exit(1);
	});
}
