import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { randomUUID } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

import { createDefaultControllers, createInitialGameState } from '../src/game/core/MatchFactory';
import { ReplayRecorder } from '../src/game/core/Replay';
import { Simulation } from '../src/game/core/Simulation';
import type {
	MoveIntent,
	SimulationStepResult,
	TankAction,
	TankController,
	TankObservation,
} from '../src/game/core/types';
import { LEVEL_CONFIGS } from '../src/game/LevelConfig';

interface ResetRequest {
	sessionId?: string;
	level?: number;
	seed?: number;
	maxTicks?: number;
	saveReplay?: boolean;
}

interface StepRequest {
	sessionId?: string;
	move?: string;
	aimAngle?: number;
	fire?: boolean;
	plantBomb?: boolean;
}

interface CloseRequest {
	sessionId?: string;
}

class ExternalController implements TankController {
	public pendingAction: TankAction = {
		move: 'none',
		aimAngle: 0,
		fire: false,
		plantBomb: false,
	};

	public reset(): void {
		return;
	}

	public act(_obs: TankObservation): TankAction {
		return this.pendingAction;
	}
}

class EnvSession {
	private readonly level: number;
	private readonly seed: number;
	private readonly maxTicks: number;
	private readonly saveReplay: boolean;
	private readonly playerTankId = 'player-0';
	private readonly playerController = new ExternalController();
	private readonly simulation: Simulation;
	private readonly replayRecorder: ReplayRecorder;
	private tick = 0;

	public readonly initPayload: Record<string, unknown>;

	public constructor(level: number, seed: number, maxTicks: number, saveReplay: boolean) {
		this.level = level;
		this.seed = seed;
		this.maxTicks = maxTicks;
		this.saveReplay = saveReplay;

		const levelConfig = LEVEL_CONFIGS[Math.max(1, Math.min(level, LEVEL_CONFIGS.length)) - 1];
		const initialState = createInitialGameState(levelConfig, seed);
		const defaultControllers = createDefaultControllers(levelConfig);
		const controllers: Record<string, TankController> = {
			...defaultControllers,
			[this.playerTankId]: this.playerController,
		};
		this.simulation = new Simulation(initialState, controllers, { debugFreeze: false });
		this.replayRecorder = new ReplayRecorder(levelConfig, seed);

		this.initPayload = {
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
			playerTankId: this.playerTankId,
			aiTankIds: [this.playerTankId],
		};
	}

	public buildObservation(): TankObservation | null {
		const state = this.simulation.getState();
		const currentPlayerTank = state.tanks.find((t) => t.id === this.playerTankId);
		if (!currentPlayerTank) {
			return null;
		}
		return {
			tick: state.tick,
			self: JSON.parse(JSON.stringify(currentPlayerTank)),
			allies: state.tanks
				.filter((t) => t.team === currentPlayerTank.team && t.id !== currentPlayerTank.id)
				.map((t) => JSON.parse(JSON.stringify(t))),
			enemies: state.tanks.filter((t) => t.team !== currentPlayerTank.team).map((t) => JSON.parse(JSON.stringify(t))),
			projectiles: state.projectiles.map((p) => JSON.parse(JSON.stringify(p))),
			bombs: state.bombs.map((b) => JSON.parse(JSON.stringify(b))),
			obstacles: state.obstacles.map((o) => JSON.parse(JSON.stringify(o))),
			arena: { ...state.arena },
		};
	}

	public step(action: { move?: MoveIntent; aimAngle?: number; fire?: boolean; plantBomb?: boolean }): {
		done: boolean;
		observation: TankObservation | null;
		result: Record<string, unknown> | null;
	} {
		const state = this.simulation.getState();
		if (state.status !== 'running') {
			return this.finalResult();
		}
		const playerTank = state.tanks.find((t) => t.id === this.playerTankId);
		if (!playerTank || playerTank.destroyed || this.tick >= this.maxTicks) {
			return this.finalResult();
		}

		this.playerController.pendingAction = {
			move: action.move ?? 'none',
			aimAngle: Number.isFinite(action.aimAngle) ? (action.aimAngle as number) : 0,
			fire: Boolean(action.fire),
			plantBomb: Boolean(action.plantBomb),
		};

		const stepResult: SimulationStepResult = this.simulation.step();
		this.replayRecorder.record(stepResult);
		this.tick += 1;

		const nextState = this.simulation.getState();
		if (nextState.status !== 'running' || this.tick >= this.maxTicks) {
			return this.finalResult();
		}

		const observation = this.buildObservation();
		if (!observation) {
			return this.finalResult();
		}
		return {
			done: false,
			observation,
			result: null,
		};
	}

	private finalResult(): { done: boolean; observation: TankObservation | null; result: Record<string, unknown> } {
		const finalState = this.simulation.getStateSnapshot();
		const playerTank = finalState.tanks.find((t) => t.id === this.playerTankId);
		const enemiesDestroyed = finalState.tanks.filter((t) => t.team === 'enemy' && t.destroyed).length;
		const totalEnemies = finalState.tanks.filter((t) => t.team === 'enemy').length;
		const finalObservation = this.buildObservation();

		const result: Record<string, unknown> = {
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
		};

		if (this.saveReplay) {
			const replayDir = path.join(__dirname, '..', 'training', 'output', 'replays');
			fs.mkdirSync(replayDir, { recursive: true });
			const replayPath = path.join(replayDir, `replay_L${this.level}_S${this.seed}.json`);
			fs.writeFileSync(replayPath, JSON.stringify(this.replayRecorder.toJSON()));
		}

		return {
			done: true,
			observation: finalObservation,
			result,
		};
	}
}

const PROTO_PATH = path.join(__dirname, '..', '..', 'training', 'proto', 'treads.proto');
const packageDef = protoLoader.loadSync(PROTO_PATH, {
	keepCase: false,
	longs: String,
	enums: String,
	defaults: true,
	oneofs: true,
});
const grpcObj = grpc.loadPackageDefinition(packageDef) as grpc.GrpcObject;
const treadsPkg = grpcObj.treads as grpc.GrpcObject;
const serviceDef = treadsPkg.TreadsEnvService as grpc.ServiceClientConstructor & { service: grpc.ServiceDefinition };

const sessions = new Map<string, EnvSession>();

function getPortArg(): number {
	const args = process.argv.slice(2);
	for (let i = 0; i < args.length; i += 1) {
		if (args[i] === '--port' && args[i + 1]) {
			return Number(args[i + 1]);
		}
	}
	return 50051;
}

function health(
	_call: grpc.ServerUnaryCall<Record<string, never>, { ok: boolean; message: string }>,
	callback: grpc.sendUnaryData<{ ok: boolean; message: string }>
): void {
	callback(null, { ok: true, message: 'ready' });
}

function reset(
	call: grpc.ServerUnaryCall<ResetRequest, { sessionId: string; initJson: string; observationJson: string }>,
	callback: grpc.sendUnaryData<{ sessionId: string; initJson: string; observationJson: string }>
): void {
	try {
		const req = call.request;
		const sessionId = req.sessionId || randomUUID();
		const level = Math.max(1, Math.min(Number(req.level ?? 1), LEVEL_CONFIGS.length));
		const seed = Number(req.seed ?? 42);
		const maxTicks = Math.max(1, Number(req.maxTicks ?? 720));
		const saveReplay = Boolean(req.saveReplay);

		const session = new EnvSession(level, seed, maxTicks, saveReplay);
		sessions.set(sessionId, session);

		const observation = session.buildObservation();
		if (!observation) {
			callback({ code: grpc.status.INTERNAL, message: 'Failed to build initial observation' }, null);
			return;
		}

		callback(null, {
			sessionId,
			initJson: JSON.stringify(session.initPayload),
			observationJson: JSON.stringify(observation),
		});
	} catch (err) {
		callback({ code: grpc.status.INTERNAL, message: String(err) }, null);
	}
}

function step(
	call: grpc.ServerUnaryCall<StepRequest, { done: boolean; observationJson: string; resultJson: string }>,
	callback: grpc.sendUnaryData<{ done: boolean; observationJson: string; resultJson: string }>
): void {
	try {
		const req = call.request;
		const sessionId = req.sessionId ?? '';
		const session = sessions.get(sessionId);
		if (!session) {
			callback({ code: grpc.status.NOT_FOUND, message: `Unknown session: ${sessionId}` }, null);
			return;
		}

		const outcome = session.step({
			move: (req.move as MoveIntent | undefined) ?? 'none',
			aimAngle: Number(req.aimAngle ?? 0),
			fire: Boolean(req.fire),
			plantBomb: Boolean(req.plantBomb),
		});

		if (outcome.done) {
			sessions.delete(sessionId);
		}

		callback(null, {
			done: outcome.done,
			observationJson: JSON.stringify(outcome.observation ?? null),
			resultJson: JSON.stringify(outcome.result ?? null),
		});
	} catch (err) {
		callback({ code: grpc.status.INTERNAL, message: String(err) }, null);
	}
}

function close(
	call: grpc.ServerUnaryCall<CloseRequest, { closed: boolean }>,
	callback: grpc.sendUnaryData<{ closed: boolean }>
): void {
	const sessionId = call.request.sessionId;
	if (sessionId && sessions.has(sessionId)) {
		sessions.delete(sessionId);
	}
	callback(null, { closed: true });
}

function main(): void {
	const port = getPortArg();
	const server = new grpc.Server();
	server.addService(serviceDef.service, {
		Health: health,
		Reset: reset,
		Step: step,
		Close: close,
	});

	const bindAddress = `127.0.0.1:${port}`;
	server.bindAsync(bindAddress, grpc.ServerCredentials.createInsecure(), (err) => {
		if (err) {
			process.stderr.write(`gRPC bind error: ${String(err)}\n`);
			process.exit(1);
		}
	});

	const shutdown = (): void => {
		server.tryShutdown(() => process.exit(0));
	};
	process.on('SIGINT', shutdown);
	process.on('SIGTERM', shutdown);
}

main();
