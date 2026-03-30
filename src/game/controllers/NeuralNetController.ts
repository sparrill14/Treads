import * as ort from 'onnxruntime-web';
import type { MatchInit, MoveIntent, TankAction, TankController, TankObservation } from '../core/types';

const ARENA_WIDTH = 1000;
const ARENA_HEIGHT = 500;
const MAX_ENEMIES = 3;
const MAX_PROJECTILES = 5;
const MAX_OBSTACLES = 3;
const SELF_DIM = 11;
const ENEMY_DIM = 5;
const PROJ_DIM = 5;
const OBS_DIM = 4;
const OBS_SIZE = SELF_DIM + MAX_ENEMIES * ENEMY_DIM + MAX_PROJECTILES * PROJ_DIM + MAX_OBSTACLES * OBS_DIM;

// Continuous action means: [move_signal, aim_signal, fire_signal, bomb_signal] in [-1, 1]
const ACTION_DIM = 4;
const FIRE_THRESHOLD = -0.2;
const BOMB_THRESHOLD = 0.8;

const MOVE_INTENTS: MoveIntent[] = ['none', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];

export class NeuralNetController implements TankController {
	private session: ort.InferenceSession | null = null;
	private ready = false;
	private inferenceInFlight = false;
	private queuedInput: Float32Array | null = null;

	constructor(private modelUrl: string) {}

	/**
	 * Load the ONNX model. Must be called before act().
	 * Can be called from any async context.
	 */
	public async loadModel(): Promise<void> {
		this.session = await ort.InferenceSession.create(this.modelUrl, {
			executionProviders: ['wasm'],
		});
		this.ready = true;
	}

	public reset(_initial: MatchInit): void {
		// no state to reset
	}

	public act(obs: TankObservation): TankAction {
		this.lastObs = obs;
		if (!this.ready || !this.session) {
			// Fallback: no-op until model is loaded
			return { move: 'none', aimAngle: obs.self.aimAngle, fire: false, plantBomb: false };
		}

		// Normalize observation synchronously
		const input = this.normalizeObservation(obs);

		// Run inference synchronously using our cached last result
		// Since ONNX inference is async, we use a fire-and-forget pattern
		// with a cached result from the previous tick
		if (this.pendingAction) {
			const action = this.pendingAction;
			this.pendingAction = null;
			this.runInferenceAsync(input);
			return action;
		}

		// First tick: return a default action and start inference
		this.runInferenceAsync(input);
		return { move: 'none', aimAngle: obs.self.aimAngle, fire: false, plantBomb: false };
	}

	private pendingAction: TankAction | null = null;
	private lastObs: TankObservation | null = null;

	private async runInferenceAsync(input: Float32Array): Promise<void> {
		if (!this.session) return;
		if (this.inferenceInFlight) {
			// Keep only the most recent input to avoid unbounded queue growth.
			this.queuedInput = input;
			return;
		}

		this.inferenceInFlight = true;
		try {
			const tensor = new ort.Tensor('float32', input, [1, OBS_SIZE]);
			const feeds = { observation: tensor };
			const results = await this.session.run(feeds);
			const output = results['action_mean'] ?? results['logits'];
			const data = output.data as Float32Array;
			this.pendingAction = this.decodeAction(Array.from(data));
		} catch (err) {
			console.error('ONNX inference error:', err);
		} finally {
			this.inferenceInFlight = false;
			if (this.queuedInput) {
				const nextInput = this.queuedInput;
				this.queuedInput = null;
				void this.runInferenceAsync(nextInput);
			}
		}
	}

	private decodeAction(logits: number[]): TankAction {
		if (logits.length < ACTION_DIM) {
			return {
				move: 'none',
				aimAngle: this.lastObs?.self.aimAngle ?? 0,
				fire: false,
				plantBomb: false,
			};
		}

		const moveSignal = Math.max(-1, Math.min(1, logits[0]));
		const aimSignal = Math.max(-1, Math.min(1, logits[1]));
		const fireSignal = Math.max(-1, Math.min(1, logits[2]));
		const bombSignal = Math.max(-1, Math.min(1, logits[3]));

		const moveIdx = Math.max(0, Math.min(8, Math.round((moveSignal + 1) * 0.5 * 8)));

		// Enemy-relative aim encoding: aim_signal=0 → pointed at nearest enemy
		// aim_signal=±1 → pointed 180° away from enemy
		const obs = this.lastObs;
		let aimAngle: number;
		if (obs) {
			const s = obs.self;
			const sx = s.x + s.size / 2;
			const sy = s.y + s.size / 2;
			const aliveEnemies = obs.enemies.filter((e) => !e.destroyed);
			if (aliveEnemies.length > 0) {
				const nearest = aliveEnemies.reduce((best, e) => {
					const dx = e.x + e.size / 2 - sx;
					const dy = e.y + e.size / 2 - sy;
					const bdx = best.x + best.size / 2 - sx;
					const bdy = best.y + best.size / 2 - sy;
					return dx * dx + dy * dy < bdx * bdx + bdy * bdy ? e : best;
				});
				const angleToEnemy = Math.atan2(nearest.y + nearest.size / 2 - sy, nearest.x + nearest.size / 2 - sx);
				aimAngle = angleToEnemy + aimSignal * Math.PI;
			} else {
				aimAngle = s.aimAngle;
			}
		} else {
			aimAngle = 0;
		}

		return {
			move: MOVE_INTENTS[moveIdx],
			aimAngle,
			fire: fireSignal > FIRE_THRESHOLD,
			plantBomb: bombSignal > BOMB_THRESHOLD,
		};
	}

	private normalizeObservation(obs: TankObservation): Float32Array {
		const result = new Float32Array(OBS_SIZE);
		let idx = 0;

		// Self state
		result[idx] = obs.self.x / ARENA_WIDTH;
		result[idx + 1] = obs.self.y / ARENA_HEIGHT;
		result[idx + 2] = obs.self.aimAngle / (2 * Math.PI);
		result[idx + 3] = obs.self.speed / 100;
		result[idx + 4] = obs.self.destroyed ? 1 : 0;
		result[idx + 5] = Math.min(obs.self.shotCooldownTicks / 300, 1);
		result[idx + 6] = obs.self.activeAmmo / Math.max(obs.self.maxAmmo, 1);
		result[idx + 7] = obs.self.maxAmmo / 5;

		// Derived aim features
		const sx = obs.self.x + obs.self.size / 2;
		const sy = obs.self.y + obs.self.size / 2;
		const livingEnemies = obs.enemies
			.filter((e) => !e.destroyed)
			.sort((a, b) => {
				const da = (a.x + a.size / 2 - sx) ** 2 + (a.y + a.size / 2 - sy) ** 2;
				const db = (b.x + b.size / 2 - sx) ** 2 + (b.y + b.size / 2 - sy) ** 2;
				return da - db;
			});

		if (livingEnemies.length > 0) {
			const nearest = livingEnemies[0];
			const ex = nearest.x + nearest.size / 2;
			const ey = nearest.y + nearest.size / 2;
			const angleToEnemy = Math.atan2(ey - sy, ex - sx);
			const distToEnemy = Math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2);
			const aimAngle = obs.self.aimAngle;
			const aimError = Math.atan2(Math.sin(aimAngle - angleToEnemy), Math.cos(aimAngle - angleToEnemy));
			const arenaDiag = Math.sqrt(ARENA_WIDTH ** 2 + ARENA_HEIGHT ** 2);
			result[idx + 8] = angleToEnemy / (2 * Math.PI) + 0.5;
			result[idx + 9] = Math.min(distToEnemy / arenaDiag, 1.0);
			result[idx + 10] = (aimError / Math.PI) * 0.5 + 0.5;
		} else {
			result[idx + 8] = 0.5;
			result[idx + 9] = 0.0;
			result[idx + 10] = 0.5;
		}
		idx += SELF_DIM;

		// Enemies sorted by distance (reuse livingEnemies computed above)
		for (let i = 0; i < MAX_ENEMIES; i++) {
			if (i < livingEnemies.length) {
				const e = livingEnemies[i];
				result[idx] = e.x / ARENA_WIDTH;
				result[idx + 1] = e.y / ARENA_HEIGHT;
				result[idx + 2] = e.aimAngle / (2 * Math.PI);
				result[idx + 3] = e.speed / 100;
				result[idx + 4] = 0; // alive
			}
			idx += ENEMY_DIM;
		}

		// Projectiles sorted by distance
		const projectiles = [...obs.projectiles].sort((a, b) => {
			const da = (a.x - sx) ** 2 + (a.y - sy) ** 2;
			const db = (b.x - sx) ** 2 + (b.y - sy) ** 2;
			return da - db;
		});

		for (let i = 0; i < MAX_PROJECTILES; i++) {
			if (i < projectiles.length) {
				const p = projectiles[i];
				result[idx] = p.x / ARENA_WIDTH;
				result[idx + 1] = p.y / ARENA_HEIGHT;
				result[idx + 2] = (p.vx / 300) * 0.5 + 0.5;
				result[idx + 3] = (p.vy / 300) * 0.5 + 0.5;
				result[idx + 4] = p.team === 'enemy' ? 1 : 0;
			}
			idx += PROJ_DIM;
		}

		// Obstacles sorted by distance
		const obstacles = [...obs.obstacles].sort((a, b) => {
			const da = (a.x + a.width / 2 - sx) ** 2 + (a.y + a.height / 2 - sy) ** 2;
			const db = (b.x + b.width / 2 - sx) ** 2 + (b.y + b.height / 2 - sy) ** 2;
			return da - db;
		});

		for (let i = 0; i < MAX_OBSTACLES; i++) {
			if (i < obstacles.length) {
				const o = obstacles[i];
				result[idx] = o.x / ARENA_WIDTH;
				result[idx + 1] = o.y / ARENA_HEIGHT;
				result[idx + 2] = o.width / ARENA_WIDTH;
				result[idx + 3] = o.height / ARENA_HEIGHT;
			}
			idx += OBS_DIM;
		}

		// Clamp to [0, 1]
		for (let i = 0; i < OBS_SIZE; i++) {
			result[i] = Math.max(0, Math.min(1, result[i]));
		}

		return result;
	}
}
