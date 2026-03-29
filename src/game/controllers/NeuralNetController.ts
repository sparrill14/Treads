import * as ort from 'onnxruntime-web';
import type { MatchInit, MoveIntent, TankAction, TankController, TankObservation } from '../core/types';

const ARENA_WIDTH = 1000;
const ARENA_HEIGHT = 500;
const MAX_ENEMIES = 3;
const MAX_PROJECTILES = 5;
const MAX_OBSTACLES = 3;
const SELF_DIM = 8;
const ENEMY_DIM = 5;
const PROJ_DIM = 5;
const OBS_DIM = 4;
const OBS_SIZE = SELF_DIM + MAX_ENEMIES * ENEMY_DIM + MAX_PROJECTILES * PROJ_DIM + MAX_OBSTACLES * OBS_DIM;

// MultiDiscrete action dimensions: [move(9), aim(16), fire(2), bomb(2)]
const ACTION_DIMS = [9, 16, 2, 2];
const NUM_AIM_BINS = 16;

const MOVE_INTENTS: MoveIntent[] = ['none', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];

export class NeuralNetController implements TankController {
	private session: ort.InferenceSession | null = null;
	private ready = false;

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

	private async runInferenceAsync(input: Float32Array): Promise<void> {
		if (!this.session) return;
		try {
			const tensor = new ort.Tensor('float32', input, [1, OBS_SIZE]);
			const feeds = { observation: tensor };
			const results = await this.session.run(feeds);
			const output = results['logits'];
			const data = output.data as Float32Array;
			this.pendingAction = this.decodeAction(Array.from(data));
		} catch (err) {
			console.error('ONNX inference error:', err);
		}
	}

	private decodeAction(logits: number[]): TankAction {
		// Logits layout: [move(9), aim(16), fire(2), bomb(2)] = 29 total
		// Take argmax within each group
		let offset = 0;
		const choices: number[] = [];
		for (const dim of ACTION_DIMS) {
			const group = logits.slice(offset, offset + dim);
			let bestIdx = 0;
			let bestVal = group[0];
			for (let i = 1; i < group.length; i++) {
				if (group[i] > bestVal) {
					bestVal = group[i];
					bestIdx = i;
				}
			}
			choices.push(bestIdx);
			offset += dim;
		}

		const moveIdx = choices[0];
		const aimBin = choices[1];
		const fireChoice = choices[2];
		const bombChoice = choices[3];

		return {
			move: MOVE_INTENTS[moveIdx],
			aimAngle: (aimBin / NUM_AIM_BINS) * 2 * Math.PI,
			fire: fireChoice === 1,
			plantBomb: bombChoice === 1,
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
		idx += SELF_DIM;

		// Enemies sorted by distance
		const sx = obs.self.x + obs.self.size / 2;
		const sy = obs.self.y + obs.self.size / 2;
		const livingEnemies = obs.enemies
			.filter((e) => !e.destroyed)
			.sort((a, b) => {
				const da = (a.x + a.size / 2 - sx) ** 2 + (a.y + a.size / 2 - sy) ** 2;
				const db = (b.x + b.size / 2 - sx) ** 2 + (b.y + b.size / 2 - sy) ** 2;
				return da - db;
			});

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
