import * as ort from 'onnxruntime-web';
import type { MatchInit, TankAction, TankController, TankObservation } from '../core/types';
import { NavigationPlanner } from '../navigation/NavigationPlanner';
import {
	ACTION_DIM,
	ACTION_HEAD_SIZES,
	ACTION_LOGITS_DIM,
	assertCompatibleContract,
	decodePolicyAction,
	OBS_SIZE,
} from './NeuralModelContract';
import { normalizePolicyObservation } from './NeuralObservation';

export class NeuralNetController implements TankController {
	private session: ort.InferenceSession | null = null;
	private navigationPlanner: NavigationPlanner | null = null;
	private ready = false;
	private inferenceInFlight = false;
	private queuedRequest: { input: Float32Array; obs: TankObservation; generation: number } | null = null;
	private lastAction: TankAction | null = null;
	private generation = 0;

	constructor(private modelUrl: string) {}

	public async loadModel(): Promise<void> {
		const contractUrl = this.modelUrl.replace(/\.onnx(?=($|\?))/, '.contract.json');
		const response = await fetch(contractUrl);
		if (!response.ok) {
			throw new Error(`Model contract request failed with HTTP ${response.status}.`);
		}
		assertCompatibleContract(await response.json());
		const session = await ort.InferenceSession.create(this.modelUrl, {
			executionProviders: ['wasm'],
		});
		const inputIndex = session.inputNames.indexOf('observation');
		const outputIndex = session.outputNames.indexOf('logits');
		const inputMetadata = inputIndex >= 0 ? session.inputMetadata[inputIndex] : null;
		const outputMetadata = outputIndex >= 0 ? session.outputMetadata[outputIndex] : null;
		const inputWidth = inputMetadata?.isTensor ? inputMetadata.shape.at(-1) : null;
		const outputWidth = outputMetadata?.isTensor ? outputMetadata.shape.at(-1) : null;
		if (inputWidth !== OBS_SIZE || outputWidth !== ACTION_LOGITS_DIM) {
			await session.release();
			throw new Error(
				`ONNX tensor contract mismatch: observation=${String(inputWidth)}, logits=${String(outputWidth)}.`
			);
		}
		this.session = session;
		this.ready = true;
	}

	public reset(initial: MatchInit): void {
		this.generation += 1;
		this.lastAction = null;
		this.queuedRequest = null;
		const self = initial.tanks.find((tank) => tank.id === initial.selfId);
		this.navigationPlanner = new NavigationPlanner(initial.arena, initial.obstacles, self?.size ?? 30);
	}

	public act(obs: TankObservation): TankAction {
		if (!this.ready || !this.session) {
			return { move: 'none', aimAngle: obs.self.aimAngle, fire: false, plantBomb: false };
		}

		const input = new Float32Array(normalizePolicyObservation(obs, 720, this.navigationPlanner));
		void this.runInferenceAsync(input, obs, this.generation);
		return this.lastAction ?? { move: 'none', aimAngle: obs.self.aimAngle, fire: false, plantBomb: false };
	}

	private async runInferenceAsync(input: Float32Array, obs: TankObservation, generation: number): Promise<void> {
		if (!this.session) return;
		if (this.inferenceInFlight) {
			this.queuedRequest = { input, obs, generation };
			return;
		}

		this.inferenceInFlight = true;
		try {
			const tensor = new ort.Tensor('float32', input, [1, OBS_SIZE]);
			const results = await this.session.run({ observation: tensor });
			const output = results['logits'];
			if (!output) {
				throw new Error('ONNX model returned no logits output.');
			}
			if (generation === this.generation) {
				this.lastAction = this.decodeAction(Array.from(output.data as Float32Array), obs);
			}
		} catch (error) {
			console.error('ONNX inference error:', error);
		} finally {
			this.inferenceInFlight = false;
			if (this.queuedRequest) {
				const nextRequest = this.queuedRequest;
				this.queuedRequest = null;
				void this.runInferenceAsync(nextRequest.input, nextRequest.obs, nextRequest.generation);
			}
		}
	}

	private decodeAction(logits: number[], obs: TankObservation): TankAction {
		if (logits.length < ACTION_LOGITS_DIM) {
			return { move: 'none', aimAngle: obs.self.aimAngle, fire: false, plantBomb: false };
		}

		const indices = new Array<number>(ACTION_DIM);
		let offset = 0;
		for (let head = 0; head < ACTION_DIM; head++) {
			const size = ACTION_HEAD_SIZES[head];
			let bestIndex = 0;
			let bestValue = Number.NEGATIVE_INFINITY;
			for (let index = 0; index < size; index++) {
				const value = logits[offset + index];
				if (value > bestValue) {
					bestValue = value;
					bestIndex = index;
				}
			}
			indices[head] = bestIndex;
			offset += size;
		}
		return decodePolicyAction(indices, obs);
	}
}
