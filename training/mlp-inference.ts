/**
 * MLP forward pass for PPO policy inference in TypeScript.
 * Matches SB3's MlpPolicy with the versioned MultiDiscrete action contract and
 * net_arch=[512, 256]:
 *   obs → policyNet: Linear(obs_dim→512) → Tanh → Linear(512→256) → Tanh
 *   obs → valueNet:  Linear(obs_dim→512) → Tanh → Linear(512→256) → Tanh
 *   policy_features → action_net → logits split into 4 heads
 *   value_features  → value_net  Linear(256→1)  → value
 *
 * Action heads:
 *   move:    9 logits → MoveIntent index
 *   aimBin: relative residual from the nearest enemy bearing
 *   fire:    2 logits → 0/1
 *   bomb:    2 logits → 0/1
 */

import { ACTION_DIM, ACTION_HEAD_SIZES, ACTION_LOGITS_DIM } from '../src/game/controllers/NeuralModelContract';

export { ACTION_DIM, ACTION_HEAD_SIZES, ACTION_LOGITS_DIM };

export interface LayerWeights {
	weight: number[][]; // [out_features][in_features]
	bias: number[]; // [out_features]
}

export interface PolicyWeights {
	policyLayers: LayerWeights[];
	valueLayers: LayerWeights[];
	actionHead: LayerWeights;
	valueHead: LayerWeights;
}

function linear(x: number[], weight: number[][], bias: number[]): number[] {
	const outDim = weight.length;
	const result = new Array<number>(outDim);
	for (let i = 0; i < outDim; i++) {
		let sum = bias[i];
		const wi = weight[i];
		for (let j = 0; j < x.length; j++) {
			sum += wi[j] * x[j];
		}
		result[i] = sum;
	}
	return result;
}

function tanhActivation(x: number[]): number[] {
	const result = new Array<number>(x.length);
	for (let i = 0; i < x.length; i++) {
		result[i] = Math.tanh(x[i]);
	}
	return result;
}

/** Numerically stable log-softmax for one head. */
function logSoftmax(logits: number[], offset: number, size: number): number[] {
	let maxV = -Infinity;
	for (let i = 0; i < size; i++) {
		const v = logits[offset + i];
		if (v > maxV) maxV = v;
	}
	let sumExp = 0;
	for (let i = 0; i < size; i++) {
		sumExp += Math.exp(logits[offset + i] - maxV);
	}
	const logSum = maxV + Math.log(sumExp);
	const out = new Array<number>(size);
	for (let i = 0; i < size; i++) {
		out[i] = logits[offset + i] - logSum;
	}
	return out;
}

/** Sample one categorical index from logits using inverse-CDF on softmax probs. */
function sampleCategorical(logProbs: number[], rng: () => number): number {
	let u = rng();
	// Avoid 0 to keep numerics safe.
	if (u <= 0) u = 1e-12;
	let cum = 0;
	for (let i = 0; i < logProbs.length; i++) {
		cum += Math.exp(logProbs[i]);
		if (u < cum) return i;
	}
	return logProbs.length - 1;
}

export class PolicyMLP {
	constructor(private weights: PolicyWeights) {
		const outDim = weights.actionHead.weight.length;
		if (outDim !== ACTION_LOGITS_DIM) {
			throw new Error(
				`PolicyMLP: action head output dim ${outDim} does not match expected ${ACTION_LOGITS_DIM} (heads ${ACTION_HEAD_SIZES.join(',')})`
			);
		}
	}

	forward(obs: number[]): { logits: number[]; value: number } {
		// Policy network
		let policyFeatures = obs;
		for (const layer of this.weights.policyLayers) {
			policyFeatures = tanhActivation(linear(policyFeatures, layer.weight, layer.bias));
		}
		const logits = linear(policyFeatures, this.weights.actionHead.weight, this.weights.actionHead.bias);

		// Value network
		let valueFeatures = obs;
		for (const layer of this.weights.valueLayers) {
			valueFeatures = tanhActivation(linear(valueFeatures, layer.weight, layer.bias));
		}
		const valueArr = linear(valueFeatures, this.weights.valueHead.weight, this.weights.valueHead.bias);

		return { logits, value: valueArr[0] };
	}

	/**
	 * Sample one action per discrete head from raw logits.
	 * Returns the per-head integer indices and the summed log-probability across heads.
	 */
	sampleMultiCategorical(logits: number[], rng: () => number = Math.random): { action: number[]; logProb: number } {
		if (logits.length !== ACTION_LOGITS_DIM) {
			throw new Error(`Expected ${ACTION_LOGITS_DIM} logits, got ${logits.length}`);
		}
		const action = new Array<number>(ACTION_DIM);
		let logProb = 0;
		let offset = 0;
		for (let h = 0; h < ACTION_DIM; h++) {
			const size = ACTION_HEAD_SIZES[h];
			const lp = logSoftmax(logits, offset, size);
			const idx = sampleCategorical(lp, rng);
			action[h] = idx;
			logProb += lp[idx];
			offset += size;
		}
		return { action, logProb };
	}

	/** Greedy (argmax) decoding — used for deterministic eval / browser. */
	argmaxMultiCategorical(logits: number[]): number[] {
		const action = new Array<number>(ACTION_DIM);
		let offset = 0;
		for (let h = 0; h < ACTION_DIM; h++) {
			const size = ACTION_HEAD_SIZES[h];
			let bestIdx = 0;
			let bestVal = -Infinity;
			for (let i = 0; i < size; i++) {
				const v = logits[offset + i];
				if (v > bestVal) {
					bestVal = v;
					bestIdx = i;
				}
			}
			action[h] = bestIdx;
			offset += size;
		}
		return action;
	}
}

/**
 * Parse weights from SB3 state_dict format into PolicyWeights.
 * Expected keys (for net_arch=[512, 256] with separate pi/vf networks and MultiDiscrete head):
 *   mlp_extractor.policy_net.0.weight  [512, obs_dim]
 *   mlp_extractor.policy_net.0.bias    [512]
 *   mlp_extractor.policy_net.2.weight  [256, 512]
 *   mlp_extractor.policy_net.2.bias    [256]
 *   mlp_extractor.value_net.0.weight   [512, obs_dim]
 *   mlp_extractor.value_net.0.bias     [512]
 *   mlp_extractor.value_net.2.weight   [256, 512]
 *   mlp_extractor.value_net.2.bias     [256]
 *   action_net.weight                  [29, 256]
 *   action_net.bias                    [29]
 *   value_net.weight                   [1, 256]
 *   value_net.bias                     [1]
 */
export function parseWeightsFromStateDict(stateDict: Record<string, number[][] | number[]>): PolicyWeights {
	return {
		policyLayers: [
			{
				weight: stateDict['mlp_extractor.policy_net.0.weight'] as number[][],
				bias: stateDict['mlp_extractor.policy_net.0.bias'] as number[],
			},
			{
				weight: stateDict['mlp_extractor.policy_net.2.weight'] as number[][],
				bias: stateDict['mlp_extractor.policy_net.2.bias'] as number[],
			},
		],
		valueLayers: [
			{
				weight: stateDict['mlp_extractor.value_net.0.weight'] as number[][],
				bias: stateDict['mlp_extractor.value_net.0.bias'] as number[],
			},
			{
				weight: stateDict['mlp_extractor.value_net.2.weight'] as number[][],
				bias: stateDict['mlp_extractor.value_net.2.bias'] as number[],
			},
		],
		actionHead: {
			weight: stateDict['action_net.weight'] as number[][],
			bias: stateDict['action_net.bias'] as number[],
		},
		valueHead: {
			weight: stateDict['value_net.weight'] as number[][],
			bias: stateDict['value_net.bias'] as number[],
		},
	};
}
