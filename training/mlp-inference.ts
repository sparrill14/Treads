/**
 * Simple MLP forward pass for PPO policy inference in TypeScript.
 * Avoids per-tick IPC by running the neural network in-process.
 *
 * Architecture matches SB3's MlpPolicy with net_arch=[512, 256]:
 *   obs (current normalized runtime vector) → policyNet: Linear(obs_dim→512) → Tanh → Linear(512→256) → Tanh → policy_features
 *   obs (current normalized runtime vector) → valueNet:  Linear(obs_dim→512) → Tanh → Linear(512→256) → Tanh → value_features
 *   policy_features → action_net Linear(256→5) → action_mean [move_x, move_y, aim, fire, bomb]
 *   value_features  → value_net  Linear(256→1)  → value
 */

export interface LayerWeights {
	weight: number[][]; // [out_features][in_features]
	bias: number[]; // [out_features]
}

export interface PolicyWeights {
	policyLayers: LayerWeights[];
	valueLayers: LayerWeights[];
	actionHead: LayerWeights;
	actionLogStd: number[];
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

export class PolicyMLP {
	constructor(private weights: PolicyWeights) {}

	forward(obs: number[]): { actionMean: number[]; value: number } {
		// Policy network
		let policyFeatures = obs;
		for (const layer of this.weights.policyLayers) {
			policyFeatures = tanhActivation(linear(policyFeatures, layer.weight, layer.bias));
		}
		const actionMean = linear(policyFeatures, this.weights.actionHead.weight, this.weights.actionHead.bias);

		// Value network
		let valueFeatures = obs;
		for (const layer of this.weights.valueLayers) {
			valueFeatures = tanhActivation(linear(valueFeatures, layer.weight, layer.bias));
		}
		const valueArr = linear(valueFeatures, this.weights.valueHead.weight, this.weights.valueHead.bias);

		return { actionMean, value: valueArr[0] };
	}

	sampleGaussianAction(actionMean: number[], rng: () => number = Math.random): { action: number[]; logProb: number } {
		const action: number[] = [];
		let logProb = 0;
		for (let i = 0; i < actionMean.length; i++) {
			const std = Math.exp(this.weights.actionLogStd[i]);
			const sample = actionMean[i] + std * sampleStandardNormal(rng);
			action.push(sample);
			const diff = sample - actionMean[i];
			logProb += -0.5 * ((diff * diff) / (std * std) + 2 * Math.log(std) + Math.log(2 * Math.PI));
		}
		return { action, logProb };
	}

	getActionLogStd(): number[] {
		return [...this.weights.actionLogStd];
	}
}

function sampleStandardNormal(rng: () => number): number {
	let u1 = 0;
	let u2 = 0;
	while (u1 <= Number.EPSILON) {
		u1 = rng();
		u2 = rng();
	}
	return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

/**
 * Parse weights from SB3 state_dict format into PolicyWeights.
 * Expected keys (for net_arch=[512, 256] with separate pi/vf networks):
 *   mlp_extractor.policy_net.0.weight  [512, obs_dim]
 *   mlp_extractor.policy_net.0.bias    [512]
 *   mlp_extractor.policy_net.2.weight  [256, 512]
 *   mlp_extractor.policy_net.2.bias    [256]
 *   mlp_extractor.value_net.0.weight   [512, obs_dim]
 *   mlp_extractor.value_net.0.bias     [512]
 *   mlp_extractor.value_net.2.weight   [256, 512]
 *   mlp_extractor.value_net.2.bias     [256]
 *   action_net.weight                  [5, 256]
 *   action_net.bias                    [5]
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
		actionLogStd: stateDict['log_std'] as number[],
		valueHead: {
			weight: stateDict['value_net.weight'] as number[][],
			bias: stateDict['value_net.bias'] as number[],
		},
	};
}
