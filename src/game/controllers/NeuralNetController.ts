import type { MatchInit, TankAction, TankController, TankObservation } from '../core/types';
import { NavigationPlanner } from '../navigation/NavigationPlanner';
import { assertCompatibleContract, decodePolicyAction } from './NeuralModelContract';
import { normalizePolicyObservation, POLICY_TICK_NORM_TICKS } from './NeuralObservation';
import { parseWeightsFromStateDict, PolicyMLP } from './PolicyMLP';

/** Runs the rollout worker's synchronous actor without delayed observations or stale actions. */
export class NeuralNetController implements TankController {
	private policy: PolicyMLP | null = null;
	private navigationPlanner: NavigationPlanner | null = null;

	constructor(private modelUrl: string) {}

	public async loadModel(): Promise<void> {
		const companion = (extension: string): string => this.modelUrl.replace(/\.onnx(?=($|\?))/, extension);
		const contract = await fetch(companion('.contract.json'));
		if (!contract.ok) throw new Error(`Model contract request failed with HTTP ${contract.status}.`);
		assertCompatibleContract(await contract.json());
		const weights = await fetch(companion('.weights.json'));
		if (!weights.ok) throw new Error(`Policy weights request failed with HTTP ${weights.status}. Re-export the checkpoint.`);
		this.policy = new PolicyMLP(parseWeightsFromStateDict(await weights.json()));
	}

	public reset(initial: MatchInit): void {
		const self = initial.tanks.find((tank) => tank.id === initial.selfId);
		this.navigationPlanner = new NavigationPlanner(initial.arena, initial.obstacles, self?.size ?? 30);
	}

	public act(obs: TankObservation): TankAction {
		if (!this.policy) return { move: 'none', aimAngle: obs.self.aimAngle, fire: false, plantBomb: false };
		const input = normalizePolicyObservation(obs, POLICY_TICK_NORM_TICKS, this.navigationPlanner);
		return decodePolicyAction(this.policy.argmaxMultiCategorical(this.policy.policyLogits(input)), obs);
	}
}
