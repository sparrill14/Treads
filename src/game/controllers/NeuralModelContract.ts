import contractJson from './neural-model-contract.json';
import type { MoveIntent, TankAction, TankObservation } from '../core/types';

export interface NeuralModelContractData {
	contractVersion: number;
	observationVersion: string;
	actionVersion: string;
	observation: {
		selfDim: number;
		enemyDim: number;
		maxEnemies: number;
		projectileDim: number;
		maxProjectiles: number;
		obstacleDim: number;
		maxObstacles: number;
		bombDim: number;
		maxBombs: number;
		summaryDim: number;
		size: number;
	};
	action: {
		headSizes: number[];
		aimMode: string;
		aimResidualMinRadians: number;
		aimResidualMaxRadians: number;
		neutralAimBin: number;
	};
}

export const NEURAL_MODEL_CONTRACT = contractJson as NeuralModelContractData;
export const ACTION_HEAD_SIZES = NEURAL_MODEL_CONTRACT.action.headSizes;
export const ACTION_DIM = ACTION_HEAD_SIZES.length;
export const ACTION_LOGITS_DIM = ACTION_HEAD_SIZES.reduce((sum, size) => sum + size, 0);
export const NUM_AIM_BINS = ACTION_HEAD_SIZES[1];
export const OBS_SIZE = NEURAL_MODEL_CONTRACT.observation.size;
export const MOVE_INTENTS_BY_INDEX: MoveIntent[] = ['none', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];

export function unitAngleFeature(angle: number): [number, number] {
	return [(Math.sin(angle) + 1) * 0.5, (Math.cos(angle) + 1) * 0.5];
}

export function aimResidualForBin(aimBin: number): number {
	const clamped = Math.max(0, Math.min(NUM_AIM_BINS - 1, Math.trunc(aimBin)));
	const min = NEURAL_MODEL_CONTRACT.action.aimResidualMinRadians;
	const max = NEURAL_MODEL_CONTRACT.action.aimResidualMaxRadians;
	return min + (clamped / (NUM_AIM_BINS - 1)) * (max - min);
}

export function nearestEnemyAimBase(obs: TankObservation): number {
	const sx = obs.self.x + obs.self.size / 2;
	const sy = obs.self.y + obs.self.size / 2;
	let nearest: TankObservation['enemies'][number] | null = null;
	let nearestDistSq = Number.POSITIVE_INFINITY;
	for (const enemy of obs.enemies) {
		if (enemy.destroyed) continue;
		const dx = enemy.x + enemy.size / 2 - sx;
		const dy = enemy.y + enemy.size / 2 - sy;
		const distSq = dx * dx + dy * dy;
		if (distSq < nearestDistSq) {
			nearest = enemy;
			nearestDistSq = distSq;
		}
	}
	if (!nearest) return obs.self.aimAngle;
	return Math.atan2(nearest.y + nearest.size / 2 - sy, nearest.x + nearest.size / 2 - sx);
}

export function decodePolicyAction(indices: readonly number[], obs: TankObservation): TankAction {
	if (indices.length !== ACTION_DIM) {
		throw new Error(`Expected ${ACTION_DIM} action heads, got ${indices.length}`);
	}
	const moveIdx = Math.max(0, Math.min(MOVE_INTENTS_BY_INDEX.length - 1, Math.trunc(indices[0])));
	const aimBin = Math.max(0, Math.min(NUM_AIM_BINS - 1, Math.trunc(indices[1])));
	return {
		move: MOVE_INTENTS_BY_INDEX[moveIdx],
		aimAngle: nearestEnemyAimBase(obs) + aimResidualForBin(aimBin),
		fire: Math.trunc(indices[2]) !== 0,
		plantBomb: Math.trunc(indices[3]) !== 0 && obs.self.maxBombs > 0,
	};
}

export function assertCompatibleContract(value: unknown): void {
	const candidate = value as Partial<NeuralModelContractData> | null;
	if (
		!candidate ||
		candidate.contractVersion !== NEURAL_MODEL_CONTRACT.contractVersion ||
		candidate.observationVersion !== NEURAL_MODEL_CONTRACT.observationVersion ||
		candidate.actionVersion !== NEURAL_MODEL_CONTRACT.actionVersion ||
		JSON.stringify(candidate.observation) !== JSON.stringify(NEURAL_MODEL_CONTRACT.observation) ||
		JSON.stringify(candidate.action) !== JSON.stringify(NEURAL_MODEL_CONTRACT.action)
	) {
		throw new Error(
			`Incompatible neural model contract. Expected v${NEURAL_MODEL_CONTRACT.contractVersion} ` +
			`${NEURAL_MODEL_CONTRACT.observationVersion}/${NEURAL_MODEL_CONTRACT.actionVersion}.`
		);
	}
}
