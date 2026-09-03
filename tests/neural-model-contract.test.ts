import assert from 'node:assert/strict';
import type { LevelConfig } from '../src/game/LevelConfig';
import {
	ACTION_HEAD_SIZES,
	assertCompatibleContract,
	decodePolicyAction,
	NEURAL_MODEL_CONTRACT,
	OBS_SIZE,
	unitAngleFeature,
} from '../src/game/controllers/NeuralModelContract';
import { createInitialGameState } from '../src/game/core/MatchFactory';
import type { TankObservation } from '../src/game/core/types';
import { NavigationPlanner } from '../src/game/navigation/NavigationPlanner';
import { normalizePolicyObservation } from '../src/game/controllers/NeuralObservation';
import { potentialBasedApproachReward } from '../training/reward-shaping';

export function runNeuralModelContractTests(): void {
	const dimensions = NEURAL_MODEL_CONTRACT.observation;
	assert.equal(
		OBS_SIZE,
		dimensions.selfDim +
			dimensions.maxEnemies * dimensions.enemyDim +
			dimensions.maxProjectiles * dimensions.projectileDim +
			dimensions.maxObstacles * dimensions.obstacleDim +
			dimensions.maxBombs * dimensions.bombDim +
			dimensions.summaryDim
	);
	assert.deepEqual(ACTION_HEAD_SIZES, [9, 65, 2, 2]);
	assert.throws(
		() =>
			assertCompatibleContract({
				...NEURAL_MODEL_CONTRACT,
				observation: { ...NEURAL_MODEL_CONTRACT.observation, size: OBS_SIZE - 1 },
			}),
		/Incompatible neural model contract/
	);

	const upper = unitAngleFeature(Math.PI / 2);
	const lower = unitAngleFeature((3 * Math.PI) / 2);
	assert.notDeepEqual(upper, lower, 'Opposite half-plane angles must not alias');

	const config: LevelConfig = {
		player: { x: 200, y: 180, bombs: { type: 'basic', count: 0 } },
		enemies: [{ type: 'stationary', x: 700, y: 320 }],
		obstacles: [],
	};
	const state = createInitialGameState(config, 123);
	const player = state.tanks.find((tank) => tank.id === 'player-0');
	assert.ok(player);
	const observation: TankObservation = {
		tick: state.tick,
		self: player,
		enemies: state.tanks.filter((tank) => tank.team === 'enemy'),
		projectiles: [],
		bombs: [],
		obstacles: [],
		arena: state.arena,
	};
	const neutral = NEURAL_MODEL_CONTRACT.action.neutralAimBin;
	const action = decodePolicyAction([0, neutral, 1, 1], observation);
	const expected = Math.atan2(140, 500);
	assert.ok(Math.abs(action.aimAngle - expected) < 1e-12, 'Neutral residual must aim exactly at nearest enemy');
	assert.equal(action.plantBomb, false, 'Bomb action must be masked when the player has no bombs');

	const obstacleConfig: LevelConfig = {
		player: { x: 405, y: 250 },
		enemies: [{ type: 'stationary', x: 800, y: 250 }],
		obstacles: [{ x: 450, y: 160, width: 40, height: 180 }],
	};
	const obstacleState = createInitialGameState(obstacleConfig, 456);
	const obstaclePlayer = obstacleState.tanks.find((tank) => tank.id === 'player-0');
	assert.ok(obstaclePlayer);
	const obstacleObservation: TankObservation = {
		tick: obstacleState.tick,
		self: obstaclePlayer,
		enemies: obstacleState.tanks.filter((tank) => tank.team === 'enemy'),
		projectiles: [],
		bombs: [],
		obstacles: obstacleState.obstacles,
		arena: obstacleState.arena,
	};
	const planner = new NavigationPlanner(obstacleState.arena, obstacleState.obstacles, obstaclePlayer.size);
	const encoded = normalizePolicyObservation(obstacleObservation, 720, planner);
	assert.equal(encoded.length, OBS_SIZE);
	assert.equal(encoded[5], 0, 'The obstacle must block direct line of sight');
	assert.equal(encoded[21], 1, 'The enemy must have a reachable A* route');
	assert.ok(encoded[20] > encoded[15], 'Path distance must exceed direct distance around the obstacle');
	assert.ok(Math.abs(encoded[18] - 0.5) > 0.1, 'Navigation bearing must route vertically around the obstacle');

	const waitingReward = potentialBasedApproachReward(600, 600, 1200, 0.997, 2.5);
	const progressReward = potentialBasedApproachReward(600, 595, 1200, 0.997, 2.5);
	assert.ok(waitingReward < 0, 'Potential shaping must penalize standing still');
	assert.ok(progressReward > waitingReward, 'Moving along the path must improve shaping reward');
}
