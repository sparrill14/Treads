import { PassiveTankController } from '../controllers/ReplayController';
import { ScriptedEnemyController } from '../controllers/ScriptedEnemyController';
import { buildBounds } from './geometry';
import { getTankSpec } from './specs';
import type { EnemyConfig, LevelConfig, NavigatorType } from '../LevelConfig';
import type { GameState, MatchBootstrap, TankController, TankStateView } from './types';

interface MatchFactoryOptions {
	playerController?: TankController;
}

function getNavigationMode(enemy: EnemyConfig): 'stationary' | NavigatorType {
	if (enemy.type === 'stationary' || enemy.type === 'stationary-random-aim') {
		return 'stationary';
	}
	return enemy.navigator?.type ?? 'astar';
}

function getRecalculationInterval(navigationMode: 'stationary' | NavigatorType): number {
	switch (navigationMode) {
		case 'simple':
			return 120;
		case 'astar-avoidance':
			return 20;
		case 'astar':
			return 60;
		default:
			return 0;
	}
}

function createPlayerTankState(id: string, config: LevelConfig['player']): TankStateView {
	const spec = getTankSpec('player');
	return {
		id,
		controllerId: id,
		team: 'player',
		kind: 'player',
		x: config.x,
		y: config.y,
		size: spec.size,
		speed: spec.speed,
		color: spec.color,
		aimAngle: 0,
		aimTargetX: null,
		aimTargetY: null,
		destroyed: false,
		bounds: buildBounds(config.x, config.y, spec.size),
		ammoType: 'basic',
		maxAmmo: 5,
		activeAmmo: 0,
		bombType: 'basic',
		maxBombs: 2,
		activeBombs: 0,
		shotCooldownTicks: spec.initialShotCooldownTicks,
		shotCooldownTicksOnFire: spec.shotCooldownTicksOnFire,
		bombCooldownTicks: spec.initialBombCooldownTicks,
		bombCooldownTicksOnPlant: spec.bombCooldownTicksOnPlant,
		wasLastMoveBlocked: false,
		lastMoveIntent: 'none',
		consecutiveDirectionMoves: 0,
		aggressionFactor: spec.aggressionFactor,
	};
}

function createEnemyTankState(id: string, config: EnemyConfig): TankStateView {
	const spec = getTankSpec(config.type);
	const aggressionFactor = config.navigator?.aggressionFactor ?? spec.aggressionFactor;
	const bombCount = config.bombs?.count ?? 0;
	return {
		id,
		controllerId: id,
		team: 'enemy',
		kind: config.type,
		x: config.x,
		y: config.y,
		size: spec.size,
		speed: spec.speed,
		color: spec.color,
		aimAngle: 0,
		aimTargetX: null,
		aimTargetY: null,
		destroyed: false,
		bounds: buildBounds(config.x, config.y, spec.size),
		ammoType: config.ammo?.type ?? (config.type === 'super-bomber' ? 'super' : 'basic'),
		maxAmmo: config.ammo?.count ?? 1,
		activeAmmo: 0,
		bombType: bombCount > 0 ? (config.bombs?.type ?? 'basic') : null,
		maxBombs: bombCount,
		activeBombs: 0,
		shotCooldownTicks: spec.initialShotCooldownTicks,
		shotCooldownTicksOnFire: spec.shotCooldownTicksOnFire,
		bombCooldownTicks: spec.initialBombCooldownTicks,
		bombCooldownTicksOnPlant: spec.bombCooldownTicksOnPlant,
		wasLastMoveBlocked: false,
		lastMoveIntent: 'none',
		consecutiveDirectionMoves: 0,
		aggressionFactor: aggressionFactor,
	};
}

export function createInitialGameState(levelConfig: LevelConfig, seed: number): GameState {
	const playerTankId = 'player-0';
	const tanks: TankStateView[] = [createPlayerTankState(playerTankId, levelConfig.player)];
	levelConfig.enemies.forEach((enemy, index) => {
		tanks.push(createEnemyTankState(`enemy-${index}`, enemy));
	});
	return {
		seed,
		rngState: seed >>> 0,
		tick: 0,
		tickRate: 60,
		status: 'running',
		arena: { width: 1000, height: 500 },
		playerTankId,
		nextEntityId: 1,
		obstacles: levelConfig.obstacles.map((obstacle, index) => ({
			id: `obstacle-${index}`,
			x: obstacle.x,
			y: obstacle.y,
			width: obstacle.width,
			height: obstacle.height,
		})),
		tanks,
		projectiles: [],
		bombs: [],
	};
}

export function createDefaultControllers(levelConfig: LevelConfig, options: MatchFactoryOptions = {}): Record<string, TankController> {
	const controllers: Record<string, TankController> = {
		'player-0': options.playerController ?? new PassiveTankController(),
	};
	levelConfig.enemies.forEach((enemy, index) => {
		const navigationMode = getNavigationMode(enemy);
		controllers[`enemy-${index}`] = new ScriptedEnemyController({
			navigationMode,
			randomAim: enemy.type === 'stationary-random-aim',
			recalculationInterval: getRecalculationInterval(navigationMode),
			aggressionFactor: enemy.navigator?.aggressionFactor ?? getTankSpec(enemy.type).aggressionFactor,
		});
	});
	return controllers;
}

export function createMatchBootstrap(
	levelConfig: LevelConfig,
	seed: number,
	options: MatchFactoryOptions = {}
): MatchBootstrap {
	return {
		initialState: createInitialGameState(levelConfig, seed),
		controllers: createDefaultControllers(levelConfig, options),
	};
}
