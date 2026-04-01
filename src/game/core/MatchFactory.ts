import { PassiveTankController } from '../controllers/ReplayController';
import { ScriptedEnemyController } from '../controllers/ScriptedEnemyController';
import {
	getLevelTankConfigs,
	type LevelConfig,
	type NavigatorType,
	type TankConfig,
	type TankKind,
} from '../LevelConfig';
import { getTankSpec } from './specs';
import type { GameState, MatchBootstrap, MatchRules, TankController, TankStateView } from './types';

interface MatchFactoryOptions {
	playerController?: TankController;
	rulesOverrides?: Partial<MatchRules>;
}

const DEFAULT_MATCH_RULES: MatchRules = {
	tankHitPoints: 3,
	projectileDamage: 1,
	bombDamage: 2,
	invulnerabilityTicks: 8,
	projectileBounces: true,
	turretSpeedMultiplier: 1,
};

function createMatchRules(levelConfig: LevelConfig, options: MatchFactoryOptions): MatchRules {
	const merged: MatchRules = {
		...DEFAULT_MATCH_RULES,
		...(levelConfig.rules ?? {}),
		...(options.rulesOverrides ?? {}),
	};
	return {
		tankHitPoints: Math.max(1, Math.round(merged.tankHitPoints)),
		projectileDamage: Math.max(1, Math.round(merged.projectileDamage)),
		bombDamage: Math.max(1, Math.round(merged.bombDamage)),
		invulnerabilityTicks: Math.max(0, Math.round(merged.invulnerabilityTicks)),
		projectileBounces: Boolean(merged.projectileBounces),
		turretSpeedMultiplier: Math.max(0.1, merged.turretSpeedMultiplier),
	};
}

function getNavigationMode(kind: TankKind, navigatorType: NavigatorType | undefined): 'stationary' | NavigatorType {
	if (kind === 'stationary' || kind === 'stationary-random-aim') {
		return 'stationary';
	}
	if (kind === 'player') {
		return navigatorType ?? 'simple';
	}
	return navigatorType ?? 'astar';
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

function createTankState(id: string, config: TankConfig, rules: MatchRules): TankStateView {
	const spec = getTankSpec(config.kind);
	const hitPoints = rules.tankHitPoints;
	const defaultAmmo = config.kind === 'super-bomber' ? 'super' : 'basic';
	const defaultBombCount =
		config.kind === 'bomber' || config.kind === 'super-bomber' ? 2 : config.kind === 'player' ? 2 : 0;
	const defaultBombType = config.kind === 'super-bomber' ? 'love' : 'basic';
	const bombCount = config.bombs?.count ?? defaultBombCount;
	return {
		id,
		controllerId: id,
		team: config.team,
		kind: config.kind,
		x: config.x,
		y: config.y,
		size: spec.size,
		speed: spec.speed,
		color: config.color ?? spec.color,
		aimAngle: 0,
		aimTargetX: null,
		aimTargetY: null,
		health: hitPoints,
		maxHealth: hitPoints,
		invulnerabilityTicksRemaining: 0,
		destroyed: false,
		ammoType: config.ammo?.type ?? defaultAmmo,
		maxAmmo: config.ammo?.count ?? (config.kind === 'player' ? 5 : 1),
		activeAmmo: 0,
		bombType: bombCount > 0 ? (config.bombs?.type ?? defaultBombType) : null,
		maxBombs: bombCount,
		activeBombs: 0,
		shotCooldownTicks: spec.initialShotCooldownTicks,
		shotCooldownTicksOnFire: spec.shotCooldownTicksOnFire,
		bombCooldownTicks: spec.initialBombCooldownTicks,
		bombCooldownTicksOnPlant: spec.bombCooldownTicksOnPlant,
		wasLastMoveBlocked: false,
		lastMoveIntent: 'none',
		consecutiveDirectionMoves: 0,
		aggressionFactor: config.navigator?.aggressionFactor ?? spec.aggressionFactor,
	};
}

export function createInitialGameState(
	levelConfig: LevelConfig,
	seed: number,
	options: MatchFactoryOptions = {}
): GameState {
	const rules = createMatchRules(levelConfig, options);
	const tankConfigs = getLevelTankConfigs(levelConfig);
	const tanks = tankConfigs.map((tankConfig, index) =>
		createTankState(tankConfig.id ?? `tank-${index}`, tankConfig, rules)
	);
	const playerTank = tankConfigs.find((tank) => tank.team === 'player' || tank.control === 'human') ?? null;
	return {
		seed,
		rngState: seed >>> 0,
		tick: 0,
		tickRate: 60,
		status: 'running',
		rules,
		arena: { width: 1000, height: 500 },
		playerTankId: playerTank?.id ?? null,
		winnerTeam: null,
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

export function createDefaultControllers(
	levelConfig: LevelConfig,
	options: MatchFactoryOptions = {}
): Record<string, TankController> {
	const controllers: Record<string, TankController> = {};
	let humanAssigned = false;
	for (const [index, tank] of getLevelTankConfigs(levelConfig).entries()) {
		const tankId = tank.id ?? `tank-${index}`;
		const control = tank.control ?? (tank.kind === 'player' ? 'human' : 'scripted');
		if (control === 'human' && !humanAssigned) {
			controllers[tankId] = options.playerController ?? new PassiveTankController();
			humanAssigned = true;
			continue;
		}

		const navigationMode = getNavigationMode(tank.kind, tank.navigator?.type);
		controllers[tankId] = new ScriptedEnemyController({
			navigationMode,
			randomAim: tank.kind === 'stationary-random-aim',
			recalculationInterval: getRecalculationInterval(navigationMode),
			aggressionFactor: tank.navigator?.aggressionFactor ?? getTankSpec(tank.kind).aggressionFactor,
			heuristicProfile: tank.navigator?.heuristicProfile,
			tacticalRole: tank.navigator?.tacticalRole,
		});
	}
	return controllers;
}

export function createMatchBootstrap(
	levelConfig: LevelConfig,
	seed: number,
	options: MatchFactoryOptions = {}
): MatchBootstrap {
	return {
		initialState: createInitialGameState(levelConfig, seed, options),
		controllers: createDefaultControllers(levelConfig, options),
	};
}
