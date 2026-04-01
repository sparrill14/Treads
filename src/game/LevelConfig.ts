export type AmmoType = 'basic' | 'super';
export type BombType = 'basic' | 'love';
export type NavigatorType = 'simple' | 'astar' | 'astar-avoidance';
export type EnemyType = 'stationary' | 'stationary-random-aim' | 'simple-moving' | 'bomber' | 'super-bomber';

export interface ObstacleConfig {
	x: number;
	y: number;
	width: number;
	height: number;
}

export interface NavigatorConfig {
	type: NavigatorType;
	aggressionFactor?: number;
	heuristicProfile?: 'default' | 'elite' | 'advanced';
	tacticalRole?: 'auto' | 'pressure' | 'flank' | 'zone';
}

export interface EnemyConfig {
	type: EnemyType;
	x: number;
	y: number;
	ammo?: { type: AmmoType; count: number };
	bombs?: { type: BombType; count: number };
	navigator?: NavigatorConfig;
}

export interface PlayerConfig {
	x: number;
	y: number;
}

export interface MatchRulesConfig {
	tankHitPoints?: number;
	projectileDamage?: number;
	bombDamage?: number;
	invulnerabilityTicks?: number;
	projectileBounces?: boolean;
	turretSpeedMultiplier?: number;
}

export interface LevelConfig {
	obstacles: ObstacleConfig[];
	enemies: EnemyConfig[];
	player: PlayerConfig;
	rules?: MatchRulesConfig;
}

export const LEVEL_CONFIGS: LevelConfig[] = [
	// Level 1
	{
		obstacles: [{ x: 300, y: 200, width: 40, height: 100 }],
		enemies: [{ type: 'stationary', x: 900, y: 240 }],
		player: { x: 100, y: 250 },
	},
	// Level 2
	{
		obstacles: [{ x: 300, y: 200, width: 40, height: 100 }],
		enemies: [
			{ type: 'stationary', x: 800, y: 100 },
			{ type: 'stationary', x: 900, y: 240 },
			{ type: 'stationary', x: 800, y: 400 },
		],
		player: { x: 100, y: 250 },
	},
	// Level 3
	{
		obstacles: [{ x: 700, y: 150, width: 30, height: 250 }],
		enemies: [{ type: 'stationary-random-aim', x: 900, y: 250, ammo: { type: 'super', count: 1 } }],
		player: { x: 200, y: 250 },
	},
	// Level 4
	{
		obstacles: [
			{ x: 300, y: 130, width: 500, height: 35 },
			{ x: 200, y: 330, width: 500, height: 35 },
		],
		enemies: [
			{
				type: 'simple-moving',
				x: 900,
				y: 50,
				ammo: { type: 'basic', count: 1 },
				navigator: { type: 'simple' },
			},
		],
		player: { x: 50, y: 450 },
	},
	// Level 5
	{
		obstacles: [
			{ x: 100, y: 100, width: 200, height: 100 },
			{ x: 700, y: 100, width: 30, height: 100 },
			{ x: 700, y: 350, width: 30, height: 100 },
		],
		enemies: [
			{
				type: 'simple-moving',
				x: 800,
				y: 300,
				ammo: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
			{ type: 'stationary-random-aim', x: 800, y: 100, ammo: { type: 'super', count: 1 } },
			{ type: 'stationary', x: 800, y: 200 },
		],
		player: { x: 200, y: 250 },
	},
	// Level 6
	{
		obstacles: [
			{ x: 100, y: 100, width: 200, height: 100 },
			{ x: 700, y: 100, width: 30, height: 300 },
		],
		enemies: [
			{
				type: 'bomber',
				x: 800,
				y: 100,
				ammo: { type: 'basic', count: 1 },
				bombs: { type: 'basic', count: 3 },
				navigator: { type: 'astar' },
			},
			{
				type: 'bomber',
				x: 800,
				y: 200,
				ammo: { type: 'basic', count: 1 },
				bombs: { type: 'basic', count: 3 },
				navigator: { type: 'astar' },
			},
			{
				type: 'bomber',
				x: 800,
				y: 300,
				ammo: { type: 'basic', count: 1 },
				bombs: { type: 'basic', count: 3 },
				navigator: { type: 'astar' },
			},
		],
		player: { x: 200, y: 250 },
	},
	// Level 7
	{
		obstacles: [
			{ x: 0, y: 120, width: 400, height: 50 },
			{ x: 600, y: 120, width: 400, height: 50 },
			{ x: 0, y: 330, width: 400, height: 50 },
			{ x: 600, y: 330, width: 400, height: 50 },
		],
		enemies: [
			{ type: 'stationary-random-aim', x: 900, y: 50, ammo: { type: 'super', count: 1 } },
			{ type: 'stationary-random-aim', x: 900, y: 250, ammo: { type: 'super', count: 1 } },
			{ type: 'stationary-random-aim', x: 900, y: 430, ammo: { type: 'super', count: 1 } },
			{ type: 'stationary-random-aim', x: 100, y: 50, ammo: { type: 'super', count: 1 } },
			{ type: 'stationary-random-aim', x: 100, y: 250, ammo: { type: 'super', count: 1 } },
		],
		player: { x: 100, y: 430 },
	},
	// Level 8
	{
		obstacles: [{ x: 700, y: 100, width: 30, height: 300 }],
		enemies: [
			{
				type: 'super-bomber',
				x: 800,
				y: 100,
				ammo: { type: 'super', count: 3 },
				bombs: { type: 'love', count: 2 },
				navigator: { type: 'astar', aggressionFactor: 5 },
			},
			{
				type: 'super-bomber',
				x: 800,
				y: 200,
				ammo: { type: 'super', count: 3 },
				bombs: { type: 'love', count: 2 },
				navigator: { type: 'astar', aggressionFactor: 10 },
			},
			{
				type: 'super-bomber',
				x: 800,
				y: 300,
				ammo: { type: 'super', count: 3 },
				bombs: { type: 'love', count: 2 },
				navigator: { type: 'astar', aggressionFactor: 15 },
			},
		],
		player: { x: 200, y: 250 },
	},
	// Level 9
	{
		obstacles: [
			{ x: 350, y: 0, width: 30, height: 200 },
			{ x: 350, y: 300, width: 30, height: 200 },
			{ x: 700, y: 100, width: 30, height: 300 },
		],
		enemies: [
			{
				type: 'super-bomber',
				x: 800,
				y: 100,
				ammo: { type: 'super', count: 3 },
				bombs: { type: 'basic', count: 2 },
				navigator: { type: 'astar-avoidance', aggressionFactor: 5 },
			},
			{
				type: 'super-bomber',
				x: 800,
				y: 350,
				ammo: { type: 'super', count: 3 },
				bombs: { type: 'basic', count: 2 },
				navigator: { type: 'astar-avoidance', aggressionFactor: 10 },
			},
		],
		player: { x: 100, y: 250 },
	},
	// Level 10
	{
		obstacles: [
			{ x: 220, y: 90, width: 30, height: 320 },
			{ x: 480, y: 0, width: 30, height: 200 },
			{ x: 480, y: 300, width: 30, height: 200 },
			{ x: 740, y: 90, width: 30, height: 320 },
		],
		enemies: [
			{
				type: 'super-bomber',
				x: 840,
				y: 90,
				ammo: { type: 'super', count: 4 },
				bombs: { type: 'love', count: 2 },
				navigator: {
					type: 'astar-avoidance',
					aggressionFactor: 7,
					heuristicProfile: 'advanced',
					tacticalRole: 'pressure',
				},
			},
			{
				type: 'super-bomber',
				x: 880,
				y: 240,
				ammo: { type: 'super', count: 4 },
				bombs: { type: 'love', count: 2 },
				navigator: {
					type: 'astar-avoidance',
					aggressionFactor: 11,
					heuristicProfile: 'advanced',
					tacticalRole: 'flank',
				},
			},
			{
				type: 'super-bomber',
				x: 840,
				y: 390,
				ammo: { type: 'super', count: 4 },
				bombs: { type: 'love', count: 2 },
				navigator: {
					type: 'astar-avoidance',
					aggressionFactor: 15,
					heuristicProfile: 'advanced',
					tacticalRole: 'zone',
				},
			},
		],
		player: { x: 90, y: 250 },
	},
];
