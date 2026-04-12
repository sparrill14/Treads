import { LEVEL_CONFIGS, type LevelConfig } from '../src/game/LevelConfig';

export const TRAINING_SCENARIOS: Record<number, LevelConfig> = {
	111: {
		player: { x: 220, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 520, y: 250, ammo: { type: 'basic', count: 0 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2.5 },
	},
	112: {
		player: { x: 220, y: 180 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 520, y: 320, ammo: { type: 'basic', count: 0 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2.5 },
	},
	113: {
		player: { x: 220, y: 250 },
		obstacles: [],
		enemies: [
			{ type: 'simple-moving', x: 620, y: 250, ammo: { type: 'basic', count: 0 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false, turretSpeedMultiplier: 3 },
	},
	121: {
		player: { x: 220, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 560, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 1.5 },
	},
	122: {
		player: { x: 220, y: 180 },
		obstacles: [],
		enemies: [{ type: 'stationary-random-aim', x: 560, y: 320, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 1.5 },
	},
	123: {
		player: { x: 220, y: 250 },
		obstacles: [],
		enemies: [
			{ type: 'simple-moving', x: 620, y: 250, ammo: { type: 'basic', count: 0 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false, turretSpeedMultiplier: 1.5 },
	},
	101: {
		player: { x: 120, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 840, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2 },
	},
	102: {
		player: { x: 120, y: 180 },
		obstacles: [],
		enemies: [{ type: 'stationary-random-aim', x: 820, y: 320, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2 },
	},
	103: {
		player: { x: 120, y: 320 },
		obstacles: [],
		enemies: [
			{ type: 'simple-moving', x: 860, y: 180, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2 },
	},
	201: {
		player: { x: 100, y: 250 },
		obstacles: [{ x: 430, y: 160, width: 40, height: 180 }],
		enemies: [{ type: 'stationary', x: 860, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	},
	202: {
		player: { x: 120, y: 250 },
		obstacles: [{ x: 360, y: 120, width: 30, height: 260 }],
		enemies: [
			{ type: 'stationary', x: 780, y: 140, ammo: { type: 'basic', count: 1 } },
			{ type: 'stationary', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
		rules: { projectileBounces: false },
	},
	203: {
		player: { x: 100, y: 400 },
		obstacles: [
			{ x: 280, y: 110, width: 220, height: 30 },
			{ x: 420, y: 260, width: 220, height: 30 },
		],
		enemies: [
			{ type: 'simple-moving', x: 850, y: 100, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	301: {
		player: { x: 140, y: 250 },
		obstacles: [{ x: 520, y: 120, width: 35, height: 260 }],
		enemies: [
			{
				type: 'bomber',
				x: 780,
				y: 250,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
		],
	},
	302: {
		player: { x: 130, y: 210 },
		obstacles: [
			{ x: 300, y: 120, width: 30, height: 260 },
			{ x: 640, y: 150, width: 30, height: 200 },
		],
		enemies: [
			{
				type: 'bomber',
				x: 820,
				y: 150,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
			{ type: 'stationary-random-aim', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
	},
	303: {
		player: { x: 150, y: 260 },
		obstacles: [
			{ x: 340, y: 110, width: 280, height: 30 },
			{ x: 340, y: 360, width: 280, height: 30 },
		],
		enemies: [
			{
				type: 'super-bomber',
				x: 800,
				y: 140,
				ammo: { type: 'super', count: 0 },
				bombs: { type: 'love', count: 1 },
				navigator: { type: 'astar', aggressionFactor: 4 },
			},
			{ type: 'stationary-random-aim', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
	},
	// ---- Phase 1 supplement: armed target practice (turret 2.5x) ----
	114: {
		player: { x: 220, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 620, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false, turretSpeedMultiplier: 2.5 },
	},
	// ---- Phase 3: full turret, no obstacles ----
	131: {
		player: { x: 200, y: 250 },
		obstacles: [],
		enemies: [{ type: 'stationary', x: 700, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	},
	132: {
		player: { x: 200, y: 180 },
		obstacles: [],
		enemies: [{ type: 'stationary-random-aim', x: 700, y: 320, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	},
	133: {
		player: { x: 200, y: 250 },
		obstacles: [],
		enemies: [
			{ type: 'simple-moving', x: 700, y: 250, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	// ---- Phase 4: obstacles intro, single enemy ----
	141: {
		player: { x: 150, y: 250 },
		obstacles: [{ x: 450, y: 160, width: 40, height: 180 }],
		enemies: [{ type: 'stationary', x: 800, y: 250, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	},
	142: {
		player: { x: 150, y: 350 },
		obstacles: [{ x: 400, y: 120, width: 30, height: 200 }],
		enemies: [{ type: 'stationary-random-aim', x: 800, y: 150, ammo: { type: 'basic', count: 1 } }],
		rules: { projectileBounces: false },
	},
	143: {
		player: { x: 150, y: 250 },
		obstacles: [{ x: 380, y: 200, width: 30, height: 150 }],
		enemies: [
			{ type: 'simple-moving', x: 800, y: 250, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	// ---- Phase 4→5 bridge: gentle multi-enemy introduction ----
	// 145: Open arena, 2 stationary enemies, one UNARMED (pure multi-target aiming)
	145: {
		player: { x: 200, y: 250 },
		obstacles: [],
		enemies: [
			{ type: 'stationary', x: 700, y: 150, ammo: { type: 'basic', count: 1 } },
			{ type: 'stationary', x: 700, y: 350, ammo: { type: 'basic', count: 0 } },
		],
		rules: { projectileBounces: false },
	},
	// 146: Open arena, 2 stationary enemies, both armed (multi-target, no obstacles)
	146: {
		player: { x: 200, y: 250 },
		obstacles: [],
		enemies: [
			{ type: 'stationary', x: 720, y: 160, ammo: { type: 'basic', count: 1 } },
			{ type: 'stationary', x: 720, y: 340, ammo: { type: 'basic', count: 1 } },
		],
		rules: { projectileBounces: false },
	},
	// 147: 1 obstacle, 1 stationary armed + 1 unarmed mobile (track two, only one shoots)
	147: {
		player: { x: 150, y: 250 },
		obstacles: [{ x: 400, y: 180, width: 30, height: 150 }],
		enemies: [
			{ type: 'stationary', x: 780, y: 150, ammo: { type: 'basic', count: 1 } },
			{ type: 'simple-moving', x: 780, y: 350, ammo: { type: 'basic', count: 0 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	// 148: Open arena, 1 random-aim armed + 1 mobile armed (full combat, no obstacles)
	148: {
		player: { x: 180, y: 250 },
		obstacles: [],
		enemies: [
			{ type: 'stationary-random-aim', x: 740, y: 160, ammo: { type: 'basic', count: 1 } },
			{ type: 'simple-moving', x: 740, y: 340, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	// ---- Phase 4.5b→4.5c bridge: multi-enemy + light obstacles ----
	// 149: 2 stationary armed + 1 wall (like 146 but with an obstacle)
	149: {
		player: { x: 150, y: 250 },
		obstacles: [{ x: 420, y: 170, width: 30, height: 160 }],
		enemies: [
			{ type: 'stationary', x: 750, y: 150, ammo: { type: 'basic', count: 1 } },
			{ type: 'stationary', x: 750, y: 350, ammo: { type: 'basic', count: 1 } },
		],
		rules: { projectileBounces: false },
	},
	// 150: 1 stationary armed + 1 mobile armed + 1 wall (like 148 but with an obstacle)
	150: {
		player: { x: 140, y: 250 },
		obstacles: [{ x: 400, y: 150, width: 30, height: 200 }],
		enemies: [
			{ type: 'stationary', x: 770, y: 160, ammo: { type: 'basic', count: 1 } },
			{ type: 'simple-moving', x: 770, y: 350, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	// ---- Phase 5: multi-enemy + obstacles ----
	151: {
		player: { x: 120, y: 250 },
		obstacles: [{ x: 430, y: 160, width: 40, height: 180 }],
		enemies: [
			{ type: 'stationary', x: 780, y: 140, ammo: { type: 'basic', count: 1 } },
			{ type: 'stationary', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
		rules: { projectileBounces: false },
	},
	152: {
		player: { x: 120, y: 250 },
		obstacles: [
			{ x: 360, y: 120, width: 30, height: 260 },
			{ x: 640, y: 200, width: 30, height: 150 },
		],
		enemies: [
			{ type: 'stationary', x: 800, y: 150, ammo: { type: 'basic', count: 1 } },
			{ type: 'simple-moving', x: 800, y: 350, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	153: {
		player: { x: 100, y: 400 },
		obstacles: [{ x: 350, y: 200, width: 30, height: 200 }],
		enemies: [
			{ type: 'stationary-random-aim', x: 800, y: 120, ammo: { type: 'basic', count: 1 } },
			{ type: 'simple-moving', x: 850, y: 350, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
		rules: { projectileBounces: false },
	},
	// ---- Phase 6: bouncing shots enabled ----
	161: {
		player: { x: 150, y: 250 },
		obstacles: [{ x: 450, y: 120, width: 35, height: 260 }],
		enemies: [{ type: 'stationary', x: 800, y: 250, ammo: { type: 'basic', count: 1 } }],
	},
	162: {
		player: { x: 130, y: 210 },
		obstacles: [{ x: 500, y: 150, width: 30, height: 200 }],
		enemies: [
			{ type: 'stationary', x: 780, y: 140, ammo: { type: 'basic', count: 1 } },
			{ type: 'stationary', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
	},
	163: {
		player: { x: 150, y: 300 },
		obstacles: [
			{ x: 350, y: 110, width: 200, height: 30 },
			{ x: 350, y: 360, width: 200, height: 30 },
		],
		enemies: [
			{ type: 'simple-moving', x: 800, y: 200, ammo: { type: 'basic', count: 1 }, navigator: { type: 'simple' } },
		],
	},
	// ---- Phase 7: bomber introduction ----
	171: {
		player: { x: 140, y: 250 },
		obstacles: [{ x: 520, y: 120, width: 35, height: 260 }],
		enemies: [
			{
				type: 'bomber',
				x: 780,
				y: 250,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
		],
	},
	172: {
		player: { x: 140, y: 250 },
		obstacles: [{ x: 500, y: 150, width: 30, height: 200 }],
		enemies: [
			{
				type: 'bomber',
				x: 780,
				y: 350,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
			{ type: 'stationary', x: 820, y: 120, ammo: { type: 'basic', count: 1 } },
		],
	},
	173: {
		player: { x: 150, y: 260 },
		obstacles: [
			{ x: 320, y: 110, width: 250, height: 30 },
			{ x: 320, y: 360, width: 250, height: 30 },
		],
		enemies: [
			{
				type: 'bomber',
				x: 800,
				y: 200,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
		],
	},
	// ---- Phase 8: full mix (bombers + shooters + super-bombers) ----
	181: {
		player: { x: 140, y: 250 },
		obstacles: [{ x: 500, y: 150, width: 30, height: 250 }],
		enemies: [
			{
				type: 'bomber',
				x: 800,
				y: 150,
				ammo: { type: 'basic', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
			{ type: 'stationary-random-aim', x: 860, y: 350, ammo: { type: 'basic', count: 1 } },
		],
	},
	182: {
		player: { x: 130, y: 210 },
		obstacles: [
			{ x: 300, y: 120, width: 30, height: 260 },
			{ x: 640, y: 150, width: 30, height: 200 },
		],
		enemies: [
			{
				type: 'super-bomber',
				x: 820,
				y: 150,
				ammo: { type: 'super', count: 0 },
				bombs: { type: 'basic', count: 1 },
				navigator: { type: 'astar' },
			},
			{ type: 'stationary', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
	},
	183: {
		player: { x: 150, y: 260 },
		obstacles: [
			{ x: 340, y: 110, width: 280, height: 30 },
			{ x: 340, y: 360, width: 280, height: 30 },
		],
		enemies: [
			{
				type: 'super-bomber',
				x: 800,
				y: 140,
				ammo: { type: 'super', count: 0 },
				bombs: { type: 'love', count: 1 },
				navigator: { type: 'astar', aggressionFactor: 4 },
			},
			{ type: 'stationary-random-aim', x: 860, y: 340, ammo: { type: 'basic', count: 1 } },
		],
	},
};

export function resolveScenarioConfig(scenarioId: number): LevelConfig {
	if (scenarioId in TRAINING_SCENARIOS) {
		return TRAINING_SCENARIOS[scenarioId];
	}
	const builtIn = LEVEL_CONFIGS[scenarioId - 1];
	if (!builtIn) {
		throw new Error(`Unknown scenario id: ${scenarioId}`);
	}
	return builtIn;
}
