import type { AmmoType, BombType, EnemyType, LevelConfig } from '../LevelConfig';

export const SIMULATION_TICK_RATE = 60;
export const SIMULATION_TICK_SECONDS = 1 / SIMULATION_TICK_RATE;

export const MOVE_INTENTS = ['none', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'] as const;
export type MoveIntent = (typeof MOVE_INTENTS)[number];
export type Team = 'player' | 'enemy';
export type TankKind = 'player' | EnemyType;
export type MatchStatus = 'running' | 'player_win' | 'enemy_win';
export type AudioCue = 'tank-fire' | 'tank-destroy' | 'bomb-explode' | 'ammunition-explode';

export interface ArenaState {
	width: number;
	height: number;
}

export interface TankAction {
	move: MoveIntent;
	aimAngle: number;
	fire: boolean;
	plantBomb: boolean;
	aimTarget?: {
		x: number;
		y: number;
	};
}

export interface ObstacleStateView {
	id: string;
	x: number;
	y: number;
	width: number;
	height: number;
}

export interface TankStateView {
	id: string;
	controllerId: string;
	team: Team;
	kind: TankKind;
	x: number;
	y: number;
	size: number;
	speed: number;
	color: string;
	aimAngle: number;
	aimTargetX: number | null;
	aimTargetY: number | null;
	destroyed: boolean;
	ammoType: AmmoType;
	maxAmmo: number;
	activeAmmo: number;
	bombType: BombType | null;
	maxBombs: number;
	activeBombs: number;
	shotCooldownTicks: number;
	shotCooldownTicksOnFire: number;
	bombCooldownTicks: number;
	bombCooldownTicksOnPlant: number;
	wasLastMoveBlocked: boolean;
	lastMoveIntent: MoveIntent;
	consecutiveDirectionMoves: number;
	aggressionFactor: number;
}

export interface ProjectileStateView {
	id: string;
	ownerTankId: string;
	team: Team;
	kind: AmmoType;
	x: number;
	y: number;
	vx: number;
	vy: number;
	speed: number;
	radius: number;
	bounces: number;
	maxBounces: number;
}

export interface BombStateView {
	id: string;
	ownerTankId: string;
	team: Team;
	kind: BombType;
	x: number;
	y: number;
	radius: number;
	blastRadius: number;
	fuseTicksRemaining: number;
}

export interface MatchInit {
	seed: number;
	tickRate: number;
	arena: ArenaState;
	obstacles: ObstacleStateView[];
	tanks: TankStateView[];
	selfId: string;
	controllerId: string;
}

export interface TankObservation {
	tick: number;
	self: TankStateView;
	enemies: TankStateView[];
	projectiles: ProjectileStateView[];
	bombs: BombStateView[];
	obstacles: ObstacleStateView[];
	arena: ArenaState;
}

export interface TankController {
	reset?(initial: MatchInit): void;
	act(obs: TankObservation): TankAction;
}

export interface GameState {
	seed: number;
	rngState: number;
	tick: number;
	tickRate: number;
	status: MatchStatus;
	arena: ArenaState;
	playerTankId: string;
	nextEntityId: number;
	obstacles: ObstacleStateView[];
	tanks: TankStateView[];
	projectiles: ProjectileStateView[];
	bombs: BombStateView[];
}

export interface SimulationEventBase {
	type: string;
	tick: number;
}

export interface ProjectileFiredEvent extends SimulationEventBase {
	type: 'projectile-fired';
	tankId: string;
	projectileId: string;
	x: number;
	y: number;
	audioCue: AudioCue;
}

export interface ProjectileDestroyedEvent extends SimulationEventBase {
	type: 'projectile-destroyed';
	projectileId: string;
	x: number;
	y: number;
	visualSeed: number;
	audioCue: AudioCue;
}

export interface BombPlantedEvent extends SimulationEventBase {
	type: 'bomb-planted';
	bombId: string;
	tankId: string;
	x: number;
	y: number;
}

export interface BombExplodedEvent extends SimulationEventBase {
	type: 'bomb-exploded';
	bombId: string;
	tankId: string;
	x: number;
	y: number;
	blastRadius: number;
	visualSeed: number;
	audioCue: AudioCue;
}

export interface TankDestroyedEvent extends SimulationEventBase {
	type: 'tank-destroyed';
	tankId: string;
	x: number;
	y: number;
	visualSeed: number;
	audioCue: AudioCue;
}

export type SimulationEvent =
	| ProjectileFiredEvent
	| ProjectileDestroyedEvent
	| BombPlantedEvent
	| BombExplodedEvent
	| TankDestroyedEvent;

export interface ControllerStepRecord {
	controllerId: string;
	tankId: string;
	observation: TankObservation;
	action: TankAction;
	reward: number;
	done: boolean;
}

export interface SimulationStepResult {
	tick: number;
	actions: Record<string, TankAction>;
	events: SimulationEvent[];
	records: ControllerStepRecord[];
}

export interface ReplayTick {
	tick: number;
	actions: Record<string, TankAction>;
}

export interface ReplayData {
	seed: number;
	levelConfig: LevelConfig;
	ticks: ReplayTick[];
}

export interface DatasetSample {
	controllerId: string;
	tankId: string;
	tick: number;
	observation: TankObservation;
	action: TankAction;
	reward: number;
	done: boolean;
}

export interface MatchBootstrap {
	initialState: GameState;
	controllers: Record<string, TankController>;
}
