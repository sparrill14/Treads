import { InputManager } from '../utils/InputManager';
import { AudioManager } from './AudioManager';
import { GameCanvas } from './GameCanvas';
import { getLevelTankConfigs, type LevelConfig } from './LevelConfig';
import { HumanInputController } from './controllers/HumanInputController';
import { createMatchBootstrap } from './core/MatchFactory';
import { Simulation } from './core/Simulation';
import type { TankController } from './core/types';

interface LevelOptions {
	audioManager?: AudioManager;
	headless?: boolean;
	seed?: number;
	playerController?: TankController;
	controllerOverrides?: Record<string, TankController>;
}

export class Level {
	public readonly simulation: Simulation;
	public readonly seed: number;
	public readonly audioManager?: AudioManager;
	private readonly headless: boolean;
	private readonly inputManager: InputManager | null = null;
	private readonly gameCanvas: GameCanvas | null = null;

	constructor(config: LevelConfig, options: LevelOptions = {}) {
		this.headless = options.headless ?? false;
		this.seed = options.seed ?? 1;
		this.audioManager = options.audioManager;

		let playerController = options.playerController;
		const hasHumanTank = getLevelTankConfigs(config).some(
			(tank) => (tank.control ?? (tank.kind === 'player' ? 'human' : 'scripted')) === 'human'
		);
		if (!this.headless && hasHumanTank) {
			const gameCanvasElement = document.querySelector('#game-canvas') as HTMLCanvasElement;
			this.inputManager = new InputManager(gameCanvasElement);
			playerController = playerController ?? new HumanInputController(this.inputManager);
		}

		const bootstrap = createMatchBootstrap(config, this.seed, { playerController });
		const controllers = {
			...bootstrap.controllers,
			...(options.controllerOverrides ?? {}),
		};
		this.simulation = new Simulation(bootstrap.initialState, controllers);

		if (!this.headless) {
			this.gameCanvas = new GameCanvas('#game-canvas', this.simulation, this.audioManager);
		}
	}

	public start(): void {
		this.gameCanvas?.start();
	}

	public stop(): void {
		this.gameCanvas?.stop();
		this.inputManager?.destroy();
	}
}
