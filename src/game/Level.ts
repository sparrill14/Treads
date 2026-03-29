import { InputManager } from '../utils/InputManager';
import { Ammunition, BasicAIAmmunition, SuperAIAmmunition } from './Ammunition';
import { AudioManager } from './AudioManager';
import { BasicBomb, Bomb, LoveBomb } from './Bomb';
import { GameCanvas } from './GameCanvas';
import { type EnemyConfig, type LevelConfig, type NavigatorConfig } from './LevelConfig';
import { Obstacle } from './Obstacle';
import { ObstacleCanvas } from './ObstacleCanvas';
import { AStarNavigator } from './navigation/AStarNavigator';
import { AStarNavigatorWithAvoidance } from './navigation/AStarNavigatorWithAvoidance';
import { NavigationGrid } from './navigation/NavigationGrid';
import { Navigator } from './navigation/Navigator';
import { SimpleNavigator } from './navigation/SimpleNavigator';
import { SimplePathfinder } from './navigation/SimplePathFinder';
import { BomberTank } from './tanks/BomberTank';
import { EnemyTank } from './tanks/EnemyTank';
import { DefaultPlayerTank } from './tanks/PlayerTank';
import { SimpleMovingTank } from './tanks/SimpleMovingTank';
import { StationaryRandomAimTank } from './tanks/StationaryRandomAimTank';
import { StationaryTank } from './tanks/StationaryTank';
import { SuperBomberMovingTank } from './tanks/SuperBomberMovingTank';

export class Level {
	public gameCanvas: GameCanvas;
	public obstacleCanvas: ObstacleCanvas;
	public canvasWidth = 1000;
	public canvasHeight = 500;
	public audioManager: AudioManager;
	public inputManager: InputManager;

	constructor(config: LevelConfig, audioManager: AudioManager) {
		this.audioManager = audioManager;

		const obstacles = config.obstacles.map((o) => new Obstacle(o.x, o.y, o.width, o.height));
		this.obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', this.canvasWidth, this.canvasHeight, obstacles);
		this.gameCanvas = new GameCanvas('#game-canvas', this.canvasWidth, this.canvasHeight, this.obstacleCanvas);
		this.inputManager = new InputManager(this.gameCanvas.gameRenderer.canvas);

		for (const enemy of config.enemies) {
			this.gameCanvas.addEnemyTank(this.createEnemy(enemy));
		}

		const playerTank = new DefaultPlayerTank(
			this.gameCanvas.gameRenderer.canvas,
			config.player.x,
			config.player.y,
			this.obstacleCanvas,
			audioManager,
			this.inputManager
		);
		this.gameCanvas.addPlayerTank(playerTank);
	}

	public stop() {
		this.gameCanvas.stop();
		this.inputManager.destroy();
	}

	public start() {
		this.gameCanvas.start();
	}

	private createAmmo(type: 'basic' | 'super', count: number): Ammunition[] {
		const Ctor = type === 'basic' ? BasicAIAmmunition : SuperAIAmmunition;
		return Array.from({ length: count }, () => new Ctor(0, 0, 0, 0, 0, true, this.audioManager));
	}

	private createBombs(type: 'basic' | 'love', count: number): Bomb[] {
		const Ctor = type === 'basic' ? BasicBomb : LoveBomb;
		return Array.from({ length: count }, () => new Ctor(0, 0, true, this.audioManager));
	}

	private createNavigator(navConfig: NavigatorConfig): Navigator {
		let nav: Navigator;
		switch (navConfig.type) {
			case 'simple': {
				const pathfinder = new SimplePathfinder(this.gameCanvas, this.obstacleCanvas, false);
				nav = new SimpleNavigator(pathfinder);
				break;
			}
			case 'astar': {
				const grid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas, false);
				nav = new AStarNavigator(grid);
				break;
			}
			case 'astar-avoidance': {
				const grid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas, false);
				nav = new AStarNavigatorWithAvoidance(grid);
				break;
			}
		}
		if (navConfig.aggressionFactor !== undefined) {
			nav.aggressionFactor = navConfig.aggressionFactor;
		}
		return nav;
	}

	private createEnemy(cfg: EnemyConfig): EnemyTank {
		const canvas = this.gameCanvas.gameRenderer.canvas;
		const obs = this.obstacleCanvas;
		const am = this.audioManager;

		switch (cfg.type) {
			case 'stationary':
				return new StationaryTank(canvas, cfg.x, cfg.y, obs, am);
			case 'stationary-random-aim':
				return new StationaryRandomAimTank(
					canvas,
					cfg.x,
					cfg.y,
					obs,
					this.createAmmo(cfg.ammo?.type ?? 'basic', cfg.ammo?.count ?? 1),
					am
				);
			case 'simple-moving':
				return new SimpleMovingTank(
					canvas,
					cfg.x,
					cfg.y,
					obs,
					this.createAmmo(cfg.ammo?.type ?? 'basic', cfg.ammo?.count ?? 1),
					this.createBombs(cfg.bombs?.type ?? 'basic', cfg.bombs?.count ?? 0),
					this.createNavigator(cfg.navigator ?? { type: 'astar' }),
					am
				);
			case 'bomber':
				return new BomberTank(
					canvas,
					cfg.x,
					cfg.y,
					obs,
					this.createAmmo(cfg.ammo?.type ?? 'basic', cfg.ammo?.count ?? 1),
					this.createBombs(cfg.bombs?.type ?? 'basic', cfg.bombs?.count ?? 0),
					this.createNavigator(cfg.navigator ?? { type: 'astar' }),
					am
				);
			case 'super-bomber':
				return new SuperBomberMovingTank(
					canvas,
					cfg.x,
					cfg.y,
					obs,
					this.createAmmo(cfg.ammo?.type ?? 'super', cfg.ammo?.count ?? 1),
					this.createBombs(cfg.bombs?.type ?? 'basic', cfg.bombs?.count ?? 0),
					this.createNavigator(cfg.navigator ?? { type: 'astar' }),
					am
				);
		}
	}
}
