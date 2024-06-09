import { Ammunition, BasicAIAmmunition, SuperAIAmmunition } from './Ammunition';
import { AudioManager } from './AudioManager';
import { BasicBomb, Bomb, LoveBomb } from './Bomb';
import { GameCanvas } from './GameCanvas';
import { NavigationGrid } from './NavigationGrid';
import { Obstacle } from './Obstacle';
import { ObstacleCanvas } from './ObstacleCanvas';
import { BomberTank } from './tanks/BomberTank';
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

	constructor(obstacleCanvas: ObstacleCanvas, audioManager: AudioManager) {
		this.obstacleCanvas = obstacleCanvas;
		this.gameCanvas = new GameCanvas('#game-canvas', this.canvasWidth, this.canvasHeight, obstacleCanvas);
		this.audioManager = audioManager;
	}

	public stop() {
		this.gameCanvas.stop();
	}

	public start() {
		this.gameCanvas.start();
	}
}

export class Level1 extends Level {
	constructor(audioManager: AudioManager) {
		const obs: Obstacle = new Obstacle(300, 200, 40, 100);
		const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs]);
		super(obstacleCanvas, audioManager);
		const stationaryTank = new StationaryTank(
			this.gameCanvas.gameRenderer.canvas,
			900,
			240,
			obstacleCanvas,
			audioManager
		);
		this.gameCanvas.addEnemyTank(stationaryTank);

		const playerTank = new DefaultPlayerTank(
			this.gameCanvas.gameRenderer.canvas,
			100,
			250,
			obstacleCanvas,
			audioManager
		);
		this.gameCanvas.addPlayerTank(playerTank);
	}
}

export class Level2 extends Level {
	constructor(audioManager: AudioManager) {
		const obs: Obstacle = new Obstacle(300, 200, 40, 100);
		const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs]);
		super(obstacleCanvas, audioManager);

		const stationaryTank1 = new StationaryTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			100,
			obstacleCanvas,
			audioManager
		);
		const stationaryTank2 = new StationaryTank(
			this.gameCanvas.gameRenderer.canvas,
			900,
			240,
			obstacleCanvas,
			audioManager
		);
		const stationaryTank3 = new StationaryTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			400,
			obstacleCanvas,
			audioManager
		);
		this.gameCanvas.addEnemyTank(stationaryTank1);
		this.gameCanvas.addEnemyTank(stationaryTank2);
		this.gameCanvas.addEnemyTank(stationaryTank3);

		const playerTank = new DefaultPlayerTank(
			this.gameCanvas.gameRenderer.canvas,
			100,
			250,
			obstacleCanvas,
			audioManager
		);
		this.gameCanvas.addPlayerTank(playerTank);
	}
}

export class Level3 extends Level {
	constructor(audioManager: AudioManager) {
		const obs: Obstacle = new Obstacle(100, 100, 200, 100);
		const obs2: Obstacle = new Obstacle(550, 200, 30, 200);
		const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2]);
		super(obstacleCanvas, audioManager);
		const superAmmo: Ammunition[] = [new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager)];
		const aiTank = new StationaryRandomAimTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			300,
			obstacleCanvas,
			superAmmo,
			audioManager
		);
		this.gameCanvas.addEnemyTank(aiTank);

		const playerTank = new DefaultPlayerTank(
			this.gameCanvas.gameRenderer.canvas,
			200,
			250,
			obstacleCanvas,
			audioManager
		);
		this.gameCanvas.addPlayerTank(playerTank);
	}
}

export class Level4 extends Level {
	constructor(audioManager: AudioManager) {
		const obs: Obstacle = new Obstacle(250, 100, 600, 40);
		const obs2: Obstacle = new Obstacle(100, 350, 600, 40);
		const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2]);
		super(obstacleCanvas, audioManager);
		const basicAmmo: Ammunition[] = [new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)];
		const basicBomb: Bomb[] = [];
		const navigationGrid: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas, false);
		const aiTank = new SimpleMovingTank(
			this.gameCanvas.gameRenderer.canvas,
			900,
			50,
			obstacleCanvas,
			basicAmmo,
			basicBomb,
			navigationGrid,
			audioManager
		);
		this.gameCanvas.addEnemyTank(aiTank);

		const playerTank = new DefaultPlayerTank(
			this.gameCanvas.gameRenderer.canvas,
			50,
			450,
			obstacleCanvas,
			audioManager
		);
		this.gameCanvas.addPlayerTank(playerTank);
	}
}

export class Level5 extends Level {
	constructor(audioManager: AudioManager) {
		const obs: Obstacle = new Obstacle(100, 100, 200, 100);
		const obs2: Obstacle = new Obstacle(700, 100, 30, 100);
		const obs3: Obstacle = new Obstacle(700, 350, 30, 100);
		const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2, obs3]);
		super(obstacleCanvas, audioManager);
		const basicAmmo: Ammunition[] = [new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)];
		const superAmmo: Ammunition[] = [new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager)];
		const basicBomb: Bomb[] = [];
		const navigationGrid: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas, false);
		const aiTank3 = new SimpleMovingTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			300,
			obstacleCanvas,
			basicAmmo,
			basicBomb,
			navigationGrid,
			audioManager
		);
		const aiTank = new StationaryRandomAimTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			100,
			obstacleCanvas,
			superAmmo,
			audioManager
		);
		const aiTank2 = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 800, 200, obstacleCanvas, audioManager);
		this.gameCanvas.addEnemyTank(aiTank3);
		this.gameCanvas.addEnemyTank(aiTank);
		this.gameCanvas.addEnemyTank(aiTank2);

		const playerTank = new DefaultPlayerTank(
			this.gameCanvas.gameRenderer.canvas,
			200,
			250,
			obstacleCanvas,
			audioManager
		);
		this.gameCanvas.addPlayerTank(playerTank);
	}
}

export class Level6 extends Level {
	constructor(audioManager: AudioManager) {
		const obs: Obstacle = new Obstacle(100, 100, 200, 100);
		const obs2: Obstacle = new Obstacle(700, 100, 30, 300);
		const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2]);
		super(obstacleCanvas, audioManager);

		const basicBombs: Bomb[] = [
			new BasicBomb(0, 0, true, audioManager),
			new BasicBomb(0, 0, true, audioManager),
			new BasicBomb(0, 0, true, audioManager),
		];
		const basicBombs2: Bomb[] = [
			new BasicBomb(0, 0, true, audioManager),
			new BasicBomb(0, 0, true, audioManager),
			new BasicBomb(0, 0, true, audioManager),
		];
		const basicBombs3: Bomb[] = [
			new BasicBomb(0, 0, true, audioManager),
			new BasicBomb(0, 0, true, audioManager),
			new BasicBomb(0, 0, true, audioManager),
		];

		const navigationGrid: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas, false);
		const navigationGrid2: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas, false);
		const navigationGrid3: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas, false);
		const aiTank = new BomberTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			100,
			obstacleCanvas,
			[new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)],
			basicBombs,
			navigationGrid,
			audioManager
		);
		const aiTank2 = new BomberTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			200,
			obstacleCanvas,
			[new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)],
			basicBombs2,
			navigationGrid2,
			audioManager
		);
		const aiTank3 = new BomberTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			300,
			obstacleCanvas,
			[new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)],
			basicBombs3,
			navigationGrid3,
			audioManager
		);

		this.gameCanvas.addEnemyTank(aiTank);
		this.gameCanvas.addEnemyTank(aiTank2);
		this.gameCanvas.addEnemyTank(aiTank3);

		const playerTank = new DefaultPlayerTank(
			this.gameCanvas.gameRenderer.canvas,
			200,
			250,
			obstacleCanvas,
			audioManager
		);
		this.gameCanvas.addPlayerTank(playerTank);
	}
}

export class Level7 extends Level {
	constructor(audioManager: AudioManager) {
		const obs: Obstacle = new Obstacle(700, 100, 30, 300);
		const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs]);
		super(obstacleCanvas, audioManager);
		const basicAmmo: Ammunition[] = [
			new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager),
		];
		const basicAmmo2: Ammunition[] = [
			new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager),
		];
		const basicAmmo3: Ammunition[] = [
			new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager),
		];

		const basicBombs: Bomb[] = [new LoveBomb(0, 0, true, audioManager), new LoveBomb(0, 0, true, audioManager)];
		const basicBombs2: Bomb[] = [new LoveBomb(0, 0, true, audioManager), new LoveBomb(0, 0, true, audioManager)];
		const basicBombs3: Bomb[] = [new LoveBomb(0, 0, true, audioManager), new LoveBomb(0, 0, true, audioManager)];

		const navigationGrid: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas, false);
		const navigationGrid2: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas, false);
		const navigationGrid3: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas, false);
		const aiTank = new SuperBomberMovingTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			100,
			obstacleCanvas,
			basicAmmo,
			basicBombs,
			navigationGrid,
			audioManager
		);
		const aiTank2 = new SuperBomberMovingTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			200,
			obstacleCanvas,
			basicAmmo2,
			basicBombs2,
			navigationGrid2,
			audioManager
		);
		const aiTank3 = new SuperBomberMovingTank(
			this.gameCanvas.gameRenderer.canvas,
			800,
			300,
			obstacleCanvas,
			basicAmmo3,
			basicBombs3,
			navigationGrid3,
			audioManager
		);

		this.gameCanvas.addEnemyTank(aiTank);
		this.gameCanvas.addEnemyTank(aiTank2);
		this.gameCanvas.addEnemyTank(aiTank3);

		const playerTank = new DefaultPlayerTank(
			this.gameCanvas.gameRenderer.canvas,
			200,
			250,
			obstacleCanvas,
			audioManager
		);
		this.gameCanvas.addPlayerTank(playerTank);
	}
}
