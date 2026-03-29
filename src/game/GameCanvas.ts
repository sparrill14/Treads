import { GameRenderer } from './GameRenderer';
import { ObstacleCanvas } from './ObstacleCanvas';
import { Tank } from './tanks/Tank';

export class GameCanvas {
	public gameRenderer: GameRenderer;
	private playerTank: Tank | null = null;
	private enemyTanks: Tank[] = [];
	private lastRenderTime: number;
	public obstacleCanvas: ObstacleCanvas;
	public width: number;
	public height: number;
	public animationFrameID: number | null = null;

	constructor(canvasSelector: string, width: number, height: number, obstacleCanvas: ObstacleCanvas) {
		this.width = width;
		this.height = height;
		this.obstacleCanvas = obstacleCanvas;
		this.animationFrameID = null;
		this.gameRenderer = new GameRenderer(document.querySelector(canvasSelector) as HTMLCanvasElement);
		this.gameRenderer.initializeCanvas(this.width, this.height);
		this.lastRenderTime = 0;
		window.addEventListener('resize', this.resizeCanvas.bind(this));
	}

	public start() {
		if (!this.animationFrameID) {
			this.animationFrameID = requestAnimationFrame(this.gameLoop.bind(this));
		}
	}

	public stop() {
		if (this.animationFrameID) {
			cancelAnimationFrame(this.animationFrameID);
			this.animationFrameID = null;
		}
	}

	// eslint-disable-next-line @typescript-eslint/no-empty-function
	private resizeCanvas(): void {}

	private gameLoop(timeStamp: number): void {
		if (this.playerTank != null) {
			const deltaTime = Math.min((timeStamp - this.lastRenderTime) / 1000, 0.1);
			this.lastRenderTime = timeStamp;
			this.gameRenderer.update(deltaTime, this.playerTank, this.enemyTanks);
			this.gameRenderer.render(this.playerTank, this.enemyTanks);
			this.animationFrameID = requestAnimationFrame(this.gameLoop.bind(this));
		}
	}

	public addEnemyTank(tank: Tank): void {
		this.enemyTanks.push(tank);
	}

	public addPlayerTank(tank: Tank): void {
		this.playerTank = tank;
	}
}
