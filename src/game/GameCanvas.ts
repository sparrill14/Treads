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
	private frameInterval = 1000 / 45;

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
			const elapsed = timeStamp - this.lastRenderTime;
			if (elapsed >= this.frameInterval) {
				this.gameRenderer.render(elapsed, this.playerTank, this.enemyTanks);
				this.lastRenderTime = timeStamp;
			}
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
