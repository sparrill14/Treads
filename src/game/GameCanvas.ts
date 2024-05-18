import { Tank } from './Tank';
import { GameRenderer } from './GameRenderer';
import { ObstacleCanvas } from './ObstacleCanvas';

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
            cancelAnimationFrame(this.animationFrameID)
            this.animationFrameID = null;
        }
    }

    private resizeCanvas(): void {

    }

    private gameLoop(timeStamp: number): void {
        if (this.playerTank != null) {
            const progress = timeStamp - this.lastRenderTime;
            this.gameRenderer.render(progress, this.playerTank, this.enemyTanks);
            if (this.gameRenderer.enemyWin || this.gameRenderer.playerWin) {
                this.stop();
                this.obstacleCanvas.clearObstacles();
                this.gameRenderer.renderLevelOverScreen();
            }
            this.lastRenderTime = timeStamp;
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
