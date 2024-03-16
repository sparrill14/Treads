import { Tank, StevesTank, GregsTank } from './Tank';
import { GameRenderer } from './GameRenderer';
import { PlayerController } from '../controllers/PlayerController';
import { Obstacle } from './Obstacle';
import { ObstacleCanvas } from './ObstacleCanvas';

export class GameCanvas {
    public gameRenderer: GameRenderer;
    private playerTank: Tank;
    private enemyTanks: Tank[] = [];
    private lastRenderTime: number;
    public obstacleCanvas: ObstacleCanvas;
    public width: number;
    public height: number;

    constructor(canvasSelector: string, width: number, height: number, obstacleCanvas: ObstacleCanvas) {
        this.width = width;
        this.height = height;
        this.obstacleCanvas = obstacleCanvas;
        this.gameRenderer = new GameRenderer(document.querySelector(canvasSelector) as HTMLCanvasElement);
        this.gameRenderer.initializeCanvas(this.width, this.height);
        this.lastRenderTime = 0;
        window.addEventListener('resize', this.resizeCanvas.bind(this));
        const playerController = new PlayerController(this.gameRenderer.canvas);
        this.playerTank = new GregsTank(playerController, 200, 250, width, height, obstacleCanvas);
        requestAnimationFrame(this.gameLoop.bind(this));
    }

    private resizeCanvas(): void {

    }

    private gameLoop(timeStamp: number): void {
        const progress = timeStamp - this.lastRenderTime;
        this.gameRenderer.render(progress, this.playerTank, this.enemyTanks);
        this.lastRenderTime = timeStamp;
        requestAnimationFrame(this.gameLoop.bind(this));
    }

    public addEnemyTank(tank: Tank): void {
        this.enemyTanks.push(tank);
    }
}
