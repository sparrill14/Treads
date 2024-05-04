import { Obstacle } from "./Obstacle";
import { StationaryTank, StationaryRandomAimTank, SimpleMovingTank } from "./Tank";
import { Ammunition, BasicAIAmmunition, SuperAIAmmunition } from "./Ammunition";
import { ObstacleCanvas } from "./ObstacleCanvas";
import { GameCanvas } from "./GameCanvas";
import { NavigationGrid } from "./NavigationGrid";


export class Level {
    public gameCanvas: GameCanvas
    public obstacleCanvas: ObstacleCanvas
    public canvasWidth: number = 1000;
    public canvasHeight: number = 500;

    constructor(obstacleCanvas: ObstacleCanvas) {
        this.obstacleCanvas = obstacleCanvas;
        this.gameCanvas = new GameCanvas('#game-canvas', this.canvasWidth, this.canvasHeight, obstacleCanvas)
    }

    public start() {
        this.gameCanvas.start();
    }
}

export class Level1 extends Level {
    constructor() {
        let obs: Obstacle = new Obstacle(100, 100, 200, 100);
        let obs2: Obstacle = new Obstacle(700, 100, 30, 100);
        let obs3: Obstacle = new Obstacle(700, 350, 30, 100);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2, obs3]);
        super(obstacleCanvas)
        let basicAmmo: Ammunition[] = [
            new BasicAIAmmunition(0, 0, 0, 0, 0, true)
        ]
        let superAmmo: Ammunition[] = [
            new SuperAIAmmunition(0, 0, 0, 0, 0, true)
        ]
        let navigationGrid: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas)
        const aiTank3 = new SimpleMovingTank(this.gameCanvas.gameRenderer.canvas, 800, 300, obstacleCanvas, basicAmmo, navigationGrid);
        const aiTank = new StationaryRandomAimTank(this.gameCanvas.gameRenderer.canvas, 800, 100, obstacleCanvas, superAmmo);
        const aiTank2 = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 800, 200, obstacleCanvas);
        this.gameCanvas.addEnemyTank(aiTank3);
        this.gameCanvas.addEnemyTank(aiTank);
        this.gameCanvas.addEnemyTank(aiTank2);
    }
}

