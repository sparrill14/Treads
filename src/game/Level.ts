import { Obstacle } from "./Obstacle";
import { StationaryTank, StationaryRandomAimTank, SimpleMovingTank, DefaultPlayerTank } from "./Tank";
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

    public stop() {
        this.gameCanvas.stop();
    }

    public start() {
        this.gameCanvas.start();
    }
}

export class Level1 extends Level {
    constructor() {
        let obs: Obstacle = new Obstacle(600, 100, 40, 300);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs]);
        super(obstacleCanvas)
        const stationaryTank = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 900, 240, obstacleCanvas);
        this.gameCanvas.addEnemyTank(stationaryTank);

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 200, 250, obstacleCanvas);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}

export class Level2 extends Level {
    constructor() {
        let obs: Obstacle = new Obstacle(600, 150, 40, 250);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs]);
        super(obstacleCanvas)

        const stationaryTank1 = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 800, 100, obstacleCanvas);
        const stationaryTank2 = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 900, 240, obstacleCanvas);
        const stationaryTank3 = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 800, 400, obstacleCanvas);
        this.gameCanvas.addEnemyTank(stationaryTank1);
        this.gameCanvas.addEnemyTank(stationaryTank2);
        this.gameCanvas.addEnemyTank(stationaryTank3);

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 200, 250, obstacleCanvas);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}

export class Level3 extends Level {
    constructor() {
        let obs: Obstacle = new Obstacle(100, 100, 200, 100);
        let obs2: Obstacle = new Obstacle(700, 100, 30, 100);
        let obs3: Obstacle = new Obstacle(700, 350, 30, 100);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2, obs3]);
        super(obstacleCanvas)
        let superAmmo: Ammunition[] = [
            new SuperAIAmmunition(0, 0, 0, 0, 0, true)
        ]
        const aiTank = new StationaryRandomAimTank(this.gameCanvas.gameRenderer.canvas, 800, 300, obstacleCanvas, superAmmo);
        this.gameCanvas.addEnemyTank(aiTank);

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 200, 250, obstacleCanvas);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}

export class Level4 extends Level {
    constructor() {
        let obs: Obstacle = new Obstacle(250, 100, 600, 40);
        let obs2: Obstacle = new Obstacle(100, 350, 600, 40);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2]);
        super(obstacleCanvas)
        let basicAmmo: Ammunition[] = [
            new BasicAIAmmunition(0, 0, 0, 0, 0, true)
        ]
        let navigationGrid: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas)
        const aiTank3 = new SimpleMovingTank(this.gameCanvas.gameRenderer.canvas, 900, 50, obstacleCanvas, basicAmmo, navigationGrid);
        this.gameCanvas.addEnemyTank(aiTank3);

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 50, 450, obstacleCanvas);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}

export class Level5 extends Level {
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

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 200, 250, obstacleCanvas);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}

