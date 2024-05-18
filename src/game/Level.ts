import { Obstacle } from "./Obstacle";
import { StationaryTank, StationaryRandomAimTank, SimpleMovingTank, DefaultPlayerTank, BomberTank } from "./Tank";
import { Ammunition, BasicAIAmmunition, SuperAIAmmunition } from "./Ammunition";
import { ObstacleCanvas } from "./ObstacleCanvas";
import { GameCanvas } from "./GameCanvas";
import { NavigationGrid } from "./NavigationGrid";
import { Bomb, BasicBomb } from "./Bomb";
import { AudioManager } from "./AudioManager";

export class Level {
    public gameCanvas: GameCanvas
    public obstacleCanvas: ObstacleCanvas
    public canvasWidth: number = 1000;
    public canvasHeight: number = 500;
    public audioManager: AudioManager;

    constructor(obstacleCanvas: ObstacleCanvas, audioManager: AudioManager) {
        this.obstacleCanvas = obstacleCanvas;
        this.gameCanvas = new GameCanvas('#game-canvas', this.canvasWidth, this.canvasHeight, obstacleCanvas)
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
        let obs: Obstacle = new Obstacle(300, 200, 40, 100);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs]);
        super(obstacleCanvas, audioManager)
        const stationaryTank = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 900, 240, obstacleCanvas, audioManager);
        this.gameCanvas.addEnemyTank(stationaryTank);

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 100, 250, obstacleCanvas, audioManager);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}

export class Level2 extends Level {
    constructor(audioManager: AudioManager) {
        let obs: Obstacle = new Obstacle(300, 200, 40, 100);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs]);
        super(obstacleCanvas, audioManager)

        const stationaryTank1 = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 800, 100, obstacleCanvas, audioManager);
        const stationaryTank2 = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 900, 240, obstacleCanvas, audioManager);
        const stationaryTank3 = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 800, 400, obstacleCanvas, audioManager);
        this.gameCanvas.addEnemyTank(stationaryTank1);
        this.gameCanvas.addEnemyTank(stationaryTank2);
        this.gameCanvas.addEnemyTank(stationaryTank3);

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 100, 250, obstacleCanvas, audioManager);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}

export class Level3 extends Level {
    constructor(audioManager: AudioManager) {
        let obs: Obstacle = new Obstacle(100, 100, 200, 100);
        let obs2: Obstacle = new Obstacle(550, 200, 30, 200);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2]);
        super(obstacleCanvas, audioManager)
        let superAmmo: Ammunition[] = [
            new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager)
        ]
        const aiTank = new StationaryRandomAimTank(this.gameCanvas.gameRenderer.canvas, 800, 300, obstacleCanvas, superAmmo, audioManager);
        this.gameCanvas.addEnemyTank(aiTank);

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 200, 250, obstacleCanvas, audioManager);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}

export class Level4 extends Level {
    constructor(audioManager: AudioManager) {
        let obs: Obstacle = new Obstacle(250, 100, 600, 40);
        let obs2: Obstacle = new Obstacle(100, 350, 600, 40);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2]);
        super(obstacleCanvas, audioManager)
        let basicAmmo: Ammunition[] = [
            new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)
        ]
        let basicBomb: Bomb[] = []
        let navigationGrid: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas)
        const aiTank = new SimpleMovingTank(this.gameCanvas.gameRenderer.canvas, 900, 50, obstacleCanvas, basicAmmo, basicBomb, navigationGrid, audioManager);
        this.gameCanvas.addEnemyTank(aiTank);

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 50, 450, obstacleCanvas, audioManager);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}

export class Level5 extends Level {
    constructor(audioManager: AudioManager) {
        let obs: Obstacle = new Obstacle(100, 100, 200, 100);
        let obs2: Obstacle = new Obstacle(700, 100, 30, 100);
        let obs3: Obstacle = new Obstacle(700, 350, 30, 100);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2, obs3]);
        super(obstacleCanvas, audioManager)
        let basicAmmo: Ammunition[] = [
            new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)
        ]
        let superAmmo: Ammunition[] = [
            new SuperAIAmmunition(0, 0, 0, 0, 0, true, audioManager)
        ]
        let basicBomb: Bomb[] = []
        let navigationGrid: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas)
        const aiTank3 = new SimpleMovingTank(this.gameCanvas.gameRenderer.canvas, 800, 300, obstacleCanvas, basicAmmo, basicBomb, navigationGrid, audioManager);
        const aiTank = new StationaryRandomAimTank(this.gameCanvas.gameRenderer.canvas, 800, 100, obstacleCanvas, superAmmo, audioManager);
        const aiTank2 = new StationaryTank(this.gameCanvas.gameRenderer.canvas, 800, 200, obstacleCanvas, audioManager);
        this.gameCanvas.addEnemyTank(aiTank3);
        this.gameCanvas.addEnemyTank(aiTank);
        this.gameCanvas.addEnemyTank(aiTank2);

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 200, 250, obstacleCanvas, audioManager);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}

export class Level6 extends Level {
    constructor(audioManager: AudioManager) {
        let obs: Obstacle = new Obstacle(100, 100, 200, 100);
        let obs2: Obstacle = new Obstacle(700, 100, 30, 300);
        const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', 1000, 500, [obs, obs2]);
        super(obstacleCanvas, audioManager)
        let basicAmmo: Ammunition[] = [
            new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)
        ]
        
        let basicBombs: Bomb[] = [
            new BasicBomb(0, 0, true, audioManager),
            new BasicBomb(0, 0, true, audioManager),
            new BasicBomb(0, 0, true, audioManager)
        ]
        let basicBombs2: Bomb[] = [
            new BasicBomb(0, 0, true, audioManager),
            new BasicBomb(0, 0, true, audioManager),
            new BasicBomb(0, 0, true, audioManager)
        ]
        let basicBombs3: Bomb[] = [
            new BasicBomb(0, 0, true, audioManager),
            new BasicBomb(0, 0, true, audioManager),
            new BasicBomb(0, 0, true, audioManager)
        ]

        let navigationGrid: NavigationGrid = new NavigationGrid(this.gameCanvas, this.obstacleCanvas)
        const aiTank = new BomberTank(this.gameCanvas.gameRenderer.canvas, 800, 100, obstacleCanvas, [new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)], basicBombs, navigationGrid, audioManager);
        const aiTank2 = new BomberTank(this.gameCanvas.gameRenderer.canvas, 800, 200, obstacleCanvas, [new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)], basicBombs2, navigationGrid, audioManager);
        const aiTank3 = new BomberTank(this.gameCanvas.gameRenderer.canvas, 800, 300, obstacleCanvas, [new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)], basicBombs3, navigationGrid, audioManager);

        this.gameCanvas.addEnemyTank(aiTank);
        this.gameCanvas.addEnemyTank(aiTank2);
        this.gameCanvas.addEnemyTank(aiTank3);

        const playerTank = new DefaultPlayerTank(this.gameCanvas.gameRenderer.canvas, 200, 250, obstacleCanvas, audioManager);
        this.gameCanvas.addPlayerTank(playerTank);
    }
}
