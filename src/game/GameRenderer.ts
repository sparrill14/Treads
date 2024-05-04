import { Tank } from "./Tank";

export class GameRenderer {
    private context: CanvasRenderingContext2D | null;

    constructor(public canvas: HTMLCanvasElement) {
        const context = this.canvas.getContext('2d');
        if (!context) {
            throw new Error('2d context not supported or canvas element not found.');
        }
        this.context = context;
    }

    public initializeCanvas(width: number, height: number): void {
        this.canvas.width = width;
        this.canvas.height = height;
    }

    render(progress: number, playerTank: Tank, enemyTanks: Tank[]): void {
        if (!this.context) {
            throw new Error('2d context not supported or canvas element not found.');
        }
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);

        playerTank.updatePosition(playerTank)
        playerTank.aim(playerTank.aimXPos, playerTank.aimYPos, playerTank)
        enemyTanks.forEach(tank => {
            tank.updatePosition(playerTank)
            tank.aim(tank.aimXPos, tank.aimYPos, playerTank)
            tank.shoot(playerTank)
        });

        enemyTanks.forEach(tank => {
            tank.draw(this.context as CanvasRenderingContext2D)
            tank.reticule.draw(this.context as CanvasRenderingContext2D, tank.xPos, tank.yPos, tank.aimXPos, tank.aimYPos);
            tank.ammunition.forEach(ammunition => {
                if(ammunition.isDestroyed) {
                    return;
                }
                ammunition.updatePosition(tank.obstacleCanvas);
                ammunition.checkPlayerHit(playerTank);
                ammunition.draw(this.context as CanvasRenderingContext2D);
            });
        });

        playerTank.draw(this.context as CanvasRenderingContext2D)
        playerTank.reticule.draw(this.context as CanvasRenderingContext2D, playerTank.xPos, playerTank.yPos, playerTank.aimXPos, playerTank.aimYPos);
        playerTank.ammunition.forEach(ammunition => {
            if(ammunition.isDestroyed) {
                return;
            }
            ammunition.updatePosition(playerTank.obstacleCanvas);
            ammunition.checkEnemyHit(enemyTanks);
            ammunition.draw(this.context as CanvasRenderingContext2D);
        });
    }
}
