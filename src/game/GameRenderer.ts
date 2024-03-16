import { Tank } from "./Tank";
import { Obstacle } from "./Obstacle";

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

        playerTank.updatePosition(playerTank.controller.aimXPos, playerTank.controller.aimYPos, playerTank)

        enemyTanks.forEach(tank => {
            tank.updatePosition(tank.controller.aimXPos, tank.controller.aimYPos, playerTank)
        });

        playerTank.draw(this.context as CanvasRenderingContext2D)
        playerTank.reticule.draw(this.context as CanvasRenderingContext2D, playerTank.xPos, playerTank.yPos, playerTank.controller.aimXPos, playerTank.controller.aimYPos);
        playerTank.controller.ammunition.forEach(ammunition => {
            if(ammunition.isDestroyed) {
                return;
            }
            ammunition.updatePosition(playerTank.obstacleCanvas);
            ammunition.draw(this.context as CanvasRenderingContext2D);
        });

        enemyTanks.forEach(tank => {
            tank.draw(this.context as CanvasRenderingContext2D)
            tank.reticule.draw(this.context as CanvasRenderingContext2D, tank.xPos, tank.yPos, tank.controller.aimXPos, tank.controller.aimYPos);
            tank.controller.ammunition.forEach(ammunition => {
                if(ammunition.isDestroyed) {
                    return;
                }
                ammunition.updatePosition(tank.obstacleCanvas);
                ammunition.draw(this.context as CanvasRenderingContext2D);
            });
        });
    }
}
