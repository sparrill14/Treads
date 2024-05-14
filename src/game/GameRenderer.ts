import { Tank } from "./Tank";

export class GameRenderer {
    public playerWin: boolean = false;
    public enemyWin: boolean = false;

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

    renderLevelOverScreen() {
        if (!this.context) {
            throw new Error('2d context not supported or canvas element not found.');
        }

        const message = this.playerWin ? "Win" : this.enemyWin ? "Lose" : "";
        if (message === "") {
            return;
        }

        const fontSize = 100;
        this.context.font = `${fontSize}px Arial`;
        this.context.lineWidth = 5;
        this.context.strokeStyle = this.playerWin ? 'green' : 'red';
        this.context.fillStyle = this.playerWin ? 'green' : 'red';

        const textWidth = this.context.measureText(message).width;
        const x = (this.canvas.width - textWidth) / 2;
        const y = this.canvas.height / 2 + fontSize / 2;

        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.context.strokeText(message, x, y);
        this.context.fillText(message, x, y);
    }

    render(progress: number, playerTank: Tank, enemyTanks: Tank[]): void {
        if (!this.context) {
            throw new Error('2d context not supported or canvas element not found.');
        }
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (playerTank.isDestroyed) {
            this.enemyWin = true;
        }
        else if (enemyTanks.every(tank => tank.isDestroyed)) {
            this.playerWin = true;
        }
        else {
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
}
