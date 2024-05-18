import { ObstacleCanvas } from "./ObstacleCanvas";
import { Tank } from "./Tank";
import { AudioFile, AudioManager } from "./AudioManager";

export class Ammunition {
    public xPos: number;
    public yPos: number;
    public xVelocity: number;
    public yVelocity: number;
    public theta: number;
    public bounces: number;
    public maxBounces: number;
    public speed: number;
    public canvasWidth: number;
    public canvasHeight: number;
    public isDestroyed: boolean;
    public audioManager: AudioManager;

    constructor (startX: number, startY: number, theta: number, speed: number, maxBounces: number, canvasWidth: number, canvasHeight: number, isDestroyed: boolean, audioManager: AudioManager) {
        this.xPos = startX;
        this.yPos = startY;
        this.theta = theta;
        this.speed = speed;
        this.xVelocity = Math.cos(this.theta) * this.speed;
        this.yVelocity = Math.sin(this.theta) * this.speed;
        this.bounces = 0;
        this.maxBounces = maxBounces;
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        this.isDestroyed = isDestroyed;
        this.audioManager = audioManager;
    }

    updatePosition (obstacleCanvas: ObstacleCanvas): void {
        this.xPos += this.xVelocity;
        this.yPos += this.yVelocity;

        if (this.xPos <= 0 || this.xPos > this.canvasWidth) {
            this.xVelocity = -this.xVelocity;
            this.bounces++;
        }

        if (this.yPos <= 0 || this.yPos > this.canvasHeight) {
            this.yVelocity = -this.yVelocity;
            this.bounces++;
        }

        // Loop through obstacles and check for collision. Increment bounces if collision is detected. Change the velocity of the ammunition based on the angle of collision.
        obstacleCanvas.obstacles.forEach(obstacle => {
            if (this.xPos > obstacle.xLeft && this.xPos < obstacle.xRight && this.yPos > obstacle.yTop && this.yPos < obstacle.yBottom) {
              this.bounces++;
              
              // Determine the side of the collision and respond accordingly
              const fromLeft = Math.abs(this.xPos - obstacle.xLeft);
              const fromRight = Math.abs(this.xPos - obstacle.xRight);
              const fromTop = Math.abs(this.yPos - obstacle.yTop);
              const fromBottom = Math.abs(this.yPos - obstacle.yBottom);
          
              // Find the minimum distance to figure out the closest collision side
              const minDistance = Math.min(fromLeft, fromRight, fromTop, fromBottom);
          
              if (minDistance === fromTop) {
                // Projectile hits obstacle from the top
                this.yPos = obstacle.yTop - 1; // Adjust position slightly to prevent overlap
                this.yVelocity = -this.yVelocity; // Invert Y velocity
              } else if (minDistance === fromBottom) {
                // Projectile hits obstacle from the bottom
                this.yPos = obstacle.yBottom + 1; // Adjust position slightly to prevent overlap
                this.yVelocity = -this.yVelocity; // Invert Y velocity
              } else if (minDistance === fromLeft) {
                // Projectile hits obstacle from the left
                this.xPos = obstacle.xLeft - 1; // Adjust position slightly to prevent overlap
                this.xVelocity = -this.xVelocity; // Invert X velocity
              } else if (minDistance === fromRight) {
                // Projectile hits obstacle from the right
                this.xPos = obstacle.xRight + 1; // Adjust position slightly to prevent overlap
                this.xVelocity = -this.xVelocity; // Invert X velocity
              }
            }
        });

        if (this.bounces > this.maxBounces) {
            this.isDestroyed = true;
        }
    }

    checkEnemyHit (enemyTanks: Tank[]): void {
        enemyTanks.forEach(enemyTank => {
            if (enemyTank.isDestroyed) {
                return;
            }
            if (this.xPos > enemyTank.xLeft && this.xPos < enemyTank.xRight && this.yPos > enemyTank.yTop && this.yPos < enemyTank.yBottom) {
                this.isDestroyed = true;
                enemyTank.isDestroyed = true;
                this.audioManager.play(AudioFile.TANK_DESTROY);
                console.log("Enemy hit!!!");
            }
        })
    }

    checkPlayerHit (playerTank: Tank): void {
        if (playerTank.isDestroyed) {
            return;
        }
        if (this.xPos > playerTank.xLeft && this.xPos < playerTank.xRight && this.yPos > playerTank.yTop && this.yPos < playerTank.yBottom) {
            // playerTank.isDestroyed = true;
            this.isDestroyed = true;
            console.log("Player Hit!!!");
        }
    }

    reload (startX: number, startY: number, theta: number, isDestroyed: boolean, canvasWidth: number, canvasHeight: number, ) {
        this.xPos = startX;
        this.yPos = startY;
        this.theta = theta;
        this.isDestroyed = isDestroyed;
        this.xVelocity = Math.cos(this.theta) * this.speed;
        this.yVelocity = Math.sin(this.theta) * this.speed;
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        this.bounces = 0;
    }

    draw (context: CanvasRenderingContext2D): void {
        context.beginPath();
        context.arc(this.xPos, this.yPos, 4, 0, 2 * Math.PI);
        context.fillStyle = 'white';
        context.fill();
        context.lineWidth = 2;
        context.strokeStyle = 'black';
        context.stroke();
        context.closePath();
    }

    willHitPlayerTank(obstacleCanvas: ObstacleCanvas, playerTank: Tank, ): boolean {
        let predictedXPosition: number = this.xPos;
        let predictedYPosition: number = this.yPos;
        let predictedXVelocity: number = this.xVelocity;
        let predictedYVelocity: number = this.yVelocity;
        let bounces: number = 0;
        while (bounces <= this.maxBounces) {
            predictedXPosition += this.xVelocity;
            predictedYPosition += this.yVelocity;
            if (predictedXPosition <= 0 || predictedXPosition > this.canvasWidth) {
                predictedXVelocity = -predictedXVelocity;
                bounces++;
            }
            if (predictedYPosition <= 0 || predictedYPosition > this.canvasHeight) {
                predictedYVelocity = -predictedYVelocity;
                bounces++;
            }
            obstacleCanvas.obstacles.forEach(obstacle => {
                if (predictedXPosition > obstacle.xLeft && predictedXPosition < obstacle.xRight && predictedYPosition > obstacle.yTop && predictedYPosition < obstacle.yBottom) {
                    bounces++;
                    predictedXVelocity = -predictedXVelocity;
                    predictedYVelocity = -predictedYVelocity;
                }
            });
            if (predictedXPosition > playerTank.xLeft && predictedXPosition < playerTank.xRight && predictedYPosition > playerTank.yTop && predictedYPosition < playerTank.yBottom) {
                return true;
            }
        }
        return false;
    }
}

export class PlayerAmmunition extends Ammunition {
    constructor (startX: number, startY: number, theta: number, canvasWidth: number, canvasHeight: number, isDestroyed: boolean, audioManager: AudioManager) {
        let playerAmmunitionMaxBounces: number = 1;
        let playerAmmunitionSpeed: number = 4;
        super(startX, startY, theta, playerAmmunitionSpeed, playerAmmunitionMaxBounces, canvasWidth, canvasHeight, isDestroyed, audioManager);
    }
}

export class BasicAIAmmunition extends Ammunition {
    constructor (startX: number, startY: number, theta: number, canvasWidth: number, canvasHeight: number, isDestroyed: boolean, audioManager: AudioManager) {
        let BasicAIAmmunitionMaxBounces: number = 0;
        let BasicAIAmmunitionSpeed: number = 4;
        super(startX, startY, theta, BasicAIAmmunitionSpeed, BasicAIAmmunitionMaxBounces, canvasWidth, canvasHeight, isDestroyed, audioManager);
    }
}

export class SuperAIAmmunition extends Ammunition {
    constructor (startX: number, startY: number, theta: number, canvasWidth: number, canvasHeight: number, isDestroyed: boolean, audioManager: AudioManager) {
        let superAIAmmunitionMaxBounces: number = 2;
        let superAIAmmunitionSpeed: number = 7;
        super(startX, startY, theta, superAIAmmunitionSpeed, superAIAmmunitionMaxBounces, canvasWidth, canvasHeight, isDestroyed, audioManager);
    }
}
