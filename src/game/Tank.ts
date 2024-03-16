import { IController } from "../controllers/IController";
import { ObstacleCanvas } from "./ObstacleCanvas";
import { Reticule, SimplePlayerReticule, CustomColorReticule, AdjustingCustomColorReticule, NoReticule } from "./Reticule";

export class Tank {
    public xPos: number;
    public yPos: number;
    public speed: number;
    public size: number;
    public tankMidpoint: number;
    public color: string;
    public reticule: Reticule;
    public controller: IController;
    public gunBarrellWidth: number = 7;
    public isDestroyed: boolean = false;
    public obstacleCanvas: ObstacleCanvas;
    public twoPi: number = 2 * Math.PI;

    protected canvasWidth: number;
    protected canvasHeight: number;

    constructor(controller: IController, reticule: Reticule, xPos: number, yPos: number, speed: number, size: number, color: string, canvasWidth: number, canvasHeight: number, obstacleCanvas: ObstacleCanvas) {
        this.controller = controller;
        this.reticule = reticule;
        this.xPos = xPos;
        this.yPos = yPos;
        this.speed = speed;
        this.size = size;
        this.color = color;
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        this.obstacleCanvas = obstacleCanvas;
        this.tankMidpoint = this.size / 2;
    }

    public draw(context: CanvasRenderingContext2D): void {
        if (this.isDestroyed) {
            return;
        }

        // Tank body
        context.fillStyle = this.color;
        context.fillRect(this.xPos, this.yPos, this.size, this.size);

        // Tank outline
        context.setLineDash([]);
        context.lineJoin = 'bevel'
        context.strokeStyle = 'black';
        context.lineWidth = 2;
        context.strokeRect(this.xPos, this.yPos, this.size, this.size);

        // Tank turret
        context.beginPath();
        context.arc(this.xPos + this.tankMidpoint, this.yPos + this.tankMidpoint, this.size / 3, 0, this.twoPi);
        context.stroke();

        // Tank gun barrell
        const endX = this.xPos + this.tankMidpoint + (Math.cos(this.controller.aimAngle) * this.size);
        const endY = this.yPos + this.tankMidpoint + (Math.sin(this.controller.aimAngle) * this.size);
        context.beginPath();
        context.moveTo(this.xPos + this.tankMidpoint, this.yPos + this.tankMidpoint);
        context.lineTo(endX, endY);
        context.lineWidth = this.gunBarrellWidth;
        context.stroke();
    }

    public updatePosition(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
        if (this.isDestroyed) {
            return;
        }

        let dy: number;
        let dx: number;

        if (this.xPos == playerTank.xPos && this.yPos == playerTank.yPos) {
            // Aim at the mouse
            dx = mouseXPos - this.xPos - this.tankMidpoint;
            dy = mouseYpos - this.yPos - this.tankMidpoint;
        }

        else {
            // Aim at the player
            dx = playerTank.xPos + (playerTank.size / 2) - this.xPos - this.tankMidpoint;
            dy = playerTank.yPos + (playerTank.size / 2) - this.yPos - this.tankMidpoint;
        }

        let theta = Math.atan2(dy, dx);
        if (theta < 0) {
            theta += 2 * Math.PI;
        }

        this.controller.aim(theta);

        // Move the tank
        if(this.controller.up() && this.controller.right()) {
            this.moveNorthEast();
        }

        else if(this.controller.up() && this.controller.left()) {
            this.moveNorthWest();
        }

        else if(this.controller.down() && this.controller.right()) {
            this.moveSouthEast();
        }

        else if(this.controller.down() && this.controller.left()) {
            this.moveSouthWest();
        }

        else if(this.controller.up()) {
            this.moveNorth();
        }

        else if(this.controller.down()) {
            this.moveSouth();
        }

        else if(this.controller.left()) {
            this.moveWest();
        }

        else if(this.controller.right()) {
            this.moveEast();
        }

        // Send position to controller
        this.controller.xPos = this.xPos;
        this.controller.yPos = this.yPos;
    }

    public moveNorth(): void {
        // Loop through all obstacles and check if the tank will collide with any of them. Set the tank y position accordingly.
        let blocked: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.yPos - this.speed < obstacle.yTop + obstacle.height && this.yPos > obstacle.yTop &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yBottom;
                blocked = true;
            }
        }

        if (!blocked) {
            this.yPos = Math.max(this.yPos - this.speed, 0);
        }
    }

    public moveSouth(): void {
        let blocked: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.yPos + this.speed + this.size > obstacle.yTop && this.yPos < obstacle.yTop + obstacle.height &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yTop - this.size;
                blocked = true;
            }
        }
        if (!blocked) {
            this.yPos = Math.min(this.yPos + this.speed, this.canvasHeight - this.size);
        }
    }

    public moveWest(): void {
        let blocked: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.xPos - this.speed < obstacle.xRight && this.xPos > obstacle.xLeft &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xRight;
                blocked = true;
            }
        }
        if (!blocked) {
            this.xPos = Math.max(this.xPos - this.speed, 0);
        }
    }

    public moveEast(): void {
        let blocked: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.xPos + this.speed + this.size > obstacle.xLeft && this.xPos < obstacle.xLeft + obstacle.width &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xLeft - this.size;
                blocked = true;
            }
        }
        if (!blocked) {
            this.xPos = Math.min(this.xPos + this.speed, this.canvasWidth - this.size);
        }
    }

    public moveNorthEast(): void {
        // Loop through all obstacles to check the north and east directions and set the tank position accordingly.
        let blockedNorth: boolean = false;
        let blockedEast: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.yPos - this.speed < obstacle.yTop + obstacle.height && this.yPos > obstacle.yTop &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yBottom;
                blockedNorth = true;
            }
            if (this.xPos + this.speed + this.size > obstacle.xLeft && this.xPos < obstacle.xLeft + obstacle.width &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xLeft - this.size;
                blockedEast = true;
            }
        }

        if (!blockedNorth) {
            this.yPos = Math.max(this.yPos - this.speed, 0);
        }

        if (!blockedEast) {
            this.xPos = Math.min(this.xPos + this.speed, this.canvasWidth - this.size);
        }
    }

    public moveNorthWest(): void {
        let blockedNorth: boolean = false;
        let blockedWest: boolean = false;

        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.yPos - this.speed < obstacle.yTop + obstacle.height && this.yPos > obstacle.yTop &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yBottom;
                blockedNorth = true;
            }
            if (this.xPos - this.speed < obstacle.xRight && this.xPos > obstacle.xLeft &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xRight;
                blockedWest = true;
            }
        }

        if (!blockedNorth) {
            this.yPos = Math.max(this.yPos - this.speed, 0);
        }
        if (!blockedWest) {
            this.xPos = Math.max(this.xPos - this.speed, 0);
        }
    }

    public moveSouthEast(): void {
        let blockedSouth: boolean = false;
        let blockedEast: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.yPos + this.speed + this.size > obstacle.yTop && this.yPos < obstacle.yTop + obstacle.height &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yTop - this.size;
                blockedSouth = true;
            }
            if (this.xPos + this.speed + this.size > obstacle.xLeft && this.xPos < obstacle.xLeft + obstacle.width &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xLeft - this.size;
                blockedEast = true;
            }
        }

        if (!blockedSouth) {
            this.yPos = Math.min(this.yPos + this.speed, this.canvasHeight - this.size);
        }
        if (!blockedEast) {
            this.xPos = Math.min(this.xPos + this.speed, this.canvasWidth - this.size);
        }
    }

    public moveSouthWest(): void {
        let blockedSouth: boolean = false;
        let blockedWest: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.yPos + this.speed + this.size > obstacle.yTop && this.yPos < obstacle.yTop + obstacle.height &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yTop - this.size;
                blockedSouth = true;
            }
            if (this.xPos - this.speed < obstacle.xRight && this.xPos > obstacle.xLeft &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xRight;
                blockedWest = true;
            }
        }
        if (!blockedSouth) {
            this.yPos = Math.min(this.yPos + this.speed, this.canvasHeight - this.size);
        }
        if (!blockedWest) {
            this.xPos = Math.max(this.xPos - this.speed, 0);
        }
    }
}

export class StationaryTank extends Tank {
    constructor(controller: IController, xPos: number, yPos: number, canvasWidth: number, canvasHeight: number, obstacleCanvas: ObstacleCanvas) {
        let fastTankSpeed: number = 0;
        let fastTankSize: number = 30;
        let fastTankColor: string = '#935217';
        super(controller, new NoReticule(), xPos, yPos, fastTankSpeed, fastTankSize, fastTankColor, canvasWidth, canvasHeight, obstacleCanvas);
    }
}

export class StevesTank extends Tank {
    constructor(controller: IController, xPos: number, yPos: number, canvasWidth: number, canvasHeight: number, obstacleCanvas: ObstacleCanvas) {
        let stevesTankSpeed: number = 5;
        let stevesTankSize: number = 30;
        let stevesTankColor: string = '#6384a1';
        super(controller, new AdjustingCustomColorReticule(stevesTankSize, stevesTankColor, canvasWidth), xPos, yPos, stevesTankSpeed, stevesTankSize, stevesTankColor, canvasWidth, canvasHeight, obstacleCanvas);
    }
}

export class LivsTank extends Tank {
    constructor(controller: IController, xPos: number, yPos: number, canvasWidth: number, canvasHeight: number, obstacleCanvas: ObstacleCanvas) {
        let livsTankSpeed: number = 10;
        let livsTankSize: number = 30;
        let livsTankColor: string = 'pink';
        super(controller, new AdjustingCustomColorReticule(livsTankSize, livsTankColor, canvasWidth), xPos, yPos, livsTankSpeed, livsTankSize, livsTankColor, canvasWidth, canvasHeight, obstacleCanvas);
    }
}

export class GregsTank extends Tank {
    constructor(controller: IController, xPos: number, yPos: number, canvasWidth: number, canvasHeight: number, obstacleCanvas: ObstacleCanvas) {
        let gregsTankSpeed: number = 3;
        let gregsTankColor: string = '#3b5232';
        super(controller, new AdjustingCustomColorReticule(controller.tankSize, gregsTankColor, canvasWidth), xPos, yPos, gregsTankSpeed, controller.tankSize, gregsTankColor, canvasWidth, canvasHeight, obstacleCanvas);
    }
}
