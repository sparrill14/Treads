import { KeyStates } from "../utils/KeyStates";
import { Ammunition, BasicAIAmmunition, PlayerAmmunition } from "./Ammunition";
import { ObstacleCanvas } from "./ObstacleCanvas";
import { Reticule, AdjustingCustomColorReticule, NoReticule } from "./Reticule";
import tankFire from "../assets/audio/tankFire.mp3"

export class Tank {
    public xPos: number;
    public yPos: number;
    public xLeft: number;
    public xRight: number;
    public yTop: number;
    public yBottom: number;
    public speed: number;
    public size: number;
    public tankMidpoint: number;
    public color: string;
    public reticule: Reticule;
    public gunBarrellWidth: number = 7;
    public isDestroyed: boolean = false;
    public obstacleCanvas: ObstacleCanvas;
    public twoPi: number = 2 * Math.PI;

    public keyStates: KeyStates = {
        ArrowUp: false,
        ArrowDown: false,
        ArrowLeft: false,
        ArrowRight: false,
        w: false,
        a: false,
        s: false,
        d: false
    }

    public aimAngle: number;
    public aimXPos: number;
    public aimYPos: number;
    public xOffset: number;
    public yOffset: number;
    // public xPos: number;
    // public yPos: number;
    // public tankSize: number;
    public ammunition: Ammunition[] = [];
    public maxAmmunition: number;

    protected canvasWidth: number;
    protected canvasHeight: number;

    constructor(canvas: HTMLCanvasElement, reticule: Reticule, xPos: number, yPos: number, speed: number, size: number, color: string, obstacleCanvas: ObstacleCanvas, ammunition: Ammunition[]) {
        this.reticule = reticule;
        this.xPos = xPos;
        this.yPos = yPos;
        this.xLeft = xPos;
        this.xRight = xPos + size;
        this.yTop = yPos;
        this.yBottom = yPos + size;
        this.speed = speed;
        this.size = size;
        this.color = color;
        this.canvasWidth = canvas.width;
        this.canvasHeight = canvas.height;
        this.obstacleCanvas = obstacleCanvas;
        this.tankMidpoint = this.size / 2;
        this.ammunition = ammunition;
        this.maxAmmunition = ammunition.length

        this.aimAngle = 90;
        const canvasRect: DOMRect = canvas.getBoundingClientRect();
        this.xOffset = canvasRect.left;
        this.yOffset = canvasRect.top;
        // Set the initital awX and Y aim position to the center of the canvas
        this.aimXPos = canvas.width / 2;
        this.aimYPos = canvas.height / 2;
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
        const endX = this.xPos + this.tankMidpoint + (Math.cos(this.aimAngle) * this.size);
        const endY = this.yPos + this.tankMidpoint + (Math.sin(this.aimAngle) * this.size);
        context.beginPath();
        context.moveTo(this.xPos + this.tankMidpoint, this.yPos + this.tankMidpoint);
        context.lineTo(endX, endY);
        context.lineWidth = this.gunBarrellWidth;
        context.stroke();
    }

    public updatePosition(): void {
        // Move the tank
        if(this.up() && this.right()) {
            this.moveNorthEast();
        }

        else if(this.up() && this.left()) {
            this.moveNorthWest();
        }

        else if(this.down() && this.right()) {
            this.moveSouthEast();
        }

        else if(this.down() && this.left()) {
            this.moveSouthWest();
        }

        else if(this.up()) {
            this.moveNorth();
        }

        else if(this.down()) {
            this.moveSouth();
        }

        else if(this.left()) {
            this.moveWest();
        }

        else if(this.right()) {
            this.moveEast();
        }

        this.xLeft = this.xPos;
        this.xRight = this.xPos + this.size;
        this.yTop = this.yPos;
        this.yBottom = this.yPos + this.size;

        // Send position to controller
        this.xPos = this.xPos;
        this.yPos = this.yPos;
    }

    public aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
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
        this.aimAngle = theta;
    }

    public shoot(playerTank: Tank): void {
        return;
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

    public up(): boolean {
        return this.keyStates.ArrowUp || this.keyStates.w;
    }

    public down(): boolean {
        return this.keyStates.ArrowDown || this.keyStates.s;
    }

    public left(): boolean {
        return this.keyStates.ArrowLeft || this.keyStates.a;
    }

    public right(): boolean {
        return this.keyStates.ArrowRight || this.keyStates.d;
    }
}

export class PlayerTank extends Tank {
    public tankFireAudio: HTMLAudioElement;

    constructor(canvas: HTMLCanvasElement, reticule: Reticule, xPos: number, yPos: number, speed: number, size: number, color: string, obstacleCanvas: ObstacleCanvas, ammunition: Ammunition[]) {
        super(canvas, reticule, xPos, yPos, speed, size, color, obstacleCanvas, ammunition)
        this.tankFireAudio = new Audio(tankFire)

        document.addEventListener('keydown', (event: KeyboardEvent) => {
            if (this.keyStates.hasOwnProperty(event.key)) {
                this.keyStates[event.key] = true;
            }
        });

        document.addEventListener('keyup', (event: KeyboardEvent) => {
            if (this.keyStates.hasOwnProperty(event.key)) {
                this.keyStates[event.key] = false;
            }
        });

        canvas.addEventListener('mousemove', (event: MouseEvent) => {
            this.aimXPos = event.clientX - this.xOffset;
            this.aimYPos = event.clientY - this.yOffset;
        });

        canvas.addEventListener('click', (event: MouseEvent) => {
            const availableAmmunitionIndex = this.ammunition.findIndex(ammunition => ammunition.isDestroyed)
            if (availableAmmunitionIndex !== -1) {
                this.tankFireAudio.play()
                this.ammunition[availableAmmunitionIndex] = new PlayerAmmunition(this.xPos + (this.size / 2), this.yPos + (this.size / 2), this.aimAngle, canvas.width, canvas.height, false);
            }
        });
    }

    public aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
        if (this.isDestroyed) {
            return;
        }
        let dy: number;
        let dx: number;
        dx = mouseXPos - this.xPos - this.tankMidpoint;
        dy = mouseYpos - this.yPos - this.tankMidpoint;
        let theta = Math.atan2(dy, dx);
        if (theta < 0) {
            theta += 2 * Math.PI;
        }
        this.aimAngle = theta;
    }

    public shoot(playerTank: Tank): void {
        // Shoot logic is in the constructor
        return;
    }
}

export class EnemyTank extends Tank {
    constructor(canvas: HTMLCanvasElement, reticule: Reticule, xPos: number, yPos: number, speed: number, size: number, color: string, obstacleCanvas: ObstacleCanvas, ammunition: Ammunition[]) {
        super(canvas, reticule, xPos, yPos, speed, size, color, obstacleCanvas, ammunition)
    }
}

export class StationaryTank extends EnemyTank {
    constructor(canvas: HTMLCanvasElement, xPos: number, yPos: number, obstacleCanvas: ObstacleCanvas) {
        let fastTankSpeed: number = 0;
        let fastTankSize: number = 30;
        let fastTankColor: string = '#935217';
        let ammunition: Ammunition[] = [
            new BasicAIAmmunition(0, 0, 0, 0, 0, true),
        ]
        super(canvas, new NoReticule(), xPos, yPos, fastTankSpeed, fastTankSize, fastTankColor, obstacleCanvas, ammunition);
        setInterval(() => {
            if (this.isDestroyed) {
                return;
            }
            const availableAmmunitionIndex = this.ammunition.findIndex(ammunition => ammunition.isDestroyed)
            if (availableAmmunitionIndex !== -1) {
                this.ammunition[availableAmmunitionIndex] = new BasicAIAmmunition(this.xPos + (this.size / 2), this.yPos + (this.size / 2), this.aimAngle, this.canvasWidth, this.canvasHeight, false);
            }
        }, 5000);
    }

    public shoot(playerTank: Tank): void {
        return;
    }

    public aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
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
        this.aimAngle = theta;
    }
}

export class StationaryRandomAimTank extends EnemyTank {
    public aimAngleChangeAmount: number = 0

    constructor(canvas: HTMLCanvasElement, xPos: number, yPos: number, obstacleCanvas: ObstacleCanvas, ammunition: Ammunition[]) {
        let fastTankSpeed: number = 0;
        let fastTankSize: number = 30;
        let fastTankColor: string = '#ffce47';
        super(canvas, new NoReticule(), xPos, yPos, fastTankSpeed, fastTankSize, fastTankColor, obstacleCanvas, ammunition);
    }

    private getAngleChangeAmount(): number {
        let max: number = 360;
        let min: number = -360;
        let randomAmount: number = Math.floor(Math.random() * (max - min + 1)) + min; 
        return randomAmount;
    }

    public shoot(playerTank: Tank): void {
        const availableAmmunitionIndex = this.ammunition.findIndex(ammunition => ammunition.isDestroyed)
        if (availableAmmunitionIndex !== -1) {
            this.ammunition[availableAmmunitionIndex].reload(this.xPos + (this.size / 2), this.yPos + (this.size / 2), this.aimAngle, true, this.canvasWidth, this.canvasHeight);
            let willHitPlayerTank: boolean = this.ammunition[availableAmmunitionIndex].willHitPlayerTank(this.obstacleCanvas, playerTank);
            if (willHitPlayerTank) {
                this.ammunition[availableAmmunitionIndex].isDestroyed = false;
            }
        }
        return;
    }

    public aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
        if (this.isDestroyed) {
            return;
        }
        if (this.aimAngleChangeAmount > 0) {
            this.aimAngle += 0.01
            this.aimAngleChangeAmount -= 1
        }
        else if (this.aimAngleChangeAmount < 0) {
            this.aimAngle -= 0.01
            this.aimAngleChangeAmount += 1
        }
        else {
            this.aimAngleChangeAmount = this.getAngleChangeAmount()
        }
    }
}

export class DefaultPlayerTank extends PlayerTank {
    constructor(canvas: HTMLCanvasElement, xPos: number, yPos: number, obstacleCanvas: ObstacleCanvas) {
        let defaultPlayerTankSpeed: number = 2;
        let defaultPlayerTankSize: number = 30;
        let defaultPlayerTankColor: string = '#6384a1';
        let ammunition: Ammunition[] = [
            new PlayerAmmunition(0, 0, 0, 0, 0, true),
            new PlayerAmmunition(0, 0, 0, 0, 0, true),
            new PlayerAmmunition(0, 0, 0, 0, 0, true),
            new PlayerAmmunition(0, 0, 0, 0, 0, true),
            new PlayerAmmunition(0, 0, 0, 0, 0, true),
        ]
        super(canvas, new AdjustingCustomColorReticule(defaultPlayerTankSize, defaultPlayerTankColor, canvas.width), xPos, yPos, defaultPlayerTankSpeed, defaultPlayerTankSize, defaultPlayerTankColor, obstacleCanvas, ammunition);
    }
}
