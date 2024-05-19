import { KeyStates } from "../utils/KeyStates";
import { Ammunition, BasicAIAmmunition, PlayerAmmunition } from "./Ammunition";
import { ObstacleCanvas } from "./ObstacleCanvas";
import { Reticule, AdjustingCustomColorReticule, NoReticule } from "./Reticule";
import { NavigationGrid, Node } from "./NavigationGrid";
import { Bomb, PlayerBomb } from "./Bomb";
import { AudioFile, AudioManager } from "./AudioManager";

export enum Direction {
    NORTH = 1,
    SOUTH = 2,
    EAST = 3,
    WEST = 4,
    NORTHEAST = 5,
    NORTHWEST = 6,
    SOUTHEAST = 7,
    SOUTHWEST = 8,
    UNKNOWN = 9
}

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
    public lastDirectionMoved: Direction = Direction.UNKNOWN;
    public wasLastMoveBlocked: boolean = false;
    public consecutiveDirectionMoves: number = 0;
    public audioManager: AudioManager;
    public aimAngle: number;
    public aimXPos: number;
    public aimYPos: number;
    public xOffset: number;
    public yOffset: number;
    public ammunition: Ammunition[] = [];
    public maxAmmunition: number;
    public bombs: Bomb[] = [];
    public maxBombs: number;
    public canvasWidth: number;
    public canvasHeight: number;

    public keyStates: KeyStates = {
        ArrowUp: false,
        ArrowDown: false,
        ArrowLeft: false,
        ArrowRight: false,
        w: false,
        a: false,
        s: false,
        d: false,
        W: false,
        A: false,
        S: false,
        D: false
    }

    constructor(canvas: HTMLCanvasElement, reticule: Reticule, xPos: number, yPos: number, speed: number, size: number, color: string, obstacleCanvas: ObstacleCanvas, ammunition: Ammunition[], bombs: Bomb[], audioManager: AudioManager) {
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
        this.maxAmmunition = ammunition.length;
        this.bombs = bombs;
        this.maxBombs = bombs.length;
        this.audioManager = audioManager;

        this.aimAngle = 90;
        const canvasRect: DOMRect = canvas.getBoundingClientRect();
        this.xOffset = canvasRect.left;
        this.yOffset = canvasRect.top;
        // Set the initital X and Y aim position to the center of the canvas
        this.aimXPos = canvas.width / 2;
        this.aimYPos = canvas.height / 2;
    }

    public draw(context: CanvasRenderingContext2D): void {
        if (this.isDestroyed) {
            return;
        }

        context.fillStyle = this.color;
        context.fillRect(this.xPos, this.yPos, this.size, this.size);

        context.setLineDash([]);
        context.lineJoin = 'bevel'
        context.strokeStyle = 'black';
        context.lineWidth = 2;
        context.strokeRect(this.xPos, this.yPos, this.size, this.size);

        context.beginPath();
        context.arc(this.xPos + this.tankMidpoint, this.yPos + this.tankMidpoint, this.size / 3, 0, this.twoPi);
        context.stroke();

        const endX = this.xPos + this.tankMidpoint + (Math.cos(this.aimAngle) * this.size);
        const endY = this.yPos + this.tankMidpoint + (Math.sin(this.aimAngle) * this.size);
        context.beginPath();
        context.moveTo(this.xPos + this.tankMidpoint, this.yPos + this.tankMidpoint);
        context.lineTo(endX, endY);
        context.lineWidth = this.gunBarrellWidth;
        context.stroke();
    }

    public updatePosition(playerTank: Tank): void {
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
        return;
    }

    public plantBomb(playerTank: Tank): void {
        if (!this.isDestroyed) {
            const availableBombIndex = this.bombs.findIndex(bomb => bomb.isDestroyed)
            if (availableBombIndex !== -1) {
                this.bombs[availableBombIndex].xPos = this.xPos + (this.size / 2);
                this.bombs[availableBombIndex].yPos = this.yPos + (this.size / 2);
                this.bombs[availableBombIndex].isDestroyed = false;
                this.bombs[availableBombIndex].setFuse();
            }
        }
        return;
    }

    public moveInLastDirectionMoved(): void {
        this.moveInCardinalDirection(this.lastDirectionMoved);
    }

    public getRandomDirection<Direction>(): Direction[keyof Direction] {
        const enumValues = Object.keys(Direction)
          .map(n => Number.parseInt(n))
          .filter(n => !Number.isNaN(n)) as unknown as Direction[keyof Direction][]
        const randomIndex = Math.floor(Math.random() * enumValues.length)
        const randomEnumValue = enumValues[randomIndex]
        return randomEnumValue;
    }

    public moveInCardinalDirection(direction: Direction): void {
        switch(direction) {
            case Direction.NORTH: {
                this.moveNorth();
                break;
            }
            case Direction.SOUTH: {
                this.moveSouth();
                break;
            }
            case Direction.EAST: {
                this.moveEast();
                break;
            }
            case Direction.WEST: {
                this.moveWest();
                break;
            }
            case Direction.NORTHEAST: {
                this.moveNorthEast();
                break;
            }
            case Direction.NORTHWEST: {
                this.moveNorthWest();
                break;
            }
            case Direction.SOUTHEAST: {
                this.moveSouthEast();
                break;
            }
            case Direction.SOUTHWEST: {
                this.moveSouthWest();
                break;
            }
            default: {
                const enumValues = Object.values(Direction).filter(value => typeof value === "number") as Direction[];
                const randomIndex = Math.floor(Math.random() * enumValues.length);
                this.moveInCardinalDirection(enumValues[randomIndex]);
                break;
            }
        }
    }

    public moveNorth(): void {
        if (this.lastDirectionMoved == Direction.NORTH) {
            this.consecutiveDirectionMoves += 1;
        }
        else {
            this.consecutiveDirectionMoves = 0;
        }
        this.lastDirectionMoved = Direction.NORTH
        let blocked: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.yPos - this.speed < obstacle.yTop + obstacle.height && this.yPos > obstacle.yTop &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yBottom;
                blocked = true;
                break;
            }
        }
        if (!blocked) {
            this.yPos = Math.max(this.yPos - this.speed, 0);
        }
        else {
            this.wasLastMoveBlocked = true;
        }
    }

    public moveSouth(): void {
        if (this.lastDirectionMoved == Direction.SOUTH) {
            this.consecutiveDirectionMoves += 1;
        }
        else {
            this.consecutiveDirectionMoves = 0;
        }
        this.lastDirectionMoved = Direction.SOUTH
        let blocked: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.yPos + this.speed + this.size > obstacle.yTop && this.yPos < obstacle.yTop + obstacle.height &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yTop - this.size;
                blocked = true;
                break;
            }
        }
        if (!blocked) {
            this.yPos = Math.min(this.yPos + this.speed, this.canvasHeight - this.size);
        }
        else {
            this.wasLastMoveBlocked = true;
        }
    }

    public moveWest(): void {
        if (this.lastDirectionMoved == Direction.WEST) {
            this.consecutiveDirectionMoves += 1;
        }
        else {
            this.consecutiveDirectionMoves = 0;
        }
        this.lastDirectionMoved = Direction.WEST
        let blocked: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.xPos - this.speed < obstacle.xRight && this.xPos > obstacle.xLeft &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xRight;
                blocked = true;
                break;
            }
        }
        if (!blocked) {
            this.xPos = Math.max(this.xPos - this.speed, 0);
        }
        else {
            this.wasLastMoveBlocked = true;
        }
    }

    public moveEast(): void {
        if (this.lastDirectionMoved == Direction.EAST) {
            this.consecutiveDirectionMoves += 1;
        }
        else {
            this.consecutiveDirectionMoves = 0;
        }
        this.lastDirectionMoved = Direction.EAST
        let blocked: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (this.xPos + this.speed + this.size > obstacle.xLeft && this.xPos < obstacle.xLeft + obstacle.width &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xLeft - this.size;
                blocked = true;
                break;
            }
        }
        if (!blocked) {
            this.xPos = Math.min(this.xPos + this.speed, this.canvasWidth - this.size);
        }
        else {
            this.wasLastMoveBlocked = true;
        }
    }

    public moveNorthEast(): void {
        if (this.lastDirectionMoved == Direction.NORTHEAST) {
            this.consecutiveDirectionMoves += 1;
        }
        else {
            this.consecutiveDirectionMoves = 0;
        }
        this.lastDirectionMoved = Direction.NORTHEAST
        let blockedNorth: boolean = false;
        let blockedEast: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (!blockedNorth && this.yPos - this.speed < obstacle.yTop + obstacle.height && this.yPos > obstacle.yTop &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yBottom;
                blockedNorth = true;
            }
            if (!blockedEast && this.xPos + this.speed + this.size > obstacle.xLeft && this.xPos < obstacle.xLeft + obstacle.width &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xLeft - this.size;
                blockedEast = true;
            }
        }
        if (blockedNorth && blockedEast) {
            this.wasLastMoveBlocked = true;
        }
        if (!blockedNorth) {
            this.yPos = Math.max(this.yPos - this.speed, 0);
        }
        if (!blockedEast) {
            this.xPos = Math.min(this.xPos + this.speed, this.canvasWidth - this.size);
        }
    }

    public moveNorthWest(): void {
        if (this.lastDirectionMoved == Direction.NORTHWEST) {
            this.consecutiveDirectionMoves += 1;
        }
        else {
            this.consecutiveDirectionMoves = 0;
        }
        this.lastDirectionMoved = Direction.NORTHWEST
        let blockedNorth: boolean = false;
        let blockedWest: boolean = false;

        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (!blockedNorth && this.yPos - this.speed < obstacle.yTop + obstacle.height && this.yPos > obstacle.yTop &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yBottom;
                blockedNorth = true;
            }
            if (!blockedWest && this.xPos - this.speed < obstacle.xRight && this.xPos > obstacle.xLeft &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xRight;
                blockedWest = true;
            }
        }
        if (blockedNorth && blockedWest) {
            this.wasLastMoveBlocked = true;
        }
        if (!blockedNorth) {
            this.yPos = Math.max(this.yPos - this.speed, 0);
        }
        if (!blockedWest) {
            this.xPos = Math.max(this.xPos - this.speed, 0);
        }
    }

    public moveSouthEast(): void {
        if (this.lastDirectionMoved == Direction.SOUTHEAST) {
            this.consecutiveDirectionMoves += 1;
        }
        else {
            this.consecutiveDirectionMoves = 0;
        }
        this.lastDirectionMoved = Direction.SOUTHEAST
        let blockedSouth: boolean = false;
        let blockedEast: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (!blockedSouth && this.yPos + this.speed + this.size > obstacle.yTop && this.yPos < obstacle.yTop + obstacle.height &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yTop - this.size;
                blockedSouth = true;
            }
            if (!blockedEast && this.xPos + this.speed + this.size > obstacle.xLeft && this.xPos < obstacle.xLeft + obstacle.width &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xLeft - this.size;
                blockedEast = true;
            }
        }
        if (blockedSouth && blockedEast) {
            this.wasLastMoveBlocked = true;
        }
        if (!blockedSouth) {
            this.yPos = Math.min(this.yPos + this.speed, this.canvasHeight - this.size);
        }
        if (!blockedEast) {
            this.xPos = Math.min(this.xPos + this.speed, this.canvasWidth - this.size);
        }
    }

    public moveSouthWest(): void {
        if (this.lastDirectionMoved == Direction.SOUTHWEST) {
            this.consecutiveDirectionMoves += 1;
        }
        else {
            this.consecutiveDirectionMoves = 0;
        }
        this.lastDirectionMoved = Direction.SOUTHWEST
        let blockedSouth: boolean = false;
        let blockedWest: boolean = false;
        for (let i = 0; i < this.obstacleCanvas.obstacles.length; i++) {
            const obstacle = this.obstacleCanvas.obstacles[i];
            if (!blockedSouth && this.yPos + this.speed + this.size > obstacle.yTop && this.yPos < obstacle.yTop + obstacle.height &&
                obstacle.xLeft < this.xPos + this.size && this.xPos < obstacle.xRight) {
                this.yPos = obstacle.yTop - this.size;
                blockedSouth = true;
            }
            if (!blockedWest && this.xPos - this.speed < obstacle.xRight && this.xPos > obstacle.xLeft &&
                obstacle.yTop < this.yPos + this.size && this.yPos < obstacle.yBottom) {
                this.xPos = obstacle.xRight;
                blockedWest = true;
            }
        }
        if (blockedSouth && blockedWest) {
            this.wasLastMoveBlocked = true;
        }
        if (!blockedSouth) {
            this.yPos = Math.min(this.yPos + this.speed, this.canvasHeight - this.size);
        }
        if (!blockedWest) {
            this.xPos = Math.max(this.xPos - this.speed, 0);
        }
    }

    public up(): boolean {
        return this.keyStates.ArrowUp || this.keyStates.w || this.keyStates.W;
    }

    public down(): boolean {
        return this.keyStates.ArrowDown || this.keyStates.s || this.keyStates.S;
    }

    public left(): boolean {
        return this.keyStates.ArrowLeft || this.keyStates.a || this.keyStates.A;
    }

    public right(): boolean {
        return this.keyStates.ArrowRight || this.keyStates.d || this.keyStates.D;
    }
}

export class PlayerTank extends Tank {
    constructor(canvas: HTMLCanvasElement, reticule: Reticule, xPos: number, yPos: number, speed: number, size: number, color: string, obstacleCanvas: ObstacleCanvas, ammunition: Ammunition[], bombs: Bomb[], audioManager: AudioManager) {
        super(canvas, reticule, xPos, yPos, speed, size, color, obstacleCanvas, ammunition, bombs, audioManager)

        document.addEventListener('keydown', (event: KeyboardEvent) => {
            if (this.keyStates.hasOwnProperty(event.key)) {
                this.keyStates[event.key] = true;
            }
        });

        document.addEventListener('keydown', (event: KeyboardEvent) => {
            if (event.code === 'Space') {
                this.plantBomb(this);
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
            this.shoot(this);
        });
    }

    public override aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
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

    public override shoot(playerTank: Tank): void {
        if (!this.isDestroyed) {
            const availableAmmunitionIndex = this.ammunition.findIndex(ammunition => ammunition.isDestroyed)
            if (availableAmmunitionIndex !== -1) {
                this.audioManager.play(AudioFile.TANK_FIRE);
                this.ammunition[availableAmmunitionIndex] = new PlayerAmmunition(this.xPos + (this.size / 2), this.yPos + (this.size / 2), this.aimAngle, this.canvasWidth, this.canvasHeight, false, this.audioManager);
            }
        }
        return;
    }
}

export class EnemyTank extends Tank {
    constructor(canvas: HTMLCanvasElement, reticule: Reticule, xPos: number, yPos: number, speed: number, size: number, color: string, obstacleCanvas: ObstacleCanvas, ammunition: Ammunition[], bombs: Bomb[], audioManager: AudioManager) {
        super(canvas, reticule, xPos, yPos, speed, size, color, obstacleCanvas, ammunition, bombs, audioManager)
    }
}

export class StationaryTank extends EnemyTank {
    minTimeBetweenShotsMS: number = 5000;
    canTakeShot: boolean = true;

    constructor(canvas: HTMLCanvasElement, xPos: number, yPos: number, obstacleCanvas: ObstacleCanvas, audioManager: AudioManager) {
        let fastTankSpeed: number = 0;
        let fastTankSize: number = 30;
        let fastTankColor: string = '#5784ba';
        let ammunition: Ammunition[] = [
            new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager),
        ]
        let bombs: Bomb[] = [];
        super(canvas, new NoReticule(), xPos, yPos, fastTankSpeed, fastTankSize, fastTankColor, obstacleCanvas, ammunition, bombs, audioManager);
    }

    public override updatePosition(playerTank: Tank): void {
        return; 
    }

    public override shoot(): void {
        if (!this.canTakeShot || this.isDestroyed) {
            return;
        }
        const availableAmmunitionIndex = this.ammunition.findIndex(ammunition => ammunition.isDestroyed)
        if (availableAmmunitionIndex !== -1) {
            this.ammunition[availableAmmunitionIndex] = new BasicAIAmmunition(this.xPos + (this.size / 2), this.yPos + (this.size / 2), this.aimAngle, this.canvasWidth, this.canvasHeight, false, this.audioManager);
            this.canTakeShot = false;
            setTimeout(() => {
                this.canTakeShot = true;
            }, this.minTimeBetweenShotsMS)
        }
        return;
    }

    public override aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
        if (this.isDestroyed) {
            return;
        }

        let dy: number;
        let dx: number;
        dx = playerTank.xPos + (playerTank.size / 2) - this.xPos - this.tankMidpoint;
        dy = playerTank.yPos + (playerTank.size / 2) - this.yPos - this.tankMidpoint;
        let theta = Math.atan2(dy, dx);
        if (theta < 0) {
            theta += 2 * Math.PI;
        }
        this.aimAngle = theta;
    }
}

export class StationaryRandomAimTank extends EnemyTank {
    public aimAngleChangeAmount: number = 0

    constructor(canvas: HTMLCanvasElement, xPos: number, yPos: number, obstacleCanvas: ObstacleCanvas, ammunition: Ammunition[], audioManager: AudioManager) {
        let fastTankSpeed: number = 0;
        let fastTankSize: number = 30;
        let fastTankColor: string = '#ebe1b9';
        let bombs: Bomb[] = [];
        super(canvas, new NoReticule(), xPos, yPos, fastTankSpeed, fastTankSize, fastTankColor, obstacleCanvas, ammunition, bombs, audioManager);
    }

    private getAngleChangeAmount(): number {
        let max: number = 360;
        let min: number = -360;
        let randomAmount: number = Math.floor(Math.random() * (max - min + 1)) + min; 
        return randomAmount;
    }

    public override updatePosition(playerTank: Tank): void {
        return;
    }

    public override shoot(playerTank: Tank): void {
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

    public override aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
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

export class SimpleMovingTank extends EnemyTank {
    public aimAngleChangeAmount: number = 0
    public navigationGrid: NavigationGrid;
    public aggressionFactor: number = 15; // Distance tank should maintain from its target
    public currentNode: Node;
    public path: Node[] | null = []
    public pathRecaculationInterval: number = 60;
    public drawNavigationGrid: boolean = false;

    constructor(canvas: HTMLCanvasElement, xPos: number, yPos: number, obstacleCanvas: ObstacleCanvas, ammunition: Ammunition[], bombs: Bomb[], navigationGrid: NavigationGrid, audioManager: AudioManager) {
        let simpleMovingTankSpeed: number = 1.2;
        let simpleMovingTankSize: number = 30;
        let simpleMovingTankColor: string = '#fd8a8a';
        super(canvas, new NoReticule(), xPos, yPos, simpleMovingTankSpeed, simpleMovingTankSize, simpleMovingTankColor, obstacleCanvas, ammunition, bombs, audioManager);
        this.navigationGrid = navigationGrid
        this.currentNode = this.navigationGrid.getNodeFromTank(this)
    }

    public override draw(context: CanvasRenderingContext2D): void {
        if (this.drawNavigationGrid) {
            context.lineWidth = 1;
            for (let i = 0; i <= this.navigationGrid.gridYLength; i++) {
                context.fillStyle = "blue"
                context.beginPath();
                context.moveTo(0, i * this.navigationGrid.gridCellWidth);
                context.lineTo(this.navigationGrid.gridXLength * this.navigationGrid.gridCellWidth, i * this.navigationGrid.gridCellWidth);
                context.stroke();
            }
            for (let j = 0; j <= this.navigationGrid.gridXLength; j++) {
                context.fillStyle = "blue"
                context.beginPath();
                context.moveTo(j * this.navigationGrid.gridCellWidth, 0);
                context.lineTo(j * this.navigationGrid.gridCellWidth, this.navigationGrid.gridXLength * this.navigationGrid.gridCellWidth);
                context.stroke();
            }
            context.fillStyle = this.color;
            this.path?.forEach((value: Node, index:  number, array: Node[])=> {
                context.beginPath();
                context.arc(value.x * this.navigationGrid.gridCellWidth + this.navigationGrid.gridCellWidth / 2, value.y * this.navigationGrid.gridCellWidth + this.navigationGrid.gridCellWidth / 2, 5, 0, 2 * Math.PI);
                context.fill();
            })
        }
        super.draw(context);
    }

    public override updatePosition(playerTank: Tank): void {
        this.pathRecaculationInterval -= 1;
        if (this.path == null || this.path.length == 0 || this.pathRecaculationInterval == 0) {
            this.navigationGrid.reset();
            let startNode: Node = this.navigationGrid.getNodeFromTank(this);
            let targetNode: Node = this.navigationGrid.getNodeFromTank(playerTank);
            let destinationNode: Node = this.navigationGrid.getRandomNodeInRadiusOfTarget(targetNode, this.aggressionFactor);
            this.path = this.navigationGrid.aStar(startNode, destinationNode);
            this.pathRecaculationInterval = 60;
            if (this.path == null) {
                console.log(`Path is null`)
            }
        } else {
            this.currentNode = this.navigationGrid.getNodeFromTank(this);
            let dx = this.path[0].x - this.currentNode.x;
            let dy = this.path[0].y - this.currentNode.y;

            if (this.wasLastMoveBlocked && this.consecutiveDirectionMoves > 2) {
                let randomDirection: Direction = this.getRandomDirection();
                this.moveInCardinalDirection(randomDirection);
                this.consecutiveDirectionMoves = 0;
                let randomNumber: number = Math.random();
                if (dx === 1 && dy === 0) {
                    if (randomNumber < 0.5) {
                        this.moveSouthEast();
                    }
                    else {
                        this.moveNorthEast();
                    }
                } else if (dx === -1 && dy === 0) {
                    if (randomNumber < 0.5) {
                        this.moveNorthWest();
                    }
                    else {
                        this.moveSouthWest();
                    }
                } else if (dx === 0 && dy === 1) {
                    if (randomNumber < 0.5) {
                        this.moveSouthEast();
                    }
                    else {
                        this.moveSouthWest();
                    }
                } else if (dx === 0 && dy === -1) {
                    if (randomNumber < 0.5) {
                        this.moveNorthWest();
                    }
                    else {
                        this.moveNorthEast();
                    }
                } else if (dx === 1 && dy === 1) {
                    if (randomNumber < 0.5) {
                        this.moveSouth();
                    }
                    else {
                        this.moveEast();
                    }
                } else if (dx === 1 && dy === -1) {
                    if (randomNumber < 0.5) {
                        this.moveNorth();
                    }
                    else {
                        this.moveEast();
                    }
                } else if (dx === -1 && dy === 1) {
                    if (randomNumber < 0.5) {
                        this.moveSouth();
                    }
                    else {
                        this.moveWest();
                    }
                } else if (dx === -1 && dy === -1) {
                    if (randomNumber < 0.5) {
                        this.moveNorth();
                    }
                    else {
                        this.moveWest();
                    }
                }
            }
            else {
                if (dx === 1 && dy === 0) {
                    this.moveEast();
                } else if (dx === -1 && dy === 0) {
                    this.moveWest();
                } else if (dx === 0 && dy === 1) {
                    this.moveSouth();
                } else if (dx === 0 && dy === -1) {
                    this.moveNorth();
                } else if (dx === 1 && dy === 1) {
                    this.moveSouthEast();
                } else if (dx === 1 && dy === -1) {
                    this.moveNorthEast();
                } else if (dx === -1 && dy === 1) {
                    this.moveSouthWest();
                } else if (dx === -1 && dy === -1) {
                    this.moveNorthWest();
                }
            }

            if(this.path[0].x == this.currentNode.x && this.path[0].y == this.currentNode.y) {
                this.path.splice(0, 1);
            }
        }

        this.xLeft = this.xPos;
        this.xRight = this.xPos + this.size;
        this.yTop = this.yPos;
        this.yBottom = this.yPos + this.size;
    }

    public override shoot(playerTank: Tank): void {
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

    public override aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
        if (this.isDestroyed) {
            return;
        }

        let dy: number;
        let dx: number;
        dx = playerTank.xPos + (playerTank.size / 2) - this.xPos - this.tankMidpoint;
        dy = playerTank.yPos + (playerTank.size / 2) - this.yPos - this.tankMidpoint;
        let theta = Math.atan2(dy, dx);
        if (theta < 0) {
            theta += 2 * Math.PI;
        }
        this.aimAngle = theta;
    }
}

export class BomberTank extends EnemyTank {
    public aimAngleChangeAmount: number = 0
    public navigationGrid: NavigationGrid;
    public aggressionFactor: number = 5; // Distance tank should maintain from its target
    public currentNode: Node;
    public path: Node[] | null = []
    public pathRecaculationInterval: number = 60;
    public drawNavigationGrid: boolean = false;

    public minTimeBetweenShotsMS: number = 20000;
    public timeBetweenShotsIsElapsed: boolean = true;
    public minTimeBetweenBombPlantsMS: number = 1000;
    public timeBetweenPlantsIsElapsed: boolean = true;

    constructor(canvas: HTMLCanvasElement, xPos: number, yPos: number, obstacleCanvas: ObstacleCanvas, ammunition: Ammunition[], bombs: Bomb[], navigationGrid: NavigationGrid, audioManager: AudioManager) {
        let simpleMovingTankSpeed: number = 2;
        let simpleMovingTankSize: number = 30;
        let simpleMovingTankColor: string = 'yellow';
        super(canvas, new NoReticule(), xPos, yPos, simpleMovingTankSpeed, simpleMovingTankSize, simpleMovingTankColor, obstacleCanvas, ammunition, bombs, audioManager);
        this.navigationGrid = navigationGrid
        this.currentNode = this.navigationGrid.getNodeFromTank(this)
    }

    public override draw(context: CanvasRenderingContext2D): void {
        if (this.drawNavigationGrid) {
            context.lineWidth = 1;
            for (let i = 0; i <= this.navigationGrid.gridYLength; i++) {
                context.fillStyle = "blue"
                context.beginPath();
                context.moveTo(0, i * this.navigationGrid.gridCellWidth);
                context.lineTo(this.navigationGrid.gridXLength * this.navigationGrid.gridCellWidth, i * this.navigationGrid.gridCellWidth);
                context.stroke();
            }
            for (let j = 0; j <= this.navigationGrid.gridXLength; j++) {
                context.fillStyle = "blue"
                context.beginPath();
                context.moveTo(j * this.navigationGrid.gridCellWidth, 0);
                context.lineTo(j * this.navigationGrid.gridCellWidth, this.navigationGrid.gridXLength * this.navigationGrid.gridCellWidth);
                context.stroke();
            }
            context.fillStyle = this.color;
            this.path?.forEach((value: Node, index:  number, array: Node[])=> {
                context.beginPath();
                context.arc(value.x * this.navigationGrid.gridCellWidth + this.navigationGrid.gridCellWidth / 2, value.y * this.navigationGrid.gridCellWidth + this.navigationGrid.gridCellWidth / 2, 5, 0, 2 * Math.PI);
                context.fill();
            })
        }
        super.draw(context);
    }

    public override updatePosition(playerTank: Tank): void {
        this.pathRecaculationInterval -= 1;
        if (this.path == null || this.path.length == 0 || this.pathRecaculationInterval == 0) {
            this.navigationGrid.reset();
            let startNode: Node = this.navigationGrid.getNodeFromTank(this);
            let targetNode: Node = this.navigationGrid.getNodeFromTank(playerTank);
            let destinationNode: Node = this.navigationGrid.getRandomNodeInRadiusOfTarget(targetNode, this.aggressionFactor);
            this.path = this.navigationGrid.aStar(startNode, destinationNode);
            this.pathRecaculationInterval = 60;
            if (this.path == null) {
                console.log(`Path is null`)
            }
        } else {
            this.currentNode = this.navigationGrid.getNodeFromTank(this);
            let dx = this.path[0].x - this.currentNode.x;
            let dy = this.path[0].y - this.currentNode.y;

            if (this.wasLastMoveBlocked && this.consecutiveDirectionMoves > 2) {
                let randomDirection: Direction = this.getRandomDirection();
                this.moveInCardinalDirection(randomDirection);
                this.consecutiveDirectionMoves = 0;
                let randomNumber: number = Math.random();
                if (dx === 1 && dy === 0) {
                    if (randomNumber < 0.5) {
                        this.moveSouthEast();
                    }
                    else {
                        this.moveNorthEast();
                    }
                } else if (dx === -1 && dy === 0) {
                    if (randomNumber < 0.5) {
                        this.moveNorthWest();
                    }
                    else {
                        this.moveSouthWest();
                    }
                } else if (dx === 0 && dy === 1) {
                    if (randomNumber < 0.5) {
                        this.moveSouthEast();
                    }
                    else {
                        this.moveSouthWest();
                    }
                } else if (dx === 0 && dy === -1) {
                    if (randomNumber < 0.5) {
                        this.moveNorthWest();
                    }
                    else {
                        this.moveNorthEast();
                    }
                } else if (dx === 1 && dy === 1) {
                    if (randomNumber < 0.5) {
                        this.moveSouth();
                    }
                    else {
                        this.moveEast();
                    }
                } else if (dx === 1 && dy === -1) {
                    if (randomNumber < 0.5) {
                        this.moveNorth();
                    }
                    else {
                        this.moveEast();
                    }
                } else if (dx === -1 && dy === 1) {
                    if (randomNumber < 0.5) {
                        this.moveSouth();
                    }
                    else {
                        this.moveWest();
                    }
                } else if (dx === -1 && dy === -1) {
                    if (randomNumber < 0.5) {
                        this.moveNorth();
                    }
                    else {
                        this.moveWest();
                    }
                }
            }
            else {
                if (dx === 1 && dy === 0) {
                    this.moveEast();
                } else if (dx === -1 && dy === 0) {
                    this.moveWest();
                } else if (dx === 0 && dy === 1) {
                    this.moveSouth();
                } else if (dx === 0 && dy === -1) {
                    this.moveNorth();
                } else if (dx === 1 && dy === 1) {
                    this.moveSouthEast();
                } else if (dx === 1 && dy === -1) {
                    this.moveNorthEast();
                } else if (dx === -1 && dy === 1) {
                    this.moveSouthWest();
                } else if (dx === -1 && dy === -1) {
                    this.moveNorthWest();
                }
            }

            if(this.path[0].x == this.currentNode.x && this.path[0].y == this.currentNode.y) {
                this.path.splice(0, 1);
            }
        }

        this.xLeft = this.xPos;
        this.xRight = this.xPos + this.size;
        this.yTop = this.yPos;
        this.yBottom = this.yPos + this.size;
    }

    public override plantBomb(playerTank: Tank): void {
        if (this.timeBetweenPlantsIsElapsed && !this.isDestroyed) {
            const availableBombIndex = this.bombs.findIndex(bomb => bomb.isDestroyed)
            if (availableBombIndex !== -1) {
                this.bombs[availableBombIndex].xPos = this.xPos + (this.size / 2);
                this.bombs[availableBombIndex].yPos = this.yPos + (this.size / 2);
                let willHitPlayerTank: boolean = this.bombs[availableBombIndex].isPointInsideBlastRadius(playerTank.xPos + playerTank.tankMidpoint, playerTank.yPos + playerTank.tankMidpoint)
                if (willHitPlayerTank) {
                    this.bombs[availableBombIndex].isDestroyed = false;
                    this.bombs[availableBombIndex].setFuse();
                    this.timeBetweenPlantsIsElapsed = false;
                    setTimeout(() => {
                        this.timeBetweenPlantsIsElapsed = true;
                    }, this.minTimeBetweenBombPlantsMS)
                }
            }
        }
        return;
    }

    public override shoot(playerTank: Tank): void {
        if (this.timeBetweenShotsIsElapsed && !this.isDestroyed) {
            const availableAmmunitionIndex = this.ammunition.findIndex(ammunition => ammunition.isDestroyed)
            if (availableAmmunitionIndex !== -1) {
                this.ammunition[availableAmmunitionIndex].reload(this.xPos + (this.size / 2), this.yPos + (this.size / 2), this.aimAngle, true, this.canvasWidth, this.canvasHeight);
                let willHitPlayerTank: boolean = this.ammunition[availableAmmunitionIndex].willHitPlayerTank(this.obstacleCanvas, playerTank);
                if (willHitPlayerTank) {
                    this.ammunition[availableAmmunitionIndex].isDestroyed = false;
                    this.timeBetweenShotsIsElapsed = false;
                    setTimeout(() => {
                        this.timeBetweenShotsIsElapsed = true;
                    }, this.minTimeBetweenShotsMS)
                }
            }
        }
        return;
    }

    public override aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
        if (this.isDestroyed) {
            return;
        }

        let dy: number;
        let dx: number;
        dx = playerTank.xPos + (playerTank.size / 2) - this.xPos - this.tankMidpoint;
        dy = playerTank.yPos + (playerTank.size / 2) - this.yPos - this.tankMidpoint;
        let theta = Math.atan2(dy, dx);
        if (theta < 0) {
            theta += 2 * Math.PI;
        }
        this.aimAngle = theta;
    }
}

export class DefaultPlayerTank extends PlayerTank {
    constructor(canvas: HTMLCanvasElement, xPos: number, yPos: number, obstacleCanvas: ObstacleCanvas, audioManager: AudioManager) {
        let defaultPlayerTankSpeed: number = 2;
        let defaultPlayerTankSize: number = 30;
        let defaultPlayerTankColor: string = '#6384a1';
        let ammunition: Ammunition[] = [
            new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
            new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
            new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
            new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
            new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
        ]
        let bombs: Bomb[] = [
            new PlayerBomb(0, 0, true, audioManager),
            new PlayerBomb(0, 0, true, audioManager)
        ];
        super(canvas, new AdjustingCustomColorReticule(defaultPlayerTankSize, defaultPlayerTankColor, canvas.width), xPos, yPos, defaultPlayerTankSpeed, defaultPlayerTankSize, defaultPlayerTankColor, obstacleCanvas, ammunition, bombs, audioManager);
    }
}
