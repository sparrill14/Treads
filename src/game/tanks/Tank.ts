import { Ammunition } from '../Ammunition';
import { AudioManager } from '../AudioManager';
import { Bomb } from '../Bomb';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { Reticule } from '../Reticule';

export enum Direction {
	NORTH = 1,
	SOUTH = 2,
	EAST = 3,
	WEST = 4,
	NORTHEAST = 5,
	NORTHWEST = 6,
	SOUTHEAST = 7,
	SOUTHWEST = 8,
	UNKNOWN = 9,
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
	public gunBarrellWidth = 7;
	public isDestroyed = false;
	public obstacleCanvas: ObstacleCanvas;
	public twoPi: number = 2 * Math.PI;
	public lastDirectionMoved: Direction = Direction.UNKNOWN;
	public wasLastMoveBlocked = false;
	public consecutiveDirectionMoves = 0;
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

	constructor(
		canvas: HTMLCanvasElement,
		reticule: Reticule,
		xPos: number,
		yPos: number,
		speed: number,
		size: number,
		color: string,
		obstacleCanvas: ObstacleCanvas,
		ammunition: Ammunition[],
		bombs: Bomb[],
		audioManager: AudioManager
	) {
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
		context.lineJoin = 'bevel';
		context.strokeStyle = 'black';
		context.lineWidth = 2;
		context.strokeRect(this.xPos, this.yPos, this.size, this.size);

		context.beginPath();
		context.arc(this.xPos + this.tankMidpoint, this.yPos + this.tankMidpoint, this.size / 3, 0, this.twoPi);
		context.stroke();

		const endX = this.xPos + this.tankMidpoint + Math.cos(this.aimAngle) * this.size;
		const endY = this.yPos + this.tankMidpoint + Math.sin(this.aimAngle) * this.size;
		context.beginPath();
		context.moveTo(this.xPos + this.tankMidpoint, this.yPos + this.tankMidpoint);
		context.lineTo(endX, endY);
		context.lineWidth = this.gunBarrellWidth;
		context.stroke();
	}

	public updatePosition(playerTank: Tank): void {
		return;
	}

	public aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
		return;
	}

	public shoot(playerTank: Tank): void {
		return;
	}

	public plantBomb(playerTank: Tank): void {
		return;
	}

	public moveInLastDirectionMoved(): void {
		this.moveInCardinalDirection(this.lastDirectionMoved);
	}

	public getRandomDirection<Direction>(): Direction[keyof Direction] {
		const enumValues = Object.keys(Direction)
			.map((n) => Number.parseInt(n))
			.filter((n) => !Number.isNaN(n)) as unknown as Direction[keyof Direction][];
		const randomIndex = Math.floor(Math.random() * enumValues.length);
		const randomEnumValue = enumValues[randomIndex];
		return randomEnumValue;
	}

	public moveInCardinalDirection(direction: Direction): void {
		switch (direction) {
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
				const enumValues = Object.values(Direction).filter((value) => typeof value === 'number') as Direction[];
				const randomIndex = Math.floor(Math.random() * enumValues.length);
				this.moveInCardinalDirection(enumValues[randomIndex]);
				break;
			}
		}
	}

	public moveNorth(): void {
		if (this.lastDirectionMoved == Direction.NORTH) {
			this.consecutiveDirectionMoves += 1;
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.NORTH;
		let blocked = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				this.yPos - this.speed < obstacle.yTop + obstacle.height &&
				this.yPos > obstacle.yTop &&
				obstacle.xLeft < this.xPos + this.size &&
				this.xPos < obstacle.xRight
			) {
				this.yPos = obstacle.yBottom;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			this.yPos = Math.max(this.yPos - this.speed, 0);
		} else {
			this.wasLastMoveBlocked = true;
		}
	}

	public moveSouth(): void {
		if (this.lastDirectionMoved == Direction.SOUTH) {
			this.consecutiveDirectionMoves += 1;
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.SOUTH;
		let blocked = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				this.yPos + this.speed + this.size > obstacle.yTop &&
				this.yPos < obstacle.yTop + obstacle.height &&
				obstacle.xLeft < this.xPos + this.size &&
				this.xPos < obstacle.xRight
			) {
				this.yPos = obstacle.yTop - this.size;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			this.yPos = Math.min(this.yPos + this.speed, this.canvasHeight - this.size);
		} else {
			this.wasLastMoveBlocked = true;
		}
	}

	public moveWest(): void {
		if (this.lastDirectionMoved == Direction.WEST) {
			this.consecutiveDirectionMoves += 1;
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.WEST;
		let blocked = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				this.xPos - this.speed < obstacle.xRight &&
				this.xPos > obstacle.xLeft &&
				obstacle.yTop < this.yPos + this.size &&
				this.yPos < obstacle.yBottom
			) {
				this.xPos = obstacle.xRight;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			this.xPos = Math.max(this.xPos - this.speed, 0);
		} else {
			this.wasLastMoveBlocked = true;
		}
	}

	public moveEast(): void {
		if (this.lastDirectionMoved == Direction.EAST) {
			this.consecutiveDirectionMoves += 1;
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.EAST;
		let blocked = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				this.xPos + this.speed + this.size > obstacle.xLeft &&
				this.xPos < obstacle.xLeft + obstacle.width &&
				obstacle.yTop < this.yPos + this.size &&
				this.yPos < obstacle.yBottom
			) {
				this.xPos = obstacle.xLeft - this.size;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			this.xPos = Math.min(this.xPos + this.speed, this.canvasWidth - this.size);
		} else {
			this.wasLastMoveBlocked = true;
		}
	}

	public moveNorthEast(): void {
		if (this.lastDirectionMoved == Direction.NORTHEAST) {
			this.consecutiveDirectionMoves += 1;
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.NORTHEAST;
		let blockedNorth = false;
		let blockedEast = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				!blockedNorth &&
				this.yPos - this.speed < obstacle.yTop + obstacle.height &&
				this.yPos > obstacle.yTop &&
				obstacle.xLeft < this.xPos + this.size &&
				this.xPos < obstacle.xRight
			) {
				this.yPos = obstacle.yBottom;
				blockedNorth = true;
			}
			if (
				!blockedEast &&
				this.xPos + this.speed + this.size > obstacle.xLeft &&
				this.xPos < obstacle.xLeft + obstacle.width &&
				obstacle.yTop < this.yPos + this.size &&
				this.yPos < obstacle.yBottom
			) {
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
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.NORTHWEST;
		let blockedNorth = false;
		let blockedWest = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				!blockedNorth &&
				this.yPos - this.speed < obstacle.yTop + obstacle.height &&
				this.yPos > obstacle.yTop &&
				obstacle.xLeft < this.xPos + this.size &&
				this.xPos < obstacle.xRight
			) {
				this.yPos = obstacle.yBottom;
				blockedNorth = true;
			}
			if (
				!blockedWest &&
				this.xPos - this.speed < obstacle.xRight &&
				this.xPos > obstacle.xLeft &&
				obstacle.yTop < this.yPos + this.size &&
				this.yPos < obstacle.yBottom
			) {
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
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.SOUTHEAST;
		let blockedSouth = false;
		let blockedEast = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				!blockedSouth &&
				this.yPos + this.speed + this.size > obstacle.yTop &&
				this.yPos < obstacle.yTop + obstacle.height &&
				obstacle.xLeft < this.xPos + this.size &&
				this.xPos < obstacle.xRight
			) {
				this.yPos = obstacle.yTop - this.size;
				blockedSouth = true;
			}
			if (
				!blockedEast &&
				this.xPos + this.speed + this.size > obstacle.xLeft &&
				this.xPos < obstacle.xLeft + obstacle.width &&
				obstacle.yTop < this.yPos + this.size &&
				this.yPos < obstacle.yBottom
			) {
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
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.SOUTHWEST;
		let blockedSouth = false;
		let blockedWest = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				!blockedSouth &&
				this.yPos + this.speed + this.size > obstacle.yTop &&
				this.yPos < obstacle.yTop + obstacle.height &&
				obstacle.xLeft < this.xPos + this.size &&
				this.xPos < obstacle.xRight
			) {
				this.yPos = obstacle.yTop - this.size;
				blockedSouth = true;
			}
			if (
				!blockedWest &&
				this.xPos - this.speed < obstacle.xRight &&
				this.xPos > obstacle.xLeft &&
				obstacle.yTop < this.yPos + this.size &&
				this.yPos < obstacle.yBottom
			) {
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
}
