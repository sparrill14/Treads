import { Ammunition } from '../Ammunition';
import { AudioManager } from '../AudioManager';
import { Bomb } from '../Bomb';
import { BombFragment } from '../BombFragment';
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
	public xPosition: number;
	public yPosition: number;
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
	public isExploding = false;
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
	public gunBarrellEndX: number;
	public gunBarrellEndY: number;
	public ammunition: Ammunition[] = [];
	public maxAmmunition: number;
	public bombs: Bomb[] = [];
	public maxBombs: number;
	public canvasWidth: number;
	public canvasHeight: number;
	private fragments: BombFragment[];

	constructor(
		canvas: HTMLCanvasElement,
		reticule: Reticule,
		xPosition: number,
		yPosition: number,
		speed: number,
		size: number,
		color: string,
		obstacleCanvas: ObstacleCanvas,
		ammunition: Ammunition[],
		bombs: Bomb[],
		audioManager: AudioManager
	) {
		this.reticule = reticule;
		this.xPosition = xPosition;
		this.yPosition = yPosition;
		this.xLeft = xPosition;
		this.xRight = xPosition + size;
		this.yTop = yPosition;
		this.yBottom = yPosition + size;
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
		const canvasRect: DOMRect = canvas.getBoundingClientRect();
		this.xOffset = canvasRect.left;
		this.yOffset = canvasRect.top;
		// Set the initital X and Y aim position to the center of the canvas
		this.aimXPos = canvas.width / 2;
		this.aimYPos = canvas.height / 2;

		const dx: number = this.aimXPos - this.xPosition - this.tankMidpoint;
		const dy: number = this.aimYPos - this.yPosition - this.tankMidpoint;
		let theta = Math.atan2(dy, dx);
		if (theta < 0) {
			theta += 2 * Math.PI;
		}
		this.aimAngle = theta;

		this.gunBarrellEndX = this.xPosition + this.tankMidpoint + Math.cos(this.aimAngle) * this.size;
		this.gunBarrellEndY = this.yPosition + this.tankMidpoint + Math.sin(this.aimAngle) * this.size;

		this.fragments = [];
	}

	public draw(context: CanvasRenderingContext2D): void {
		if (this.isDestroyed) {
			if (this.isExploding) {
				this.updateExplosion(context);
			}
			context.strokeStyle = this.color;
			context.lineWidth = 5;
			context.setLineDash([]);
			const xLength = 12;
			context.beginPath();
			context.moveTo(this.xPosition + this.tankMidpoint - xLength, this.yPosition + this.tankMidpoint - xLength);
			context.lineTo(this.xPosition + this.tankMidpoint + xLength, this.yPosition + this.tankMidpoint + xLength);
			context.stroke();

			context.beginPath();
			context.moveTo(this.xPosition + this.tankMidpoint - xLength, this.yPosition + this.tankMidpoint + xLength);
			context.lineTo(this.xPosition + this.tankMidpoint + xLength, this.yPosition + this.tankMidpoint - xLength);
			context.stroke();
		} else {
			context.fillStyle = this.color;
			context.fillRect(this.xPosition, this.yPosition, this.size, this.size);

			context.setLineDash([]);
			context.lineJoin = 'bevel';
			context.strokeStyle = 'black';
			context.lineWidth = 2;
			context.strokeRect(this.xPosition, this.yPosition, this.size, this.size);

			context.beginPath();
			context.arc(this.xPosition + this.tankMidpoint, this.yPosition + this.tankMidpoint, this.size / 3, 0, this.twoPi);
			context.stroke();

			this.gunBarrellEndX = this.xPosition + this.tankMidpoint + Math.cos(this.aimAngle) * this.size;
			this.gunBarrellEndY = this.yPosition + this.tankMidpoint + Math.sin(this.aimAngle) * this.size;
			context.beginPath();
			context.moveTo(this.xPosition + this.tankMidpoint, this.yPosition + this.tankMidpoint);
			context.lineTo(this.gunBarrellEndX, this.gunBarrellEndY);
			context.lineWidth = this.gunBarrellWidth;
			context.stroke();
		}
	}

	public destroy(): void {
		this.isDestroyed = true;
		this.isExploding = true;
		this.createFragments();
		setTimeout((): void => {
			this.isExploding = false;
		}, 1000);
	}

	createFragments(): void {
		const fragmentCount = 50;
		for (let i = 0; i < fragmentCount; i++) {
			const angle = Math.random() * 2 * Math.PI;
			const speed = Math.random() * 5 + 2;
			const velocityX = Math.cos(angle) * speed;
			const velocityY = Math.sin(angle) * speed;
			const grayValue = Math.floor(Math.random() * 256);
			const alphaValue = Math.random() * 0.5 + 0.5;
			const fragmentColor = `rgba(${grayValue}, ${grayValue}, ${grayValue}, ${alphaValue})`;
			const fragment = new BombFragment(
				this.xPosition + this.tankMidpoint,
				this.yPosition + this.tankMidpoint,
				Math.random() * 2 + 1,
				fragmentColor,
				velocityX,
				velocityY,
				10
			);
			this.fragments.push(fragment);
		}
	}

	updateExplosion(context: CanvasRenderingContext2D): void {
		this.fragments.forEach((particle) => {
			particle.update();
			particle.draw(context);
		});

		this.fragments = this.fragments.filter((particle) => particle.life > 0);
	}

	public updatePosition(playerTank: Tank, enemyTanks: Tank[], ammunition: Ammunition[], bombs: Bomb[]): void {
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
		if (this.isDestroyed) {
			return;
		}
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
		if (this.isDestroyed) {
			return;
		}
		if (this.lastDirectionMoved == Direction.NORTH) {
			this.consecutiveDirectionMoves += 1;
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.NORTH;
		let blocked = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				this.yPosition - this.speed < obstacle.yTop + obstacle.height &&
				this.yPosition > obstacle.yTop &&
				obstacle.xLeft < this.xPosition + this.size &&
				this.xPosition < obstacle.xRight
			) {
				this.yPosition = obstacle.yBottom;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			this.yPosition = Math.max(this.yPosition - this.speed, 0);
		} else {
			this.wasLastMoveBlocked = true;
		}
	}

	public moveSouth(): void {
		if (this.isDestroyed) {
			return;
		}
		if (this.lastDirectionMoved == Direction.SOUTH) {
			this.consecutiveDirectionMoves += 1;
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.SOUTH;
		let blocked = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				this.yPosition + this.speed + this.size > obstacle.yTop &&
				this.yPosition < obstacle.yTop + obstacle.height &&
				obstacle.xLeft < this.xPosition + this.size &&
				this.xPosition < obstacle.xRight
			) {
				this.yPosition = obstacle.yTop - this.size;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			this.yPosition = Math.min(this.yPosition + this.speed, this.canvasHeight - this.size);
		} else {
			this.wasLastMoveBlocked = true;
		}
	}

	public moveWest(): void {
		if (this.isDestroyed) {
			return;
		}
		if (this.lastDirectionMoved == Direction.WEST) {
			this.consecutiveDirectionMoves += 1;
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.WEST;
		let blocked = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				this.xPosition - this.speed < obstacle.xRight &&
				this.xPosition > obstacle.xLeft &&
				obstacle.yTop < this.yPosition + this.size &&
				this.yPosition < obstacle.yBottom
			) {
				this.xPosition = obstacle.xRight;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			this.xPosition = Math.max(this.xPosition - this.speed, 0);
		} else {
			this.wasLastMoveBlocked = true;
		}
	}

	public moveEast(): void {
		if (this.isDestroyed) {
			return;
		}
		if (this.lastDirectionMoved == Direction.EAST) {
			this.consecutiveDirectionMoves += 1;
		} else {
			this.consecutiveDirectionMoves = 0;
		}
		this.lastDirectionMoved = Direction.EAST;
		let blocked = false;
		for (const obstacle of this.obstacleCanvas.obstacles) {
			if (
				this.xPosition + this.speed + this.size > obstacle.xLeft &&
				this.xPosition < obstacle.xLeft + obstacle.width &&
				obstacle.yTop < this.yPosition + this.size &&
				this.yPosition < obstacle.yBottom
			) {
				this.xPosition = obstacle.xLeft - this.size;
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			this.xPosition = Math.min(this.xPosition + this.speed, this.canvasWidth - this.size);
		} else {
			this.wasLastMoveBlocked = true;
		}
	}

	public moveNorthEast(): void {
		if (this.isDestroyed) {
			return;
		}
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
				this.yPosition - this.speed < obstacle.yTop + obstacle.height &&
				this.yPosition > obstacle.yTop &&
				obstacle.xLeft < this.xPosition + this.size &&
				this.xPosition < obstacle.xRight
			) {
				this.yPosition = obstacle.yBottom;
				blockedNorth = true;
			}
			if (
				!blockedEast &&
				this.xPosition + this.speed + this.size > obstacle.xLeft &&
				this.xPosition < obstacle.xLeft + obstacle.width &&
				obstacle.yTop < this.yPosition + this.size &&
				this.yPosition < obstacle.yBottom
			) {
				this.xPosition = obstacle.xLeft - this.size;
				blockedEast = true;
			}
		}
		if (blockedNorth && blockedEast) {
			this.wasLastMoveBlocked = true;
		}
		if (!blockedNorth) {
			this.yPosition = Math.max(this.yPosition - this.speed, 0);
		}
		if (!blockedEast) {
			this.xPosition = Math.min(this.xPosition + this.speed, this.canvasWidth - this.size);
		}
	}

	public moveNorthWest(): void {
		if (this.isDestroyed) {
			return;
		}
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
				this.yPosition - this.speed < obstacle.yTop + obstacle.height &&
				this.yPosition > obstacle.yTop &&
				obstacle.xLeft < this.xPosition + this.size &&
				this.xPosition < obstacle.xRight
			) {
				this.yPosition = obstacle.yBottom;
				blockedNorth = true;
			}
			if (
				!blockedWest &&
				this.xPosition - this.speed < obstacle.xRight &&
				this.xPosition > obstacle.xLeft &&
				obstacle.yTop < this.yPosition + this.size &&
				this.yPosition < obstacle.yBottom
			) {
				this.xPosition = obstacle.xRight;
				blockedWest = true;
			}
		}
		if (blockedNorth && blockedWest) {
			this.wasLastMoveBlocked = true;
		}
		if (!blockedNorth) {
			this.yPosition = Math.max(this.yPosition - this.speed, 0);
		}
		if (!blockedWest) {
			this.xPosition = Math.max(this.xPosition - this.speed, 0);
		}
	}

	public moveSouthEast(): void {
		if (this.isDestroyed) {
			return;
		}
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
				this.yPosition + this.speed + this.size > obstacle.yTop &&
				this.yPosition < obstacle.yTop + obstacle.height &&
				obstacle.xLeft < this.xPosition + this.size &&
				this.xPosition < obstacle.xRight
			) {
				this.yPosition = obstacle.yTop - this.size;
				blockedSouth = true;
			}
			if (
				!blockedEast &&
				this.xPosition + this.speed + this.size > obstacle.xLeft &&
				this.xPosition < obstacle.xLeft + obstacle.width &&
				obstacle.yTop < this.yPosition + this.size &&
				this.yPosition < obstacle.yBottom
			) {
				this.xPosition = obstacle.xLeft - this.size;
				blockedEast = true;
			}
		}
		if (blockedSouth && blockedEast) {
			this.wasLastMoveBlocked = true;
		}
		if (!blockedSouth) {
			this.yPosition = Math.min(this.yPosition + this.speed, this.canvasHeight - this.size);
		}
		if (!blockedEast) {
			this.xPosition = Math.min(this.xPosition + this.speed, this.canvasWidth - this.size);
		}
	}

	public moveSouthWest(): void {
		if (this.isDestroyed) {
			return;
		}
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
				this.yPosition + this.speed + this.size > obstacle.yTop &&
				this.yPosition < obstacle.yTop + obstacle.height &&
				obstacle.xLeft < this.xPosition + this.size &&
				this.xPosition < obstacle.xRight
			) {
				this.yPosition = obstacle.yTop - this.size;
				blockedSouth = true;
			}
			if (
				!blockedWest &&
				this.xPosition - this.speed < obstacle.xRight &&
				this.xPosition > obstacle.xLeft &&
				obstacle.yTop < this.yPosition + this.size &&
				this.yPosition < obstacle.yBottom
			) {
				this.xPosition = obstacle.xRight;
				blockedWest = true;
			}
		}
		if (blockedSouth && blockedWest) {
			this.wasLastMoveBlocked = true;
		}
		if (!blockedSouth) {
			this.yPosition = Math.min(this.yPosition + this.speed, this.canvasHeight - this.size);
		}
		if (!blockedWest) {
			this.xPosition = Math.max(this.xPosition - this.speed, 0);
		}
	}
}
