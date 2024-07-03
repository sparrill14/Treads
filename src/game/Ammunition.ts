import * as d3 from 'd3';
import { PastelColorPalette } from '../ui/PastelColorPalette';
import { AudioFile, AudioManager } from './AudioManager';
import { Bomb } from './Bomb';
import { BombFragment } from './BombFragment';
import { ObstacleCanvas } from './ObstacleCanvas';
import { Tank } from './tanks/Tank';

export class Ammunition {
	public xPosition: number;
	public yPosition: number;
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
	public isExploding: boolean = false;
	public radius: number = 4;
	public fragmentationRadius: number = 10;
	private fragments: BombFragment[] = [];
	private fragmentColorScale = d3
		.scaleLinear<string>()
		.domain([0, 0.5, 1])
		.range([PastelColorPalette.PALE_BLACK, PastelColorPalette.PALE_GRAY, PastelColorPalette.WHITE_SMOKE]);

	constructor(
		startX: number,
		startY: number,
		theta: number,
		speed: number,
		maxBounces: number,
		canvasWidth: number,
		canvasHeight: number,
		isDestroyed: boolean,
		audioManager: AudioManager
	) {
		this.xPosition = startX;
		this.yPosition = startY;
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

	updatePosition(obstacleCanvas: ObstacleCanvas): void {
		if (this.isExploding) {
			return;
		}
		this.xPosition += this.xVelocity;
		this.yPosition += this.yVelocity;

		if (this.xPosition <= 0 || this.xPosition > this.canvasWidth) {
			this.xVelocity = -this.xVelocity;
			this.bounces++;
		}

		if (this.yPosition <= 0 || this.yPosition > this.canvasHeight) {
			this.yVelocity = -this.yVelocity;
			this.bounces++;
		}

		obstacleCanvas.obstacles.forEach((obstacle) => {
			if (
				this.xPosition > obstacle.xLeft &&
				this.xPosition < obstacle.xRight &&
				this.yPosition > obstacle.yTop &&
				this.yPosition < obstacle.yBottom
			) {
				this.bounces++;
				const fromLeft = Math.abs(this.xPosition - obstacle.xLeft);
				const fromRight = Math.abs(this.xPosition - obstacle.xRight);
				const fromTop = Math.abs(this.yPosition - obstacle.yTop);
				const fromBottom = Math.abs(this.yPosition - obstacle.yBottom);
				const minDistance = Math.min(fromLeft, fromRight, fromTop, fromBottom);

				if (minDistance === fromTop) {
					this.yPosition = obstacle.yTop - 1;
					this.yVelocity = -this.yVelocity;
				} else if (minDistance === fromBottom) {
					this.yPosition = obstacle.yBottom + 1;
					this.yVelocity = -this.yVelocity;
				} else if (minDistance === fromLeft) {
					this.xPosition = obstacle.xLeft - 1;
					this.xVelocity = -this.xVelocity;
				} else if (minDistance === fromRight) {
					this.xPosition = obstacle.xRight + 1;
					this.xVelocity = -this.xVelocity;
				}
			}
		});

		if (this.bounces > this.maxBounces) {
			this.destroy();
		}
	}

	public destroy(): void {
		if (this.isDestroyed || this.isExploding) {
			return;
		}
		this.createFragments();
		this.audioManager.play(AudioFile.AMMUNITION_EXPLODE);
		this.isExploding = true;
		setTimeout((): void => {
			this.isExploding = false;
			this.isDestroyed = true;
		}, 500);
	}

	checkEnemyHit(enemyTanks: Tank[]): void {
		enemyTanks.forEach((enemyTank) => {
			if (enemyTank.isDestroyed) {
				return;
			}
			if (
				this.xPosition > enemyTank.xLeft &&
				this.xPosition < enemyTank.xRight &&
				this.yPosition > enemyTank.yTop &&
				this.yPosition < enemyTank.yBottom
			) {
				this.destroy();
				enemyTank.destroy();
				this.audioManager.play(AudioFile.TANK_DESTROY);
				console.log('Enemy hit!!!');
			}
		});
	}

	checkPlayerHit(playerTank: Tank): void {
		if (playerTank.isDestroyed) {
			return;
		}
		if (
			this.xPosition > playerTank.xLeft &&
			this.xPosition < playerTank.xRight &&
			this.yPosition > playerTank.yTop &&
			this.yPosition < playerTank.yBottom
		) {
			playerTank.destroy();
			this.destroy();
			console.log('Player Hit!!!');
		}
	}

	checkAmmunitionCollision(ammunitions: Ammunition[]): void {
		for (const ammunition of ammunitions) {
			if (ammunition !== this && !ammunition.isDestroyed) {
				const dx = this.xPosition - ammunition.xPosition;
				const dy = this.yPosition - ammunition.yPosition;
				const distance = Math.sqrt(dx * dx + dy * dy);
				if (distance < this.radius + ammunition.radius) {
					this.destroy();
					ammunition.destroy();
				}
			}
		}
	}

	checkBombCollision(bombs: Bomb[]): void {
		for (const bomb of bombs) {
			if (!bomb.isDestroyed && !bomb.isExploding()) {
				const dx = this.xPosition - bomb.xPosition;
				const dy = this.yPosition - bomb.yPosition;
				const distance = Math.sqrt(dx * dx + dy * dy);
				if (distance < this.radius + bomb.radius) {
					this.destroy();
					bomb.destroy();
				}
			}
		}
	}

	reload(
		startX: number,
		startY: number,
		theta: number,
		isDestroyed: boolean,
		canvasWidth: number,
		canvasHeight: number
	) {
		this.xPosition = startX;
		this.yPosition = startY;
		this.theta = theta;
		this.isDestroyed = isDestroyed;
		this.xVelocity = Math.cos(this.theta) * this.speed;
		this.yVelocity = Math.sin(this.theta) * this.speed;
		this.canvasWidth = canvasWidth;
		this.canvasHeight = canvasHeight;
		this.bounces = 0;
	}

	draw(context: CanvasRenderingContext2D): void {
		if (this.isExploding) {
			this.updateExplosion(context);
			return;
		}

		context.beginPath();
		context.arc(this.xPosition, this.yPosition, this.radius, 0, 2 * Math.PI);
		context.fillStyle = 'white';
		context.fill();
		context.lineWidth = 2;
		context.strokeStyle = 'black';
		context.stroke();
		context.closePath();
	}

	private updateExplosion(context: CanvasRenderingContext2D): void {
		this.fragments.forEach((particle) => {
			particle.update();
			particle.draw(context);
		});

		this.fragments = this.fragments.filter((particle) => particle.life > 0);
	}

	private createFragments(): void {
		const fragmentCount = 15;
		for (let i = 0; i < fragmentCount; i++) {
			const angle = Math.random() * 2 * Math.PI;
			const speed = Math.random() * 5 + 2;
			const velocityX = Math.cos(angle) * speed;
			const velocityY = Math.sin(angle) * speed;
			const fragment = new BombFragment(
				this.xPosition,
				this.yPosition,
				Math.random() * 2 + 1,
				this.fragmentColorScale(Math.random()),
				velocityX,
				velocityY,
				this.fragmentationRadius
			);
			this.fragments.push(fragment);
		}
	}

	willHitPlayerTank(obstacleCanvas: ObstacleCanvas, playerTank: Tank): boolean {
		let predictedXPosition: number = this.xPosition;
		let predictedYPosition: number = this.yPosition;
		let predictedXVelocity: number = this.xVelocity;
		let predictedYVelocity: number = this.yVelocity;
		let predictedBounces = 0;
		while (predictedBounces <= this.maxBounces) {
			predictedXPosition += predictedXVelocity;
			predictedYPosition += predictedYVelocity;
			if (predictedXPosition <= 0 || predictedXPosition > this.canvasWidth) {
				predictedXVelocity = -predictedXVelocity;
				predictedBounces++;
			}
			if (predictedYPosition <= 0 || predictedYPosition > this.canvasHeight) {
				predictedYVelocity = -predictedYVelocity;
				predictedBounces++;
			}
			obstacleCanvas.obstacles.forEach((obstacle) => {
				if (
					predictedXPosition > obstacle.xLeft &&
					predictedXPosition < obstacle.xRight &&
					predictedYPosition > obstacle.yTop &&
					predictedYPosition < obstacle.yBottom
				) {
					predictedBounces++;
					predictedXVelocity = -predictedXVelocity;
					predictedYVelocity = -predictedYVelocity;
				}
			});
			if (
				predictedXPosition > playerTank.xLeft &&
				predictedXPosition < playerTank.xRight &&
				predictedYPosition > playerTank.yTop &&
				predictedYPosition < playerTank.yBottom
			) {
				return true;
			}
		}
		return false;
	}
}

export class PlayerAmmunition extends Ammunition {
	constructor(
		startX: number,
		startY: number,
		theta: number,
		canvasWidth: number,
		canvasHeight: number,
		isDestroyed: boolean,
		audioManager: AudioManager
	) {
		const playerAmmunitionMaxBounces = 1;
		const playerAmmunitionSpeed = 4;
		super(
			startX,
			startY,
			theta,
			playerAmmunitionSpeed,
			playerAmmunitionMaxBounces,
			canvasWidth,
			canvasHeight,
			isDestroyed,
			audioManager
		);
	}
}

export class BasicAIAmmunition extends Ammunition {
	constructor(
		startX: number,
		startY: number,
		theta: number,
		canvasWidth: number,
		canvasHeight: number,
		isDestroyed: boolean,
		audioManager: AudioManager
	) {
		const BasicAIAmmunitionMaxBounces = 1;
		const BasicAIAmmunitionSpeed = 4;
		super(
			startX,
			startY,
			theta,
			BasicAIAmmunitionSpeed,
			BasicAIAmmunitionMaxBounces,
			canvasWidth,
			canvasHeight,
			isDestroyed,
			audioManager
		);
	}
}

export class SuperAIAmmunition extends Ammunition {
	constructor(
		startX: number,
		startY: number,
		theta: number,
		canvasWidth: number,
		canvasHeight: number,
		isDestroyed: boolean,
		audioManager: AudioManager
	) {
		const superAIAmmunitionMaxBounces = 2;
		const superAIAmmunitionSpeed = 6;
		super(
			startX,
			startY,
			theta,
			superAIAmmunitionSpeed,
			superAIAmmunitionMaxBounces,
			canvasWidth,
			canvasHeight,
			isDestroyed,
			audioManager
		);
	}
}
