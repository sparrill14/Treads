import { AudioFile, AudioManager } from './AudioManager';
import { BombFragment } from './BombFragment';
import { Tank } from './tanks/Tank';

export class Bomb {
	public xPos: number;
	public yPos: number;
	public blastRadius: number;
	public blastDelayMS: number;
	public isDestroyed: boolean;
	public isExploding: boolean;
	public fuseStartTime: number;
	public shouldFlashRed: boolean;
	public audioManager: AudioManager;

	private fragments: BombFragment[];

	constructor(startX: number, startY: number, blastRadius: number, isDestroyed: boolean, audioManager: AudioManager) {
		this.xPos = startX;
		this.yPos = startY;
		this.blastRadius = blastRadius;
		this.isDestroyed = isDestroyed;
		this.audioManager = audioManager;
		this.blastDelayMS = 5000;
		this.fuseStartTime = 0;
		this.shouldFlashRed = false;
		this.isExploding = false;
		this.fragments = [];
	}

	checkEnemyHit(enemyTanks: Tank[]): void {
		enemyTanks.forEach((enemyTank) => {
			if (enemyTank.isDestroyed) {
				return;
			}
			if (this.isExploding) {
				if (
					this.isExploding &&
					this.isPointInsideBlastRadius(
						enemyTank.xPos + enemyTank.tankMidpoint,
						enemyTank.yPos + enemyTank.tankMidpoint
					)
				) {
					enemyTank.destroy();
					console.log('Enemy hit with bomb!!!');
				}
			}
		});
	}

	checkPlayerHit(playerTank: Tank): void {
		if (playerTank.isDestroyed) {
			return;
		}
		if (
			this.isExploding &&
			(this.isPointInsideBlastRadius(playerTank.xLeft, playerTank.yTop) ||
				this.isPointInsideBlastRadius(playerTank.xRight, playerTank.yTop) ||
				this.isPointInsideBlastRadius(playerTank.xLeft, playerTank.yBottom) ||
				this.isPointInsideBlastRadius(playerTank.xRight, playerTank.yBottom) ||
				this.isPointInsideBlastRadius(playerTank.xPos, playerTank.yPos))
		) {
			playerTank.destroy();
			console.log('Player Hit with bomb!!!');
		}
	}

	setFuse(): void {
		if (this.isDestroyed) {
			return;
		}
		this.fuseStartTime = performance.now();
		this.animateFuse();
		setTimeout((): void => {
			this.createFragments();
			this.audioManager.play(AudioFile.BOMB_EXPLODE);
			this.isExploding = true;
			setTimeout((): void => {
				this.isExploding = false;
				this.isDestroyed = true;
			}, 1000);
		}, this.blastDelayMS);
	}

	private animateFuse(): void {
		if (this.isDestroyed) {
			return;
		}
		const elapsedTime = performance.now() - this.fuseStartTime;
		const remainingTime = this.blastDelayMS - elapsedTime;
		if (remainingTime <= 0) {
			this.isDestroyed = true;
			return;
		}
		const fractionElapsed = elapsedTime / this.blastDelayMS;
		const flashThresholds = [0.25, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.90625, 0.9375, 0.96875, 0.984375];
		this.shouldFlashRed = flashThresholds.some(
			(threshold) => fractionElapsed > threshold && fractionElapsed < threshold + 0.01
		);
		requestAnimationFrame(() => this.animateFuse());
	}

	isPointInsideBlastRadius(x: number, y: number): boolean {
		const dx = x - this.xPos;
		const dy = y - this.yPos;
		return Math.sqrt(dx * dx + dy * dy) <= this.blastRadius;
	}

	createFragments(): void {
		const fragmentCount = 50;
		for (let i = 0; i < fragmentCount; i++) {
			const angle = Math.random() * 2 * Math.PI;
			const speed = Math.random() * 5 + 2;
			const velocityX = Math.cos(angle) * speed;
			const velocityY = Math.sin(angle) * speed;
			const fragment = new BombFragment(
				this.xPos,
				this.yPos,
				Math.random() * 2 + 1,
				'rgba(255, 69, 0, 1)',
				velocityX,
				velocityY,
				this.blastRadius
			);
			this.fragments.push(fragment);
		}
	}

	draw(context: CanvasRenderingContext2D): void {
		if (this.isExploding) {
			this.updateExplosion(context);
			return;
		}

		context.beginPath();
		context.arc(this.xPos, this.yPos, 15, 0, 2 * Math.PI);
		context.fillStyle = this.shouldFlashRed ? 'red' : 'yellow';
		context.fill();
		context.lineWidth = 3;
		context.strokeStyle = 'black';
		context.stroke();
		context.closePath();

		context.beginPath();
		context.arc(this.xPos, this.yPos, 10, 0, 2 * Math.PI);
		context.lineWidth = 1.5;
		context.strokeStyle = 'black';
		context.stroke();
		context.closePath();
	}

	updateExplosion(context: CanvasRenderingContext2D): void {
		this.fragments.forEach((particle) => {
			particle.update();
			particle.draw(context);
		});

		this.fragments = this.fragments.filter((particle) => particle.life > 0);
	}
}

export class PlayerBomb extends Bomb {
	constructor(startX: number, startY: number, isDestroyed: boolean, audioManager: AudioManager) {
		const playerBombBlastRadius = 50;
		super(startX, startY, playerBombBlastRadius, isDestroyed, audioManager);
	}
}

export class BasicBomb extends Bomb {
	constructor(startX: number, startY: number, isDestroyed: boolean, audioManager: AudioManager) {
		const basicBombBlastRadius = 80;
		super(startX, startY, basicBombBlastRadius, isDestroyed, audioManager);
	}
}

export class SuperBomb extends Bomb {
	constructor(startX: number, startY: number, isDestroyed: boolean, audioManager: AudioManager) {
		const superBombBlastRadius = 20;
		super(startX, startY, superBombBlastRadius, isDestroyed, audioManager);
	}
}

export class LoveBomb extends Bomb {
	constructor(startX: number, startY: number, isDestroyed: boolean, audioManager: AudioManager) {
		const loveBombBlastRadius = 100;
		super(startX, startY, loveBombBlastRadius, isDestroyed, audioManager);
	}

	override draw(context: CanvasRenderingContext2D): void {
		if (this.isExploding) {
			this.updateExplosion(context);
			return;
		}
		context.save();
		context.translate(this.xPos, this.yPos);
		context.scale(1, 1.3);
		context.beginPath();
		context.moveTo(0, 0);
		context.bezierCurveTo(12, -12, 24, 0, 0, 24);
		context.bezierCurveTo(-24, 0, -12, -12, 0, 0);
		context.fillStyle = this.shouldFlashRed ? 'red' : '#FFC2D1';
		context.fill();
		context.lineWidth = 3;
		context.strokeStyle = 'black';
		context.stroke();
		context.closePath();
		context.restore();
	}
}
