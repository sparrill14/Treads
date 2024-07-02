import * as d3 from 'd3';
import { AudioFile, AudioManager } from './AudioManager';
import { BombFragment } from './BombFragment';
import { Tank } from './tanks/Tank';

export class Bomb {
	public xPosition: number;
	public yPosition: number;
	public blastRadius: number;
	public blastDelayMS: number;
	public isDestroyed: boolean;
	public isExploding: boolean;
	public fuseStartTime: number;
	public shouldFlashRed: boolean;
	public audioManager: AudioManager;
	public radius: number = 15;

	private fragments: BombFragment[];
	private fragmentColorScale = d3.scaleLinear<string>().domain([0, 0.5, 1]).range(['red', 'yellow', 'orange']);
	private fuseTimeoutId: number | null = null;

	constructor(startX: number, startY: number, blastRadius: number, isDestroyed: boolean, audioManager: AudioManager) {
		this.xPosition = startX;
		this.yPosition = startY;
		this.blastRadius = blastRadius;
		this.isDestroyed = isDestroyed;
		this.audioManager = audioManager;
		this.blastDelayMS = 5000;
		this.fuseStartTime = 0;
		this.shouldFlashRed = false;
		this.isExploding = false;
		this.fragments = [];
	}

	public destroy(): void {
		if (this.isDestroyed || this.isExploding) {
			return;
		}
		this.createFragments();
		this.audioManager.play(AudioFile.BOMB_EXPLODE);
		this.isExploding = true;
		if (this.fuseTimeoutId !== null) {
			clearTimeout(this.fuseTimeoutId);
			this.fuseTimeoutId = null;
		}
		setTimeout((): void => {
			this.isExploding = false;
			this.isDestroyed = true;
		}, 500);
	}

	public checkEnemyHit(enemyTanks: Tank[]): void {
		enemyTanks.forEach((enemyTank) => {
			if (enemyTank.isDestroyed) {
				return;
			}
			if (
				this.isExploding &&
				this.isPointInsideBlastRadius(
					enemyTank.xPosition + enemyTank.tankMidpoint,
					enemyTank.yPosition + enemyTank.tankMidpoint
				)
			) {
				enemyTank.destroy();
				console.log('Enemy hit with bomb!!!');
			}
		});
	}

	public checkPlayerHit(playerTank: Tank): void {
		if (playerTank.isDestroyed) {
			return;
		}
		if (
			this.isExploding &&
			(this.isPointInsideBlastRadius(playerTank.xLeft, playerTank.yTop) ||
				this.isPointInsideBlastRadius(playerTank.xRight, playerTank.yTop) ||
				this.isPointInsideBlastRadius(playerTank.xLeft, playerTank.yBottom) ||
				this.isPointInsideBlastRadius(playerTank.xRight, playerTank.yBottom) ||
				this.isPointInsideBlastRadius(playerTank.xPosition, playerTank.yPosition))
		) {
			playerTank.destroy();
			console.log('Player Hit with bomb!!!');
		}
	}

	public setFuse(): void {
		if (this.isDestroyed) {
			return;
		}
		this.fuseStartTime = performance.now();
		this.animateFuse();
		this.fuseTimeoutId = window.setTimeout((): void => {
			this.destroy();
		}, this.blastDelayMS);
	}

	public isPointInsideBlastRadius(x: number, y: number): boolean {
		const dx = x - this.xPosition;
		const dy = y - this.yPosition;
		return Math.sqrt(dx * dx + dy * dy) <= this.blastRadius;
	}

	protected updateExplosion(context: CanvasRenderingContext2D): void {
		this.fragments.forEach((particle) => {
			particle.update();
			particle.draw(context);
		});

		this.fragments = this.fragments.filter((particle) => particle.life > 0);
	}

	public draw(context: CanvasRenderingContext2D): void {
		if (this.isExploding) {
			this.updateExplosion(context);
			return;
		}

		context.beginPath();
		context.arc(this.xPosition, this.yPosition, this.radius, 0, 2 * Math.PI);
		context.fillStyle = this.shouldFlashRed ? 'red' : 'yellow';
		context.fill();
		context.lineWidth = 3;
		context.strokeStyle = 'black';
		context.stroke();
		context.closePath();

		context.beginPath();
		context.arc(this.xPosition, this.yPosition, 10, 0, 2 * Math.PI);
		context.lineWidth = 1.5;
		context.strokeStyle = 'black';
		context.stroke();
		context.closePath();
	}

	private animateFuse(): void {
		if (this.isDestroyed) {
			return;
		}
		const elapsedTime = performance.now() - this.fuseStartTime;
		const fractionElapsed = elapsedTime / this.blastDelayMS;
		const flashThresholds = [0.25, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.90625, 0.9375, 0.96875, 0.984375];
		this.shouldFlashRed = flashThresholds.some(
			(threshold) => fractionElapsed > threshold && fractionElapsed < threshold + 0.01
		);
		requestAnimationFrame(() => this.animateFuse());
	}

	private createFragments(): void {
		const fragmentCount = 50;
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
				this.blastRadius
			);
			this.fragments.push(fragment);
		}
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
		const superBombBlastRadius = 100;
		super(startX, startY, superBombBlastRadius, isDestroyed, audioManager);
	}
}

export class LoveBomb extends Bomb {
	constructor(startX: number, startY: number, isDestroyed: boolean, audioManager: AudioManager) {
		const loveBombBlastRadius = 150;
		super(startX, startY, loveBombBlastRadius, isDestroyed, audioManager);
	}

	override draw(context: CanvasRenderingContext2D): void {
		if (this.isExploding) {
			this.updateExplosion(context);
			return;
		}
		context.save();
		context.translate(this.xPosition, this.yPosition);
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
