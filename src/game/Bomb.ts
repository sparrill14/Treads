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
	public fuseStartTime: number;
	public shouldFlashRed: boolean;
	public audioManager: AudioManager;
	public radius: number = 15;

	protected fragments: BombFragment[] = [];
	private fragmentColorScale = d3.scaleLinear<string>().domain([0, 0.5, 1]).range(['red', 'yellow', 'orange']);
	private fuseTimeoutId: number | null = null;

	constructor(startX: number, startY: number, blastRadius: number, isDestroyed: boolean, audioManager: AudioManager) {
		this.xPosition = startX;
		this.yPosition = startY;
		this.blastRadius = blastRadius;
		this.isDestroyed = isDestroyed;
		this.audioManager = audioManager;
		this.blastDelayMS = 6000;
		this.fuseStartTime = 0;
		this.shouldFlashRed = false;
	}

	public destroy(): void {
		if (this.isDestroyed || this.isExploding()) {
			return;
		}
		this.isDestroyed = true;
		this.createFragments();
		this.audioManager.play(AudioFile.BOMB_EXPLODE);
		if (this.fuseTimeoutId !== null) {
			clearTimeout(this.fuseTimeoutId);
			this.fuseTimeoutId = null;
		}
	}

	public checkEnemyHit(enemyTanks: Tank[]): void {
		enemyTanks.forEach((enemyTank) => {
			if (enemyTank.isDestroyed) {
				return;
			}
			for (const fragment of this.fragments) {
				if (fragment.checkHit(enemyTank)) {
					enemyTank.destroy();
					break;
				}
			}
		});
	}

	public checkPlayerHit(playerTank: Tank): void {
		if (playerTank.isDestroyed) {
			return;
		}
		for (const fragment of this.fragments) {
			if (fragment.checkHit(playerTank)) {
				playerTank.destroy();
				break;
			}
		}
	}

	public setFuse(): void {
		this.fragments = [];
		this.isDestroyed = false;
		this.fuseStartTime = performance.now();
		this.animateFuse();
		this.fuseTimeoutId = window.setTimeout((): void => {
			this.destroy();
		}, this.blastDelayMS);
	}

	public isTankInsideBlastRadius(tank: Tank): boolean {
		const squareXCenter: number = tank.xPosition + tank.tankMidpoint;
		const squareYCenter: number = tank.yPosition + tank.tankMidpoint;
		const circleXCenter: number = this.xPosition;
		const circleYCenter: number = this.yPosition;
		const halfSideLength: number = tank.tankMidpoint;
		const closestX: number = Math.max(
			squareXCenter - halfSideLength,
			Math.min(circleXCenter, squareXCenter + halfSideLength)
		);
		const closestY: number = Math.max(
			squareYCenter - halfSideLength,
			Math.min(circleYCenter, squareYCenter + halfSideLength)
		);
		const dx: number = closestX - circleXCenter;
		const dy: number = closestY - circleYCenter;
		const distance: number = Math.sqrt(dx * dx + dy * dy);
		return distance <= this.blastRadius;
	}

	public isExploding(): boolean {
		const isExploding: boolean = this.fragments.some((fragment) => fragment.life > 0);
		return isExploding;
	}

	protected updateExplosion(context: CanvasRenderingContext2D): void {
		this.fragments.forEach((particle) => {
			particle.update();
			particle.draw(context);
		});

		this.fragments = this.fragments.filter((particle) => particle.life > 0);
	}

	public draw(context: CanvasRenderingContext2D): void {
		const isExploding: boolean = this.isExploding();
		if (isExploding) {
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
		const basicBombBlastRadius = 50;
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
		const loveBombBlastRadius = 80;
		super(startX, startY, loveBombBlastRadius, isDestroyed, audioManager);
	}

	override draw(context: CanvasRenderingContext2D): void {
		const isExploding = this.fragments.some((fragment: BombFragment) => {
			return fragment.life > 0;
		});
		if (isExploding) {
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
