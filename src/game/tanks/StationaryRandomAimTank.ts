import { Ammunition } from '../Ammunition';
import { AudioManager } from '../AudioManager';
import { Bomb } from '../Bomb';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { NoReticule } from '../Reticule';
import { EnemyTank } from './EnemyTank';
import { Tank } from './Tank';

export class StationaryRandomAimTank extends EnemyTank {
	public aimAngleChangeAmount = 0;

	constructor(
		canvas: HTMLCanvasElement,
		xPos: number,
		yPos: number,
		obstacleCanvas: ObstacleCanvas,
		ammunition: Ammunition[],
		audioManager: AudioManager
	) {
		const fastTankSpeed = 0;
		const fastTankSize = 30;
		const fastTankColor = '#ebe1b9';
		const bombs: Bomb[] = [];
		super(
			canvas,
			new NoReticule(),
			xPos,
			yPos,
			fastTankSpeed,
			fastTankSize,
			fastTankColor,
			obstacleCanvas,
			ammunition,
			bombs,
			audioManager
		);
	}

	private getAngleChangeAmount(): number {
		const max = 360;
		const min = -360;
		const randomAmount: number = Math.floor(Math.random() * (max - min + 1)) + min;
		return randomAmount;
	}

	public override updatePosition(playerTank: Tank): void {
		return;
	}

	public override shoot(playerTank: Tank): void {
		const availableAmmunitionIndex = this.ammunition.findIndex((ammunition) => ammunition.isDestroyed);
		if (availableAmmunitionIndex !== -1) {
			this.ammunition[availableAmmunitionIndex].reload(
				this.xPos + this.size / 2,
				this.yPos + this.size / 2,
				this.aimAngle,
				true,
				this.canvasWidth,
				this.canvasHeight
			);
			const willHitPlayerTank: boolean = this.ammunition[availableAmmunitionIndex].willHitPlayerTank(
				this.obstacleCanvas,
				playerTank
			);
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
			this.aimAngle += 0.01;
			this.aimAngleChangeAmount -= 1;
		} else if (this.aimAngleChangeAmount < 0) {
			this.aimAngle -= 0.01;
			this.aimAngleChangeAmount += 1;
		} else {
			this.aimAngleChangeAmount = this.getAngleChangeAmount();
		}
	}
}
