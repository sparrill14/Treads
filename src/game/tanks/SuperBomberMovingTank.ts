import { PastelColorPalette } from '../../ui/PastelColorPalette';
import { Ammunition } from '../Ammunition';
import { AudioManager } from '../AudioManager';
import { Bomb } from '../Bomb';
import { Navigator } from '../navigation/Navigator';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { NoReticule } from '../Reticule';
import { EnemyTank } from './EnemyTank';
import { Tank } from './Tank';

export class SuperBomberMovingTank extends EnemyTank {
	public minTimeBetweenShotsMS = 1000;
	public timeBetweenShotsIsElapsed = false;
	public minTimeBetweenBombPlantsMS = 1000;
	public timeBetweenPlantsIsElapsed = true;

	constructor(
		canvas: HTMLCanvasElement,
		xPosition: number,
		yPosition: number,
		obstacleCanvas: ObstacleCanvas,
		ammunition: Ammunition[],
		bombs: Bomb[],
		navigator: Navigator,
		audioManager: AudioManager
	) {
		const superBomberMovingTankSpeed = 2.2;
		const superBomberMovingTankSize = 30;
		const superBomberMovingTankAggressionFactor = 5;
		const superBomberMovingTankColor = PastelColorPalette.BLUSH_PINK;
		super(
			canvas,
			new NoReticule(),
			xPosition,
			yPosition,
			superBomberMovingTankSpeed,
			superBomberMovingTankSize,
			superBomberMovingTankAggressionFactor,
			superBomberMovingTankColor,
			obstacleCanvas,
			ammunition,
			bombs,
			audioManager,
			navigator
		);
		setTimeout(() => {
			this.timeBetweenShotsIsElapsed = true;
		}, 1000);
	}

	public override plantBomb(playerTank: Tank): void {
		if (this.timeBetweenPlantsIsElapsed && !this.isDestroyed) {
			const availableBombIndex = this.bombs.findIndex((bomb) => bomb.isDestroyed && !bomb.isExploding());
			if (availableBombIndex !== -1) {
				this.bombs[availableBombIndex].xPosition = this.xPosition + this.size / 2;
				this.bombs[availableBombIndex].yPosition = this.yPosition + this.size / 2;
				const willHitPlayerTank: boolean = this.bombs[availableBombIndex].isTankInsideBlastRadius(playerTank);
				if (willHitPlayerTank) {
					this.bombs[availableBombIndex].setFuse();
					this.timeBetweenPlantsIsElapsed = false;
					setTimeout(() => {
						this.timeBetweenPlantsIsElapsed = true;
					}, this.minTimeBetweenBombPlantsMS);
				}
			}
		}
		return;
	}

	public override shoot(playerTank: Tank): void {
		if (this.timeBetweenShotsIsElapsed && !this.isDestroyed) {
			const availableAmmunitionIndex = this.ammunition.findIndex((ammunition) => ammunition.isDestroyed);
			if (availableAmmunitionIndex !== -1) {
				this.ammunition[availableAmmunitionIndex].reload(
					this.gunBarrellEndX,
					this.gunBarrellEndY,
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
					this.timeBetweenShotsIsElapsed = false;
					setTimeout(() => {
						this.timeBetweenShotsIsElapsed = true;
					}, this.minTimeBetweenShotsMS);
				}
			}
		}
		return;
	}

	public override aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
		if (this.isDestroyed) {
			return;
		}

		const dx: number = playerTank.xPosition + playerTank.size / 2 - this.xPosition - this.tankMidpoint;
		const dy: number = playerTank.yPosition + playerTank.size / 2 - this.yPosition - this.tankMidpoint;
		let theta = Math.atan2(dy, dx);
		if (theta < 0) {
			theta += 2 * Math.PI;
		}
		this.aimAngle = theta;
	}
}
