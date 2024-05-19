import { Ammunition } from '../Ammunition';
import { AudioManager } from '../AudioManager';
import { Bomb } from '../Bomb';
import { NavigationGrid } from '../NavigationGrid';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { NoReticule } from '../Reticule';
import { EnemyTank } from './EnemyTank';
import { Tank } from './Tank';

export class BomberTank extends EnemyTank {
	public minTimeBetweenShotsMS = 20000;
	public timeBetweenShotsIsElapsed = true;
	public minTimeBetweenBombPlantsMS = 1000;
	public timeBetweenPlantsIsElapsed = true;

	constructor(
		canvas: HTMLCanvasElement,
		xPos: number,
		yPos: number,
		obstacleCanvas: ObstacleCanvas,
		ammunition: Ammunition[],
		bombs: Bomb[],
		navigationGrid: NavigationGrid,
		audioManager: AudioManager
	) {
		const bomberTankSpeed = 2;
		const bomberTankSize = 30;
		const bomberTankAggressionFactor = 4;
		const bomberTankColor = 'yellow';
		super(
			canvas,
			new NoReticule(),
			xPos,
			yPos,
			bomberTankSpeed,
			bomberTankSize,
			bomberTankAggressionFactor,
			bomberTankColor,
			obstacleCanvas,
			ammunition,
			bombs,
			navigationGrid,
			audioManager
		);
	}

	public override plantBomb(playerTank: Tank): void {
		if (this.timeBetweenPlantsIsElapsed && !this.isDestroyed) {
			const availableBombIndex = this.bombs.findIndex((bomb) => bomb.isDestroyed && !bomb.isExploding);
			if (availableBombIndex !== -1) {
				this.bombs[availableBombIndex].xPos = this.xPos + this.size / 2;
				this.bombs[availableBombIndex].yPos = this.yPos + this.size / 2;
				const willHitPlayerTank: boolean = this.bombs[availableBombIndex].isPointInsideBlastRadius(
					playerTank.xPos + playerTank.tankMidpoint,
					playerTank.yPos + playerTank.tankMidpoint
				);
				if (willHitPlayerTank) {
					this.bombs[availableBombIndex].isDestroyed = false;
					this.bombs[availableBombIndex].setFuse();
					this.timeBetweenPlantsIsElapsed = false;
					setTimeout((): void => {
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

		const dx: number = playerTank.xPos + playerTank.size / 2 - this.xPos - this.tankMidpoint;
		const dy: number = playerTank.yPos + playerTank.size / 2 - this.yPos - this.tankMidpoint;
		let theta = Math.atan2(dy, dx);
		if (theta < 0) {
			theta += 2 * Math.PI;
		}
		this.aimAngle = theta;
	}
}
