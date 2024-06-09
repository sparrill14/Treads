import { PastelColorPalette } from '../../ui/PastelColorPalette';
import { Ammunition, BasicAIAmmunition } from '../Ammunition';
import { AudioManager } from '../AudioManager';
import { Bomb } from '../Bomb';
import { NavigationGrid } from '../NavigationGrid';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { NoReticule } from '../Reticule';
import { EnemyTank } from './EnemyTank';
import { Tank } from './Tank';

export class StationaryTank extends EnemyTank {
	minTimeBetweenShotsMS = 5000;
	timeBetweenShotsIsElapsed = false;

	constructor(
		canvas: HTMLCanvasElement,
		xPosition: number,
		yPosition: number,
		obstacleCanvas: ObstacleCanvas,
		audioManager: AudioManager
	) {
		const stationaryTankSpeed = 0;
		const stationaryTankSize = 30;
		const stationaryTankAggressionFactor = 0;
		const stationaryTankColor = PastelColorPalette.PALE_GRAY;
		const ammunition: Ammunition[] = [new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)];
		const bombs: Bomb[] = [];
		const navigationGrid: NavigationGrid = new NavigationGrid();
		super(
			canvas,
			new NoReticule(),
			xPosition,
			yPosition,
			stationaryTankSpeed,
			stationaryTankSize,
			stationaryTankAggressionFactor,
			stationaryTankColor,
			obstacleCanvas,
			ammunition,
			bombs,
			navigationGrid,
			audioManager
		);
		setTimeout(() => {
			this.timeBetweenShotsIsElapsed = true;
		}, 1000);
	}

	public override updatePosition(playerTank: Tank): void {
		return;
	}

	public override shoot(playerTank: Tank): void {
		if (!this.timeBetweenShotsIsElapsed || this.isDestroyed) {
			return;
		}
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
