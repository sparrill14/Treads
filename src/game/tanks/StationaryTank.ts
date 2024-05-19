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
	canTakeShot = true;

	constructor(
		canvas: HTMLCanvasElement,
		xPos: number,
		yPos: number,
		obstacleCanvas: ObstacleCanvas,
		audioManager: AudioManager
	) {
		const stationaryTankSpeed = 0;
		const stationaryTankSize = 30;
		const stationaryTankAggressionFactor = 0;
		const stationaryTankColor = '#5784ba';
		const ammunition: Ammunition[] = [new BasicAIAmmunition(0, 0, 0, 0, 0, true, audioManager)];
		const bombs: Bomb[] = [];
		const navigationGrid: NavigationGrid = new NavigationGrid();
		super(
			canvas,
			new NoReticule(),
			xPos,
			yPos,
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
	}

	public override updatePosition(playerTank: Tank): void {
		return;
	}

	public override shoot(): void {
		if (!this.canTakeShot || this.isDestroyed) {
			return;
		}
		const availableAmmunitionIndex = this.ammunition.findIndex((ammunition) => ammunition.isDestroyed);
		if (availableAmmunitionIndex !== -1) {
			this.ammunition[availableAmmunitionIndex] = new BasicAIAmmunition(
				this.xPos + this.size / 2,
				this.yPos + this.size / 2,
				this.aimAngle,
				this.canvasWidth,
				this.canvasHeight,
				false,
				this.audioManager
			);
			this.canTakeShot = false;
			setTimeout(() => {
				this.canTakeShot = true;
			}, this.minTimeBetweenShotsMS);
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
