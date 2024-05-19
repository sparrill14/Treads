import { Ammunition } from '../Ammunition';
import { AudioManager } from '../AudioManager';
import { Bomb } from '../Bomb';
import { NavigationGrid } from '../NavigationGrid';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { NoReticule } from '../Reticule';
import { EnemyTank } from './EnemyTank';
import { Tank } from './Tank';

export class SimpleMovingTank extends EnemyTank {
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
		const simpleMovingTankSpeed = 1.2;
		const simpleMovingTankSize = 30;
		const simpleMovingTankAggressionFactor = 15;
		const simpleMovingTankColor = '#fd8a8a';
		super(
			canvas,
			new NoReticule(),
			xPos,
			yPos,
			simpleMovingTankSpeed,
			simpleMovingTankSize,
			simpleMovingTankAggressionFactor,
			simpleMovingTankColor,
			obstacleCanvas,
			ammunition,
			bombs,
			navigationGrid,
			audioManager
		);
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

		const dx: number = playerTank.xPos + playerTank.size / 2 - this.xPos - this.tankMidpoint;
		const dy: number = playerTank.yPos + playerTank.size / 2 - this.yPos - this.tankMidpoint;
		let theta = Math.atan2(dy, dx);
		if (theta < 0) {
			theta += 2 * Math.PI;
		}
		this.aimAngle = theta;
	}
}
