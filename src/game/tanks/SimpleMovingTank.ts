import { PastelColorPalette } from '../../ui/PastelColorPalette';
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
		xPosition: number,
		yPosition: number,
		obstacleCanvas: ObstacleCanvas,
		ammunition: Ammunition[],
		bombs: Bomb[],
		navigationGrid: NavigationGrid,
		audioManager: AudioManager
	) {
		const simpleMovingTankSpeed = 1.2;
		const simpleMovingTankSize = 30;
		const simpleMovingTankAggressionFactor = 15;
		const simpleMovingTankColor = PastelColorPalette.CORAL_ORANGE;
		super(
			canvas,
			new NoReticule(),
			xPosition,
			yPosition,
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
