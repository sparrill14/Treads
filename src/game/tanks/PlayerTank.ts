import { Ammunition, PlayerAmmunition } from '../Ammunition';
import { AudioFile, AudioManager } from '../AudioManager';
import { Bomb, PlayerBomb } from '../Bomb';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { AdjustingCustomColorReticule, Reticule } from '../Reticule';
import { Tank } from './Tank';

export class PlayerTank extends Tank {
	constructor(
		canvas: HTMLCanvasElement,
		reticule: Reticule,
		xPos: number,
		yPos: number,
		speed: number,
		size: number,
		color: string,
		obstacleCanvas: ObstacleCanvas,
		ammunition: Ammunition[],
		bombs: Bomb[],
		audioManager: AudioManager
	) {
		super(canvas, reticule, xPos, yPos, speed, size, color, obstacleCanvas, ammunition, bombs, audioManager);

		document.addEventListener('keydown', (event: KeyboardEvent) => {
			if (this.keyStates.hasOwnProperty(event.key)) {
				this.keyStates[event.key] = true;
			}
		});

		document.addEventListener('keydown', (event: KeyboardEvent) => {
			if (event.code === 'Space') {
				this.plantBomb(this);
			}
		});

		document.addEventListener('keyup', (event: KeyboardEvent) => {
			if (this.keyStates.hasOwnProperty(event.key)) {
				this.keyStates[event.key] = false;
			}
		});

		canvas.addEventListener('mousemove', (event: MouseEvent) => {
			this.aimXPos = event.clientX - this.xOffset;
			this.aimYPos = event.clientY - this.yOffset;
		});

		canvas.addEventListener('click', (event: MouseEvent) => {
			this.shoot(this);
		});
	}

	public override aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
		if (this.isDestroyed) {
			return;
		}
		const dx: number = mouseXPos - this.xPos - this.tankMidpoint;
		const dy: number = mouseYpos - this.yPos - this.tankMidpoint;
		let theta = Math.atan2(dy, dx);
		if (theta < 0) {
			theta += 2 * Math.PI;
		}
		this.aimAngle = theta;
	}

	public override shoot(playerTank: Tank): void {
		if (!this.isDestroyed) {
			const availableAmmunitionIndex = this.ammunition.findIndex((ammunition) => ammunition.isDestroyed);
			if (availableAmmunitionIndex !== -1) {
				this.audioManager.play(AudioFile.TANK_FIRE);
				this.ammunition[availableAmmunitionIndex] = new PlayerAmmunition(
					this.xPos + this.size / 2,
					this.yPos + this.size / 2,
					this.aimAngle,
					this.canvasWidth,
					this.canvasHeight,
					false,
					this.audioManager
				);
			}
		}
		return;
	}
}

export class DefaultPlayerTank extends PlayerTank {
	constructor(
		canvas: HTMLCanvasElement,
		xPos: number,
		yPos: number,
		obstacleCanvas: ObstacleCanvas,
		audioManager: AudioManager
	) {
		const defaultPlayerTankSpeed = 2;
		const defaultPlayerTankSize = 30;
		const defaultPlayerTankColor = '#6384a1';
		const ammunition: Ammunition[] = [
			new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
		];
		const bombs: Bomb[] = [new PlayerBomb(0, 0, true, audioManager), new PlayerBomb(0, 0, true, audioManager)];
		super(
			canvas,
			new AdjustingCustomColorReticule(defaultPlayerTankSize, defaultPlayerTankColor, canvas.width),
			xPos,
			yPos,
			defaultPlayerTankSpeed,
			defaultPlayerTankSize,
			defaultPlayerTankColor,
			obstacleCanvas,
			ammunition,
			bombs,
			audioManager
		);
	}
}
