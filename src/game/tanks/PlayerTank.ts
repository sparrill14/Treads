import { InputManager } from '../../utils/InputManager';
import { Ammunition, PlayerAmmunition } from '../Ammunition';
import { AudioFile, AudioManager } from '../AudioManager';
import { Bomb, PlayerBomb } from '../Bomb';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { AdjustingCustomColorReticule, Reticule } from '../Reticule';
import { Tank } from './Tank';

export class PlayerTank extends Tank {
	private inputManager: InputManager;

	constructor(
		canvas: HTMLCanvasElement,
		reticule: Reticule,
		xPosition: number,
		yPosition: number,
		speed: number,
		size: number,
		color: string,
		obstacleCanvas: ObstacleCanvas,
		ammunition: Ammunition[],
		bombs: Bomb[],
		audioManager: AudioManager,
		inputManager: InputManager
	) {
		super(canvas, reticule, xPosition, yPosition, speed, size, color, obstacleCanvas, ammunition, bombs, audioManager);
		this.inputManager = inputManager;
	}

	public override updatePosition(
		_currentTank: Tank,
		_playerTank: Tank,
		_enemyTanks: Tank[],
		_ammunition: Ammunition[],
		_bombs: Bomb[]
	): void {
		// Handle queued actions
		if (this.inputManager.consumeShoot()) {
			this.shoot(this);
		}
		if (this.inputManager.consumeBomb()) {
			this.plantBomb(this);
		}

		// Update aim from mouse position
		this.aimXPos = this.inputManager.mouseX;
		this.aimYPos = this.inputManager.mouseY;

		// Move the tank
		if (this.inputManager.up() && this.inputManager.right()) {
			this.moveNorthEast();
		} else if (this.inputManager.up() && this.inputManager.left()) {
			this.moveNorthWest();
		} else if (this.inputManager.down() && this.inputManager.right()) {
			this.moveSouthEast();
		} else if (this.inputManager.down() && this.inputManager.left()) {
			this.moveSouthWest();
		} else if (this.inputManager.up()) {
			this.moveNorth();
		} else if (this.inputManager.down()) {
			this.moveSouth();
		} else if (this.inputManager.left()) {
			this.moveWest();
		} else if (this.inputManager.right()) {
			this.moveEast();
		}

		this.xLeft = this.xPosition;
		this.xRight = this.xPosition + this.size;
		this.yTop = this.yPosition;
		this.yBottom = this.yPosition + this.size;
	}

	public override plantBomb(_playerTank: Tank): void {
		if (this.isDestroyed) {
			return;
		}
		const availableBombIndex = this.bombs.findIndex((bomb) => bomb.isDestroyed && !bomb.isExploding());
		if (availableBombIndex !== -1) {
			this.bombs[availableBombIndex].xPosition = this.xPosition + this.size / 2;
			this.bombs[availableBombIndex].yPosition = this.yPosition + this.size / 2;
			this.bombs[availableBombIndex].setFuse();
		}
	}

	public override aim(mouseXPos: number, mouseYpos: number, _playerTank: Tank): void {
		if (this.isDestroyed) {
			return;
		}
		const dx: number = mouseXPos - this.xPosition - this.tankMidpoint;
		const dy: number = mouseYpos - this.yPosition - this.tankMidpoint;
		let theta = Math.atan2(dy, dx);
		if (theta < 0) {
			theta += 2 * Math.PI;
		}
		this.aimAngle = theta;
	}

	public override shoot(_playerTank: Tank): void {
		if (!this.isDestroyed) {
			const availableAmmunitionIndex = this.ammunition.findIndex((ammunition) => ammunition.isDestroyed);
			if (availableAmmunitionIndex !== -1) {
				this.audioManager.play(AudioFile.TANK_FIRE);
				this.ammunition[availableAmmunitionIndex] = new PlayerAmmunition(
					this.gunBarrellEndX,
					this.gunBarrellEndY,
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
		audioManager: AudioManager,
		inputManager: InputManager
	) {
		const defaultPlayerTankSpeed = 90;
		const defaultPlayerTankSize = 30;
		const defaultPlayerTankColor = '#4f6d7a';
		const ammunition: Ammunition[] = [
			new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
			new PlayerAmmunition(0, 0, 0, 0, 0, true, audioManager),
		];
		const bombs: Bomb[] = [new PlayerBomb(0, 0, true, audioManager), new PlayerBomb(0, 0, true, audioManager)];
		const rect: DOMRect = canvas.getBoundingClientRect();
		const viewportWidth: number = window.innerWidth;
		const distanceFromLeft: number = rect.left;
		const distanceFromRight: number = viewportWidth - rect.right;
		const maxReticuleLength: number = canvas.width + Math.max(distanceFromLeft, distanceFromRight);
		super(
			canvas,
			new AdjustingCustomColorReticule(defaultPlayerTankSize, defaultPlayerTankColor, maxReticuleLength),
			xPos,
			yPos,
			defaultPlayerTankSpeed,
			defaultPlayerTankSize,
			defaultPlayerTankColor,
			obstacleCanvas,
			ammunition,
			bombs,
			audioManager,
			inputManager
		);
	}
}
