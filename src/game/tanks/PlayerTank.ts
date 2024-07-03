import { KeyStates } from '../../utils/KeyStates';
import { Ammunition, PlayerAmmunition } from '../Ammunition';
import { AudioFile, AudioManager } from '../AudioManager';
import { Bomb, PlayerBomb } from '../Bomb';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { AdjustingCustomColorReticule, Reticule } from '../Reticule';
import { Tank } from './Tank';

export class PlayerTank extends Tank {
	public keyStates: KeyStates = {
		ArrowUp: false,
		ArrowDown: false,
		ArrowLeft: false,
		ArrowRight: false,
		w: false,
		a: false,
		s: false,
		d: false,
		W: false,
		A: false,
		S: false,
		D: false,
	};

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
		audioManager: AudioManager
	) {
		super(canvas, reticule, xPosition, yPosition, speed, size, color, obstacleCanvas, ammunition, bombs, audioManager);

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
		document.addEventListener('mousemove', (event: MouseEvent) => {
			this.aimXPos = event.clientX - this.xOffset;
			this.aimYPos = event.clientY - this.yOffset;
		});
		document.addEventListener('click', (event: MouseEvent) => {
			if (canvas.contains(event.target as Node)) {
				this.shoot(this);
			}
		});
	}

	public override updatePosition(
		currentTank: Tank,
		playerTank: Tank,
		enemyTanks: Tank[],
		ammunition: Ammunition[],
		bombs: Bomb[]
	): void {
		// Move the tank
		if (this.up() && this.right()) {
			this.moveNorthEast();
		} else if (this.up() && this.left()) {
			this.moveNorthWest();
		} else if (this.down() && this.right()) {
			this.moveSouthEast();
		} else if (this.down() && this.left()) {
			this.moveSouthWest();
		} else if (this.up()) {
			this.moveNorth();
		} else if (this.down()) {
			this.moveSouth();
		} else if (this.left()) {
			this.moveWest();
		} else if (this.right()) {
			this.moveEast();
		}

		this.xLeft = this.xPosition;
		this.xRight = this.xPosition + this.size;
		this.yTop = this.yPosition;
		this.yBottom = this.yPosition + this.size;
	}

	public override plantBomb(playerTank: Tank): void {
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

	public override aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
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

	public override shoot(playerTank: Tank): void {
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

	public up(): boolean {
		return this.keyStates.ArrowUp || this.keyStates.w || this.keyStates.W;
	}

	public down(): boolean {
		return this.keyStates.ArrowDown || this.keyStates.s || this.keyStates.S;
	}

	public left(): boolean {
		return this.keyStates.ArrowLeft || this.keyStates.a || this.keyStates.A;
	}

	public right(): boolean {
		return this.keyStates.ArrowRight || this.keyStates.d || this.keyStates.D;
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
			audioManager
		);
	}
}
