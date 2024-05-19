import { Ammunition } from '../Ammunition';
import { AudioManager } from '../AudioManager';
import { Bomb } from '../Bomb';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { Reticule } from '../Reticule';
import { Tank } from './Tank';

export class EnemyTank extends Tank {
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
	}
}
