import { Ammunition } from '../Ammunition';
import { AudioManager } from '../AudioManager';
import { Bomb } from '../Bomb';
import { Node } from '../Node';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { Reticule } from '../Reticule';
import { Navigator } from '../navigation/Navigator';
import { Tank } from './Tank';

export class EnemyTank extends Tank {
	public navigator: Navigator;
	public aggressionFactor = 15; // Distance tank should maintain from its target
	public path: Node[] | null = [];
	public pathRecaculationInterval = 60;
	public drawNavigation = false;

	constructor(
		canvas: HTMLCanvasElement,
		reticule: Reticule,
		xPosition: number,
		yPosition: number,
		speed: number,
		size: number,
		aggressionFactor: number,
		color: string,
		obstacleCanvas: ObstacleCanvas,
		ammunition: Ammunition[],
		bombs: Bomb[],
		audioManager: AudioManager,
		navigator: Navigator
	) {
		super(canvas, reticule, xPosition, yPosition, speed, size, color, obstacleCanvas, ammunition, bombs, audioManager);
		this.aggressionFactor = aggressionFactor;
		this.navigator = navigator;
	}

	public override draw(context: CanvasRenderingContext2D): void {
		if (this.drawNavigation) {
			this.navigator.draw(context);
		}
		super.draw(context);
	}

	public override updatePosition(
		currentTank: Tank,
		playerTank: Tank,
		enemyTanks: Tank[],
		ammunition: Ammunition[],
		bombs: Bomb[]
	): void {
		this.navigator.updatePosition(currentTank, playerTank, enemyTanks, ammunition, bombs);
	}
}
