import { Ammunition } from '../Ammunition';
import { Bomb } from '../Bomb';
import { Tank } from '../tanks/Tank';

export abstract class Navigator {
	abstract aggressionFactor: number;

	abstract draw(context: CanvasRenderingContext2D): void;

	abstract updatePosition(
		currentTank: Tank,
		playerTank: Tank,
		enemyTanks: Tank[],
		ammunition: Ammunition[],
		bombs: Bomb[]
	): void;
}
