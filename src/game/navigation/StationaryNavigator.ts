import { Ammunition } from '../Ammunition';
import { Bomb } from '../Bomb';
import { Tank } from '../tanks/Tank';
import { Navigator } from './Navigator';

export class StationaryNavigator extends Navigator {
	public aggressionFactor: number = 0;

	public draw(context: CanvasRenderingContext2D): void {
		return;
	}

	updatePosition(
		currentTank: Tank,
		playerTank: Tank,
		enemyTanks: Tank[],
		ammunition: Ammunition[],
		bombs: Bomb[]
	): void {
		return;
	}
}
