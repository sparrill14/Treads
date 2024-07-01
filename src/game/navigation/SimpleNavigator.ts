import { Ammunition } from '../Ammunition';
import { Bomb } from '../Bomb';
import { Node } from '../Node';
import { Tank } from '../tanks/Tank';
import { Navigator } from './Navigator';
import { SimplePathfinder } from './SimplePathFinder';

export class SimpleNavigator extends Navigator {
	public pathRecaculationInterval = 120;
	public aggressionFactor = 10;
	private simplePathFinder: SimplePathfinder;
	private currentNode: Node | null = null;
	private path: Node[] | null = [];

	constructor(pathfinder: SimplePathfinder) {
		super();
		this.simplePathFinder = pathfinder;
	}

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
		this.pathRecaculationInterval -= 1;
		if (this.path == null || this.path.length == 0 || this.pathRecaculationInterval == 0) {
			const startNode: Node = this.simplePathFinder.getNodeFromTank(currentTank);
			const targetNode: Node = this.simplePathFinder.getNodeFromTank(playerTank);
			const destinationNode: Node = this.simplePathFinder.getRandomNodeInRadiusOfTarget(
				targetNode,
				this.aggressionFactor
			);
			this.pathRecaculationInterval = 120;
			this.path = this.simplePathFinder.findPath(startNode, destinationNode);
		} else {
			this.currentNode = this.simplePathFinder.getNodeFromTank(currentTank);
			const dx = this.path[0].x - this.currentNode.x;
			const dy = this.path[0].y - this.currentNode.y;

			if (this.path && this.path.length > 1) {
				if (dx === 1 && dy === 0) {
					currentTank.moveEast();
				} else if (dx === -1 && dy === 0) {
					currentTank.moveWest();
				} else if (dx === 0 && dy === 1) {
					currentTank.moveSouth();
				} else if (dx === 0 && dy === -1) {
					currentTank.moveNorth();
				} else if (dx === 1 && dy === 1) {
					currentTank.moveSouthEast();
				} else if (dx === 1 && dy === -1) {
					currentTank.moveNorthEast();
				} else if (dx === -1 && dy === 1) {
					currentTank.moveSouthWest();
				} else if (dx === -1 && dy === -1) {
					currentTank.moveNorthWest();
				}
			}

			if (this.path[0].x == this.currentNode.x && this.path[0].y == this.currentNode.y) {
				this.path.splice(0, 1);
			}
		}

		currentTank.xLeft = currentTank.xPosition;
		currentTank.xRight = currentTank.xPosition + currentTank.size;
		currentTank.yTop = currentTank.yPosition;
		currentTank.yBottom = currentTank.yPosition + currentTank.size;
	}
}
