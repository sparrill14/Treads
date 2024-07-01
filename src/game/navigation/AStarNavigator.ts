import { Ammunition } from '../Ammunition';
import { Bomb } from '../Bomb';
import { Node } from '../Node';
import { Direction, Tank } from '../tanks/Tank';
import { NavigationGrid } from './NavigationGrid';
import { Navigator } from './Navigator';

export class AStarNavigator extends Navigator {
	public aggressionFactor = 15;
	public pathRecaculationInterval = 60;
	private navigationGrid: NavigationGrid;
	private currentNode: Node | null = null;
	private path: Node[] | null = [];

	constructor(navigationGrid: NavigationGrid) {
		super();
		this.navigationGrid = navigationGrid;
	}

	public draw(context: CanvasRenderingContext2D): void {
		this.navigationGrid.draw(context);
	}

	public updatePosition(
		currentTank: Tank,
		playerTank: Tank,
		enemyTanks: Tank[],
		ammunition: Ammunition[],
		bombs: Bomb[]
	): void {
		this.pathRecaculationInterval -= 1;
		if (this.path == null || this.path.length == 0 || this.pathRecaculationInterval == 0) {
			this.navigationGrid.reset();
			const startNode: Node = this.navigationGrid.getNodeFromTank(currentTank);
			const targetNode: Node = this.navigationGrid.getNodeFromTank(playerTank);
			const destinationNode: Node = this.navigationGrid.getRandomNodeInRadiusOfTarget(
				targetNode,
				this.aggressionFactor
			);
			this.path = this.navigationGrid.aStar(startNode, destinationNode);
			this.pathRecaculationInterval = 60;
			if (this.path == null) {
				console.log(`Path is null`);
			}
		} else {
			this.currentNode = this.navigationGrid.getNodeFromTank(currentTank);
			const dx = this.path[0].x - this.currentNode.x;
			const dy = this.path[0].y - this.currentNode.y;

			if (currentTank.wasLastMoveBlocked && currentTank.consecutiveDirectionMoves > 2) {
				const randomDirection: Direction = currentTank.getRandomDirection();
				currentTank.moveInCardinalDirection(randomDirection);
				currentTank.consecutiveDirectionMoves = 0;
				const randomNumber: number = Math.random();
				if (dx === 1 && dy === 0) {
					if (randomNumber < 0.5) {
						currentTank.moveSouthEast();
					} else {
						currentTank.moveNorthEast();
					}
				} else if (dx === -1 && dy === 0) {
					if (randomNumber < 0.5) {
						currentTank.moveNorthWest();
					} else {
						currentTank.moveSouthWest();
					}
				} else if (dx === 0 && dy === 1) {
					if (randomNumber < 0.5) {
						currentTank.moveSouthEast();
					} else {
						currentTank.moveSouthWest();
					}
				} else if (dx === 0 && dy === -1) {
					if (randomNumber < 0.5) {
						currentTank.moveNorthWest();
					} else {
						currentTank.moveNorthEast();
					}
				} else if (dx === 1 && dy === 1) {
					if (randomNumber < 0.5) {
						currentTank.moveSouth();
					} else {
						currentTank.moveEast();
					}
				} else if (dx === 1 && dy === -1) {
					if (randomNumber < 0.5) {
						currentTank.moveNorth();
					} else {
						currentTank.moveEast();
					}
				} else if (dx === -1 && dy === 1) {
					if (randomNumber < 0.5) {
						currentTank.moveSouth();
					} else {
						currentTank.moveWest();
					}
				} else if (dx === -1 && dy === -1) {
					if (randomNumber < 0.5) {
						currentTank.moveNorth();
					} else {
						currentTank.moveWest();
					}
				}
			} else {
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
