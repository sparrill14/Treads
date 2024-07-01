import { GameCanvas } from '../GameCanvas';
import { Node } from '../Node';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { Tank } from '../tanks/Tank';

export class SimplePathfinder {
	public grid: Node[][] = [];
	public gridCellWidth = 30;
	public gridXLength: number;
	public gridYLength: number;
	public path: Node[] = [];
	public stationary: boolean;

	constructor(gameCanvas?: GameCanvas, obstacleCanvas?: ObstacleCanvas, stationary: boolean = true) {
		if (!gameCanvas || !obstacleCanvas || stationary) {
			this.gridXLength = 0;
			this.gridYLength = 0;
			this.stationary = true;
			return;
		}
		this.stationary = false;
		this.gridXLength = Math.floor(gameCanvas.width / this.gridCellWidth);
		this.gridYLength = Math.floor(gameCanvas.height / this.gridCellWidth);
		for (let x = 0; x < this.gridXLength; x++) {
			this.grid[x] = [];
			for (let y = 0; y < this.gridYLength; y++) {
				this.grid[x][y] = new Node(x, y);
				const gridXLeft = x * this.gridCellWidth;
				const gridXRight = gridXLeft + this.gridCellWidth;
				const gridYTop = y * this.gridCellWidth;
				const gridYBottom = gridYTop + this.gridCellWidth;
				this.grid[x][y].walkable = !obstacleCanvas.obstacles.some(
					(obs) => gridXRight > obs.xLeft && gridXLeft < obs.xRight && gridYBottom > obs.yTop && gridYTop < obs.yBottom
				);
			}
		}
	}

	moveTowardsGoal(currentPos: Node, goalPos: Node): Node {
		if (currentPos === goalPos) {
			return currentPos;
		}

		const possibleMoves = [
			new Node(currentPos.x + 1, currentPos.y), // Move right
			new Node(currentPos.x - 1, currentPos.y), // Move left
			new Node(currentPos.x, currentPos.y + 1), // Move down
			new Node(currentPos.x, currentPos.y - 1), // Move up
		];

		let bestMove = currentPos;
		let minDistance = Number.MAX_SAFE_INTEGER;
		for (const move of possibleMoves) {
			if (this.isWithinBounds(move) && this.isWalkable(move)) {
				const distance = Math.abs(move.x - goalPos.x) + Math.abs(move.y - goalPos.y);
				if (distance < minDistance) {
					minDistance = distance;
					bestMove = move;
				}
			}
		}
		return bestMove;
	}

	isWithinBounds(position: Node): boolean {
		return 0 <= position.x && position.x < this.gridXLength && 0 <= position.y && position.y < this.gridYLength;
	}

	isWalkable(position: Node): boolean {
		return this.grid[position.x][position.y].walkable;
	}

	findPath(start: Node, goal: Node): Node[] {
		let currentPos = start;
		const path: Node[] = [currentPos];
		let count: number = 10;
		while (count > 0) {
			currentPos = this.moveTowardsGoal(currentPos, goal);
			path.push(currentPos);
			count--;
		}
		return path;
	}

	getNodeFromTank(tank: Tank): Node {
		let xGridCoordinate: number = Math.floor((tank.xPosition + tank.size / 2) / this.gridCellWidth);
		let yGridCoordinate: number = Math.floor((tank.yPosition + tank.size / 2) / this.gridCellWidth);
		xGridCoordinate = Math.max(0, Math.min(xGridCoordinate, this.gridXLength - 1));
		yGridCoordinate = Math.max(0, Math.min(yGridCoordinate, this.gridYLength - 1));
		return this.grid[xGridCoordinate][yGridCoordinate];
	}

	getRandomNodeInRadiusOfTarget(target: Node, radius: number): Node {
		const candidateNodes: Node[] = [];
		for (let x = 0; x < this.gridXLength; x++) {
			for (let y = 0; y < this.gridYLength; y++) {
				const node = this.grid[x][y];
				const distance = Math.sqrt(Math.pow(node.x - target.x, 2) + Math.pow(node.y - target.y, 2));
				if (distance <= radius + 1 && distance >= radius - 1 && node.walkable) {
					candidateNodes.push(node);
				}
			}
		}
		const randomIndex = Math.floor(Math.random() * candidateNodes.length);
		return candidateNodes[randomIndex];
	}
}
