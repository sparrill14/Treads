import { GameCanvas } from './GameCanvas';
import { ObstacleCanvas } from './ObstacleCanvas';
import { Tank } from './tanks/Tank';

export class Node {
	public x: number;
	public y: number;
	public g = 0;
	public h = 0;
	public f = 0;
	public walkable = true;
	public parent: Node | null = null;

	constructor(x: number, y: number) {
		this.x = x;
		this.y = y;
	}
}

export class NavigationGrid {
	public grid: Node[][] = [];
	public gridCellWidth = 30;
	public gridXLength: number;
	public gridYLength: number;
	public path: Node[] = [];

	constructor(gameCanvas: GameCanvas, obstacleCanvas: ObstacleCanvas) {
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

	reset(): void {
		for (let x = 0; x < this.gridXLength; x++) {
			for (let y = 0; y < this.gridYLength; y++) {
				this.grid[x][y].f = 0;
				this.grid[x][y].g = 0;
				this.grid[x][y].h = 0;
				this.grid[x][y].parent = null;
			}
		}
	}

	getNodeFromTank(tank: Tank): Node {
		let xGridCoordinate: number = Math.floor((tank.xPos + tank.size / 2) / this.gridCellWidth);
		let yGridCoordinate: number = Math.floor((tank.yPos + tank.size / 2) / this.gridCellWidth);

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

	aStar(start: Node, target: Node): Node[] | null {
		const openSet: Node[] = [start];
		const closedSet = new Set<Node>();

		while (openSet.length > 0) {
			let current: Node | null = openSet.sort((a, b) => a.f - b.f)[0];

			if (current.x === target.x && current.y === target.y) {
				const path = [];
				while (current !== null) {
					path.unshift(current);
					current = current.parent;
				}
				return path;
			}

			openSet.splice(openSet.indexOf(current), 1);
			closedSet.add(current);

			const neighbors = this.getWalkableNeighbors(current);

			for (const neighbor of neighbors) {
				if (closedSet.has(neighbor)) {
					continue;
				}

				const tentativeG = current.g + (neighbor.x - current.x === 0 || neighbor.y - current.y === 0 ? 1 : Math.SQRT2);
				if (!openSet.includes(neighbor)) {
					openSet.push(neighbor);
				} else if (tentativeG >= neighbor.g) {
					continue;
				}
				neighbor.parent = current;
				neighbor.g = tentativeG;
				neighbor.h = Math.round(Math.sqrt(Math.pow(neighbor.x - target.x, 2) + Math.pow(neighbor.y - target.y, 2)));
				neighbor.f = neighbor.g + neighbor.h;
			}
		}
		return null;
	}

	getWalkableNeighbors(node: Node): Node[] {
		// Get the 8 nodes surrounding the current only if its walkable
		const neighbors: Node[] = [];
		for (let dx = -1; dx <= 1; dx++) {
			for (let dy = -1; dy <= 1; dy++) {
				if (dx === 0 && dy === 0) {
					continue;
				}

				const x = node.x + dx;
				const y = node.y + dy;

				if (x >= 0 && y >= 0 && x < this.gridXLength && y < this.gridYLength && this.grid[x][y].walkable) {
					neighbors.push(this.grid[x][y]);
				}
			}
		}
		return neighbors;
	}
}
