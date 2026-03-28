import { Ammunition } from '../Ammunition';
import { Bomb } from '../Bomb';
import { GameCanvas } from '../GameCanvas';
import { Node } from '../Node';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { Tank } from '../tanks/Tank';

export class NavigationGrid {
	public grid: Node[][] = [];
	public gridCellWidth = 30;
	public gridXLength: number;
	public gridYLength: number;
	public path: Node[] = [];
	public stationary: boolean;
	public color: string;
	private gameCanvas: GameCanvas;
	private obstacleCanvas: ObstacleCanvas;

	constructor(gameCanvas: GameCanvas, obstacleCanvas: ObstacleCanvas, stationary = true, color = 'gray') {
		this.color = color;
		this.gameCanvas = gameCanvas;
		this.obstacleCanvas = obstacleCanvas;
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

	public draw(context: CanvasRenderingContext2D): void {
		context.lineWidth = 1;
		for (let i = 0; i <= this.gridYLength; i++) {
			context.fillStyle = 'blue';
			context.beginPath();
			context.moveTo(0, i * this.gridCellWidth);
			context.lineTo(this.gridXLength * this.gridCellWidth, i * this.gridCellWidth);
			context.stroke();
		}
		for (let j = 0; j <= this.gridXLength; j++) {
			context.fillStyle = 'blue';
			context.beginPath();
			context.moveTo(j * this.gridCellWidth, 0);
			context.lineTo(j * this.gridCellWidth, this.gridXLength * this.gridCellWidth);
			context.stroke();
		}
		context.fillStyle = 'red';
		for (let x = 0; x < this.gridXLength; x++) {
			for (let y = 0; y < this.gridYLength; y++) {
				if (this.grid[x][y].dangerous) {
					context.beginPath();
					context.arc(
						x * this.gridCellWidth + this.gridCellWidth / 2,
						y * this.gridCellWidth + this.gridCellWidth / 2,
						5,
						0,
						2 * Math.PI
					);
					context.fill();
				}
			}
		}
		context.fillStyle = this.color;
		this.path?.forEach((value: Node) => {
			context.beginPath();
			context.arc(
				value.x * this.gridCellWidth + this.gridCellWidth / 2,
				value.y * this.gridCellWidth + this.gridCellWidth / 2,
				5,
				0,
				2 * Math.PI
			);
			context.fill();
		});
	}

	public reset(): void {
		for (let x = 0; x < this.gridXLength; x++) {
			for (let y = 0; y < this.gridYLength; y++) {
				this.grid[x][y].f = 0;
				this.grid[x][y].g = 0;
				this.grid[x][y].h = 0;
				this.grid[x][y].parent = null;
				this.grid[x][y].dangerous = false;
			}
		}
	}

	getNodeFromTank(tank: Tank): Node {
		let xGridCoordinate: number = Math.floor((tank.xPosition + tank.size / 2) / this.gridCellWidth);
		let yGridCoordinate: number = Math.floor((tank.yPosition + tank.size / 2) / this.gridCellWidth);
		xGridCoordinate = Math.max(0, Math.min(xGridCoordinate, this.gridXLength - 1));
		yGridCoordinate = Math.max(0, Math.min(yGridCoordinate, this.gridYLength - 1));
		return this.grid[xGridCoordinate][yGridCoordinate];
	}

	getNodeFromAmmunition(ammunition: Ammunition): Node {
		let xGridCoordinate: number = Math.floor(ammunition.xPosition / this.gridCellWidth);
		let yGridCoordinate: number = Math.floor(ammunition.yPosition / this.gridCellWidth);
		xGridCoordinate = Math.max(0, Math.min(xGridCoordinate, this.gridXLength - 1));
		yGridCoordinate = Math.max(0, Math.min(yGridCoordinate, this.gridYLength - 1));
		return this.grid[xGridCoordinate][yGridCoordinate];
	}

	getNodeFromBomb(bomb: Bomb): Node {
		let xGridCoordinate: number = Math.floor(bomb.xPosition / this.gridCellWidth);
		let yGridCoordinate: number = Math.floor(bomb.yPosition / this.gridCellWidth);
		xGridCoordinate = Math.max(0, Math.min(xGridCoordinate, this.gridXLength - 1));
		yGridCoordinate = Math.max(0, Math.min(yGridCoordinate, this.gridYLength - 1));
		return this.grid[xGridCoordinate][yGridCoordinate];
	}

	getNodeFromXYCoordintate(x: number, y: number) {
		let xGridCoordinate: number = Math.floor(x / this.gridCellWidth);
		let yGridCoordinate: number = Math.floor(y / this.gridCellWidth);
		xGridCoordinate = Math.max(0, Math.min(xGridCoordinate, this.gridXLength - 1));
		yGridCoordinate = Math.max(0, Math.min(yGridCoordinate, this.gridYLength - 1));
		return this.grid[xGridCoordinate][yGridCoordinate];
	}

	private markDangerousWithBuffer(cx: number, cy: number, buffer: number): void {
		for (let dx = -buffer; dx <= buffer; dx++) {
			for (let dy = -buffer; dy <= buffer; dy++) {
				const nx = cx + dx;
				const ny = cy + dy;
				if (nx >= 0 && ny >= 0 && nx < this.gridXLength && ny < this.gridYLength) {
					this.grid[nx][ny].dangerous = true;
				}
			}
		}
	}

	getSafeNode(target: Node, radius: number, allAmmunition: Ammunition[], allBombs: Bomb[]): Node {
		for (let x = 0; x < this.gridXLength; x++) {
			for (let y = 0; y < this.gridYLength; y++) {
				this.grid[x][y].dangerous = false;
			}
		}
		for (const ammunition of allAmmunition) {
			if (ammunition.isDestroyed) {
				continue;
			}
			let predictedXPosition: number = ammunition.xPosition;
			let predictedYPosition: number = ammunition.yPosition;
			let predictedXVelocity: number = ammunition.xVelocity;
			let predictedYVelocity: number = ammunition.yVelocity;
			let predictedBounces: number = ammunition.bounces;
			while (predictedBounces <= ammunition.maxBounces) {
				predictedXPosition += predictedXVelocity;
				predictedYPosition += predictedYVelocity;
				if (predictedXPosition <= 0 || predictedXPosition > this.gameCanvas.width) {
					predictedXVelocity = -predictedXVelocity;
					predictedBounces++;
				}
				if (predictedYPosition <= 0 || predictedYPosition > this.gameCanvas.height) {
					predictedYVelocity = -predictedYVelocity;
					predictedBounces++;
				}
				this.obstacleCanvas.obstacles.forEach((obstacle) => {
					if (
						predictedXPosition > obstacle.xLeft &&
						predictedXPosition < obstacle.xRight &&
						predictedYPosition > obstacle.yTop &&
						predictedYPosition < obstacle.yBottom
					) {
						predictedBounces++;
						predictedXVelocity = -predictedXVelocity;
						predictedYVelocity = -predictedYVelocity;
					}
				});
				const node: Node = this.getNodeFromXYCoordintate(predictedXPosition, predictedYPosition);
				this.markDangerousWithBuffer(node.x, node.y, 1);
			}
		}
		for (const bomb of allBombs) {
			if (bomb.isDestroyed) {
				continue;
			}
			const blastRadiusCells = Math.ceil(bomb.blastRadius / this.gridCellWidth);
			const bombNode: Node = this.getNodeFromBomb(bomb);
			for (let dx = -blastRadiusCells; dx <= blastRadiusCells; dx++) {
				for (let dy = -blastRadiusCells; dy <= blastRadiusCells; dy++) {
					const nx = bombNode.x + dx;
					const ny = bombNode.y + dy;
					if (nx >= 0 && ny >= 0 && nx < this.gridXLength && ny < this.gridYLength) {
						this.grid[nx][ny].dangerous = true;
					}
				}
			}
		}
		const candidateNodes: Node[] = [];
		for (let x = 0; x < this.gridXLength; x++) {
			for (let y = 0; y < this.gridYLength; y++) {
				const node = this.grid[x][y];
				const distance = Math.sqrt(Math.pow(node.x - target.x, 2) + Math.pow(node.y - target.y, 2));
				if (distance <= radius + 1 && distance >= radius - 1 && node.walkable && !node.dangerous) {
					candidateNodes.push(node);
				}
			}
		}
		const randomIndex = Math.floor(Math.random() * candidateNodes.length);
		if (candidateNodes.length == 0) {
			const randomBackupNode: Node = this.getRandomNodeInRadiusOfTarget(target, radius);
			return randomBackupNode;
		} else {
			return candidateNodes[randomIndex];
		}
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
				this.path = path;
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
