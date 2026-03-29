import { predictProjectileWillHitTank } from '../core/physics';
import { SeededRandom } from '../core/prng';
import { MinPriorityQueue } from './MinPriorityQueue';
import type {
	ArenaState,
	BombStateView,
	ObstacleStateView,
	ProjectileStateView,
	TankStateView,
} from '../core/types';

interface GridNode {
	x: number;
	y: number;
	walkable: boolean;
	dangerous: boolean;
	g: number;
	h: number;
	f: number;
	parent: GridNode | null;
}

export type NavigationMode = 'simple' | 'astar' | 'astar-avoidance';

export class NavigationPlanner {
	private readonly gridCellWidth = 30;
	private readonly gridXLength: number;
	private readonly gridYLength: number;
	private readonly grid: GridNode[][];

	constructor(private arena: ArenaState, private obstacles: ObstacleStateView[]) {
		this.gridXLength = Math.floor(arena.width / this.gridCellWidth);
		this.gridYLength = Math.floor(arena.height / this.gridCellWidth);
		this.grid = [];
		for (let x = 0; x < this.gridXLength; x++) {
			this.grid[x] = [];
			for (let y = 0; y < this.gridYLength; y++) {
				const cellLeft = x * this.gridCellWidth;
				const cellTop = y * this.gridCellWidth;
				const cellRight = cellLeft + this.gridCellWidth;
				const cellBottom = cellTop + this.gridCellWidth;
				this.grid[x][y] = {
					x,
					y,
					walkable: !obstacles.some(
						(obstacle) =>
							cellRight > obstacle.x &&
							cellLeft < obstacle.x + obstacle.width &&
							cellBottom > obstacle.y &&
							cellTop < obstacle.y + obstacle.height
					),
					dangerous: false,
					g: Number.POSITIVE_INFINITY,
					h: 0,
					f: Number.POSITIVE_INFINITY,
					parent: null,
				};
			}
		}
	}

	public getPath(
		mode: NavigationMode,
		currentTank: TankStateView,
		targetTank: TankStateView,
		aggressionFactor: number,
		closeApproach: boolean,
		projectiles: ProjectileStateView[],
		bombs: BombStateView[],
		rng: SeededRandom
	): Array<{ x: number; y: number }> {
		const start = this.getNodeFromTank(currentTank);
		const target = this.getNodeFromTank(targetTank);
		const getDestination = (): GridNode =>
			closeApproach ? this.getRandomNodeWithinRadius(target, aggressionFactor, rng) : this.getRandomNodeInRadius(target, aggressionFactor, rng);
		if (mode === 'simple') {
			const destination = getDestination();
			return this.findSimplePath(start, destination).map((node) => ({ x: node.x, y: node.y }));
		}

		this.reset();
		const destination =
			mode === 'astar-avoidance'
				? this.getSafeNode(target, aggressionFactor, closeApproach, currentTank, projectiles, bombs, rng)
				: getDestination();
		return (this.aStar(start, destination) ?? []).map((node) => ({ x: node.x, y: node.y }));
	}

	public getMoveForNextStep(
		currentTank: TankStateView,
		nextNode: { x: number; y: number } | undefined
	): 'none' | 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw' {
		if (!nextNode) {
			return 'none';
		}
		const currentNode = this.getNodeFromTank(currentTank);
		const dx = nextNode.x - currentNode.x;
		const dy = nextNode.y - currentNode.y;
		if (dx === 1 && dy === 0) return 'e';
		if (dx === -1 && dy === 0) return 'w';
		if (dx === 0 && dy === 1) return 's';
		if (dx === 0 && dy === -1) return 'n';
		if (dx === 1 && dy === 1) return 'se';
		if (dx === 1 && dy === -1) return 'ne';
		if (dx === -1 && dy === 1) return 'sw';
		if (dx === -1 && dy === -1) return 'nw';
		return 'none';
	}

	private reset(): void {
		for (let x = 0; x < this.gridXLength; x++) {
			for (let y = 0; y < this.gridYLength; y++) {
				const node = this.grid[x][y];
				node.g = Number.POSITIVE_INFINITY;
				node.h = 0;
				node.f = Number.POSITIVE_INFINITY;
				node.parent = null;
				node.dangerous = false;
			}
		}
	}

	private getNodeFromTank(tank: TankStateView): GridNode {
		const xCoordinate = Math.max(0, Math.min(Math.floor((tank.x + tank.size / 2) / this.gridCellWidth), this.gridXLength - 1));
		const yCoordinate = Math.max(0, Math.min(Math.floor((tank.y + tank.size / 2) / this.gridCellWidth), this.gridYLength - 1));
		return this.grid[xCoordinate][yCoordinate];
	}

	private getNodeFromPoint(x: number, y: number): GridNode {
		const xCoordinate = Math.max(0, Math.min(Math.floor(x / this.gridCellWidth), this.gridXLength - 1));
		const yCoordinate = Math.max(0, Math.min(Math.floor(y / this.gridCellWidth), this.gridYLength - 1));
		return this.grid[xCoordinate][yCoordinate];
	}

	private findSimplePath(start: GridNode, goal: GridNode): GridNode[] {
		let current = start;
		const path = [current];
		let remaining = 10;
		while (remaining > 0) {
			current = this.moveTowardsGoal(current, goal);
			path.push(current);
			remaining -= 1;
		}
		return path;
	}

	private moveTowardsGoal(current: GridNode, goal: GridNode): GridNode {
		if (current === goal) {
			return current;
		}
		const candidates = [
			{ x: current.x + 1, y: current.y },
			{ x: current.x - 1, y: current.y },
			{ x: current.x, y: current.y + 1 },
			{ x: current.x, y: current.y - 1 },
		];
		let best = current;
		let minDistance = Number.MAX_SAFE_INTEGER;
		for (const candidate of candidates) {
			if (this.isWithinBounds(candidate.x, candidate.y) && this.grid[candidate.x][candidate.y].walkable) {
				const distance = Math.abs(candidate.x - goal.x) + Math.abs(candidate.y - goal.y);
				if (distance < minDistance) {
					minDistance = distance;
					best = this.grid[candidate.x][candidate.y];
				}
			}
		}
		return best;
	}

	private getRandomNodeInRadius(target: GridNode, radius: number, rng: SeededRandom): GridNode {
		const candidates: GridNode[] = [];
		for (let x = 0; x < this.gridXLength; x++) {
			for (let y = 0; y < this.gridYLength; y++) {
				const node = this.grid[x][y];
				const distance = Math.sqrt((node.x - target.x) ** 2 + (node.y - target.y) ** 2);
				if (distance <= radius + 1 && distance >= radius - 1 && node.walkable) {
					candidates.push(node);
				}
			}
		}
		return candidates.length > 0 ? rng.pick(candidates) : target;
	}

	private getRandomNodeWithinRadius(target: GridNode, radius: number, rng: SeededRandom): GridNode {
		const candidates: GridNode[] = [];
		for (let x = 0; x < this.gridXLength; x++) {
			for (let y = 0; y < this.gridYLength; y++) {
				const node = this.grid[x][y];
				const distance = Math.sqrt((node.x - target.x) ** 2 + (node.y - target.y) ** 2);
				if (distance <= radius && node.walkable) {
					candidates.push(node);
				}
			}
		}
		return candidates.length > 0 ? rng.pick(candidates) : target;
	}

	private getSafeNode(
		target: GridNode,
		radius: number,
		closeApproach: boolean,
		currentTank: TankStateView,
		projectiles: ProjectileStateView[],
		bombs: BombStateView[],
		rng: SeededRandom
	): GridNode {
		for (const projectile of projectiles) {
			if (predictProjectileWillHitTank(projectile, currentTank, this.arena, this.obstacles)) {
				const node = this.getNodeFromPoint(projectile.x, projectile.y);
				this.markDangerous(node.x, node.y, 1);
			}
		}
		for (const bomb of bombs) {
			const bombNode = this.getNodeFromPoint(bomb.x, bomb.y);
			const blastRadiusCells = Math.ceil(bomb.blastRadius / this.gridCellWidth);
			this.markDangerous(bombNode.x, bombNode.y, blastRadiusCells);
		}

		const candidates: GridNode[] = [];
		for (let x = 0; x < this.gridXLength; x++) {
			for (let y = 0; y < this.gridYLength; y++) {
				const node = this.grid[x][y];
				const distance = Math.sqrt((node.x - target.x) ** 2 + (node.y - target.y) ** 2);
				const inRange = closeApproach ? distance <= radius : distance <= radius + 1 && distance >= radius - 1;
				if (inRange && node.walkable && !node.dangerous) {
					candidates.push(node);
				}
			}
		}
		if (candidates.length === 0) {
			return this.getRandomNodeInRadius(target, radius, rng);
		}
		return rng.pick(candidates);
	}

	private markDangerous(centerX: number, centerY: number, buffer: number): void {
		for (let dx = -buffer; dx <= buffer; dx++) {
			for (let dy = -buffer; dy <= buffer; dy++) {
				const nx = centerX + dx;
				const ny = centerY + dy;
				if (this.isWithinBounds(nx, ny)) {
					this.grid[nx][ny].dangerous = true;
				}
			}
		}
	}

	private aStar(start: GridNode, target: GridNode): GridNode[] | null {
		start.g = 0;
		start.h = this.getHeuristic(start, target);
		start.f = start.h;
		const openSet = new MinPriorityQueue<GridNode>();
		openSet.push(start, start.f);
		const closedSet = new Set<GridNode>();
		while (openSet.size > 0) {
			const currentEntry = openSet.pop();
			if (!currentEntry) {
				break;
			}
			const current = currentEntry.value;
			if (closedSet.has(current) || currentEntry.priority !== current.f) {
				continue;
			}
			if (current.x === target.x && current.y === target.y) {
				const path: GridNode[] = [];
				let cursor: GridNode | null = current;
				while (cursor !== null) {
					path.unshift(cursor);
					cursor = cursor.parent;
				}
				return path;
			}

			closedSet.add(current);
			for (const neighbor of this.getWalkableNeighbors(current)) {
				if (closedSet.has(neighbor)) {
					continue;
				}
				const tentativeG = current.g + (neighbor.x === current.x || neighbor.y === current.y ? 1 : Math.SQRT2);
				if (tentativeG >= neighbor.g) {
					continue;
				}
				neighbor.parent = current;
				neighbor.g = tentativeG;
				neighbor.h = this.getHeuristic(neighbor, target);
				neighbor.f = neighbor.g + neighbor.h;
				openSet.push(neighbor, neighbor.f);
			}
		}
		return null;
	}

	private getHeuristic(node: GridNode, target: GridNode): number {
		return Math.hypot(node.x - target.x, node.y - target.y);
	}

	private getWalkableNeighbors(node: GridNode): GridNode[] {
		const neighbors: GridNode[] = [];
		for (let dx = -1; dx <= 1; dx++) {
			for (let dy = -1; dy <= 1; dy++) {
				if (dx === 0 && dy === 0) {
					continue;
				}
				const x = node.x + dx;
				const y = node.y + dy;
				if (this.isWithinBounds(x, y) && this.grid[x][y].walkable) {
					neighbors.push(this.grid[x][y]);
				}
			}
		}
		return neighbors;
	}

	private isWithinBounds(x: number, y: number): boolean {
		return x >= 0 && y >= 0 && x < this.gridXLength && y < this.gridYLength;
	}
}
