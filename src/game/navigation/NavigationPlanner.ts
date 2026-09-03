import { predictProjectileWillHitTank, stepProjectile } from '../core/physics';
import { SeededRandom } from '../core/prng';
import type { ArenaState, BombStateView, ObstacleStateView, ProjectileStateView, TankStateView } from '../core/types';
import { MinPriorityQueue } from './MinPriorityQueue';

interface GridNode {
	x: number;
	y: number;
	walkable: boolean;
	dangerous: boolean;
	g: number;
	h: number;
	f: number;
	parent: GridNode | null;
	closed: number; // generation counter — matches currentGeneration when closed
}

export type NavigationMode = 'simple' | 'astar' | 'astar-avoidance';

export class NavigationPlanner {
	private readonly gridCellWidth = 15;
	private readonly projectileDangerHorizonTicks = 24;
	private readonly gridXLength: number;
	private readonly gridYLength: number;
	private readonly grid: GridNode[][];

	// Reusable buffers to avoid allocations in hot paths
	private readonly neighborBuf: GridNode[] = new Array(8);
	private readonly openSet = new MinPriorityQueue<GridNode>();
	private currentGeneration = 0;

	constructor(
		private arena: ArenaState,
		private obstacles: ObstacleStateView[],
		tankSize = 30
	) {
		this.gridXLength = Math.floor(arena.width / this.gridCellWidth);
		this.gridYLength = Math.floor(arena.height / this.gridCellWidth);
		// Inflate obstacles by half the tank size so A* paths keep the tank
		// body clear of obstacle edges (configuration-space approach).
		const padding = tankSize / 2;
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
							cellRight > obstacle.x - padding &&
							cellLeft < obstacle.x + obstacle.width + padding &&
							cellBottom > obstacle.y - padding &&
							cellTop < obstacle.y + obstacle.height + padding
					),
					dangerous: false,
					g: Number.POSITIVE_INFINITY,
					h: 0,
					f: Number.POSITIVE_INFINITY,
					parent: null,
					closed: -1,
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
	): { x: number; y: number }[] {
		const start = this.getNodeFromTank(currentTank);
		const target = this.getNodeFromTank(targetTank);
		const getDestination = (): GridNode =>
			closeApproach
				? this.getRandomNodeWithinRadius(target, aggressionFactor, rng)
				: this.getRandomNodeInRadius(target, aggressionFactor, rng);
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
		const xCoordinate = Math.max(
			0,
			Math.min(Math.floor((tank.x + tank.size / 2) / this.gridCellWidth), this.gridXLength - 1)
		);
		const yCoordinate = Math.max(
			0,
			Math.min(Math.floor((tank.y + tank.size / 2) / this.gridCellWidth), this.gridYLength - 1)
		);
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
		const r2lo = (radius - 1) * (radius - 1);
		const r2hi = (radius + 1) * (radius + 1);
		const xMin = Math.max(0, Math.ceil(target.x - radius - 1));
		const xMax = Math.min(this.gridXLength - 1, Math.floor(target.x + radius + 1));
		const yMin = Math.max(0, Math.ceil(target.y - radius - 1));
		const yMax = Math.min(this.gridYLength - 1, Math.floor(target.y + radius + 1));
		for (let x = xMin; x <= xMax; x++) {
			for (let y = yMin; y <= yMax; y++) {
				const node = this.grid[x][y];
				const d2 = (node.x - target.x) * (node.x - target.x) + (node.y - target.y) * (node.y - target.y);
				if (d2 <= r2hi && d2 >= r2lo && node.walkable) {
					candidates.push(node);
				}
			}
		}
		return candidates.length > 0 ? rng.pick(candidates) : target;
	}

	private getRandomNodeWithinRadius(target: GridNode, radius: number, rng: SeededRandom): GridNode {
		const candidates: GridNode[] = [];
		const r2 = radius * radius;
		const xMin = Math.max(0, Math.ceil(target.x - radius));
		const xMax = Math.min(this.gridXLength - 1, Math.floor(target.x + radius));
		const yMin = Math.max(0, Math.ceil(target.y - radius));
		const yMax = Math.min(this.gridYLength - 1, Math.floor(target.y + radius));
		for (let x = xMin; x <= xMax; x++) {
			for (let y = yMin; y <= yMax; y++) {
				const node = this.grid[x][y];
				const d2 = (node.x - target.x) * (node.x - target.x) + (node.y - target.y) * (node.y - target.y);
				if (d2 <= r2 && node.walkable) {
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
				this.markProjectileDangerTrail(projectile);
			}
		}
		for (const bomb of bombs) {
			const bombNode = this.getNodeFromPoint(bomb.x, bomb.y);
			const blastRadiusCells = Math.ceil(bomb.blastRadius / this.gridCellWidth);
			this.markDangerous(bombNode.x, bombNode.y, blastRadiusCells);
		}

		const candidates: GridNode[] = [];
		const r2lo = closeApproach ? 0 : (radius - 1) * (radius - 1);
		const r2hi = closeApproach ? radius * radius : (radius + 1) * (radius + 1);
		const scanRadius = closeApproach ? radius : radius + 1;
		const xMin = Math.max(0, Math.ceil(target.x - scanRadius));
		const xMax = Math.min(this.gridXLength - 1, Math.floor(target.x + scanRadius));
		const yMin = Math.max(0, Math.ceil(target.y - scanRadius));
		const yMax = Math.min(this.gridYLength - 1, Math.floor(target.y + scanRadius));
		for (let x = xMin; x <= xMax; x++) {
			for (let y = yMin; y <= yMax; y++) {
				const node = this.grid[x][y];
				const d2 = (node.x - target.x) * (node.x - target.x) + (node.y - target.y) * (node.y - target.y);
				if (d2 >= r2lo && d2 <= r2hi && node.walkable && !node.dangerous) {
					candidates.push(node);
				}
			}
		}
		if (candidates.length === 0) {
			return this.getRandomNodeInRadius(target, radius, rng);
		}
		return rng.pick(candidates);
	}

	private markProjectileDangerTrail(projectile: ProjectileStateView): void {
		const simulated: ProjectileStateView = { ...projectile };
		for (let tick = 0; tick < this.projectileDangerHorizonTicks; tick++) {
			if (simulated.bounces > simulated.maxBounces) {
				break;
			}
			const node = this.getNodeFromPoint(simulated.x, simulated.y);
			this.markDangerous(node.x, node.y, 1);
			stepProjectile(simulated, this.arena, this.obstacles, true);
		}
	}

	private markDangerous(centerX: number, centerY: number, buffer: number): void {
		const xMin = Math.max(0, centerX - buffer);
		const xMax = Math.min(this.gridXLength - 1, centerX + buffer);
		const yMin = Math.max(0, centerY - buffer);
		const yMax = Math.min(this.gridYLength - 1, centerY + buffer);
		for (let nx = xMin; nx <= xMax; nx++) {
			for (let ny = yMin; ny <= yMax; ny++) {
				this.grid[nx][ny].dangerous = true;
			}
		}
	}

	private aStar(start: GridNode, target: GridNode): GridNode[] | null {
		this.currentGeneration += 1;
		const gen = this.currentGeneration;
		const openSet = this.openSet;
		openSet.clear();

		start.g = 0;
		start.h = this.heuristic(start, target);
		start.f = start.h;
		openSet.push(start, start.f);

		while (openSet.size > 0) {
			if (!openSet.pop()) {
				break;
			}
			const current = openSet.popValue as GridNode;
			const currentPriority = openSet.popPriority;

			if (current.closed === gen || currentPriority !== current.f) {
				continue;
			}
			if (current.x === target.x && current.y === target.y) {
				return this.reconstructPath(current);
			}

			current.closed = gen;
			const neighborCount = this.fillWalkableNeighbors(current);
			for (let i = 0; i < neighborCount; i++) {
				const neighbor = this.neighborBuf[i];
				if (neighbor.closed === gen) {
					continue;
				}
				const tentativeG = current.g + (neighbor.x === current.x || neighbor.y === current.y ? 1 : Math.SQRT2);
				if (tentativeG >= neighbor.g) {
					continue;
				}
				neighbor.parent = current;
				neighbor.g = tentativeG;
				neighbor.h = this.heuristic(neighbor, target);
				neighbor.f = neighbor.g + neighbor.h;
				openSet.push(neighbor, neighbor.f);
			}
		}
		return null;
	}

	private reconstructPath(end: GridNode): GridNode[] {
		const path: GridNode[] = [];
		let cursor: GridNode | null = end;
		while (cursor !== null) {
			path.push(cursor);
			cursor = cursor.parent;
		}
		path.reverse();
		return path;
	}

	private heuristic(node: GridNode, target: GridNode): number {
		const dx = node.x - target.x;
		const dy = node.y - target.y;
		return Math.sqrt(dx * dx + dy * dy);
	}

	/** Fill neighborBuf with walkable neighbors, return count. Zero allocations. */
	private fillWalkableNeighbors(node: GridNode): number {
		let count = 0;
		const nx = node.x;
		const ny = node.y;
		const gx = this.gridXLength;
		const gy = this.gridYLength;
		const grid = this.grid;
		for (let dx = -1; dx <= 1; dx++) {
			const x = nx + dx;
			if (x < 0 || x >= gx) continue;
			for (let dy = -1; dy <= 1; dy++) {
				if (dx === 0 && dy === 0) continue;
				const y = ny + dy;
				if (y < 0 || y >= gy) continue;
				const candidate = grid[x][y];
				if (!candidate.walkable) continue;
				// Prevent corner-cutting: diagonal moves require both
				// adjacent cardinal cells to be walkable.
				if (dx !== 0 && dy !== 0) {
					if (!grid[nx + dx][ny].walkable || !grid[nx][ny + dy].walkable) continue;
				}
				this.neighborBuf[count++] = candidate;
			}
		}
		return count;
	}

	private isWithinBounds(x: number, y: number): boolean {
		return x >= 0 && y >= 0 && x < this.gridXLength && y < this.gridYLength;
	}

	/**
	 * Return the A* path distance (in world units) between two points.
	 * Falls back to Euclidean distance when either point is inside an
	 * obstacle or no walkable path exists.
	 */
	public getPathDistance(fromX: number, fromY: number, toX: number, toY: number): number {
		return this.getPathGuidance(fromX, fromY, toX, toY).distance;
	}

	/**
	 * Return the first collision-free A* step and total path distance between two points.
	 * The next point is expressed in world coordinates so policy observations can expose
	 * an immediately actionable navigation bearing without coupling the policy to grid size.
	 */
	public getPathGuidance(
		fromX: number,
		fromY: number,
		toX: number,
		toY: number
	): { nextX: number; nextY: number; distance: number; reachable: boolean } {
		const startNode = this.getNodeFromPoint(fromX, fromY);
		const endNode = this.getNodeFromPoint(toX, toY);
		if (startNode === endNode) {
			const dx = toX - fromX;
			const dy = toY - fromY;
			return { nextX: toX, nextY: toY, distance: Math.sqrt(dx * dx + dy * dy), reachable: true };
		}
		if (!startNode.walkable || !endNode.walkable) {
			const dx = toX - fromX;
			const dy = toY - fromY;
			return { nextX: toX, nextY: toY, distance: Math.sqrt(dx * dx + dy * dy), reachable: false };
		}
		this.reset();
		const path = this.aStar(startNode, endNode);
		if (!path) {
			const dx = toX - fromX;
			const dy = toY - fromY;
			return { nextX: toX, nextY: toY, distance: Math.sqrt(dx * dx + dy * dy), reachable: false };
		}
		const nextNode = path[Math.min(1, path.length - 1)];
		return {
			nextX: (nextNode.x + 0.5) * this.gridCellWidth,
			nextY: (nextNode.y + 0.5) * this.gridCellWidth,
			distance: path[path.length - 1].g * this.gridCellWidth,
			reachable: true,
		};
	}
}
