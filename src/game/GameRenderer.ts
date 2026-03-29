import { SeededRandom } from './core/prng';
import type { GameState, SimulationEvent, TankStateView } from './core/types';

interface VisualParticle {
	x: number;
	y: number;
	vx: number;
	vy: number;
	radius: number;
	life: number;
	maxLife: number;
	color: string;
}

export class GameRenderer {
	private context: CanvasRenderingContext2D;
	private particles: VisualParticle[] = [];

	constructor(public canvas: HTMLCanvasElement) {
		const context = this.canvas.getContext('2d');
		if (!context) {
			throw new Error('2d context not supported or canvas element not found.');
		}
		this.context = context;
	}

	public initializeCanvas(width: number, height: number): void {
		this.canvas.width = width;
		this.canvas.height = height;
	}

	public consumeEvents(events: SimulationEvent[]): void {
		for (const event of events) {
			switch (event.type) {
				case 'projectile-destroyed':
					this.spawnParticles(event.x, event.y, event.visualSeed, 15, ['#A9A9A9', '#ECE4E4', '#F5F5F5'], 0.35, 50);
					break;
				case 'bomb-exploded':
					this.spawnParticles(event.x, event.y, event.visualSeed, 50, ['red', 'yellow', 'orange'], 0.65, 80);
					break;
				case 'tank-destroyed':
					this.spawnParticles(event.x, event.y, event.visualSeed, 50, ['#A9A9A9', '#D3D3D3', '#F5F5F5'], 0.65, 70);
					break;
			}
		}
	}

	public updateVisuals(deltaSeconds: number): void {
		const frameScale = deltaSeconds * 60;
		for (const particle of this.particles) {
			particle.x += particle.vx * frameScale;
			particle.y += particle.vy * frameScale;
			particle.life -= deltaSeconds;
		}
		this.particles = this.particles.filter((particle) => particle.life > 0);
	}

	public render(currentState: GameState, previousState: GameState | null, alpha: number): void {
		this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
		this.drawObstacles(currentState);
		this.drawBombs(currentState, previousState, alpha);
		this.drawProjectiles(currentState, previousState, alpha);
		this.drawTanks(currentState, previousState, alpha);
		this.drawParticles();
		this.drawOverlay(currentState);
	}

	private drawObstacles(state: GameState): void {
		this.context.fillStyle = '#1d1c1a';
		for (const obstacle of state.obstacles) {
			this.context.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
		}
	}

	private drawBombs(state: GameState, previousState: GameState | null, alpha: number): void {
		for (const bomb of state.bombs) {
			const previousBomb = previousState?.bombs.find((candidate) => candidate.id === bomb.id) ?? null;
			const x = this.interpolate(previousBomb?.x ?? bomb.x, bomb.x, alpha);
			const y = this.interpolate(previousBomb?.y ?? bomb.y, bomb.y, alpha);
			this.context.beginPath();
			this.context.arc(x, y, bomb.radius, 0, Math.PI * 2);
			this.context.fillStyle = bomb.fuseTicksRemaining % 20 < 10 ? 'yellow' : 'red';
			this.context.fill();
			this.context.lineWidth = 3;
			this.context.strokeStyle = 'black';
			this.context.stroke();
			this.context.closePath();
		}
	}

	private drawProjectiles(state: GameState, previousState: GameState | null, alpha: number): void {
		for (const projectile of state.projectiles) {
			const previousProjectile = previousState?.projectiles.find((candidate) => candidate.id === projectile.id) ?? null;
			const x = this.interpolate(previousProjectile?.x ?? projectile.x, projectile.x, alpha);
			const y = this.interpolate(previousProjectile?.y ?? projectile.y, projectile.y, alpha);
			this.context.beginPath();
			this.context.arc(x, y, projectile.radius, 0, Math.PI * 2);
			this.context.fillStyle = 'white';
			this.context.fill();
			this.context.lineWidth = 2;
			this.context.strokeStyle = 'black';
			this.context.stroke();
			this.context.closePath();
		}
	}

	private drawTanks(state: GameState, previousState: GameState | null, alpha: number): void {
		const enemyTanks = state.tanks.filter((tank) => tank.team === 'enemy');
		const playerTank = state.tanks.find((tank) => tank.team === 'player') ?? null;
		for (const tank of enemyTanks) {
			this.drawTank(tank, previousState, alpha, false);
		}
		if (playerTank) {
			this.drawTank(playerTank, previousState, alpha, true);
		}
	}

	private drawTank(tank: TankStateView, previousState: GameState | null, alpha: number, isPlayer: boolean): void {
		const previousTank = previousState?.tanks.find((candidate) => candidate.id === tank.id) ?? null;
		const x = this.interpolate(previousTank?.x ?? tank.x, tank.x, alpha);
		const y = this.interpolate(previousTank?.y ?? tank.y, tank.y, alpha);
		if (tank.destroyed) {
			this.context.strokeStyle = tank.color;
			this.context.lineWidth = 5;
			const xLength = 12;
			const centerX = x + tank.size / 2;
			const centerY = y + tank.size / 2;
			this.context.beginPath();
			this.context.moveTo(centerX - xLength, centerY - xLength);
			this.context.lineTo(centerX + xLength, centerY + xLength);
			this.context.stroke();
			this.context.beginPath();
			this.context.moveTo(centerX - xLength, centerY + xLength);
			this.context.lineTo(centerX + xLength, centerY - xLength);
			this.context.stroke();
			return;
		}

		this.context.fillStyle = tank.color;
		this.context.fillRect(x, y, tank.size, tank.size);
		this.context.strokeStyle = 'black';
		this.context.lineWidth = 2;
		this.context.strokeRect(x, y, tank.size, tank.size);
		this.context.beginPath();
		this.context.arc(x + tank.size / 2, y + tank.size / 2, tank.size / 3, 0, Math.PI * 2);
		this.context.stroke();
		this.context.beginPath();
		this.context.moveTo(x + tank.size / 2, y + tank.size / 2);
		this.context.lineTo(x + tank.size / 2 + Math.cos(tank.aimAngle) * tank.size, y + tank.size / 2 + Math.sin(tank.aimAngle) * tank.size);
		this.context.lineWidth = 7;
		this.context.stroke();

		if (isPlayer && tank.aimTargetX !== null && tank.aimTargetY !== null) {
			this.context.strokeStyle = tank.color;
			this.context.setLineDash([4, 8]);
			this.context.beginPath();
			this.context.moveTo(x + tank.size / 2, y + tank.size / 2);
			this.context.lineTo(tank.aimTargetX, tank.aimTargetY);
			this.context.lineWidth = 5;
			this.context.stroke();
			this.context.setLineDash([]);
			const markerSize = 8;
			this.context.beginPath();
			this.context.moveTo(tank.aimTargetX - markerSize, tank.aimTargetY - markerSize);
			this.context.lineTo(tank.aimTargetX + markerSize, tank.aimTargetY + markerSize);
			this.context.stroke();
			this.context.beginPath();
			this.context.moveTo(tank.aimTargetX - markerSize, tank.aimTargetY + markerSize);
			this.context.lineTo(tank.aimTargetX + markerSize, tank.aimTargetY - markerSize);
			this.context.stroke();
		}
	}

	private drawParticles(): void {
		for (const particle of this.particles) {
			const radiusScale = particle.life / particle.maxLife;
			this.context.beginPath();
			this.context.arc(particle.x, particle.y, particle.radius * radiusScale, 0, Math.PI * 2);
			this.context.fillStyle = particle.color;
			this.context.fill();
			this.context.closePath();
		}
	}

	private drawOverlay(state: GameState): void {
		if (state.status === 'running') {
			return;
		}
		const message = state.status === 'player_win' ? 'Win' : 'Lose';
		const fontSize = 100;
		this.context.font = `${fontSize}px Arial`;
		this.context.lineWidth = 5;
		this.context.strokeStyle = state.status === 'player_win' ? 'green' : 'red';
		this.context.fillStyle = state.status === 'player_win' ? 'green' : 'red';
		const textWidth = this.context.measureText(message).width;
		const x = (this.canvas.width - textWidth) / 2;
		const y = this.canvas.height / 2 + fontSize / 2;
		this.context.strokeText(message, x, y);
		this.context.fillText(message, x, y);
	}

	private interpolate(previousValue: number, currentValue: number, alpha: number): number {
		return previousValue + (currentValue - previousValue) * alpha;
	}

	private spawnParticles(
		x: number,
		y: number,
		seed: number,
		count: number,
		colors: string[],
		lifeSeconds: number,
		velocityMagnitude: number
	): void {
		const rng = new SeededRandom(seed);
		for (let index = 0; index < count; index++) {
			const angle = rng.nextRange(0, Math.PI * 2);
			const speed = rng.nextRange(2, velocityMagnitude / 10);
			this.particles.push({
				x,
				y,
				vx: Math.cos(angle) * speed,
				vy: Math.sin(angle) * speed,
				radius: rng.nextRange(1, 3),
				life: lifeSeconds,
				maxLife: lifeSeconds,
				color: colors[rng.nextInt(0, colors.length - 1)],
			});
		}
	}
}
