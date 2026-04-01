import { SeededRandom } from './core/prng';
import type { DeepReadonly } from './core/stateUtils';
import type { GameState, MoveIntent, SimulationEvent, TankAction, TankStateView } from './core/types';

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

const MOVE_ARROWS: Record<MoveIntent, { dx: number; dy: number } | null> = {
	none: null,
	n: { dx: 0, dy: -1 },
	s: { dx: 0, dy: 1 },
	e: { dx: 1, dy: 0 },
	w: { dx: -1, dy: 0 },
	ne: { dx: 0.707, dy: -0.707 },
	nw: { dx: -0.707, dy: -0.707 },
	se: { dx: 0.707, dy: 0.707 },
	sw: { dx: -0.707, dy: 0.707 },
};

export class GameRenderer {
	private context: CanvasRenderingContext2D;
	private particles: VisualParticle[] = [];
	private lastActions: Record<string, TankAction> = {};
	public diagnosticsEnabled = false;

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

	public setLastActions(actions: Record<string, TankAction>): void {
		this.lastActions = actions;
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

	public render(currentState: DeepReadonly<GameState>, previousState: GameState | null, alpha: number): void {
		this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
		this.drawObstacles(currentState);
		this.drawBombs(currentState, previousState, alpha);
		this.drawProjectiles(currentState, previousState, alpha);
		this.drawTanks(currentState, previousState, alpha);
		this.drawParticles();
		if (this.diagnosticsEnabled) {
			this.drawDiagnostics(currentState, previousState, alpha);
		}
		this.drawOverlay(currentState);
	}

	private drawObstacles(state: DeepReadonly<GameState>): void {
		this.context.fillStyle = '#1d1c1a';
		for (const obstacle of state.obstacles) {
			this.context.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
		}
	}

	private drawBombs(state: DeepReadonly<GameState>, previousState: GameState | null, alpha: number): void {
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

	private drawProjectiles(state: DeepReadonly<GameState>, previousState: GameState | null, alpha: number): void {
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

	private drawTanks(state: DeepReadonly<GameState>, previousState: GameState | null, alpha: number): void {
		const tanks = [...state.tanks].sort((left, right) => {
			if (left.team === 'player' && right.team !== 'player') return 1;
			if (left.team !== 'player' && right.team === 'player') return -1;
			return left.id.localeCompare(right.id);
		});
		for (const tank of tanks) {
			this.drawTank(tank, previousState, alpha, tank.team === 'player');
		}
	}

	private drawTank(
		tank: DeepReadonly<TankStateView>,
		previousState: GameState | null,
		alpha: number,
		isPlayer: boolean
	): void {
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
		if (tank.invulnerabilityTicksRemaining > 0) {
			this.context.save();
			this.context.strokeStyle = 'rgba(255,255,255,0.9)';
			this.context.lineWidth = 3;
			this.context.setLineDash([4, 3]);
			this.context.strokeRect(x - 2, y - 2, tank.size + 4, tank.size + 4);
			this.context.restore();
		}
		this.context.strokeStyle = 'black';
		this.context.lineWidth = 2;
		this.context.strokeRect(x, y, tank.size, tank.size);
		this.context.beginPath();
		this.context.arc(x + tank.size / 2, y + tank.size / 2, tank.size / 3, 0, Math.PI * 2);
		this.context.stroke();
		this.context.beginPath();
		this.context.moveTo(x + tank.size / 2, y + tank.size / 2);
		this.context.lineTo(
			x + tank.size / 2 + Math.cos(tank.aimAngle) * tank.size,
			y + tank.size / 2 + Math.sin(tank.aimAngle) * tank.size
		);
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

		this.drawHealthBar(x, y, tank);
	}

	private drawHealthBar(x: number, y: number, tank: DeepReadonly<TankStateView>): void {
		const barWidth = tank.size;
		const barHeight = 5;
		const barX = x;
		const barY = y - 10;
		this.context.fillStyle = 'rgba(0,0,0,0.55)';
		this.context.fillRect(barX - 1, barY - 1, barWidth + 2, barHeight + 2);
		this.context.fillStyle = '#4b1f1f';
		this.context.fillRect(barX, barY, barWidth, barHeight);
		const ratio = tank.maxHealth > 0 ? tank.health / tank.maxHealth : 0;
		this.context.fillStyle = ratio > 0.66 ? '#47c96d' : ratio > 0.33 ? '#f1c94a' : '#db5b5b';
		this.context.fillRect(barX, barY, barWidth * ratio, barHeight);
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

	private drawOverlay(state: DeepReadonly<GameState>): void {
		if (state.status === 'running') {
			return;
		}
		let message = 'Draw';
		let color = '#d8d8d8';
		if (state.status === 'player_win') {
			message = 'Win';
			color = 'green';
		} else if (state.status === 'enemy_win') {
			message = 'Lose';
			color = 'red';
		} else if (state.status === 'team_win') {
			message = state.winnerTeam ? `${state.winnerTeam} wins` : 'Team wins';
			color = '#6ef3a5';
		}
		const fontSize = 100;
		this.context.font = `${fontSize}px Arial`;
		this.context.lineWidth = 5;
		this.context.strokeStyle = color;
		this.context.fillStyle = color;
		const textWidth = this.context.measureText(message).width;
		const x = (this.canvas.width - textWidth) / 2;
		const y = this.canvas.height / 2 + fontSize / 2;
		this.context.strokeText(message, x, y);
		this.context.fillText(message, x, y);
	}

	private drawDiagnostics(state: DeepReadonly<GameState>, previousState: GameState | null, alpha: number): void {
		const ctx = this.context;
		ctx.save();

		// Draw diagnostics badge
		ctx.font = '12px monospace';
		ctx.fillStyle = 'rgba(0,0,0,0.6)';
		ctx.fillRect(0, 0, 230, 20 + state.tanks.length * 14);
		ctx.fillStyle = '#0f0';
		ctx.fillText('DIAGNOSTICS ON', 5, 14);
		ctx.fillStyle = '#fff';
		state.tanks.forEach((tank, index) => {
			ctx.fillText(
				`${tank.id} HP ${tank.health}/${tank.maxHealth} invul=${tank.invulnerabilityTicksRemaining}`,
				5,
				32 + index * 14
			);
		});

		for (const tank of state.tanks) {
			if (tank.destroyed) continue;
			const action = this.lastActions[tank.id];
			if (!action) continue;

			const prevTank = previousState?.tanks.find((t) => t.id === tank.id);
			const cx = this.interpolate(
				prevTank ? prevTank.x + prevTank.size / 2 : tank.x + tank.size / 2,
				tank.x + tank.size / 2,
				alpha
			);
			const cy = this.interpolate(
				prevTank ? prevTank.y + prevTank.size / 2 : tank.y + tank.size / 2,
				tank.y + tank.size / 2,
				alpha
			);

			// Move direction arrow
			const moveDir = MOVE_ARROWS[action.move];
			if (moveDir) {
				const arrowLen = tank.size * 1.2;
				const ax = cx + moveDir.dx * arrowLen;
				const ay = cy + moveDir.dy * arrowLen;
				ctx.strokeStyle = '#0ff';
				ctx.lineWidth = 2;
				ctx.setLineDash([3, 3]);
				ctx.beginPath();
				ctx.moveTo(cx, cy);
				ctx.lineTo(ax, ay);
				ctx.stroke();
				// Arrow head
				const headLen = 6;
				const angle = Math.atan2(ay - cy, ax - cx);
				ctx.beginPath();
				ctx.moveTo(ax, ay);
				ctx.lineTo(ax - headLen * Math.cos(angle - 0.4), ay - headLen * Math.sin(angle - 0.4));
				ctx.moveTo(ax, ay);
				ctx.lineTo(ax - headLen * Math.cos(angle + 0.4), ay - headLen * Math.sin(angle + 0.4));
				ctx.stroke();
				ctx.setLineDash([]);
			}

			// Aim direction line (magenta, extending to arena edge)
			const aimLen = Math.max(state.arena.width, state.arena.height);
			const aimX = cx + Math.cos(action.aimAngle) * aimLen;
			const aimY = cy + Math.sin(action.aimAngle) * aimLen;
			ctx.strokeStyle = 'rgba(255,0,255,0.4)';
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(cx, cy);
			ctx.lineTo(aimX, aimY);
			ctx.stroke();

			// Fire indicator
			if (action.fire) {
				ctx.strokeStyle = 'red';
				ctx.lineWidth = 2;
				ctx.beginPath();
				ctx.arc(cx, cy, tank.size * 0.8, 0, Math.PI * 2);
				ctx.stroke();
			}

			// Bomb indicator
			if (action.plantBomb) {
				ctx.strokeStyle = 'orange';
				ctx.lineWidth = 2;
				ctx.beginPath();
				ctx.arc(cx, cy, tank.size, 0, Math.PI * 2);
				ctx.stroke();
			}

			// Label: move intent + fire/bomb
			const labelParts: string[] = [action.move];
			if (action.fire) labelParts.push('F');
			if (action.plantBomb) labelParts.push('B');
			ctx.font = '10px monospace';
			ctx.fillStyle = '#fff';
			ctx.strokeStyle = '#000';
			ctx.lineWidth = 2;
			const label = labelParts.join(' ');
			ctx.strokeText(label, cx - ctx.measureText(label).width / 2, cy - tank.size * 0.7);
			ctx.fillText(label, cx - ctx.measureText(label).width / 2, cy - tank.size * 0.7);
		}

		ctx.restore();
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
