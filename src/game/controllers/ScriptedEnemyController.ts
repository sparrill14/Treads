import {
	aimAngleAtTarget,
	computeClearGunBarrelEnd,
	deriveAimTarget,
	getMoveDelta,
	normalizeAngle,
	obstacleIntersectsTank,
} from '../core/geometry';
import { bombWouldHitTank, predictProjectileWillHitTank } from '../core/physics';
import { mixSeed, SeededRandom } from '../core/prng';
import { getBombSpec, getProjectileSpec } from '../core/specs';
import type {
	MatchInit,
	MoveIntent,
	ProjectileStateView,
	TankAction,
	TankController,
	TankObservation,
	TankStateView,
} from '../core/types';
import { NavigationPlanner, type NavigationMode } from '../navigation/NavigationPlanner';

import { SIMULATION_TICK_SECONDS as TICK_SECONDS } from '../core/types';

type HeuristicProfile = 'default' | 'elite' | 'advanced';
type TacticalRole = 'auto' | 'pressure' | 'flank' | 'zone';

const ALL_MOVES: MoveIntent[] = ['none', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];

interface ScriptedEnemyControllerOptions {
	navigationMode: NavigationMode | 'stationary';
	randomAim: boolean;
	recalculationInterval: number;
	aggressionFactor: number;
	heuristicProfile?: HeuristicProfile;
	tacticalRole?: TacticalRole;
}

export class ScriptedEnemyController implements TankController {
	private selfId = '';
	private planner: NavigationPlanner | null = null;
	private rng = new SeededRandom(1);
	private path: { x: number; y: number }[] = [];
	private pathTicksRemaining = 0;
	private aimAngleChangeAmount = 0;
	private lastSeenEnemyPositions = new Map<string, { x: number; y: number }>();

	constructor(private options: ScriptedEnemyControllerOptions) {}

	public reset(initial: MatchInit): void {
		this.selfId = initial.selfId;
		const selfTank = initial.tanks.find((t) => t.id === initial.selfId);
		this.planner = new NavigationPlanner(initial.arena, initial.obstacles, selfTank?.size);
		const salt = [...this.selfId].reduce((sum, character) => sum + character.charCodeAt(0), 0);
		this.rng = new SeededRandom(mixSeed(initial.seed, salt));
		this.path = [];
		this.pathTicksRemaining = 0;
		this.aimAngleChangeAmount = 0;
		this.lastSeenEnemyPositions.clear();
	}

	public act(obs: TankObservation): TankAction {
		const target = this.selectTarget(obs.enemies, obs.self);
		const profile = this.options.heuristicProfile ?? 'default';
		const evasiveMove = profile === 'default' ? 'none' : this.getThreatEvasiveMove(obs, profile === 'advanced' ? 2 : 1);
		const fallbackAimAngle = this.getAimAngle(obs, target);
		const firingSolution = this.getFiringSolution(obs, target, fallbackAimAngle, profile);
		const aimAngle = firingSolution.aimAngle;
		const baseMove = this.getMoveIntent(obs, target);
		const move =
			evasiveMove !== 'none'
				? evasiveMove
				: profile === 'advanced'
					? this.getAdvancedMoveIntent(obs, target, baseMove)
					: baseMove;
		const plantBomb = this.shouldPlantBomb(obs, target, profile);

		if (target && profile !== 'default') {
			this.lastSeenEnemyPositions.set(target.id, { x: target.x, y: target.y });
		}

		return {
			move,
			aimAngle,
			fire: firingSolution.fire,
			plantBomb,
			aimTarget: target
				? deriveAimTarget(
						aimAngle,
						obs.self.x + obs.self.size / 2,
						obs.self.y + obs.self.size / 2,
						obs.arena,
						Math.max(obs.arena.width, obs.arena.height)
					)
				: undefined,
		};
	}

	private selectTarget(enemies: TankObservation['enemies'], self: TankStateView): TankStateView | null {
		const livingEnemies = enemies.filter((enemy) => !enemy.destroyed);
		if (livingEnemies.length === 0) {
			return null;
		}
		const profile = this.options.heuristicProfile ?? 'default';
		if (profile !== 'default') {
			livingEnemies.sort((left, right) => {
				const leftDistance = Math.hypot(left.x - self.x, left.y - self.y);
				const rightDistance = Math.hypot(right.x - self.x, right.y - self.y);
				const leftThreat = left.maxAmmo - left.activeAmmo + left.maxBombs - left.activeBombs;
				const rightThreat = right.maxAmmo - right.activeAmmo + right.maxBombs - right.activeBombs;
				const leftScore = leftDistance - leftThreat * 18 + left.health * 8;
				const rightScore = rightDistance - rightThreat * 18 + right.health * 8;
				if (leftScore !== rightScore) {
					return leftScore - rightScore;
				}
				return left.id.localeCompare(right.id);
			});
			return livingEnemies[0];
		}
		livingEnemies.sort((left, right) => {
			const leftDistance = (left.x - self.x) ** 2 + (left.y - self.y) ** 2;
			const rightDistance = (right.x - self.x) ** 2 + (right.y - self.y) ** 2;
			if (leftDistance !== rightDistance) {
				return leftDistance - rightDistance;
			}
			return left.id.localeCompare(right.id);
		});
		return livingEnemies[0];
	}

	private getAimAngle(obs: TankObservation, target: TankStateView | null): number {
		if (!this.options.randomAim) {
			const profile = this.options.heuristicProfile ?? 'default';
			if (profile === 'elite' && target) {
				return this.getEliteAimAngle(obs, target);
			}
			if (profile === 'advanced' && target) {
				return this.getAdvancedAimAngle(obs, target);
			}
			return target ? aimAngleAtTarget(obs.self, target) : obs.self.aimAngle;
		}
		if (this.aimAngleChangeAmount > 0) {
			this.aimAngleChangeAmount -= 1;
			return normalizeAngle(obs.self.aimAngle + 0.01);
		}
		if (this.aimAngleChangeAmount < 0) {
			this.aimAngleChangeAmount += 1;
			return normalizeAngle(obs.self.aimAngle - 0.01);
		}
		this.aimAngleChangeAmount = this.rng.nextInt(-360, 360);
		return obs.self.aimAngle;
	}

	private getEliteAimAngle(obs: TankObservation, target: TankStateView): number {
		const last = this.lastSeenEnemyPositions.get(target.id);
		const targetCenterX = target.x + target.size / 2;
		const targetCenterY = target.y + target.size / 2;
		const selfCenterX = obs.self.x + obs.self.size / 2;
		const selfCenterY = obs.self.y + obs.self.size / 2;
		if (!last) {
			return normalizeAngle(Math.atan2(targetCenterY - selfCenterY, targetCenterX - selfCenterX));
		}

		const velocityX = target.x - last.x;
		const velocityY = target.y - last.y;
		const projectileSpeedPerTick = getProjectileSpec(obs.self.ammoType).speed * TICK_SECONDS;
		const distance = Math.hypot(targetCenterX - selfCenterX, targetCenterY - selfCenterY);
		const travelTicks = projectileSpeedPerTick > 0 ? distance / projectileSpeedPerTick : 0;
		const predictedX = this.clampTankCoordinate(
			targetCenterX + velocityX * Math.min(travelTicks, 36),
			target.size,
			obs.arena.width
		);
		const predictedY = this.clampTankCoordinate(
			targetCenterY + velocityY * Math.min(travelTicks, 36),
			target.size,
			obs.arena.height
		);
		return normalizeAngle(Math.atan2(predictedY - selfCenterY, predictedX - selfCenterX));
	}

	private getAdvancedAimAngle(obs: TankObservation, target: TankStateView): number {
		const eliteAngle = this.getEliteAimAngle(obs, target);
		const candidates = this.getBounceCandidateAngles(obs.self, target, obs.arena.width, obs.arena.height);
		let bestAngle = eliteAngle;
		let bestHit = false;
		for (const angle of [eliteAngle, ...candidates]) {
			const projectile = this.createPredictionProjectile(obs.self, angle, obs);
			const willHit = predictProjectileWillHitTank(projectile, target, obs.arena, obs.obstacles);
			if (!bestHit && willHit) {
				bestAngle = angle;
				bestHit = true;
				continue;
			}
			if (
				bestHit &&
				willHit &&
				this.angleDistance(obs.self.aimAngle, angle) < this.angleDistance(obs.self.aimAngle, bestAngle)
			) {
				bestAngle = angle;
			}
		}
		return bestAngle;
	}

	private getMoveIntent(obs: TankObservation, target: TankStateView | null): MoveIntent {
		if (this.options.navigationMode === 'stationary' || !target || !this.planner) {
			return 'none';
		}
		this.pathTicksRemaining -= 1;
		if (this.path.length === 0 || this.pathTicksRemaining <= 0) {
			const { aggressionFactor, closeApproach } = this.getNavigationProfile(obs.self);
			this.path = this.planner.getPath(
				this.options.navigationMode,
				obs.self,
				target,
				aggressionFactor,
				closeApproach,
				obs.projectiles,
				obs.bombs,
				this.rng
			);
			this.pathTicksRemaining = this.options.recalculationInterval;
		}

		let desiredMove = this.planner.getMoveForNextStep(obs.self, this.path[0]);
		if (desiredMove === 'none' && this.path.length > 1) {
			this.path.shift();
			desiredMove = this.planner.getMoveForNextStep(obs.self, this.path[0]);
		}
		if (
			(this.options.navigationMode === 'astar' || this.options.navigationMode === 'astar-avoidance') &&
			obs.self.wasLastMoveBlocked &&
			obs.self.consecutiveDirectionMoves > 2
		) {
			this.pathTicksRemaining = 0;
			return this.getBlockedRecoveryMove(desiredMove);
		}
		return desiredMove;
	}

	private getNavigationProfile(self: TankStateView): { aggressionFactor: number; closeApproach: boolean } {
		if (!self.bombType || self.activeBombs >= self.maxBombs) {
			return { aggressionFactor: this.options.aggressionFactor, closeApproach: false };
		}
		return {
			aggressionFactor: 0,
			closeApproach: true,
		};
	}

	private getBlockedRecoveryMove(desiredMove: MoveIntent): MoveIntent {
		switch (desiredMove) {
			case 'e':
				return this.rng.nextFloat() < 0.5 ? 'se' : 'ne';
			case 'w':
				return this.rng.nextFloat() < 0.5 ? 'nw' : 'sw';
			case 's':
				return this.rng.nextFloat() < 0.5 ? 'se' : 'sw';
			case 'n':
				return this.rng.nextFloat() < 0.5 ? 'nw' : 'ne';
			case 'se':
				return this.rng.nextFloat() < 0.5 ? 's' : 'e';
			case 'ne':
				return this.rng.nextFloat() < 0.5 ? 'n' : 'e';
			case 'sw':
				return this.rng.nextFloat() < 0.5 ? 's' : 'w';
			case 'nw':
				return this.rng.nextFloat() < 0.5 ? 'n' : 'w';
			default:
				return this.rng.pick<MoveIntent>(['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw']);
		}
	}

	private getThreatEvasiveMove(obs: TankObservation, lookaheadSteps: number): MoveIntent {
		const immediateThreat = obs.projectiles.some(
			(projectile) =>
				projectile.team !== obs.self.team &&
				predictProjectileWillHitTank(projectile, obs.self, obs.arena, obs.obstacles)
		);
		if (!immediateThreat) {
			return 'none';
		}
		let bestMove: MoveIntent = 'none';
		let bestScore = Number.NEGATIVE_INFINITY;
		for (const move of ALL_MOVES) {
			let candidate = this.projectTankAfterMove(obs.self, move, obs.arena.width, obs.arena.height, obs.obstacles);
			let totalDanger = this.computeDangerScore(candidate, obs, 1);
			for (let step = 1; step < lookaheadSteps; step++) {
				candidate = this.projectTankAfterMove(candidate, 'none', obs.arena.width, obs.arena.height, obs.obstacles);
				totalDanger += this.computeDangerScore(candidate, obs, step + 1) * 0.7;
			}
			const moveBonus = move === 'none' ? 0 : 4;
			const score = -totalDanger + moveBonus;
			if (score > bestScore) {
				bestScore = score;
				bestMove = move;
			}
		}
		return bestMove;
	}

	private getAdvancedMoveIntent(
		obs: TankObservation,
		target: TankStateView | null,
		preferredMove: MoveIntent
	): MoveIntent {
		if (!target) {
			return preferredMove;
		}

		const firstMoves = this.getAdvancedFirstMoves(preferredMove);
		let bestMove = preferredMove;
		let bestScore = Number.NEGATIVE_INFINITY;

		for (const firstMove of firstMoves) {
			const firstState = this.projectTankAfterMove(
				obs.self,
				firstMove,
				obs.arena.width,
				obs.arena.height,
				obs.obstacles
			);
			const firstScore = this.scoreCombatState(firstState, obs, target, firstMove);
			let bestSecond = Number.NEGATIVE_INFINITY;
			for (const secondMove of ALL_MOVES) {
				const secondState = this.projectTankAfterMove(
					firstState,
					secondMove,
					obs.arena.width,
					obs.arena.height,
					obs.obstacles
				);
				bestSecond = Math.max(bestSecond, this.scoreCombatState(secondState, obs, target, secondMove));
			}
			const sequenceScore = firstScore * 0.65 + bestSecond * 0.35;
			if (sequenceScore > bestScore) {
				bestScore = sequenceScore;
				bestMove = firstMove;
			}
		}
		return bestMove;
	}

	private getAdvancedFirstMoves(preferredMove: MoveIntent): MoveIntent[] {
		const ordered: MoveIntent[] = [preferredMove, 'none', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];
		const deduped: MoveIntent[] = [];
		for (const move of ordered) {
			if (!deduped.includes(move)) {
				deduped.push(move);
			}
		}
		return deduped;
	}

	private scoreCombatState(
		state: TankStateView,
		obs: TankObservation,
		target: TankStateView,
		move: MoveIntent
	): number {
		const role = this.resolveRole();
		const distance = Math.hypot(target.x - state.x, target.y - state.y);
		const preferredDistance = role === 'pressure' ? 130 : role === 'flank' ? 190 : 250;
		const distancePenalty = Math.abs(distance - preferredDistance) * 0.08;
		const dangerPenalty = this.computeDangerScore(state, obs, 1) * 1.2;
		const movementPenalty = move === 'none' ? 3 : 0;
		const centerX = state.x + state.size / 2;
		const centerY = state.y + state.size / 2;
		const arenaCenterX = obs.arena.width / 2;
		const arenaCenterY = obs.arena.height / 2;
		const centerDistance = Math.hypot(centerX - arenaCenterX, centerY - arenaCenterY);
		const centerBias = role === 'zone' ? -centerDistance * 0.01 : 0;
		return 100 - distancePenalty - dangerPenalty - movementPenalty + centerBias;
	}

	private computeDangerScore(candidateTank: TankStateView, obs: TankObservation, lookaheadTick: number): number {
		let score = 0;
		const candidateCenterX = candidateTank.x + candidateTank.size / 2;
		const candidateCenterY = candidateTank.y + candidateTank.size / 2;
		for (const projectile of obs.projectiles) {
			if (projectile.team === obs.self.team) {
				continue;
			}
			const projected = {
				...projectile,
				x: projectile.x + projectile.vx * TICK_SECONDS * lookaheadTick,
				y: projectile.y + projectile.vy * TICK_SECONDS * lookaheadTick,
			};
			const px = Math.max(0, Math.min(projected.x, obs.arena.width));
			const py = Math.max(0, Math.min(projected.y, obs.arena.height));
			const distance = Math.hypot(candidateCenterX - px, candidateCenterY - py);
			if (predictProjectileWillHitTank(projected, candidateTank, obs.arena, obs.obstacles)) {
				score += 220;
			}
			score += Math.max(0, 120 - distance) * 0.7;
		}
		for (const bomb of obs.bombs) {
			if (bomb.team === obs.self.team) {
				continue;
			}
			const distance = Math.hypot(candidateCenterX - bomb.x, candidateCenterY - bomb.y);
			const fuseUrgency = Math.max(0, 1 - bomb.fuseTicksRemaining / 360);
			const overlapPenalty = Math.max(0, bomb.blastRadius + candidateTank.size / 2 - distance);
			score += overlapPenalty * (0.6 + fuseUrgency);
		}
		return score;
	}

	private projectTankAfterMove(
		tank: TankStateView,
		move: MoveIntent,
		arenaWidth: number,
		arenaHeight: number,
		obstacles: TankObservation['obstacles']
	): TankStateView {
		const delta = getMoveDelta(move, tank.speed, TICK_SECONDS);
		const nextX = Math.max(0, Math.min(tank.x + delta.dx, arenaWidth - tank.size));
		const nextY = Math.max(0, Math.min(tank.y + delta.dy, arenaHeight - tank.size));
		const blocked = obstacles.some((obstacle) => obstacleIntersectsTank(nextX, nextY, tank.size, obstacle));
		return {
			...tank,
			x: blocked ? tank.x : nextX,
			y: blocked ? tank.y : nextY,
		};
	}

	private getFiringSolution(
		obs: TankObservation,
		target: TankStateView | null,
		fallbackAimAngle: number,
		profile: HeuristicProfile
	): { aimAngle: number; fire: boolean } {
		if (!target || obs.self.destroyed) {
			return { aimAngle: fallbackAimAngle, fire: false };
		}
		if (obs.self.shotCooldownTicks > 0 || obs.self.activeAmmo >= obs.self.maxAmmo) {
			return { aimAngle: fallbackAimAngle, fire: false };
		}

		const candidateAngles =
			profile === 'default'
				? [fallbackAimAngle]
				: this.getAdvancedCandidateAngles(obs, target, fallbackAimAngle, profile);

		for (const angle of candidateAngles) {
			const projectile = this.createPredictionProjectile(obs.self, angle, obs);
			if (
				predictProjectileWillHitTank(projectile, target, obs.arena, obs.obstacles) &&
				(profile !== 'advanced' || !this.wouldCauseFriendlyFire(projectile, obs))
			) {
				return { aimAngle: angle, fire: true };
			}
		}
		return { aimAngle: fallbackAimAngle, fire: false };
	}

	private getAdvancedCandidateAngles(
		obs: TankObservation,
		target: TankStateView,
		baseAngle: number,
		profile: HeuristicProfile
	): number[] {
		const offsets = profile === 'advanced' ? [0, 0.1, -0.1, 0.25, -0.25, 0.4, -0.4] : [0, 0.2, -0.2, 0.4, -0.4];
		const set = new Set<number>();
		for (const offset of offsets) {
			set.add(Number(normalizeAngle(baseAngle + offset).toFixed(5)));
		}
		if (profile === 'advanced') {
			for (const bounceAngle of this.getBounceCandidateAngles(obs.self, target, obs.arena.width, obs.arena.height)) {
				set.add(Number(bounceAngle.toFixed(5)));
			}
		}
		return Array.from(set.values());
	}

	private getBounceCandidateAngles(
		self: TankStateView,
		target: TankStateView,
		arenaWidth: number,
		arenaHeight: number
	): number[] {
		const selfCenterX = self.x + self.size / 2;
		const selfCenterY = self.y + self.size / 2;
		const targetCenterX = target.x + target.size / 2;
		const targetCenterY = target.y + target.size / 2;
		const mirrors = [
			{ x: -targetCenterX, y: targetCenterY },
			{ x: arenaWidth * 2 - targetCenterX, y: targetCenterY },
			{ x: targetCenterX, y: -targetCenterY },
			{ x: targetCenterX, y: arenaHeight * 2 - targetCenterY },
		];
		return mirrors.map((mirror) => normalizeAngle(Math.atan2(mirror.y - selfCenterY, mirror.x - selfCenterX)));
	}

	private createPredictionProjectile(
		self: TankStateView,
		aimAngle: number,
		obs: Pick<TankObservation, 'arena' | 'obstacles'>
	): ProjectileStateView {
		const projectileSpec = getProjectileSpec(self.ammoType);
		const barrelEnd = computeClearGunBarrelEnd(
			{
				x: self.x,
				y: self.y,
				size: self.size,
				aimAngle,
			},
			obs.obstacles,
			obs.arena
		);
		return {
			id: 'prediction',
			ownerTankId: self.id,
			team: self.team,
			kind: self.ammoType,
			x: barrelEnd.x,
			y: barrelEnd.y,
			vx: Math.cos(aimAngle) * projectileSpec.speed,
			vy: Math.sin(aimAngle) * projectileSpec.speed,
			speed: projectileSpec.speed,
			radius: projectileSpec.radius,
			bounces: 0,
			maxBounces: projectileSpec.maxBounces,
		};
	}

	private shouldPlantBomb(obs: TankObservation, target: TankStateView | null, profile: HeuristicProfile): boolean {
		if (!target || !obs.self.bombType || obs.self.destroyed) {
			return false;
		}
		if (obs.self.bombCooldownTicks > 0 || obs.self.activeBombs >= obs.self.maxBombs) {
			return false;
		}
		const bombSpec = getBombSpec(obs.self.bombType);
		const centerX = obs.self.x + obs.self.size / 2;
		const centerY = obs.self.y + obs.self.size / 2;
		if (!bombWouldHitTank(centerX, centerY, bombSpec.blastRadius, target)) {
			if (profile !== 'advanced') {
				return false;
			}
			if (!this.isLikelyBombTrap(centerX, centerY, bombSpec.blastRadius, bombSpec.fuseTicks, target, obs)) {
				return false;
			}
		}
		if (profile === 'default') {
			return true;
		}
		if (profile === 'advanced' && this.wouldBombAlly(centerX, centerY, bombSpec.blastRadius, obs)) {
			return false;
		}
		const selfWouldBeHit = bombWouldHitTank(centerX, centerY, bombSpec.blastRadius * 0.85, obs.self);
		if (profile === 'advanced') {
			return !selfWouldBeHit;
		}
		return !selfWouldBeHit || obs.self.health > 1;
	}

	private wouldCauseFriendlyFire(projectile: ProjectileStateView, obs: TankObservation): boolean {
		if (predictProjectileWillHitTank(projectile, obs.self, obs.arena, obs.obstacles)) {
			return true;
		}
		const allies = obs.allies ?? [];
		for (const ally of allies) {
			if (!ally.destroyed && predictProjectileWillHitTank(projectile, ally, obs.arena, obs.obstacles)) {
				return true;
			}
		}
		return false;
	}

	private wouldBombAlly(centerX: number, centerY: number, blastRadius: number, obs: TankObservation): boolean {
		const allies = obs.allies ?? [];
		for (const ally of allies) {
			if (!ally.destroyed && bombWouldHitTank(centerX, centerY, blastRadius, ally)) {
				return true;
			}
		}
		return false;
	}

	private isLikelyBombTrap(
		bombX: number,
		bombY: number,
		blastRadius: number,
		fuseTicks: number,
		target: TankStateView,
		obs: TankObservation
	): boolean {
		const predicted = this.predictTargetPosition(target, Math.min(fuseTicks, 120));
		const predictedTank: TankStateView = {
			...target,
			x: predicted.x,
			y: predicted.y,
		};
		if (!bombWouldHitTank(bombX, bombY, blastRadius, predictedTank)) {
			return false;
		}
		const escapeMove = this.getThreatEvasiveMove(obs, 2);
		return escapeMove !== 'none' || obs.self.health > 1;
	}

	private predictTargetPosition(target: TankStateView, ticks: number): { x: number; y: number } {
		const last = this.lastSeenEnemyPositions.get(target.id);
		if (!last) {
			return { x: target.x, y: target.y };
		}
		const vx = target.x - last.x;
		const vy = target.y - last.y;
		const predictedX = target.x + vx * Math.min(ticks, 36);
		const predictedY = target.y + vy * Math.min(ticks, 36);
		return { x: predictedX, y: predictedY };
	}

	private resolveRole(): Exclude<TacticalRole, 'auto'> {
		if (this.options.tacticalRole && this.options.tacticalRole !== 'auto') {
			return this.options.tacticalRole;
		}
		const bucket = [...this.selfId].reduce((acc, char) => acc + char.charCodeAt(0), 0) % 3;
		if (bucket === 0) {
			return 'pressure';
		}
		if (bucket === 1) {
			return 'flank';
		}
		return 'zone';
	}

	private angleDistance(a: number, b: number): number {
		const delta = Math.abs(a - b) % (Math.PI * 2);
		return Math.min(delta, Math.PI * 2 - delta);
	}

	private clampTankCoordinate(value: number, tankSize: number, arenaBound: number): number {
		const half = tankSize / 2;
		return Math.max(half, Math.min(value, arenaBound - half));
	}
}
