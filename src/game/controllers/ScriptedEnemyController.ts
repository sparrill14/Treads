import { aimAngleAtTarget, computeGunBarrelEnd, deriveAimTarget, normalizeAngle } from '../core/geometry';
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

interface ScriptedEnemyControllerOptions {
	navigationMode: NavigationMode | 'stationary';
	randomAim: boolean;
	recalculationInterval: number;
	aggressionFactor: number;
}

export class ScriptedEnemyController implements TankController {
	private selfId = '';
	private planner: NavigationPlanner | null = null;
	private rng = new SeededRandom(1);
	private path: { x: number; y: number }[] = [];
	private pathTicksRemaining = 0;
	private aimAngleChangeAmount = 0;

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
	}

	public act(obs: TankObservation): TankAction {
		const target = this.selectTarget(obs.enemies, obs.self);
		const aimAngle = this.getAimAngle(obs, target);
		return {
			move: this.getMoveIntent(obs, target),
			aimAngle,
			fire: this.shouldFire(obs, target, aimAngle),
			plantBomb: this.shouldPlantBomb(obs, target),
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

	private shouldFire(obs: TankObservation, target: TankStateView | null, aimAngle: number): boolean {
		if (!target || obs.self.destroyed) {
			return false;
		}
		if (obs.self.shotCooldownTicks > 0 || obs.self.activeAmmo >= obs.self.maxAmmo) {
			return false;
		}
		const projectileSpec = getProjectileSpec(obs.self.ammoType);
		const barrelEnd = computeGunBarrelEnd({
			x: obs.self.x,
			y: obs.self.y,
			size: obs.self.size,
			aimAngle,
		});
		const projectile: ProjectileStateView = {
			id: 'prediction',
			ownerTankId: obs.self.id,
			team: obs.self.team,
			kind: obs.self.ammoType,
			x: barrelEnd.x,
			y: barrelEnd.y,
			vx: Math.cos(aimAngle) * projectileSpec.speed,
			vy: Math.sin(aimAngle) * projectileSpec.speed,
			speed: projectileSpec.speed,
			radius: projectileSpec.radius,
			bounces: 0,
			maxBounces: projectileSpec.maxBounces,
		};
		return predictProjectileWillHitTank(projectile, target, obs.arena, obs.obstacles);
	}

	private shouldPlantBomb(obs: TankObservation, target: TankStateView | null): boolean {
		if (!target || !obs.self.bombType || obs.self.destroyed) {
			return false;
		}
		if (obs.self.bombCooldownTicks > 0 || obs.self.activeBombs >= obs.self.maxBombs) {
			return false;
		}
		const bombSpec = getBombSpec(obs.self.bombType);
		const centerX = obs.self.x + obs.self.size / 2;
		const centerY = obs.self.y + obs.self.size / 2;
		return bombWouldHitTank(centerX, centerY, bombSpec.blastRadius, target);
	}
}
