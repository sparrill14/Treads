import * as ort from 'onnxruntime-web';
import type { MatchInit, MoveIntent, TankAction, TankController, TankObservation } from '../core/types';

const ARENA_WIDTH = 1000;
const ARENA_HEIGHT = 500;
const MAX_ENEMIES = 6;
const MAX_PROJECTILES = 15;
const MAX_OBSTACLES = 5;
const MAX_BOMBS = 6;
const SELF_DIM = 15; // pos(2), aim, speed, LOS, blocked, invuln, tick, hp, ammoRatio, shotCD, bombRatio, aimFeatures(3)
const ENEMY_DIM = 9; // [relX, relY, aimAngle, speed, hasBomb, health, aimed_at_me, ammoThreat, isApproaching]
const PROJ_DIM = 5;
const OBS_DIM = 4;
const BOMB_DIM = 5;
const SUMMARY_DIM = 6;
const MAX_FUSE_TICKS = 360;
const MAX_BLAST_RADIUS = 100;
const PROJECTILE_SPEED_NORM = 300; // Normalizer for projectile velocity (max super=270)
const OBS_SIZE =
	SELF_DIM +
	MAX_ENEMIES * ENEMY_DIM +
	MAX_PROJECTILES * PROJ_DIM +
	MAX_OBSTACLES * OBS_DIM +
	MAX_BOMBS * BOMB_DIM +
	SUMMARY_DIM;

// Continuous action means: [move_x, move_y, aim_signal, fire_signal, bomb_signal] in [-1, 1]
const ACTION_DIM = 5;
const FIRE_THRESHOLD = 0.0;
const BOMB_THRESHOLD = 0.5;
const AIM_OFFSET_LIMIT = Math.PI / 18; // ±10° — must match training (rollout-worker.ts)
const ARENA_DIAGONAL = Math.sqrt(ARENA_WIDTH * ARENA_WIDTH + ARENA_HEIGHT * ARENA_HEIGHT);
const MOVE_DEAD_ZONE = 0.33;

// ---- MoveIntent → unit direction vector mapping ----
const SQRT2_2 = Math.SQRT2 / 2;
const MOVE_DIR_MAP: Record<MoveIntent, [number, number]> = {
	none: [0, 0],
	n: [0, -1],
	s: [0, 1],
	e: [1, 0],
	w: [-1, 0],
	ne: [SQRT2_2, -SQRT2_2],
	nw: [-SQRT2_2, -SQRT2_2],
	se: [SQRT2_2, SQRT2_2],
	sw: [-SQRT2_2, SQRT2_2],
};
function moveIntentToDir(intent: MoveIntent): [number, number] {
	return MOVE_DIR_MAP[intent] ?? [0, 0];
}

// ---- Line-of-sight utility (segment-AABB intersection) ----
function segmentIntersectsRect(
	x1: number,
	y1: number,
	x2: number,
	y2: number,
	rx: number,
	ry: number,
	rw: number,
	rh: number
): boolean {
	const dx = x2 - x1;
	const dy = y2 - y1;
	let tMin = 0;
	let tMax = 1;
	if (Math.abs(dx) < 1e-10) {
		if (x1 < rx || x1 > rx + rw) return false;
	} else {
		let t1 = (rx - x1) / dx;
		let t2 = (rx + rw - x1) / dx;
		if (t1 > t2) {
			const tmp = t1;
			t1 = t2;
			t2 = tmp;
		}
		tMin = Math.max(tMin, t1);
		tMax = Math.min(tMax, t2);
		if (tMin > tMax) return false;
	}
	if (Math.abs(dy) < 1e-10) {
		if (y1 < ry || y1 > ry + rh) return false;
	} else {
		let t1 = (ry - y1) / dy;
		let t2 = (ry + rh - y1) / dy;
		if (t1 > t2) {
			const tmp = t1;
			t1 = t2;
			t2 = tmp;
		}
		tMin = Math.max(tMin, t1);
		tMax = Math.min(tMax, t2);
		if (tMin > tMax) return false;
	}
	return true;
}

function nnHasLineOfSight(
	sx: number,
	sy: number,
	tx: number,
	ty: number,
	obstacles: readonly { x: number; y: number; width: number; height: number }[]
): boolean {
	for (const o of obstacles) {
		if (segmentIntersectsRect(sx, sy, tx, ty, o.x, o.y, o.width, o.height)) {
			return false;
		}
	}
	return true;
}

export class NeuralNetController implements TankController {
	private session: ort.InferenceSession | null = null;
	private ready = false;
	private inferenceInFlight = false;
	private queuedRequest: { input: Float32Array; obs: TankObservation; generation: number } | null = null;
	private pendingAction: TankAction | null = null;
	private generation = 0;

	constructor(private modelUrl: string) {}

	/**
	 * Load the ONNX model. Must be called before act().
	 * Can be called from any async context.
	 */
	public async loadModel(): Promise<void> {
		this.session = await ort.InferenceSession.create(this.modelUrl, {
			executionProviders: ['wasm'],
		});
		this.ready = true;
	}

	public reset(_initial: MatchInit): void {
		this.generation += 1;
		this.pendingAction = null;
		this.queuedRequest = null;
	}

	public act(obs: TankObservation): TankAction {
		if (!this.ready || !this.session) {
			// Fallback: no-op until model is loaded
			return { move: 'none', aimAngle: obs.self.aimAngle, fire: false, plantBomb: false };
		}

		// Normalize observation synchronously
		const input = this.normalizeObservation(obs);

		// Run inference synchronously using our cached last result
		// Since ONNX inference is async, we use a fire-and-forget pattern
		// with a cached result from the previous tick
		if (this.pendingAction) {
			const action = this.pendingAction;
			this.pendingAction = null;
			void this.runInferenceAsync(input, obs, this.generation);
			return action;
		}

		// First tick: return a default action and start inference
		void this.runInferenceAsync(input, obs, this.generation);
		return { move: 'none', aimAngle: obs.self.aimAngle, fire: false, plantBomb: false };
	}

	private async runInferenceAsync(input: Float32Array, obs: TankObservation, generation: number): Promise<void> {
		if (!this.session) return;
		if (this.inferenceInFlight) {
			// Keep only the most recent input to avoid unbounded queue growth.
			this.queuedRequest = { input, obs, generation };
			return;
		}

		this.inferenceInFlight = true;
		try {
			const tensor = new ort.Tensor('float32', input, [1, OBS_SIZE]);
			const feeds = { observation: tensor };
			const results = await this.session.run(feeds);
			const output = results['action_mean'] ?? results['logits'];
			const data = output.data as Float32Array;
			if (generation === this.generation) {
				this.pendingAction = this.decodeAction(Array.from(data), obs);
			}
		} catch (err) {
			console.error('ONNX inference error:', err);
		} finally {
			this.inferenceInFlight = false;
			if (this.queuedRequest) {
				const nextRequest = this.queuedRequest;
				this.queuedRequest = null;
				void this.runInferenceAsync(nextRequest.input, nextRequest.obs, nextRequest.generation);
			}
		}
	}

	private decodeAction(logits: number[], obs: TankObservation): TankAction {
		if (logits.length < ACTION_DIM) {
			return {
				move: 'none',
				aimAngle: obs.self.aimAngle,
				fire: false,
				plantBomb: false,
			};
		}

		const mx = Math.max(-1, Math.min(1, logits[0]));
		const my = Math.max(-1, Math.min(1, logits[1]));
		const aimSignal = Math.max(-1, Math.min(1, logits[2]));
		const fireSignal = Math.max(-1, Math.min(1, logits[3]));
		const bombSignal = Math.max(-1, Math.min(1, logits[4]));

		// 2D movement decode: (move_x, move_y) → MoveIntent
		const goE = mx > MOVE_DEAD_ZONE;
		const goW = mx < -MOVE_DEAD_ZONE;
		const goS = my > MOVE_DEAD_ZONE;
		const goN = my < -MOVE_DEAD_ZONE;
		let moveIntent: MoveIntent;
		if (goN && goE) moveIntent = 'ne';
		else if (goN && goW) moveIntent = 'nw';
		else if (goS && goE) moveIntent = 'se';
		else if (goS && goW) moveIntent = 'sw';
		else if (goN) moveIntent = 'n';
		else if (goS) moveIntent = 's';
		else if (goE) moveIntent = 'e';
		else if (goW) moveIntent = 'w';
		else moveIntent = 'none';

		// Enemy-relative aim encoding: aim_signal=0 points at nearest enemy.
		const s = obs.self;
		const sx = s.x + s.size / 2;
		const sy = s.y + s.size / 2;
		const aliveEnemies = obs.enemies.filter((e) => !e.destroyed);
		let aimAngle: number;
		if (aliveEnemies.length > 0) {
			const nearest = aliveEnemies.reduce((best, e) => {
				const dx = e.x + e.size / 2 - sx;
				const dy = e.y + e.size / 2 - sy;
				const bdx = best.x + best.size / 2 - sx;
				const bdy = best.y + best.size / 2 - sy;
				return dx * dx + dy * dy < bdx * bdx + bdy * bdy ? e : best;
			});
			const angleToEnemy = Math.atan2(nearest.y + nearest.size / 2 - sy, nearest.x + nearest.size / 2 - sx);
			aimAngle = angleToEnemy + aimSignal * AIM_OFFSET_LIMIT;
		} else {
			aimAngle = s.aimAngle;
		}

		return {
			move: moveIntent,
			aimAngle,
			fire: fireSignal > FIRE_THRESHOLD,
			plantBomb: bombSignal > BOMB_THRESHOLD,
		};
	}

	private normalizeObservation(obs: TankObservation): Float32Array {
		const result = new Float32Array(OBS_SIZE);
		let idx = 0;

		const sx = obs.self.x + obs.self.size / 2;
		const sy = obs.self.y + obs.self.size / 2;
		const arenaDiag = ARENA_DIAGONAL;
		const livingEnemies = obs.enemies
			.filter((e) => !e.destroyed)
			.sort((a, b) => {
				const da = (a.x + a.size / 2 - sx) ** 2 + (a.y + a.size / 2 - sy) ** 2;
				const db = (b.x + b.size / 2 - sx) ** 2 + (b.y + b.size / 2 - sy) ** 2;
				return da - db;
			});

		// ── Self features (SELF_DIM = 12) ──
		result[idx] = obs.self.x / ARENA_WIDTH;
		result[idx + 1] = obs.self.y / ARENA_HEIGHT;
		result[idx + 2] = obs.self.aimAngle / (2 * Math.PI) + 0.5; // Fix 1: map [-π,π] → [0,1]
		result[idx + 3] = obs.self.speed / 100;
		let hasLOS = 0.0;
		if (livingEnemies.length > 0) {
			const nearest = livingEnemies[0];
			hasLOS = nnHasLineOfSight(sx, sy, nearest.x + nearest.size / 2, nearest.y + nearest.size / 2, obs.obstacles)
				? 1.0
				: 0.0;
		}
		result[idx + 4] = hasLOS;
		result[idx + 5] = obs.self.wasLastMoveBlocked ? 1 : 0;
		result[idx + 6] = Math.min(obs.self.invulnerabilityTicksRemaining / 8, 1);
		result[idx + 7] = Math.min(obs.tick / 1080, 1);
		result[idx + 8] = obs.self.health / Math.max(obs.self.maxHealth, 1);

		// Resource features — ammo, cooldown, bombs
		result[idx + 9] = obs.self.activeAmmo / Math.max(obs.self.maxAmmo, 1); // ammo ratio (1=full, 0=empty)
		result[idx + 10] = obs.self.shotCooldownTicks / Math.max(obs.self.shotCooldownTicksOnFire, 1); // shot cooldown (0=ready, 1=just fired)
		result[idx + 11] = obs.self.activeBombs / Math.max(obs.self.maxBombs, 1); // bomb ratio (0=none/empty)

		// Derived aim features (relative to nearest enemy)
		if (livingEnemies.length > 0) {
			const nearest = livingEnemies[0];
			const ex = nearest.x + nearest.size / 2;
			const ey = nearest.y + nearest.size / 2;
			const angleToEnemy = Math.atan2(ey - sy, ex - sx);
			const distToEnemy = Math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2);
			const aimAngle = obs.self.aimAngle;
			const aimError = Math.atan2(Math.sin(aimAngle - angleToEnemy), Math.cos(aimAngle - angleToEnemy));
			result[idx + 12] = angleToEnemy / (2 * Math.PI) + 0.5;
			result[idx + 13] = Math.min(distToEnemy / arenaDiag, 1.0);
			result[idx + 14] = (aimError / Math.PI) * 0.5 + 0.5;
		} else {
			result[idx + 12] = 0.5;
			result[idx + 13] = 0.0;
			result[idx + 14] = 0.5;
		}
		idx += SELF_DIM;

		// ── Enemies (ENEMY_DIM = 9, player-relative positions) ──
		for (let i = 0; i < MAX_ENEMIES; i++) {
			if (i < livingEnemies.length) {
				const e = livingEnemies[i];
				const ecx = e.x + e.size / 2;
				const ecy = e.y + e.size / 2;
				result[idx] = ((ecx - sx) / arenaDiag) * 0.5 + 0.5; // player-relative dx
				result[idx + 1] = ((ecy - sy) / arenaDiag) * 0.5 + 0.5; // player-relative dy
				result[idx + 2] = e.aimAngle / (2 * Math.PI) + 0.5; // angle normalization
				result[idx + 3] = e.speed / 100;
				result[idx + 4] = e.bombType ? 1 : 0;
				result[idx + 5] = e.health / Math.max(e.maxHealth, 1);
				// aimed_at_me — how much enemy turret points toward player (1=direct, 0=away)
				const angleFromEnemyToPlayer = Math.atan2(sy - ecy, sx - ecx);
				const enemyAimError = Math.atan2(
					Math.sin(e.aimAngle - angleFromEnemyToPlayer),
					Math.cos(e.aimAngle - angleFromEnemyToPlayer)
				);
				result[idx + 6] = 1.0 - Math.abs(enemyAimError) / Math.PI;
				// ammoThreat — 0=none, 0.5=basic, 1.0=super
				result[idx + 7] = e.maxAmmo > 0 ? (e.ammoType === 'super' ? 1.0 : 0.5) : 0.0;
				// isApproaching — dot(moveDir, enemyToPlayer) mapped to [0,1]
				const moveDir = moveIntentToDir(e.lastMoveIntent);
				if (moveDir[0] !== 0 || moveDir[1] !== 0) {
					const toPlayerDx = sx - ecx;
					const toPlayerDy = sy - ecy;
					const toPlayerDist = Math.sqrt(toPlayerDx * toPlayerDx + toPlayerDy * toPlayerDy);
					if (toPlayerDist > 1e-6) {
						const dot = (moveDir[0] * toPlayerDx + moveDir[1] * toPlayerDy) / toPlayerDist;
						result[idx + 8] = dot * 0.5 + 0.5; // [-1,1] → [0,1]
					} else {
						result[idx + 8] = 0.5;
					}
				} else {
					result[idx + 8] = 0.5; // stationary → neutral
				}
			}
			idx += ENEMY_DIM;
		}

		// ── Projectiles (PROJ_DIM = 5, player-relative positions) ──
		const projectiles = [...obs.projectiles].sort((a, b) => {
			const da = (a.x - sx) ** 2 + (a.y - sy) ** 2;
			const db = (b.x - sx) ** 2 + (b.y - sy) ** 2;
			return da - db;
		});
		for (let i = 0; i < MAX_PROJECTILES; i++) {
			if (i < projectiles.length) {
				const p = projectiles[i];
				result[idx] = ((p.x - sx) / arenaDiag) * 0.5 + 0.5; // Fix 3: player-relative
				result[idx + 1] = ((p.y - sy) / arenaDiag) * 0.5 + 0.5; // Fix 3: player-relative
				result[idx + 2] = (p.vx / PROJECTILE_SPEED_NORM) * 0.5 + 0.5;
				result[idx + 3] = (p.vy / PROJECTILE_SPEED_NORM) * 0.5 + 0.5;
				result[idx + 4] = p.team === 'enemy' ? 1 : 0;
			}
			idx += PROJ_DIM;
		}

		// ── Obstacles (OBS_DIM = 4, player-relative center positions) ──
		const obstacles = [...obs.obstacles].sort((a, b) => {
			const da = (a.x + a.width / 2 - sx) ** 2 + (a.y + a.height / 2 - sy) ** 2;
			const db = (b.x + b.width / 2 - sx) ** 2 + (b.y + b.height / 2 - sy) ** 2;
			return da - db;
		});
		for (let i = 0; i < MAX_OBSTACLES; i++) {
			if (i < obstacles.length) {
				const o = obstacles[i];
				result[idx] = ((o.x + o.width / 2 - sx) / arenaDiag) * 0.5 + 0.5; // Fix 3: player-relative center
				result[idx + 1] = ((o.y + o.height / 2 - sy) / arenaDiag) * 0.5 + 0.5;
				result[idx + 2] = o.width / ARENA_WIDTH;
				result[idx + 3] = o.height / ARENA_HEIGHT;
			}
			idx += OBS_DIM;
		}

		// ── Bombs (BOMB_DIM = 5, player-relative positions) ──
		const bombs = [...obs.bombs].sort((a, b) => {
			const da = (a.x - sx) ** 2 + (a.y - sy) ** 2;
			const db = (b.x - sx) ** 2 + (b.y - sy) ** 2;
			return da - db;
		});
		for (let i = 0; i < MAX_BOMBS; i++) {
			if (i < bombs.length) {
				const b = bombs[i];
				result[idx] = ((b.x - sx) / arenaDiag) * 0.5 + 0.5; // Fix 3: player-relative
				result[idx + 1] = ((b.y - sy) / arenaDiag) * 0.5 + 0.5; // Fix 3: player-relative
				result[idx + 2] = Math.min(b.fuseTicksRemaining / MAX_FUSE_TICKS, 1.0);
				result[idx + 3] = Math.min(b.blastRadius / MAX_BLAST_RADIUS, 1.0);
				result[idx + 4] = b.team === 'enemy' ? 1 : 0;
			}
			idx += BOMB_DIM;
		}

		// ── Summary features (SUMMARY_DIM = 6) ──
		result[idx] = Math.min(livingEnemies.length / MAX_ENEMIES, 1.0);
		if (livingEnemies.length > 0) {
			let farthestEnemyDistSq = 0;
			for (const e of livingEnemies) {
				const dx = e.x + e.size / 2 - sx;
				const dy = e.y + e.size / 2 - sy;
				const dSq = dx * dx + dy * dy;
				if (dSq > farthestEnemyDistSq) farthestEnemyDistSq = dSq;
			}
			result[idx + 1] = Math.min(Math.sqrt(farthestEnemyDistSq) / arenaDiag, 1.0);
		} else {
			result[idx + 1] = 0;
		}
		result[idx + 2] = Math.min(obs.projectiles.length / MAX_PROJECTILES, 1.0);
		result[idx + 3] = Math.min(obs.bombs.length / MAX_BOMBS, 1.0);
		const enemyBombs = obs.bombs.filter((b) => b.team === 'enemy');
		if (enemyBombs.length > 0) {
			let closestBombDistSq = Infinity;
			for (const b of enemyBombs) {
				const dx = b.x - sx;
				const dy = b.y - sy;
				const dSq = dx * dx + dy * dy;
				if (dSq < closestBombDistSq) closestBombDistSq = dSq;
			}
			result[idx + 4] = Math.min(Math.sqrt(closestBombDistSq) / arenaDiag, 1.0);
		} else {
			result[idx + 4] = 1.0;
		}
		if (obs.projectiles.length > 0) {
			let farthestProjDistSq = 0;
			for (const p of obs.projectiles) {
				const dx = p.x - sx;
				const dy = p.y - sy;
				const dSq = dx * dx + dy * dy;
				if (dSq > farthestProjDistSq) farthestProjDistSq = dSq;
			}
			result[idx + 5] = Math.min(Math.sqrt(farthestProjDistSq) / arenaDiag, 1.0);
		} else {
			result[idx + 5] = 0;
		}

		// Clamp to [0, 1]
		for (let i = 0; i < OBS_SIZE; i++) {
			result[i] = Math.max(0, Math.min(1, result[i]));
		}

		return result;
	}
}
