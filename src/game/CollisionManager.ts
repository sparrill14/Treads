import { Ammunition } from './Ammunition';
import { AudioFile, AudioManager } from './AudioManager';
import { Bomb } from './Bomb';
import { ObstacleCanvas } from './ObstacleCanvas';
import { Tank } from './tanks/Tank';

function pointInAABB(px: number, py: number, left: number, right: number, top: number, bottom: number): boolean {
	return px > left && px < right && py > top && py < bottom;
}

function circlesOverlap(x1: number, y1: number, r1: number, x2: number, y2: number, r2: number): boolean {
	const dx = x1 - x2;
	const dy = y1 - y2;
	return dx * dx + dy * dy < (r1 + r2) * (r1 + r2);
}

export class CollisionManager {
	constructor(private audioManager: AudioManager) {}

	public update(playerTank: Tank, enemyTanks: Tank[], obstacleCanvas: ObstacleCanvas, deltaTime: number): void {
		const allAmmunition: Ammunition[] = [...enemyTanks.flatMap((t) => t.ammunition), ...playerTank.ammunition];
		const allBombs: Bomb[] = [...enemyTanks.flatMap((t) => t.bombs), ...playerTank.bombs];

		// Enemy ammunition
		for (const enemyTank of enemyTanks) {
			for (const ammo of enemyTank.ammunition) {
				if (ammo.isDestroyed) continue;
				this.checkAmmoVsAmmo(ammo, allAmmunition);
				this.checkAmmoVsBombs(ammo, playerTank.bombs);
				ammo.updatePosition(obstacleCanvas, deltaTime);
				this.checkAmmoVsTank(ammo, playerTank);
			}
			for (const bomb of enemyTank.bombs) {
				if (bomb.isDestroyed && !bomb.isExploding()) continue;
				this.checkBombVsTank(bomb, playerTank);
			}
		}

		// Player ammunition
		for (const ammo of playerTank.ammunition) {
			if (ammo.isDestroyed) continue;
			this.checkAmmoVsAmmo(ammo, allAmmunition);
			this.checkAmmoVsBombs(ammo, allBombs);
			ammo.updatePosition(obstacleCanvas, deltaTime);
			this.checkAmmoVsEnemies(ammo, enemyTanks);
			this.checkAmmoVsTank(ammo, playerTank);
		}

		// Player bombs
		for (const bomb of playerTank.bombs) {
			if (bomb.isDestroyed && !bomb.isExploding()) continue;
			this.checkBombVsEnemies(bomb, enemyTanks);
			this.checkBombVsTank(bomb, playerTank);
		}
	}

	private checkAmmoVsTank(ammo: Ammunition, tank: Tank): void {
		if (tank.isDestroyed || ammo.isDestroyed) return;
		if (pointInAABB(ammo.xPosition, ammo.yPosition, tank.xLeft, tank.xRight, tank.yTop, tank.yBottom)) {
			tank.destroy();
			ammo.destroy();
		}
	}

	private checkAmmoVsEnemies(ammo: Ammunition, enemyTanks: Tank[]): void {
		for (const enemy of enemyTanks) {
			if (enemy.isDestroyed) continue;
			if (pointInAABB(ammo.xPosition, ammo.yPosition, enemy.xLeft, enemy.xRight, enemy.yTop, enemy.yBottom)) {
				ammo.destroy();
				enemy.destroy();
				this.audioManager.play(AudioFile.TANK_DESTROY);
				return;
			}
		}
	}

	private checkAmmoVsAmmo(ammo: Ammunition, allAmmunition: Ammunition[]): void {
		for (const other of allAmmunition) {
			if (other === ammo || other.isDestroyed) continue;
			if (circlesOverlap(ammo.xPosition, ammo.yPosition, ammo.radius, other.xPosition, other.yPosition, other.radius)) {
				ammo.destroy();
				other.destroy();
				return;
			}
		}
	}

	private checkAmmoVsBombs(ammo: Ammunition, bombs: Bomb[]): void {
		for (const bomb of bombs) {
			if (bomb.isDestroyed || bomb.isExploding()) continue;
			if (circlesOverlap(ammo.xPosition, ammo.yPosition, ammo.radius, bomb.xPosition, bomb.yPosition, bomb.radius)) {
				ammo.destroy();
				bomb.destroy();
				return;
			}
		}
	}

	private checkBombVsEnemies(bomb: Bomb, enemyTanks: Tank[]): void {
		for (const enemy of enemyTanks) {
			if (enemy.isDestroyed) continue;
			if (this.checkFragmentsVsTank(bomb, enemy)) {
				enemy.destroy();
			}
		}
	}

	private checkBombVsTank(bomb: Bomb, tank: Tank): void {
		if (tank.isDestroyed) return;
		if (this.checkFragmentsVsTank(bomb, tank)) {
			tank.destroy();
		}
	}

	private checkFragmentsVsTank(bomb: Bomb, tank: Tank): boolean {
		for (const fragment of bomb.fragments) {
			if (
				fragment.life > 0 &&
				fragment.x >= tank.xLeft &&
				fragment.x <= tank.xRight &&
				fragment.y >= tank.yTop &&
				fragment.y <= tank.yBottom
			) {
				return true;
			}
		}
		return false;
	}
}
