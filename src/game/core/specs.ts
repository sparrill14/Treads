import { PastelColorPalette } from '../../ui/PastelColorPalette';
import type { AmmoType, BombType, EnemyType } from '../LevelConfig';
import type { TankKind } from './types';

export interface ProjectileSpec {
	speed: number;
	maxBounces: number;
	radius: number;
}

export interface BombSpec {
	radius: number;
	blastRadius: number;
	fuseTicks: number;
}

export interface TankSpec {
	speed: number;
	size: number;
	color: string;
	shotCooldownTicksOnFire: number;
	bombCooldownTicksOnPlant: number;
	initialShotCooldownTicks: number;
	initialBombCooldownTicks: number;
	aggressionFactor: number;
}

export function getProjectileSpec(ammoType: AmmoType): ProjectileSpec {
	if (ammoType === 'super') {
		return { speed: 270, maxBounces: 2, radius: 4 };
	}
	return { speed: 180, maxBounces: 1, radius: 4 };
}

export function getBombSpec(bombType: BombType): BombSpec {
	if (bombType === 'love') {
		return { radius: 15, blastRadius: 80, fuseTicks: 360 };
	}
	return { radius: 15, blastRadius: 50, fuseTicks: 360 };
}

export function getTankSpec(kind: TankKind): TankSpec {
	switch (kind) {
		case 'player':
			return {
				speed: 90,
				size: 30,
				color: '#4f6d7a',
				shotCooldownTicksOnFire: 0,
				bombCooldownTicksOnPlant: 0,
				initialShotCooldownTicks: 0,
				initialBombCooldownTicks: 0,
				aggressionFactor: 0,
			};
		case 'stationary':
			return {
				speed: 0,
				size: 30,
				color: PastelColorPalette.PALE_GRAY,
				shotCooldownTicksOnFire: 300,
				bombCooldownTicksOnPlant: 0,
				initialShotCooldownTicks: 60,
				initialBombCooldownTicks: 0,
				aggressionFactor: 0,
			};
		case 'stationary-random-aim':
			return {
				speed: 0,
				size: 30,
				color: PastelColorPalette.BABY_BLUE,
				shotCooldownTicksOnFire: 0,
				bombCooldownTicksOnPlant: 0,
				initialShotCooldownTicks: 0,
				initialBombCooldownTicks: 0,
				aggressionFactor: 0,
			};
		case 'simple-moving':
			return {
				speed: 54,
				size: 30,
				color: PastelColorPalette.CORAL_ORANGE,
				shotCooldownTicksOnFire: 0,
				bombCooldownTicksOnPlant: 0,
				initialShotCooldownTicks: 0,
				initialBombCooldownTicks: 0,
				aggressionFactor: 15,
			};
		case 'bomber':
			return {
				speed: 90,
				size: 30,
				color: PastelColorPalette.PALE_YELLOW,
				shotCooldownTicksOnFire: 1200,
				bombCooldownTicksOnPlant: 60,
				initialShotCooldownTicks: 0,
				initialBombCooldownTicks: 0,
				aggressionFactor: 4,
			};
		case 'super-bomber':
			return {
				speed: 99,
				size: 30,
				color: PastelColorPalette.BLUSH_PINK,
				shotCooldownTicksOnFire: 60,
				bombCooldownTicksOnPlant: 60,
				initialShotCooldownTicks: 60,
				initialBombCooldownTicks: 0,
				aggressionFactor: 5,
			};
	}
}

export function isEnemyKind(kind: TankKind): kind is EnemyType {
	return kind !== 'player';
}
