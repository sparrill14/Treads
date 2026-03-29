import type { GameState, TankObservation } from './types';

export function cloneJson<T>(value: T): T {
	return JSON.parse(JSON.stringify(value)) as T;
}

function freezeObject<T>(value: T): T {
	if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
		Object.freeze(value);
		for (const nestedValue of Object.values(value as Record<string, unknown>)) {
			freezeObject(nestedValue);
		}
	}
	return value;
}

export function freezeObservation<T extends TankObservation>(observation: T): Readonly<T> {
	return freezeObject(observation);
}

export function cloneGameState(state: GameState): GameState {
	return cloneJson(state);
}

export function serializeGameState(state: GameState): string {
	return JSON.stringify(state);
}
