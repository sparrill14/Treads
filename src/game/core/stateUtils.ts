import type { GameState, TankObservation } from './types';

export type DeepReadonly<T> = T extends (...args: never[]) => unknown
	? T
	: T extends readonly (infer U)[]
		? readonly DeepReadonly<U>[]
		: T extends object
			? { readonly [K in keyof T]: DeepReadonly<T[K]> }
			: T;

export function cloneJson<T>(value: T): T {
	return JSON.parse(JSON.stringify(value)) as T;
}

function freezeObject<T>(value: T): DeepReadonly<T> {
	if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
		Object.freeze(value);
		for (const nestedValue of Object.values(value as Record<string, unknown>)) {
			freezeObject(nestedValue);
		}
	}
	return value as DeepReadonly<T>;
}

export function freezeObservation<T extends TankObservation>(observation: T): T {
	return freezeObject(observation) as T;
}

export function createReadonlySnapshot<T>(value: T): DeepReadonly<T> {
	return freezeObject(cloneJson(value));
}

export function cloneGameState(state: GameState | DeepReadonly<GameState>): GameState {
	return cloneJson(state) as GameState;
}

export function serializeGameState(state: GameState | DeepReadonly<GameState>): string {
	return JSON.stringify(state);
}
