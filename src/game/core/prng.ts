export function normalizeSeed(seed: number): number {
	const normalized = (seed >>> 0) || 0x6d2b79f5;
	return normalized;
}

export function mixSeed(seed: number, salt: number): number {
	let value = normalizeSeed(seed ^ salt);
	value ^= value >>> 16;
	value = Math.imul(value, 0x7feb352d);
	value ^= value >>> 15;
	value = Math.imul(value, 0x846ca68b);
	value ^= value >>> 16;
	return normalizeSeed(value);
}

export class SeededRandom {
	private state: number;

	constructor(seed: number) {
		this.state = normalizeSeed(seed);
	}

	public nextUint32(): number {
		let value = (this.state += 0x6d2b79f5) >>> 0;
		value = Math.imul(value ^ (value >>> 15), value | 1);
		value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
		const nextValue = (value ^ (value >>> 14)) >>> 0;
		this.state = nextValue;
		return nextValue;
	}

	public nextFloat(): number {
		return this.nextUint32() / 4294967296;
	}

	public nextInt(minInclusive: number, maxInclusive: number): number {
		if (maxInclusive < minInclusive) {
			return minInclusive;
		}
		const span = maxInclusive - minInclusive + 1;
		return minInclusive + Math.floor(this.nextFloat() * span);
	}

	public nextRange(minInclusive: number, maxExclusive: number): number {
		return minInclusive + this.nextFloat() * (maxExclusive - minInclusive);
	}

	public pick<T>(items: T[]): T {
		if (items.length === 0) {
			throw new Error('Cannot pick from an empty collection.');
		}
		const index = this.nextInt(0, items.length - 1);
		return items[index];
	}

	public getState(): number {
		return this.state >>> 0;
	}

	public setState(seed: number): void {
		this.state = normalizeSeed(seed);
	}
}
