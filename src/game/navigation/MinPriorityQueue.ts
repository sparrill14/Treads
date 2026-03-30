export class MinPriorityQueue<T> {
	private values: T[] = [];
	private priorities: number[] = [];

	public push(value: T, priority: number): void {
		const idx = this.values.length;
		this.values.push(value);
		this.priorities.push(priority);
		this.bubbleUp(idx);
	}

	public popValue: T | undefined;
	public popPriority = 0;

	public pop(): boolean {
		const len = this.values.length;
		if (len === 0) {
			return false;
		}
		this.popValue = this.values[0];
		this.popPriority = this.priorities[0];
		const lastIdx = len - 1;
		if (lastIdx > 0) {
			this.values[0] = this.values[lastIdx];
			this.priorities[0] = this.priorities[lastIdx];
		}
		this.values.length = lastIdx;
		this.priorities.length = lastIdx;
		if (lastIdx > 0) {
			this.sinkDown(0);
		}
		return true;
	}

	public get size(): number {
		return this.values.length;
	}

	public clear(): void {
		this.values.length = 0;
		this.priorities.length = 0;
	}

	private bubbleUp(index: number): void {
		const priorities = this.priorities;
		const values = this.values;
		while (index > 0) {
			const parentIndex = (index - 1) >> 1;
			if (priorities[parentIndex] <= priorities[index]) {
				return;
			}
			const tmpP = priorities[parentIndex];
			priorities[parentIndex] = priorities[index];
			priorities[index] = tmpP;
			const tmpV = values[parentIndex];
			values[parentIndex] = values[index];
			values[index] = tmpV;
			index = parentIndex;
		}
	}

	private sinkDown(index: number): void {
		const priorities = this.priorities;
		const values = this.values;
		const length = priorities.length;
		for (;;) {
			const leftIndex = index * 2 + 1;
			const rightIndex = leftIndex + 1;
			let smallestIndex = index;

			if (leftIndex < length && priorities[leftIndex] < priorities[smallestIndex]) {
				smallestIndex = leftIndex;
			}
			if (rightIndex < length && priorities[rightIndex] < priorities[smallestIndex]) {
				smallestIndex = rightIndex;
			}
			if (smallestIndex === index) {
				return;
			}
			const tmpP = priorities[index];
			priorities[index] = priorities[smallestIndex];
			priorities[smallestIndex] = tmpP;
			const tmpV = values[index];
			values[index] = values[smallestIndex];
			values[smallestIndex] = tmpV;
			index = smallestIndex;
		}
	}
}
