interface QueueEntry<T> {
	value: T;
	priority: number;
}

export class MinPriorityQueue<T> {
	private heap: QueueEntry<T>[] = [];

	public push(value: T, priority: number): void {
		this.heap.push({ value, priority });
		this.bubbleUp(this.heap.length - 1);
	}

	public pop(): QueueEntry<T> | undefined {
		if (this.heap.length === 0) {
			return undefined;
		}
		const min = this.heap[0];
		const end = this.heap.pop();
		if (end && this.heap.length > 0) {
			this.heap[0] = end;
			this.sinkDown(0);
		}
		return min;
	}

	public get size(): number {
		return this.heap.length;
	}

	private bubbleUp(index: number): void {
		while (index > 0) {
			const parentIndex = Math.floor((index - 1) / 2);
			if (this.heap[parentIndex].priority <= this.heap[index].priority) {
				return;
			}
			[this.heap[parentIndex], this.heap[index]] = [this.heap[index], this.heap[parentIndex]];
			index = parentIndex;
		}
	}

	private sinkDown(index: number): void {
		for (;;) {
			const leftIndex = index * 2 + 1;
			const rightIndex = leftIndex + 1;
			let smallestIndex = index;

			if (leftIndex < this.heap.length && this.heap[leftIndex].priority < this.heap[smallestIndex].priority) {
				smallestIndex = leftIndex;
			}
			if (rightIndex < this.heap.length && this.heap[rightIndex].priority < this.heap[smallestIndex].priority) {
				smallestIndex = rightIndex;
			}
			if (smallestIndex === index) {
				return;
			}
			[this.heap[index], this.heap[smallestIndex]] = [this.heap[smallestIndex], this.heap[index]];
			index = smallestIndex;
		}
	}
}
