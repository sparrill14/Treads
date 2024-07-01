export class Node {
	public x: number;
	public y: number;
	public g = 0;
	public h = 0;
	public f = 0;
	public walkable = true;
	public parent: Node | null = null;

	constructor(x: number, y: number) {
		this.x = x;
		this.y = y;
	}
}
