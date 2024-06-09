export class Obstacle {
	public xLeft: number;
	public xRight: number;
	public yTop: number;
	public yBottom: number;
	public width: number;
	public height: number;

	constructor(startX: number, startY: number, width: number, height: number) {
		this.xLeft = startX;
		this.yTop = startY;
		this.width = width;
		this.height = height;
		this.yBottom = this.yTop + this.height;
		this.xRight = this.xLeft + this.width;
	}

	draw(context: CanvasRenderingContext2D): void {
		context.beginPath();
		context.rect(this.xLeft, this.yTop, this.width, this.height);
		context.fillStyle = '#1d1c1a';
		context.fill();
	}
}
