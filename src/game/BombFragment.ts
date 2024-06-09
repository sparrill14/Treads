export class BombFragment {
	public life: number;

	private x: number;
	private y: number;
	private fragmentRadius: number;
	private fragmentColor: string;
	private velocityX: number;
	private velocityY: number;
	private blastRadius: number;

	constructor(x: number, y: number, radius: number, color: string, velocityX: number, velocityY: number, life: number) {
		this.x = x;
		this.y = y;
		this.fragmentRadius = radius;
		this.fragmentColor = color;
		this.velocityX = velocityX;
		this.velocityY = velocityY;
		this.blastRadius = life;
		this.life = life;
	}

	update(): void {
		this.x += this.velocityX;
		this.y += this.velocityY;
		this.life -= 1;
	}

	draw(context: CanvasRenderingContext2D): void {
		context.beginPath();
		context.arc(this.x, this.y, this.fragmentRadius * (this.life / this.blastRadius), 0, Math.PI * 2, false);
		context.fillStyle = this.fragmentColor;
		context.fill();
		context.closePath();
	}
}
