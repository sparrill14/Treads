import { CollisionManager } from './CollisionManager';
import { Tank } from './tanks/Tank';

export class GameRenderer {
	public playerWin = false;
	public enemyWin = false;

	private context: CanvasRenderingContext2D | null;
	private collisionManager: CollisionManager | null = null;

	constructor(public canvas: HTMLCanvasElement) {
		const context = this.canvas.getContext('2d');
		if (!context) {
			throw new Error('2d context not supported or canvas element not found.');
		}
		this.context = context;
	}

	public initializeCanvas(width: number, height: number): void {
		this.canvas.width = width;
		this.canvas.height = height;
	}

	public setCollisionManager(collisionManager: CollisionManager): void {
		this.collisionManager = collisionManager;
	}

	renderLevelOverScreen() {
		if (!this.context) {
			throw new Error('2d context not supported or canvas element not found.');
		}

		const message = this.playerWin ? 'Win' : this.enemyWin ? 'Lose' : '';
		if (message === '') {
			return;
		}

		const fontSize = 100;
		this.context.font = `${fontSize}px Arial`;
		this.context.lineWidth = 5;
		this.context.strokeStyle = this.playerWin ? 'green' : 'red';
		this.context.fillStyle = this.playerWin ? 'green' : 'red';

		const textWidth = this.context.measureText(message).width;
		const x = (this.canvas.width - textWidth) / 2;
		const y = this.canvas.height / 2 + fontSize / 2;

		this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
		this.context.strokeText(message, x, y);
		this.context.fillText(message, x, y);
	}

	update(deltaTime: number, playerTank: Tank, enemyTanks: Tank[]): void {
		if (playerTank.isDestroyed) {
			this.enemyWin = true;
		} else if (enemyTanks.every((tank) => tank.isDestroyed)) {
			this.playerWin = true;
		}

		if (this.enemyWin || this.playerWin) {
			return;
		}

		const allAmmunition = [...enemyTanks.flatMap((enemyTank) => enemyTank.ammunition), ...playerTank.ammunition];
		const allBombs = [...enemyTanks.flatMap((enemyTank) => enemyTank.bombs), ...playerTank.bombs];

		playerTank.dt = deltaTime;
		playerTank.updatePosition(playerTank, playerTank, enemyTanks, allAmmunition, allBombs);
		playerTank.aim(playerTank.aimXPos, playerTank.aimYPos, playerTank);

		enemyTanks.forEach((enemyTank) => {
			if (!enemyTank.isDestroyed) {
				enemyTank.dt = deltaTime;
				enemyTank.updatePosition(enemyTank, playerTank, enemyTanks, allAmmunition, allBombs);
				enemyTank.aim(enemyTank.aimXPos, enemyTank.aimYPos, playerTank);
				enemyTank.shoot(playerTank);
				enemyTank.plantBomb(playerTank);
			}
		});

		this.collisionManager?.update(playerTank, enemyTanks, playerTank.obstacleCanvas, deltaTime);
	}

	render(playerTank: Tank, enemyTanks: Tank[]): void {
		if (!this.context) {
			throw new Error('2d context not supported or canvas element not found.');
		}
		this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);

		if (this.enemyWin || this.playerWin) {
			this.renderLevelOverScreen();
			return;
		}

		enemyTanks.forEach((enemyTank) => {
			enemyTank.draw(this.context as CanvasRenderingContext2D);
			enemyTank.ammunition.forEach((ammunition) => {
				if (!ammunition.isDestroyed) {
					ammunition.draw(this.context as CanvasRenderingContext2D);
				}
			});
			enemyTank.bombs.forEach((bomb) => {
				if (!bomb.isDestroyed || bomb.isExploding()) {
					bomb.draw(this.context as CanvasRenderingContext2D);
				}
			});
		});

		playerTank.draw(this.context as CanvasRenderingContext2D);
		if (!playerTank.isDestroyed) {
			playerTank.reticule.draw(
				this.context as CanvasRenderingContext2D,
				playerTank.xPosition,
				playerTank.yPosition,
				playerTank.aimXPos,
				playerTank.aimYPos
			);
		}
		playerTank.ammunition.forEach((ammunition) => {
			if (!ammunition.isDestroyed) {
				ammunition.draw(this.context as CanvasRenderingContext2D);
			}
		});
		playerTank.bombs.forEach((bomb) => {
			if (!bomb.isDestroyed || bomb.isExploding()) {
				bomb.draw(this.context as CanvasRenderingContext2D);
			}
		});
	}
}
