import { Ammunition } from './Ammunition';
import { Bomb } from './Bomb';
import { Tank } from './tanks/Tank';

export class GameRenderer {
	public playerWin = false;
	public enemyWin = false;

	private context: CanvasRenderingContext2D | null;

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

	render(progress: number, playerTank: Tank, enemyTanks: Tank[]): void {
		if (!this.context) {
			throw new Error('2d context not supported or canvas element not found.');
		}
		this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);

		if (playerTank.isDestroyed) {
			this.enemyWin = true;
		} else if (enemyTanks.every((tank) => tank.isDestroyed)) {
			this.playerWin = true;
		}
		if (this.enemyWin || this.playerWin) {
			this.renderLevelOverScreen();
		}
		const allAmmunition: Ammunition[] = [
			...enemyTanks.flatMap((enemyTank) => enemyTank.ammunition),
			...playerTank.ammunition,
		];
		const allBombs: Bomb[] = [...enemyTanks.flatMap((enemyTank) => enemyTank.bombs), ...playerTank.bombs];
		playerTank.updatePosition(playerTank, playerTank, enemyTanks, allAmmunition, allBombs);
		playerTank.aim(playerTank.aimXPos, playerTank.aimYPos, playerTank);

		enemyTanks.forEach((enemyTank) => {
			if (!enemyTank.isDestroyed) {
				enemyTank.updatePosition(enemyTank, playerTank, enemyTanks, allAmmunition, allBombs);
				enemyTank.aim(enemyTank.aimXPos, enemyTank.aimYPos, playerTank);
				enemyTank.shoot(playerTank);
				enemyTank.plantBomb(playerTank);
			}
		});
		enemyTanks.forEach((enemyTank) => {
			enemyTank.draw(this.context as CanvasRenderingContext2D);
			enemyTank.ammunition.forEach((ammunition) => {
				if (ammunition.isDestroyed) {
					return;
				}
				ammunition.checkAmmunitionCollision(allAmmunition);
				ammunition.checkBombCollision([...playerTank.bombs]);
				ammunition.updatePosition(enemyTank.obstacleCanvas);
				ammunition.checkPlayerHit(playerTank);
				ammunition.draw(this.context as CanvasRenderingContext2D);
			});
			enemyTank.bombs.forEach((bomb) => {
				if (bomb.isDestroyed && !bomb.isExploding()) {
					return;
				}
				bomb.checkPlayerHit(playerTank);
				bomb.draw(this.context as CanvasRenderingContext2D);
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
			if (ammunition.isDestroyed) {
				return;
			}
			ammunition.checkAmmunitionCollision(allAmmunition);
			ammunition.checkBombCollision(allBombs);
			ammunition.updatePosition(playerTank.obstacleCanvas);
			ammunition.checkEnemyHit(enemyTanks);
			ammunition.checkPlayerHit(playerTank);
			ammunition.draw(this.context as CanvasRenderingContext2D);
		});
		playerTank.bombs.forEach((bomb) => {
			if (bomb.isDestroyed && !bomb.isExploding()) {
				return;
			}
			bomb.checkEnemyHit(enemyTanks);
			bomb.checkPlayerHit(playerTank);
			bomb.draw(this.context as CanvasRenderingContext2D);
		});
	}
}
