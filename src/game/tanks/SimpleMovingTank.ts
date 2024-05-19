import { Ammunition } from '../Ammunition';
import { AudioManager } from '../AudioManager';
import { Bomb } from '../Bomb';
import { NavigationGrid, Node } from '../NavigationGrid';
import { ObstacleCanvas } from '../ObstacleCanvas';
import { NoReticule } from '../Reticule';
import { EnemyTank } from './EnemyTank';
import { Direction, Tank } from './Tank';

export class SimpleMovingTank extends EnemyTank {
	public aimAngleChangeAmount = 0;
	public navigationGrid: NavigationGrid;
	public aggressionFactor = 15; // Distance tank should maintain from its target
	public currentNode: Node;
	public path: Node[] | null = [];
	public pathRecaculationInterval = 60;
	public drawNavigationGrid = false;

	constructor(
		canvas: HTMLCanvasElement,
		xPos: number,
		yPos: number,
		obstacleCanvas: ObstacleCanvas,
		ammunition: Ammunition[],
		bombs: Bomb[],
		navigationGrid: NavigationGrid,
		audioManager: AudioManager
	) {
		const simpleMovingTankSpeed = 1.2;
		const simpleMovingTankSize = 30;
		const simpleMovingTankColor = '#fd8a8a';
		super(
			canvas,
			new NoReticule(),
			xPos,
			yPos,
			simpleMovingTankSpeed,
			simpleMovingTankSize,
			simpleMovingTankColor,
			obstacleCanvas,
			ammunition,
			bombs,
			audioManager
		);
		this.navigationGrid = navigationGrid;
		this.currentNode = this.navigationGrid.getNodeFromTank(this);
	}

	public override draw(context: CanvasRenderingContext2D): void {
		if (this.drawNavigationGrid) {
			context.lineWidth = 1;
			for (let i = 0; i <= this.navigationGrid.gridYLength; i++) {
				context.fillStyle = 'blue';
				context.beginPath();
				context.moveTo(0, i * this.navigationGrid.gridCellWidth);
				context.lineTo(
					this.navigationGrid.gridXLength * this.navigationGrid.gridCellWidth,
					i * this.navigationGrid.gridCellWidth
				);
				context.stroke();
			}
			for (let j = 0; j <= this.navigationGrid.gridXLength; j++) {
				context.fillStyle = 'blue';
				context.beginPath();
				context.moveTo(j * this.navigationGrid.gridCellWidth, 0);
				context.lineTo(
					j * this.navigationGrid.gridCellWidth,
					this.navigationGrid.gridXLength * this.navigationGrid.gridCellWidth
				);
				context.stroke();
			}
			context.fillStyle = this.color;
			this.path?.forEach((value: Node, index: number, array: Node[]) => {
				context.beginPath();
				context.arc(
					value.x * this.navigationGrid.gridCellWidth + this.navigationGrid.gridCellWidth / 2,
					value.y * this.navigationGrid.gridCellWidth + this.navigationGrid.gridCellWidth / 2,
					5,
					0,
					2 * Math.PI
				);
				context.fill();
			});
		}
		super.draw(context);
	}

	public override updatePosition(playerTank: Tank): void {
		this.pathRecaculationInterval -= 1;
		if (this.path == null || this.path.length == 0 || this.pathRecaculationInterval == 0) {
			this.navigationGrid.reset();
			const startNode: Node = this.navigationGrid.getNodeFromTank(this);
			const targetNode: Node = this.navigationGrid.getNodeFromTank(playerTank);
			const destinationNode: Node = this.navigationGrid.getRandomNodeInRadiusOfTarget(
				targetNode,
				this.aggressionFactor
			);
			this.path = this.navigationGrid.aStar(startNode, destinationNode);
			this.pathRecaculationInterval = 60;
			if (this.path == null) {
				console.log(`Path is null`);
			}
		} else {
			this.currentNode = this.navigationGrid.getNodeFromTank(this);
			const dx = this.path[0].x - this.currentNode.x;
			const dy = this.path[0].y - this.currentNode.y;

			if (this.wasLastMoveBlocked && this.consecutiveDirectionMoves > 2) {
				const randomDirection: Direction = this.getRandomDirection();
				this.moveInCardinalDirection(randomDirection);
				this.consecutiveDirectionMoves = 0;
				const randomNumber: number = Math.random();
				if (dx === 1 && dy === 0) {
					if (randomNumber < 0.5) {
						this.moveSouthEast();
					} else {
						this.moveNorthEast();
					}
				} else if (dx === -1 && dy === 0) {
					if (randomNumber < 0.5) {
						this.moveNorthWest();
					} else {
						this.moveSouthWest();
					}
				} else if (dx === 0 && dy === 1) {
					if (randomNumber < 0.5) {
						this.moveSouthEast();
					} else {
						this.moveSouthWest();
					}
				} else if (dx === 0 && dy === -1) {
					if (randomNumber < 0.5) {
						this.moveNorthWest();
					} else {
						this.moveNorthEast();
					}
				} else if (dx === 1 && dy === 1) {
					if (randomNumber < 0.5) {
						this.moveSouth();
					} else {
						this.moveEast();
					}
				} else if (dx === 1 && dy === -1) {
					if (randomNumber < 0.5) {
						this.moveNorth();
					} else {
						this.moveEast();
					}
				} else if (dx === -1 && dy === 1) {
					if (randomNumber < 0.5) {
						this.moveSouth();
					} else {
						this.moveWest();
					}
				} else if (dx === -1 && dy === -1) {
					if (randomNumber < 0.5) {
						this.moveNorth();
					} else {
						this.moveWest();
					}
				}
			} else {
				if (dx === 1 && dy === 0) {
					this.moveEast();
				} else if (dx === -1 && dy === 0) {
					this.moveWest();
				} else if (dx === 0 && dy === 1) {
					this.moveSouth();
				} else if (dx === 0 && dy === -1) {
					this.moveNorth();
				} else if (dx === 1 && dy === 1) {
					this.moveSouthEast();
				} else if (dx === 1 && dy === -1) {
					this.moveNorthEast();
				} else if (dx === -1 && dy === 1) {
					this.moveSouthWest();
				} else if (dx === -1 && dy === -1) {
					this.moveNorthWest();
				}
			}

			if (this.path[0].x == this.currentNode.x && this.path[0].y == this.currentNode.y) {
				this.path.splice(0, 1);
			}
		}

		this.xLeft = this.xPos;
		this.xRight = this.xPos + this.size;
		this.yTop = this.yPos;
		this.yBottom = this.yPos + this.size;
	}

	public override shoot(playerTank: Tank): void {
		const availableAmmunitionIndex = this.ammunition.findIndex((ammunition) => ammunition.isDestroyed);
		if (availableAmmunitionIndex !== -1) {
			this.ammunition[availableAmmunitionIndex].reload(
				this.xPos + this.size / 2,
				this.yPos + this.size / 2,
				this.aimAngle,
				true,
				this.canvasWidth,
				this.canvasHeight
			);
			const willHitPlayerTank: boolean = this.ammunition[availableAmmunitionIndex].willHitPlayerTank(
				this.obstacleCanvas,
				playerTank
			);
			if (willHitPlayerTank) {
				this.ammunition[availableAmmunitionIndex].isDestroyed = false;
			}
		}
		return;
	}

	public override aim(mouseXPos: number, mouseYpos: number, playerTank: Tank): void {
		if (this.isDestroyed) {
			return;
		}

		const dx: number = playerTank.xPos + playerTank.size / 2 - this.xPos - this.tankMidpoint;
		const dy: number = playerTank.yPos + playerTank.size / 2 - this.yPos - this.tankMidpoint;
		let theta = Math.atan2(dy, dx);
		if (theta < 0) {
			theta += 2 * Math.PI;
		}
		this.aimAngle = theta;
	}
}
