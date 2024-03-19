import './css/style.css';
import './assets/computer.ico';
import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
import '../node_modules/bootstrap/dist/js/bootstrap.min.js';

import { GameCanvas } from './game/GameCanvas';
import { StationaryTank } from './game/Tank';
import { Obstacle } from './game/Obstacle';
import { ObstacleCanvas } from './game/ObstacleCanvas';

const canvasWidth = 1000;
const canvasHeight = 500;

let obs: Obstacle = new Obstacle(100, 100, 200, 100);
let obs2: Obstacle = new Obstacle(700, 100, 30, 300);

const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', canvasWidth, canvasHeight, [obs, obs2]);
const gameArea = new GameCanvas('#game-canvas', canvasWidth, canvasHeight, obstacleCanvas);
const aiTank = new StationaryTank(gameArea.gameRenderer.canvas, 800, 100, obstacleCanvas);
const aiTank2 = new StationaryTank(gameArea.gameRenderer.canvas, 800, 200, obstacleCanvas);
const aiTank3 = new StationaryTank(gameArea.gameRenderer.canvas, 800, 300, obstacleCanvas);

gameArea.addEnemyTank(aiTank);
gameArea.addEnemyTank(aiTank2);
gameArea.addEnemyTank(aiTank3);
