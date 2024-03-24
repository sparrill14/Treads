import './css/style.css';
import './assets/computer.ico';
import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
import '../node_modules/bootstrap/dist/js/bootstrap.min.js';

import { GameCanvas } from './game/GameCanvas';
import { StationaryRandomAimTank, StationaryTank } from './game/Tank';
import { Obstacle } from './game/Obstacle';
import { ObstacleCanvas } from './game/ObstacleCanvas';
import { Ammunition, BasicAIAmmunition, SuperAIAmmunition } from './game/Ammunition';

const canvasWidth = 1000;
const canvasHeight = 500;

let obs: Obstacle = new Obstacle(100, 100, 200, 100);
let obs2: Obstacle = new Obstacle(700, 100, 30, 100);
let obs3: Obstacle = new Obstacle(700, 350, 30, 100);

const obstacleCanvas = new ObstacleCanvas('#obstacle-canvas', canvasWidth, canvasHeight, [obs, obs2, obs3]);
const gameArea = new GameCanvas('#game-canvas', canvasWidth, canvasHeight, obstacleCanvas);
const aiTank2 = new StationaryTank(gameArea.gameRenderer.canvas, 800, 200, obstacleCanvas);

let basicAmmo: Ammunition[] = [
    new BasicAIAmmunition(0, 0, 0, 0, 0, true)
]
let superAmmo: Ammunition[] = [
    new SuperAIAmmunition(0, 0, 0, 0, 0, true)
]
const aiTank = new StationaryRandomAimTank(gameArea.gameRenderer.canvas, 800, 100, obstacleCanvas, superAmmo);
const aiTank3 = new StationaryRandomAimTank(gameArea.gameRenderer.canvas, 800, 300, obstacleCanvas, basicAmmo);

gameArea.addEnemyTank(aiTank);
gameArea.addEnemyTank(aiTank2);
gameArea.addEnemyTank(aiTank3);
