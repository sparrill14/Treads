import * as d3 from 'd3';
import './css/style.css';
import './assets/computer.ico';
import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
import '../node_modules/bootstrap/dist/js/bootstrap.min.js';

import { GameCanvas } from './game/GameCanvas';
import { Tank, StationaryTank, StevesTank, LivsTank } from './game/Tank';
import { StationaryCPUController } from './controllers/StationaryCPUController';
import { Obstacle } from './game/Obstacle';
import { ObstacleCanvas } from './game/ObstacleCanvas';

const canvasWidth = 1000;
const canvasHeight = 500;

let obs: Obstacle = new Obstacle(100, 100, 200, 100);
let obs2: Obstacle = new Obstacle(700, 100, 30, 300);

const obstacleConvas = new ObstacleCanvas('#obstacle-canvas', canvasWidth, canvasHeight, [obs, obs2]);
const gameArea = new GameCanvas('#game-canvas', canvasWidth, canvasHeight, obstacleConvas);
const aiTank = new StationaryTank(new StationaryCPUController(gameArea.gameRenderer.canvas), 800, 100, gameArea.width, gameArea.height, obstacleConvas);
const aiTank2 = new StationaryTank(new StationaryCPUController(gameArea.gameRenderer.canvas), 800, 200, gameArea.width, gameArea.height, obstacleConvas);
const aiTank3 = new StationaryTank(new StationaryCPUController(gameArea.gameRenderer.canvas), 800, 300, gameArea.width, gameArea.height, obstacleConvas);

gameArea.addEnemyTank(aiTank);
gameArea.addEnemyTank(aiTank2);
gameArea.addEnemyTank(aiTank3);
