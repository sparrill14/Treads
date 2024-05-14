import './css/style.css';
import './assets/computer.ico';
import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
import '../node_modules/bootstrap/dist/js/bootstrap.min.js';

import { Level1 } from './game/Level';
import { LevelSelector } from './ui/LevelSelector';

const levelSelector = new LevelSelector(5);

levelSelector.startActiveLevel();