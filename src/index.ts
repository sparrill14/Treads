import './css/style.css';
import './assets/computer.ico';
import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
import '../node_modules/bootstrap/dist/js/bootstrap.min.js';

import { LevelSelector } from './ui/LevelSelector';

const levelSelector: LevelSelector = new LevelSelector(6);

levelSelector.startActiveLevel();