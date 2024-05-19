import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
import '../node_modules/bootstrap/dist/js/bootstrap.min.js';
import './assets/computer.ico';
import './css/style.css';

import { LevelSelector } from './ui/LevelSelector';

const levelSelector: LevelSelector = new LevelSelector(7);

levelSelector.startActiveLevel();
