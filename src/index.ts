import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
import '../node_modules/bootstrap/dist/js/bootstrap.min.js';
import './css/style.css';

import { LEVEL_CONFIGS } from './game/LevelConfig';
import { LevelSelector } from './ui/LevelSelector';

new LevelSelector(LEVEL_CONFIGS.length);
