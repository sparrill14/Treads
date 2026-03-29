import './css/style.css';

import { LEVEL_CONFIGS } from './game/LevelConfig';
import { LevelSelector } from './ui/LevelSelector';

new LevelSelector(LEVEL_CONFIGS.length);
