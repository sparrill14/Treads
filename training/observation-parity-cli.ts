import fs from 'fs';

import type { TankObservation } from '../src/game/core/types';
import { normalizeWorkerObservation } from './rollout-worker';

const payload = JSON.parse(fs.readFileSync(0, 'utf8')) as TankObservation[];
process.stdout.write(JSON.stringify(payload.map((observation) => normalizeWorkerObservation(observation))));
