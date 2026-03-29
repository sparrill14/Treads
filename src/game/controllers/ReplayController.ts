import type { ReplayData } from '../core/types';
import type { TankAction, TankController, TankObservation } from '../core/types';

const NO_OP_ACTION: TankAction = {
	move: 'none',
	aimAngle: 0,
	fire: false,
	plantBomb: false,
};

export class ReplayController implements TankController {
	constructor(private tankId: string, private replay: ReplayData) {}

	public act(obs: TankObservation): TankAction {
		const tickRecord = this.replay.ticks.find((tick) => tick.tick === obs.tick);
		return tickRecord?.actions[this.tankId] ?? NO_OP_ACTION;
	}
}

export class PassiveTankController implements TankController {
	public act(obs: TankObservation): TankAction {
		return {
			move: 'none',
			aimAngle: obs.self.aimAngle,
			fire: false,
			plantBomb: false,
		};
	}
}
