import type { ReplayData } from '../core/types';
import type { TankAction, TankController, TankObservation } from '../core/types';

const NO_OP_ACTION: TankAction = {
	move: 'none',
	aimAngle: 0,
	fire: false,
	plantBomb: false,
};

export class ReplayController implements TankController {
	private readonly actionsByTick = new Map<number, TankAction>();

	constructor(private tankId: string, replay: ReplayData) {
		for (const tickRecord of replay.ticks) {
			const action = tickRecord.actions[this.tankId];
			if (action) {
				this.actionsByTick.set(tickRecord.tick, action);
			}
		}
	}

	public act(obs: TankObservation): TankAction {
		return this.actionsByTick.get(obs.tick) ?? NO_OP_ACTION;
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
