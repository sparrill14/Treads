import { normalizeAngle } from '../core/geometry';
import type { TankAction, TankController, TankObservation } from '../core/types';
import { InputManager } from '../../utils/InputManager';

function getMoveIntent(inputManager: InputManager): TankAction['move'] {
	if (inputManager.up() && inputManager.right()) return 'ne';
	if (inputManager.up() && inputManager.left()) return 'nw';
	if (inputManager.down() && inputManager.right()) return 'se';
	if (inputManager.down() && inputManager.left()) return 'sw';
	if (inputManager.up()) return 'n';
	if (inputManager.down()) return 's';
	if (inputManager.left()) return 'w';
	if (inputManager.right()) return 'e';
	return 'none';
}

export class HumanInputController implements TankController {
	constructor(private inputManager: InputManager) {}

	public act(obs: TankObservation): TankAction {
		const centerX = obs.self.x + obs.self.size / 2;
		const centerY = obs.self.y + obs.self.size / 2;
		const aimAngle = normalizeAngle(Math.atan2(this.inputManager.mouseY - centerY, this.inputManager.mouseX - centerX));
		return {
			move: getMoveIntent(this.inputManager),
			aimAngle,
			fire: this.inputManager.consumeShoot(),
			plantBomb: this.inputManager.consumeBomb(),
			aimTarget: {
				x: this.inputManager.mouseX,
				y: this.inputManager.mouseY,
			},
		};
	}
}
