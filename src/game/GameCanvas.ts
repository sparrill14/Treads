import { AudioFile, AudioManager } from './AudioManager';
import { Simulation } from './core/Simulation';
import type { GameState, SimulationEvent } from './core/types';
import { GameRenderer } from './GameRenderer';

export class GameCanvas {
	public gameRenderer: GameRenderer;
	public animationFrameID: number | null = null;
	public readonly width: number;
	public readonly height: number;
	private readonly tickMs: number;
	private lastFrameTime = 0;
	private accumulatorMs = 0;
	private previousState: GameState | null = null;
	private readonly onKeyDown = (e: KeyboardEvent): void => {
		if (e.key === 'F2') {
			this.gameRenderer.diagnosticsEnabled = !this.gameRenderer.diagnosticsEnabled;
		}
	};

	constructor(
		canvasSelector: string,
		private simulation: Simulation,
		private audioManager?: AudioManager
	) {
		const canvas = document.querySelector(canvasSelector) as HTMLCanvasElement;
		this.gameRenderer = new GameRenderer(canvas);
		const initialState = this.simulation.getState();
		const { width, height } = initialState.arena;
		this.width = width;
		this.height = height;
		this.gameRenderer.initializeCanvas(width, height);
		this.tickMs = 1000 / initialState.tickRate;
		this.previousState = this.simulation.getStateSnapshot();
		document.addEventListener('keydown', this.onKeyDown);
	}

	public start(): void {
		if (this.animationFrameID === null) {
			this.lastFrameTime = performance.now();
			this.animationFrameID = requestAnimationFrame(this.gameLoop.bind(this));
		}
	}

	public stop(): void {
		if (this.animationFrameID !== null) {
			cancelAnimationFrame(this.animationFrameID);
			this.animationFrameID = null;
		}
		document.removeEventListener('keydown', this.onKeyDown);
	}

	private gameLoop(timeStamp: number): void {
		const frameDelta = Math.min(timeStamp - this.lastFrameTime, 250);
		this.lastFrameTime = timeStamp;
		this.accumulatorMs += frameDelta;
		let currentState = this.simulation.getState();

		while (this.accumulatorMs >= this.tickMs && currentState.status === 'running') {
			this.previousState = this.simulation.getStateSnapshot();
			const stepResult = this.simulation.step();
			this.gameRenderer.setLastActions(stepResult.actions);
			this.handleEvents(stepResult.events);
			this.accumulatorMs -= this.tickMs;
			currentState = this.simulation.getState();
		}

		this.gameRenderer.updateVisuals(frameDelta / 1000);
		this.gameRenderer.render(currentState, this.previousState, this.accumulatorMs / this.tickMs);
		this.animationFrameID = requestAnimationFrame(this.gameLoop.bind(this));
	}

	private handleEvents(events: SimulationEvent[]): void {
		this.gameRenderer.consumeEvents(events);
		if (!this.audioManager) {
			return;
		}
		for (const event of events) {
			if (!('audioCue' in event)) {
				continue;
			}
			switch (event.audioCue) {
				case 'tank-fire':
					this.audioManager.play(AudioFile.TANK_FIRE);
					break;
				case 'tank-destroy':
					this.audioManager.play(AudioFile.TANK_DESTROY);
					break;
				case 'bomb-explode':
					this.audioManager.play(AudioFile.BOMB_EXPLODE);
					break;
				case 'ammunition-explode':
					this.audioManager.play(AudioFile.AMMUNITION_EXPLODE);
					break;
			}
		}
	}
}
