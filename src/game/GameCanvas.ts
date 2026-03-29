import { AudioFile, AudioManager } from './AudioManager';
import { Simulation } from './core/Simulation';
import type { GameState, SimulationEvent } from './core/types';
import { cloneGameState } from './core/stateUtils';
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

	constructor(canvasSelector: string, private simulation: Simulation, private audioManager?: AudioManager) {
		const canvas = document.querySelector(canvasSelector) as HTMLCanvasElement;
		this.gameRenderer = new GameRenderer(canvas);
		const { width, height } = this.simulation.getState().arena;
		this.width = width;
		this.height = height;
		this.gameRenderer.initializeCanvas(width, height);
		this.tickMs = 1000 / this.simulation.getState().tickRate;
		this.previousState = cloneGameState(this.simulation.getState());
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
	}

	private gameLoop(timeStamp: number): void {
		const frameDelta = Math.min(timeStamp - this.lastFrameTime, 250);
		this.lastFrameTime = timeStamp;
		this.accumulatorMs += frameDelta;

		while (this.accumulatorMs >= this.tickMs && this.simulation.getState().status === 'running') {
			this.previousState = cloneGameState(this.simulation.getState());
			const stepResult = this.simulation.step();
			this.handleEvents(stepResult.events);
			this.accumulatorMs -= this.tickMs;
		}

		this.gameRenderer.updateVisuals(frameDelta / 1000);
		this.gameRenderer.render(this.simulation.getState(), this.previousState, this.accumulatorMs / this.tickMs);
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

