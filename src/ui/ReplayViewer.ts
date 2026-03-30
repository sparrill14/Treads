import { GameRenderer } from '../game/GameRenderer';
import { createInitialGameState } from '../game/core/MatchFactory';
import { createReplayControllers } from '../game/core/Replay';
import { Simulation } from '../game/core/Simulation';
import type { GameState, ReplayData } from '../game/core/types';

export class ReplayViewer {
	private simulation: Simulation | null = null;
	private renderer: GameRenderer | null = null;
	private replay: ReplayData | null = null;
	private animationFrameId: number | null = null;
	private currentTick = 0;
	private totalTicks = 0;
	private playing = false;
	private playbackSpeed = 1;
	private previousState: GameState | null = null;
	private lastFrameTime = 0;
	private accumulatorMs = 0;
	private tickMs = 1000 / 60;

	private container: HTMLDivElement | null = null;
	private scrubBar: HTMLInputElement | null = null;
	private tickLabel: HTMLSpanElement | null = null;
	private playBtn: HTMLButtonElement | null = null;
	private onClose: (() => void) | null = null;

	public show(canvas: HTMLCanvasElement, onClose: () => void): void {
		this.onClose = onClose;
		this.renderer = new GameRenderer(canvas);
		this.buildControls(canvas);
	}

	public destroy(): void {
		this.stop();
		this.container?.remove();
		this.container = null;
		this.replay = null;
		this.simulation = null;
		this.renderer = null;
	}

	private buildControls(canvas: HTMLCanvasElement): void {
		this.container = document.createElement('div');
		this.container.id = 'replay-controls';
		this.container.style.cssText =
			'text-align:center; margin:8px 0; display:flex; gap:6px; align-items:center; justify-content:center; flex-wrap:wrap;';

		const loadBtn = this.makeButton('Load Replay', () => this.openFilePicker());
		this.playBtn = this.makeButton('Play', () => this.togglePlay());
		this.playBtn.disabled = true;

		const speedLabel = document.createElement('span');
		speedLabel.style.cssText = 'color:#ccc; font-size:12px;';
		speedLabel.textContent = '1x';

		const speedDown = this.makeButton('-', () => {
			this.playbackSpeed = Math.max(0.25, this.playbackSpeed / 2);
			speedLabel.textContent = `${this.playbackSpeed}x`;
		});
		const speedUp = this.makeButton('+', () => {
			this.playbackSpeed = Math.min(8, this.playbackSpeed * 2);
			speedLabel.textContent = `${this.playbackSpeed}x`;
		});

		this.scrubBar = document.createElement('input');
		this.scrubBar.type = 'range';
		this.scrubBar.min = '0';
		this.scrubBar.max = '0';
		this.scrubBar.value = '0';
		this.scrubBar.style.cssText = 'width:200px;';
		this.scrubBar.addEventListener('input', () => {
			const tick = parseInt(this.scrubBar?.value ?? '0', 10);
			this.seekToTick(tick);
		});

		this.tickLabel = document.createElement('span');
		this.tickLabel.style.cssText = 'color:#ccc; font-size:12px; min-width:80px;';
		this.tickLabel.textContent = '0 / 0';

		const closeBtn = this.makeButton('Close', () => {
			this.destroy();
			this.onClose?.();
		});

		this.container.append(
			loadBtn,
			this.playBtn,
			speedDown,
			speedLabel,
			speedUp,
			this.scrubBar,
			this.tickLabel,
			closeBtn
		);
		canvas.parentElement?.insertBefore(this.container, canvas);
	}

	private makeButton(text: string, onclick: () => void): HTMLButtonElement {
		const btn = document.createElement('button');
		btn.textContent = text;
		btn.className = 'btn btn-outline-secondary btn-sm';
		btn.addEventListener('click', onclick);
		return btn;
	}

	private openFilePicker(): void {
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.json';
		input.addEventListener('change', () => {
			const file = input.files?.[0];
			if (!file) return;
			const reader = new FileReader();
			reader.onload = () => {
				try {
					const data = JSON.parse(reader.result as string) as ReplayData;
					this.loadReplay(data);
				} catch (err) {
					console.error('Failed to parse replay:', err);
				}
			};
			reader.readAsText(file);
		});
		input.click();
	}

	private loadReplay(replay: ReplayData): void {
		this.stop();
		this.replay = replay;
		this.totalTicks = replay.ticks.length;
		this.currentTick = 0;

		if (this.scrubBar) {
			this.scrubBar.max = String(this.totalTicks);
			this.scrubBar.value = '0';
		}
		if (this.playBtn) this.playBtn.disabled = false;
		this.updateTickLabel();

		this.seekToTick(0);
	}

	private seekToTick(targetTick: number): void {
		if (!this.replay || !this.renderer) return;
		targetTick = Math.max(0, Math.min(targetTick, this.totalTicks));

		const initialState = createInitialGameState(this.replay.levelConfig, this.replay.seed);
		const controllers = createReplayControllers(initialState, this.replay);
		this.simulation = new Simulation(initialState, controllers);
		this.renderer.initializeCanvas(initialState.arena.width, initialState.arena.height);

		this.previousState = null;
		for (let t = 0; t < targetTick; t++) {
			this.previousState = this.simulation.getStateSnapshot();
			const stepResult = this.simulation.step();
			this.renderer.setLastActions(stepResult.actions);
			this.renderer.consumeEvents(stepResult.events);
		}

		this.currentTick = targetTick;
		this.updateTickLabel();
		if (this.scrubBar) this.scrubBar.value = String(targetTick);

		const state = this.simulation.getState();
		this.renderer.render(state, this.previousState, 1);
	}

	private togglePlay(): void {
		if (this.playing) {
			this.stop();
		} else {
			this.play();
		}
	}

	private play(): void {
		if (!this.simulation || !this.renderer || !this.replay) return;
		this.playing = true;
		if (this.playBtn) this.playBtn.textContent = 'Pause';
		this.lastFrameTime = performance.now();
		this.accumulatorMs = 0;
		this.animationFrameId = requestAnimationFrame((t) => this.gameLoop(t));
	}

	private stop(): void {
		this.playing = false;
		if (this.playBtn) this.playBtn.textContent = 'Play';
		if (this.animationFrameId !== null) {
			cancelAnimationFrame(this.animationFrameId);
			this.animationFrameId = null;
		}
	}

	private gameLoop(timestamp: number): void {
		if (!this.playing || !this.simulation || !this.renderer || !this.replay) return;

		const frameDelta = Math.min(timestamp - this.lastFrameTime, 250);
		this.lastFrameTime = timestamp;
		this.accumulatorMs += frameDelta * this.playbackSpeed;

		while (this.accumulatorMs >= this.tickMs && this.currentTick < this.totalTicks) {
			this.previousState = this.simulation.getStateSnapshot();
			const stepResult = this.simulation.step();
			this.renderer.setLastActions(stepResult.actions);
			this.renderer.consumeEvents(stepResult.events);
			this.currentTick++;
			this.accumulatorMs -= this.tickMs;
		}

		this.updateTickLabel();
		if (this.scrubBar) this.scrubBar.value = String(this.currentTick);

		const state = this.simulation.getState();
		this.renderer.updateVisuals(frameDelta / 1000);
		this.renderer.render(state, this.previousState, this.accumulatorMs / this.tickMs);

		if (this.currentTick >= this.totalTicks) {
			this.stop();
			return;
		}

		this.animationFrameId = requestAnimationFrame((t) => this.gameLoop(t));
	}

	private updateTickLabel(): void {
		if (this.tickLabel) {
			this.tickLabel.textContent = `${this.currentTick} / ${this.totalTicks}`;
		}
	}
}
