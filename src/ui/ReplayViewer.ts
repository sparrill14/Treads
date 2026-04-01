import { GameRenderer } from '../game/GameRenderer';
import { createInitialGameState } from '../game/core/MatchFactory';
import { createReplayControllers } from '../game/core/Replay';
import { Simulation } from '../game/core/Simulation';
import type { GameState, ReplayData } from '../game/core/types';
import { DashboardApiClient, type ReplaySummary } from './dashboardApi';

interface ReplayViewerOptions {
	onReplayLoaded?: (summary: ReplaySummary) => void;
	onReplayCleared?: () => void;
}

function formatBytes(value: number): string {
	if (value >= 1_000_000) {
		return `${(value / 1_000_000).toFixed(1)} MB`;
	}
	if (value >= 1_000) {
		return `${(value / 1_000).toFixed(1)} KB`;
	}
	return `${value} B`;
}

export class ReplayViewer {
	private readonly api: DashboardApiClient;
	private readonly options: ReplayViewerOptions;
	private simulation: Simulation | null = null;
	private renderer: GameRenderer | null = null;
	private replay: ReplayData | null = null;
	private activeReplay: ReplaySummary | null = null;
	private animationFrameId: number | null = null;
	private currentTick = 0;
	private totalTicks = 0;
	private playing = false;
	private playbackSpeed = 1;
	private previousState: GameState | null = null;
	private lastFrameTime = 0;
	private accumulatorMs = 0;
	private readonly tickMs = 1000 / 60;
	private selectedRunId: string | null = null;
	private allReplays: ReplaySummary[] = [];
	private filteredReplays: ReplaySummary[] = [];

	private canvas: HTMLCanvasElement | null = null;
	private container: HTMLElement | null = null;
	private libraryList: HTMLDivElement | null = null;
	private searchInput: HTMLInputElement | null = null;
	private resultMeta: HTMLDivElement | null = null;
	private scrubBar: HTMLInputElement | null = null;
	private tickLabel: HTMLSpanElement | null = null;
	private playBtn: HTMLButtonElement | null = null;
	private speedLabel: HTMLSpanElement | null = null;
	private currentReplayCard: HTMLDivElement | null = null;
	private loadingReplayFileName: string | null = null;
	private loadErrorMessage: string | null = null;

	public constructor(api: DashboardApiClient, options: ReplayViewerOptions = {}) {
		this.api = api;
		this.options = options;
	}

	public mount(container: HTMLElement, canvas: HTMLCanvasElement): void {
		this.container = container;
		this.canvas = canvas;
		this.renderer = new GameRenderer(canvas);
		container.innerHTML = '';
		container.classList.add('replay-library');

		const header = document.createElement('div');
		header.className = 'panel-header';
		const copy = document.createElement('div');
		copy.className = 'panel-header-copy';
		const eyebrow = document.createElement('span');
		eyebrow.className = 'panel-eyebrow';
		eyebrow.textContent = 'Replay Browser';
		const title = document.createElement('h2');
		title.textContent = 'Saved Matches';
		copy.append(eyebrow, title);
		header.appendChild(copy);

		this.searchInput = document.createElement('input');
		this.searchInput.type = 'search';
		this.searchInput.placeholder = 'Filter by level, outcome, or episode';
		this.searchInput.className = 'replay-search';
		this.searchInput.addEventListener('input', () => this.applyReplayFilter());

		this.resultMeta = document.createElement('div');
		this.resultMeta.className = 'replay-result-meta';

		this.currentReplayCard = document.createElement('div');
		this.currentReplayCard.className = 'panel-card replay-current-card';

		const controls = document.createElement('div');
		controls.className = 'replay-controls';

		this.playBtn = this.makeButton('Play', () => this.togglePlay());
		this.playBtn.disabled = true;

		const speedDown = this.makeButton('Slower', () => this.setPlaybackSpeed(this.playbackSpeed / 2));
		const speedUp = this.makeButton('Faster', () => this.setPlaybackSpeed(this.playbackSpeed * 2));
		this.speedLabel = document.createElement('span');
		this.speedLabel.className = 'speed-label';
		this.speedLabel.textContent = '1x';

		const liveBtn = this.makeButton('Return To Arena', () => this.clearReplay());
		liveBtn.classList.add('ghost');

		controls.append(this.playBtn, speedDown, this.speedLabel, speedUp, liveBtn);

		this.scrubBar = document.createElement('input');
		this.scrubBar.type = 'range';
		this.scrubBar.min = '0';
		this.scrubBar.max = '0';
		this.scrubBar.value = '0';
		this.scrubBar.className = 'replay-scrubber';
		this.scrubBar.addEventListener('input', () => {
			const tick = Number(this.scrubBar?.value ?? '0');
			this.seekToTick(tick);
		});

		this.tickLabel = document.createElement('span');
		this.tickLabel.className = 'tick-label';
		this.tickLabel.textContent = 'No replay loaded';

		this.libraryList = document.createElement('div');
		this.libraryList.className = 'replay-list';

		container.append(
			header,
			this.searchInput,
			this.resultMeta,
			this.currentReplayCard,
			controls,
			this.scrubBar,
			this.tickLabel,
			this.libraryList
		);
		this.renderCurrentReplay();
	}

	public destroy(): void {
		this.stop();
		this.container = null;
		this.canvas = null;
		this.simulation = null;
		this.replay = null;
		this.activeReplay = null;
	}

	public hasActiveReplay(): boolean {
		return this.activeReplay !== null;
	}

	public exitReplay(): void {
		this.clearReplay();
	}

	public async setRun(runId: string | null): Promise<void> {
		this.selectedRunId = runId;
		if (!runId || !this.libraryList) {
			this.allReplays = [];
			this.filteredReplays = [];
			this.renderReplayList();
			return;
		}
		try {
			this.allReplays = await this.api.fetchReplays(runId);
		} catch (error) {
			console.error(`Failed to load replays for ${runId}:`, error);
			this.allReplays = [];
		}
		this.applyReplayFilter();
	}

	private makeButton(text: string, onClick: () => void): HTMLButtonElement {
		const button = document.createElement('button');
		button.type = 'button';
		button.className = 'control-button';
		button.textContent = text;
		button.addEventListener('click', onClick);
		return button;
	}

	private setPlaybackSpeed(nextSpeed: number): void {
		this.playbackSpeed = Math.max(0.25, Math.min(8, nextSpeed));
		if (this.speedLabel) {
			this.speedLabel.textContent = `${this.playbackSpeed}x`;
		}
	}

	private applyReplayFilter(): void {
		const query = this.searchInput?.value.trim().toLowerCase() ?? '';
		this.filteredReplays = this.allReplays.filter((replay) => {
			if (!query) {
				return true;
			}
			return (
				replay.fileName.toLowerCase().includes(query) ||
				String(replay.episode ?? '').includes(query) ||
				String(replay.level ?? '').includes(query) ||
				replay.outcome.toLowerCase().includes(query)
			);
		});
		this.renderReplayList();
	}

	private renderReplayList(): void {
		if (!this.libraryList || !this.resultMeta) {
			return;
		}
		this.libraryList.innerHTML = '';
		if (this.loadingReplayFileName) {
			this.resultMeta.textContent = `Loading ${this.loadingReplayFileName}...`;
		} else {
			this.resultMeta.textContent = `${this.filteredReplays.length} replay${this.filteredReplays.length === 1 ? '' : 's'} ready`;
		}

		if (this.filteredReplays.length === 0) {
			const empty = document.createElement('div');
			empty.className = 'empty-state';
			empty.textContent = this.selectedRunId
				? 'This run has no saved replay JSON files yet.'
				: 'Select a run from the console to browse its replay archive.';
			this.libraryList.appendChild(empty);
			return;
		}

		for (const replay of this.filteredReplays) {
			const card = document.createElement('button');
			card.type = 'button';
			card.className = `replay-card${this.activeReplay?.fileName === replay.fileName ? ' selected' : ''}${
				this.loadingReplayFileName === replay.fileName ? ' loading' : ''
			}`;
			card.disabled = this.loadingReplayFileName !== null;
			card.addEventListener('click', () => {
				void this.loadReplayFromSummary(replay);
			});

			const top = document.createElement('div');
			top.className = 'replay-card-top';
			const title = document.createElement('strong');
			title.textContent = `Episode ${replay.episode ?? '--'} • L${replay.level ?? '--'}`;
			const outcome = document.createElement('span');
			outcome.className = `status-pill ${replay.outcome.includes('win') ? 'running' : 'idle'}`;
			outcome.textContent = replay.outcome.replace(/_/g, ' ');
			top.append(title, outcome);

			const meta = document.createElement('div');
			meta.className = 'replay-card-meta';
			meta.textContent = `${formatBytes(replay.sizeBytes)} • ${new Date(replay.updatedAt).toLocaleString()}`;

			card.append(top, meta);
			this.libraryList.appendChild(card);
		}
	}

	private async loadReplayFromSummary(summary: ReplaySummary): Promise<void> {
		if (this.loadingReplayFileName) {
			return;
		}

		this.loadingReplayFileName = summary.fileName;
		this.loadErrorMessage = null;
		if (this.playBtn) {
			this.playBtn.disabled = true;
		}
		if (this.scrubBar) {
			this.scrubBar.disabled = true;
		}
		if (this.tickLabel) {
			this.tickLabel.textContent = 'Loading replay...';
		}
		this.renderCurrentReplay();
		this.renderReplayList();

		try {
			const replay = await this.api.fetchReplayJson<ReplayData>(summary.runId, summary.fileName);
			this.loadReplay(replay, summary);
		} catch (error) {
			console.error(`Failed to load replay ${summary.fileName}:`, error);
			this.loadErrorMessage = `Failed to load ${summary.fileName}.`;
			this.renderCurrentReplay();
			if (this.tickLabel) {
				this.tickLabel.textContent = 'Replay load failed';
			}
		} finally {
			this.loadingReplayFileName = null;
			if (this.playBtn) {
				this.playBtn.disabled = this.activeReplay === null;
			}
			if (this.scrubBar) {
				this.scrubBar.disabled = false;
			}
			this.renderCurrentReplay();
			this.renderReplayList();
		}
	}

	private loadReplay(replay: ReplayData, summary: ReplaySummary): void {
		this.stop();
		this.replay = replay;
		this.activeReplay = summary;
		this.totalTicks = replay.ticks.length;
		this.currentTick = 0;

		if (this.scrubBar) {
			this.scrubBar.max = String(this.totalTicks);
			this.scrubBar.value = '0';
		}
		if (this.playBtn) {
			this.playBtn.disabled = false;
		}

		this.renderCurrentReplay();
		this.seekToTick(0);
		this.options.onReplayLoaded?.(summary);
		this.renderReplayList();
	}

	private clearReplay(): void {
		this.stop();
		this.loadingReplayFileName = null;
		this.loadErrorMessage = null;
		this.replay = null;
		this.activeReplay = null;
		this.simulation = null;
		this.currentTick = 0;
		this.totalTicks = 0;
		if (this.scrubBar) {
			this.scrubBar.max = '0';
			this.scrubBar.value = '0';
		}
		if (this.playBtn) {
			this.playBtn.disabled = true;
		}
		if (this.tickLabel) {
			this.tickLabel.textContent = 'Arena live';
		}
		this.renderCurrentReplay();
		this.renderReplayList();
		this.options.onReplayCleared?.();
	}

	private renderCurrentReplay(): void {
		if (!this.currentReplayCard) {
			return;
		}
		this.currentReplayCard.innerHTML = '';
		this.currentReplayCard.classList.remove('loading');

		if (this.loadingReplayFileName) {
			this.currentReplayCard.classList.add('loading');
			const loadingWrap = document.createElement('div');
			loadingWrap.className = 'replay-loading';

			const spinner = document.createElement('div');
			spinner.className = 'replay-loading-spinner';

			const title = document.createElement('h3');
			title.textContent = 'Loading Replay';

			const meta = document.createElement('p');
			meta.textContent = this.loadingReplayFileName;

			loadingWrap.append(spinner, title, meta);
			this.currentReplayCard.appendChild(loadingWrap);
			return;
		}

		if (this.loadErrorMessage) {
			const errorState = document.createElement('div');
			errorState.className = 'empty-state';
			errorState.textContent = this.loadErrorMessage;
			this.currentReplayCard.appendChild(errorState);
			return;
		}

		if (!this.activeReplay) {
			const empty = document.createElement('div');
			empty.className = 'empty-state';
			empty.textContent = 'Choose a replay to take over the arena canvas.';
			this.currentReplayCard.appendChild(empty);
			return;
		}

		const title = document.createElement('h3');
		title.textContent = `Episode ${this.activeReplay.episode ?? '--'} • Level ${this.activeReplay.level ?? '--'}`;
		const meta = document.createElement('p');
		meta.textContent = `${this.activeReplay.outcome.replace(/_/g, ' ')} • ${formatBytes(this.activeReplay.sizeBytes)} • Seed ${this.activeReplay.seed ?? '--'}`;
		this.currentReplayCard.append(title, meta);
	}

	private seekToTick(targetTick: number): void {
		if (!this.replay || !this.renderer) {
			return;
		}

		const clampedTick = Math.max(0, Math.min(targetTick, this.totalTicks));
		const initialState = createInitialGameState(this.replay.levelConfig, this.replay.seed);
		const controllers = createReplayControllers(initialState, this.replay);
		this.simulation = new Simulation(initialState, controllers);
		this.renderer.initializeCanvas(initialState.arena.width, initialState.arena.height);

		this.previousState = null;
		for (let tick = 0; tick < clampedTick; tick += 1) {
			this.previousState = this.simulation.getStateSnapshot();
			const stepResult = this.simulation.step();
			this.renderer.setLastActions(stepResult.actions);
			this.renderer.consumeEvents(stepResult.events);
		}

		this.currentTick = clampedTick;
		if (this.scrubBar) {
			this.scrubBar.value = String(clampedTick);
		}
		if (this.tickLabel) {
			this.tickLabel.textContent = `${clampedTick} / ${this.totalTicks}`;
		}

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
		if (!this.simulation || !this.renderer || !this.replay) {
			return;
		}
		this.playing = true;
		if (this.playBtn) {
			this.playBtn.textContent = 'Pause';
		}
		this.lastFrameTime = performance.now();
		this.accumulatorMs = 0;
		this.animationFrameId = requestAnimationFrame((timestamp) => this.gameLoop(timestamp));
	}

	private stop(): void {
		this.playing = false;
		if (this.playBtn) {
			this.playBtn.textContent = 'Play';
		}
		if (this.animationFrameId !== null) {
			cancelAnimationFrame(this.animationFrameId);
			this.animationFrameId = null;
		}
	}

	private gameLoop(timestamp: number): void {
		if (!this.playing || !this.simulation || !this.renderer || !this.replay) {
			return;
		}

		const frameDelta = Math.min(timestamp - this.lastFrameTime, 250);
		this.lastFrameTime = timestamp;
		this.accumulatorMs += frameDelta * this.playbackSpeed;

		while (this.accumulatorMs >= this.tickMs && this.currentTick < this.totalTicks) {
			this.previousState = this.simulation.getStateSnapshot();
			const stepResult = this.simulation.step();
			this.renderer.setLastActions(stepResult.actions);
			this.renderer.consumeEvents(stepResult.events);
			this.currentTick += 1;
			this.accumulatorMs -= this.tickMs;
		}

		if (this.scrubBar) {
			this.scrubBar.value = String(this.currentTick);
		}
		if (this.tickLabel) {
			this.tickLabel.textContent = `${this.currentTick} / ${this.totalTicks}`;
		}

		const state = this.simulation.getState();
		this.renderer.updateVisuals(frameDelta / 1000);
		this.renderer.render(state, this.previousState, this.accumulatorMs / this.tickMs);

		if (this.currentTick >= this.totalTicks) {
			this.stop();
			return;
		}

		this.animationFrameId = requestAnimationFrame((nextTimestamp) => this.gameLoop(nextTimestamp));
	}
}
