import packageJson from '../../package.json';
import { AudioManager } from '../game/AudioManager';
import { NeuralNetController } from '../game/controllers/NeuralNetController';
import type { TankController } from '../game/core/types';
import { Level } from '../game/Level';
import { LEVEL_CONFIGS, type LevelConfig } from '../game/LevelConfig';
import { DashboardApiClient, type ModelStatus, type RunsResponse } from './dashboardApi';
import { LevelEditor } from './LevelEditor';
import { ReplayViewer } from './ReplayViewer';
import { TrainingDashboard } from './TrainingDashboard';

function levelSubtitle(levelNumber: number): string {
	const config = LEVEL_CONFIGS[levelNumber - 1];
	const tags = [
		`${config.enemies?.length ?? 0} enemies`,
		`${config.obstacles.length} obstacles`,
		config.rules?.projectileBounces === false ? 'no bounces' : 'bounces on',
	];
	return tags.join(' | ');
}

export class LevelSelector {
	public static createHeadlessLevel(
		levelNumber: number,
		seed: number = levelNumber,
		playerController?: TankController
	): Level {
		const configIndex = Math.max(0, Math.min(levelNumber - 1, LEVEL_CONFIGS.length - 1));
		return new Level(LEVEL_CONFIGS[configIndex], { headless: true, seed, playerController });
	}

	private readonly numLevels: number;
	private readonly audioManager: AudioManager;
	private readonly api = new DashboardApiClient();
	private activeLevelNumber = 1;
	private activeLevel: Level | null = null;
	private aiMode = false;
	private aiController: NeuralNetController | null = null;
	private aiModelUrl: string | null = null;
	private selectedRunId: string | null = null;
	private latestRunsResponse: RunsResponse | null = null;
	private modelStatus: ModelStatus | null = null;
	private replayViewer: ReplayViewer | null = null;
	private trainingDashboard: TrainingDashboard | null = null;
	private levelEditor: LevelEditor | null = null;
	private editorMode = false;
	private editorPlaying = false;

	private aiToggleBtn: HTMLButtonElement | null = null;
	private editorToggleBtn: HTMLButtonElement | null = null;
	private modelBadge: HTMLDivElement | null = null;
	private stageModeBadge: HTMLDivElement | null = null;
	private stageTitle: HTMLHeadingElement | null = null;
	private stageSubtitle: HTMLParagraphElement | null = null;
	private heroMeta: HTMLDivElement | null = null;
	private levelGrid: HTMLDivElement | null = null;
	private canvas: HTMLCanvasElement | null = null;
	private canvasFrame: HTMLDivElement | null = null;

	public constructor(levels: number) {
		this.numLevels = levels;
		this.audioManager = new AudioManager();
		void this.audioManager.loadAllAudio().then(() => {
			this.audioManager.playBackgroundMusic();
		});

		this.buildShell();
		this.mountPanels();
		this.renderLevelGrid();
		void this.refreshModelStatus();
		this.startActiveLevel();
	}

	private buildShell(): void {
		const appRoot = document.getElementById('app');
		if (!appRoot) {
			throw new Error('App root #app not found.');
		}
		appRoot.innerHTML = '';

		const shell = document.createElement('div');
		shell.className = 'app-shell';

		const hero = document.createElement('header');
		hero.className = 'hero-shell';

		const heroCopy = document.createElement('div');
		heroCopy.className = 'hero-copy';
		const eyebrow = document.createElement('span');
		eyebrow.className = 'panel-eyebrow';
		eyebrow.textContent = 'Local Training Command Center';
		const title = document.createElement('h1');
		title.textContent = `Treads v${packageJson.version}`;
		const subtitle = document.createElement('p');
		subtitle.textContent =
			'Watch training runs evolve, browse saved replay JSON files in the browser, and load the current policy into the arena without rebuilding.';
		heroCopy.append(eyebrow, title, subtitle);

		const heroStatus = document.createElement('div');
		heroStatus.className = 'hero-status';
		this.heroMeta = document.createElement('div');
		this.heroMeta.className = 'hero-metrics';
		this.heroMeta.textContent = 'Scanning runs and model artifacts';
		heroStatus.appendChild(this.heroMeta);

		hero.append(heroCopy, heroStatus);

		const workspace = document.createElement('main');
		workspace.className = 'workspace';

		const stageColumn = document.createElement('section');
		stageColumn.className = 'stage-column';

		const stageCard = document.createElement('div');
		stageCard.className = 'stage-card';
		const stageHeader = document.createElement('div');
		stageHeader.className = 'stage-header';
		const stageCopy = document.createElement('div');
		stageCopy.className = 'panel-header-copy';
		const stageEyebrow = document.createElement('span');
		stageEyebrow.className = 'panel-eyebrow';
		stageEyebrow.textContent = 'Arena';
		this.stageTitle = document.createElement('h2');
		this.stageTitle.textContent = 'Level 1';
		this.stageSubtitle = document.createElement('p');
		this.stageSubtitle.textContent = levelSubtitle(1);
		stageCopy.append(stageEyebrow, this.stageTitle, this.stageSubtitle);

		const stageActions = document.createElement('div');
		stageActions.className = 'stage-actions';
		this.stageModeBadge = document.createElement('div');
		this.stageModeBadge.className = 'status-pill idle';
		this.stageModeBadge.textContent = 'Live arena';

		this.modelBadge = document.createElement('div');
		this.modelBadge.className = 'model-badge';
		this.modelBadge.textContent = 'Model unavailable';

		this.aiToggleBtn = document.createElement('button');
		this.aiToggleBtn.type = 'button';
		this.aiToggleBtn.className = 'ai-toggle';
		this.aiToggleBtn.textContent = 'AI Pilot Off';
		this.aiToggleBtn.addEventListener('click', () => {
			void this.toggleAiMode();
		});

		this.editorToggleBtn = document.createElement('button');
		this.editorToggleBtn.type = 'button';
		this.editorToggleBtn.className = 'control-button';
		this.editorToggleBtn.textContent = 'Level Editor';
		this.editorToggleBtn.addEventListener('click', () => {
			this.setEditorMode(!this.editorMode);
		});

		stageActions.append(this.stageModeBadge, this.modelBadge, this.aiToggleBtn, this.editorToggleBtn);
		stageHeader.append(stageCopy, stageActions);

		this.canvasFrame = document.createElement('div');
		this.canvasFrame.className = 'canvas-frame';
		this.canvas = document.createElement('canvas');
		this.canvas.id = 'game-canvas';
		this.canvas.className = 'game-canvas';
		this.canvasFrame.appendChild(this.canvas);

		stageCard.append(stageHeader, this.canvasFrame);

		const levelPanel = document.createElement('div');
		levelPanel.className = 'panel-card level-panel';
		const levelHeader = document.createElement('div');
		levelHeader.className = 'panel-header';
		const levelCopy = document.createElement('div');
		levelCopy.className = 'panel-header-copy';
		const levelEyebrow = document.createElement('span');
		levelEyebrow.className = 'panel-eyebrow';
		levelEyebrow.textContent = 'Scenario Deck';
		const levelTitle = document.createElement('h2');
		levelTitle.textContent = 'Challenge Ladder';
		levelCopy.append(levelEyebrow, levelTitle);
		levelHeader.appendChild(levelCopy);
		this.levelGrid = document.createElement('div');
		this.levelGrid.className = 'level-grid';
		levelPanel.append(levelHeader, this.levelGrid);

		stageColumn.append(stageCard, levelPanel);

		const inspectorColumn = document.createElement('aside');
		inspectorColumn.className = 'inspector-column';
		const trainingPanel = document.createElement('section');
		trainingPanel.className = 'panel-card inspector-panel';
		const replayPanel = document.createElement('section');
		replayPanel.className = 'panel-card inspector-panel';
		inspectorColumn.append(trainingPanel, replayPanel);

		workspace.append(stageColumn, inspectorColumn);
		shell.append(hero, workspace);
		appRoot.appendChild(shell);

		this.trainingDashboard = new TrainingDashboard(this.api, {
			onRunSelected: (runId) => {
				void this.handleRunSelection(runId);
			},
			onRunsUpdated: (response) => {
				this.handleRunsUpdated(response);
			},
		});
		this.trainingDashboard.mount(trainingPanel);

		if (!this.canvas) {
			throw new Error('Game canvas was not created.');
		}
		if (!this.canvasFrame) {
			throw new Error('Canvas frame was not created.');
		}

		this.levelEditor = new LevelEditor({
			canvas: this.canvas,
			onPlaytest: (config) => {
				this.startEditorPlaytest(config);
			},
			onStopPlaytest: () => {
				this.stopEditorPlaytest();
			},
		});
		this.levelEditor.mount(this.canvasFrame);
		this.levelEditor.setVisible(false);

		this.replayViewer = new ReplayViewer(this.api, {
			onReplayLoaded: (summary) => {
				if (this.editorMode) {
					this.setEditorMode(false);
				}
				this.activeLevel?.stop();
				if (this.stageModeBadge) {
					this.stageModeBadge.className = 'status-pill running';
					this.stageModeBadge.textContent = 'Replay';
				}
				if (this.stageTitle) {
					this.stageTitle.textContent = `Replay L${summary.level ?? '--'} | E${summary.episode ?? '--'}`;
				}
				if (this.stageSubtitle) {
					this.stageSubtitle.textContent = summary.fileName;
				}
			},
			onReplayCleared: () => {
				if (this.stageModeBadge) {
					this.stageModeBadge.className = 'status-pill idle';
					this.stageModeBadge.textContent = 'Live arena';
				}
				this.startActiveLevel();
			},
		});
		this.replayViewer.mount(replayPanel, this.canvas);
		if (this.selectedRunId) {
			void this.replayViewer.setRun(this.selectedRunId);
		}
	}

	private mountPanels(): void {
		this.updateStageLabels();
		this.updateAiControls();
	}

	private renderLevelGrid(): void {
		if (!this.levelGrid) {
			return;
		}
		this.levelGrid.innerHTML = '';
		for (let level = 1; level <= this.numLevels; level += 1) {
			const config = LEVEL_CONFIGS[level - 1];
			const card = document.createElement('button');
			card.type = 'button';
			card.className = `level-card${level === this.activeLevelNumber ? ' selected' : ''}`;
			card.addEventListener('click', () => {
				if (this.editorMode) {
					this.setEditorMode(false);
				}
				this.activeLevelNumber = level;
				this.updateStageLabels();
				this.renderLevelGrid();
				if (this.replayViewer?.hasActiveReplay()) {
					this.replayViewer.exitReplay();
				} else {
					this.startActiveLevel();
				}
			});

			const title = document.createElement('strong');
			title.textContent = `Level ${level}`;
			const meta = document.createElement('span');
			meta.textContent = `${config.enemies?.length ?? 0} enemy | ${config.obstacles.length} obstacle`;
			const rules = document.createElement('small');
			rules.textContent = config.rules?.projectileBounces === false ? 'No bounce ruleset' : 'Standard bounce ruleset';
			card.append(title, meta, rules);
			this.levelGrid.appendChild(card);
		}
	}

	private async handleRunSelection(runId: string): Promise<void> {
		this.selectedRunId = runId;
		await this.replayViewer?.setRun(runId);
		await this.refreshModelStatus();
		if (this.aiMode && !this.replayViewer?.hasActiveReplay()) {
			await this.reloadAiController();
			this.startActiveLevel();
		}
		this.updateHeroMeta();
	}

	private handleRunsUpdated(response: RunsResponse): void {
		this.latestRunsResponse = response;
		if (!this.selectedRunId) {
			this.selectedRunId = response.activeRunId ?? response.runs[0]?.id ?? null;
		}
		if (this.selectedRunId) {
			void this.replayViewer?.setRun(this.selectedRunId);
		}
		this.updateHeroMeta();
	}

	private updateHeroMeta(): void {
		if (!this.heroMeta) {
			return;
		}
		const selectedRun =
			this.latestRunsResponse?.runs.find((run) => run.id === this.selectedRunId) ??
			this.latestRunsResponse?.runs[0] ??
			null;
		if (!selectedRun || !selectedRun.latestMetrics) {
			this.heroMeta.textContent = this.modelStatus?.available
				? `Model ready from ${this.modelStatus.runId ?? 'current output'}`
				: 'Waiting for run metrics and a browser-ready ONNX export';
			return;
		}
		const metrics = selectedRun.latestMetrics;
		const rewardLabel =
			metrics.avgReward50 !== null && metrics.avgReward50 !== undefined ? metrics.avgReward50.toFixed(2) : '--';
		const winRateLabel =
			metrics.avgWinrate50 !== null && metrics.avgWinrate50 !== undefined
				? `${(metrics.avgWinrate50 * 100).toFixed(1)}%`
				: '--';
		const speedLabel =
			metrics.stepsPerSec !== null && metrics.stepsPerSec !== undefined ? metrics.stepsPerSec.toFixed(0) : '--';
		this.heroMeta.textContent =
			`${selectedRun.label} | Reward ${rewardLabel} | Win ${winRateLabel} | ` +
			`${speedLabel} steps/s | ${selectedRun.replayCount} replays`;
	}

	private buildRequestedModelUrl(): string | null {
		if (!this.modelStatus?.available) {
			return null;
		}
		const params = new URLSearchParams();
		if (this.selectedRunId) {
			params.set('run', this.selectedRunId);
		}
		if (this.modelStatus.updatedAtMs !== null && this.modelStatus.updatedAtMs !== undefined) {
			params.set('v', String(Math.round(this.modelStatus.updatedAtMs)));
		}
		return `/api/model/current.onnx?${params.toString()}`;
	}

	private async refreshModelStatus(): Promise<void> {
		try {
			this.modelStatus = await this.api.fetchModel(this.selectedRunId ?? undefined);
		} catch (error) {
			console.error('Failed to fetch model status:', error);
			this.modelStatus = null;
		}
		this.updateAiControls();
		this.updateHeroMeta();
	}

	private updateAiControls(loading = false): void {
		if (this.aiToggleBtn) {
			this.aiToggleBtn.disabled = loading;
			this.aiToggleBtn.textContent = loading ? 'Loading AI...' : this.aiMode ? 'AI Pilot On' : 'AI Pilot Off';
			this.aiToggleBtn.classList.toggle('active', this.aiMode);
		}
		if (this.modelBadge) {
			this.modelBadge.textContent = this.modelStatus?.available
				? `Model ${this.modelStatus.runId === '__root__' ? 'current output' : (this.modelStatus.runId ?? 'ready')}`
				: 'Model unavailable';
		}
		if (this.editorToggleBtn) {
			this.editorToggleBtn.classList.toggle('active', this.editorMode);
			this.editorToggleBtn.textContent = this.editorMode ? 'Exit Editor' : 'Level Editor';
		}
	}

	private async reloadAiController(): Promise<void> {
		const modelUrl = this.buildRequestedModelUrl();
		if (!modelUrl) {
			throw new Error('No ONNX model is available for the selected run.');
		}
		if (this.aiController && this.aiModelUrl === modelUrl) {
			return;
		}
		this.aiController = new NeuralNetController(modelUrl);
		await this.aiController.loadModel();
		this.aiModelUrl = modelUrl;
	}

	private async toggleAiMode(): Promise<void> {
		if (!this.aiMode) {
			this.updateAiControls(true);
			await this.refreshModelStatus();
			try {
				await this.reloadAiController();
				this.aiMode = true;
			} catch (error) {
				console.error('Failed to enable AI mode:', error);
				this.aiMode = false;
			}
		} else {
			this.aiMode = false;
		}
		this.updateAiControls();
		if (!this.replayViewer?.hasActiveReplay()) {
			this.startActiveLevel();
		}
	}

	private updateStageLabels(): void {
		if (this.editorMode) {
			if (this.stageTitle) {
				this.stageTitle.textContent = this.editorPlaying ? 'Custom Level Playtest' : 'Custom Level Editor';
			}
			if (this.stageSubtitle) {
				this.stageSubtitle.textContent = this.editorPlaying
					? 'Testing your custom map in the live arena'
					: 'Use tools below to place units, draw obstacles, and tune rules';
			}
			return;
		}
		if (this.stageTitle) {
			this.stageTitle.textContent = `Level ${this.activeLevelNumber}`;
		}
		if (this.stageSubtitle) {
			this.stageSubtitle.textContent = levelSubtitle(this.activeLevelNumber);
		}
	}

	private startActiveLevel(): void {
		if (this.editorMode) {
			return;
		}
		if (this.replayViewer?.hasActiveReplay()) {
			return;
		}

		this.activeLevel?.stop();
		const configIndex = Math.max(0, Math.min(this.activeLevelNumber - 1, LEVEL_CONFIGS.length - 1));
		const playerController = this.aiMode ? (this.aiController ?? undefined) : undefined;
		this.activeLevel = new Level(LEVEL_CONFIGS[configIndex], {
			audioManager: this.audioManager,
			seed: this.activeLevelNumber,
			playerController,
		});
		this.activeLevel.start();
		this.updateStageLabels();
		if (this.stageModeBadge) {
			this.stageModeBadge.className = 'status-pill idle';
			this.stageModeBadge.textContent = this.aiMode ? 'AI live' : 'Live arena';
		}
	}

	private setEditorMode(enabled: boolean): void {
		if (!this.levelEditor) {
			return;
		}
		if (enabled === this.editorMode) {
			return;
		}

		if (enabled) {
			if (this.replayViewer?.hasActiveReplay()) {
				this.replayViewer.exitReplay();
			}
			this.activeLevel?.stop();
			this.activeLevel = null;
			this.editorMode = true;
			this.editorPlaying = false;
			this.levelEditor.setVisible(true);
			this.levelEditor.setPlaytesting(false);
			if (this.stageModeBadge) {
				this.stageModeBadge.className = 'status-pill running';
				this.stageModeBadge.textContent = 'Editor';
			}
			this.updateStageLabels();
			this.updateAiControls();
			return;
		}

		this.stopEditorPlaytest();
		this.editorMode = false;
		this.editorPlaying = false;
		this.levelEditor.setVisible(false);
		this.updateAiControls();
		this.startActiveLevel();
	}

	private startEditorPlaytest(config: LevelConfig): void {
		if (!this.editorMode) {
			return;
		}
		this.activeLevel?.stop();
		const playerController = this.aiMode ? (this.aiController ?? undefined) : undefined;
		this.activeLevel = new Level(config, {
			audioManager: this.audioManager,
			seed: 9991,
			playerController,
		});
		this.activeLevel.start();
		this.editorPlaying = true;
		this.levelEditor?.setPlaytesting(true);
		if (this.stageModeBadge) {
			this.stageModeBadge.className = 'status-pill running';
			this.stageModeBadge.textContent = this.aiMode ? 'Editor AI playtest' : 'Editor playtest';
		}
		this.updateStageLabels();
	}

	private stopEditorPlaytest(): void {
		if (!this.editorMode) {
			return;
		}
		if (this.activeLevel) {
			this.activeLevel.stop();
			this.activeLevel = null;
		}
		this.editorPlaying = false;
		this.levelEditor?.setPlaytesting(false);
		if (this.stageModeBadge) {
			this.stageModeBadge.className = 'status-pill running';
			this.stageModeBadge.textContent = 'Editor';
		}
		this.updateStageLabels();
	}
}
