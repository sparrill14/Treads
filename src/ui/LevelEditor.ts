import {
	getLevelTankConfigs,
	type MatchRulesConfig,
	type TankConfig,
	type TankKind,
	type ControlType,
	type LevelConfig,
	type ObstacleConfig,
} from '../game/LevelConfig';

type EditorTool = 'move' | 'tank' | 'obstacle' | 'erase';

type DragTarget =
	| { kind: 'tank'; index: number; offsetX: number; offsetY: number }
	| { kind: 'obstacle'; index: number; offsetX: number; offsetY: number };

interface LevelEditorOptions {
	canvas: HTMLCanvasElement;
	onPlaytest: (config: LevelConfig) => void;
	onStopPlaytest: () => void;
}

const ARENA_WIDTH = 1000;
const ARENA_HEIGHT = 500;
const GRID_SIZE = 25;
const TANK_SIZE = 30;
const DEFAULT_OBSTACLE_WIDTH = 80;
const DEFAULT_OBSTACLE_HEIGHT = 60;
const STORAGE_KEY = 'treads.custom-level.v2';

const DEFAULT_RULES: MatchRulesConfig = {
	tankHitPoints: 3,
	projectileDamage: 1,
	bombDamage: 2,
	invulnerabilityTicks: 8,
	projectileBounces: true,
	turretSpeedMultiplier: 1,
};

const DEFAULT_LEVEL_CONFIG: LevelConfig = {
	obstacles: [],
	tanks: [
		{ id: 'tank-0', team: 'alpha', kind: 'player', control: 'human', x: 120, y: 250 },
		{ id: 'tank-1', team: 'beta', kind: 'stationary', control: 'scripted', x: 860, y: 250 },
	],
	rules: { ...DEFAULT_RULES },
};

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function clampTankX(x: number): number {
	return clamp(Math.round(x), 0, ARENA_WIDTH - TANK_SIZE);
}

function clampTankY(y: number): number {
	return clamp(Math.round(y), 0, ARENA_HEIGHT - TANK_SIZE);
}

function sanitizeObstacle(obstacle: ObstacleConfig): ObstacleConfig {
	const width = clamp(Math.round(obstacle.width), 8, ARENA_WIDTH);
	const height = clamp(Math.round(obstacle.height), 8, ARENA_HEIGHT);
	const x = clamp(Math.round(obstacle.x), 0, ARENA_WIDTH - width);
	const y = clamp(Math.round(obstacle.y), 0, ARENA_HEIGHT - height);
	return { x, y, width, height };
}

function normalizeRules(rules: MatchRulesConfig | undefined): MatchRulesConfig {
	return {
		tankHitPoints: clamp(Math.round(rules?.tankHitPoints ?? 3), 1, 20),
		projectileDamage: clamp(Math.round(rules?.projectileDamage ?? 1), 1, 20),
		bombDamage: clamp(Math.round(rules?.bombDamage ?? 2), 1, 20),
		invulnerabilityTicks: clamp(Math.round(rules?.invulnerabilityTicks ?? 8), 0, 120),
		projectileBounces: rules?.projectileBounces ?? true,
		turretSpeedMultiplier: clamp(rules?.turretSpeedMultiplier ?? 1, 0.1, 3),
	};
}

function deepClone(config: LevelConfig): LevelConfig {
	return {
		obstacles: config.obstacles.map((obstacle) => ({ ...obstacle })),
		tanks: getLevelTankConfigs(config).map((tank) => ({
			...tank,
			ammo: tank.ammo ? { ...tank.ammo } : undefined,
			bombs: tank.bombs ? { ...tank.bombs } : undefined,
			navigator: tank.navigator ? { ...tank.navigator } : undefined,
		})),
		rules: config.rules ? { ...config.rules } : undefined,
	};
}

function isPointInRect(x: number, y: number, rect: ObstacleConfig): boolean {
	return x >= rect.x && x <= rect.x + rect.width && y >= rect.y && y <= rect.y + rect.height;
}

function teamColor(team: string): string {
	const value = [...team].reduce((sum, char) => sum + char.charCodeAt(0), 0);
	const hue = value % 360;
	return `hsl(${hue} 68% 58%)`;
}

function defaultTankForKind(kind: TankKind): Pick<TankConfig, 'ammo' | 'bombs' | 'navigator'> {
	if (kind === 'bomber') {
		return { bombs: { type: 'basic', count: 2 }, navigator: { type: 'astar' } };
	}
	if (kind === 'super-bomber') {
		return {
			ammo: { type: 'super', count: 2 },
			bombs: { type: 'love', count: 2 },
			navigator: { type: 'astar-avoidance', aggressionFactor: 8 },
		};
	}
	if (kind === 'simple-moving') {
		return { navigator: { type: 'simple' } };
	}
	if (kind === 'stationary-random-aim') {
		return { ammo: { type: 'super', count: 1 } };
	}
	return {};
}

function makeTank(kind: TankKind, team: string, control: ControlType, x: number, y: number, index: number): TankConfig {
	return {
		id: `tank-${index}`,
		kind,
		team,
		control,
		x: clampTankX(x),
		y: clampTankY(y),
		...defaultTankForKind(kind),
	};
}

export class LevelEditor {
	private readonly canvas: HTMLCanvasElement;
	private readonly context: CanvasRenderingContext2D;
	private readonly onPlaytest: (config: LevelConfig) => void;
	private readonly onStopPlaytest: () => void;

	private host: HTMLElement | null = null;
	private overlay: HTMLDivElement | null = null;
	private toolSelect: HTMLSelectElement | null = null;
	private kindSelect: HTMLSelectElement | null = null;
	private controlSelect: HTMLSelectElement | null = null;
	private teamInput: HTMLInputElement | null = null;
	private obstacleWidthInput: HTMLInputElement | null = null;
	private obstacleHeightInput: HTMLInputElement | null = null;
	private jsonArea: HTMLTextAreaElement | null = null;
	private status: HTMLParagraphElement | null = null;
	private tanksList: HTMLDivElement | null = null;
	private obstaclesList: HTMLDivElement | null = null;
	private playtestButton: HTMLButtonElement | null = null;
	private stopPlaytestButton: HTMLButtonElement | null = null;

	private config: LevelConfig = deepClone(DEFAULT_LEVEL_CONFIG);
	private visible = false;
	private playtesting = false;
	private drag: DragTarget | null = null;
	private draftObstacleStart: { x: number; y: number } | null = null;
	private draftObstacleCurrent: { x: number; y: number } | null = null;
	private selectedTankIndex: number | null = null;
	private activeTool: EditorTool = 'move';

	public constructor(options: LevelEditorOptions) {
		this.canvas = options.canvas;
		const context = this.canvas.getContext('2d');
		if (!context) {
			throw new Error('Could not create 2D context for level editor.');
		}
		this.context = context;
		this.onPlaytest = options.onPlaytest;
		this.onStopPlaytest = options.onStopPlaytest;
		this.restoreFromStorage();
		this.canvas.width = ARENA_WIDTH;
		this.canvas.height = ARENA_HEIGHT;
		this.bindCanvasEvents();
	}

	public mount(host: HTMLElement): void {
		this.host = host;
		this.overlay = document.createElement('div');
		this.overlay.className = 'editor-overlay';
		host.appendChild(this.overlay);

		const topBar = document.createElement('div');
		topBar.className = 'editor-topbar';

		this.toolSelect = document.createElement('select');
		this.toolSelect.className = 'editor-select';
		for (const tool of ['move', 'tank', 'obstacle', 'erase'] as EditorTool[]) {
			const option = document.createElement('option');
			option.value = tool;
			option.textContent = `Tool: ${tool}`;
			this.toolSelect.appendChild(option);
		}
		this.toolSelect.value = 'move';

		this.kindSelect = document.createElement('select');
		this.kindSelect.className = 'editor-select';
		for (const kind of [
			'player',
			'stationary',
			'stationary-random-aim',
			'simple-moving',
			'bomber',
			'super-bomber',
		] as TankKind[]) {
			const option = document.createElement('option');
			option.value = kind;
			option.textContent = `Tank: ${kind}`;
			this.kindSelect.appendChild(option);
		}

		this.controlSelect = document.createElement('select');
		this.controlSelect.className = 'editor-select';
		for (const control of ['human', 'scripted'] as ControlType[]) {
			const option = document.createElement('option');
			option.value = control;
			option.textContent = `Control: ${control}`;
			this.controlSelect.appendChild(option);
		}
		this.controlSelect.value = 'human';

		const toolPalette = document.createElement('div');
		toolPalette.className = 'editor-tool-palette';
		for (const tool of ['move', 'tank', 'obstacle', 'erase'] as EditorTool[]) {
			const button = document.createElement('button');
			button.type = 'button';
			button.className = 'editor-palette-button';
			button.dataset.editorTool = tool;
			button.textContent = tool === 'tank' ? 'Place Tank' : tool === 'obstacle' ? 'Draw Obstacle' : tool;
			button.addEventListener('click', () => {
				this.activeTool = tool;
				if (this.toolSelect) {
					this.toolSelect.value = tool;
				}
				this.syncPaletteButtons();
			});
			toolPalette.appendChild(button);
		}

		const kindPalette = document.createElement('div');
		kindPalette.className = 'editor-kind-palette';
		for (const kind of ['player', 'stationary', 'simple-moving', 'bomber', 'super-bomber'] as TankKind[]) {
			const button = document.createElement('button');
			button.type = 'button';
			button.className = 'editor-palette-button';
			button.dataset.editorKind = kind;
			button.textContent = `Tank: ${kind}`;
			button.addEventListener('click', () => {
				if (this.kindSelect) {
					this.kindSelect.value = kind;
				}
				this.activeTool = 'tank';
				if (this.toolSelect) {
					this.toolSelect.value = 'tank';
				}
				this.syncPaletteButtons();
			});
			kindPalette.appendChild(button);
		}

		this.teamInput = document.createElement('input');
		this.teamInput.className = 'editor-input';
		this.teamInput.value = 'alpha';
		this.teamInput.placeholder = 'team';

		this.obstacleWidthInput = document.createElement('input');
		this.obstacleWidthInput.type = 'number';
		this.obstacleWidthInput.className = 'editor-input compact';
		this.obstacleWidthInput.value = String(DEFAULT_OBSTACLE_WIDTH);
		this.obstacleWidthInput.title = 'Obstacle width';

		this.obstacleHeightInput = document.createElement('input');
		this.obstacleHeightInput.type = 'number';
		this.obstacleHeightInput.className = 'editor-input compact';
		this.obstacleHeightInput.value = String(DEFAULT_OBSTACLE_HEIGHT);
		this.obstacleHeightInput.title = 'Obstacle height';

		const newButton = document.createElement('button');
		newButton.type = 'button';
		newButton.className = 'control-button';
		newButton.textContent = 'New';
		newButton.addEventListener('click', () => {
			this.config = deepClone(DEFAULT_LEVEL_CONFIG);
			this.selectedTankIndex = null;
			this.persistToStorage();
			this.refreshOverlay();
			this.draw();
		});

		this.playtestButton = document.createElement('button');
		this.playtestButton.type = 'button';
		this.playtestButton.className = 'control-button';
		this.playtestButton.textContent = 'Playtest';
		this.playtestButton.addEventListener('click', () => {
			if (this.playtesting) return;
			this.playtesting = true;
			this.onPlaytest(this.getConfig());
			this.refreshPlaytestButtons();
		});

		this.stopPlaytestButton = document.createElement('button');
		this.stopPlaytestButton.type = 'button';
		this.stopPlaytestButton.className = 'control-button';
		this.stopPlaytestButton.textContent = 'Stop';
		this.stopPlaytestButton.addEventListener('click', () => {
			if (!this.playtesting) return;
			this.playtesting = false;
			this.onStopPlaytest();
			this.refreshPlaytestButtons();
			this.draw();
		});

		topBar.append(
			toolPalette,
			kindPalette,
			this.controlSelect,
			this.teamInput,
			this.obstacleWidthInput,
			this.obstacleHeightInput,
			newButton,
			this.playtestButton,
			this.stopPlaytestButton
		);

		const palette = document.createElement('div');
		palette.className = 'editor-palette';
		for (const kind of ['player', 'stationary', 'simple-moving', 'bomber', 'super-bomber'] as TankKind[]) {
			const chip = document.createElement('button');
			chip.type = 'button';
			chip.className = 'editor-chip';
			chip.draggable = true;
			chip.textContent = `Drop ${kind}`;
			chip.addEventListener('click', () => {
				if (this.kindSelect) this.kindSelect.value = kind;
				if (this.toolSelect) this.toolSelect.value = 'tank';
			});
			chip.addEventListener('dragstart', (event) => {
				event.dataTransfer?.setData('text/tank-kind', kind);
			});
			palette.appendChild(chip);
		}

		this.status = document.createElement('p');
		this.status.className = 'editor-status';
		this.status.textContent =
			'Drag a tank chip onto the arena, click-drag to move items, and use teams/controls for AI vs AI or human vs AI.';

		const details = document.createElement('details');
		details.className = 'editor-details';
		details.open = true;
		const summary = document.createElement('summary');
		summary.textContent = 'Advanced settings';
		details.appendChild(summary);

		const inspector = document.createElement('div');
		inspector.className = 'editor-inspector';
		const tanksSection = document.createElement('section');
		tanksSection.className = 'editor-section';
		const tanksTitle = document.createElement('h3');
		tanksTitle.textContent = 'Tanks';
		this.tanksList = document.createElement('div');
		this.tanksList.className = 'editor-list';
		tanksSection.append(tanksTitle, this.tanksList);

		const obstaclesSection = document.createElement('section');
		obstaclesSection.className = 'editor-section';
		const obstaclesTitle = document.createElement('h3');
		obstaclesTitle.textContent = 'Obstacles';
		this.obstaclesList = document.createElement('div');
		this.obstaclesList.className = 'editor-list';
		obstaclesSection.append(obstaclesTitle, this.obstaclesList);

		const rulesSection = document.createElement('section');
		rulesSection.className = 'editor-section';
		const rulesTitle = document.createElement('h3');
		rulesTitle.textContent = 'Rules';
		const rulesGrid = document.createElement('div');
		rulesGrid.className = 'editor-rules-grid';
		rulesSection.append(rulesTitle, rulesGrid);
		this.buildRulesInputs(rulesGrid);

		inspector.append(tanksSection, obstaclesSection, rulesSection);
		details.appendChild(inspector);

		const jsonSection = document.createElement('section');
		jsonSection.className = 'editor-section';
		const jsonTitle = document.createElement('h3');
		jsonTitle.textContent = 'JSON';
		const jsonButtons = document.createElement('div');
		jsonButtons.className = 'editor-field-row';
		const applyJsonBtn = document.createElement('button');
		applyJsonBtn.type = 'button';
		applyJsonBtn.className = 'control-button';
		applyJsonBtn.textContent = 'Apply JSON';
		applyJsonBtn.addEventListener('click', () => this.applyJson());
		const copyJsonBtn = document.createElement('button');
		copyJsonBtn.type = 'button';
		copyJsonBtn.className = 'control-button';
		copyJsonBtn.textContent = 'Copy JSON';
		copyJsonBtn.addEventListener('click', async () => {
			if (!this.jsonArea) return;
			try {
				await navigator.clipboard.writeText(this.jsonArea.value);
				this.setStatus('JSON copied to clipboard.');
			} catch {
				this.setStatus('Clipboard unavailable. Copy from the JSON box.');
			}
		});
		jsonButtons.append(applyJsonBtn, copyJsonBtn);
		this.jsonArea = document.createElement('textarea');
		this.jsonArea.className = 'editor-json';
		jsonSection.append(jsonTitle, jsonButtons, this.jsonArea);

		this.overlay.append(topBar, palette, this.status, details, jsonSection);
		this.syncPaletteButtons();
		this.refreshOverlay();
		this.refreshPlaytestButtons();
		this.setVisible(false);
	}

	private syncPaletteButtons(): void {
		if (!this.overlay) {
			return;
		}
		this.overlay.querySelectorAll<HTMLButtonElement>('.editor-palette-button[data-editor-tool]').forEach((button) => {
			button.classList.toggle('active', button.dataset.editorTool === this.activeTool);
		});
		const kind = this.kindSelect?.value ?? 'player';
		this.overlay.querySelectorAll<HTMLButtonElement>('.editor-palette-button[data-editor-kind]').forEach((button) => {
			button.classList.toggle('active', button.dataset.editorKind === kind);
		});
	}

	public setVisible(visible: boolean): void {
		this.visible = visible;
		if (this.overlay) {
			this.overlay.style.display = visible ? 'grid' : 'none';
		}
		this.canvas.classList.toggle('editor-active', visible && !this.playtesting);
		if (visible && !this.playtesting) this.draw();
	}

	public setPlaytesting(playtesting: boolean): void {
		this.playtesting = playtesting;
		this.canvas.classList.toggle('editor-active', this.visible && !this.playtesting);
		this.refreshPlaytestButtons();
		if (!playtesting) this.draw();
	}

	public setConfig(config: LevelConfig): void {
		this.config = this.sanitizeConfig(config);
		this.persistToStorage();
		this.refreshOverlay();
		if (!this.playtesting) this.draw();
	}

	public getConfig(): LevelConfig {
		return deepClone(this.config);
	}

	private sanitizeConfig(input: LevelConfig): LevelConfig {
		const obstacles = (input.obstacles ?? []).map(sanitizeObstacle);
		const tanks = getLevelTankConfigs(input).map((tank, index) => ({
			id: tank.id ?? `tank-${index}`,
			team: tank.team?.trim() || 'alpha',
			kind: tank.kind,
			control: tank.control ?? (tank.kind === 'player' ? 'human' : 'scripted'),
			x: clampTankX(tank.x),
			y: clampTankY(tank.y),
			color: tank.color,
			ammo: tank.ammo ? { ...tank.ammo } : undefined,
			bombs: tank.bombs ? { ...tank.bombs } : undefined,
			navigator: tank.navigator ? { ...tank.navigator } : undefined,
		}));
		return {
			obstacles,
			tanks,
			rules: normalizeRules(input.rules),
		};
	}

	private bindCanvasEvents(): void {
		this.canvas.addEventListener('mousedown', (event) => {
			if (!this.visible || this.playtesting) return;
			const point = this.getCanvasPoint(event);
			const tool = (this.toolSelect?.value ?? this.activeTool) as EditorTool;
			this.activeTool = tool;
			const dragCandidate = this.findDragTarget(point.x, point.y);
			if (dragCandidate && tool !== 'erase') {
				this.drag = dragCandidate;
				return;
			}
			if (tool === 'obstacle') {
				this.draftObstacleStart = point;
				this.draftObstacleCurrent = point;
				event.preventDefault();
				this.draw();
			}
		});

		this.canvas.addEventListener('mousemove', (event) => {
			if (!this.visible || this.playtesting) return;
			const point = this.getCanvasPoint(event);
			if (this.drag) {
				this.applyDrag(point.x, point.y);
				this.draw();
				return;
			}
			if (this.draftObstacleStart) {
				this.draftObstacleCurrent = point;
				this.draw();
			}
		});

		this.canvas.addEventListener('mouseup', (event) => {
			if (!this.visible || this.playtesting) return;
			const point = this.getCanvasPoint(event);
			if (this.drag) {
				this.drag = null;
				this.persistToStorage();
				this.refreshOverlay();
				return;
			}
			const tool = (this.toolSelect?.value ?? this.activeTool) as EditorTool;
			this.activeTool = tool;
			if (tool === 'tank') {
				this.addTankAt(point.x, point.y);
			} else if (tool === 'obstacle') {
				this.commitDraftObstacle(point.x, point.y);
			} else if (tool === 'erase') {
				this.eraseAt(point.x, point.y);
			}
			this.persistToStorage();
			this.refreshOverlay();
			this.draw();
		});

		this.canvas.addEventListener('mouseleave', () => {
			this.drag = null;
			this.draftObstacleStart = null;
			this.draftObstacleCurrent = null;
		});

		this.canvas.addEventListener('dragover', (event) => {
			if (!this.visible || this.playtesting) return;
			event.preventDefault();
		});

		this.canvas.addEventListener('drop', (event) => {
			if (!this.visible || this.playtesting) return;
			event.preventDefault();
			const kind = (event.dataTransfer?.getData('text/tank-kind') ?? '') as TankKind;
			if (!kind) return;
			const point = this.getCanvasPoint(event);
			if (this.kindSelect) this.kindSelect.value = kind;
			if (this.toolSelect) this.toolSelect.value = 'tank';
			this.addTankAt(point.x, point.y);
			this.persistToStorage();
			this.refreshOverlay();
			this.draw();
		});
	}

	private addTankAt(x: number, y: number): void {
		const kind = (this.kindSelect?.value ?? 'player') as TankKind;
		const team = this.teamInput?.value.trim() || 'alpha';
		const control = (this.controlSelect?.value ?? (kind === 'player' ? 'human' : 'scripted')) as ControlType;
		this.config.tanks = this.config.tanks ?? [];
		this.config.tanks.push(
			makeTank(kind, team, control, x - TANK_SIZE / 2, y - TANK_SIZE / 2, this.config.tanks.length)
		);
	}

	private findDragTarget(x: number, y: number): DragTarget | null {
		const tanks = this.config.tanks ?? [];
		for (let index = tanks.length - 1; index >= 0; index -= 1) {
			const tank = tanks[index];
			if (isPointInRect(x, y, { x: tank.x, y: tank.y, width: TANK_SIZE, height: TANK_SIZE })) {
				this.selectedTankIndex = index;
				return { kind: 'tank', index, offsetX: x - tank.x, offsetY: y - tank.y };
			}
		}
		for (let index = this.config.obstacles.length - 1; index >= 0; index -= 1) {
			const obstacle = this.config.obstacles[index];
			if (isPointInRect(x, y, obstacle)) {
				return { kind: 'obstacle', index, offsetX: x - obstacle.x, offsetY: y - obstacle.y };
			}
		}
		return null;
	}

	private applyDrag(x: number, y: number): void {
		if (!this.drag) return;
		if (this.drag.kind === 'tank') {
			const tank = this.config.tanks?.[this.drag.index];
			if (!tank) return;
			tank.x = clampTankX(x - this.drag.offsetX);
			tank.y = clampTankY(y - this.drag.offsetY);
			return;
		}
		const obstacle = this.config.obstacles[this.drag.index];
		if (!obstacle) return;
		obstacle.x = clamp(Math.round(x - this.drag.offsetX), 0, ARENA_WIDTH - obstacle.width);
		obstacle.y = clamp(Math.round(y - this.drag.offsetY), 0, ARENA_HEIGHT - obstacle.height);
	}

	private commitDraftObstacle(endX: number, endY: number): void {
		if (!this.draftObstacleStart) {
			const width = clamp(Number(this.obstacleWidthInput?.value ?? DEFAULT_OBSTACLE_WIDTH), 8, ARENA_WIDTH);
			const height = clamp(Number(this.obstacleHeightInput?.value ?? DEFAULT_OBSTACLE_HEIGHT), 8, ARENA_HEIGHT);
			this.config.obstacles.push(sanitizeObstacle({ x: endX - width / 2, y: endY - height / 2, width, height }));
			return;
		}
		const minX = Math.min(this.draftObstacleStart.x, endX);
		const minY = Math.min(this.draftObstacleStart.y, endY);
		const width = Math.abs(endX - this.draftObstacleStart.x);
		const height = Math.abs(endY - this.draftObstacleStart.y);
		if (width >= 8 && height >= 8) {
			this.config.obstacles.push(sanitizeObstacle({ x: minX, y: minY, width, height }));
		} else {
			const fallbackWidth = clamp(Number(this.obstacleWidthInput?.value ?? DEFAULT_OBSTACLE_WIDTH), 8, ARENA_WIDTH);
			const fallbackHeight = clamp(Number(this.obstacleHeightInput?.value ?? DEFAULT_OBSTACLE_HEIGHT), 8, ARENA_HEIGHT);
			this.config.obstacles.push(
				sanitizeObstacle({
					x: endX - fallbackWidth / 2,
					y: endY - fallbackHeight / 2,
					width: fallbackWidth,
					height: fallbackHeight,
				})
			);
		}
		this.draftObstacleStart = null;
		this.draftObstacleCurrent = null;
	}

	private eraseAt(x: number, y: number): void {
		const tanks = this.config.tanks ?? [];
		for (let index = tanks.length - 1; index >= 0; index -= 1) {
			if (isPointInRect(x, y, { x: tanks[index].x, y: tanks[index].y, width: TANK_SIZE, height: TANK_SIZE })) {
				tanks.splice(index, 1);
				if (this.selectedTankIndex === index) this.selectedTankIndex = null;
				return;
			}
		}
		for (let index = this.config.obstacles.length - 1; index >= 0; index -= 1) {
			if (isPointInRect(x, y, this.config.obstacles[index])) {
				this.config.obstacles.splice(index, 1);
				return;
			}
		}
	}

	private buildRulesInputs(container: HTMLElement): void {
		const fields: {
			key: keyof Required<MatchRulesConfig>;
			label: string;
			type: 'number' | 'checkbox';
			step?: string;
		}[] = [
			{ key: 'tankHitPoints', label: 'HP', type: 'number' },
			{ key: 'projectileDamage', label: 'Projectile', type: 'number' },
			{ key: 'bombDamage', label: 'Bomb', type: 'number' },
			{ key: 'invulnerabilityTicks', label: 'Invuln Ticks', type: 'number' },
			{ key: 'projectileBounces', label: 'Bounces', type: 'checkbox' },
			{ key: 'turretSpeedMultiplier', label: 'Turret Speed', type: 'number', step: '0.1' },
		];
		for (const field of fields) {
			const row = document.createElement('label');
			row.className = 'editor-rule-row';
			const name = document.createElement('span');
			name.textContent = field.label;
			const input = document.createElement('input');
			input.className = 'editor-input compact';
			input.dataset.ruleField = String(field.key);
			if (field.type === 'checkbox') {
				input.type = 'checkbox';
			} else {
				input.type = 'number';
				if (field.step) input.step = field.step;
			}
			input.addEventListener('change', () => {
				const rules = normalizeRules(this.config.rules);
				if (field.type === 'checkbox') {
					(rules[field.key] as boolean) = input.checked;
				} else {
					(rules[field.key] as number) = Number(input.value);
				}
				this.config.rules = normalizeRules(rules);
				this.persistToStorage();
				this.refreshOverlay();
			});
			row.append(name, input);
			container.appendChild(row);
		}
	}

	private refreshOverlay(): void {
		if (!this.overlay) return;
		const tanks = this.config.tanks ?? [];
		if (this.tanksList) {
			this.tanksList.innerHTML = '';
			if (tanks.length === 0) {
				const empty = document.createElement('p');
				empty.className = 'editor-empty';
				empty.textContent = 'No tanks yet.';
				this.tanksList.appendChild(empty);
			}
			tanks.forEach((tank, index) => {
				const row = document.createElement('div');
				row.className = `editor-list-row${this.selectedTankIndex === index ? ' selected' : ''}`;
				const kind = document.createElement('select');
				kind.className = 'editor-select';
				for (const value of [
					'player',
					'stationary',
					'stationary-random-aim',
					'simple-moving',
					'bomber',
					'super-bomber',
				] as TankKind[]) {
					const option = document.createElement('option');
					option.value = value;
					option.textContent = value;
					kind.appendChild(option);
				}
				kind.value = tank.kind;
				kind.addEventListener('change', () => {
					tank.kind = kind.value as TankKind;
					Object.assign(tank, defaultTankForKind(tank.kind));
					this.persistToStorage();
					this.refreshOverlay();
					this.draw();
				});

				const team = document.createElement('input');
				team.className = 'editor-input compact';
				team.value = tank.team;
				team.addEventListener('change', () => {
					tank.team = team.value.trim() || 'alpha';
					this.persistToStorage();
					this.draw();
				});

				const control = document.createElement('select');
				control.className = 'editor-select';
				for (const mode of ['human', 'scripted'] as ControlType[]) {
					const option = document.createElement('option');
					option.value = mode;
					option.textContent = mode;
					control.appendChild(option);
				}
				control.value = tank.control ?? (tank.kind === 'player' ? 'human' : 'scripted');
				control.addEventListener('change', () => {
					tank.control = control.value as ControlType;
					this.persistToStorage();
				});

				const remove = document.createElement('button');
				remove.type = 'button';
				remove.className = 'control-button danger';
				remove.textContent = 'Remove';
				remove.addEventListener('click', () => {
					tanks.splice(index, 1);
					this.persistToStorage();
					this.refreshOverlay();
					this.draw();
				});

				row.append(kind, team, control, remove);
				this.tanksList?.appendChild(row);
			});
		}

		if (this.obstaclesList) {
			this.obstaclesList.innerHTML = '';
			if (this.config.obstacles.length === 0) {
				const empty = document.createElement('p');
				empty.className = 'editor-empty';
				empty.textContent = 'No obstacles yet.';
				this.obstaclesList.appendChild(empty);
			}
			this.config.obstacles.forEach((obstacle, index) => {
				const row = document.createElement('div');
				row.className = 'editor-list-row';
				const coords = document.createElement('input');
				coords.className = 'editor-input';
				coords.value = `${obstacle.x},${obstacle.y},${obstacle.width},${obstacle.height}`;
				coords.addEventListener('change', () => {
					const values = coords.value.split(',').map((value) => Number(value.trim()));
					if (values.length !== 4 || values.some((value) => Number.isNaN(value))) {
						this.setStatus('Obstacle format: x,y,width,height');
						return;
					}
					this.config.obstacles[index] = sanitizeObstacle({
						x: values[0],
						y: values[1],
						width: values[2],
						height: values[3],
					});
					this.persistToStorage();
					this.refreshOverlay();
					this.draw();
				});
				const remove = document.createElement('button');
				remove.type = 'button';
				remove.className = 'control-button danger';
				remove.textContent = 'Remove';
				remove.addEventListener('click', () => {
					this.config.obstacles.splice(index, 1);
					this.persistToStorage();
					this.refreshOverlay();
					this.draw();
				});
				row.append(coords, remove);
				this.obstaclesList?.appendChild(row);
			});
		}

		const ruleInputs = this.overlay.querySelectorAll<HTMLInputElement>('input[data-rule-field]');
		const rules = normalizeRules(this.config.rules);
		ruleInputs.forEach((input) => {
			const key = input.dataset.ruleField as keyof Required<MatchRulesConfig>;
			if (!key) return;
			if (input.type === 'checkbox') {
				input.checked = Boolean(rules[key]);
			} else {
				input.value = String(rules[key]);
			}
		});

		if (this.jsonArea) {
			this.jsonArea.value = JSON.stringify(this.getConfig(), null, 2);
		}
	}

	private refreshPlaytestButtons(): void {
		if (this.playtestButton) this.playtestButton.disabled = this.playtesting;
		if (this.stopPlaytestButton) this.stopPlaytestButton.disabled = !this.playtesting;
	}

	private applyJson(): void {
		if (!this.jsonArea) return;
		try {
			const parsed = JSON.parse(this.jsonArea.value) as LevelConfig;
			this.config = this.sanitizeConfig(parsed);
			this.persistToStorage();
			this.refreshOverlay();
			this.draw();
			this.setStatus('JSON applied.');
		} catch {
			this.setStatus('Invalid JSON.');
		}
	}

	private draw(): void {
		if (!this.visible || this.playtesting) return;
		const ctx = this.context;
		ctx.clearRect(0, 0, ARENA_WIDTH, ARENA_HEIGHT);
		ctx.fillStyle = '#0d141f';
		ctx.fillRect(0, 0, ARENA_WIDTH, ARENA_HEIGHT);

		ctx.strokeStyle = 'rgba(255,255,255,0.08)';
		ctx.lineWidth = 1;
		for (let x = 0; x <= ARENA_WIDTH; x += GRID_SIZE) {
			ctx.beginPath();
			ctx.moveTo(x + 0.5, 0);
			ctx.lineTo(x + 0.5, ARENA_HEIGHT);
			ctx.stroke();
		}
		for (let y = 0; y <= ARENA_HEIGHT; y += GRID_SIZE) {
			ctx.beginPath();
			ctx.moveTo(0, y + 0.5);
			ctx.lineTo(ARENA_WIDTH, y + 0.5);
			ctx.stroke();
		}

		for (const obstacle of this.config.obstacles) {
			ctx.fillStyle = '#273444';
			ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
			ctx.strokeStyle = '#8ec5ff';
			ctx.lineWidth = 2;
			ctx.strokeRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
		}

		const tanks = this.config.tanks ?? [];
		tanks.forEach((tank, index) => {
			ctx.fillStyle = teamColor(tank.team);
			ctx.fillRect(tank.x, tank.y, TANK_SIZE, TANK_SIZE);
			ctx.strokeStyle = this.selectedTankIndex === index ? '#ffffff' : 'rgba(0,0,0,0.65)';
			ctx.lineWidth = this.selectedTankIndex === index ? 3 : 2;
			ctx.strokeRect(tank.x, tank.y, TANK_SIZE, TANK_SIZE);
			ctx.fillStyle = 'rgba(0,0,0,0.75)';
			ctx.font = '11px Consolas, monospace';
			ctx.fillText(`${tank.team}:${tank.control ?? 'scripted'}`, tank.x - 2, Math.max(12, tank.y - 6));
		});

		if (this.draftObstacleStart && this.draftObstacleCurrent) {
			const x = Math.min(this.draftObstacleStart.x, this.draftObstacleCurrent.x);
			const y = Math.min(this.draftObstacleStart.y, this.draftObstacleCurrent.y);
			const width = Math.abs(this.draftObstacleCurrent.x - this.draftObstacleStart.x);
			const height = Math.abs(this.draftObstacleCurrent.y - this.draftObstacleStart.y);
			ctx.fillStyle = 'rgba(255,122,24,0.24)';
			ctx.fillRect(x, y, width, height);
			ctx.strokeStyle = '#ff9a4d';
			ctx.lineWidth = 2;
			ctx.strokeRect(x, y, width, height);
		}

		const teams = new Set((this.config.tanks ?? []).map((tank) => tank.team));
		ctx.fillStyle = 'rgba(0,0,0,0.62)';
		ctx.fillRect(12, 12, 560, 58);
		ctx.fillStyle = '#eef5ff';
		ctx.font = '13px Consolas, monospace';
		ctx.fillText(`Tanks ${tanks.length} | Teams ${teams.size} | Obstacles ${this.config.obstacles.length}`, 20, 34);
		ctx.fillText(`Drop tanks for AI vs AI, and set any tank control to human for player input`, 20, 54);
	}

	private getCanvasPoint(event: MouseEvent | DragEvent): { x: number; y: number } {
		const bounds = this.canvas.getBoundingClientRect();
		const scaleX = this.canvas.width / bounds.width;
		const scaleY = this.canvas.height / bounds.height;
		const x = Math.round(((event.clientX ?? 0) - bounds.left) * scaleX);
		const y = Math.round(((event.clientY ?? 0) - bounds.top) * scaleY);
		return { x: clamp(x, 0, ARENA_WIDTH), y: clamp(y, 0, ARENA_HEIGHT) };
	}

	private setStatus(message: string): void {
		if (this.status) this.status.textContent = message;
	}

	private persistToStorage(): void {
		try {
			window.localStorage.setItem(STORAGE_KEY, JSON.stringify(this.config));
		} catch {
			// Ignore local storage failures.
		}
	}

	private restoreFromStorage(): void {
		try {
			const raw = window.localStorage.getItem(STORAGE_KEY);
			if (!raw) {
				this.config = deepClone(DEFAULT_LEVEL_CONFIG);
				return;
			}
			this.config = this.sanitizeConfig(JSON.parse(raw) as LevelConfig);
		} catch {
			this.config = deepClone(DEFAULT_LEVEL_CONFIG);
		}
	}
}
