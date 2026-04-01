import * as d3 from 'd3';

import {
	DashboardApiClient,
	type LivePayload,
	type MetricsSnapshot,
	type RunSummary,
	type RunsResponse,
} from './dashboardApi';

interface TrainingDashboardOptions {
	onRunSelected?: (runId: string) => void;
	onRunsUpdated?: (response: RunsResponse) => void;
}

interface SeriesDef {
	key: keyof MetricsSnapshot;
	label: string;
	color: string;
}

const SERIES_COLORS = ['#ff7a18', '#ffb627', '#7bdff2', '#6ef3a5', '#f96db6', '#b48ef7'];

function formatCompact(value: number | null | undefined, digits = 1): string {
	if (value === null || value === undefined || Number.isNaN(value)) {
		return '--';
	}
	const abs = Math.abs(value);
	if (abs >= 1_000_000) {
		return `${(value / 1_000_000).toFixed(digits)}M`;
	}
	if (abs >= 1_000) {
		return `${(value / 1_000).toFixed(digits)}k`;
	}
	return abs >= 100 ? value.toFixed(0) : value.toFixed(digits);
}

function formatPercent(value: number | null | undefined): string {
	if (value === null || value === undefined || Number.isNaN(value)) {
		return '--';
	}
	return `${(value * 100).toFixed(1)}%`;
}

function formatAge(isoString: string | null | undefined): string {
	if (!isoString) {
		return 'no activity';
	}
	const ageMs = Date.now() - Date.parse(isoString);
	if (!Number.isFinite(ageMs)) {
		return 'unknown';
	}
	const ageSec = Math.max(0, Math.round(ageMs / 1000));
	if (ageSec < 60) {
		return `${ageSec}s ago`;
	}
	if (ageSec < 3600) {
		return `${Math.round(ageSec / 60)}m ago`;
	}
	return `${Math.round(ageSec / 3600)}h ago`;
}

function chartSeries(history: MetricsSnapshot[]): SeriesDef[] {
	const preferred: SeriesDef[] = [
		{ key: 'avgHit50', label: 'Hit', color: SERIES_COLORS[0] },
		{ key: 'avgHurt50', label: 'Hurt', color: SERIES_COLORS[4] },
		{ key: 'avgKill50', label: 'Kill', color: SERIES_COLORS[1] },
		{ key: 'avgDeath50', label: 'Death', color: '#d64045' },
		{ key: 'avgApproach50', label: 'Approach', color: SERIES_COLORS[2] },
		{ key: 'avgDodge50', label: 'Dodge', color: SERIES_COLORS[3] },
		{ key: 'avgAimJitter50', label: 'Aim Jitter', color: SERIES_COLORS[5] },
		{ key: 'avgMoveJitter50', label: 'Move Jitter', color: '#5f6caf' },
	];
	return preferred.filter((series) =>
		history.some((entry) => typeof entry[series.key] === 'number' && Number.isFinite(entry[series.key] as number))
	);
}

export class TrainingDashboard {
	private container: HTMLElement | null = null;
	private readonly api: DashboardApiClient;
	private readonly options: TrainingDashboardOptions;
	private runs: RunSummary[] = [];
	private history: MetricsSnapshot[] = [];
	private selectedRunId: string | null = null;
	private liveSource: EventSource | null = null;
	private refreshTimer: number | null = null;
	private resizeListener: (() => void) | null = null;

	private liveBadge: HTMLDivElement | null = null;
	private runList: HTMLDivElement | null = null;
	private summaryCard: HTMLDivElement | null = null;
	private kpiGrid: HTMLDivElement | null = null;
	private chartGrid: HTMLDivElement | null = null;

	public constructor(api: DashboardApiClient, options: TrainingDashboardOptions = {}) {
		this.api = api;
		this.options = options;
	}

	public mount(container: HTMLElement): void {
		this.container = container;
		this.container.innerHTML = '';
		this.container.classList.add('training-console');

		const header = document.createElement('div');
		header.className = 'panel-header';

		const titleGroup = document.createElement('div');
		titleGroup.className = 'panel-header-copy';
		const eyebrow = document.createElement('span');
		eyebrow.className = 'panel-eyebrow';
		eyebrow.textContent = 'Training Command';
		const title = document.createElement('h2');
		title.textContent = 'Run Console';
		titleGroup.append(eyebrow, title);

		this.liveBadge = document.createElement('div');
		this.liveBadge.className = 'live-badge';
		this.liveBadge.textContent = 'Scanning runs';

		header.append(titleGroup, this.liveBadge);

		this.runList = document.createElement('div');
		this.runList.className = 'run-list';

		this.summaryCard = document.createElement('div');
		this.summaryCard.className = 'panel-card run-summary-card';

		this.kpiGrid = document.createElement('div');
		this.kpiGrid.className = 'kpi-grid';

		this.chartGrid = document.createElement('div');
		this.chartGrid.className = 'chart-grid';

		this.container.append(header, this.runList, this.summaryCard, this.kpiGrid, this.chartGrid);

		this.resizeListener = () => this.renderCharts();
		window.addEventListener('resize', this.resizeListener);
		void this.refreshRuns();
		this.refreshTimer = window.setInterval(() => {
			void this.refreshRuns(false);
		}, 10_000);
	}

	public destroy(): void {
		if (this.liveSource) {
			this.liveSource.close();
			this.liveSource = null;
		}
		if (this.refreshTimer !== null) {
			window.clearInterval(this.refreshTimer);
			this.refreshTimer = null;
		}
		if (this.resizeListener) {
			window.removeEventListener('resize', this.resizeListener);
			this.resizeListener = null;
		}
		this.container = null;
	}

	public getSelectedRunId(): string | null {
		return this.selectedRunId;
	}

	public async setRun(runId: string): Promise<void> {
		if (runId === this.selectedRunId) {
			return;
		}
		this.selectedRunId = runId;
		this.options.onRunSelected?.(runId);
		await this.loadRunHistory(runId);
		this.connectLive(runId);
		this.render();
	}

	private async refreshRuns(allowAutoSelection = true): Promise<void> {
		try {
			const response = await this.api.fetchRuns();
			this.runs = response.runs;
			this.options.onRunsUpdated?.(response);

			if (allowAutoSelection) {
				const nextRunId =
					this.selectedRunId && this.runs.some((run) => run.id === this.selectedRunId)
						? this.selectedRunId
						: (response.activeRunId ?? this.runs[0]?.id ?? null);
				if (nextRunId) {
					await this.setRun(nextRunId);
				}
			}

			if (!this.selectedRunId && this.runs.length > 0) {
				await this.setRun(this.runs[0].id);
			}

			if (this.liveBadge) {
				const activeRun = this.runs.find((run) => run.isRunning) ?? this.runs[0] ?? null;
				this.liveBadge.textContent = activeRun
					? activeRun.isRunning
						? `Live on ${activeRun.label}`
						: `Latest run ${activeRun.label}`
					: 'No runs yet';
			}

			this.renderRunList();
			this.render();
		} catch (error) {
			console.error('Failed to refresh runs:', error);
			if (this.liveBadge) {
				this.liveBadge.textContent = 'Dashboard API unavailable';
			}
		}
	}

	private async loadRunHistory(runId: string): Promise<void> {
		try {
			this.history = await this.api.fetchMetrics(runId);
		} catch (error) {
			console.error(`Failed to load metrics for ${runId}:`, error);
			this.history = [];
		}
	}

	private connectLive(runId: string): void {
		if (this.liveSource) {
			this.liveSource.close();
			this.liveSource = null;
		}
		this.liveSource = this.api.subscribeLive(runId, (payload) => this.applyLiveUpdate(payload));
		this.liveSource.onerror = () => {
			if (this.liveBadge) {
				this.liveBadge.textContent = 'Live stream reconnecting';
			}
		};
	}

	private applyLiveUpdate(payload: LivePayload): void {
		if (!payload.run || payload.run.id !== this.selectedRunId) {
			return;
		}
		const incoming = payload.run.latestMetrics;
		if (incoming) {
			const latestIteration = this.history[this.history.length - 1]?.iteration;
			if (latestIteration === null || latestIteration === undefined || incoming.iteration !== latestIteration) {
				this.history = [...this.history, incoming];
			} else {
				this.history = [...this.history.slice(0, -1), incoming];
			}
		}
		const runIndex = this.runs.findIndex((run) => run.id === payload.run?.id);
		if (runIndex >= 0) {
			this.runs[runIndex] = payload.run;
		}
		if (this.liveBadge) {
			this.liveBadge.textContent = payload.run.isRunning
				? `Streaming ${payload.run.label}`
				: `Watching ${payload.run.label}`;
		}
		this.renderRunList();
		this.render();
	}

	private renderRunList(): void {
		if (!this.runList) {
			return;
		}
		this.runList.innerHTML = '';
		if (this.runs.length === 0) {
			const empty = document.createElement('div');
			empty.className = 'empty-state';
			empty.textContent = 'No training runs found yet.';
			this.runList.appendChild(empty);
			return;
		}

		for (const run of this.runs) {
			const card = document.createElement('button');
			card.className = `run-card${run.id === this.selectedRunId ? ' selected' : ''}`;
			card.type = 'button';
			card.addEventListener('click', () => {
				void this.setRun(run.id);
			});

			const titleRow = document.createElement('div');
			titleRow.className = 'run-card-top';
			const label = document.createElement('span');
			label.className = 'run-card-title';
			label.textContent = run.label;
			const status = document.createElement('span');
			status.className = `status-pill ${run.isRunning ? 'running' : 'idle'}`;
			status.textContent = run.isRunning ? 'live' : 'saved';
			titleRow.append(label, status);

			const meta = document.createElement('div');
			meta.className = 'run-card-meta';
			meta.textContent = `${run.replayCount} replays • ${formatAge(run.updatedAt)}`;

			const reward = run.latestMetrics?.avgReward50;
			const winrate = run.latestMetrics?.avgWinrate50;
			const stat = document.createElement('div');
			stat.className = 'run-card-stats';
			stat.textContent = `Reward ${formatCompact(reward)} • Win ${formatPercent(winrate)}`;

			card.append(titleRow, meta, stat);
			this.runList.appendChild(card);
		}
	}

	private render(): void {
		if (!this.summaryCard || !this.kpiGrid || !this.chartGrid) {
			return;
		}
		const selectedRun = this.runs.find((run) => run.id === this.selectedRunId) ?? null;
		const latest = this.history[this.history.length - 1] ?? selectedRun?.latestMetrics ?? null;

		if (!selectedRun) {
			this.summaryCard.innerHTML = '<div class="empty-state">Pick a run to inspect metrics.</div>';
			this.kpiGrid.innerHTML = '';
			this.chartGrid.innerHTML = '';
			return;
		}

		this.summaryCard.innerHTML = '';
		const summaryHeader = document.createElement('div');
		summaryHeader.className = 'run-summary-header';
		const heading = document.createElement('div');
		const eyebrow = document.createElement('span');
		eyebrow.className = 'panel-eyebrow';
		eyebrow.textContent = selectedRun.isRunning ? 'Live Focus' : 'Replayable Run';
		const title = document.createElement('h3');
		title.textContent = selectedRun.label;
		heading.append(eyebrow, title);

		const freshness = document.createElement('div');
		freshness.className = 'summary-freshness';
		freshness.textContent = formatAge(selectedRun.updatedAt);
		summaryHeader.append(heading, freshness);

		const summaryBody = document.createElement('div');
		summaryBody.className = 'run-summary-body';
		const phase = document.createElement('p');
		phase.textContent = latest?.curriculumPhase
			? `${latest.curriculumPhase} • Scenarios ${latest.activeScenarios.join(', ')}`
			: `Output path ${selectedRun.path}`;
		const detail = document.createElement('p');
		detail.textContent = `Replays ${selectedRun.replayCount} • ONNX ${selectedRun.onnxAvailable ? 'ready' : 'pending'} • Status ${selectedRun.status}`;
		summaryBody.append(phase, detail);
		this.summaryCard.append(summaryHeader, summaryBody);

		this.renderKpis(latest);
		this.renderCharts();
	}

	private renderKpis(latest: MetricsSnapshot | null): void {
		if (!this.kpiGrid) {
			return;
		}
		this.kpiGrid.innerHTML = '';
		const cards = [
			{ label: 'Episodes', value: formatCompact(latest?.episodes) },
			{ label: 'Timesteps', value: formatCompact(latest?.timesteps) },
			{ label: 'Reward (50)', value: formatCompact(latest?.avgReward50, 2) },
			{ label: 'Win Rate (50)', value: formatPercent(latest?.avgWinrate50) },
			{ label: 'Phase Win (500)', value: formatPercent(latest?.phaseRecentWinrate500) },
			{ label: 'Steps / s', value: formatCompact(latest?.stepsPerSec) },
		];

		for (const cardDef of cards) {
			const card = document.createElement('div');
			card.className = 'kpi-card';
			const label = document.createElement('span');
			label.className = 'kpi-label';
			label.textContent = cardDef.label;
			const value = document.createElement('strong');
			value.className = 'kpi-value';
			value.textContent = cardDef.value;
			card.append(label, value);
			this.kpiGrid.appendChild(card);
		}
	}

	private renderCharts(): void {
		if (!this.chartGrid) {
			return;
		}
		this.chartGrid.innerHTML = '';
		if (this.history.length === 0) {
			const empty = document.createElement('div');
			empty.className = 'empty-state';
			empty.textContent = 'This run has not emitted metrics yet.';
			this.chartGrid.appendChild(empty);
			return;
		}

		this.renderLineChart('Reward Trend', [{ key: 'avgReward50', label: 'Avg Reward', color: '#ff7a18' }]);
		this.renderLineChart('Win Rate', [{ key: 'avgWinrate50', label: 'Win Rate', color: '#6ef3a5' }], true);
		this.renderLineChart('Throughput', [{ key: 'stepsPerSec', label: 'Steps / s', color: '#7bdff2' }]);

		const breakdownSeries = chartSeries(this.history).slice(0, 4);
		if (breakdownSeries.length > 0) {
			this.renderLineChart('Reward Mix', breakdownSeries);
		}
	}

	private renderLineChart(title: string, seriesDefs: SeriesDef[], percentageAxis = false): void {
		if (!this.chartGrid) {
			return;
		}
		const card = document.createElement('div');
		card.className = 'chart-card';
		const heading = document.createElement('div');
		heading.className = 'chart-heading';
		heading.textContent = title;
		card.appendChild(heading);

		const mount = document.createElement('div');
		mount.className = 'chart-mount';
		card.appendChild(mount);
		this.chartGrid.appendChild(card);

		const width = Math.max(280, Math.floor(mount.clientWidth || 320));
		const height = 190;
		const margin = { top: 14, right: 16, bottom: 26, left: 44 };
		const svg = d3
			.select(mount)
			.append('svg')
			.attr('viewBox', `0 0 ${width} ${height}`)
			.attr('preserveAspectRatio', 'none');

		const xValues = this.history.map((entry, index) => entry.iteration ?? index + 1);
		const xScale = d3
			.scaleLinear()
			.domain(d3.extent(xValues) as [number, number])
			.range([margin.left, width - margin.right]);

		const yCandidates = this.history.flatMap((entry) =>
			seriesDefs
				.map((series) => entry[series.key])
				.filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
		);
		const yDomain = d3.extent(yCandidates.length > 0 ? yCandidates : [0, 1]) as [number, number];
		const safeMin = yDomain[0] === yDomain[1] ? yDomain[0] - 1 : yDomain[0];
		const safeMax = yDomain[0] === yDomain[1] ? yDomain[1] + 1 : yDomain[1];
		const yScale = d3
			.scaleLinear()
			.domain([safeMin, safeMax])
			.nice()
			.range([height - margin.bottom, margin.top]);

		svg
			.append('g')
			.attr('transform', `translate(0,${height - margin.bottom})`)
			.call(
				d3
					.axisBottom(xScale)
					.ticks(5)
					.tickFormat((value) => formatCompact(Number(value), 0))
			)
			.attr('class', 'chart-axis');

		svg
			.append('g')
			.attr('transform', `translate(${margin.left},0)`)
			.call(
				d3
					.axisLeft(yScale)
					.ticks(4)
					.tickFormat((value) =>
						percentageAxis ? `${(Number(value) * 100).toFixed(0)}%` : formatCompact(Number(value), 1)
					)
			)
			.attr('class', 'chart-axis');

		svg
			.append('g')
			.selectAll('line')
			.data(yScale.ticks(4))
			.join('line')
			.attr('x1', margin.left)
			.attr('x2', width - margin.right)
			.attr('y1', (value) => yScale(value))
			.attr('y2', (value) => yScale(value))
			.attr('class', 'chart-grid-line');

		for (const series of seriesDefs) {
			const values = this.history.filter(
				(entry): entry is MetricsSnapshot =>
					typeof entry[series.key] === 'number' && Number.isFinite(entry[series.key] as number)
			);
			if (values.length === 0) {
				continue;
			}

			const line = d3
				.line<MetricsSnapshot>()
				.x((entry, index) => xScale(entry.iteration ?? index + 1))
				.y((entry) => yScale(entry[series.key] as number))
				.curve(d3.curveMonotoneX);

			svg
				.append('path')
				.datum(values)
				.attr('fill', 'none')
				.attr('stroke', series.color)
				.attr('stroke-width', 2.2)
				.attr('d', line);

			const lastPoint = values[values.length - 1];
			svg
				.append('circle')
				.attr('cx', xScale(lastPoint.iteration ?? values.length))
				.attr('cy', yScale(lastPoint[series.key] as number))
				.attr('r', 3.5)
				.attr('fill', series.color);
		}

		const legend = document.createElement('div');
		legend.className = 'chart-legend';
		for (const series of seriesDefs) {
			const item = document.createElement('span');
			item.className = 'chart-legend-item';
			const swatch = document.createElement('i');
			swatch.style.background = series.color;
			const label = document.createElement('span');
			label.textContent = series.label;
			item.append(swatch, label);
			legend.appendChild(item);
		}
		card.appendChild(legend);
	}
}
