/**
 * Enhanced Training Dashboard with Real-time Metrics and Charts
 * Replaces the basic CSV loader with a polished training experience
 */

import * as d3 from 'd3';

interface TrainingMetric {
	iteration: number;
	timesteps: number;
	episodes: number;
	curriculum_phase: string;
	active_scenarios: string;
	phase_recent_winrate_500: number;
	avg_reward_50: number;
	avg_winrate_50: number;
	avg_tick_50: number;
	avg_hit_50: number;
	avg_hurt_50: number;
	avg_kill_50: number;
	avg_death_50: number;
	avg_terminal_win_50: number;
	avg_terminal_loss_50: number;
	avg_timeout_50: number;
	avg_approach_50: number;
	avg_dodge_50: number;
	steps_per_sec: number;
	elapsed_sec: number;
}

interface Replay {
	name: string;
	size: number;
	mtime: number;
}

export class EnhancedTrainingDashboard {
	private container: HTMLDivElement | null = null;
	private onClose: (() => void) | null = null;
	private metrics: TrainingMetric[] = [];
	private updateInterval: NodeJS.Timeout | null = null;
	private readonly serverUrl = 'http://localhost:3007';
	private activeTab: 'dashboard' | 'replays' = 'dashboard';

	public show(anchorElement: HTMLElement, onClose: () => void): void {
		this.onClose = onClose;
		this.container = document.createElement('div');
		this.container.id = 'enhanced-training-dashboard';
		this.container.style.cssText = `
			position: fixed;
			top: 0;
			left: 0;
			right: 0;
			bottom: 0;
			z-index: 10000;
			background: #f8f9fa;
			overflow-y: auto;
		`;

		// Header with tabs
		const header = document.createElement('div');
		header.style.cssText = `
			background: #2c3e50;
			color: white;
			padding: 20px;
			display: flex;
			justify-content: space-between;
			align-items: center;
			box-shadow: 0 2px 4px rgba(0,0,0,0.1);
		`;

		const title = document.createElement('h1');
		title.textContent = '🎮 Treads Training Dashboard';
		title.style.cssText = 'margin: 0; font-size: 24px;';

		const closeBtn = document.createElement('button');
		closeBtn.textContent = '✕ Close';
		closeBtn.className = 'btn btn-outline-light btn-sm';
		closeBtn.addEventListener('click', () => this.destroy());

		header.appendChild(title);
		header.appendChild(closeBtn);

		// Tab buttons
		const tabContainer = document.createElement('div');
		tabContainer.style.cssText = `
			background: white;
			padding: 10px 20px;
			display: flex;
			gap: 10px;
			border-bottom: 1px solid #e0e0e0;
		`;

		const dashboardTab = document.createElement('button');
		dashboardTab.textContent = 'Live Metrics';
		dashboardTab.className = 'btn btn-sm btn-primary';
		dashboardTab.addEventListener('click', () => this.switchTab('dashboard', dashboardTab));

		const replaysTab = document.createElement('button');
		replaysTab.textContent = 'Replays';
		replaysTab.className = 'btn btn-sm btn-outline-primary';
		replaysTab.addEventListener('click', () => this.switchTab('replays', replaysTab));

		tabContainer.appendChild(dashboardTab);
		tabContainer.appendChild(replaysTab);

		// Content area
		const content = document.createElement('div');
		content.id = 'dashboard-content';
		content.style.cssText = 'padding: 20px; max-width: 1400px; margin: 0 auto;';

		this.container.appendChild(header);
		this.container.appendChild(tabContainer);
		this.container.appendChild(content);

		anchorElement.parentElement?.insertBefore(this.container, anchorElement);

		// Start loading data
		this.loadMetricsFromServer();
		this.updateInterval = setInterval(() => this.loadMetricsFromServer(), 3000);

		// Initial render
		this.renderDashboard();
	}

	private switchTab(tab: 'dashboard' | 'replays', button: HTMLButtonElement): void {
		this.activeTab = tab;

		// Update button styles
		const buttons = this.container?.querySelectorAll('[role="button"]');
		buttons?.forEach((btn) => {
			if (btn === button) {
				btn.className = 'btn btn-sm btn-primary';
			} else {
				btn.className = 'btn btn-sm btn-outline-primary';
			}
		});

		// Re-render with new tab
		if (tab === 'dashboard') {
			this.renderDashboard();
		} else {
			this.renderReplays();
		}
	}

	private async loadMetricsFromServer(): Promise<void> {
		try {
			const response = await fetch(`${this.serverUrl}/api/training/metrics`);
			if (!response.ok) throw new Error(`HTTP ${response.status}`);
			this.metrics = await response.json();

			// Re-render if we're on the dashboard tab
			if (this.activeTab === 'dashboard') {
				this.renderDashboard();
			}
		} catch (err) {
			console.error('Failed to load metrics:', err);
		}
	}

	private async loadReplaysFromServer(): Promise<Replay[]> {
		try {
			const response = await fetch(`${this.serverUrl}/api/replays`);
			if (!response.ok) throw new Error(`HTTP ${response.status}`);
			return await response.json();
		} catch (err) {
			console.error('Failed to load replays:', err);
			return [];
		}
	}

	private renderDashboard(): void {
		if (!this.container) return;

		const content = this.container.querySelector('#dashboard-content');
		if (!content) return;
		content.innerHTML = '';

		if (this.metrics.length === 0) {
			content.innerHTML =
				'<p style="text-align: center; color: #999;">No training data yet. Start training to see metrics.</p>';
			return;
		}

		// Status cards
		const latest = this.metrics[this.metrics.length - 1];
		const statusHtml = this.renderStatusCards(latest);
		content.appendChild(statusHtml);

		// Charts container
		const chartsContainer = document.createElement('div');
		chartsContainer.style.cssText = `
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
			gap: 20px;
			margin-top: 20px;
		`;

		// Reward chart
		const rewardChart = document.createElement('div');
		rewardChart.style.cssText =
			'background: white; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);';
		chartsContainer.appendChild(rewardChart);
		this.renderLineChart(rewardChart, 'Avg Reward (50 ep)', this.metrics, (d) => d.avg_reward_50, '#4caf50');

		// Win rate chart
		const winRateChart = document.createElement('div');
		winRateChart.style.cssText =
			'background: white; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);';
		chartsContainer.appendChild(winRateChart);
		this.renderLineChart(winRateChart, 'Win Rate (50 ep)', this.metrics, (d) => d.avg_winrate_50 * 100, '#2196f3', '%');

		// Reward breakdown chart
		const breakdownChart = document.createElement('div');
		breakdownChart.style.cssText =
			'background: white; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);';
		chartsContainer.appendChild(breakdownChart);
		this.renderStackedChart(breakdownChart);

		content.appendChild(chartsContainer);
	}

	private renderStatusCards(latest: TrainingMetric): HTMLElement {
		const container = document.createElement('div');
		container.style.cssText = `
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
			gap: 15px;
		`;

		interface Card {
			label: string;
			value: number | string;
			format: (v: number | string) => string;
		}

		const cards: Card[] = [
			{ label: 'Episodes', value: latest.episodes || 0, format: (v) => (v as number).toLocaleString() },
			{ label: 'Timesteps', value: latest.timesteps || 0, format: (v) => ((v as number) / 1000).toFixed(1) + 'k' },
			{ label: 'Avg Reward', value: latest.avg_reward_50 || 0, format: (v) => (v as number).toFixed(2) },
			{ label: 'Win Rate', value: (latest.avg_winrate_50 || 0) * 100, format: (v) => (v as number).toFixed(1) + '%' },
			{ label: 'Phase', value: latest.curriculum_phase?.substring(0, 15) || '—', format: (v) => String(v) },
			{ label: 'Steps/sec', value: latest.steps_per_sec || 0, format: (v) => Math.round(v as number).toLocaleString() },
			{ label: 'Elapsed', value: latest.elapsed_sec || 0, format: (v) => this.formatSeconds(v as number) },
			{
				label: 'Recent Win%',
				value: (latest.phase_recent_winrate_500 || 0) * 100,
				format: (v) => (v as number).toFixed(1) + '%',
			},
		];

		cards.forEach(({ label, value, format }) => {
			const card = document.createElement('div');
			card.style.cssText = `
				background: white;
				padding: 15px;
				border-radius: 6px;
				box-shadow: 0 2px 4px rgba(0,0,0,0.1);
				border-left: 4px solid #4caf50;
			`;

			const labelEl = document.createElement('div');
			labelEl.style.cssText = 'font-size: 12px; color: #666; text-transform: uppercase; font-weight: 500;';
			labelEl.textContent = label;

			const valueEl = document.createElement('div');
			valueEl.style.cssText = 'font-size: 20px; font-weight: bold; color: #333; margin-top: 5px;';
			valueEl.textContent = format(value);

			card.appendChild(labelEl);
			card.appendChild(valueEl);
			container.appendChild(card);
		});

		return container;
	}

	private renderLineChart(
		container: HTMLElement,
		title: string,
		data: TrainingMetric[],
		accessor: (d: TrainingMetric) => number,
		color: string,
		suffix = ''
	): void {
		const width = container.clientWidth - 30;
		const height = 300;
		const margin = { top: 20, right: 30, bottom: 30, left: 60 };

		const svg = d3
			.select(container)
			.append('svg')
			.attr('width', width)
			.attr('height', height)
			.append('g')
			.attr('transform', `translate(${margin.left},${margin.top})`);

		const innerWidth = width - margin.left - margin.right;
		const innerHeight = height - margin.top - margin.bottom;

		const xScale = d3
			.scaleLinear()
			.domain([0, data.length - 1])
			.range([0, innerWidth]);
		const yScale = d3
			.scaleLinear()
			.domain([0, d3.max(data, accessor) || 1])
			.range([innerHeight, 0]);

		const line = d3
			.line<TrainingMetric>()
			.x((d, i) => xScale(i))
			.y((d) => yScale(accessor(d)));

		// Add title
		d3.select(container).insert('h4', 'svg').style('margin', '0 0 10px 0').text(title);

		// Path
		svg.append('path').datum(data).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2).attr('d', line);

		// X axis
		svg
			.append('g')
			.attr('transform', `translate(0,${innerHeight})`)
			.call(d3.axisBottom(xScale).tickFormat((d) => `Iter ${d}`));

		// Y axis
		svg.append('g').call(d3.axisLeft(yScale).tickFormat((d) => `${d}${suffix}`));
	}

	private renderStackedChart(container: HTMLElement): void {
		const rewardKeys = ['avg_hit_50', 'avg_kill_50', 'avg_terminal_win_50', 'avg_approach_50'];
		const width = container.clientWidth - 30;
		const height = 300;
		const margin = { top: 20, right: 30, bottom: 30, left: 60 };

		d3.select(container).append('h4').style('margin', '0 0 10px 0').text('Reward Breakdown (50 ep)');

		const svg = d3
			.select(container)
			.append('svg')
			.attr('width', width)
			.attr('height', height)
			.append('g')
			.attr('transform', `translate(${margin.left},${margin.top})`);

		const innerWidth = width - margin.left - margin.right;
		const innerHeight = height - margin.top - margin.bottom;

		const xScale = d3
			.scaleLinear()
			.domain([0, this.metrics.length - 1])
			.range([0, innerWidth]);
		const yScale = d3.scaleLinear().domain([0, 1]).range([innerHeight, 0]);

		const colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4d96ff'];
		const stack = d3.stack<TrainingMetric>().keys(rewardKeys);
		const stackedData = stack(this.metrics);

		stackedData.forEach((series, i) => {
			const area = d3
				.area<(typeof series)[number]>()
				.x((d, idx) => xScale(idx))
				.y0((d) => yScale(d[0]))
				.y1((d) => yScale(d[1]));

			svg
				.append('path')
				.datum(series)
				.attr('fill', colors[i % colors.length])
				.attr('opacity', 0.7)
				.attr('d', area);
		});

		// Axes
		svg.append('g').attr('transform', `translate(0,${innerHeight})`).call(d3.axisBottom(xScale));
		svg.append('g').call(d3.axisLeft(yScale));
	}

	private async renderReplays(): Promise<void> {
		if (!this.container) return;

		const content = this.container.querySelector('#dashboard-content');
		if (!content) return;
		content.innerHTML = '<p>Loading replays...</p>';

		const replays = await this.loadReplaysFromServer();

		content.innerHTML = '';

		if (replays.length === 0) {
			content.innerHTML = '<p style="text-align: center; color: #999;">No replays found.</p>';
			return;
		}

		const listContainer = document.createElement('div');
		listContainer.style.cssText =
			'background: white; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);';

		const table = document.createElement('table');
		table.style.cssText = 'width: 100%; border-collapse: collapse;';

		const header = document.createElement('thead');
		header.innerHTML = `
			<tr style="background: #f5f5f5; border-bottom: 1px solid #e0e0e0;">
				<th style="padding: 12px; text-align: left; font-weight: 600;">Replay</th>
				<th style="padding: 12px; text-align: right; font-weight: 600;">Size</th>
				<th style="padding: 12px; text-align: right; font-weight: 600;">Date</th>
				<th style="padding: 12px; text-align: center; font-weight: 600;">Action</th>
			</tr>
		`;
		table.appendChild(header);

		const body = document.createElement('tbody');
		replays.forEach((replay, idx) => {
			const row = document.createElement('tr');
			row.style.cssText = `border-bottom: 1px solid #e0e0e0; ${idx % 2 === 0 ? 'background: #fafafa;' : ''}`;

			const nameCell = document.createElement('td');
			nameCell.style.cssText = 'padding: 12px;';
			nameCell.textContent = replay.name;

			const sizeCell = document.createElement('td');
			sizeCell.style.cssText = 'padding: 12px; text-align: right; color: #666;';
			sizeCell.textContent = this.formatBytes(replay.size);

			const dateCell = document.createElement('td');
			dateCell.style.cssText = 'padding: 12px; text-align: right; color: #666;';
			dateCell.textContent = new Date(replay.mtime).toLocaleString();

			const actionCell = document.createElement('td');
			actionCell.style.cssText = 'padding: 12px; text-align: center;';
			const viewBtn = document.createElement('button');
			viewBtn.className = 'btn btn-sm btn-outline-primary';
			viewBtn.textContent = 'View';
			viewBtn.addEventListener('click', () => this.viewReplay(replay.name));
			actionCell.appendChild(viewBtn);

			row.appendChild(nameCell);
			row.appendChild(sizeCell);
			row.appendChild(dateCell);
			row.appendChild(actionCell);
			body.appendChild(row);
		});

		table.appendChild(body);
		listContainer.appendChild(table);
		content.appendChild(listContainer);
	}

	private async viewReplay(filename: string): Promise<void> {
		try {
			const replay = await fetch(`${this.serverUrl}/api/replays/${filename}`).then((r) => r.json());

			// Store replay in window for ReplayViewer to access
			(window as unknown as Record<string, unknown>).__selectedReplay = replay;

			// Dispatch custom event so ReplayViewer can pick it up
			window.dispatchEvent(new CustomEvent('replay-selected', { detail: replay }));

			console.log('Replay loaded:', replay);
		} catch (err) {
			console.error('Failed to load replay:', err);
			alert('Failed to load replay');
		}
	}

	private formatSeconds(seconds: number): string {
		if (seconds < 60) return `${Math.round(seconds)}s`;
		if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
		return `${(seconds / 3600).toFixed(1)}h`;
	}

	private formatBytes(bytes: number): string {
		if (bytes < 1024) return `${bytes}B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
	}

	public destroy(): void {
		if (this.updateInterval) {
			clearInterval(this.updateInterval);
		}
		this.container?.remove();
		this.container = null;
		this.onClose?.();
	}
}
