import * as d3 from 'd3';

interface TrainingLogRow {
	episode: number;
	timestep: number;
	reward: number;
	length: number;
	win: number;
	avgReward50: number;
	avgWinrate50: number;
	elapsedSec: number;
}

export class TrainingDashboard {
	private container: HTMLDivElement | null = null;
	private data: TrainingLogRow[] = [];
	private onClose: (() => void) | null = null;

	public show(anchorElement: HTMLElement, onClose: () => void): void {
		this.onClose = onClose;
		this.container = document.createElement('div');
		this.container.id = 'training-dashboard';
		this.container.style.cssText = 'padding:10px; text-align:center;';

		const controls = document.createElement('div');
		controls.style.cssText = 'margin-bottom:10px; display:flex; gap:6px; justify-content:center;';

		const loadBtn = document.createElement('button');
		loadBtn.className = 'btn btn-outline-secondary btn-sm';
		loadBtn.textContent = 'Load CSV';
		loadBtn.addEventListener('click', () => this.openCsvPicker());

		const closeBtn = document.createElement('button');
		closeBtn.className = 'btn btn-outline-secondary btn-sm';
		closeBtn.textContent = 'Close';
		closeBtn.addEventListener('click', () => this.destroy());

		controls.append(loadBtn, closeBtn);
		this.container.appendChild(controls);

		anchorElement.parentElement?.insertBefore(this.container, anchorElement);
	}

	public destroy(): void {
		this.container?.remove();
		this.container = null;
		this.onClose?.();
	}

	private openCsvPicker(): void {
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.csv';
		input.addEventListener('change', () => {
			const file = input.files?.[0];
			if (!file) return;
			const reader = new FileReader();
			reader.onload = () => {
				this.parseCsv(reader.result as string);
				this.renderCharts();
			};
			reader.readAsText(file);
		});
		input.click();
	}

	private parseCsv(text: string): void {
		const lines = text.trim().split('\n');
		if (lines.length < 2) return;
		this.data = [];
		for (let i = 1; i < lines.length; i++) {
			const cols = lines[i].split(',');
			if (cols.length < 8) continue;
			this.data.push({
				episode: parseFloat(cols[0]),
				timestep: parseFloat(cols[1]),
				reward: parseFloat(cols[2]),
				length: parseFloat(cols[3]),
				win: parseFloat(cols[4]),
				avgReward50: parseFloat(cols[5]),
				avgWinrate50: parseFloat(cols[6]),
				elapsedSec: parseFloat(cols[7]),
			});
		}
	}

	private renderCharts(): void {
		if (!this.container || this.data.length === 0) return;

		// Remove old charts
		this.container.querySelectorAll('.dashboard-chart').forEach((el) => el.remove());

		this.renderLineChart(
			'Avg Reward (50 ep)',
			this.data,
			(d) => d.timestep,
			(d) => d.avgReward50,
			'#4caf50'
		);
		this.renderLineChart(
			'Win Rate (50 ep)',
			this.data,
			(d) => d.timestep,
			(d) => d.avgWinrate50,
			'#2196f3'
		);
		this.renderLineChart(
			'Episode Length',
			this.data,
			(d) => d.timestep,
			(d) => d.length,
			'#ff9800'
		);
	}

	private renderLineChart(
		title: string,
		data: TrainingLogRow[],
		xFn: (d: TrainingLogRow) => number,
		yFn: (d: TrainingLogRow) => number,
		color: string
	): void {
		const wrapper = document.createElement('div');
		wrapper.className = 'dashboard-chart';
		wrapper.style.cssText = 'display:inline-block; margin:5px;';

		const margin = { top: 25, right: 15, bottom: 30, left: 50 };
		const width = 300;
		const height = 180;

		const svg = d3
			.select(wrapper)
			.append('svg')
			.attr('width', width + margin.left + margin.right)
			.attr('height', height + margin.top + margin.bottom);

		const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

		const filteredData = data.filter((d) => !isNaN(yFn(d)));

		const xScale = d3
			.scaleLinear()
			.domain(d3.extent(filteredData, xFn) as [number, number])
			.range([0, width]);

		const yScale = d3
			.scaleLinear()
			.domain(d3.extent(filteredData, yFn) as [number, number])
			.nice()
			.range([height, 0]);

		g.append('g')
			.attr('transform', `translate(0,${height})`)
			.call(
				d3
					.axisBottom(xScale)
					.ticks(5)
					.tickFormat((d) => `${Number(d) / 1000}k`)
			)
			.selectAll('text')
			.style('fill', '#aaa');
		g.selectAll('.domain, .tick line').style('stroke', '#555');

		g.append('g').call(d3.axisLeft(yScale).ticks(4)).selectAll('text').style('fill', '#aaa');

		const line = d3
			.line<TrainingLogRow>()
			.x((d) => xScale(xFn(d)))
			.y((d) => yScale(yFn(d)));

		g.append('path')
			.datum(filteredData)
			.attr('fill', 'none')
			.attr('stroke', color)
			.attr('stroke-width', 1.5)
			.attr('d', line);

		svg
			.append('text')
			.attr('x', (width + margin.left + margin.right) / 2)
			.attr('y', 16)
			.attr('text-anchor', 'middle')
			.attr('fill', '#ccc')
			.attr('font-size', '12px')
			.text(title);

		this.container?.appendChild(wrapper);
	}
}
