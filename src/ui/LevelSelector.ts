import * as d3 from 'd3';
import packageJson from '../../package.json';
import { AudioManager } from '../game/AudioManager';
import { Level } from '../game/Level';
import { LEVEL_CONFIGS } from '../game/LevelConfig';
import { NeuralNetController } from '../game/controllers/NeuralNetController';
import type { TankController } from '../game/core/types';

export class LevelSelector {
	public static createHeadlessLevel(
		levelNumber: number,
		seed: number = levelNumber,
		playerController?: TankController
	): Level {
		const configIndex = Math.max(0, Math.min(levelNumber - 1, LEVEL_CONFIGS.length - 1));
		return new Level(LEVEL_CONFIGS[configIndex], { headless: true, seed, playerController });
	}
	private numLevels: number;
	private activeLevelNumber: number;
	private activeLevel: Level | null = null;
	private sliderWidth: number = Math.min(window.innerWidth * 0.8, 600);
	private audioManager: AudioManager;
	private aiMode = false;
	private neuralNetController: NeuralNetController | null = null;

	constructor(levels: number) {
		this.numLevels = levels;
		this.audioManager = new AudioManager();
		const audioPromise = this.audioManager.loadAllAudio();
		audioPromise.then((): void => {
			this.audioManager.playBackgroundMusic();
		});
		this.activeLevelNumber = 1;
		this.setHeader();
		this.createAiToggle();
		this.createSlider();
		this.createJumbotron();
	}

	public setHeader() {
		const header: HTMLElement | null = document.getElementById('main-header');
		if (header) {
			header.textContent = `Treads V${packageJson.version}`;
		}
	}

	public startActiveLevel() {
		this.activeLevel?.stop();
		const configIndex = Math.max(0, Math.min(this.activeLevelNumber - 1, LEVEL_CONFIGS.length - 1));
		const config = LEVEL_CONFIGS[configIndex];

		const controllerOverrides: Record<string, TankController> = {};
		if (this.aiMode && this.neuralNetController) {
			// Override all enemy controllers with the neural net controller
			const nnController = this.neuralNetController;
			config.enemies.forEach((_enemy, index) => {
				controllerOverrides[`enemy-${index}`] = nnController;
			});
		}

		this.activeLevel = new Level(LEVEL_CONFIGS[configIndex], {
			audioManager: this.audioManager,
			seed: this.activeLevelNumber,
			controllerOverrides,
		});
		this.activeLevel.start();
	}

	private createAiToggle(): void {
		const container = document.createElement('div');
		container.id = 'ai-toggle-container';
		container.style.cssText = 'text-align: center; margin: 10px 0;';

		const btn = document.createElement('button');
		btn.id = 'ai-toggle-btn';
		btn.textContent = 'AI Mode: OFF';
		btn.className = 'btn btn-outline-secondary btn-sm';
		btn.addEventListener('click', () => this.toggleAiMode(btn));

		container.appendChild(btn);

		const header = document.getElementById('main-header');
		if (header?.parentElement) {
			header.parentElement.insertBefore(container, header.nextSibling);
		}
	}

	private async toggleAiMode(btn: HTMLButtonElement): Promise<void> {
		if (!this.aiMode) {
			btn.textContent = 'AI Mode: Loading...';
			btn.disabled = true;
			try {
				if (!this.neuralNetController) {
					this.neuralNetController = new NeuralNetController('models/treads_policy.onnx');
					await this.neuralNetController.loadModel();
				}
				this.aiMode = true;
				btn.textContent = 'AI Mode: ON';
				btn.className = 'btn btn-success btn-sm';
			} catch (err) {
				console.error('Failed to load AI model:', err);
				btn.textContent = 'AI Mode: Error';
				btn.className = 'btn btn-danger btn-sm';
				setTimeout(() => {
					btn.textContent = 'AI Mode: OFF';
					btn.className = 'btn btn-outline-secondary btn-sm';
					btn.disabled = false;
				}, 2000);
				return;
			}
			btn.disabled = false;
		} else {
			this.aiMode = false;
			btn.textContent = 'AI Mode: OFF';
			btn.className = 'btn btn-outline-secondary btn-sm';
		}
		this.startActiveLevel();
	}

	private createSlider(): void {
		const margin = { top: 10, right: 10, bottom: 20, left: 10 };
		const effectiveWidth: number = this.sliderWidth - margin.left - margin.right;
		const scale: d3.ScaleLinear<number, number, never> = d3
			.scaleLinear()
			.domain([1, this.numLevels])
			.range([0, effectiveWidth])
			.clamp(true);
		const svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, undefined> = d3
			.select('#slider')
			.append('svg')
			.attr('width', this.sliderWidth)
			.attr('height', 50);
		const sliderGroup: d3.Selection<SVGGElement, unknown, HTMLElement, undefined> = svg
			.append('g')
			.attr('transform', `translate(${margin.left}, 30)`);
		sliderGroup.append('g').call(d3.axisBottom(scale).ticks(this.numLevels).tickFormat(d3.format('1')));
		const handle: d3.Selection<SVGCircleElement, unknown, HTMLElement, undefined> = sliderGroup
			.append('circle')
			.attr('cx', scale(this.activeLevelNumber))
			.attr('cy', -10)
			.attr('r', 10)
			.style('fill', 'red')
			.style('cursor', 'ew-resize');
		const dragHandler: d3.DragBehavior<SVGCircleElement, unknown, unknown> = d3
			.drag<SVGCircleElement, unknown>()
			.on('drag', (event) => {
				const x = event.x - margin.left;
				const level = Math.round(scale.invert(x));
				handle.attr('cx', scale(level));
				this.updateActiveLevel(level);
			});
		handle.call(dragHandler);
	}

	private createJumbotron(): void {
		const jumbotron: d3.Selection<d3.BaseType, unknown, HTMLElement, undefined> = d3.select('#jumbotron');
		const colorScale = d3.scaleLinear<string>().domain([1, this.numLevels]).range(['lightblue', 'lightcoral']);
		for (let i = 1; i <= this.numLevels; i++) {
			const box: d3.Selection<HTMLDivElement, unknown, HTMLElement, undefined> = jumbotron
				.append('div')
				.attr('class', 'jumbotron-box inactive')
				.on('click', () => this.updateActiveLevel(i));
			const svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, undefined> = box
				.append('svg')
				.attr('width', '100%')
				.attr('height', '100%');
			svg.append('rect').attr('width', '100%').attr('height', '100%').attr('fill', colorScale(i));
			svg
				.append('text')
				.attr('x', '50%')
				.attr('y', '50%')
				.attr('dominant-baseline', 'middle')
				.attr('text-anchor', 'middle')
				.text(`Level ${i}`);
		}
		this.updateActiveLevel(this.activeLevelNumber);
	}

	private updateActiveLevel(level: number): void {
		this.activeLevelNumber = level;
		d3.selectAll('.jumbotron-box')
			.classed('active', (_, i) => i + 1 === level)
			.classed('inactive', (_, i) => i + 1 !== level);
		d3.select('circle').attr(
			'cx',
			d3
				.scaleLinear()
				.domain([1, this.numLevels])
				.range([0, this.sliderWidth - 20])(level)
		);
		this.startActiveLevel();
	}
}
