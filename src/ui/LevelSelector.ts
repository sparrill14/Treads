import * as d3 from 'd3';
import packageJson from '../../package.json';
import { Level, Level1, Level2, Level3, Level4, Level5, Level6 } from '../game/Level';
import { AudioManager } from '../game/AudioManager';

export class LevelSelector {
    private levels: number;
    private activeLevelNumber: number;
    private activeLevel: Level;
    private sliderWidth: number = Math.min(window.innerWidth * 0.8, 600); // Responsive width
    private audioManager: AudioManager;

    constructor(levels: number) {
        this.levels = levels;
        this.audioManager = new AudioManager();
        let audioPromise: Promise<void[]> = this.audioManager.loadAllAudio();
        audioPromise.then((): void => {
            this.audioManager.playBackgroundMusic();
        })
        this.activeLevelNumber = 1;
        this.activeLevel = new Level1(this.audioManager);
        this.setHeader();
        this.createSlider();
        this.createJumbotron();
    }

    public setHeader() {
        let header: HTMLElement | null = document.getElementById('main-header')
        if (header) {
            header.textContent = `Treads V${packageJson.version}`
        }
    }

    public startActiveLevel() {
        this.activeLevel.stop();
        switch (this.activeLevelNumber) {
            case 1:
                this.activeLevel = new Level1(this.audioManager);
                break;
            case 2:
                this.activeLevel = new Level2(this.audioManager);
                break;
            case 3:
                this.activeLevel = new Level3(this.audioManager);
                break;
            case 4:
                this.activeLevel = new Level4(this.audioManager);
                break;
            case 5:
                this.activeLevel = new Level5(this.audioManager);
                break;
            case 6:
                this.activeLevel = new Level6(this.audioManager);
                break;
            default:
                this.activeLevel = new Level1(this.audioManager);
                break;
        }
        this.activeLevel.start();
    }

    private createSlider(): void {
        const margin: { top: number; right: number; bottom: number; left: number; } = { top: 10, right: 10, bottom: 20, left: 10 };
        const effectiveWidth: number = this.sliderWidth - margin.left - margin.right;
    
        const scale: d3.ScaleLinear<number, number, never> = d3.scaleLinear()
            .domain([1, this.levels])
            .range([0, effectiveWidth])
            .clamp(true);
    
        const svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any> = d3.select('#slider').append('svg')
            .attr('width', this.sliderWidth)
            .attr('height', 50);
    
        const sliderGroup: d3.Selection<SVGGElement, unknown, HTMLElement, any> = svg.append('g')
            .attr('transform', `translate(${margin.left}, 30)`);
    
        sliderGroup.append('g')
            .call(d3.axisBottom(scale).ticks(this.levels).tickFormat(d3.format('1')));
    
        const handle: d3.Selection<SVGCircleElement, unknown, HTMLElement, any> = sliderGroup.append('circle')
            .attr('cx', scale(this.activeLevelNumber))
            .attr('cy', -10)
            .attr('r', 10)
            .style('fill', 'red')
            .style('cursor', 'ew-resize');
        const dragHandler: d3.DragBehavior<SVGCircleElement, unknown, unknown> = d3.drag<SVGCircleElement, unknown>()
            .on('drag', (event) => {
                const x = event.x - margin.left;  // Adjusting for the left margin
                const level = Math.round(scale.invert(x));
                handle.attr('cx', scale(level));
                this.updateActiveLevel(level);
            });
    
        handle.call(dragHandler);
    }

    private createJumbotron(): void {
        const jumbotron: d3.Selection<d3.BaseType, unknown, HTMLElement, any> = d3.select('#jumbotron');
        for (let i = 1; i <= this.levels; i++) {
            const box: d3.Selection<HTMLDivElement, unknown, HTMLElement, any> = jumbotron.append('div')
                .attr('class', 'jumbotron-box inactive')
                .on('click', () => this.updateActiveLevel(i));

            // Example of how you can append an SVG to a jumbotron box.
            const svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any> = box.append('svg')
                .attr('width', '100%')
                .attr('height', '100%');

            svg.append('rect') // Placeholder for actual SVG content.
                .attr('width', '100%')
                .attr('height', '100%')
                .attr('fill', 'transparent');
            svg.append('text')
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
        d3.select('circle').attr('cx', d3.scaleLinear()
            .domain([1, this.levels])
            .range([0, this.sliderWidth - 20])(level));
        this.startActiveLevel();
    }
}
