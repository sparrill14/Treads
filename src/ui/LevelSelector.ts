import * as d3 from 'd3';
import { Level, Level1, Level2, Level3, Level4, Level5 } from '../game/Level';

export class LevelSelector {
    private levels: number;
    private activeLevelNumber: number;
    private activeLevel: Level
    private sliderWidth: number = Math.min(window.innerWidth * 0.8, 600); // Responsive width

    constructor(levels: number) {
        this.levels = levels;
        this.activeLevelNumber = 1;
        this.activeLevel = new Level1()
        this.createSlider();
        this.createJumbotron();
    }

    public startActiveLevel() {
        this.activeLevel.stop()
        switch (this.activeLevelNumber) {
            case 1:
                this.activeLevel = new Level1()
                break;
            case 2:
                this.activeLevel = new Level2()
                break;
            case 3:
                this.activeLevel = new Level3()
                break;
            case 4:
                this.activeLevel = new Level4()
                break;
            case 5:
                this.activeLevel = new Level5()
                break;
            default:
                this.activeLevel = new Level1()
                break;
        }
        this.activeLevel.start();
    }

    private createSlider(): void {
        const margin = { top: 10, right: 10, bottom: 20, left: 10 };
        const effectiveWidth = this.sliderWidth - margin.left - margin.right;
    
        const scale = d3.scaleLinear()
            .domain([1, this.levels])
            .range([0, effectiveWidth])
            .clamp(true);
    
        const svg = d3.select('#slider').append('svg')
            .attr('width', this.sliderWidth)
            .attr('height', 50);
    
        const sliderGroup = svg.append('g')
            .attr('transform', `translate(${margin.left}, 30)`);
    
        sliderGroup.append('g')
            .call(d3.axisBottom(scale).ticks(this.levels).tickFormat(d3.format('1')));
    
        const handle = sliderGroup.append('circle')
            .attr('cx', scale(this.activeLevelNumber))
            .attr('cy', -10)
            .attr('r', 10)
            .style('fill', 'red')
            .style('cursor', 'ew-resize');
        const dragHandler = d3.drag<SVGCircleElement, unknown>()
            .on('drag', (event) => {
                const x = event.x - margin.left;  // Adjusting for the left margin
                const level = Math.round(scale.invert(x));
                handle.attr('cx', scale(level));
                this.updateActiveLevel(level);
            });
    
        handle.call(dragHandler);
    }

    private createJumbotron(): void {
        const jumbotron = d3.select('#jumbotron');
        for (let i = 1; i <= this.levels; i++) {
            const box = jumbotron.append('div')
                .attr('class', 'jumbotron-box inactive')
                .on('click', () => this.updateActiveLevel(i));

            // Example of how you can append an SVG to a jumbotron box.
            const svg = box.append('svg')
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
