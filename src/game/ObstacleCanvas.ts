import { Obstacle } from './Obstacle';

export class ObstacleCanvas {
    private obstacleCanvas: HTMLCanvasElement;
    public obstacles: Obstacle[] = [];
    public width: number;
    public height: number;

    constructor(obstacleCanvasSelector: string, width: number, height: number, obstacles: Obstacle[]) {
        this.width = width;
        this.height = height;
        this.obstacles = obstacles;
        window.addEventListener('resize', this.resizeCanvas.bind(this));
        this.obstacleCanvas = document.querySelector(obstacleCanvasSelector) as HTMLCanvasElement
        this.obstacleCanvas.width = width;
        this.obstacleCanvas.height = height;
        obstacles.forEach(obstacle => {
            obstacle.draw(this.obstacleCanvas.getContext('2d') as CanvasRenderingContext2D);
        });
    }

    public clearObstacles(): void {
        const context = this.obstacleCanvas.getContext('2d');
        if (context) {
            context.clearRect(0, 0, this.obstacleCanvas.width, this.obstacleCanvas.height);
        }
        this.obstacles = [];
    }

    private resizeCanvas(): void {

    }
}
