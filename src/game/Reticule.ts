export class Reticule {
    protected color: string;
    protected dashPattern: [number, number]; // [dashLengthInPixels, spaceLengthInPixels]
    protected renderReticule: boolean;
    protected tankSize: number;
    protected reticuleWidth: number = 5;

    constructor (dashPattern: [number, number], color: string, renderReticule: boolean, tankSize: number) {
        this.color = color;
        this.dashPattern = dashPattern;
        this.renderReticule = renderReticule;
        this.tankSize = tankSize;
    }

    public draw(context: CanvasRenderingContext2D, tankXPos: number, tankYPos: number, mouseXPos: number, mouseYpos: number): void {
        if (this.renderReticule) {
            context.strokeStyle = this.color;
            context.setLineDash(this.dashPattern);
            context.beginPath();
            context.moveTo(tankXPos + (this.tankSize / 2), tankYPos + (this.tankSize / 2));
            context.lineTo(mouseXPos, mouseYpos);
            context.lineWidth = this.reticuleWidth
            context.stroke();
        }
    }
}

export class NoReticule extends Reticule {
    constructor() {
        super([0, 0], 'blue', false, 0);
    }
}

export class SimplePlayerReticule extends Reticule {
    constructor(tankSize: number) {
        super([5, 5], 'blue', true, tankSize);
    }
}

export class CustomColorReticule extends Reticule {
    constructor(tankSize: number, color: string) {
        super([4, 1], color, true, tankSize);
    }
}

export class AdjustingCustomColorReticule extends Reticule {
    private maxReticuleLength: number;

    constructor (tankSize: number, color: string, maxReticuleLength: number) {
        super([1, 1], color, true, tankSize);
        this.maxReticuleLength = maxReticuleLength;
    }

    override draw(context: CanvasRenderingContext2D, tankXPos: number, tankYPos: number, mouseXPos: number, mouseYpos: number): void {
        // Calculate distance between tank and mouse
        const xDist = mouseXPos - tankXPos;
        const yDist = mouseYpos - tankYPos;
        const distance = Math.sqrt(Math.pow(xDist, 2) + Math.pow(yDist, 2));

        // map distance / maxReticuleLength to the range [1, 10] and [10, 1]
        const distanceToMaxDistanceRatio = distance / this.maxReticuleLength;
        const distanceToMaxDistanceRatioInverse = 1 - distanceToMaxDistanceRatio;
        let dashLength = distanceToMaxDistanceRatio * 10 + 2;
        let spaceLength = distanceToMaxDistanceRatioInverse * 10 + 5;

        if (this.renderReticule) {
            context.strokeStyle = this.color;
            context.setLineDash([dashLength, spaceLength]);
            context.beginPath();
            context.moveTo(tankXPos + (this.tankSize / 2), tankYPos + (this.tankSize / 2),);
            context.lineTo(mouseXPos, mouseYpos);
            context.lineWidth = this.reticuleWidth
            context.stroke();
        }
    }
}
