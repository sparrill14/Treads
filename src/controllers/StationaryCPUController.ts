import { IController } from "./IController";
import { Ammunition, PlayerAmmunition, BasicAIAmmunition } from "../game/Ammunition";

export class StationaryCPUController implements IController {
    public aimAngle: number;
    public aimXPos: number;
    public aimYPos: number;
    public xOffset: number;
    public yOffset: number;
    public xPos: number;
    public yPos: number;
    public tankSize: number;
    public ammunition: BasicAIAmmunition[] = [];
    public maxAmmunition: number = 1;

    constructor(public canvas: HTMLCanvasElement, tankSize: number = 30) {
        this.tankSize = tankSize;
        this.aimAngle = 90;
        const canvasRect: DOMRect = canvas.getBoundingClientRect();
        this.xOffset = canvasRect.left;
        this.yOffset = canvasRect.top;
        this.xPos = canvas.width / 5;
        this.yPos = canvas.height / 2;

        for (let i = 0; i < this.maxAmmunition; i++) {
            this.ammunition.push(new BasicAIAmmunition(0, 0, 0, 0, 0, true))
        }

        setInterval(() => {
            const availableAmmunitionIndex = this.ammunition.findIndex(ammunition => ammunition.isDestroyed)
            if (availableAmmunitionIndex !== -1) {
                this.ammunition[availableAmmunitionIndex] = new BasicAIAmmunition(this.xPos + (this.tankSize / 2), this.yPos + (this.tankSize / 2), this.aimAngle, this.canvas.width, this.canvas.height, false);
            }
        }, 5000);
        // Set the initital awX and Y aim position to the center of the canvas
        this.aimXPos = canvas.width / 2;
        this.aimYPos = canvas.height / 2;
    }

    public aim(angle: number): void {
        this.aimAngle = angle;
    }

    public up(): boolean {
        return false;
    }

    public down(): boolean {
        return false;
    }

    public left(): boolean {
        return false;
    }

    public right(): boolean {
        return false;
    }
}