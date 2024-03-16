import { IController } from "./IController";
import { KeyStates } from "./KeyStates";
import { Ammunition, PlayerAmmunition } from "../game/Ammunition"

export class PlayerController implements IController {
    private keyStates: KeyStates = {
        ArrowUp: false,
        ArrowDown: false,
        ArrowLeft: false,
        ArrowRight: false,
        w: false,
        a: false,
        s: false,
        d: false
    }

    public aimAngle: number;
    public aimXPos: number;
    public aimYPos: number;
    public xOffset: number;
    public yOffset: number;
    public xPos: number;
    public yPos: number;
    public tankSize: number;
    public ammunition: PlayerAmmunition[] = [];
    public maxAmmunition: number = 5;

    constructor(canvas: HTMLCanvasElement, tankSize: number = 30) {
        this.tankSize = tankSize;
        this.aimAngle = 90;
        const canvasRect: DOMRect = canvas.getBoundingClientRect();
        this.xOffset = canvasRect.left;
        this.yOffset = canvasRect.top;

        this.xPos = canvas.width / 5;
        this.yPos = canvas.height / 2;
        // Set the initital awX and Y aim position to the center of the canvas
        this.aimXPos = canvas.width / 2;
        this.aimYPos = canvas.height / 2;

        for (let i = 0; i < this.maxAmmunition; i++) {
            this.ammunition.push(new PlayerAmmunition(0, 0, 0, 0, 0, true))
        }

        document.addEventListener('keydown', (event: KeyboardEvent) => {
            if (this.keyStates.hasOwnProperty(event.key)) {
                this.keyStates[event.key] = true;
            }
        });

        document.addEventListener('keyup', (event: KeyboardEvent) => {
            if (this.keyStates.hasOwnProperty(event.key)) {
                this.keyStates[event.key] = false;
            }
        });

        canvas.addEventListener('mousemove', (event: MouseEvent) => {
            this.aimXPos = event.clientX - this.xOffset;
            this.aimYPos = event.clientY - this.yOffset;
        });

        canvas.addEventListener('click', (event: MouseEvent) => {
            const availableAmmunitionIndex = this.ammunition.findIndex(ammunition => ammunition.isDestroyed)
            if (availableAmmunitionIndex !== -1) {
                this.ammunition[availableAmmunitionIndex] = new PlayerAmmunition(this.xPos + (tankSize / 2), this.yPos + (tankSize / 2), this.aimAngle, canvas.width, canvas.height, false);
            }
        });
    }

    public aim(angle: number): void {
        this.aimAngle = angle;
    }

    public up(): boolean {
        return this.keyStates.ArrowUp || this.keyStates.w;
    }

    public down(): boolean {
        return this.keyStates.ArrowDown || this.keyStates.s;
    }

    public left(): boolean {
        return this.keyStates.ArrowLeft || this.keyStates.a;
    }

    public right(): boolean {
        return this.keyStates.ArrowRight || this.keyStates.d;
    }
}
