import { Ammunition } from "../game/Ammunition";

export interface IController {
    aimAngle: number;
    aimXPos: number;
    aimYPos: number;
    xPos: number;
    yPos: number;
    tankSize: number;
    ammunition: Ammunition[];

    up: () => boolean;
    down: () => boolean;
    left: () => boolean;
    right: () => boolean;
    aim: (angle: number) => void;
}
