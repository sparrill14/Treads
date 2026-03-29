import { KeyStates } from './KeyStates';

export class InputManager {
	public keyStates: KeyStates = {
		ArrowUp: false,
		ArrowDown: false,
		ArrowLeft: false,
		ArrowRight: false,
		w: false,
		a: false,
		s: false,
		d: false,
		W: false,
		A: false,
		S: false,
		D: false,
	};

	public mouseX = 0;
	public mouseY = 0;
	public shootRequested = false;
	public bombRequested = false;

	private canvas: HTMLCanvasElement;
	private xOffset: number;
	private yOffset: number;

	private onKeyDown: (e: KeyboardEvent) => void;
	private onKeyUp: (e: KeyboardEvent) => void;
	private onMouseMove: (e: MouseEvent) => void;
	private onClick: (e: MouseEvent) => void;

	constructor(canvas: HTMLCanvasElement) {
		this.canvas = canvas;
		const rect = canvas.getBoundingClientRect();
		this.xOffset = rect.left;
		this.yOffset = rect.top;

		this.onKeyDown = (event: KeyboardEvent) => {
			if (Object.prototype.hasOwnProperty.call(this.keyStates, event.key)) {
				this.keyStates[event.key] = true;
			}
			if (event.code === 'Space') {
				this.bombRequested = true;
			}
		};

		this.onKeyUp = (event: KeyboardEvent) => {
			if (Object.prototype.hasOwnProperty.call(this.keyStates, event.key)) {
				this.keyStates[event.key] = false;
			}
		};

		this.onMouseMove = (event: MouseEvent) => {
			this.mouseX = event.clientX - this.xOffset;
			this.mouseY = event.clientY - this.yOffset;
		};

		this.onClick = (event: MouseEvent) => {
			if (this.canvas.contains(event.target as Node)) {
				this.shootRequested = true;
			}
		};

		document.addEventListener('keydown', this.onKeyDown);
		document.addEventListener('keyup', this.onKeyUp);
		document.addEventListener('mousemove', this.onMouseMove);
		document.addEventListener('click', this.onClick);
	}

	public up(): boolean {
		return this.keyStates.ArrowUp || this.keyStates.w || this.keyStates.W;
	}

	public down(): boolean {
		return this.keyStates.ArrowDown || this.keyStates.s || this.keyStates.S;
	}

	public left(): boolean {
		return this.keyStates.ArrowLeft || this.keyStates.a || this.keyStates.A;
	}

	public right(): boolean {
		return this.keyStates.ArrowRight || this.keyStates.d || this.keyStates.D;
	}

	public consumeShoot(): boolean {
		if (this.shootRequested) {
			this.shootRequested = false;
			return true;
		}
		return false;
	}

	public consumeBomb(): boolean {
		if (this.bombRequested) {
			this.bombRequested = false;
			return true;
		}
		return false;
	}

	public destroy(): void {
		document.removeEventListener('keydown', this.onKeyDown);
		document.removeEventListener('keyup', this.onKeyUp);
		document.removeEventListener('mousemove', this.onMouseMove);
		document.removeEventListener('click', this.onClick);
	}
}
