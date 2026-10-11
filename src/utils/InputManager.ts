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

	private onKeyDown: (e: KeyboardEvent) => void;
	private onKeyUp: (e: KeyboardEvent) => void;
	private onMouseMove: (e: MouseEvent) => void;
	private onClick: (e: MouseEvent) => void;
	private readonly clearInput = (): void => {
		for (const key of Object.keys(this.keyStates)) this.keyStates[key] = false;
		this.shootRequested = false;
		this.bombRequested = false;
	};

	constructor(canvas: HTMLCanvasElement) {
		this.canvas = canvas;
		this.canvas.tabIndex = 0;

		this.onKeyDown = (event: KeyboardEvent) => {
			if (document.activeElement !== this.canvas) return;
			if (Object.prototype.hasOwnProperty.call(this.keyStates, event.key)) {
				event.preventDefault();
				this.keyStates[event.key] = true;
			}
			if (event.code === 'Space') {
				event.preventDefault();
				if (!event.repeat) this.bombRequested = true;
			}
		};

		this.onKeyUp = (event: KeyboardEvent) => {
			if (Object.prototype.hasOwnProperty.call(this.keyStates, event.key)) {
				this.keyStates[event.key] = false;
			}
		};

		this.onMouseMove = (event: MouseEvent) => {
			if (!this.canvas) return;
			const rect = this.canvas.getBoundingClientRect();
			const localX = event.clientX - rect.left;
			const localY = event.clientY - rect.top;
			const scaleX = rect.width > 0 ? this.canvas.width / rect.width : 1;
			const scaleY = rect.height > 0 ? this.canvas.height / rect.height : 1;
			this.mouseX = localX * scaleX;
			this.mouseY = localY * scaleY;
		};

		this.onClick = (event: MouseEvent) => {
			if (this.canvas && this.canvas.contains(event.target as Node)) {
				this.canvas.focus({ preventScroll: true });
				this.onMouseMove(event);
				this.shootRequested = true;
			}
		};

		document.addEventListener('keydown', this.onKeyDown);
		document.addEventListener('keyup', this.onKeyUp);
		document.addEventListener('mousemove', this.onMouseMove);
		document.addEventListener('click', this.onClick);
		window.addEventListener('blur', this.clearInput);
		this.canvas.addEventListener('blur', this.clearInput);
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
		window.removeEventListener('blur', this.clearInput);
		this.canvas.removeEventListener('blur', this.clearInput);
	}
}
