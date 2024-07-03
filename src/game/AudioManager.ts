import ammunitionExplode from '../assets/audio/ammunitionExplode.mp3';
import backgroundMusic from '../assets/audio/backgroundMusic.mp3';
import bombExplode from '../assets/audio/bombExplode.mp3';
import tankDestroy from '../assets/audio/tankDestroy.mp3';
import tankFire from '../assets/audio/tankFire.mp3';

export enum AudioFile {
	TANK_FIRE = 1,
	TANK_DESTROY = 2,
	BOMB_EXPLODE = 3,
	BACKGROUND_MUSIC = 4,
	AMMUNITION_EXPLODE = 5,
}

export class AudioManager {
	private audioContext: AudioContext;
	private audioBuffers: Map<AudioFile, AudioBuffer>;
	private backgroundMusicSource: AudioBufferSourceNode | null;

	constructor() {
		this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
		this.audioBuffers = new Map();
		this.backgroundMusicSource = null;
	}

	loadAllAudio(): Promise<void[]> {
		const promises = [
			this.loadAudio(AudioFile.TANK_FIRE, tankFire),
			this.loadAudio(AudioFile.TANK_DESTROY, tankDestroy),
			this.loadAudio(AudioFile.BOMB_EXPLODE, bombExplode),
			this.loadAudio(AudioFile.BACKGROUND_MUSIC, backgroundMusic),
			this.loadAudio(AudioFile.AMMUNITION_EXPLODE, ammunitionExplode),
		];
		return Promise.all(promises);
	}

	loadAudio(audioFile: AudioFile, url: string): Promise<void> {
		return fetch(url)
			.then((response) => response.arrayBuffer())
			.then((arrayBuffer) => this.audioContext.decodeAudioData(arrayBuffer))
			.then((audioBuffer) => {
				this.audioBuffers.set(audioFile, audioBuffer);
			})
			.catch((error) => console.error(`Error loading audio file: ${url}`, error));
	}

	play(audioFile: AudioFile): void {
		const buffer = this.audioBuffers.get(audioFile);
		if (buffer) {
			const source = this.audioContext.createBufferSource();
			source.buffer = buffer;
			source.connect(this.audioContext.destination);
			source.start(0);
		} else {
			console.warn(`Audio buffer for ${audioFile} not found`);
		}
	}

	getAudioContext(): AudioContext {
		return this.audioContext;
	}

	playBackgroundMusic(): void {
		const buffer = this.audioBuffers.get(AudioFile.BACKGROUND_MUSIC);
		if (buffer) {
			this.stopBackgroundMusic();
			this.backgroundMusicSource = this.audioContext.createBufferSource();
			this.backgroundMusicSource.buffer = buffer;
			this.backgroundMusicSource.loop = true;
			this.backgroundMusicSource.connect(this.audioContext.destination);
			this.backgroundMusicSource.start(0);
		} else {
			console.warn(`Background music buffer not found`);
		}
	}

	stopBackgroundMusic(): void {
		if (this.backgroundMusicSource) {
			this.backgroundMusicSource.stop(0);
			this.backgroundMusicSource.disconnect();
			this.backgroundMusicSource = null;
		}
	}
}
