export interface MetricsSnapshot {
	timestamp: string | null;
	iteration: number | null;
	timesteps: number | null;
	episodes: number | null;
	curriculumPhase: string;
	currentPhaseIndex: number | null;
	activeScenarios: number[];
	rehearsalScenarios: number[];
	phaseRecentWinrate500: number | null;
	avgReward50: number | null;
	avgWinrate50: number | null;
	avgTick50: number | null;
	avgHit50: number | null;
	avgHurt50: number | null;
	avgKill50: number | null;
	avgDeath50: number | null;
	avgTerminalWin50: number | null;
	avgTerminalLoss50: number | null;
	avgTimeout50: number | null;
	avgApproach50: number | null;
	avgDodge50: number | null;
	avgAimJitter50: number | null;
	avgMoveJitter50: number | null;
	stepsPerSec: number | null;
	elapsedSec: number | null;
	bestWinRate: number | null;
	replayCount: number | null;
	onnxAvailable: boolean;
	metrics: Record<string, number>;
}

export interface RunSummary {
	id: string;
	label: string;
	isRoot: boolean;
	path: string;
	status: string;
	isRunning: boolean;
	startedAt: string | null;
	finishedAt: string | null;
	updatedAt: string;
	updatedAtMs: number;
	replayCount: number;
	onnxAvailable: boolean;
	latestMetrics: MetricsSnapshot | null;
	manifest: Record<string, unknown> | null;
}

export interface ReplaySummary {
	fileName: string;
	runId: string;
	episode: number | null;
	level: number | null;
	seed: number | null;
	outcome: string;
	sizeBytes: number;
	updatedAt: string;
	updatedAtMs: number;
	downloadUrl: string;
}

export interface ModelStatus {
	available: boolean;
	runId: string | null;
	filePath: string | null;
	updatedAt: string | null;
	updatedAtMs: number | null;
	sizeBytes: number;
	url: string | null;
}

export interface RunsResponse {
	serverTime: string;
	activeRunId: string | null;
	model: ModelStatus;
	runs: RunSummary[];
}

export interface LivePayload {
	timestamp: string;
	activeRunId: string | null;
	selectedRunId: string | null;
	run: RunSummary | null;
	model: ModelStatus;
}

async function readJson<T>(url: string): Promise<T> {
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error(`${response.status} ${response.statusText}`);
	}
	return (await response.json()) as T;
}

export class DashboardApiClient {
	public async fetchRuns(): Promise<RunsResponse> {
		return readJson<RunsResponse>('/api/runs');
	}

	public async fetchMetrics(runId: string): Promise<MetricsSnapshot[]> {
		const response = await readJson<{ runId: string; history: MetricsSnapshot[] }>(
			`/api/metrics?run=${encodeURIComponent(runId)}`
		);
		return response.history;
	}

	public async fetchReplays(runId: string): Promise<ReplaySummary[]> {
		const response = await readJson<{ runId: string; replays: ReplaySummary[] }>(
			`/api/replays?run=${encodeURIComponent(runId)}`
		);
		return response.replays;
	}

	public async fetchReplayJson<T>(runId: string, fileName: string): Promise<T> {
		return readJson<T>(`/api/replay?run=${encodeURIComponent(runId)}&file=${encodeURIComponent(fileName)}`);
	}

	public async fetchModel(runId?: string): Promise<ModelStatus> {
		const query = runId ? `?run=${encodeURIComponent(runId)}` : '';
		return readJson<ModelStatus>(`/api/model${query}`);
	}

	public subscribeLive(runId: string | null, onMessage: (payload: LivePayload) => void): EventSource {
		const query = runId ? `?run=${encodeURIComponent(runId)}` : '';
		const eventSource = new EventSource(`/api/live${query}`);
		eventSource.onmessage = (event) => {
			const payload = JSON.parse(event.data) as LivePayload;
			onMessage(payload);
		};
		return eventSource;
	}
}
