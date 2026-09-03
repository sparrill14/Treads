import express, { type Express, type Request, type Response } from 'express';
import fs from 'fs';
import path from 'path';

const PORT = Number(process.env.TREADS_DASHBOARD_PORT ?? 3007);
const ROOT_RUN_ID = '__root__';
const REPO_ROOT = process.cwd();
const OUTPUT_ROOT = process.env.TREADS_OUTPUT_ROOT
	? path.resolve(process.env.TREADS_OUTPUT_ROOT)
	: path.join(REPO_ROOT, 'training', 'output');
const DIST_DIR = path.join(REPO_ROOT, 'dist');

interface MetricsSnapshot {
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

interface RunSummary {
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

interface ReplaySummary {
	fileName: string;
	runId: string;
	worker: number | null;
	episode: number | null;
	level: number | null;
	seed: number | null;
	outcome: string;
	sizeBytes: number;
	updatedAt: string;
	updatedAtMs: number;
	downloadUrl: string;
}

interface ModelStatus {
	available: boolean;
	runId: string | null;
	filePath: string | null;
	updatedAt: string | null;
	updatedAtMs: number | null;
	sizeBytes: number;
	url: string | null;
}

function fileExists(filePath: string): boolean {
	try {
		return fs.existsSync(filePath);
	} catch {
		return false;
	}
}

function readJsonIfExists(filePath: string): Record<string, unknown> | null {
	if (!fileExists(filePath)) {
		return null;
	}
	try {
		return JSON.parse(fs.readFileSync(filePath, 'utf-8')) as Record<string, unknown>;
	} catch {
		return null;
	}
}

function isCompatibleModel(modelPath: string): boolean {
	if (!fileExists(modelPath)) return false;
	const expected = readJsonIfExists(
		path.join(REPO_ROOT, 'src', 'game', 'controllers', 'neural-model-contract.json')
	);
	const actual = readJsonIfExists(modelPath.replace(/\.onnx$/i, '.contract.json'));
	return (
		expected !== null &&
		actual !== null &&
		actual.contractVersion === expected.contractVersion &&
		actual.observationVersion === expected.observationVersion &&
		actual.actionVersion === expected.actionVersion &&
		JSON.stringify(actual.observation) === JSON.stringify(expected.observation) &&
		JSON.stringify(actual.action) === JSON.stringify(expected.action)
	);
}

function parseNumber(value: unknown): number | null {
	if (value === null || value === undefined || value === '') {
		return null;
	}
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : null;
}

function parseCsvLine(line: string): string[] {
	const cols: string[] = [];
	let current = '';
	let inQuotes = false;

	for (let i = 0; i < line.length; i += 1) {
		const char = line[i];
		if (char === '"') {
			const next = line[i + 1];
			if (inQuotes && next === '"') {
				current += '"';
				i += 1;
			} else {
				inQuotes = !inQuotes;
			}
			continue;
		}
		if (char === ',' && !inQuotes) {
			cols.push(current);
			current = '';
			continue;
		}
		current += char;
	}

	cols.push(current);
	return cols;
}

function parseScenarioList(value: unknown): number[] {
	if (Array.isArray(value)) {
		return value.map((entry) => Number(entry)).filter((entry) => Number.isFinite(entry));
	}
	if (typeof value !== 'string' || value.trim() === '') {
		return [];
	}
	return value
		.split(',')
		.map((entry) => Number(entry.trim()))
		.filter((entry) => Number.isFinite(entry));
}

function normalizeMetricRecord(record: Record<string, unknown>): MetricsSnapshot {
	const numericMetrics: Record<string, number> = {};
	for (const [key, value] of Object.entries(record)) {
		const parsed = parseNumber(value);
		if (parsed !== null) {
			numericMetrics[key] = parsed;
		}
	}

	const pickMetric = (...keys: string[]): number | null => {
		for (const key of keys) {
			if (numericMetrics[key] !== undefined) {
				return numericMetrics[key];
			}
		}
		return null;
	};

	return {
		timestamp:
			typeof record.timestamp === 'string'
				? record.timestamp
				: typeof record.updatedAt === 'string'
					? record.updatedAt
					: null,
		iteration: pickMetric('iteration'),
		timesteps: pickMetric('timesteps'),
		episodes: pickMetric('episodes'),
		curriculumPhase:
			typeof record.curriculumPhase === 'string'
				? record.curriculumPhase
				: typeof record.curriculum_phase === 'string'
					? record.curriculum_phase
					: '',
		currentPhaseIndex: pickMetric('currentPhaseIndex'),
		activeScenarios: parseScenarioList(record.activeScenarios ?? record.active_scenarios),
		rehearsalScenarios: parseScenarioList(record.rehearsalScenarios ?? record.rehearsal_scenarios),
		phaseRecentWinrate500: pickMetric('phaseRecentWinrate500', 'phase_recent_winrate_500'),
		avgReward50: pickMetric('avgReward50', 'avg_reward_50'),
		avgWinrate50: pickMetric('avgWinrate50', 'avg_winrate_50'),
		avgTick50: pickMetric('avgTick50', 'avg_tick_50'),
		avgHit50: pickMetric('avgHit50', 'avg_hit_50'),
		avgHurt50: pickMetric('avgHurt50', 'avg_hurt_50'),
		avgKill50: pickMetric('avgKill50', 'avg_kill_50'),
		avgDeath50: pickMetric('avgDeath50', 'avg_death_50'),
		avgTerminalWin50: pickMetric('avgTerminalWin50', 'avg_terminal_win_50'),
		avgTerminalLoss50: pickMetric('avgTerminalLoss50', 'avg_terminal_loss_50'),
		avgTimeout50: pickMetric('avgTimeout50', 'avg_timeout_50'),
		avgApproach50: pickMetric('avgApproach50', 'avg_approach_50'),
		avgDodge50: pickMetric('avgDodge50', 'avg_dodge_50'),
		avgAimJitter50: pickMetric('avgAimJitter50', 'avg_aim_jitter_50'),
		avgMoveJitter50: pickMetric('avgMoveJitter50', 'avg_move_jitter_50'),
		stepsPerSec: pickMetric('stepsPerSec', 'steps_per_sec'),
		elapsedSec: pickMetric('elapsedSec', 'elapsed_sec'),
		bestWinRate: pickMetric('bestWinRate'),
		replayCount: pickMetric('replayCount'),
		onnxAvailable:
			record.onnxAvailable === true ||
			record.onnxAvailable === 'true' ||
			record.onnx_available === true ||
			record.onnx_available === 'true',
		metrics: numericMetrics,
	};
}

function readMetricsHistory(runDir: string): MetricsSnapshot[] {
	const jsonlPath = path.join(runDir, 'metrics_history.jsonl');
	if (fileExists(jsonlPath)) {
		return fs
			.readFileSync(jsonlPath, 'utf-8')
			.split(/\r?\n/)
			.map((line) => line.trim())
			.filter(Boolean)
			.map((line) => {
				try {
					return normalizeMetricRecord(JSON.parse(line) as Record<string, unknown>);
				} catch {
					return null;
				}
			})
			.filter((entry): entry is MetricsSnapshot => entry !== null);
	}

	const livePath = path.join(runDir, 'live_metrics.json');
	const liveMetrics = readJsonIfExists(livePath);
	if (liveMetrics) {
		return [normalizeMetricRecord(liveMetrics)];
	}

	const csvPath = path.join(runDir, 'training_log.csv');
	if (!fileExists(csvPath)) {
		return [];
	}
	const lines = fs
		.readFileSync(csvPath, 'utf-8')
		.split(/\r?\n/)
		.filter((line) => line.trim().length > 0);
	if (lines.length < 2) {
		return [];
	}
	const headers = parseCsvLine(lines[0]);
	return lines.slice(1).map((line) => {
		const cols = parseCsvLine(line);
		const row: Record<string, unknown> = {};
		headers.forEach((header, index) => {
			row[header] = cols[index] ?? '';
		});
		return normalizeMetricRecord(row);
	});
}

function parseReplayFileName(fileName: string): {
	worker: number | null;
	episode: number | null;
	level: number | null;
	seed: number | null;
	outcome: string;
} {
	const workerMatch = /^episode_(\d+)_W(\d+)_L(\d+)_S(\d+)_([^.]+)\.json$/i.exec(fileName);
	if (workerMatch) {
		return {
			worker: Number(workerMatch[2]),
			episode: Number(workerMatch[1]),
			level: Number(workerMatch[3]),
			seed: Number(workerMatch[4]),
			outcome: workerMatch[5],
		};
	}

	const legacyMatch = /^episode_(\d+)_L(\d+)_S(\d+)_([^.]+)\.json$/i.exec(fileName);
	if (legacyMatch) {
		return {
			worker: null,
			episode: Number(legacyMatch[1]),
			level: Number(legacyMatch[2]),
			seed: Number(legacyMatch[3]),
			outcome: legacyMatch[4],
		};
	}
	return {
		worker: null,
		episode: null,
		level: null,
		seed: null,
		outcome: 'unknown',
	};
}

function listReplayFiles(runDir: string, runId: string): ReplaySummary[] {
	const replayDir = path.join(runDir, 'replays');
	if (!fileExists(replayDir)) {
		return [];
	}
	return fs
		.readdirSync(replayDir, { withFileTypes: true })
		.filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
		.map((entry) => {
			const fullPath = path.join(replayDir, entry.name);
			const stats = fs.statSync(fullPath);
			const meta = parseReplayFileName(entry.name);
			return {
				fileName: entry.name,
				runId,
				worker: meta.worker,
				episode: meta.episode,
				level: meta.level,
				seed: meta.seed,
				outcome: meta.outcome,
				sizeBytes: stats.size,
				updatedAt: stats.mtime.toISOString(),
				updatedAtMs: stats.mtimeMs,
				downloadUrl: `/api/replay?run=${encodeURIComponent(runId)}&file=${encodeURIComponent(entry.name)}`,
			};
		})
		.sort((a, b) => b.updatedAtMs - a.updatedAtMs);
}

function directoryLooksLikeRun(dirPath: string): boolean {
	return (
		fileExists(path.join(dirPath, 'training_log.csv')) ||
		fileExists(path.join(dirPath, 'metrics_history.jsonl')) ||
		fileExists(path.join(dirPath, 'run_manifest.json')) ||
		fileExists(path.join(dirPath, 'live_metrics.json')) ||
		fileExists(path.join(dirPath, 'replays'))
	);
}

function listRunDirectories(): { id: string; label: string; dir: string; isRoot: boolean }[] {
	const runs: { id: string; label: string; dir: string; isRoot: boolean }[] = [];
	if (directoryLooksLikeRun(OUTPUT_ROOT)) {
		runs.push({
			id: ROOT_RUN_ID,
			label: 'Current Output',
			dir: OUTPUT_ROOT,
			isRoot: true,
		});
	}

	if (!fileExists(OUTPUT_ROOT)) {
		return runs;
	}

	for (const entry of fs.readdirSync(OUTPUT_ROOT, { withFileTypes: true })) {
		if (!entry.isDirectory() || entry.name === 'replays') {
			continue;
		}
		const dirPath = path.join(OUTPUT_ROOT, entry.name);
		if (!directoryLooksLikeRun(dirPath)) {
			continue;
		}
		runs.push({
			id: entry.name,
			label: entry.name,
			dir: dirPath,
			isRoot: false,
		});
	}

	return runs;
}

function getRunUpdatedMs(
	runDir: string,
	manifest: Record<string, unknown> | null,
	latestMetrics: MetricsSnapshot | null,
	replayCount: number
): number {
	const candidates: number[] = [];
	for (const fileName of [
		'run_manifest.json',
		'live_metrics.json',
		'training_log.csv',
		'metrics_history.jsonl',
		'treads_policy.onnx',
	]) {
		const filePath = path.join(runDir, fileName);
		if (fileExists(filePath)) {
			candidates.push(fs.statSync(filePath).mtimeMs);
		}
	}
	const replayDir = path.join(runDir, 'replays');
	if (fileExists(replayDir) && replayCount > 0) {
		candidates.push(fs.statSync(replayDir).mtimeMs);
	}
	if (typeof manifest?.updatedAt === 'string') {
		const parsed = Date.parse(manifest.updatedAt);
		if (Number.isFinite(parsed)) {
			candidates.push(parsed);
		}
	}
	if (latestMetrics?.timestamp) {
		const parsed = Date.parse(latestMetrics.timestamp);
		if (Number.isFinite(parsed)) {
			candidates.push(parsed);
		}
	}
	return candidates.length > 0 ? Math.max(...candidates) : Date.now();
}

function getRunSummary(run: { id: string; label: string; dir: string; isRoot: boolean }): RunSummary {
	const manifest = readJsonIfExists(path.join(run.dir, 'run_manifest.json'));
	const history = readMetricsHistory(run.dir);
	const latestMetrics = history.length > 0 ? history[history.length - 1] : null;
	const replayCount = listReplayFiles(run.dir, run.id).length;
	const updatedAtMs = getRunUpdatedMs(run.dir, manifest, latestMetrics, replayCount);
	const liveMetricsPath = path.join(run.dir, 'live_metrics.json');
	const liveUpdatedAtMs = fileExists(liveMetricsPath) ? fs.statSync(liveMetricsPath).mtimeMs : 0;
	const manifestPath = path.join(run.dir, 'run_manifest.json');
	const manifestUpdatedAtMs = fileExists(manifestPath) ? fs.statSync(manifestPath).mtimeMs : 0;
	const manifestStatus = typeof manifest?.status === 'string' ? manifest.status : '';
	const reportsActive = manifestStatus === 'running' || manifestStatus === 'initializing';
	const isRunning = reportsActive && Date.now() - Math.max(liveUpdatedAtMs, manifestUpdatedAtMs) < 45_000;
	return {
		id: run.id,
		label: run.label,
		isRoot: run.isRoot,
		path: path.relative(OUTPUT_ROOT, run.dir) || '.',
		status: reportsActive && !isRunning ? 'stale' : manifestStatus || 'complete',
		isRunning,
		startedAt: typeof manifest?.startedAt === 'string' ? manifest.startedAt : null,
		finishedAt: typeof manifest?.finishedAt === 'string' ? manifest.finishedAt : null,
		updatedAt: new Date(updatedAtMs).toISOString(),
		updatedAtMs,
		replayCount,
		onnxAvailable: isCompatibleModel(path.join(run.dir, 'treads_policy.onnx')),
		latestMetrics,
		manifest,
	};
}

function scanRuns(): { runs: RunSummary[]; activeRunId: string | null } {
	const runs = listRunDirectories()
		.map((run) => getRunSummary(run))
		.sort((a, b) => b.updatedAtMs - a.updatedAtMs);
	const activeRun = runs.find((run) => run.isRunning) ?? runs[0] ?? null;
	return {
		runs,
		activeRunId: activeRun ? activeRun.id : null,
	};
}

function resolveRunDir(runId: string | null): string {
	if (!runId || runId === ROOT_RUN_ID) {
		return OUTPUT_ROOT;
	}
	return path.join(OUTPUT_ROOT, runId);
}

function resolveCurrentModel(preferredRunId: string | null = null): ModelStatus {
	const { runs, activeRunId } = scanRuns();
	const candidates = (preferredRunId ? [preferredRunId] : [activeRunId, ...runs.map((run) => run.id)]).filter(
		(runId): runId is string => typeof runId === 'string' && runId.length > 0
	);
	const seen = new Set<string>();

	for (const runId of candidates) {
		if (seen.has(runId)) {
			continue;
		}
		seen.add(runId);
		const filePath = path.join(resolveRunDir(runId), 'treads_policy.onnx');
		if (!isCompatibleModel(filePath)) {
			continue;
		}
		const stats = fs.statSync(filePath);
		return {
			available: true,
			runId,
			filePath,
			updatedAt: stats.mtime.toISOString(),
			updatedAtMs: stats.mtimeMs,
			sizeBytes: stats.size,
			url: `/api/model/current.onnx?v=${Math.round(stats.mtimeMs)}`,
		};
	}

	const fallbackPath = path.join(OUTPUT_ROOT, 'treads_policy.onnx');
	if (!preferredRunId && isCompatibleModel(fallbackPath)) {
		const stats = fs.statSync(fallbackPath);
		return {
			available: true,
			runId: ROOT_RUN_ID,
			filePath: fallbackPath,
			updatedAt: stats.mtime.toISOString(),
			updatedAtMs: stats.mtimeMs,
			sizeBytes: stats.size,
			url: `/api/model/current.onnx?v=${Math.round(stats.mtimeMs)}`,
		};
	}

	return {
		available: false,
		runId: null,
		filePath: null,
		updatedAt: null,
		updatedAtMs: null,
		sizeBytes: 0,
		url: null,
	};
}

function livePayload(requestedRunId: string | null): {
	timestamp: string;
	activeRunId: string | null;
	selectedRunId: string | null;
	run: RunSummary | null;
	model: ModelStatus;
} {
	const { runs, activeRunId } = scanRuns();
	const selectedRunId = requestedRunId || activeRunId || (runs[0] ? runs[0].id : null);
	return {
		timestamp: new Date().toISOString(),
		activeRunId,
		selectedRunId,
		run: runs.find((run) => run.id === selectedRunId) ?? null,
		model: resolveCurrentModel(selectedRunId),
	};
}

const app: Express = express();
app.use(express.json());

app.get('/api/runs', (_req: Request, res: Response) => {
	const { runs, activeRunId } = scanRuns();
	res.json({
		serverTime: new Date().toISOString(),
		activeRunId,
		model: resolveCurrentModel(activeRunId),
		runs,
	});
});

app.get('/api/metrics', (req: Request, res: Response) => {
	const runId = typeof req.query.run === 'string' ? req.query.run : ROOT_RUN_ID;
	const runDir = resolveRunDir(runId);
	if (!directoryLooksLikeRun(runDir)) {
		res.status(404).json({ error: `Run not found: ${runId}` });
		return;
	}
	res.json({
		runId,
		history: readMetricsHistory(runDir),
	});
});

app.get('/api/replays', (req: Request, res: Response) => {
	const runId = typeof req.query.run === 'string' ? req.query.run : ROOT_RUN_ID;
	const runDir = resolveRunDir(runId);
	if (!directoryLooksLikeRun(runDir)) {
		res.status(404).json({ error: `Run not found: ${runId}` });
		return;
	}
	res.json({
		runId,
		replays: listReplayFiles(runDir, runId),
	});
});

app.get('/api/replay', (req: Request, res: Response) => {
	const runId = typeof req.query.run === 'string' ? req.query.run : ROOT_RUN_ID;
	const fileName = typeof req.query.file === 'string' ? req.query.file : '';
	if (!fileName) {
		res.status(400).json({ error: 'Missing replay file.' });
		return;
	}
	const replayPath = path.join(resolveRunDir(runId), 'replays', path.basename(fileName));
	if (!fileExists(replayPath)) {
		res.status(404).json({ error: `Replay not found: ${fileName}` });
		return;
	}
	res.sendFile(replayPath);
});

app.get('/api/model', (req: Request, res: Response) => {
	const runId = typeof req.query.run === 'string' ? req.query.run : null;
	res.json(resolveCurrentModel(runId));
});

app.get('/api/model/current.onnx', (req: Request, res: Response) => {
	const runId = typeof req.query.run === 'string' ? req.query.run : null;
	const model = resolveCurrentModel(runId);
	if (!model.available || !model.filePath) {
		res.status(404).json({ error: 'No ONNX model is available yet.' });
		return;
	}
	res.sendFile(model.filePath);
});

app.get('/api/model/current.contract.json', (req: Request, res: Response) => {
	const runId = typeof req.query.run === 'string' ? req.query.run : null;
	const model = resolveCurrentModel(runId);
	if (!model.available || !model.filePath) {
		res.status(404).json({ error: 'No ONNX model contract is available yet.' });
		return;
	}
	const contractPath = model.filePath.replace(/\.onnx$/i, '.contract.json');
	if (!fileExists(contractPath)) {
		res.status(409).json({ error: 'The selected model predates the versioned model contract.' });
		return;
	}
	res.sendFile(contractPath);
});

app.get('/api/live', (req: Request, res: Response) => {
	const requestedRunId = typeof req.query.run === 'string' ? req.query.run : null;
	res.setHeader('Content-Type', 'text/event-stream');
	res.setHeader('Cache-Control', 'no-cache, no-transform');
	res.setHeader('Connection', 'keep-alive');

	const send = () => {
		res.write(`data: ${JSON.stringify(livePayload(requestedRunId))}\n\n`);
	};

	send();
	const interval = setInterval(send, 2_500);
	req.on('close', () => {
		clearInterval(interval);
		res.end();
	});
});

app.use(express.static(DIST_DIR));
app.get('/{*path}', (_req: Request, res: Response) => {
	res.sendFile(path.join(DIST_DIR, 'index.html'));
});

app.listen(PORT, () => {
	console.log(`Training dashboard server running at http://localhost:${PORT}`);
	console.log(`Reading runs from ${OUTPUT_ROOT}`);
});
