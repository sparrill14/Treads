/**
 * Determinism + performance benchmark for Simulation data-plumbing optimizations.
 *
 * Usage:
 *   npm run test           (compiles)
 *   node .test-dist/tests/benchmark.js [--baseline] [--verify] [--bench] [--profile]
 *
 * --baseline   Run 1000 ticks, print the serialized final-state hash (SHA-256).
 * --verify     Run 1000 ticks, compare hash to the stored baseline.
 * --bench      Run 10 000 ticks, report steps/sec.
 * --profile    Run 5000 ticks with profiling enabled, print phase breakdown.
 *
 * When no flag is given, all steps run in order.
 */

import { createHash } from 'node:crypto';
import { LEVEL_CONFIGS } from '../src/game/LevelConfig';
import { PassiveTankController } from '../src/game/controllers/ReplayController';
import { createMatchBootstrap } from '../src/game/core/MatchFactory';
import { Simulation } from '../src/game/core/Simulation';
import { serializeGameState } from '../src/game/core/stateUtils';

const LEVEL = LEVEL_CONFIGS[8]; // Same level as existing tests
const SEED = 42;
const DETERMINISM_TICKS = 1000;
const BENCH_TICKS = 10_000;
const PROFILE_TICKS = 5_000;

// Hardcoded baseline hash — set once, never changes.
let BASELINE_HASH = '';

function buildSimulation(opts: { debugFreeze?: boolean; profiling?: boolean } = {}): Simulation {
	const bootstrap = createMatchBootstrap(LEVEL, SEED, {
		playerController: new PassiveTankController(),
	});
	return new Simulation(bootstrap.initialState, bootstrap.controllers, opts);
}

function runTicks(sim: Simulation, ticks: number): void {
	for (let i = 0; i < ticks; i++) {
		sim.step();
	}
}

function hashState(sim: Simulation): string {
	const serialized = serializeGameState(sim.getStateSnapshot());
	return createHash('sha256').update(serialized).digest('hex');
}

// ── baseline ────────────────────────────────────────────────────────────
function baseline(): string {
	const sim = buildSimulation({ debugFreeze: true });
	runTicks(sim, DETERMINISM_TICKS);
	const hash = hashState(sim);
	console.log(`Baseline hash (${DETERMINISM_TICKS} ticks, seed=${SEED}): ${hash}`);
	return hash;
}

// ── verify ──────────────────────────────────────────────────────────────
function verify(expectedHash: string, label: string, opts: { debugFreeze?: boolean } = {}): void {
	const sim = buildSimulation({ debugFreeze: opts.debugFreeze ?? false });
	runTicks(sim, DETERMINISM_TICKS);
	const hash = hashState(sim);
	if (hash !== expectedHash) {
		throw new Error(`DETERMINISM FAILURE [${label}]: expected ${expectedHash}, got ${hash}`);
	}
	console.log(`  ✓ ${label} — hash matches`);
}

// ── bench ───────────────────────────────────────────────────────────────
function bench(label: string, opts: { debugFreeze?: boolean } = {}): number {
	// Warm up
	const warmSim = buildSimulation(opts);
	runTicks(warmSim, 200);

	const sim = buildSimulation(opts);
	const start = performance.now();
	runTicks(sim, BENCH_TICKS);
	const elapsed = performance.now() - start;
	const stepsPerSec = (BENCH_TICKS / elapsed) * 1000;
	console.log(`  ${label}: ${BENCH_TICKS} ticks in ${elapsed.toFixed(1)} ms — ${stepsPerSec.toFixed(0)} steps/sec`);
	return stepsPerSec;
}

// ── profile ─────────────────────────────────────────────────────────────
function profile(): void {
	const sim = buildSimulation({ debugFreeze: false, profiling: true });
	runTicks(sim, PROFILE_TICKS);
	sim.printProfilingReport();
}

// ── main ────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const runAll = args.length === 0;

function getBaseline(): string {
	if (!BASELINE_HASH) {
		BASELINE_HASH = baseline();
	}
	return BASELINE_HASH;
}

if (runAll || args.includes('--baseline')) {
	getBaseline();
}

if (runAll || args.includes('--verify')) {
	const hash = getBaseline();
	console.log('\nDeterminism checks:');
	verify(hash, 'freeze ON', { debugFreeze: true });
	verify(hash, 'freeze OFF', { debugFreeze: false });
}

if (runAll || args.includes('--bench')) {
	getBaseline();
	console.log('\nBenchmark:');
	const oldSpeed = bench('freeze ON  (unoptimized path)', { debugFreeze: true });
	const newSpeed = bench('freeze OFF (optimized path)', { debugFreeze: false });
	console.log(`\n  Speedup: ${(newSpeed / oldSpeed).toFixed(2)}x`);
}

if (runAll || args.includes('--profile')) {
	console.log('\nProfiling:');
	profile();
}
