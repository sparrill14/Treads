/**
 * Simple test that pipes no-op actions to the CLI runner to verify it works end-to-end.
 */
const { spawn } = require('child_process');
const path = require('path');

const runner = spawn('node', [
    path.join(__dirname, '..', '.training-dist', 'training', 'cli-runner.js'),
    '--level', '1',
    '--seed', '42',
    '--max-ticks', '300',
]);

let buffer = '';
let messageCount = 0;

runner.stdout.on('data', (data) => {
    buffer += data.toString();
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep incomplete line in buffer

    for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        messageCount++;

        if (msg.type === 'init') {
            console.log('INIT:', JSON.stringify({ level: msg.level, seed: msg.seed, arena: msg.arena, playerTankId: msg.playerTankId }));
        } else if (msg.type === 'observation') {
            // Send no-op action
            const action = JSON.stringify({ move: 'none', aimAngle: 0, fire: false, plantBomb: false }) + '\n';
            runner.stdin.write(action);
            if (msg.tick % 60 === 0) {
                console.log(`Tick ${msg.tick}: player at (${msg.observation.self.x.toFixed(0)}, ${msg.observation.self.y.toFixed(0)}), enemies alive: ${msg.observation.enemies.filter(e => !e.destroyed).length}`);
            }
        } else if (msg.type === 'result') {
            console.log('RESULT:', JSON.stringify(msg, null, 2));
            console.log(`Total messages: ${messageCount}`);
        }
    }
});

runner.stderr.on('data', (data) => {
    console.error('STDERR:', data.toString());
});

runner.on('close', (code) => {
    console.log(`Runner exited with code ${code}`);
});
