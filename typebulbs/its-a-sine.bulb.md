---
format: typebulb/v1
name: "It's a sine"
---

**code.tsx**

```tsx
import { Module, compile, Linear, mul, sub, mean, relu, type Tensor } from 'tensorgrad'

const HIDDEN = 64
const BATCH_SIZE = 256
const LR = 0.005

class MLP extends Module {
  l1 = new Linear(1, HIDDEN)
  l2 = new Linear(HIDDEN, HIDDEN)
  l3 = new Linear(HIDDEN, 1)
}

function modelFwd(p: MLP, x: Tensor): Tensor {
  return p.l3.fwd(relu(p.l2.fwd(relu(p.l1.fwd(x)))))
}

function lossFn(p: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  const diff = sub(modelFwd(p, x), y)
  return mean(mul(diff, diff))
}

function makeBatch(): { x: Float32Array; y: Float32Array } {
  const x = new Float32Array(BATCH_SIZE)
  const y = new Float32Array(BATCH_SIZE)
  for (let i = 0; i < BATCH_SIZE; i++) {
    const v = (Math.random() * 2 - 1) * Math.PI
    x[i] = v
    y[i] = Math.sin(v)
  }
  return { x, y }
}

// Build the training graph + a polymorphic-batch inference graph that shares
// its params. Returns both plus the timestamped compile duration so the UI
// can show how long it took.
async function buildGraphs() {
  const t0 = performance.now()
  const model = new MLP()
  const train = await compile({
    model,
    loss: lossFn,
    optimizer: { kind: 'adam', lr: LR },
    inputs: { x: [BATCH_SIZE, 1], y: [BATCH_SIZE, 1] },
  })
  const infer = await train.attach({
    forward: (m, { x }) => modelFwd(m, x),
    inputs: { x: [null, 1] },
  })
  return { train, infer, compileMs: performance.now() - t0 }
}

const PLOT_N = 200
const canvas = document.getElementById('plot') as HTMLCanvasElement
const ctx = canvas.getContext('2d')!
const statusEl = document.getElementById('status') as HTMLDivElement
const runBtn = document.getElementById('run') as HTMLButtonElement
const stopBtn = document.getElementById('stop') as HTMLButtonElement

const plotXs = new Float32Array(PLOT_N)
for (let i = 0; i < PLOT_N; i++) plotXs[i] = -Math.PI + (2 * Math.PI) * i / (PLOT_N - 1)

function drawPlot(modelYs: Float32Array): void {
  const w = canvas.width, h = canvas.height
  const style = getComputedStyle(document.documentElement)
  ctx.clearRect(0, 0, w, h)
  drawCurve(style.getPropertyValue('--axis').trim(),   1, (_, i) => h / 2)
  drawCurve(style.getPropertyValue('--target').trim(), 1, (x) => h / 2 - Math.sin(x) * (h * 0.4))
  drawCurve(style.getPropertyValue('--model').trim(),  2, (_, i) => h / 2 - modelYs[i]! * (h * 0.4))

  function drawCurve(color: string, width: number, y: (x: number, i: number) => number): void {
    ctx.strokeStyle = color
    ctx.lineWidth = width
    ctx.beginPath()
    for (let i = 0; i < PLOT_N; i++) {
      const px = (plotXs[i]! + Math.PI) / (2 * Math.PI) * w
      if (i === 0) ctx.moveTo(px, y(plotXs[i]!, i)); else ctx.lineTo(px, y(plotXs[i]!, i))
    }
    ctx.stroke()
  }
}

drawPlot(new Float32Array(PLOT_N))   // initial paint so the target curve is visible before training

let stopRequested = false
stopBtn.onclick = () => { stopRequested = true }

runBtn.onclick = async () => {
  runBtn.disabled = true
  stopBtn.disabled = false
  stopRequested = false
  statusEl.textContent = 'Compiling…'

  const { train, infer, compileMs } = await buildGraphs()
  statusEl.textContent = `Compiled in ${compileMs.toFixed(0)} ms. Training…`

  let step = 0, lastViz = 0
  try {
    while (!stopRequested) {
      step++
      const r = await train.step(makeBatch())
      if (r.kind === 'aborted') break
      const now = performance.now()
      if (now - lastViz > 200) {
        lastViz = now
        const pr = await infer.run({ x: plotXs })
        if (pr.kind === 'completed') drawPlot(pr.output)
        statusEl.textContent = `step ${step}   loss ${r.loss.toFixed(6)}`
      }
      if (step % 5 === 0) await new Promise(r => setTimeout(r, 0))
    }
    statusEl.textContent = `Stopped at step ${step}.`
  } catch (e) {
    statusEl.textContent = `error: ${(e as { message?: string })?.message ?? e}`
  } finally {
    infer.destroy()
    train.destroy()
    runBtn.disabled = false
    stopBtn.disabled = true
  }
}
```
**styles.css**

```css
:root {
  color-scheme: light;
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #666666;
  --border: #dddddd;
  --axis: #eeeeee;
  --target: #9a9a9a;
  --model: #0d9488;
  --accent: #0d9488;
  --accent-fg: #ffffff;
  --btn-bg: #ffffff;
  --btn-border: #bbbbbb;
}

html[data-theme="dark"] {
  color-scheme: dark;
  --bg: #1c1c1c;
  --fg: #ececec;
  --muted: #a0a0a0;
  --border: #333333;
  --axis: #2a2a2a;
  --target: #707070;
  --model: #5eead4;
  --accent: #14b8a6;
  --accent-fg: #0f1f1d;
  --btn-bg: #2a2a2a;
  --btn-border: #444444;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.page {
  max-width: 900px;
  margin: 2rem auto;
  padding: 0 1rem;
}

h1 {
  font-size: 1.4rem;
  margin-bottom: 0.25rem;
  color: var(--accent);
}

.subtitle {
  color: var(--muted);
  margin-top: 0;
  margin-bottom: 1.5rem;
}

.controls {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

button {
  padding: 0.5rem 1rem;
  font-size: 0.95rem;
  cursor: pointer;
  background: var(--btn-bg);
  color: var(--fg);
  border: 1px solid var(--btn-border);
  border-radius: 4px;
}

button.primary {
  background: var(--accent);
  color: var(--accent-fg);
  border-color: var(--accent);
}

button:disabled {
  opacity: 0.5;
  cursor: default;
}

canvas {
  display: block;
  border: 1px solid var(--border);
  border-radius: 6px;
  max-width: 100%;
  height: auto;
}

#status {
  font-family: ui-monospace, monospace;
  font-size: 0.9rem;
  color: var(--muted);
  margin-top: 0.75rem;
  min-height: 1.4em;
}

code {
  font-family: ui-monospace, monospace;
  font-size: 0.9em;
  background: var(--border);
  padding: 0.05rem 0.3rem;
  border-radius: 3px;
}
```
**index.html**

```html
<div class="page">
  <h1>It's a sine</h1>
  <p class="subtitle">
    A 3-layer MLP (1 → 64 → 64 → 1) learns to approximate <code>y = sin(x)</code>
    over [-π, π]. The faint curve is the target; the bold curve is the
    model's prediction, updated every ~200 ms during training.
  </p>
  <div class="controls">
    <button id="run" class="primary">Train</button>
    <button id="stop" disabled>Stop</button>
  </div>
  <canvas id="plot" width="800" height="280"></canvas>
  <div id="status">ready.</div>
</div>
```
**config.json**

```json
{
  "dependencies": {
    "tensorgrad": "0.1.0"
  },
  "description": "The smallest end-to-end tensorgrad training loop: a 3-layer MLP learns y = sin(x)."
}
```