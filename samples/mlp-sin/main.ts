// MLP regressor fitting y = sin(x) over [-π, π].
// Same tensorgrad pipeline as the transformer sample (Module + autograd + Adam +
// WGSL), but the model is ~3 layers and the loss is plain MSE — a useful sanity
// test that the library works for non-transformer shapes of problem.
//
// File layout: ML + training above, UI below. The ML section exposes a small
// set of entry points (startTraining, stopTraining) and emits updates via the
// `onStatus` and `onPlot` hooks the UI registers at boot. DOM access lives
// entirely in the UI section.

import {
  Module, compile, nn,
  mul, sub, mean, relu,
  type Tensor, type CompiledTraining, type CompiledForward,
} from 'tensorgrad'

// ========== MODEL / TRAINING ==========

const HIDDEN = 64
const B = 256                 // batch size
const LR = 0.005
const PLOT_N = 200

class MLP extends Module {
  l1 = new nn.Linear(1, HIDDEN)
  l2 = new nn.Linear(HIDDEN, HIDDEN)
  l3 = new nn.Linear(HIDDEN, 1)
}

function modelFwd(p: MLP, x: Tensor): Tensor {
  return p.l3.fwd(relu(p.l2.fwd(relu(p.l1.fwd(x)))))
}

function lossFn(p: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  const diff = sub(modelFwd(p, x), y)
  return mean(mul(diff, diff))
}

function predictFn(p: MLP, { x }: { x: Tensor }): Tensor {
  return modelFwd(p, x)
}

const plotXs = new Float32Array(PLOT_N)
for (let i = 0; i < PLOT_N; i++) plotXs[i] = -Math.PI + (2 * Math.PI) * i / (PLOT_N - 1)

let train: CompiledTraining<MLP> | null = null
let infer: CompiledForward<MLP> | null = null
let running = false
let step = 0

// UI-supplied sinks; assigned in the UI section so the ML side has zero DOM
// dependencies. Defaults let this section behave in isolation.
let onStatus: (msg: string, cls?: 'err' | 'ok') => void = () => {}
let onPlot: (xs: Float32Array, ys: Float32Array) => void = () => {}

function makeBatch(): { x: Float32Array; y: Float32Array } {
  const x = new Float32Array(B)
  const y = new Float32Array(B)
  for (let i = 0; i < B; i++) {
    const v = (Math.random() * 2 - 1) * Math.PI   // uniform in [-π, π]
    x[i] = v
    y[i] = Math.sin(v)
  }
  return { x, y }
}

async function startTraining(): Promise<void> {
  if (running) return
  running = true
  try {
    if (!train) await buildGraphs()
    await runTraining()
  } catch (e) {
    running = false
    onStatus(`error: ${(e as { message?: string })?.message ?? e}`, 'err')
    throw e
  }
}

function stopTraining(): void {
  running = false
}

async function buildGraphs(): Promise<void> {
  onStatus('Compiling MLP + Adam...')
  const t0 = performance.now()
  const model = new MLP()
  train = await compile({
    model,
    loss: lossFn,
    optimizer: { kind: 'adam', lr: LR },
    inputs: { x: [B, 1], y: [B, 1] },
  })
  // Inference graph for plotting: shares param buffers with the training
  // graph, polymorphic over the batch dim so we can run it at PLOT_N=200
  // without recompiling per-shape.
  infer = await train.attach({
    forward: predictFn,
    inputs: { x: [null, 1] },
  })
  onStatus(`  ${train.kernels.length} kernels, compile ${(performance.now() - t0).toFixed(0)} ms`, 'ok')
}

async function runTraining(): Promise<void> {
  if (!train || !infer) return
  onStatus('Training...')
  let lastViz = 0
  while (running) {
    step++
    const r = await train.step(makeBatch())
    if (r.kind === 'aborted') break

    if (step === 1 || step % 100 === 0) {
      onStatus(`  step ${step.toString().padStart(4)}  loss ${r.loss.toFixed(6)}`)
    }

    const now = performance.now()
    if (now - lastViz > 250 || step === 1) {
      lastViz = now
      const out = await infer.run({ x: plotXs })
      if (out.kind === 'completed') onPlot(plotXs, out.output)
    }

    if (step % 5 === 0) await new Promise(r => setTimeout(r, 0))
  }
  onStatus(`Stopped at step ${step}.`, 'ok')
}

// ========== UI ==========

const canvas = document.getElementById('plot') as HTMLCanvasElement
const cctx = canvas.getContext('2d')!
const logEl = document.getElementById('log')!
const runBtn = document.getElementById('run') as HTMLButtonElement
const stopBtn = document.getElementById('stop') as HTMLButtonElement

function log(msg: string, cls?: 'err' | 'ok'): void {
  const line = document.createElement('div')
  if (cls) line.className = cls
  line.textContent = msg
  logEl.appendChild(line)
  logEl.scrollTop = logEl.scrollHeight
  console.log(msg)
  fetch('/__log', { method: 'POST', body: JSON.stringify({ msg }) }).catch(() => {})
}

function drawPlot(xs: Float32Array, ys: Float32Array): void {
  const w = canvas.width, h = canvas.height
  cctx.clearRect(0, 0, w, h)
  // Axis
  cctx.strokeStyle = '#ddd'
  cctx.beginPath()
  cctx.moveTo(0, h / 2); cctx.lineTo(w, h / 2)
  cctx.stroke()
  // True sin curve
  cctx.strokeStyle = '#888'
  cctx.beginPath()
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i]!
    const px = (x + Math.PI) / (2 * Math.PI) * w
    const py = h / 2 - Math.sin(x) * (h * 0.4)
    if (i === 0) cctx.moveTo(px, py); else cctx.lineTo(px, py)
  }
  cctx.stroke()
  // Model prediction
  cctx.strokeStyle = '#06c'
  cctx.lineWidth = 2
  cctx.beginPath()
  for (let i = 0; i < xs.length; i++) {
    const px = (xs[i]! + Math.PI) / (2 * Math.PI) * w
    const py = h / 2 - ys[i]! * (h * 0.4)
    if (i === 0) cctx.moveTo(px, py); else cctx.lineTo(px, py)
  }
  cctx.stroke()
  cctx.lineWidth = 1
}

onStatus = log
onPlot = drawPlot

window.addEventListener('error', e => log(`[error] ${e.message}`, 'err'))
window.addEventListener('unhandledrejection', e => log(`[promise] ${String((e as any).reason?.message ?? (e as any).reason)}`, 'err'))
const origConsoleError = console.error.bind(console)
console.error = (...args: unknown[]) => {
  origConsoleError(...args)
  const text = args.map(a => typeof a === 'string' ? a : (a instanceof Error ? a.stack ?? a.message : JSON.stringify(a))).join(' ')
  log(`[console.error] ${text}`, 'err')
}

runBtn.addEventListener('click', () => {
  runBtn.disabled = true
  stopBtn.disabled = false
  logEl.innerHTML = ''
  startTraining().finally(() => {
    runBtn.disabled = false
    stopBtn.disabled = true
  })
})

stopBtn.addEventListener('click', () => { stopTraining() })
