// MLP regressor fitting y = sin(x) over [-π, π].
// Same tensorgrad pipeline as the transformer sample (Module + autograd + Adam +
// WGSL), but the model is ~3 layers and the loss is plain MSE — a useful sanity
// test that the library works for non-transformer shapes of problem.

import {
  Module, compileModule, nn,
  mul, sub, mean, relu,
  type Tensor,
} from 'tensorgrad'

// Hyperparameters. Small to keep iteration fast.
const HIDDEN = 64
const B = 256                 // batch size
const LR = 0.005

// ---------- Model: 1 → HIDDEN → HIDDEN → 1 MLP ----------------------------

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

// ---------- Batch generation ----------------------------------------------

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

// ---------- Plot the model's current prediction over [-π, π] --------------

const canvas = document.getElementById('plot') as HTMLCanvasElement
const cctx = canvas.getContext('2d')!
const PLOT_N = 200
const plotXs = new Float32Array(PLOT_N)
for (let i = 0; i < PLOT_N; i++) plotXs[i] = -Math.PI + (2 * Math.PI) * i / (PLOT_N - 1)

function drawPlot(modelXs: Float32Array, modelYs: Float32Array) {
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
  for (let i = 0; i < PLOT_N; i++) {
    const x = plotXs[i]!
    const px = (x + Math.PI) / (2 * Math.PI) * w
    const py = h / 2 - Math.sin(x) * (h * 0.4)
    if (i === 0) cctx.moveTo(px, py); else cctx.lineTo(px, py)
  }
  cctx.stroke()
  // Model prediction
  cctx.strokeStyle = '#06c'
  cctx.lineWidth = 2
  cctx.beginPath()
  for (let i = 0; i < modelXs.length; i++) {
    const px = (modelXs[i]! + Math.PI) / (2 * Math.PI) * w
    const py = h / 2 - modelYs[i]! * (h * 0.4)
    if (i === 0) cctx.moveTo(px, py); else cctx.lineTo(px, py)
  }
  cctx.stroke()
  cctx.lineWidth = 1
}

// ---------- Logging UI ----------------------------------------------------

const logEl = document.getElementById('log')!
const runBtn = document.getElementById('run') as HTMLButtonElement
const stopBtn = document.getElementById('stop') as HTMLButtonElement
let stopRequested = false

function log(msg: string, cls?: 'err' | 'ok') {
  const line = document.createElement('div')
  if (cls) line.className = cls
  line.textContent = msg
  logEl.appendChild(line)
  logEl.scrollTop = logEl.scrollHeight
  console.log(msg)
  fetch('/__log', { method: 'POST', body: JSON.stringify({ msg }) }).catch(() => {})
}

window.addEventListener('error', e => log(`[error] ${e.message}`, 'err'))
window.addEventListener('unhandledrejection', e => log(`[promise] ${String((e as any).reason?.message ?? (e as any).reason)}`, 'err'))
const origConsoleError = console.error.bind(console)
console.error = (...args: unknown[]) => {
  origConsoleError(...args)
  const text = args.map(a => typeof a === 'string' ? a : (a instanceof Error ? a.stack ?? a.message : JSON.stringify(a))).join(' ')
  log(`[console.error] ${text}`, 'err')
}

stopBtn.onclick = () => { stopRequested = true }

// ---------- Main ----------------------------------------------------------

async function run() {
  runBtn.disabled = true; stopBtn.disabled = false; stopRequested = false
  logEl.innerHTML = ''
  // Draw initial state (target sin curve + zeros for model).
  drawPlot(plotXs, new Float32Array(PLOT_N))

  log('Compiling MLP + Adam...')
  const t0 = performance.now()
  const compiled = await compileModule({
    factory: () => new MLP(),
    loss: lossFn,
    adam: { lr: LR },
    inputs: { x: [B, 1], y: [B, 1] },
  })
  log(`  ${compiled.kernelCount} kernels, compile ${(performance.now() - t0).toFixed(0)} ms`, 'ok')

  // Inference graph for plotting: shares param buffers with the training
  // graph, polymorphic over the batch dim so we can run it at PLOT_N=200
  // without recompiling per-shape.
  const predict = await compiled.compileForward({
    forward: predictFn,
    inputs: { x: [null, 1] },
  })

  // Stretch the plot x's into [PLOT_N, 1] for the [B, 1] input shape.
  const plotInput = new Float32Array(PLOT_N)
  for (let i = 0; i < PLOT_N; i++) plotInput[i] = plotXs[i]!

  log('Training...')
  let step = 0
  let lastViz = 0
  while (!stopRequested) {
    step++
    const { x, y } = makeBatch()
    const lossVal = await compiled.step({ x, y })

    if (step === 1 || step % 100 === 0) {
      log(`  step ${step.toString().padStart(4)}  loss ${lossVal.toFixed(6)}`)
    }

    // Update plot every ~250 ms.
    const now = performance.now()
    if (now - lastViz > 250 || step === 1) {
      lastViz = now
      const modelYs = await predict.run({ x: plotInput })
      drawPlot(plotXs, modelYs)
    }

    if (step % 5 === 0) await new Promise(r => setTimeout(r, 0))
  }
  log(`Stopped at step ${step}.`, 'ok')
  predict.destroy()
  compiled.destroy()
  runBtn.disabled = false; stopBtn.disabled = true
}

runBtn.onclick = () => { run().catch(e => log(`error: ${e?.message ?? e}\n${e?.stack ?? ''}`, 'err')) }
