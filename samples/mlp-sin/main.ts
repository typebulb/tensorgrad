// MLP regressor fitting y = sin(x) over [-π, π].
// Same tensorgrad pipeline as the transformer sample (Module + autograd + Adam +
// WGSL), but the model is ~3 layers and the loss is plain MSE — a useful sanity
// test that the library works for non-transformer shapes of problem.

import {
  Module, compileModule, init,
  add, mul, sub, meanAll, matmul, relu,
  type Tensor,
} from 'tensorgrad'

// Hyperparameters. Small to keep iteration fast.
const HIDDEN = 64
const B = 256                 // batch size
const LR = 0.005

// ---------- Modules: a 1 → HIDDEN → HIDDEN → 1 MLP -------------------------

class Linear extends Module {
  W: Tensor; b: Tensor
  constructor(public readonly inDim: number, public readonly outDim: number) {
    super()
    this.W = this.param([inDim, outDim], { init: init.kaiming() })
    this.b = this.param([outDim], { init: 'zeros' })
  }
}

class MLP extends Module {
  l1: Linear; l2: Linear; l3: Linear
  constructor() {
    super()
    this.l1 = new Linear(1, HIDDEN)
    this.l2 = new Linear(HIDDEN, HIDDEN)
    this.l3 = new Linear(HIDDEN, 1)
  }
}

// ---------- Forward + loss -------------------------------------------------

function linearFwd(p: Linear, x: Tensor): Tensor {
  return add(matmul(x, p.W), p.b)
}

function modelFwd(p: MLP, x: Tensor): Tensor {
  // x: [B, 1] -> [B, 1]
  const h1 = relu(linearFwd(p.l1, x))
  const h2 = relu(linearFwd(p.l2, h1))
  return linearFwd(p.l3, h2)
}

function lossFn(p: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  // Inputs are [B, 1] each; loss is mean squared error.
  const diff = sub(modelFwd(p, x), y)
  return meanAll(mul(diff, diff))
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

  // For visualisation we run a separate forward-only mini-pipeline. The simple
  // approach here: use the same step but with a fixed plot batch padded to B.
  // For brevity in this sample, we just compute model output via a plain JS
  // re-walk of the trained params after each viz interval. (Not as efficient as
  // a second compiled pipeline, but trivially correct and fine for B=200.)

  log('Training...')
  let step = 0
  let lastViz = 0
  while (!stopRequested) {
    step++
    const { x, y } = makeBatch()
    const lossVal = await compiled.step({
      x: new Float32Array(x),
      y: new Float32Array(y),
    })

    if (step === 1 || step % 100 === 0) {
      log(`  step ${step.toString().padStart(4)}  loss ${lossVal.toFixed(6)}`)
    }

    // Update plot every ~250 ms.
    const now = performance.now()
    if (now - lastViz > 250 || step === 1) {
      lastViz = now
      // Read params back, run model in JS for plotXs.
      const params = await compiled.downloadParams()
      const modelYs = forwardJS(params, plotXs)
      drawPlot(plotXs, modelYs)
    }

    if (step % 5 === 0) await new Promise(r => setTimeout(r, 0))
  }
  log(`Stopped at step ${step}.`, 'ok')
  compiled.destroy()
  runBtn.disabled = false; stopBtn.disabled = true
}

// JS reference forward. Lets us plot without a second compiled pipeline.
// Takes the typed param tree returned by downloadParams() — the layout
// mirrors the MLP class (l1/l2/l3, each with W and b).
function forwardJS(
  params: { l1: { W: Float32Array; b: Float32Array }; l2: { W: Float32Array; b: Float32Array }; l3: { W: Float32Array; b: Float32Array } },
  xs: Float32Array,
): Float32Array {
  const N = xs.length
  // l1: 1 -> HIDDEN
  const W1 = params.l1.W, b1 = params.l1.b
  const W2 = params.l2.W, b2 = params.l2.b
  const W3 = params.l3.W, b3 = params.l3.b
  const out = new Float32Array(N)
  const h1 = new Float32Array(HIDDEN)
  const h2 = new Float32Array(HIDDEN)
  for (let n = 0; n < N; n++) {
    const x = xs[n]!
    // h1 = relu(x * W1 + b1)
    for (let j = 0; j < HIDDEN; j++) {
      let s = b1[j]!
      s += x * W1[j]!  // W1 is [1, HIDDEN], row-major
      h1[j] = s > 0 ? s : 0
    }
    // h2 = relu(h1 @ W2 + b2)
    for (let j = 0; j < HIDDEN; j++) {
      let s = b2[j]!
      for (let k = 0; k < HIDDEN; k++) s += h1[k]! * W2[k * HIDDEN + j]!
      h2[j] = s > 0 ? s : 0
    }
    // y = h2 @ W3 + b3
    let s = b3[0]!
    for (let k = 0; k < HIDDEN; k++) s += h2[k]! * W3[k]!
    out[n] = s
  }
  return out
}

runBtn.onclick = () => { run().catch(e => log(`error: ${e?.message ?? e}\n${e?.stack ?? ''}`, 'err')) }
