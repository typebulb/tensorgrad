// KAN-tiny: Kolmogorov-Arnold Network fitting y = sin(πx). Each "edge" of the
// network is a learnable univariate function (cubic B-spline) rather than a
// fixed nonlinearity times a scalar weight. The headline demo is the live
// per-edge spline grid: extract each spline's coefficients via
// `downloadParams` between training steps and watch the curves evolve as
// the network finds a decomposition of the target.
//
// Per layer the compute is:
//   1. B(x) = cubic B-spline basis values at `KNOTS` integer-spaced positions
//      → shape [batch, I, KNOTS]
//   2. Y_spline = einsum('bik,oik->bo', B, C)   (implemented as reshape + matmul)
//   3. Y_res    = silu(x) @ W_res
//   4. output   = Y_spline + Y_res
//
// File layout: model + training at the top, UI at the bottom. Same convention
// as the other samples.

import {
  isWebGPUAvailable,
  type CompiledTraining, type CompiledForward,
} from 'tensorgrad'
import {
  KAN, irSpec, compileTraining,
  KNOTS, HIDDEN, BATCH,
} from './spec.ts'

// ========== MODEL / TRAINING ==========
// Model + loss + predict live in ./spec.ts.

// ---------- Batch generation ----------------------------------------------

// x ∈ [-1, 1] is the normalized input the network sees; the curve is
// y = sin(πx) over that domain. Equivalent to mlp-sin's y = sin(x) over
// [-π, π], pre-scaled so x lands inside the spline grid.
function makeBatch(): { x: Float32Array; y: Float32Array } {
  const x = new Float32Array(BATCH)
  const y = new Float32Array(BATCH)
  for (let i = 0; i < BATCH; i++) {
    const v = Math.random() * 2 - 1
    x[i] = v
    y[i] = Math.sin(Math.PI * v)
  }
  return { x, y }
}

// ---------- CPU-side spline eval (for the per-edge plot) ------------------

function basisCpu(u: number): number {
  const a = Math.abs(u)
  if (a < 1) return 2 / 3 - u * u + 0.5 * a * a * a
  if (a < 2) return Math.pow(2 - a, 3) / 6
  return 0
}

// φ_{o,i}(x) = Σ_k C[o,i,k] · N(((x+1)/2)·(KNOTS-1) - k), evaluated for the
// per-edge panels. C is laid out [O, I, KNOTS], row-major.
function evalEdge(C: Float32Array, I: number, oIdx: number, iIdx: number, x: number): number {
  const xs = (x + 1) * (KNOTS - 1) / 2
  const base = oIdx * I * KNOTS + iIdx * KNOTS
  let sum = 0
  for (let k = 0; k < KNOTS; k++) sum += C[base + k]! * basisCpu(xs - k)
  return sum
}

// ---------- Lifecycle -----------------------------------------------------

let train: CompiledTraining<KAN> | null = null
let infer:    CompiledForward<KAN> | null = null
let running  = false
let step     = 0

let onStatus: (msg: string) => void = () => {}
let onRender: (modelYs: Float32Array, params: { l1C: Float32Array; l2C: Float32Array }) => void = () => {}

const PLOT_N = 200
const PLOT_XS = new Float32Array(PLOT_N)
for (let i = 0; i < PLOT_N; i++) PLOT_XS[i] = -1 + 2 * i / (PLOT_N - 1)

async function renderAll(): Promise<void> {
  if (!infer || !train) return
  const r = await infer.run({ x: PLOT_XS })
  if (r.kind === 'aborted') return
  const params = await train.downloadParams()
  onRender(r.output, { l1C: params['l1.C']!, l2C: params['l2.C']! })
}

async function buildGraphs(): Promise<void> {
  onStatus('compiling…')
  const t0 = performance.now()
  train = await compileTraining()
  infer = await train.attach({
    forward: irSpec.predict,
    inputs: { x: [null, 1] },
  })
  step = 0
  onStatus(`compiled (${train.kernels.length} kernels, ${(performance.now() - t0).toFixed(0)} ms)`)
}

async function trainLoop(): Promise<void> {
  let lastRender = 0
  let lastLoss = 0
  while (running && train) {
    const sr = await train.step(makeBatch())
    if (sr.kind === 'aborted') return
    lastLoss = sr.loss
    step += 1
    if (!Number.isFinite(lastLoss)) {
      onStatus(`step ${step}: loss is ${lastLoss} — diverged.`)
      running = false
      return
    }
    const now = Date.now()
    if (now - lastRender > 250) {
      lastRender = now
      onStatus(`step ${step}  loss ${lastLoss.toFixed(5)}`)
      await renderAll()
    }
    if (step % 4 === 0) await new Promise(r => setTimeout(r, 0))
  }
  onStatus(`stopped at step ${step}  loss ${lastLoss.toFixed(5)}`)
}

function startTraining(): void { if (!running) { running = true; void trainLoop() } }
function stopTraining(): void { running = false }

async function resetWeights(): Promise<void> {
  if (!train) return
  const wasRunning = running
  running = false
  await new Promise<void>(r => setTimeout(r, 0))
  await train.reset()
  step = 0
  await renderAll()
  onStatus(`weights re-initialized (seed ${train.seed})`)
  if (wasRunning) { running = true; void trainLoop() }
}

// ========== UI ==========

const statusEl   = document.getElementById('status')      as HTMLDivElement
const trainBtn   = document.getElementById('train')       as HTMLButtonElement
const stopBtn    = document.getElementById('stop')        as HTMLButtonElement
const resetBtn   = document.getElementById('reset')       as HTMLButtonElement
const fitCanvas  = document.getElementById('fit-canvas')  as HTMLCanvasElement
const edgeCanvas = document.getElementById('edge-canvas') as HTMLCanvasElement
const fitCtx  = fitCanvas.getContext('2d')!
const edgeCtx = edgeCanvas.getContext('2d')!

function drawFit(modelYs: Float32Array): void {
  const w = fitCanvas.width
  const h = fitCanvas.height
  fitCtx.clearRect(0, 0, w, h)
  fitCtx.strokeStyle = '#ddd'
  fitCtx.beginPath(); fitCtx.moveTo(0, h / 2); fitCtx.lineTo(w, h / 2); fitCtx.stroke()
  // Target y = sin(π x)
  fitCtx.strokeStyle = '#888'
  fitCtx.beginPath()
  for (let i = 0; i < PLOT_N; i++) {
    const x = PLOT_XS[i]!
    const px = (x + 1) / 2 * w
    const py = h / 2 - Math.sin(Math.PI * x) * (h * 0.4)
    if (i === 0) fitCtx.moveTo(px, py); else fitCtx.lineTo(px, py)
  }
  fitCtx.stroke()
  // Model prediction
  fitCtx.strokeStyle = '#06c'
  fitCtx.lineWidth = 2
  fitCtx.beginPath()
  for (let i = 0; i < PLOT_N; i++) {
    const px = (PLOT_XS[i]! + 1) / 2 * w
    const py = h / 2 - (modelYs[i] ?? 0) * (h * 0.4)
    if (i === 0) fitCtx.moveTo(px, py); else fitCtx.lineTo(px, py)
  }
  fitCtx.stroke()
  fitCtx.lineWidth = 1
}

// Per-edge spline panels. Row 1: layer 1 edges (network's [-1, 1] input
// directly). Row 2: layer 2 edges (taking a hidden activation from layer 1).
// Both rows plot over the normalized [-1, 1] window — layer 2 hidden
// activations may extend beyond this in practice, but [-1, 1] is what the
// spline grid covers.
const PANEL_SAMPLES = 60
const SAMPLE_XS = new Float32Array(PANEL_SAMPLES)
for (let i = 0; i < PANEL_SAMPLES; i++) SAMPLE_XS[i] = -1 + 2 * i / (PANEL_SAMPLES - 1)

function drawEdges(l1C: Float32Array, l2C: Float32Array): void {
  const w = edgeCanvas.width
  const h = edgeCanvas.height
  edgeCtx.clearRect(0, 0, w, h)
  const labelW   = 60
  const padTop   = 16
  const padRight = 12
  const padBot   = 8
  const cols     = HIDDEN
  const rows     = 2
  const colGap   = 12
  const rowGap   = 18
  const usableW  = w - labelW - padRight
  const usableH  = h - padTop - padBot
  const panelW   = (usableW - colGap * (cols - 1)) / cols
  const panelH   = (usableH - rowGap * (rows - 1)) / rows

  edgeCtx.font = '11px ui-monospace, monospace'
  edgeCtx.fillStyle = '#555'
  edgeCtx.textBaseline = 'middle'
  edgeCtx.textAlign = 'right'
  edgeCtx.fillText('layer 1', labelW - 6, padTop + panelH / 2)
  edgeCtx.fillText('layer 2', labelW - 6, padTop + panelH + rowGap + panelH / 2)

  const drawPanel = (originX: number, originY: number, getY: (xn: number) => number): void => {
    edgeCtx.strokeStyle = '#ddd'
    edgeCtx.strokeRect(originX, originY, panelW, panelH)
    edgeCtx.strokeStyle = '#eee'
    edgeCtx.beginPath()
    edgeCtx.moveTo(originX, originY + panelH / 2)
    edgeCtx.lineTo(originX + panelW, originY + panelH / 2)
    edgeCtx.stroke()

    const ys = new Float32Array(PANEL_SAMPLES)
    let minY = Infinity, maxY = -Infinity
    for (let i = 0; i < PANEL_SAMPLES; i++) {
      const y = getY(SAMPLE_XS[i]!)
      ys[i] = y
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    const span = Math.max(maxY - minY, 0.4)
    const mid  = (maxY + minY) / 2
    const yMin = mid - span / 2
    const yMax = mid + span / 2

    edgeCtx.strokeStyle = '#06c'
    edgeCtx.lineWidth = 1.5
    edgeCtx.beginPath()
    for (let i = 0; i < PANEL_SAMPLES; i++) {
      const px = originX + (SAMPLE_XS[i]! + 1) / 2 * panelW
      const py = originY + panelH - (ys[i]! - yMin) / (yMax - yMin) * panelH
      if (i === 0) edgeCtx.moveTo(px, py); else edgeCtx.lineTo(px, py)
    }
    edgeCtx.stroke()
    edgeCtx.lineWidth = 1
  }

  // Row 1: layer 1 edges, C: [HIDDEN, 1, KNOTS] — one edge per output.
  for (let o = 0; o < HIDDEN; o++) {
    const x = labelW + o * (panelW + colGap)
    drawPanel(x, padTop, xn => evalEdge(l1C, 1, o, 0, xn))
  }
  // Row 2: layer 2 edges, C: [1, HIDDEN, KNOTS] — one edge per input.
  for (let i = 0; i < HIDDEN; i++) {
    const x = labelW + i * (panelW + colGap)
    drawPanel(x, padTop + panelH + rowGap, xn => evalEdge(l2C, HIDDEN, 0, i, xn))
  }
}

onStatus = msg => { statusEl.textContent = msg }
onRender = (modelYs, { l1C, l2C }) => { drawFit(modelYs); drawEdges(l1C, l2C) }

trainBtn.addEventListener('click', () => {
  trainBtn.disabled = true
  stopBtn.disabled = false
  startTraining()
})

stopBtn.addEventListener('click', () => {
  stopTraining()
  trainBtn.disabled = false
  stopBtn.disabled = true
})

resetBtn.addEventListener('click', () => { void resetWeights() })

// ---------- Boot ----------------------------------------------------------

async function boot(): Promise<void> {
  if (!isWebGPUAvailable()) {
    onStatus('WebGPU not available. Try Chrome 113+ or Safari 17.4+.')
    return
  }
  await buildGraphs()
  await renderAll()
  trainBtn.disabled = false
  resetBtn.disabled = false
}

boot().catch((e: unknown) => {
  const msg = (e as { message?: string })?.message ?? String(e)
  onStatus(`error: ${msg}`)
  console.error(e)
})
