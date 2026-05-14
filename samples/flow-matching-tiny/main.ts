// Tiny rectified-flow / flow-matching model on MNIST. The model learns the
// velocity field `v_θ(x_t, t)` that transports samples along the straight
// line from data to noise; generation is Euler integration of that field
// backward in time. Typically reaches diffusion-quality outputs at fewer
// sampling steps.
//
// Training: draw x_1 ~ N(0,1) in-graph, mix with the clean image as
// `x_t = (1 − t) · x_0 + t · x_1` = `x_0 + t · (x_1 − x_0)`, regress on the
// constant velocity target `v* = x_1 − x_0` with MSE. The `(x_1 − x_0)` term
// is computed once and reused as both the path mixer and the loss target.
//
// Sampling: start at x = N(0,1) (t=1) and Euler-step toward t=0,
// `x ← x − dt · v_θ(x, t)`. A discrete timestep embedding (an `Embedding`
// table indexed by integer t) handles the time conditioning — continuous t
// is snapped to the nearest of T_STEPS buckets at training and sampling time.
//
// File layout: ML + app logic at the top, UI at the bottom. Pattern shared
// with the other samples.

import {
  isWebGPUAvailable,
  type CompiledTraining, type CompiledForward,
} from 'tensorgrad'
import {
  TinyFlow, irSpec, compileTraining,
  IMG_H, IMG_W, IMG_LEN, T_STEPS, BATCH_SIZE,
} from './spec.ts'

// ========== MODEL / TRAINING ==========

const MNIST_PREFIX = 'https://s3.eu-west-2.amazonaws.com/solenya-media/'
const SAMPLE_STEPS = 25          // Euler steps at generation time

// ---------------------------------------------------------------------------
// Time schedule. Flow matching just needs t ∈ [0, 1]; we discretize into
// T_STEPS buckets so an integer-indexed `Embedding` handles the time
// conditioning. Index 0 is unused; the math reads naturally with 1-indexed t.
// `tNormTable[t] = t / T_STEPS` is the only schedule table the loss needs —
// `(1 − t)` falls out of `x_0 + t · (x_1 − x_0)`.
// ---------------------------------------------------------------------------

const tNormTable = (() => {
  const out = new Float32Array(T_STEPS + 1)
  for (let t = 1; t <= T_STEPS; t++) out[t] = t / T_STEPS
  return out
})()

// Box-Muller into an existing Float32Array. Only needed for the sampling
// loop — training noise is drawn in-graph via `randn`.
function fillRandn(out: Float32Array): void {
  for (let i = 0; i < out.length; i += 2) {
    const u1 = Math.max(1e-10, Math.random())
    const u2 = Math.random()
    const mag = Math.sqrt(-2 * Math.log(u1))
    out[i] = mag * Math.cos(2 * Math.PI * u2)
    if (i + 1 < out.length) out[i + 1] = mag * Math.sin(2 * Math.PI * u2)
  }
}

// ---------------------------------------------------------------------------
// MNIST loading. idx-ubyte gzipped, normalized to [-1, 1] so the unit-variance
// noise endpoint is well-matched to the data range.
// ---------------------------------------------------------------------------

interface MnistSet { images: Float32Array; count: number }

async function inflateGzip(buf: ArrayBuffer): Promise<Uint8Array> {
  const stream = new Response(buf).body!.pipeThrough(new DecompressionStream('gzip'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

async function loadImages(file: string): Promise<MnistSet> {
  const buf = await fetch(MNIST_PREFIX + file).then(r => r.arrayBuffer())
  const bytes = await inflateGzip(buf)
  const view = new DataView(bytes.buffer, bytes.byteOffset)
  const count = view.getUint32(4)
  const images = new Float32Array(count * IMG_LEN)
  for (let i = 0; i < images.length; i++) images[i] = (bytes[16 + i]! / 255) * 2 - 1
  return { images, count }
}

// Model + loss + predict live in ./spec.ts.

// ---------------------------------------------------------------------------
// State + lifecycle. UI calls into the entry points; status / sample frames
// flow back through hooks registered by the UI section at boot.
// ---------------------------------------------------------------------------

let train: CompiledTraining<TinyFlow> | null = null
let infer:    CompiledForward<TinyFlow> | null = null
let trainData: MnistSet | null = null
let trainOrder: number[] = []
let trainCursor = 0
let running = false
let step = 0

let onStatus: (msg: string) => void = () => {}
let onSampleFrame: (img: Float32Array, t: number) => void = () => {}

function shuffleOrder(arr: number[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j]!, arr[i]!]
  }
}

// One training batch: clean images, per-sample timesteps, and the tNorm
// table (constant — same every step, just plumbed through).
function nextBatch(): {
  x_0: Float32Array; t: Int32Array; tNorm_table: Float32Array
} {
  if (!trainData) throw new Error('nextBatch: no data')
  const x_0 = new Float32Array(BATCH_SIZE * IMG_LEN)
  const t = new Int32Array(BATCH_SIZE)
  for (let b = 0; b < BATCH_SIZE; b++) {
    if (trainCursor >= trainOrder.length) { shuffleOrder(trainOrder); trainCursor = 0 }
    const idx = trainOrder[trainCursor++]!
    t[b] = 1 + Math.floor(Math.random() * T_STEPS)
    x_0.set(trainData.images.subarray(idx * IMG_LEN, (idx + 1) * IMG_LEN), b * IMG_LEN)
  }
  return { x_0, t, tNorm_table: tNormTable }
}

async function runTraining(): Promise<void> {
  let lastEval = 0
  while (running && train) {
    const r = await train.step(nextBatch())
    if (r.kind === 'aborted') return
    const lastLoss = r.loss
    step += 1
    if (!Number.isFinite(lastLoss)) {
      onStatus(`step ${step}: loss is ${lastLoss} — NaN, aborting.`)
      running = false
      return
    }
    const now = Date.now()
    if (now - lastEval > 1000) {
      lastEval = now
      onStatus(`step ${step}  loss ${lastLoss.toFixed(4)}`)
    }
    if (step % 4 === 0) await new Promise(r => setTimeout(r, 0))
  }
}

// Euler integration of dx/dt = v_θ(x, t) from t=1 down to t=0. At each step
// snap continuous t to its nearest embedding bucket so the model sees the
// same discrete index space it was trained on. SAMPLE_STEPS < T_STEPS is the
// whole point — flow matching trades schedule complexity for sample-step count.
async function generateOne(): Promise<Float32Array | null> {
  if (!infer) return null
  const x = new Float32Array(IMG_LEN)
  const tBuf = new Int32Array(1)
  fillRandn(x)
  const dt = 1 / SAMPLE_STEPS
  for (let i = 0; i < SAMPLE_STEPS; i++) {
    const tCont = 1 - i / SAMPLE_STEPS                          // 1, (N−1)/N, …, 1/N
    const tBucket = Math.max(1, Math.min(T_STEPS, Math.round(tCont * T_STEPS)))
    tBuf[0] = tBucket
    const vr = await infer.run({ x_t: x, t: tBuf })
    if (vr.kind === 'aborted') return null
    const v = vr.output
    for (let p = 0; p < IMG_LEN; p++) x[p] = x[p]! - dt * v[p]!
    if (i === SAMPLE_STEPS - 1 || i % 2 === 0) {
      onSampleFrame(x.slice(), tBucket)
      await new Promise(r => setTimeout(r, 0))
    }
  }
  return x
}

async function loadMnist(): Promise<void> {
  trainData = await loadImages('train-images-idx3-ubyte.gz')
  trainOrder = Array.from({ length: trainData.count }, (_, i) => i)
  shuffleOrder(trainOrder)
}

async function buildGraphs(): Promise<void> {
  onStatus('compiling flow-matching model…')
  const t0 = performance.now()
  train = await compileTraining()
  infer = await train.attach({
    forward: irSpec.predict,
    inputs: {
      x_t: [1, 1, IMG_H, IMG_W],
      t:   { shape: [1], dtype: 'i32' },
    },
  })
  step = 0
  onStatus(`compiled (${train.kernels.length} kernels, ${(performance.now() - t0).toFixed(0)} ms, seed ${train.seed})`)
}

function startTraining(): void {
  if (running) return
  running = true
  void runTraining()
}

function stopTraining(): void {
  running = false
}

// ========== UI ==========

const statusEl  = document.getElementById('status')   as HTMLDivElement
const trainBtn  = document.getElementById('train')    as HTMLButtonElement
const stopBtn   = document.getElementById('stop')     as HTMLButtonElement
const sampleBtn = document.getElementById('sample')   as HTMLButtonElement
const canvas    = document.getElementById('sample-canvas') as HTMLCanvasElement
const tLabel    = document.getElementById('t-label')  as HTMLSpanElement

const PIXEL_SCALE = 10
canvas.width  = IMG_W * PIXEL_SCALE
canvas.height = IMG_H * PIXEL_SCALE
const ctx = canvas.getContext('2d')!
ctx.imageSmoothingEnabled = false

const small = document.createElement('canvas')
small.width = IMG_W
small.height = IMG_H
const sctx = small.getContext('2d')!
const imgData = sctx.createImageData(IMG_W, IMG_H)

function renderImage(img: Float32Array): void {
  for (let i = 0; i < IMG_LEN; i++) {
    const v = Math.max(0, Math.min(255, Math.round(((img[i]! + 1) / 2) * 255)))
    const off = i * 4
    imgData.data[off]     = v
    imgData.data[off + 1] = v
    imgData.data[off + 2] = v
    imgData.data[off + 3] = 255
  }
  sctx.putImageData(imgData, 0, 0)
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height)
}

onStatus = (msg) => { statusEl.textContent = msg }
onSampleFrame = (img, t) => {
  renderImage(img)
  tLabel.textContent = `t=${t}`
}

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

sampleBtn.addEventListener('click', async () => {
  sampleBtn.disabled = true
  try { await generateOne() }
  finally { sampleBtn.disabled = false }
})

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  if (!isWebGPUAvailable()) {
    onStatus('WebGPU not available. Try Chrome 113+ or Safari 17.4+.')
    return
  }
  onStatus('loading MNIST…')
  await loadMnist()
  await buildGraphs()
  trainBtn.disabled = false
  sampleBtn.disabled = false
}

boot().catch((e: unknown) => {
  const msg = (e as { message?: string })?.message ?? String(e)
  onStatus(`error: ${msg}`)
  console.error(e)
})
