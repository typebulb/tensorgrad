// 2D implicit neural representation. A tiny MLP learns the mapping
// `(x, y) → (r, g, b)` for the pixels of a target image, then renders the
// reconstruction continuously as it trains. Same paradigm as NeRF / SIREN /
// Fourier Features — the headline trick is sinusoidal positional encoding
// (`sin` / `cos` at multiple frequency bands), without which a plain MLP can
// only fit the low-frequency part of the signal.
//
// Architecture: pos-enc (4·L features) → 3 hidden layers of 64 → sigmoid →
// RGB. L = 8 frequency bands ≈ log2(image side), enough detail for 64×64.
//
// Training: random pixel batches (1024 coords + RGBs per step), Adam, MSE
// loss in pixel space. Default image is procedurally drawn (gradient + the
// word "tensorgrad" rendered on top) so high-frequency edges and smooth
// regions both appear. Users can swap in their own image via the upload
// button.
//
// File layout: ML + app logic at the top, UI at the bottom. Same convention
// as the other samples.

import {
  isWebGPUAvailable,
  type CompiledTraining, type CompiledForward,
} from 'tensorgrad'
import {
  NeRFTiny, predictFn, compileTraining,
  IMG_H, IMG_W, N_PIXELS, BATCH_SIZE, L_FREQS,
} from './spec.ts'

// ========== MODEL / TRAINING ==========

// Frequency bands π·2^k, used once per step as a tensor input. Pre-allocated
// so we're not re-creating a 32-byte buffer every step.
const FREQS = new Float32Array(L_FREQS)
for (let k = 0; k < L_FREQS; k++) FREQS[k] = Math.PI * Math.pow(2, k)

// All grid coordinates in [-1, 1], row-major. Used by the inference render.
const GRID_COORDS = new Float32Array(N_PIXELS * 2)
for (let row = 0; row < IMG_H; row++) {
  for (let col = 0; col < IMG_W; col++) {
    const i = row * IMG_W + col
    GRID_COORDS[i * 2]     = 2 * (col + 0.5) / IMG_W - 1
    GRID_COORDS[i * 2 + 1] = 2 * (row + 0.5) / IMG_H - 1
  }
}

// Model + loss + predict live in ./spec.ts.

// ---------------------------------------------------------------------------
// State + lifecycle
// ---------------------------------------------------------------------------

let train: CompiledTraining<NeRFTiny> | null = null
let infer:    CompiledForward<NeRFTiny> | null = null
let targetRgb: Float32Array | null = null  // [N_PIXELS * 3], the image we're fitting
let running = false
let step = 0

let onStatus: (msg: string) => void = () => {}
let onReconstruction: (rgb: Float32Array) => void = () => {}

function nextBatch(): { coords: Float32Array; rgb: Float32Array; freqs: Float32Array } {
  if (!targetRgb) throw new Error('nextBatch: no target image')
  const coords = new Float32Array(BATCH_SIZE * 2)
  const rgb = new Float32Array(BATCH_SIZE * 3)
  for (let b = 0; b < BATCH_SIZE; b++) {
    const idx = Math.floor(Math.random() * N_PIXELS)
    coords[b * 2]     = GRID_COORDS[idx * 2]!
    coords[b * 2 + 1] = GRID_COORDS[idx * 2 + 1]!
    rgb[b * 3]     = targetRgb[idx * 3]!
    rgb[b * 3 + 1] = targetRgb[idx * 3 + 1]!
    rgb[b * 3 + 2] = targetRgb[idx * 3 + 2]!
  }
  return { coords, rgb, freqs: FREQS }
}

async function renderReconstruction(): Promise<void> {
  if (!infer) return
  const r = await infer.run({ coords: GRID_COORDS, freqs: FREQS })
  if (r.kind === 'completed') onReconstruction(r.output)
}

async function runTraining(): Promise<void> {
  let lastRender = 0
  let lastLoss = 0
  while (running && train) {
    const sr = await train.step(nextBatch())
    if (sr.kind === 'aborted') return
    lastLoss = sr.loss
    step += 1
    if (!Number.isFinite(lastLoss)) {
      onStatus(`step ${step}: loss is ${lastLoss} — NaN, aborting.`)
      running = false
      return
    }
    const now = Date.now()
    if (now - lastRender > 250) {
      lastRender = now
      onStatus(`step ${step}  loss ${lastLoss.toFixed(5)}`)
      await renderReconstruction()
    }
    if (step % 4 === 0) await new Promise(r => setTimeout(r, 0))
  }
}

async function buildGraphs(): Promise<void> {
  onStatus('compiling…')
  const t0 = performance.now()
  train = await compileTraining()
  infer = await train.attach({
    forward: predictFn,
    inputs: {
      coords: [N_PIXELS, 2],
      freqs:  [L_FREQS],
    },
  })
  step = 0
  onStatus(`compiled (${train.kernels.length} kernels, ${(performance.now() - t0).toFixed(0)} ms)`)
}

function startTraining(): void {
  if (running) return
  running = true
  void runTraining()
}

function stopTraining(): void {
  running = false
}

async function resetWeights(): Promise<void> {
  if (!train) return
  const wasRunning = running
  running = false
  await new Promise<void>(r => setTimeout(r, 0))
  await train.reset()
  step = 0
  await renderReconstruction()
  onStatus(`weights re-initialized (seed ${train.seed})`)
  if (wasRunning) { running = true; void runTraining() }
}

function setTargetImage(rgb: Float32Array): void {
  if (rgb.length !== N_PIXELS * 3) {
    throw new Error(`setTargetImage: expected ${N_PIXELS * 3} values, got ${rgb.length}`)
  }
  targetRgb = rgb
}

// ========== UI ==========

const statusEl   = document.getElementById('status')         as HTMLDivElement
const trainBtn   = document.getElementById('train')          as HTMLButtonElement
const stopBtn    = document.getElementById('stop')           as HTMLButtonElement
const resetBtn   = document.getElementById('reset')          as HTMLButtonElement
const uploadBtn  = document.getElementById('upload')         as HTMLInputElement
const targetCanvas = document.getElementById('target-canvas') as HTMLCanvasElement
const reconCanvas  = document.getElementById('recon-canvas')  as HTMLCanvasElement

const PIXEL_SCALE = 4
targetCanvas.width  = reconCanvas.width  = IMG_W * PIXEL_SCALE
targetCanvas.height = reconCanvas.height = IMG_H * PIXEL_SCALE
const targetCtx = targetCanvas.getContext('2d')!
const reconCtx  = reconCanvas.getContext('2d')!
targetCtx.imageSmoothingEnabled = false
reconCtx.imageSmoothingEnabled  = false

// Offscreen 64×64 buffer used for both target and reconstruction; drawImage
// upscales it to the visible canvas.
const small = document.createElement('canvas')
small.width = IMG_W
small.height = IMG_H
const smallCtx = small.getContext('2d')!
const imgData = smallCtx.createImageData(IMG_W, IMG_H)

function rgbToCanvas(rgb: Float32Array, dst: CanvasRenderingContext2D): void {
  for (let i = 0; i < N_PIXELS; i++) {
    const off = i * 4
    imgData.data[off]     = Math.max(0, Math.min(255, Math.round(rgb[i * 3]!     * 255)))
    imgData.data[off + 1] = Math.max(0, Math.min(255, Math.round(rgb[i * 3 + 1]! * 255)))
    imgData.data[off + 2] = Math.max(0, Math.min(255, Math.round(rgb[i * 3 + 2]! * 255)))
    imgData.data[off + 3] = 255
  }
  smallCtx.putImageData(imgData, 0, 0)
  dst.drawImage(small, 0, 0, dst.canvas.width, dst.canvas.height)
}

onStatus = (msg) => { statusEl.textContent = msg }
onReconstruction = (rgb) => rgbToCanvas(rgb, reconCtx)

// ---------------------------------------------------------------------------
// Default target image. Procedurally drawn: radial gradient + the word
// "tensorgrad" centered on it. The gradient gives the network something
// low-frequency to fit fast; the text edges are the high-frequency detail
// that only emerges once the positional encoding's higher bands start
// contributing.
// ---------------------------------------------------------------------------

function makeDefaultImage(): Float32Array {
  const c = document.createElement('canvas')
  c.width = IMG_W
  c.height = IMG_H
  const ctx = c.getContext('2d')!
  const grad = ctx.createRadialGradient(IMG_W / 2, IMG_H / 2, 4, IMG_W / 2, IMG_H / 2, IMG_W * 0.7)
  grad.addColorStop(0, '#f97316')
  grad.addColorStop(1, '#1e293b')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, IMG_W, IMG_H)
  ctx.fillStyle = '#ffffff'
  ctx.font = 'bold 10px ui-sans-serif, system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('tensorgrad', IMG_W / 2, IMG_H / 2)
  const px = ctx.getImageData(0, 0, IMG_W, IMG_H).data
  const rgb = new Float32Array(N_PIXELS * 3)
  for (let i = 0; i < N_PIXELS; i++) {
    rgb[i * 3]     = px[i * 4]!     / 255
    rgb[i * 3 + 1] = px[i * 4 + 1]! / 255
    rgb[i * 3 + 2] = px[i * 4 + 2]! / 255
  }
  return rgb
}

// Load a user-uploaded image. Draws it into the 64×64 offscreen canvas
// (browser handles downscaling), reads back the pixels.
async function loadUploadedImage(file: File): Promise<Float32Array> {
  const bitmap = await createImageBitmap(file)
  smallCtx.clearRect(0, 0, IMG_W, IMG_H)
  smallCtx.drawImage(bitmap, 0, 0, IMG_W, IMG_H)
  const px = smallCtx.getImageData(0, 0, IMG_W, IMG_H).data
  const rgb = new Float32Array(N_PIXELS * 3)
  for (let i = 0; i < N_PIXELS; i++) {
    rgb[i * 3]     = px[i * 4]!     / 255
    rgb[i * 3 + 1] = px[i * 4 + 1]! / 255
    rgb[i * 3 + 2] = px[i * 4 + 2]! / 255
  }
  return rgb
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

resetBtn.addEventListener('click', () => { void resetWeights() })

uploadBtn.addEventListener('change', async () => {
  const file = uploadBtn.files?.[0]
  if (!file) return
  const wasRunning = running
  running = false
  await new Promise<void>(r => setTimeout(r, 0))
  const rgb = await loadUploadedImage(file)
  setTargetImage(rgb)
  rgbToCanvas(rgb, targetCtx)
  if (train) {
    await train.reset()
    step = 0
    await renderReconstruction()
  }
  uploadBtn.value = ''
  onStatus(`loaded ${file.name} (${rgb.length / 3} px) — weights reset`)
  if (wasRunning) { running = true; void runTraining() }
})

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  if (!isWebGPUAvailable()) {
    onStatus('WebGPU not available. Try Chrome 113+ or Safari 17.4+.')
    return
  }
  const rgb = makeDefaultImage()
  setTargetImage(rgb)
  rgbToCanvas(rgb, targetCtx)
  await buildGraphs()
  await renderReconstruction()
  trainBtn.disabled = false
  resetBtn.disabled = false
  uploadBtn.disabled = false
}

boot().catch((e: unknown) => {
  const msg = (e as { message?: string })?.message ?? String(e)
  onStatus(`error: ${msg}`)
  console.error(e)
})
