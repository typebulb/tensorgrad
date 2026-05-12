// Variational autoencoder on MNIST. Encoder MLP maps each image to a
// distribution over latent codes (μ, log σ²); the decoder reverses it. The
// trick — reparameterization — lets the gradient flow through the sampling
// step by writing `z = μ + σ · ε` where `ε ~ N(0, 1)` is treated as an
// input rather than a draw, so log-prob math stays differentiable in μ and σ.
// `randn` shipped exactly for this.
//
// Three demos in the page:
//   1. A 3×3 grid of random samples — decode random latent vectors. Watch
//      these go from noise to recognisable digits as training proceeds.
//   2. Latent interpolation: pick two test digits, project them to their
//      latent means, slide between them, decode at each step. The smooth
//      morph is the canonical "look at the latent space" demo.
//   3. Loss curve in status text (recon + KL).
//
// File layout: ML + app logic at the top, UI at the bottom. Same convention
// as the other samples.

import {
  Module, compile, isWebGPUAvailable, nn,
  add, sub, mul, sum, exp, sigmoid, relu,
  randn, square,
  type Tensor, type CompiledTraining, type CompiledForward,
} from 'tensorgrad'

// ============================================================================
//                          MODEL / TRAINING
// ============================================================================

const MNIST_PREFIX = 'https://s3.eu-west-2.amazonaws.com/solenya-media/'
const INPUT_DIM = 28 * 28
const LATENT_DIM = 8
const HIDDEN = 256
const BATCH_SIZE = 128
const N_SAMPLES = 9                   // 3×3 grid
const BETA = 1.0                       // KL weight (vanilla VAE)

// ---------------------------------------------------------------------------
// MNIST loading. Same idx-ubyte/gzip pattern as the other MNIST samples.
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
  const images = new Float32Array(count * INPUT_DIM)
  for (let i = 0; i < images.length; i++) images[i] = bytes[16 + i]! / 255
  return { images, count }
}

// ---------------------------------------------------------------------------
// Model. MLP encoder + decoder; separate heads for μ and log σ².
// ---------------------------------------------------------------------------

class VAE extends Module {
  enc1     = new nn.Linear(INPUT_DIM, HIDDEN)
  enc2     = new nn.Linear(HIDDEN,    HIDDEN)
  encMu    = new nn.Linear(HIDDEN,    LATENT_DIM)
  encLogV  = new nn.Linear(HIDDEN,    LATENT_DIM)
  dec1     = new nn.Linear(LATENT_DIM, HIDDEN)
  dec2     = new nn.Linear(HIDDEN,    HIDDEN)
  decOut   = new nn.Linear(HIDDEN,    INPUT_DIM)
}

function encoder(m: VAE, x: Tensor): { mu: Tensor; logVar: Tensor } {
  let h = relu(m.enc1.fwd(x))
  h = relu(m.enc2.fwd(h))
  return { mu: m.encMu.fwd(h), logVar: m.encLogV.fwd(h) }
}

function decoder(m: VAE, z: Tensor): Tensor {
  let h = relu(m.dec1.fwd(z))
  h = relu(m.dec2.fwd(h))
  return sigmoid(m.decOut.fwd(h))
}

// Training loss = sum-over-pixels recon + β · sum-over-latent KL, both
// averaged over the batch. KL closed form against N(0, 1):
// `KL = -0.5 · sum(1 + log σ² − μ² − σ²)`. Recon is MSE in pixel space —
// simpler than BCE-with-logits and converges fine on MNIST.
function lossFn(m: VAE, { x }: { x: Tensor }): Tensor {
  const B = x.shape[0]!
  const { mu, logVar } = encoder(m, x)
  // Reparameterize: z = μ + σ · ε, ε ~ N(0, 1).
  const eps = randn([B, LATENT_DIM])
  const sigma = exp(mul(logVar, 0.5))
  const z = add(mu, mul(sigma, eps))
  const xHat = decoder(m, z)
  // Recon: per-batch mean of per-image sum of squared pixel error.
  const recon = mul(sum(square(sub(xHat, x))), 1 / B)
  // KL: per-batch mean of per-image sum over the latent axis.
  const klElem = mul(sub(sub(add(logVar, 1), square(mu)), exp(logVar)), -0.5)
  const kl = mul(sum(klElem), 1 / B)
  return add(recon, mul(kl, BETA))
}

// Inference forwards. We expose encode-to-mean (drops σ — only the mode is
// needed for the interpolation demo) and decode-from-latent (used for both
// the random-samples grid and the interpolation output).
function encodeFn(m: VAE, { x }: { x: Tensor }): Tensor {
  return encoder(m, x).mu
}

function decodeFn(m: VAE, { z }: { z: Tensor }): Tensor {
  return decoder(m, z)
}

// ---------------------------------------------------------------------------
// State + lifecycle
// ---------------------------------------------------------------------------

let train: CompiledTraining<VAE> | null = null
let encode:   CompiledForward<VAE> | null = null
let decode:   CompiledForward<VAE> | null = null
let trainData: MnistSet | null = null
let testData:  MnistSet | null = null
let trainOrder: number[] = []
let trainCursor = 0
let running = false
let step = 0

let onStatus: (msg: string) => void = () => {}
let onSamples: (rgb: Float32Array) => void = () => {}
let onInterp: (rgb: Float32Array) => void = () => {}

function shuffleOrder(arr: number[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j]!, arr[i]!]
  }
}

function nextBatch(): { x: Float32Array } {
  if (!trainData) throw new Error('nextBatch: no data')
  const x = new Float32Array(BATCH_SIZE * INPUT_DIM)
  for (let b = 0; b < BATCH_SIZE; b++) {
    if (trainCursor >= trainOrder.length) { shuffleOrder(trainOrder); trainCursor = 0 }
    const idx = trainOrder[trainCursor++]!
    x.set(trainData.images.subarray(idx * INPUT_DIM, (idx + 1) * INPUT_DIM), b * INPUT_DIM)
  }
  return { x }
}

// Box-Muller into a buffer for the random-sample-grid latent draws.
function fillRandn(out: Float32Array): void {
  for (let i = 0; i < out.length; i += 2) {
    const u1 = Math.max(1e-10, Math.random())
    const u2 = Math.random()
    const mag = Math.sqrt(-2 * Math.log(u1))
    out[i] = mag * Math.cos(2 * Math.PI * u2)
    if (i + 1 < out.length) out[i + 1] = mag * Math.sin(2 * Math.PI * u2)
  }
}

async function renderSamples(): Promise<void> {
  if (!decode) return
  const z = new Float32Array(N_SAMPLES * LATENT_DIM)
  fillRandn(z)
  const r = await decode.run({ z })
  if (r.kind === 'completed') onSamples(r.output)
}

// Two endpoint latents for the interpolation demo. Picked from random test
// images; re-rolled on demand.
let interpA = new Float32Array(LATENT_DIM)
let interpB = new Float32Array(LATENT_DIM)
let interpAImg = new Float32Array(INPUT_DIM)
let interpBImg = new Float32Array(INPUT_DIM)

async function rerollInterpEndpoints(): Promise<void> {
  if (!encode || !testData) return
  const x = new Float32Array(2 * INPUT_DIM)
  const idxA = Math.floor(Math.random() * testData.count)
  const idxB = Math.floor(Math.random() * testData.count)
  x.set(testData.images.subarray(idxA * INPUT_DIM, (idxA + 1) * INPUT_DIM), 0)
  x.set(testData.images.subarray(idxB * INPUT_DIM, (idxB + 1) * INPUT_DIM), INPUT_DIM)
  const mr = await encode.run({ x })   // [2, LATENT_DIM]
  if (mr.kind === 'aborted') return
  const mus = mr.output
  interpA = new Float32Array(mus.subarray(0, LATENT_DIM))
  interpB = new Float32Array(mus.subarray(LATENT_DIM, 2 * LATENT_DIM))
  interpAImg = new Float32Array(x.subarray(0, INPUT_DIM))
  interpBImg = new Float32Array(x.subarray(INPUT_DIM, 2 * INPUT_DIM))
}

async function renderInterp(t: number): Promise<void> {
  if (!decode) return
  const z = new Float32Array(LATENT_DIM)
  for (let i = 0; i < LATENT_DIM; i++) z[i] = (1 - t) * interpA[i]! + t * interpB[i]!
  const r = await decode.run({ z })
  if (r.kind === 'completed') onInterp(r.output)
}

async function runTraining(): Promise<void> {
  let lastEval = 0
  while (running && train) {
    const sr = await train.step(nextBatch())
    if (sr.kind === 'aborted') return
    const loss = sr.loss
    step += 1
    if (!Number.isFinite(loss)) {
      onStatus(`step ${step}: loss is ${loss} — NaN, aborting.`)
      running = false
      return
    }
    const now = Date.now()
    if (now - lastEval > 500) {
      lastEval = now
      onStatus(`step ${step}  loss ${loss.toFixed(3)}`)
      await renderSamples()
      await renderCurrentInterp()
    }
    if (step % 4 === 0) await new Promise(r => setTimeout(r, 0))
  }
}

async function loadMnist(): Promise<void> {
  ;[trainData, testData] = await Promise.all([
    loadImages('train-images-idx3-ubyte.gz'),
    loadImages('t10k-images-idx3-ubyte.gz'),
  ])
  trainOrder = Array.from({ length: trainData.count }, (_, i) => i)
  shuffleOrder(trainOrder)
}

async function buildGraphs(): Promise<void> {
  onStatus('compiling…')
  const t0 = performance.now()
  const model = new VAE()
  train = await compile({
    model,
    loss: lossFn,
    optimizer: { kind: 'adam', lr: 1e-3, clipGradNorm: 1.0 },
    inputs: { x: [BATCH_SIZE, INPUT_DIM] },
  })
  // Polymorphic batch for both inference proxies: encode is called at B=2
  // (interpolation endpoints), decode at B=1 (interpolation output) and
  // B=N_SAMPLES (random grid).
  encode = await train.attach({
    forward: encodeFn,
    inputs: { x: [null, INPUT_DIM] },
  })
  decode = await train.attach({
    forward: decodeFn,
    inputs: { z: [null, LATENT_DIM] },
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
  await rerollInterpEndpoints()
  await renderSamples()
  await renderCurrentInterp()
  onStatus(`weights re-initialized (seed ${train.seed})`)
  if (wasRunning) { running = true; void runTraining() }
}

// ============================================================================
//                                   UI
// ============================================================================

const statusEl     = document.getElementById('status')         as HTMLDivElement
const trainBtn     = document.getElementById('train')          as HTMLButtonElement
const stopBtn      = document.getElementById('stop')           as HTMLButtonElement
const resetBtn     = document.getElementById('reset')          as HTMLButtonElement
const rerollBtn    = document.getElementById('reroll')         as HTMLButtonElement
const samplesCanvas  = document.getElementById('samples-canvas')  as HTMLCanvasElement
const interpACanvas  = document.getElementById('interp-a-canvas') as HTMLCanvasElement
const interpBCanvas  = document.getElementById('interp-b-canvas') as HTMLCanvasElement
const interpOutCanvas = document.getElementById('interp-out-canvas') as HTMLCanvasElement
const interpSlider   = document.getElementById('interp-slider') as HTMLInputElement

const PIXEL_SCALE = 3
const GRID_N = 3

// Samples canvas shows a 3×3 mosaic; interp canvases each show one image.
samplesCanvas.width  = 28 * GRID_N * PIXEL_SCALE
samplesCanvas.height = 28 * GRID_N * PIXEL_SCALE
for (const c of [interpACanvas, interpBCanvas, interpOutCanvas]) {
  c.width  = 28 * PIXEL_SCALE
  c.height = 28 * PIXEL_SCALE
  c.getContext('2d')!.imageSmoothingEnabled = false
}
samplesCanvas.getContext('2d')!.imageSmoothingEnabled = false
const samplesCtx = samplesCanvas.getContext('2d')!

// Offscreen 28×28 buffer reused for every render.
const small = document.createElement('canvas')
small.width = 28
small.height = 28
const smallCtx = small.getContext('2d')!
const imgData = smallCtx.createImageData(28, 28)

function drawImage(rgb: Float32Array, offset: number, dst: CanvasRenderingContext2D, dx: number, dy: number, dw: number, dh: number): void {
  for (let i = 0; i < 28 * 28; i++) {
    const v = Math.max(0, Math.min(255, Math.round(rgb[offset + i]! * 255)))
    const o = i * 4
    imgData.data[o]     = v
    imgData.data[o + 1] = v
    imgData.data[o + 2] = v
    imgData.data[o + 3] = 255
  }
  smallCtx.putImageData(imgData, 0, 0)
  dst.drawImage(small, dx, dy, dw, dh)
}

onStatus = (msg) => { statusEl.textContent = msg }

onSamples = (flat) => {
  // flat: [N_SAMPLES * 784], laid out as N consecutive 28×28 images.
  const tile = 28 * PIXEL_SCALE
  for (let i = 0; i < N_SAMPLES; i++) {
    const row = Math.floor(i / GRID_N)
    const col = i % GRID_N
    drawImage(flat, i * INPUT_DIM, samplesCtx, col * tile, row * tile, tile, tile)
  }
}

onInterp = (flat) => {
  const ctx = interpOutCanvas.getContext('2d')!
  drawImage(flat, 0, ctx, 0, 0, interpOutCanvas.width, interpOutCanvas.height)
}

async function renderCurrentInterp(): Promise<void> {
  // Endpoint canvases show the original test images (constant per reroll).
  drawImage(interpAImg, 0, interpACanvas.getContext('2d')!, 0, 0, interpACanvas.width, interpACanvas.height)
  drawImage(interpBImg, 0, interpBCanvas.getContext('2d')!, 0, 0, interpBCanvas.width, interpBCanvas.height)
  await renderInterp(parseFloat(interpSlider.value))
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

rerollBtn.addEventListener('click', async () => {
  await rerollInterpEndpoints()
  await renderCurrentInterp()
})

interpSlider.addEventListener('input', () => { void renderInterp(parseFloat(interpSlider.value)) })

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
  await rerollInterpEndpoints()
  await renderSamples()
  await renderCurrentInterp()
  trainBtn.disabled = false
  resetBtn.disabled = false
  rerollBtn.disabled = false
}

boot().catch((e: unknown) => {
  const msg = (e as { message?: string })?.message ?? String(e)
  onStatus(`error: ${msg}`)
  console.error(e)
})
