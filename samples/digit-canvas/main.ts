// Digit Canvas: MNIST classifier trained live in the browser.
//
// This sample is intended as the broadest demonstration of the tensorgrad
// API surface in one place. It exercises:
//
//   * compileForward with a parametric batch dim (`null` wildcard) — the
//     same predict graph serves B=1 for the canvas and B=EVAL_BATCH for
//     the periodic accuracy probe. One compileForward, two resolved shapes.
//   * dropout in the training forward only, omitted from the inference
//     forward. No `.train()/.eval()` mode flag — the two are separate
//     forward functions, each compiled into its own graph. Dropout is
//     literally absent from the inference path.
//   * replaceModel for changing the hidden layer size without spawning a
//     new worker. The sibling forward proxy stays valid; its per-shape
//     kernel cache recompiles lazily on the next run.
//   * setOptimizerConfig for changing the learning rate mid-training.
//   * clipGradNorm for training stability, baked into AdamConfig.
//   * singleFlight wrapping the canvas predict so rapid-stroke predictions
//     don't queue up — only the latest call resolves; older callers reject
//     with AbortError.
//   * mean(crossEntropyLast(logits, targets)) as the canonical
//     classification loss tail.
//
// MNIST data is served from solenya-media S3 — same URLs the in-repo bulbs
// use. ~11 MB on first load; cached after that.

import {
  Module, compileModule, isWebGPUAvailable, nn,
  mean, relu, dropout, softmaxLast, singleFlight,
  type Tensor, type CompiledModule, type CompiledForwardModule,
} from 'tensorgrad'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MNIST_PREFIX = 'https://s3.eu-west-2.amazonaws.com/solenya-media/'
const INPUT_DIM = 784       // 28 * 28 grayscale pixels
const N_CLASSES = 10
const BATCH_SIZE = 64       // training-step batch
const EVAL_BATCH = 256      // per-probe test-accuracy sample size
const DROP_P = 0.1          // dropout probability on hidden activations

// ---------------------------------------------------------------------------
// MNIST loading. The files are the standard 60k-train / 10k-test idx-ubyte
// format, gzipped. We use the browser's DecompressionStream (Chrome 80+,
// Firefox 113+, Safari 16.4+) instead of a userspace gzip lib.
// ---------------------------------------------------------------------------

interface MnistSet { images: Float32Array; labels: Int32Array; count: number }

async function inflateGzip(buf: ArrayBuffer): Promise<Uint8Array> {
  const stream = new Response(buf).body!.pipeThrough(new DecompressionStream('gzip'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

async function loadSet(imgFile: string, lblFile: string): Promise<MnistSet> {
  const [imgBuf, lblBuf] = await Promise.all([
    fetch(MNIST_PREFIX + imgFile).then(r => r.arrayBuffer()),
    fetch(MNIST_PREFIX + lblFile).then(r => r.arrayBuffer()),
  ])
  const imgBytes = await inflateGzip(imgBuf)
  const lblBytes = await inflateGzip(lblBuf)
  // idx-ubyte format: 16-byte image header (magic, count, rows, cols);
  // 8-byte label header (magic, count). Pixels are u8 0..255.
  const view = new DataView(imgBytes.buffer, imgBytes.byteOffset)
  const count = view.getUint32(4)
  const images = new Float32Array(count * INPUT_DIM)
  for (let i = 0; i < images.length; i++) images[i] = imgBytes[16 + i]! / 255
  const labels = new Int32Array(count)
  for (let i = 0; i < count; i++) labels[i] = lblBytes[8 + i]!
  return { images, labels, count }
}

// ---------------------------------------------------------------------------
// Model. Plain MLP — INPUT_DIM → hidden → N_CLASSES with ReLU activations.
// Train- and inference-forward share the structural pass through layers,
// differing only in whether dropout is applied between hidden activations.
// ---------------------------------------------------------------------------

class MLP extends Module {
  layers: nn.Linear[]
  constructor(sizes: readonly number[]) {
    super()
    this.layers = []
    for (let i = 0; i < sizes.length - 1; i++) {
      this.layers.push(new nn.Linear(sizes[i]!, sizes[i + 1]!))
    }
  }
}

function netFwd(m: MLP, x: Tensor, applyDropout: boolean): Tensor {
  let h = x
  for (let i = 0; i < m.layers.length; i++) {
    h = m.layers[i]!.fwd(h)
    if (i < m.layers.length - 1) {
      h = relu(h)
      if (applyDropout) h = dropout(h, DROP_P)
    }
  }
  return h
}

function lossFn(m: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  // Training-only: dropout active on hidden activations.
  const logits = netFwd(m, x, true)
  return mean(nn.crossEntropyLast(logits, y))
}

function predictFn(m: MLP, { x }: { x: Tensor }): Tensor {
  // Inference: no dropout. Returns softmax probabilities so the canvas can
  // show a confidence breakdown without a second compileForward call.
  return softmaxLast(netFwd(m, x, false))
}

// ---------------------------------------------------------------------------
// DOM references + state
// ---------------------------------------------------------------------------

const canvas = document.getElementById('draw') as HTMLCanvasElement
const ctx = canvas.getContext('2d')!
const trainBtn = document.getElementById('train') as HTMLButtonElement
const stopBtn = document.getElementById('stop') as HTMLButtonElement
const clearBtn = document.getElementById('clear') as HTMLButtonElement
const resetBtn = document.getElementById('reset-weights') as HTMLButtonElement
const layerSelect = document.getElementById('layer-size') as HTMLSelectElement
const lrSelect = document.getElementById('lr') as HTMLSelectElement
const statusEl = document.getElementById('status') as HTMLDivElement
const predEl = document.getElementById('prediction') as HTMLDivElement
const probsEl = document.getElementById('probs') as HTMLDivElement

let trainData: MnistSet
let testData: MnistSet
let trainOrder: number[] = []
let trainCursor = 0

// `any` here is the polymorphic-shape input type that includes `null` —
// the public type signature doesn't yet narrow nicely with wildcards.
let compiled: CompiledModule<MLP, { x: readonly [number, number]; y: { shape: readonly [number]; dtype: 'i32' } }> | null = null
let predict: CompiledForwardModule<MLP, { x: readonly [null, number] }> | null = null
let predictCanvas: ((input: Float32Array) => Promise<Float32Array>) | null = null

let running = false
let trainingActive = false
let step = 0

// ---------------------------------------------------------------------------
// Canvas drawing — pointer events, white strokes on a black background.
// ---------------------------------------------------------------------------

function resetCanvas(): void {
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
}
resetCanvas()
ctx.strokeStyle = '#fff'
ctx.lineWidth = 20
ctx.lineCap = 'round'
ctx.lineJoin = 'round'

let drawing = false
function pos(e: PointerEvent): { x: number; y: number } {
  const r = canvas.getBoundingClientRect()
  return {
    x: (e.clientX - r.left) * (canvas.width / r.width),
    y: (e.clientY - r.top) * (canvas.height / r.height),
  }
}
canvas.addEventListener('pointerdown', e => {
  drawing = true
  canvas.setPointerCapture(e.pointerId)
  const p = pos(e)
  ctx.beginPath()
  ctx.moveTo(p.x, p.y)
  // Single tap should leave a dot — draw a tiny segment.
  ctx.lineTo(p.x + 0.1, p.y + 0.1)
  ctx.stroke()
  schedulePredict()
})
canvas.addEventListener('pointermove', e => {
  if (!drawing) return
  const p = pos(e)
  ctx.lineTo(p.x, p.y)
  ctx.stroke()
  schedulePredict()
})
canvas.addEventListener('pointerup', e => {
  drawing = false
  try { canvas.releasePointerCapture(e.pointerId) } catch { /* ignore */ }
  schedulePredict()
})
clearBtn.addEventListener('click', () => {
  resetCanvas()
  predEl.textContent = ''
  probsEl.innerHTML = ''
})

// Capture the 280×280 canvas as a 28×28 Float32Array in [0, 1]. Browser
// downscaling handles the antialiasing; we read the red channel (== green
// == blue since we drew white on black).
function captureInput(): Float32Array {
  const small = document.createElement('canvas')
  small.width = 28
  small.height = 28
  const sctx = small.getContext('2d')!
  sctx.imageSmoothingEnabled = true
  sctx.drawImage(canvas, 0, 0, 28, 28)
  const data = sctx.getImageData(0, 0, 28, 28).data
  const out = new Float32Array(INPUT_DIM)
  for (let i = 0; i < INPUT_DIM; i++) out[i] = data[i * 4]! / 255
  return out
}

// ---------------------------------------------------------------------------
// Prediction — schedule on each pointermove via rAF, run via singleFlight so
// rapid strokes never queue. Older calls reject with AbortError when a
// newer one supersedes; we ignore those.
// ---------------------------------------------------------------------------

let predictScheduled = false
function schedulePredict(): void {
  if (predictScheduled || !predictCanvas) return
  predictScheduled = true
  requestAnimationFrame(async () => {
    predictScheduled = false
    if (!predictCanvas) return
    try {
      const probs = await predictCanvas(captureInput())
      updatePredictionUI(probs)
    } catch (e: unknown) {
      const err = e as { name?: string }
      if (err?.name === 'AbortError') return  // superseded — newer call will paint
      console.error('predict error:', e)
    }
  })
}

function updatePredictionUI(probs: Float32Array): void {
  let best = 0
  for (let i = 1; i < N_CLASSES; i++) if (probs[i]! > probs[best]!) best = i
  predEl.textContent = String(best)
  const rows: string[] = []
  for (let i = 0; i < N_CLASSES; i++) {
    const p = probs[i]!
    rows.push(
      `<div><span class="d">${i}</span>` +
      `<span class="bar" style="width:${(p * 100).toFixed(1)}%"></span>` +
      `<span class="pct">${(p * 100).toFixed(1)}%</span></div>`,
    )
  }
  probsEl.innerHTML = rows.join('')
}

// ---------------------------------------------------------------------------
// Training loop — assembles a batch from the shuffled order, calls
// compiled.step(), yields to the UI between dispatches. Periodic accuracy
// probe via the same predict graph at B=EVAL_BATCH (parametric batch
// triggers a sibling compile on first call, then cache-hits forever).
// ---------------------------------------------------------------------------

function shuffleOrder(arr: number[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j]!, arr[i]!]
  }
}

function nextTrainBatch(): { x: Float32Array; y: Int32Array } {
  const x = new Float32Array(BATCH_SIZE * INPUT_DIM)
  const y = new Int32Array(BATCH_SIZE)
  for (let b = 0; b < BATCH_SIZE; b++) {
    if (trainCursor >= trainOrder.length) { shuffleOrder(trainOrder); trainCursor = 0 }
    const idx = trainOrder[trainCursor++]!
    for (let j = 0; j < INPUT_DIM; j++) x[b * INPUT_DIM + j] = trainData.images[idx * INPUT_DIM + j]!
    y[b] = trainData.labels[idx]!
  }
  return { x, y }
}

async function probeAccuracy(): Promise<number> {
  if (!predict) return 0
  const x = new Float32Array(EVAL_BATCH * INPUT_DIM)
  const truth = new Int32Array(EVAL_BATCH)
  for (let b = 0; b < EVAL_BATCH; b++) {
    const idx = Math.floor(Math.random() * testData.count)
    for (let j = 0; j < INPUT_DIM; j++) x[b * INPUT_DIM + j] = testData.images[idx * INPUT_DIM + j]!
    truth[b] = testData.labels[idx]!
  }
  // First call at B=EVAL_BATCH triggers a sibling compile + cache;
  // subsequent calls hit the cache.
  const probs = await predict.run({ x })
  let correct = 0
  for (let b = 0; b < EVAL_BATCH; b++) {
    let best = 0
    const off = b * N_CLASSES
    for (let c = 1; c < N_CLASSES; c++) if (probs[off + c]! > probs[off + best]!) best = c
    if (best === truth[b]) correct++
  }
  return correct / EVAL_BATCH
}

async function runTraining(): Promise<void> {
  if (trainingActive) return
  trainingActive = true
  let lastEval = 0
  let lastLoss = 0
  try {
    while (running && compiled) {
      const batch = nextTrainBatch()
      const r = await compiled.step(batch, { onAbort: 'value' })
      if (r.kind === 'aborted') return   // graph was replaced; quietly bail
      lastLoss = r.loss
      step += 1
      const now = Date.now()
      if (now - lastEval > 1000) {
        lastEval = now
        const acc = await probeAccuracy()
        statusEl.textContent =
          `step ${step}  loss ${lastLoss.toFixed(4)}  test acc ${(acc * 100).toFixed(1)}%`
      } else if (step % 10 === 0) {
        statusEl.textContent = `step ${step}  loss ${lastLoss.toFixed(4)}`
      }
      // Yield every few steps so the canvas predict + UI stay responsive.
      if (step % 4 === 0) await new Promise(r => setTimeout(r, 0))
    }
  } finally {
    trainingActive = false
  }
}

// ---------------------------------------------------------------------------
// Compile / lifecycle. Build the training graph + one polymorphic inference
// graph at startup. Layer-size dropdown calls replaceModel; LR dropdown
// calls setOptimizerConfig.
// ---------------------------------------------------------------------------

function currentLayerSpec(): number[] {
  const hidden = parseInt(layerSelect.value, 10)
  return [INPUT_DIM, hidden, N_CLASSES]
}

async function compile(): Promise<void> {
  const layers = currentLayerSpec()
  statusEl.textContent = `compiling MLP ${layers.join(' → ')}…`
  const t0 = performance.now()
  compiled = await compileModule({
    factory: () => new MLP(layers),
    loss: lossFn,
    adam: {
      lr: parseFloat(lrSelect.value),
      weightDecay: 0.01,
      clipGradNorm: 1.0,
    },
    inputs: {
      x: [BATCH_SIZE, INPUT_DIM],
      y: { shape: [BATCH_SIZE], dtype: 'i32' },
    },
  })
  // One polymorphic inference graph — the same predict serves the canvas
  // (B=1) and the accuracy probe (B=EVAL_BATCH).
  predict = await compiled.compileForward({
    forward: predictFn,
    inputs: { x: [null, INPUT_DIM] },
  })
  // singleFlight: rapid strokes supersede each other; only the latest call
  // resolves. Captured via closure so the wrapper survives replaceModel
  // (which invalidates the per-shape kernel cache, not the proxy object).
  const inferRef = predict
  predictCanvas = singleFlight(async (input: Float32Array) => inferRef.run({ x: input }))
  step = 0
  statusEl.textContent =
    `ready (${compiled.kernelCount} kernels, ${(performance.now() - t0).toFixed(0)} ms, seed ${compiled.seed})`
}

async function changeLayerSize(): Promise<void> {
  if (!compiled) return
  const wasRunning = running
  running = false
  // Wait one tick so any in-flight step finishes and the training loop exits.
  await new Promise<void>(r => setTimeout(r, 0))
  const layers = currentLayerSpec()
  statusEl.textContent = `replacing model with ${layers.join(' → ')}…`
  await compiled.replaceModel(() => new MLP(layers))
  // The forward proxy (predict) stays the same object — its per-shape kernel
  // cache was invalidated, so the next run() recompiles against the new
  // topology. predictCanvas (singleFlight wrapper) is still valid.
  step = 0
  statusEl.textContent = `replaced (seed ${compiled.seed})`
  resetBtn.disabled = false
  if (wasRunning) { running = true; void runTraining() }
}

async function changeLR(): Promise<void> {
  if (!compiled) return
  await compiled.setOptimizerConfig({ lr: parseFloat(lrSelect.value) })
}

async function resetWeights(): Promise<void> {
  if (!compiled) return
  const wasRunning = running
  running = false
  await new Promise<void>(r => setTimeout(r, 0))
  await compiled.reset()
  step = 0
  statusEl.textContent = `weights re-initialized (seed ${compiled.seed})`
  if (wasRunning) { running = true; void runTraining() }
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

trainBtn.addEventListener('click', () => {
  if (running) return
  running = true
  trainBtn.disabled = true
  stopBtn.disabled = false
  void runTraining()
})

stopBtn.addEventListener('click', () => {
  running = false
  trainBtn.disabled = false
  stopBtn.disabled = true
})

layerSelect.addEventListener('change', () => { void changeLayerSize() })
lrSelect.addEventListener('change', () => { void changeLR() })
resetBtn.addEventListener('click', () => { void resetWeights() })

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  if (!isWebGPUAvailable()) {
    statusEl.textContent = 'WebGPU not available in this browser. Try Chrome 113+ or Safari 17.4+.'
    trainBtn.disabled = true
    return
  }
  statusEl.textContent = 'loading MNIST (~11 MB, cached after first load)…'
  ;[trainData, testData] = await Promise.all([
    loadSet('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
    loadSet('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'),
  ])
  trainOrder = Array.from({ length: trainData.count }, (_, i) => i)
  shuffleOrder(trainOrder)
  await compile()
}

boot().catch((e: unknown) => {
  const msg = (e as { message?: string })?.message ?? String(e)
  statusEl.textContent = `error: ${msg}`
  console.error(e)
})
