// MNIST CNN — primary purpose is GPU verification of the conv2d /
// maxPool2d / nn.Conv2d / flatten path. If anything in the WGSL is broken,
// training either crashes at compile, NaNs, or fails to converge.
//
// Architecture: two conv+pool blocks → flatten → MLP head. Standard
// MNIST-CNN template (the kind of model an LLM would write from a PyTorch
// tutorial port).
//
//   [B, 1, 28, 28] -> Conv2d(1,16,3,pad=1) + ReLU
//                  -> MaxPool2D(2)          [B, 16, 14, 14]
//                  -> Conv2d(16,32,3,pad=1) + ReLU
//                  -> MaxPool2D(2)          [B, 32, 7, 7]
//                  -> flatten               [B, 1568]
//                  -> Linear(1568, 64) + ReLU
//                  -> Linear(64, 10)        [B, 10] logits
//
// Expected: test accuracy reaches >97% within a couple thousand batches.
//
// Status is streamed to the vite dev server via POST /__log so you can
// tail it in the terminal instead of copying from the browser console.

import {
  Module, compileModule, isWebGPUAvailable, nn,
  mean, relu, flatten, maxPool2d,
  type Tensor, type CompiledModule, type CompiledForwardModule,
} from 'tensorgrad'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MNIST_PREFIX = 'https://s3.eu-west-2.amazonaws.com/solenya-media/'
const BATCH_SIZE = 64
const EVAL_BATCH = 256
const N_CLASSES = 10
const CONV1_OUT = 16
const CONV2_OUT = 32
const HIDDEN = 64

// UI-supplied sink; assigned in the UI section so the ML side has zero DOM
// dependencies. Default no-op lets this section behave in isolation.
let log: (msg: string) => void = () => {}

// ---------------------------------------------------------------------------
// MNIST loading. Pixels are stored as flat Float32Array; the compileModule
// declared input shape is [B, 1, 28, 28], so per batch we just slice the
// flat buffer — WGSL reads contiguously regardless of declared shape.
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
  const view = new DataView(imgBytes.buffer, imgBytes.byteOffset)
  const count = view.getUint32(4)
  const images = new Float32Array(count * 28 * 28)
  for (let i = 0; i < images.length; i++) images[i] = imgBytes[16 + i]! / 255
  const labels = new Int32Array(count)
  for (let i = 0; i < count; i++) labels[i] = lblBytes[8 + i]!
  return { images, labels, count }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

class CNN extends Module {
  conv1 = new nn.Conv2d(1, CONV1_OUT, 3, { padding: 1 })
  conv2 = new nn.Conv2d(CONV1_OUT, CONV2_OUT, 3, { padding: 1 })
  // After two 2x2 pools: 28 → 14 → 7. Conv2 output is [B, 32, 7, 7] → 1568.
  fc1 = new nn.Linear(CONV2_OUT * 7 * 7, HIDDEN)
  fc2 = new nn.Linear(HIDDEN, N_CLASSES)
}

function forwardLogits(m: CNN, x: Tensor): Tensor {
  let h = relu(m.conv1.fwd(x))     // [B, 16, 28, 28]
  h = maxPool2d(h, 2)              // [B, 16, 14, 14]
  h = relu(m.conv2.fwd(h))         // [B, 32, 14, 14]
  h = maxPool2d(h, 2)              // [B, 32, 7, 7]
  h = flatten(h, 1)                // [B, 1568]
  h = relu(m.fc1.fwd(h))           // [B, 64]
  return m.fc2.fwd(h)              // [B, 10] logits
}

function lossFn(m: CNN, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  return mean(nn.crossEntropy(forwardLogits(m, x), y))
}

function predictFn(m: CNN, { x }: { x: Tensor }): Tensor {
  return forwardLogits(m, x)
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let compiled: CompiledModule<CNN> | null = null
let predict: CompiledForwardModule<CNN> | null = null
let trainData: MnistSet, testData: MnistSet
let trainOrder: number[] = []
let trainCursor = 0
let running = true
let step = 0

function shuffleOrder(arr: number[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j]!, arr[i]!]
  }
}

function nextTrainBatch(): { x: Float32Array; y: Int32Array } {
  const x = new Float32Array(BATCH_SIZE * 28 * 28)
  const y = new Int32Array(BATCH_SIZE)
  for (let b = 0; b < BATCH_SIZE; b++) {
    if (trainCursor >= trainOrder.length) { shuffleOrder(trainOrder); trainCursor = 0 }
    const idx = trainOrder[trainCursor++]!
    for (let j = 0; j < 28 * 28; j++) x[b * 28 * 28 + j] = trainData.images[idx * 28 * 28 + j]!
    y[b] = trainData.labels[idx]!
  }
  return { x, y }
}

async function probeAccuracy(): Promise<number> {
  if (!predict) return 0
  const x = new Float32Array(EVAL_BATCH * 28 * 28)
  const truth = new Int32Array(EVAL_BATCH)
  for (let b = 0; b < EVAL_BATCH; b++) {
    const idx = Math.floor(Math.random() * testData.count)
    for (let j = 0; j < 28 * 28; j++) x[b * 28 * 28 + j] = testData.images[idx * 28 * 28 + j]!
    truth[b] = testData.labels[idx]!
  }
  const logits = await predict.run({ x })
  let correct = 0
  for (let b = 0; b < EVAL_BATCH; b++) {
    let best = 0
    const off = b * N_CLASSES
    for (let c = 1; c < N_CLASSES; c++) if (logits[off + c]! > logits[off + best]!) best = c
    if (best === truth[b]) correct++
  }
  return correct / EVAL_BATCH
}

async function runTraining(): Promise<void> {
  let lastEval = 0
  let lastLoss = 0
  while (running && compiled) {
    const batch = nextTrainBatch()
    lastLoss = await compiled.step(batch)
    step += 1
    if (!Number.isFinite(lastLoss)) {
      log(`step ${step}: loss is ${lastLoss} — training NaN'd. WGSL bug likely.`)
      running = false
      return
    }
    const now = Date.now()
    if (now - lastEval > 2000) {
      lastEval = now
      const acc = await probeAccuracy()
      log(`step ${step}  loss ${lastLoss.toFixed(4)}  test acc ${(acc * 100).toFixed(1)}%`)
    } else if (step % 20 === 0) {
      log(`step ${step}  loss ${lastLoss.toFixed(4)}`)
    }
    if (step % 4 === 0) await new Promise(r => setTimeout(r, 0))
  }
}

// ---------------------------------------------------------------------------
// UI + boot
// ---------------------------------------------------------------------------

const statusEl = document.getElementById('status') as HTMLDivElement

log = (msg) => {
  statusEl.textContent = msg
  // Stream to dev server so it lands in terminal stdout.
  void fetch('/__log', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ msg }),
  })
}

async function boot(): Promise<void> {
  if (!isWebGPUAvailable()) {
    log('WebGPU not available. Use Chrome 113+ or Safari 17.4+.')
    return
  }
  log('loading MNIST…')
  ;[trainData, testData] = await Promise.all([
    loadSet('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
    loadSet('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'),
  ])
  trainOrder = Array.from({ length: trainData.count }, (_, i) => i)
  shuffleOrder(trainOrder)

  log('compiling CNN…')
  const t0 = performance.now()
  compiled = await compileModule({
    factory: () => new CNN(),
    loss: lossFn,
    adam: { lr: 1e-3, weightDecay: 0.01, clipGradNorm: 1.0 },
    inputs: {
      x: [BATCH_SIZE, 1, 28, 28],
      y: { shape: [BATCH_SIZE], dtype: 'i32' },
    },
  })
  predict = await compiled.compileForward({
    forward: predictFn,
    inputs: { x: [EVAL_BATCH, 1, 28, 28] },
  })
  log(`compiled in ${(performance.now() - t0).toFixed(0)} ms (${compiled.kernelCount} kernels) — training…`)

  void runTraining()
}

boot().catch((e: unknown) => {
  const msg = (e as { message?: string })?.message ?? String(e)
  log(`error: ${msg}`)
  console.error(e)
})
