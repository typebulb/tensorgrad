---
format: typebulb/v1
name: Neural Network (tensorgrad)
---

**code.tsx**

```tsx
import presetWind3 from '@unocss/preset-wind3'
import { App, Component, div, h1, button, span, p, svg, circle, line, path, inputSelect, formField, type VElement, type SelectOption } from 'domeleon'
import { UnoThemeManager, type ThemeProxy } from 'domeleon/unocss'
import { inflate } from 'pako'
import * as fabric from 'fabric'

import {
  Module, compile, isWebGPUAvailable, Linear, crossEntropy, relu, singleFlight,
  type Tensor, type CompiledTraining, type CompiledForward,
} from 'tensorgrad'

// ---------- Constants ----------
const INPUT_DIM = 784           // 28 * 28 MNIST pixels
const N_CLASSES = 10            // digits 0..9
// GPU batch size for both training and eval. 64 keeps the gradient noisy enough
// that the "stochastic" feel survives while amortizing the per-step mapAsync cost.
const BATCH_SIZE = 64
const TALLY_WINDOW = 10000      // sliding window for train/test success ratio
// Run an accuracy eval every Nth training batch. Each eval is one extra forward
// pass at B=BATCH_SIZE on a fresh sample — cheap, but doing it every step would
// double GPU work.
const EVAL_EVERY_K_BATCHES = 5
// Per-layer-pair cap on rendered connection lines. Input→hidden alone is up to
// 784 × 50 = 39,200 candidates above the |w| > 0.2 threshold; rendering each
// as its own <line> kills the DOM. We keep the top-K by |w| per layer pair.
const MAX_CONN_PER_LAYER = 300

const range = (n: number) => Array.from({ length: n }, (_, i) => i)
const clamp = (n: number, min: number, max: number) => Math.min(Math.max(n, min), max)

// ---------- Network ----------
class NeuralNetwork extends Module {
  layers: Linear[]
  constructor(sizes: number[]) {
    super()
    this.layers = []
    for (let i = 0; i < sizes.length - 1; i++) {
      this.layers.push(new Linear(sizes[i]!, sizes[i + 1]!))
    }
  }
}

// Shared by lossFn (B=BATCH_SIZE) and predictFn (B=null parametric).
function forward(net: NeuralNetwork, x: Tensor): Tensor {
  let h = x
  for (let i = 0; i < net.layers.length; i++) {
    h = net.layers[i]!.fwd(h)
    if (i < net.layers.length - 1) h = relu(h)
  }
  return h
}

function lossFn(net: NeuralNetwork, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  return crossEntropy(forward(net, x), y)
}

function predictFn(net: NeuralNetwork, { x }: { x: Tensor }): Tensor {
  return forward(net, x)
}

type Weight = { l1: number; i1: number; l2: number; i2: number; w: number }

// ---------- Model component ----------
// Sibling-facing surface of Model. Excludes training internals (GPU pipelines,
// batch buffers, tallies) — UI panels see only stats + actions.
interface IModel {
  status: string
  isReady: boolean
  isRunning: boolean
  trainIter: number
  trainRatio: number
  testRatio: number
  manualPredictionResult: number[]
  layers: number[]
  readonly activeLayers: number[]
  biases: number[][]
  weights: Weight[]
  hiddenLayer1: number
  hiddenLayer2: number
  lRate: number
  steps: number
  toggleRun(): void
  initNetwork(): Promise<void>
  predictDrawing(bytes: Uint8Array): Promise<void>
}

// Owns parameters, training, prediction, and the MNIST dataset.
class Model extends Component implements IModel {
  status = "Initializing..."
  isReady = false
  isRunning = false
  stepsPerAnimation = 1000
  layers = [INPUT_DIM, 20, 0, N_CLASSES]
  learningRate = 0.01

  trainIter = 0
  trainRatio = 0
  testRatio = 0
  manualPredictionResult: number[] = []
  // biases[0] is the input layer (all zeros, kept for index alignment with the
  // viz which uses `biases[layerIdx][neuronIdx]`); biases[k] for k>=1 is layer k.
  biases: number[][] = []
  // Pre-filtered (|w| > 0.2, top-K per layer pair); see #syncVizData.
  weights: Weight[] = []

  #train: CompiledTraining<NeuralNetwork> | null = null
  #infer: CompiledForward<NeuralNetwork> | null = null
  #trainingActive = false

  #trainSet: { input: Uint8Array; correct: number }[] = []
  #testSet: { input: Uint8Array; correct: number }[] = []

  #batchX = new Float32Array(BATCH_SIZE * INPUT_DIM)
  #batchY = new Int32Array(BATCH_SIZE)

  // Sliding window — capped to TALLY_WINDOW in #evaluateAccuracy.
  #trainTally: boolean[] = []
  #testTally: boolean[] = []
  #batchesSinceEval = 0

  // Fabric mouse:move fires faster than one inference round-trip; drop displaced
  // requests so the in-flight predict isn't queued behind 50 stale strokes.
  #predict = singleFlight(async (bytes: Uint8Array) => {
    if (!this.#infer) return
    const x = new Float32Array(INPUT_DIM)
    for (let i = 0; i < INPUT_DIM; i++) x[i] = bytes[i]! / 256
    const r = await this.#infer.run({ x })
    if (r.kind === 'aborted') return
    this.manualPredictionResult = Array.from(r.output)
    this.update()
  })

  get activeLayers() { return this.layers.filter(x => x > 0) }

  // Hyperparam setters that trigger a graph rebuild.
  get hiddenLayer1() { return this.layers[1]! }
  set hiddenLayer1(v) { this.layers[1] = v; this.initNetwork() }

  get hiddenLayer2() { return this.layers[2]! }
  set hiddenLayer2(v) { this.layers[2] = v; this.initNetwork() }

  get lRate() { return this.learningRate }
  set lRate(v) { this.learningRate = v; this.initNetwork() }

  get steps() { return this.stepsPerAnimation }
  set steps(v) { this.stepsPerAnimation = v; this.update() }

  constructor() {
    super()
    // Gate the entire UI on WebGPU. Without this, the main UI renders and the
    // predict/train fail silently on first lazy compile — looking broken rather
    // than unsupported.
    if (!isWebGPUAvailable()) {
      this.status = 'This demo requires WebGPU. Try a recent Chrome, Edge, or Safari (17.4+).'
      return
    }
    this.loadData().catch(err => {
      this.status = "Error: " + err.message
      this.update()
    })
  }

  async #compile() {
    const activeLayers = this.activeLayers
    if (this.#train) {
      // Explicit fresh seed → different weights on every rebuild (replaceModel
      // also defaults to fresh, but being explicit guards against future default flips).
      await this.#train.replaceModel(
        new NeuralNetwork(activeLayers),
        { seed: Math.floor(Math.random() * 0x7fffffff) },
      )
      await this.#train.setLR(this.learningRate)
      return
    }
    this.status = 'Compiling WGSL kernels...'
    this.update()
    const model = new NeuralNetwork(activeLayers)
    this.#train = await compile({
      model,
      loss: lossFn,
      optimizer: { kind: 'adam', lr: this.learningRate },
      inputs: {
        x: [BATCH_SIZE, INPUT_DIM],
        y: { shape: [BATCH_SIZE], dtype: 'i32' },
      },
    })
    // Polymorphic batch: B=1 from drawing, B=BATCH_SIZE from eval.
    this.#infer = await this.#train.attach({
      forward: predictFn,
      inputs: { x: [null, INPUT_DIM] },
    })
  }

  async initNetwork() {
    this.manualPredictionResult = []
    this.trainIter = 0
    this.trainRatio = 0
    this.testRatio = 0
    this.#trainTally = []
    this.#testTally = []
    this.#batchesSinceEval = 0
    this.biases = []
    this.weights = []
    try {
      await this.#compile()
      await this.#syncVizData()
    } catch { /* error already surfaced via status */ }
    this.update()
  }

  async loadData() {
    const prefix = "https://s3.eu-west-2.amazonaws.com/solenya-media/"

    const loadSet = async (imgFile: string, lblFile: string, name: string) => {
      this.status = `Fetching ${name}...`
      this.update()

      const [imgRes, lblRes] = await Promise.all([
        fetch(prefix + imgFile),
        fetch(prefix + lblFile)
      ])

      const imgBytes = inflate(new Uint8Array(await imgRes.arrayBuffer()))
      const lblBytes = inflate(new Uint8Array(await lblRes.arrayBuffer()))
      const view = new DataView(imgBytes.buffer, imgBytes.byteOffset)

      const count = view.getUint32(4)
      const rows = view.getUint32(8)
      const cols = view.getUint32(12)
      const size = rows * cols

      return range(count).map(i => ({
        input: imgBytes.subarray(16 + i * size, 16 + (i + 1) * size),
        correct: lblBytes[i + 8]
      }))
    }

    this.#trainSet = await loadSet("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "Training Data")
    this.#testSet = await loadSet("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", "Test Data")

    await this.initNetwork()
    this.status = "Ready"
    this.isReady = true
    this.update()
  }

  #makeBatch(set: { input: Uint8Array; correct: number }[]) {
    for (let i = 0; i < BATCH_SIZE; i++) {
      const ex = set[Math.floor(Math.random() * set.length)]!
      const off = i * INPUT_DIM
      for (let j = 0; j < INPUT_DIM; j++) this.#batchX[off + j] = ex.input[j]! / 256
      this.#batchY[i] = ex.correct
    }
  }

  toggleRun() {
    this.isRunning = !this.isRunning
    if (this.isRunning && !this.#trainingActive) this.runTrainingLoop()
    this.update()
  }

  // Train N batches per frame, rAF-yield between to keep UI responsive.
  async runTrainingLoop() {
    this.#trainingActive = true
    let lastUiUpdate = 0
    let lastVizSync = 0
    while (this.isRunning && this.#train) {
      const batchesPerFrame = Math.max(1, Math.round(this.stepsPerAnimation / BATCH_SIZE))
      for (let i = 0; i < batchesPerFrame && this.isRunning; i++) {
        await this.#trainOneBatch()
      }
      await new Promise<void>(r => requestAnimationFrame(() => r()))
      const now = performance.now()
      if (now - lastUiUpdate > 100) {
        lastUiUpdate = now
        this.update()
      }
      if (now - lastVizSync > 250) {
        lastVizSync = now
        await this.#syncVizData()
      }
    }
    this.update()
    this.#trainingActive = false
  }

  async #trainOneBatch() {
    if (!this.#train) return
    this.#makeBatch(this.#trainSet)
    try {
      const r = await this.#train.step({ x: this.#batchX, y: this.#batchY })
      if (r.kind === 'aborted') return
    } catch (e: any) {
      this.isRunning = false
      this.status = `Step error: ${e?.message ?? e}`
      this.update()
      return
    }
    this.trainIter += BATCH_SIZE

    this.#batchesSinceEval++
    if (this.#batchesSinceEval >= EVAL_EVERY_K_BATCHES) {
      this.#batchesSinceEval = 0
      await this.#evaluateAccuracy()
    }
  }

  async #evaluateAccuracy() {
    if (!this.#infer) return

    // Returns false on abort so the caller can short-circuit before ratios update.
    const evalOn = async (set: { input: Uint8Array; correct: number }[], tally: boolean[]) => {
      this.#makeBatch(set)
      const labels = Array.from(this.#batchY)
      const r = await this.#infer!.run({ x: this.#batchX })
      if (r.kind === 'aborted') return false
      const logits = r.output
      for (let i = 0; i < BATCH_SIZE; i++) {
        let best = 0
        const off = i * N_CLASSES
        for (let c = 1; c < N_CLASSES; c++) {
          if (logits[off + c]! > logits[off + best]!) best = c
        }
        tally.push(best === labels[i])
      }
      if (tally.length > TALLY_WINDOW) tally.splice(0, tally.length - TALLY_WINDOW)
      return true
    }

    try {
      if (!(await evalOn(this.#trainSet, this.#trainTally))) return
      if (!(await evalOn(this.#testSet, this.#testTally))) return
      this.trainRatio = this.#trainTally.filter(x => x).length / Math.max(1, this.#trainTally.length)
      this.testRatio = this.#testTally.filter(x => x).length / Math.max(1, this.#testTally.length)
    } catch { /* AbortError or transient — next eval retries */ }
  }

  // Reshape GPU param buffers into the (biases, weights) form the viz expects.
  async #syncVizData() {
    if (!this.#train) return
    let params: Record<string, Float32Array>
    try {
      params = await this.#train.downloadParams()
    } catch { return }
    const activeLayers = this.activeLayers
    const newBiases: number[][] = [new Array(activeLayers[0]).fill(0)]
    for (let i = 0; i < activeLayers.length - 1; i++) {
      const b = params[`layers.${i}.b`]
      newBiases.push(b ? Array.from(b) : new Array(activeLayers[i + 1]).fill(0))
    }
    this.biases = newBiases

    const weights: Weight[] = []
    for (let li = 0; li < activeLayers.length - 1; li++) {
      const W = params[`layers.${li}.W`]
      if (!W) continue
      const inSize = activeLayers[li]!
      const outSize = activeLayers[li + 1]!
      const layerWeights: Weight[] = []
      for (let i = 0; i < inSize; i++) {
        for (let j = 0; j < outSize; j++) {
          const w = W[i * outSize + j]!
          if (Math.abs(w) > 0.2) layerWeights.push({ l1: li, i1: i, l2: li + 1, i2: j, w })
        }
      }
      layerWeights.sort((a, b) => Math.abs(b.w) - Math.abs(a.w))
      if (layerWeights.length > MAX_CONN_PER_LAYER) layerWeights.length = MAX_CONN_PER_LAYER
      weights.push(...layerWeights)
    }
    this.weights = weights
  }

  async predictDrawing(bytes: Uint8Array): Promise<void> {
    await this.#predict(bytes)
  }
}

// ========== UI ==========

// ---------- UI constants ----------
const WIDE_LAYOUT_THRESHOLD = 1250
const MEDIUM_LAYOUT_THRESHOLD = 768

// ---------- UI interfaces ----------
interface IRoot {
  layout: 'wide' | 'medium' | 'narrow'
  isH: boolean
  controlSize: number
  model: IModel
}

// ---------- Theme ----------
const lightTheme = {
  colors: {
    primary: 'rgb(0, 128, 0)',
    primaryStrong: 'rgb(0, 100, 0)',
    secondary: 'rgb(128, 128, 128)',
    background: 'rgb(255, 255, 255)',
    surface: 'rgb(245, 245, 245)',
    text: 'rgb(15, 15, 15)',
    textMuted: 'rgb(85, 85, 85)',
    controlStrip: 'rgb(250, 250, 250)',
    border: 'rgb(220, 220, 220)',
    inputNeuron: 'rgb(200, 200, 200)',
    neuron: {
      positive: 'rgb(0, 255, 0)',
      negative: 'rgb(255, 0, 0)',
      neutral: 'rgb(180, 180, 180)'
    }
  }
}

const darkTheme = {
  colors: {
    primary: 'rgb(0, 200, 0)',
    primaryStrong: 'rgb(0, 255, 0)',
    secondary: 'rgb(100, 100, 100)',
    background: 'rgb(30, 30, 30)',
    surface: 'rgb(40, 40, 40)',
    text: 'rgb(255, 255, 255)',
    textMuted: 'rgb(165, 165, 165)',
    controlStrip: 'rgb(35, 35, 35)',
    border: 'rgb(45, 45, 45)',
    inputNeuron: 'rgb(60, 60, 60)',
    neuron: {
      positive: 'rgb(0, 255, 0)',
      negative: 'rgb(255, 0, 0)',
      neutral: 'rgb(80, 80, 80)'
    }
  }
}

const globalUnoCss = (theme: ThemeProxy<typeof lightTheme>) => ({
  'body': `m-0 bg-${theme.colors.background} text-${theme.colors.text} font-sans antialiased overflow-x-hidden`,
  'button': `transition-colors duration-200 ease-in-out`,
  '*': `box-border`
})

const getInitialTheme = () => {
  const hostTheme = document.documentElement.getAttribute('data-theme')
  if (hostTheme === 'light' || hostTheme === 'dark') return hostTheme
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

const themeMgr = new UnoThemeManager({
  id: 'app',
  themes: { light: lightTheme, dark: darkTheme },
  initialTheme: getInitialTheme(),
  unoCssConfig: { presets: [presetWind3()] },
  globalUnoCss
})

const styles = themeMgr.styles('neural', (theme) => {
  const { text, textMuted, primary, primaryStrong, surface, secondary, controlStrip, background, border } = theme.colors
  return {
    layout: `flex flex-col min-h-screen w-full`,
    header: `flex-none pt-4 pb-6 bg-${background} border-b border-${surface}`,
    headerContainer: `max-w-[1300px] mx-auto px-4 md:px-6 flex flex-col items-center text-center md:items-start md:text-left`,
    headerTitle: `text-${text} text-3xl font-bold tracking-tight`,
    headerDesc: `text-${textMuted} mt-1`,
    content: `flex-1`,
    controlStrip: `bg-${controlStrip} py-3 px-2 md:px-4 border-b border-${surface}`,
    controlStripInner: `max-w-[1300px] mx-auto px-1 flex flex-wrap items-center justify-center md:justify-between gap-x-4 gap-y-4`,
    statsRow: `flex flex-nowrap items-center justify-center gap-3 sm:gap-8`,
    controlGroup: `flex items-center gap-3`,
    selectorsWrap: `flex flex-wrap items-center justify-center md:justify-start gap-2 md:gap-4`,
    mainContainer: `max-w-[1300px] mx-auto px-4 md:px-6 py-4`,
    labelText: `text-xs text-${textMuted} whitespace-nowrap`,
    valueText: `font-bold text-${text} whitespace-nowrap`,
    iconBtn: `bg-transparent border-none p-0 cursor-pointer text-${primary} hover:text-${primaryStrong} transition-colors outline-none flex items-center justify-center`,
    iconBtnPlay: `w-12 h-12`,
    iconBtnSmall: `w-8 h-8`,
    formField: `flex flex-col shrink-0`,
    formLabel: `text-xs text-${textMuted}`,
    formControl: `px-3 py-1 border border-${secondary} rounded-lg bg-${background} text-${text} text-sm focus:outline-none focus:ring-2 focus:ring-${primary} focus:border-transparent transition-all cursor-pointer appearance-none`,
    canvasContainer: `inline-block`,
    canvasControls: `relative`,
    canvasHint: `absolute bottom-3 left-3 text-${textMuted} text-xs w-48 pointer-events-none select-none z-10`,
    neuron: `rounded-full border border-${secondary} flex items-center justify-center text-white text-xs shrink-0`,
    inputNeuronSvg: `flex`,
    predictionSquare: `border border-solid border-${border} rounded-lg overflow-hidden flex items-center justify-center`,
    predictionDisplay: `text-[220px] font-normal text-${text} leading-none select-none flex items-center justify-center -translate-y-2`,
    helpSection: `my-8 text-${textMuted}`,
    helpLink: `text-${primary} hover:underline cursor-pointer`,
    waitContainer: `flex flex-col items-center justify-center min-h-screen`,
    waitText: `text-${text} mb-4`,
    waitSpinner: `mt-4 animate-bounce text-2xl`,
    svgConnections: `absolute inset-0 pointer-events-none`,
    successHigh: `text-${primary}`,
    successLow: `text-red-500`,
    mainRowWide: `flex flex-row flex-nowrap mt-3 justify-center gap-8`,
    mainRowMedium: `flex flex-col mt-3 items-center gap-8`,
    mainRowNarrow: `flex flex-col mt-4 items-center gap-8`,
    mediumBottomRow: `flex flex-row gap-8 justify-center`,
    networkOuterH: `flex flex-row relative gap-8`,
    networkInputWrapH: `flex flex-row`,
    networkLayersH: `flex flex-row justify-between`,
    neuronLayerH: `flex flex-col justify-between h-full`,
    inputNeuronRowH: `flex flex-col`,
    scrawlerWrapH: `shrink-0 border border-solid border-${border} rounded-lg overflow-hidden`,
    mainRowV: `flex flex-col mt-4 items-center gap-8`,
    networkOuterV: `flex flex-col relative items-center gap-8`,
    networkInputWrapV: `flex flex-col`,
    networkLayersV: `flex flex-col justify-between items-center`,
    neuronLayerV: `flex flex-row justify-between w-full`,
    inputNeuronRowV: `flex flex-row`,
    scrawlerWrapV: `shrink-0 flex justify-center border border-solid border-${border} rounded-lg overflow-hidden`,
    predictionWrap: `shrink-0 flex justify-center`
  }
})

const icon = (d: string) => svg({ viewBox: '0 0 24 24', fill: 'currentColor', class: 'w-full h-full' }, path({ d }))
const iconPlay = () => icon('M8 5v14l11-7z')
const iconPause = () => icon('M6 19h4V5H6v14zm8-14v14h4V5h-4z')
const iconDelete = () => svg({ viewBox: '0 0 24 24', class: 'w-full h-full', fill: 'none', stroke: 'currentColor', strokeWidth: '2', strokeLineCap: 'round',  strokeLineJoin: 'round' },
  path({ d: 'M3 6h18m-2 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2' }),
  line({ x1: '10', y1: '11', x2: '10', y2: '17', strokeWidth: '2' }),
  line({ x1: '14', y1: '11', x2: '14', y2: '17', strokeWidth: '2' })
)

// ---------- UI helpers ----------

// Bias / weight → diverging color: green for positive, red for negative, intensity
// = |x| clamped to [0, 1]. Reads theme directly so callers don't have to thread it.
function weightedColor(x: number): string {
  const factor = clamp(Math.abs(x), 0, 1)
  const { positive, negative, neutral } = themeMgr.theme.colors.neuron
  const target = x > 0 ? positive.rawValue : negative.rawValue

  const parse = (c: string) => c.match(/\d+/g)?.map(Number) ?? [128, 128, 128]
  const [r1, g1, b1] = parse(neutral.rawValue)
  const [r2, g2, b2] = parse(target)

  const r = Math.round(r1 + (r2 - r1) * factor)
  const g = Math.round(g1 + (g2 - g1) * factor)
  const b = Math.round(b1 + (b2 - b1) * factor)

  return `rgb(${r}, ${g}, ${b})`
}

// Crop tightly to the inked pixels and re-center inside a width×height target
// canvas. Returns the alpha channel as a flat Uint8Array — MNIST style.
function extractCenteredImage(canvas1: HTMLCanvasElement, width: number, height: number) {
  const canvas2 = document.createElement('canvas')
  canvas2.width = width
  canvas2.height = height

  const ctx2 = canvas2.getContext('2d')!
  ctx2.imageSmoothingEnabled = false
  ctx2.drawImage(canvas1, 0, 0, width, height)

  const data = ctx2.getImageData(0, 0, width, height)
  const bytes = Array.from({ length: width * height }, (_, i) => data.data[i * 4 + 3])

  const b = { l: width, t: height, r: 0, b: 0 }
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      if (bytes[y * width + x]) {
        b.l = Math.min(b.l, x)
        b.r = Math.max(b.r, x + 1)
        b.t = Math.min(b.t, y)
        b.b = Math.max(b.b, y + 1)
      }
    }
  }

  const bw = b.r - b.l
  const bh = b.b - b.t
  const canvas3 = document.createElement('canvas')
  canvas3.width = width
  canvas3.height = height

  const ctx3 = canvas3.getContext('2d')!
  if (bw > 0 && bh > 0) {
    const targetX = b.l + ((width - b.r) - b.l) / 2
    const targetY = b.t + ((height - b.b) - b.t) / 2
    ctx3.drawImage(canvas2, b.l, b.t, bw, bh, targetX, targetY, bw, bh)
  }

  const res = ctx3.getImageData(0, 0, width, height)
  return new Uint8Array(Array.from({ length: width * height }, (_, i) => res.data[i * 4 + 3]))
}

// ---------- Scrawler ----------
// Drawing canvas (fabric.js PencilBrush). Captures live mid-stroke (compositing
// fabric's upper canvas onto the lower) so predictions update during a stroke,
// not just on mouse-up.
class Scrawler extends Component {
  isClear = true
  private fc: fabric.Canvas | null = null

  clear() {
    this.fc?.clear()
    this.isClear = true
    this.update()
  }

  view(size: number, onStart: () => void, onFinish: (bytes: Uint8Array) => void) {
    const { text } = themeMgr.theme.colors

    if (this.fc) {
      this.fc.freeDrawingBrush!.color = text.rawValue
      this.fc.getObjects().forEach(obj => {
        if (obj instanceof fabric.Path) obj.set({ stroke: text.rawValue })
      })
      this.fc.renderAll()
    }

    return div({
      class: 'relative',
      style: { width: size + 'px', height: size + 'px' }
    },
      div({
        class: 'relative',
        key: 'fabric-wrapper',
        onMounted: el => {
          if (this.fc) return
          const canvas = el.appendChild(document.createElement('canvas'))
          const fc = this.fc = new fabric.Canvas(canvas, {
            isDrawingMode: true,
            width: size,
            height: size
          })

          fc.freeDrawingBrush = new fabric.PencilBrush(fc)
          fc.freeDrawingBrush.width = 20
          fc.freeDrawingBrush.color = text.rawValue

          // Composite lower (committed strokes) + upper (in-progress freehand
          // stroke from PencilBrush) so mid-stroke captures include the live
          // pen. Lower alone is empty until path:created fires on mouse-up.
          const captureLive = () => {
            const lower = fc.getElement()
            const upper = fc.upperCanvasEl as HTMLCanvasElement
            const temp = document.createElement('canvas')
            temp.width = lower.width
            temp.height = lower.height
            const tctx = temp.getContext('2d')!
            tctx.drawImage(lower, 0, 0)
            tctx.drawImage(upper, 0, 0)
            onFinish(extractCenteredImage(temp, 28, 28))
          }
          let isDrawing = false
          let movePending = false
          fc.on('mouse:down', () => { isDrawing = true; onStart() })
          fc.on('mouse:up', () => { isDrawing = false })
          fc.on('mouse:move', () => {
            if (!isDrawing || movePending) return
            movePending = true
            requestAnimationFrame(() => {
              movePending = false
              if (!this.fc) return
              captureLive()
            })
          })
          fc.on('path:created', () => {
             requestAnimationFrame(() => {
               onFinish(extractCenteredImage(fc.getElement(), 28, 28))
               this.isClear = false
               this.update()
             })
          })

          return () => {
            this.fc?.dispose()
            this.fc = null
          }
        }
      }),
      this.isClear ?
        div({ class: styles.canvasHint }, "Draw a digit in this box when success reaches 90%.") :
        button({
        class: [styles.iconBtn, 'absolute top-2 right-2 flex-none z-20', styles.iconBtnSmall],
        onClick: (e: MouseEvent) => {
          e.stopPropagation()
          this.clear()
        },
        title: 'Clear Drawing'
      }, iconDelete())
    )
  }
}

// ---------- Controls panel ----------
// Selector changes write back via the model's setters; each triggers a graph rebuild.
class ControlsPanel extends Component {
  get root() { return this.ctx.root as any as IRoot }
  get model() { return this.root.model }

  view() {
    return div({ class: styles.controlStrip },
      div({ class: styles.controlStripInner },
        div({ class: styles.statsRow },
          div({ class: styles.controlGroup },
            button({
              onClick: () => this.model.toggleRun(),
              class: [styles.iconBtn, styles.iconBtnPlay]
            }, this.model.isRunning ? iconPause() : iconPlay()),
            button({
              onClick: () => this.model.initNetwork(),
              class: [styles.iconBtn, styles.iconBtnSmall]
            }, iconDelete())
          ),
          div({ class: styles.formField },
            div({ class: styles.labelText }, 'Steps'),
            div({ class: styles.valueText }, this.model.trainIter.toLocaleString())
          ),
          div({ class: styles.formField },
            div({ class: styles.labelText }, 'Success (Train / Test)'),
            div({ class: styles.valueText },
              span({ class: this.model.trainRatio > 0.9 ? styles.successHigh : styles.successLow }, (this.model.trainRatio * 100).toFixed(1) + '%'),
              ' / ',
              span({ class: this.model.testRatio > 0.9 ? styles.successHigh : styles.successLow }, (this.model.testRatio * 100).toFixed(1) + '%')
            )
          )
        ),
        div({ class: styles.selectorsWrap }, ...this.selectors())
      )
    )
  }

  selectors() {
    const opts = [
      { value: 0, label: 'None' },
      { value: 10, label: '10' },
      { value: 20, label: '20' },
      { value: 50, label: '50' }
    ]

    const field = (prop: () => any, label: string, options: SelectOption<number>[]) =>
      formField({
        target: this.model as any as Component,
        prop,
        inputFn: inputSelect,
        label,
        labelAttrs: { class: styles.formLabel },
        inputProps: {
          options,
          attrs: { class: styles.formControl }
        }
      })

    return [
      field(() => this.model.steps, 'Step', [10, 100, 500, 1000, 2000].map(v => ({ value: v, label: String(v) }))),
      field(() => this.model.lRate, 'L. Rate', [0.001, 0.01, 0.03, 0.1].map(v => ({ value: v, label: String(v) }))),
      field(() => this.model.hiddenLayer1, 'Layer 1', opts),
      field(() => this.model.hiddenLayer2, 'Layer 2', opts)
    ]
  }
}

// ---------- Network panel ----------
// 28×28 input grid + hidden/output neuron columns (colored by bias) + top-K
// connection lines (colored by weight) + distribution strip of softmax probs.
class NetworkPanel extends Component {
  get root() { return this.ctx.root as any as IRoot }
  get model() { return this.root.model }

  view() {
    const isH = this.root.isH
    const layout = this.root.layout
    const size = this.root.controlSize

    return div({
      class: isH ? styles.networkOuterH : styles.networkOuterV,
      key: 'net-outer',
      style: {
        border: '1px solid transparent', // Fixes some layout shifts
        ...(layout === 'wide' ? { marginRight: '-24px' } : { marginBottom: '-24px' })
      }
    },
      this.inputGridView(size, isH),
      this.layersView(size, isH),
      this.connectionsView(),
      this.distributionView(size, isH, layout)
    )
  }

  inputGridView(size: number, isH: boolean) {
    const cellSize = size / 28
    const inputNeuronColor = themeMgr.theme.colors.inputNeuron.rawValue
    return div({
      class: [isH ? styles.networkInputWrapH : styles.networkInputWrapV],
      style: { width: size + 'px', height: size + 'px' }
    },
      ...range(28).map(y => div({ class: isH ? styles.inputNeuronRowH : styles.inputNeuronRowV },
        ...range(28).map(x => svg({ width: cellSize, height: cellSize, class: styles.inputNeuronSvg },
          circle({
            cx: cellSize / 2,
            cy: cellSize / 2,
            r: Math.max(0, (cellSize / 2) - 0.5),
            fill: 'transparent',
            stroke: inputNeuronColor,
            id: `neuron-0-${y * 28 + x}`,
            style: { pointerEvents: 'none' }
          })
        ))
      ))
    )
  }

  layersView(size: number, isH: boolean) {
    const activeLayers = this.model.activeLayers
    return div({ class: isH ? styles.networkLayersH : styles.networkLayersV, style: { width: size + 'px', height: size + 'px' } },
      ...activeLayers.slice(1).map((count, lIdx) =>
        div({ class: isH ? styles.neuronLayerH : styles.neuronLayerV, style: isH ? {} : { width: '100%' } },
          ...range(count).map(i => {
            const b = this.model.biases[lIdx + 1]?.[i] ?? 0
            const s = Math.max(2, Math.min(24, Math.floor(size / count)))
            const isOutput = (lIdx + 1 === activeLayers.length - 1)

            return div({
              id: `neuron-${lIdx + 1}-${i}`,
              class: styles.neuron,
              style: { backgroundColor: weightedColor(b), width: s + 'px', height: s + 'px' }
            }, isOutput ? i : '')
          })
        )
      )
    )
  }

  connectionsView() {
    const h = this.root.isH
    const lines: VElement[] = []

    this.model.weights.forEach(w => {
        const r1 = document.getElementById(`neuron-${w.l1}-${w.i1}`)?.getBoundingClientRect()
        const r2 = document.getElementById(`neuron-${w.l2}-${w.i2}`)?.getBoundingClientRect()
        const parent = document.getElementById('connections-svg')?.getBoundingClientRect()

        if (r1 && r2 && parent) {
            const x1 = (h ? (w.l1 === 0 ? r1.left + r1.width/2 : r1.right) : r1.left + r1.width/2) - parent.left
            const y1 = (h ? r1.top + r1.height/2 : (w.l1 === 0 ? r1.top + r1.height/2 : r1.bottom)) - parent.top
            const x2 = (h ? r2.left : r2.left + r2.width/2) - parent.left
            const y2 = (h ? r2.top + r2.height/2 : r2.top) - parent.top

            lines.push(line({ x1, y1, x2, y2, stroke: weightedColor(w.w), strokeWidth: 1 }))
        }
    })

    return svg({
      id: 'connections-svg',
      class: styles.svgConnections,
      width: '100%',
      height: '100%'
    }, ...lines)
  }

  distributionView(size: number, isH: boolean, layout: 'wide' | 'medium' | 'narrow') {
    const STRIP = 50
    const primary = themeMgr.theme.colors.primary.rawValue

    const logits = this.model.manualPredictionResult
    let probs: number[] = []
    if (logits.length === N_CLASSES) {
      const m = Math.max(...logits)
      const exps = logits.map(l => Math.exp(l - m))
      const sum = exps.reduce((a, b) => a + b, 0)
      probs = exps.map(e => e / sum)
    }

    const s = Math.max(2, Math.min(24, Math.floor(size / N_CLASSES)))
    const thickness = Math.max(2, Math.floor(s * 0.6))

    // In medium layout, the AB / CD rows are stacked and each centered. If the
    // chart contributes to the network's flex width it shifts AB right of CD.
    // So in medium we absolute-position it off the right of networkOuterH —
    // protruding past the centered row instead of widening it.
    const isMedium = layout === 'medium'
    const posStyle = isMedium
      ? { position: 'absolute', top: '0', right: '-' + (STRIP + 8) + 'px' }
      : { position: 'relative', ...(isH ? { marginLeft: '-24px' } : { marginTop: '-24px' }) }

    return div({
      style: isH
        ? { width: STRIP + 'px', height: size + 'px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', zIndex: '1', ...posStyle }
        : { width: size + 'px', height: STRIP + 'px', display: 'flex', flexDirection: 'row', justifyContent: 'space-between', zIndex: '1', ...posStyle }
    }, ...range(N_CLASSES).map(i => {
      const p = probs[i] ?? 0
      return div({
        style: isH
          ? { width: STRIP + 'px', height: s + 'px', display: 'flex', alignItems: 'center' }
          : { width: s + 'px', height: STRIP + 'px', display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }
      },
        div({
          style: isH
            ? { width: (p * STRIP) + 'px', height: thickness + 'px', backgroundColor: primary, borderRadius: '2px' }
            : { height: (p * STRIP) + 'px', width: thickness + 'px', backgroundColor: primary, borderRadius: '2px' }
        })
      )
    }))
  }
}

// ---------- Prediction display ----------
class PredictionDisplay extends Component {
  get root() { return this.ctx.root as any as IRoot }

  view() {
    const size = this.root.controlSize
    const arr = this.root.model.manualPredictionResult
    const digit = arr.length > 0 ? arr.indexOf(Math.max(...arr)) : ''
    return div({
      class: styles.predictionSquare,
      style: { width: size + 'px', height: size + 'px' }
    },
      div({ class: styles.predictionDisplay }, digit)
    )
  }
}

// ---------- Root ----------
class Root extends Component implements IRoot {
  model = new Model()
  controls = new ControlsPanel()
  network = new NetworkPanel()
  scrawler = new Scrawler()
  prediction = new PredictionDisplay()

  #layout: 'wide' | 'medium' | 'narrow' = window.innerWidth >= WIDE_LAYOUT_THRESHOLD ? 'wide' : (window.innerWidth >= MEDIUM_LAYOUT_THRESHOLD ? 'medium' : 'narrow')

  get layout() { return this.#layout }
  get isH() { return this.#layout !== 'narrow' }
  get controlSize() {
    if (this.#layout === 'narrow') return Math.max(100, Math.min(300, window.innerWidth - 40))
    return 280
  }

  constructor() {
    super()

    new MutationObserver(() => {
      const theme = document.documentElement.getAttribute('data-theme')
      if (theme && (theme === 'light' || theme === 'dark') && theme !== themeMgr.themeName) {
        themeMgr.themeName = theme
        this.update()
      }
    }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })

    const onResize = () => {
      const w = window.innerWidth
      this.#layout = w >= WIDE_LAYOUT_THRESHOLD ? 'wide' : (w >= MEDIUM_LAYOUT_THRESHOLD ? 'medium' : 'narrow')
      this.update()
      requestAnimationFrame(() => this.update())
    }
    window.addEventListener('resize', onResize)
    document.addEventListener('fullscreenchange', onResize)
  }

  view() {
    if (!this.model.isReady) {
      return div({ class: styles.waitContainer },
        div({ class: styles.waitText }, this.model.status),
        div({ class: styles.waitSpinner }, "⏳")
      )
    }

    return div({ class: styles.layout },
      div({ class: styles.header },
        div({ class: styles.headerContainer },
          h1({ class: styles.headerTitle }, "Neural Network"),
          p({ class: styles.headerDesc }, "Recognise handwritten digits with a neural network.")
        )
      ),
      div({ class: styles.content },
        this.controls.view(),
        div({ class: styles.mainContainer }, this.renderMainContent(), this.helpSection())
      )
    )
  }

  renderMainContent() {
    const network = this.network.view()
    const scrawler = div(
      { class: this.isH ? styles.scrawlerWrapH : styles.scrawlerWrapV },
      this.scrawler.view(
        this.controlSize,
        () => {},
        (bytes) => { this.model.predictDrawing(bytes) }
      )
    )
    const prediction = div({ class: styles.predictionWrap }, this.prediction.view())

    if (this.#layout === 'wide') {
      return div({ class: styles.mainRowWide }, network, scrawler, prediction)
    }
    if (this.#layout === 'medium') {
      return div(
        { class: styles.mainRowMedium },
        network,
        div({ class: styles.mediumBottomRow }, scrawler, prediction)
      )
    }
    return div({ class: styles.mainRowNarrow }, network, scrawler, prediction)
  }

  helpSection() {
    const { helpLink } = styles
    return div({ class: styles.helpSection },
      p("A dense feed-forward network where forward + backward + Adam update run as one fused GPU kernel pipeline per batch step via ", span({ class: helpLink, onClick: () => window.open("https://www.npmjs.com/package/tensorgrad", "_blank") }, "tensorgrad"), "."),
      p("When you click 'play' the neural network starts learning from 60,000 hand-written images of digits, courtesy of ",
        span({ class: helpLink, onClick: () => window.open("http://yann.lecun.com/exdb/mnist/", "_blank") }, "the MNIST database"), "."
      ),
      p("The 'training' vs. 'testing' success indicates how many successful predictions the neural network makes on the 60,000 training images vs. 10,000 test images that were not used during training."),
      p("Every neuron connects to every other neuron in adjacent layers, but for clarity (and to not overwork your GPU!) only the strongest connections are displayed.")
    )
  }
}

new App({
  root: new Root(),
  id: 'app',
  cssAdapter: themeMgr.unoCssAdapter
})
```

**index.html**

```html
<div id="app"></div>
```
**config.json**

```json
{
  "dependencies": {
    "domeleon": "^0.6.0",
    "@unocss/preset-wind3": "^66.5.3",
    "pako": "^2.1.0",
    "fabric": "^7.0.0",
    "tensorgrad": "0.1.0"
  },
  "description": "Train a neural network to recognize handwritten digits (MNIST) live in your browser. Draw on the canvas and watch the model improve in real time. Built with tensorgrad (autograd + WebGPU)."
}
```