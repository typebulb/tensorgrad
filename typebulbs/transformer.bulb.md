---
format: typebulb/v1
name: Transformer (tensorgrad)
---

**code.tsx**

```tsx
import presetWind3 from '@unocss/preset-wind3'
import { App, Component, div, h1, button, span, p, a, svg, g, rect, path, line, text, circle, animate, formField, inputRange, type VElement } from 'domeleon'
import { UnoThemeManager, type ThemeProxy } from 'domeleon/unocss'
import { inputNumber } from 'domeleon/maskito'

import {
  Module, compile, isWebGPUAvailable, lr, Linear, LayerNorm, crossEntropy, capture, singleFlight,
  add, mul, sum, swapAxes,
  relu, matmul, embedding, arange,
  softmaxCausal, splitHeads, mergeHeads,
  type Tensor,
  type CompiledTraining, type CompiledForward,
} from 'tensorgrad'

// ========== MODEL / TRAINING ==========

// ---------- Constants ----------
// Vocab: digits 0..9 are tokens 0..9; '+' = 10, '=' = 11.
const VOCAB = 12
const TOK_PLUS = 10
const TOK_EQ = 11
const N_OP_DIGITS = 2
const N_RESULT_DIGITS = 3
// Sequence layout: [d d + d d = r r r], result digits reversed (LSB first) so
// the model can do left-to-right carry.
const SEQ_LEN = N_OP_DIGITS + 1 + N_OP_DIGITS + 1 + N_RESULT_DIGITS  // 9
const RESULT_START = N_OP_DIGITS + 1 + N_OP_DIGITS + 1               // 6

const D_MODEL = 64
const N_LAYERS = 3
const N_HEADS = 4
const D_HEAD = D_MODEL / N_HEADS
const BATCH_SIZE = 128
const T_LEN = SEQ_LEN - 1
const LR_SCHEDULE = lr.linear({ peak: 0.005, final: 0.0005, steps: 1500 })

const range = (n: number) => Array.from({ length: n }, (_, i) => i)
const clamp = (n: number, min: number, max: number) => Math.min(Math.max(n, min), max)

// 6-token addition prompt: [a_tens, a_ones, +, b_tens, b_ones, =].
const promptTokens = (a: number, b: number): number[] => [
  Math.floor(a / 10), a % 10, TOK_PLUS,
  Math.floor(b / 10), b % 10, TOK_EQ,
]

// Deterministic train/test split by (a, b). ~20% held out.
const isTestPair = (a: number, b: number) => ((a * 100 + b) * 31 + 7) % 5 === 0

const N_TEST = 50

// ---------- Model hierarchy ----------

class Attention extends Module {
  q = new Linear(D_MODEL, D_MODEL, { bias: false })
  k = new Linear(D_MODEL, D_MODEL, { bias: false })
  v = new Linear(D_MODEL, D_MODEL, { bias: false })
  o = new Linear(D_MODEL, D_MODEL, { bias: false })
}

class MLP extends Module {
  up = new Linear(D_MODEL, 4 * D_MODEL)
  down = new Linear(4 * D_MODEL, D_MODEL)
}

class Block extends Module {
  ln1 = new LayerNorm(D_MODEL)
  attn = new Attention()
  ln2 = new LayerNorm(D_MODEL)
  mlp = new MLP()
}

class Transformer extends Module {
  tok_emb: Tensor
  pos_emb: Tensor
  layers: Block[]
  lnf = new LayerNorm(D_MODEL)
  constructor() {
    super()
    this.tok_emb = this.param([VOCAB, D_MODEL])
    this.pos_emb = this.param([SEQ_LEN, D_MODEL])
    this.layers = range(N_LAYERS).map(() => new Block())
  }
}

// ---------- Forward pass + loss ----------
// Capture sites (`capture(name, t)`) make intermediates readable post-run via
// `result.captures.get(name)`. Tensorgrad always reads captures back alongside
// the loss, so leaving viz captures in the training graph would mean ~7 MB
// GPU→main per step. `cap` gates them by trace mode: off during the training
// compile, on for the inference compile.
let captureMode: 'on' | 'off' = 'on'
const cap = (name: string, t: Tensor): Tensor => captureMode === 'on' ? capture(name, t) : t

function attentionFwd(p: Attention, x: Tensor, layerIdx: number): Tensor {
  const q = cap(`q.${layerIdx}`, splitHeads(p.q.fwd(x), N_HEADS))
  const k = cap(`k.${layerIdx}`, splitHeads(p.k.fwd(x), N_HEADS))
  const v = cap(`v.${layerIdx}`, splitHeads(p.v.fwd(x), N_HEADS))
  const scores = mul(matmul(q, swapAxes(k, -1, -2)), 1 / Math.sqrt(D_HEAD))
  const attn = cap(`attn.${layerIdx}`, softmaxCausal(scores))
  return p.o.fwd(mergeHeads(matmul(attn, v)))
}

// MLP is inlined so the post-ReLU hidden state can be captured.
function blockFwd(p: Block, x: Tensor, layerIdx: number): Tensor {
  const x1 = add(x, attentionFwd(p.attn, p.ln1.fwd(x), layerIdx))
  const xn2 = p.ln2.fwd(x1)
  const h1 = cap(`mlp_hidden.${layerIdx}`, relu(p.mlp.up.fwd(xn2)))
  return add(x1, p.mlp.down.fwd(h1))
}

function modelFwd(p: Transformer, tokens: Tensor): Tensor {
  const tokE = embedding(p.tok_emb, tokens)                      // [B, T, D]
  const posE = embedding(p.pos_emb, arange(T_LEN))               // [T, D] — broadcasts over batch
  let x = add(tokE, posE)
  for (let i = 0; i < p.layers.length; i++) {
    cap(`residual.${i}`, x)
    x = blockFwd(p.layers[i]!, x, i)
  }
  cap(`residual.${N_LAYERS}`, x)
  const xnf = p.lnf.fwd(x)
  // Tied embedding head: logits = xnf @ tok_emb^T.
  return matmul(xnf, swapAxes(p.tok_emb, -1, -2))
}

// `mask` is float32 [T] with 1s on result-digit positions and 0s elsewhere —
// loss is only computed over the digits the model has to predict.
function lossFn(
  p: Transformer,
  { tokens, targets, mask }: { tokens: Tensor; targets: Tensor; mask: Tensor },
): Tensor {
  const ce = crossEntropy(modelFwd(p, tokens), targets, { reduction: 'none' })   // [B, T] of -log p(target)
  return mul(sum(mul(ce, mask)), 1 / (BATCH_SIZE * N_RESULT_DIGITS))
}

// Inference forward extends modelFwd with logit-lens captures: at every
// residual depth, project the residual through the final LN + tied unembed.
// The final-layer logit-lens IS the model's prediction, so it's the output.
function predictFn(p: Transformer, { tokens }: { tokens: Tensor }): Tensor {
  const tokE = embedding(p.tok_emb, tokens)
  const posE = embedding(p.pos_emb, arange(T_LEN))
  const tokEmbT = swapAxes(p.tok_emb, -1, -2)
  const lensAt = (r: Tensor) => matmul(p.lnf.fwd(r), tokEmbT)
  let x = add(tokE, posE)
  for (let i = 0; i < p.layers.length; i++) {
    capture(`residual.${i}`, x)
    capture(`logit_lens.${i}`, lensAt(x))
    x = blockFwd(p.layers[i]!, x, i)
  }
  capture(`residual.${N_LAYERS}`, x)
  return capture(`logit_lens.${N_LAYERS}`, lensAt(x))
}

// Loss masks out non-result positions.
const RESULT_MASK = Float32Array.from({ length: T_LEN }, (_, t) => t >= RESULT_START - 1 ? 1 : 0)

// Excludes test pairs. Encodes with reversed result digits.
function makeBatch(): { tokens: Int32Array; targets: Int32Array } {
  const tokens = new Int32Array(BATCH_SIZE * T_LEN)
  const targets = new Int32Array(BATCH_SIZE * T_LEN)
  for (let bi = 0; bi < BATCH_SIZE; bi++) {
    let a, c
    do { a = Math.floor(Math.random() * 100); c = Math.floor(Math.random() * 100) } while (isTestPair(a, c))
    const sum = a + c
    const seq = [
      ...promptTokens(a, c),
      sum % 10, Math.floor(sum / 10) % 10, Math.floor(sum / 100),
    ]
    for (let t = 0; t < T_LEN; t++) {
      tokens[bi * T_LEN + t] = seq[t]!
      targets[bi * T_LEN + t] = seq[t + 1]!
    }
  }
  return { tokens, targets }
}

// ---------- Inference: forward-only graph at B=1 ----------

const HIDDEN = 4 * D_MODEL

// Stable softmax over a slice of a flat array.
function softmaxNumeric(arr: Float32Array, offset = 0, len = arr.length - offset): number[] {
  let mx = -Infinity
  for (let i = 0; i < len; i++) if (arr[offset + i]! > mx) mx = arr[offset + i]!
  const probs = new Array<number>(len)
  let sum = 0
  for (let i = 0; i < len; i++) { probs[i] = Math.exp(arr[offset + i]! - mx); sum += probs[i]! }
  for (let i = 0; i < len; i++) probs[i] = probs[i]! / sum
  return probs
}
function argmaxNumeric(arr: number[]): number {
  let best = 0
  for (let i = 1; i < arr.length; i++) if (arr[i]! > arr[best]!) best = i
  return best
}

type AttnMap = { data: number[]; T: number; q: number[]; k: number[]; v: number[] }
type LatticeCell = { topToken: number; topProb: number; probs: number[]; residual: number[] }
type InsideStep = { inputTokens: number[]; generated: number; lattice: LatticeCell[][]; mlpHiddens: number[][][] }

// Autoregressive generation. `infer` shares param buffers with training, so
// every step's update is visible on the next call.
async function predictAddition(
  infer: CompiledForward,
  a: number,
  b: number,
  opts: { collectViz?: boolean } = {},
): Promise<{
  generated: number[]
  generatedProbs: number[][]
  attnMaps: AttnMap[][]
  inside: InsideStep[]
}> {
  // collectViz=false skips lattice/attention assembly for accuracy-only probes.
  const collectViz = opts.collectViz !== false

  const prefix = promptTokens(a, b)
  const generated: number[] = []
  const generatedProbs: number[][] = []
  const attnMapsPerStep: AttnMap[][] = []
  const insideSteps: InsideStep[] = []
  const tokensBuf = new Int32Array(T_LEN)

  for (let step = 0; step < N_RESULT_DIGITS; step++) {
    const realLen = prefix.length + generated.length     // 6, 7, or 8
    tokensBuf.fill(0)
    for (let i = 0; i < prefix.length; i++) tokensBuf[i] = prefix[i]!
    for (let i = 0; i < generated.length; i++) tokensBuf[prefix.length + i] = generated[i]!
    const inputTokens = prefix.concat(generated)         // before this step's push

    const r = await infer.run({ tokens: tokensBuf })
    if (r.kind !== 'completed') return { generated, generatedProbs, attnMaps: attnMapsPerStep, inside: insideSteps }
    const logitsAll = r.output
    const captures = r.captures

    // Restrict to digits 0..9 — '+' and '=' aren't valid at result positions.
    const probs = softmaxNumeric(logitsAll, (realLen - 1) * VOCAB, 10)
    const best = argmaxNumeric(probs)
    generated.push(best)
    generatedProbs.push(probs)

    if (!collectViz) continue

    const stepAttnMaps: AttnMap[] = []
    for (let l = 0; l < N_LAYERS; l++) {
      const attnH = captures.perHead(`attn.${l}`)
      const qH = captures.perHead(`q.${l}`)
      const kH = captures.perHead(`k.${l}`)
      const vH = captures.perHead(`v.${l}`)
      for (let h = 0; h < N_HEADS; h++) {
        // attnH[h] is [T_LEN, T_LEN] flat; copy out the realLen×realLen block.
        const attn: number[] = new Array(realLen * realLen)
        for (let i = 0; i < realLen; i++) for (let j = 0; j < realLen; j++) {
          attn[i * realLen + j] = attnH[h]![i * T_LEN + j]!
        }
        stepAttnMaps.push({
          data: attn,
          T: realLen,
          q: Array.from(qH[h]!.subarray(0, realLen * D_HEAD)),
          k: Array.from(kH[h]!.subarray(0, realLen * D_HEAD)),
          v: Array.from(vH[h]!.subarray(0, realLen * D_HEAD)),
        })
      }
    }
    attnMapsPerStep.push(stepAttnMaps)

    const stepLattice: LatticeCell[][] = []
    for (let l = 0; l <= N_LAYERS; l++) {
      const ll = captures.get(`logit_lens.${l}`)         // [1, T, V] flat
      const r = captures.get(`residual.${l}`)            // [1, T, D] flat
      const cells: LatticeCell[] = []
      for (let pos = 0; pos < realLen; pos++) {
        const ps = softmaxNumeric(ll, pos * VOCAB, VOCAB)
        const topI = argmaxNumeric(ps)
        cells.push({
          topToken: topI,
          topProb: ps[topI]!,
          probs: ps,
          residual: Array.from(r.subarray(pos * D_MODEL, (pos + 1) * D_MODEL)),
        })
      }
      stepLattice.push(cells)
    }

    const stepMlp: number[][][] = []
    for (let l = 0; l < N_LAYERS; l++) {
      const m = captures.get(`mlp_hidden.${l}`)          // [1, T, HIDDEN] flat
      const perPos: number[][] = []
      for (let pos = 0; pos < realLen; pos++) {
        perPos.push(Array.from(m.subarray(pos * HIDDEN, (pos + 1) * HIDDEN)))
      }
      stepMlp.push(perPos)
    }

    insideSteps.push({ inputTokens, generated: best, lattice: stepLattice, mlpHiddens: stepMlp })
  }

  return { generated, generatedProbs, attnMaps: attnMapsPerStep, inside: insideSteps }
}

// Sibling-facing surface of Model — what UI panels read. Excludes training
// internals and Root-only stats (avgLoss, isRunning, paramCount).
interface IModel {
  trainStep: number
  predictedDigits: number[]
  digitProbs: number[][]
  attnMaps: AttnMap[][]
  inside: InsideStep[]
  recentFailures: { a: number; b: number; got: number; want: number }[]
  embeddingHistory: { step: number; tokEmb: Float32Array }[]
  getTokEmbSnapshot(): Float32Array | null
  getMlpWeights(layerIdx: number): { w1: Float32Array; w2: Float32Array; b2: Float32Array } | null
  getAttnWeights(layerIdx: number): { wQ: Float32Array; wK: Float32Array; wV: Float32Array; wO: Float32Array } | null
  refreshPrediction(): Promise<void>
}

// ---------- Model ----------

class Model extends Component implements IModel {
  status = 'Initializing...'
  isReady = false
  isRunning = false
  trainStep = 0
  examplesSeen = 0
  avgLoss = 0
  examplesPerSec = 0
  testAcc = 0

  predictedDigits: number[] = []  // LSB-first: [ones, tens, hundreds]
  digitProbs: number[][] = []
  attnMaps: AttnMap[][] = []
  inside: InsideStep[] = []

  lossHistory: { step: number; value: number }[] = []
  accHistory: { step: number; value: number }[] = []

  recentFailures: { a: number; b: number; got: number; want: number }[] = []
  embeddingHistory: { step: number; tokEmb: Float32Array }[] = []

  // CPU mirror of params, kept in sync from GPU for viz panels.
  #params: Record<string, Float32Array> = {}
  #train: CompiledTraining<Transformer> | null = null
  // #infer is polymorphic over batch dim — first run() at B=1 (predictAddition)
  // and at B=N_TEST (#writeDiagnostic) each trigger a sibling compile + cache.
  #infer: CompiledForward | null = null
  #compilePromise: Promise<void> | null = null
  #trainingActive = false
  #lossWindow: number[] = []
  #stepTimes: number[] = []
  #lastStepTs = 0
  #diagnosticBusy = false

  get root() { return this.ctx.root as any as IRoot }

  constructor() {
    super()
    if (!isWebGPUAvailable()) {
      this.isReady = false
      this.status = 'This demo requires WebGPU. Try a recent Chrome, Edge, or Safari (17.4+).'
      return
    }
    this.isReady = true
    this.status = 'Ready (will compile WGSL on first run)'
  }

  // refreshPrediction needs this.root.ctx wired, which only happens at attach time.
  override onAttached() {
    if (!this.isReady) return
    this.refreshPrediction()
    if (!this.isRunning) this.toggleRun()
  }

  get paramCount(): number {
    return Object.values(this.#params).reduce((a, b) => a + b.length, 0)
  }

  getTokEmbSnapshot(): Float32Array | null {
    const t = this.#params['tok_emb']
    return t ? new Float32Array(t) : null
  }

  getMlpWeights(layerIdx: number): { w1: Float32Array; w2: Float32Array; b2: Float32Array } | null {
    const lp = `layers.${layerIdx}.mlp`
    const w1 = this.#params[`${lp}.up.W`]
    const w2 = this.#params[`${lp}.down.W`]
    const b2 = this.#params[`${lp}.down.b`]
    return w1 && w2 && b2 ? { w1, w2, b2 } : null
  }

  getAttnWeights(layerIdx: number): { wQ: Float32Array; wK: Float32Array; wV: Float32Array; wO: Float32Array } | null {
    const lp = `layers.${layerIdx}.attn`
    const wQ = this.#params[`${lp}.q.W`]
    const wK = this.#params[`${lp}.k.W`]
    const wV = this.#params[`${lp}.v.W`]
    const wO = this.#params[`${lp}.o.W`]
    return wQ && wK && wV && wO ? { wQ, wK, wV, wO } : null
  }

  // Idempotent — concurrent callers await the same in-flight compile.
  #ensureCompiled(): Promise<void> {
    if (this.#train) return Promise.resolve()
    if (this.#compilePromise) return this.#compilePromise
    this.#compilePromise = this.#compile()
    return this.#compilePromise
  }

  async #compile(): Promise<void> {
    this.status = 'Compiling WGSL kernels...'
    this.update()
    const t0 = performance.now()
    const model = new Transformer()
    // Trace the training graph without captures — they'd be readback every step.
    captureMode = 'off'
    this.#train = await compile({
      model,
      loss: lossFn,
      optimizer: {
        kind: 'adamw',
        lr: LR_SCHEDULE,
        // Without weight decay, train loss bottoms out around 0.9 and held-out
        // accuracy stays near 0% — the model memorizes.
        weightDecay: 0.01,
        clipGradNorm: 1.0,
      },
      inputs: {
        tokens:  { shape: [BATCH_SIZE, T_LEN], dtype: 'i32' },
        targets: { shape: [BATCH_SIZE, T_LEN], dtype: 'i32' },
        mask:    [T_LEN],
      },
    })
    captureMode = 'on'
    this.#infer = await this.#train.attach({
      forward: predictFn,
      inputs: { tokens: { shape: [null, T_LEN], dtype: 'i32' } },
    })
    this.#params = await this.#train.downloadParams()
    const compileMs = performance.now() - t0
    this.status = `Ready (${this.#train.kernels.length} kernels, ${compileMs.toFixed(0)} ms)`
    this.update()
  }

  async reset() {
    // In-place: no recompile. Inference siblings see new params via shared buffers.
    if (this.#train) {
      await this.#train.reset()
      this.#params = await this.#train.downloadParams()
    } else {
      this.#params = {}
    }
    this.isReady = true
    this.status = this.#train ? `Ready` : 'Ready (will compile WGSL on first run)'
    this.trainStep = 0
    this.examplesSeen = 0
    this.avgLoss = 0
    this.testAcc = 0
    this.#lossWindow = []
    this.#stepTimes = []
    this.#lastStepTs = 0
    this.predictedDigits = []
    this.digitProbs = []
    this.attnMaps = []
    this.inside = []
    this.lossHistory = []
    this.accHistory = []
    this.recentFailures = []
    this.embeddingHistory = []
    this.update()
    this.refreshPrediction()
  }

  toggleRun() {
    this.isRunning = !this.isRunning
    if (this.isRunning && !this.#trainingActive) this.runTrainingLoop()
    this.update()
  }

  async trainOneStep() {
    await this.#ensureCompiled()
    if (!this.#train) return
    const { tokens, targets } = makeBatch()
    let lossVal: number
    try {
      const r = await this.#train.step({ tokens, targets, mask: RESULT_MASK })
      if (r.kind !== 'completed') return
      lossVal = r.loss
    } catch (e: any) {
      this.status = `Step error: ${e?.message ?? e}`
      this.isRunning = false
      this.update()
      return
    }
    this.trainStep++
    this.examplesSeen += BATCH_SIZE

    if (!Number.isFinite(lossVal)) {
      this.status = 'Loss diverged (NaN/Inf)'
      this.isRunning = false
      this.update()
      return
    }
    this.#lossWindow.push(lossVal)
    if (this.#lossWindow.length > 50) this.#lossWindow.shift()
    this.avgLoss = this.#lossWindow.reduce((a, b) => a + b, 0) / this.#lossWindow.length

    this.lossHistory.push({ step: this.trainStep, value: this.avgLoss })
    if (this.lossHistory.length > 600) this.lossHistory.shift()

    const now = performance.now()
    if (this.#lastStepTs > 0) {
      const dt = now - this.#lastStepTs
      this.#stepTimes.push(dt)
      if (this.#stepTimes.length > 30) this.#stepTimes.shift()
      const avgDt = this.#stepTimes.reduce((a, b) => a + b, 0) / this.#stepTimes.length
      this.examplesPerSec = avgDt > 0 ? BATCH_SIZE * 1000 / avgDt : 0
    }
    this.#lastStepTs = now
  }

  async runTrainingLoop() {
    this.#trainingActive = true
    let lastUiUpdate = 0
    let lastPredictUpdate = 0
    let lastDiagnostic = 0
    while (this.isRunning) {
      await this.trainOneStep()
      await new Promise<void>(r => requestAnimationFrame(() => r()))
      const now = performance.now()
      if (now - lastUiUpdate > 100) {
        lastUiUpdate = now
        this.update()
      }
      if (now - lastPredictUpdate > 500) {
        lastPredictUpdate = now
        this.refreshPrediction()
      }
      if (now - lastDiagnostic > 3000) {
        lastDiagnostic = now
        this.#writeDiagnostic()
      }
    }
    // The last trainOneStep may have completed *after* isRunning went false (we
    // were awaiting it when pause was clicked); flush its state changes.
    this.update()
    this.#trainingActive = false
  }

  async #syncParamsFromGpu() {
    if (!this.#train) return
    try {
      this.#params = await this.#train.downloadParams()
    } catch { /* next sync will retry */ }
  }

  // Latest-wins coalescing: rapid input changes drop stale predictions in flight.
  // Sync CPU mirror for viz panels; inference itself reads from training's
  // GPU buffers, no readback needed for the forward pass.
  #refreshFlight = singleFlight(async (_: void) => {
    if (!this.#infer) return null
    await this.#syncParamsFromGpu()
    return predictAddition(this.#infer, this.root.inputPanel.operandA, this.root.inputPanel.operandB)
  })

  async refreshPrediction() {
    const r = await this.#refreshFlight()
    if (r.kind === 'aborted' || !r.value) return
    this.predictedDigits = r.value.generated
    this.digitProbs = r.value.generatedProbs
    this.attnMaps = r.value.attnMaps
    this.inside = r.value.inside
    this.update()
  }

  // Batched eval: N_TEST pairs decoded in parallel — 3 forward passes total
  // (one per result digit) vs 150 for a per-pair loop.
  async #writeDiagnostic() {
    if (this.#diagnosticBusy) return
    const evalCompiled = this.#infer
    if (!evalCompiled) return
    this.#diagnosticBusy = true
    try {
      await this.#syncParamsFromGpu()
      const testPairs: { a: number; b: number }[] = []
      let attempts = 0
      while (testPairs.length < N_TEST && attempts < 5000) {
        attempts++
        const a = Math.floor(Math.random() * 100), b = Math.floor(Math.random() * 100)
        if (isTestPair(a, b)) testPairs.push({ a, b })
      }
      // Defensive pad if isTestPair fails to yield N_TEST hits in 5000 tries.
      while (testPairs.length < N_TEST) testPairs.push(testPairs[0]!)

      const tokensBuf = new Int32Array(N_TEST * T_LEN)
      for (let row = 0; row < N_TEST; row++) {
        const { a, b } = testPairs[row]!
        const off = row * T_LEN
        const prompt = promptTokens(a, b)
        for (let i = 0; i < prompt.length; i++) tokensBuf[off + i] = prompt[i]!
      }

      const generated: number[][] = Array.from({ length: N_TEST }, () => [])
      for (let step = 0; step < N_RESULT_DIGITS; step++) {
        const r = await evalCompiled.run({ tokens: tokensBuf })  // [N_TEST, T_LEN, VOCAB]
        if (r.kind !== 'completed') return
        const logits = r.output
        const realLen = RESULT_START + step                                 // 6, 7, 8
        const readPos = realLen - 1
        for (let row = 0; row < N_TEST; row++) {
          const offset = (row * T_LEN + readPos) * VOCAB
          let best = 0
          for (let v = 1; v < 10; v++) {
            if (logits[offset + v]! > logits[offset + best]!) best = v
          }
          generated[row]!.push(best)
          if (step < N_RESULT_DIGITS - 1) tokensBuf[row * T_LEN + realLen] = best
        }
      }

      let correct = 0
      const failures: { a: number; b: number; got: number; want: number }[] = []
      for (let row = 0; row < N_TEST; row++) {
        const { a, b } = testPairs[row]!
        const got = decodeReversedTokens(generated[row]!)
        const want = a + b
        if (got === want) correct++
        else if (failures.length < 5) failures.push({ a, b, got, want })
      }

      const acc = correct / N_TEST
      this.testAcc = acc
      this.accHistory.push({ step: this.trainStep, value: acc })
      if (this.accHistory.length > 200) this.accHistory.shift()
      this.recentFailures = failures

      this.embeddingHistory.push({ step: this.trainStep, tokEmb: new Float32Array(this.#params['tok_emb']) })
      if (this.embeddingHistory.length > 120) this.embeddingHistory.shift()
      this.update()
    } catch (e: any) {
      // reset() mid-dispatch surfaces as AbortError; next tick retries.
      if (e?.name === 'AbortError') return
      throw e
    } finally {
      this.#diagnosticBusy = false
    }
  }
}

// ========== UI ==========

// ---------- UI constants ----------

const STRIP_CELL_H = 20
const TOKEN_LABELS: string[] = ['0','1','2','3','4','5','6','7','8','9','+','=']
// LSB-first to match generation order.
const RESULT_DIGIT_NAMES = ['ones', 'tens', 'hundreds'] as const

const bareStageName = (s: string) => s.replace(/^After /, '')

// Sequence layout during prediction: hundreds digit is being generated at the
// final step, so it never appears as a captured position in attention.
const POSITION_LABELS: string[] = ['A tens', 'A ones', '+', 'B tens', 'B ones', '=', 'ones', 'tens']

// ---------- UI interfaces ----------

interface IRoot {
  isNarrow: boolean
  model: IModel
  inputPanel: IInputPanel
}

interface IInputPanel {
  operandA: number
  operandB: number
}

// Layer index 0 = embedding, 1..N_LAYERS = layer 1..N. Default falls back to
// the bottom-right cell.
interface IInsidePanel {
  effectiveSelection: { layer: number; pos: number }
}

// ---------- Theme ----------
const lightTheme = {
  colors: {
    primary: 'rgb(0, 128, 0)',
    primaryStrong: 'rgb(0, 100, 0)',
    accent: 'rgb(147, 51, 234)',
    background: 'rgb(255, 255, 255)',
    surface: 'rgb(245, 245, 245)',
    text: 'rgb(95, 95, 95)',
    textMuted: 'rgb(85, 85, 85)',
    hoverBg: 'rgb(250, 250, 250)',
    border: 'rgb(220, 220, 220)',
    attn: 'rgb(70, 140, 230)',
    mlp: 'rgb(180, 100, 0)',
    error: 'rgb(220, 38, 38)'
  }
}
const darkTheme = {
  colors: {
    primary: 'rgb(0, 200, 0)',
    primaryStrong: 'rgb(0, 255, 0)',
    accent: 'rgb(192, 132, 252)',
    background: 'rgb(30, 30, 30)',
    surface: 'rgb(40, 40, 40)',
    text: 'rgb(245, 245, 245)',
    textMuted: 'rgb(170, 170, 170)',
    hoverBg: 'rgb(35, 35, 35)',
    border: 'rgb(55, 55, 55)',
    attn: 'rgb(70, 140, 230)',
    mlp: 'rgb(220, 140, 0)',
    error: 'rgb(239, 68, 68)'
  }
}

const globalUnoCss = (theme: ThemeProxy<typeof lightTheme>) => ({
  'body': `m-0 bg-${theme.colors.background} text-${theme.colors.text} font-sans antialiased overflow-x-hidden`,
  'button': `transition-colors duration-200 ease-in-out`,
  'a': `text-${theme.colors.primary} underline hover:text-${theme.colors.primaryStrong} transition-colors`,
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

const styles = themeMgr.styles('xfm', (theme) => {
  const { text, textMuted, primary, primaryStrong, surface, hoverBg, background, border, accent, error } = theme.colors
  return {
    layout: `flex flex-col min-h-screen w-full`,
    header: `flex-none pt-3 bg-${background}`,
    headerContainer: `max-w-[1300px] mx-auto px-4 md:px-6 flex flex-col items-center text-center md:items-start md:text-left`,
    headerTitle: `text-${text} text-2xl font-bold tracking-tight my-2`,
    content: `flex-1`,
    controlStripBar: `w-full max-w-[1300px] mx-auto px-0 md:px-6`,
    controlStripContent: `w-full bg-${surface} px-2 py-2 rounded-none md:rounded-lg`,
    controlsRow: `w-full flex flex-wrap items-center justify-center md:justify-start gap-x-4 gap-y-2 mt-1`,
    statsRow: `flex flex-wrap items-center justify-center gap-x-3 gap-y-1 sm:gap-x-8`,
    controlGroup: `flex items-center gap-1 sm:gap-6`,
    mainContainer: `max-w-[1300px] mx-auto px-0 md:px-6 py-4`,
    labelText: `text-xs text-${textMuted} font-bold`,
    valueText: `font-bold text-${text} whitespace-nowrap`,
    iconBtn: `bg-transparent border-none p-0 cursor-pointer text-${primary} hover:text-${primaryStrong} transition-colors outline-none flex items-center justify-center`,
    iconBtnPlay: `w-12 h-12`,
    iconBtnSmall: `w-8 h-8`,
    sectionTitle: `text-${text} text-lg font-semibold mb-3`,
    body: `text-${textMuted} text-base`,
    bodyText: `text-${text} text-base`,
    bodyMono: `text-${text} text-base font-mono`,
    panel: `bg-${surface} rounded-none md:rounded-lg p-4`,
    btn: `px-3 py-1.5 rounded-md border border-solid border-${border} bg-${background} text-${text} text-xs cursor-pointer hover:bg-${hoverBg} transition-colors`,
    waitContainer: `flex flex-col items-center justify-center min-h-screen`,
    waitText: `text-${text} mb-4`,
    waitSpinner: `mt-4 animate-bounce text-2xl`,
    successHigh: `text-${primary}`,
    successLow: `text-${error}`,
    additionRow: `flex items-baseline gap-2 my-3`,
    numberInput: `w-20 px-2 py-1 text-2xl text-center font-mono rounded-md border border-solid border-${border} bg-${background} text-${text} focus:outline-none focus:ring-2 focus:ring-${primary}`,
    bigOp: `text-3xl font-mono text-${text}`,
    bigResult: `text-3xl font-mono font-bold`,
    presetRow: `flex flex-wrap gap-2 mt-3`,
    digitProbs: `mt-3 flex flex-col gap-2`,
    attnGrid: `grid gap-3`,
    attnHeadCell: `flex flex-col items-center gap-1`,
    attnHeadLabel: `text-${textMuted} text-[11px]`,
    mainGrid: `grid gap-4`,
    chartBox: `w-full h-auto block bg-${background} rounded-md`,
    explainerContainer: `text-${textMuted}`,
    detailPanel: `mt-5 p-3 rounded-md bg-${background} border border-${border} block max-w-full box-border`,
    detailPanelTight: `mt-4 p-3 rounded-md bg-${background} border border-${border} block max-w-full box-border`,
    monoLabelTiny: `text-[11px] font-mono text-${textMuted}`,
    panelTitle: `text-[13px] text-${text} font-bold mb-2`,
    tabBtn: `px-3 py-1.5 text-sm cursor-pointer rounded-md transition-colors`,
    tabBtnActive: `bg-${primary} text-white border-none font-bold`,
    tabBtnInactive: `bg-${background} text-${textMuted} border border-solid border-${border} font-normal hover:bg-${hoverBg}`,
    subTabBtn: `px-2 py-1.5 text-sm cursor-pointer bg-transparent border-0 border-b-2 border-solid transition-colors rounded-none`,
    subTabBtnActive: `text-${text} font-bold border-${primary}`,
    subTabBtnInactive: `text-${textMuted} border-transparent hover:text-${text}`,
    statCol: `flex flex-col items-center`,
    statValueRow: `flex items-center gap-1.5`,
    latticeCell: `flex items-center justify-center font-mono text-[13px] font-bold cursor-pointer box-border text-${text}`,
    stripRow: `flex items-center gap-2 mb-0.5`,
    stripRowLabel: `text-[11px] font-mono text-right pr-1 flex-shrink-0`,
    rowLabel: `text-[11px] pr-2 text-right text-${textMuted}`,
    pickMark: `text-[10px] h-3 text-${text}`,
    vocabLabel: `text-[10px] mt-0.5 text-${textMuted}`,
    softmaxPosLabel: `text-[9px] font-mono mt-0.5 text-${textMuted}`,
    sparklineBox: `w-20 h-3.5 block flex-shrink-0 bg-${background} mt-0.5`,
    freshSwatch: `w-3.5 h-3.5 bg-${accent}/40 border border-${accent}`,
    attnDrillTitle: `text-sm font-mono text-${text}`,
    mutedText: `text-${textMuted}`,
    credit: `text-[0.85em] opacity-65 mt-1 mb-0 text-center`,
    attnDrillHeatmapBox: `flex-shrink-0 max-w-[180px]`,
    badge: `flex items-center justify-center rounded-full bg-${background}`,
  }
})

const icon = (d: string) => svg({ viewBox: '0 0 24 24', fill: 'currentColor', class: 'w-full h-full' }, path({ d }))
const iconPlay = () => icon('M8 5v14l11-7z')
const iconPause = () => icon('M6 19h4V5H6v14zm8-14v14h4V5h-4z')
const iconReset = () => icon('M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z')

// ---------- UI helpers ----------
function decodeReversedTokens(digits: number[]): number {
  let n = 0
  for (let i = digits.length - 1; i >= 0; i--) n = n * 10 + digits[i]
  return n
}

// LSB-first: [ones, tens, hundreds]. Inverse of decodeReversedTokens.
function splitResultDigits(n: number): number[] {
  return [n % 10, Math.floor(n / 10) % 10, Math.floor(n / 100)]
}

function maxAbs(values: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < values.length; i++) {
    const a = Math.abs(values[i])
    if (a > m) m = a
  }
  return m
}

// Green for positive, blue for negative, intensity = |v| / maxVal.
function makeDivergingStripColor(maxVal: number): (v: number) => string {
  const bg = themeMgr.theme.colors.surface.rawValue
  const pos = themeMgr.theme.colors.primary.rawValue
  const neg = themeMgr.theme.colors.attn.rawValue
  if (maxVal === 0) return () => bg
  return (v) => {
    const t = Math.min(1, Math.abs(v) / maxVal)
    return v >= 0 ? lerpRgb(bg, pos, t) : lerpRgb(bg, neg, t)
  }
}

// 1D array → colored strip cells. cellW omitted = flex-grow.
function stripCells(
  values: ArrayLike<number>,
  color: (v: number) => string,
  cellH: number,
  tooltip: (i: number, v: number) => string,
  cellW?: number
) {
  return range(values.length).map(i => div({
    class: cellW != null ? 'box-border' : 'flex-1 min-w-0 box-border',
    style: cellW != null
      ? { width: cellW + 'px', height: cellH + 'px', backgroundColor: color(values[i]) }
      : { height: cellH + 'px', backgroundColor: color(values[i]) },
    title: tooltip(i, values[i])
  }))
}

// Probability bar row with truth/pick highlighting.
//   bar height/opacity = prob; ▼ marks pick;
//   green bar = truth, red bar = picked-but-wrong, gray = neither.
//   When confidently wrong, the truth bar can never reach green's max brightness
//   (bounded by 1 - picked_prob) — presence of red disambiguates.
function probBarsView(opts: {
  probs: number[]
  picked: number
  truth: number
  labels: string[]
  maxBarH: number
}) {
  const { probs, picked, truth, labels, maxBarH } = opts
  const primary = themeMgr.theme.colors.primary.css
  const error = themeMgr.theme.colors.error.css
  const textCol = themeMgr.theme.colors.text.css
  // Frame height = maxBarH + 12 so the ▼ marker can ride on top of a max-height bar.
  return div({ class: 'flex gap-1' },
    probs.map((p, i) => {
      const isPicked = i === picked
      const isTruth = i === truth
      const heightPx = Math.max(2, Math.round(p * maxBarH))
      const bg = isTruth ? primary : (isPicked ? error : textCol)
      return div({ class: 'flex flex-col items-center flex-1 min-w-0' },
        div({ class: 'w-full flex flex-col items-center justify-end',
              style: { height: (maxBarH + 12) + 'px' } },
          div({ class: styles.pickMark }, isPicked ? '▼' : ''),
          div({ class: 'w-full rounded-sm', style: {
            height: heightPx + 'px',
            backgroundColor: bg,
            opacity: String(0.2 + 0.8 * p)
          }})
        ),
        div({ class: styles.vocabLabel }, labels[i])
      )
    })
  )
}

function lerpRgb(from: string, to: string, t: number): string {
  const parse = (s: string): [number, number, number] => {
    const m = s.match(/\d+/g)
    return m ? [Number(m[0]), Number(m[1]), Number(m[2])] : [128, 128, 128]
  }
  const [r1, g1, b1] = parse(from)
  const [r2, g2, b2] = parse(to)
  const k = clamp(t, 0, 1)
  return `rgb(${Math.round(r1 + (r2 - r1) * k)},${Math.round(g1 + (g2 - g1) * k)},${Math.round(b1 + (b2 - b1) * k)})`
}

function pcaBasis(X: number[][]): { mean: number[]; v1: number[]; v2: number[] } {
  const N = X.length, D = X[0].length
  const mean = new Array(D).fill(0)
  for (const row of X) for (let j = 0; j < D; j++) mean[j] += row[j]
  for (let j = 0; j < D; j++) mean[j] /= N
  const centered = X.map(row => row.map((v, j) => v - mean[j]))
  const cov: number[][] = Array.from({ length: D }, () => new Array(D).fill(0))
  for (let i = 0; i < D; i++) {
    for (let j = i; j < D; j++) {
      let s = 0
      for (let k = 0; k < N; k++) s += centered[k][i] * centered[k][j]
      cov[i][j] = cov[j][i] = s / N
    }
  }
  const v1 = powerIter(cov)
  const Mv1 = mvm(cov, v1)
  const lambda1 = v1.reduce((s, x, i) => s + x * Mv1[i], 0)
  for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) cov[i][j] -= lambda1 * v1[i] * v1[j]
  const v2 = powerIter(cov)
  return { mean, v1, v2 }
}

function projectThroughBasis(X: number[][], basis: { mean: number[]; v1: number[]; v2: number[] }): [number, number][] {
  return X.map(row => {
    let p1 = 0, p2 = 0
    for (let i = 0; i < row.length; i++) {
      const c = row[i] - basis.mean[i]
      p1 += c * basis.v1[i]
      p2 += c * basis.v2[i]
    }
    return [p1, p2] as [number, number]
  })
}

function powerIter(M: number[][], iters = 80): number[] {
  const D = M.length
  let v = new Array(D).fill(0).map(() => Math.random() - 0.5)
  let n = Math.sqrt(v.reduce((s, x) => s + x * x, 0))
  for (let i = 0; i < D; i++) v[i] /= n
  for (let it = 0; it < iters; it++) {
    const Mv = mvm(M, v)
    n = Math.sqrt(Mv.reduce((s, x) => s + x * x, 0))
    if (n < 1e-12) return v
    for (let i = 0; i < D; i++) v[i] = Mv[i] / n
  }
  return v
}

function mvm(M: number[][], v: number[]): number[] {
  return M.map(row => {
    let s = 0
    for (let i = 0; i < v.length; i++) s += row[i] * v[i]
    return s
  })
}

// Top-N entries by |w| as (i, j, w) triples sorted descending.
function pickTopWeights(w: Float32Array, rows: number, cols: number, topN: number): { i: number; j: number; w: number }[] {
  const all: { i: number; j: number; w: number }[] = []
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      all.push({ i, j, w: w[i * cols + j] })
    }
  }
  all.sort((a, b) => Math.abs(b.w) - Math.abs(a.w))
  return all.slice(0, topN)
}

function countFiring(vals: ArrayLike<number>): number {
  let n = 0
  for (let i = 0; i < vals.length; i++) if (vals[i] > 0) n++
  return n
}

// ---------- Attention panel ----------
type AttnDetailTab = 'project' | 'score' | 'output'

class AttentionPanel extends Component {
  selectedAttnCell: { layer: number; head: number; qPos: number; kPos: number } | null = null
  // Default to the last step; sticky across prediction refreshes.
  activeAttnStep = N_RESULT_DIGITS - 1
  // Sticky across cell selections.
  attnDetailTab: AttnDetailTab = 'project'

  get root() { return this.ctx.root as any as IRoot }

  get currentStepAttnMaps(): AttnMap[] {
    return this.root.model.attnMaps[this.activeAttnStep] ?? []
  }

  // -1 = "all positions fresh, no cache yet" (first step); otherwise the last
  // position, which is the only K/V freshly computed this step.
  get freshColIdx(): number {
    const T_len = this.currentStepAttnMaps[0]?.T ?? 0
    return this.activeAttnStep === 0 ? -1 : T_len - 1
  }

  view() {
    if (this.selectedAttnCell !== null) {
      return this.attnDrillDownView()
    }
    return div(
      this.kvCacheStepNav(),
      div({ class: [styles.body, 'mb-2'] }, 'Each row shows where that position is looking; brighter cells = more attention. Click any cell to drill in and see the math behind it (Q, K, V).'),
      this.attnGridView(),
      this.freshColumnHint()
    )
  }

  // K/V are recomputed every step (vs cached in real LLM inference). Caption
  // calls out which column would be the "fresh" one in a cached impl.
  kvCacheStepNav() {
    if (this.root.model.attnMaps.length === 0) return div()
    const numSteps = this.root.model.attnMaps.length
    const stepNames = RESULT_DIGIT_NAMES
    const active = this.activeAttnStep
    const T_len = this.currentStepAttnMaps[0]?.T ?? 0
    const freshIdx = this.freshColIdx
    const freshLabel = freshIdx >= 0 ? POSITION_LABELS[freshIdx] : null
    const caption = active === 0
      ? `Generation Step 1/${numSteps}: the model sees the prompt (${T_len} tokens). It's about to generate the ${stepNames[active]} digit.`
      : `Generation Step ${active + 1}/${numSteps}: the model has just generated the ${stepNames[active - 1]} digit (position ${freshIdx} = "${freshLabel}"). Only this new position needed fresh K and V values this step; positions 0–${freshIdx - 1} already had theirs from earlier steps (real LLMs save and reuse those past values — this is the K/V cache).`
    return div({ class: 'mb-4' },
      div({ class: 'flex items-center gap-3 flex-wrap mb-2' },
        range(numSteps).map(s => {
          const isActive = s === active
          return button({
            class: [styles.subTabBtn, isActive ? styles.subTabBtnActive : styles.subTabBtnInactive],
            onClick: () => {
              if (this.activeAttnStep === s) return
              this.activeAttnStep = s
              this.selectedAttnCell = null
              this.update()
            }
          }, stepNames[s])
        })
      ),
      div({ class: styles.body }, caption)
    )
  }

  freshColumnHint() {
    if (this.root.model.attnMaps.length === 0) return div()
    const freshIdx = this.freshColIdx
    if (freshIdx < 0) return div()
    const freshLabel = POSITION_LABELS[freshIdx]
    return div({ class: [styles.labelText, 'mt-2 flex items-center justify-center gap-1.5'] },
      div({ class: styles.freshSwatch }),
      `= column "${freshLabel}" — K/V freshly computed at this step.`
    )
  }

  attnDrillDownView() {
    const sel = this.selectedAttnCell!
    const idx = sel.layer * N_HEADS + sel.head
    const am = this.currentStepAttnMaps[idx]
    if (!am) {
      this.selectedAttnCell = null
      return div()
    }
    const qLabel = POSITION_LABELS[sel.qPos] ?? `pos ${sel.qPos}`
    const kLabel = POSITION_LABELS[sel.kPos] ?? `pos ${sel.kPos}`

    return div(
      div({ class: 'flex items-center gap-3 mb-3 flex-wrap' },
        button({
          class: styles.btn,
          onClick: () => { this.selectedAttnCell = null; this.update() }
        }, '← Back to all heads'),
        div({ class: styles.attnDrillTitle }, `Layer ${sel.layer + 1} · Head ${sel.head + 1}`)
      ),
      div({ class: ['flex gap-4 items-center mb-1', this.root.isNarrow ? 'flex-col' : 'flex-row'] },
        div({ class: styles.attnDrillHeatmapBox },
          this.heatmapView(am, sel.layer, sel.head)
        ),
        div({ class: 'flex-1 min-w-0' },
          div({ class: styles.panelTitle },
            `Attention weight: query at "${qLabel}" → key at "${kLabel}"`
          ),
          div({ class: styles.body },
            `Attention lets each position selectively read context from itself and earlier positions. Each position acts as three things: a query ("what am I looking for?"), a key ("what do I offer?"), and a value (the payload to share). Every query weights each available key, and those weights blend the values into the query's read.`
          ),
          div({ class: [styles.body, 'mt-2'] },
            `Every cell in this heatmap is one such weight, post-softmax. You clicked the cell where the query at "${qLabel}" attends to the key at "${kLabel}".`
          )
        )
      ),
      this.attnDetailView()
    )
  }

  attnDetailView() {
    const sel = this.selectedAttnCell
    if (!sel || this.currentStepAttnMaps.length === 0) return div()
    const idx = sel.layer * N_HEADS + sel.head
    const am = this.currentStepAttnMaps[idx]
    if (!am) return div()
    const T_len = am.T
    if (sel.qPos >= T_len || sel.kPos >= T_len) return div()

    const qVec = am.q.slice(sel.qPos * D_HEAD, (sel.qPos + 1) * D_HEAD)
    const kVec = am.k.slice(sel.kPos * D_HEAD, (sel.kPos + 1) * D_HEAD)
    const vVec = am.v.slice(sel.kPos * D_HEAD, (sel.kPos + 1) * D_HEAD)
    const products = qVec.map((q, i) => q * kVec[i])
    const dotProduct = products.reduce((a, b) => a + b, 0)
    const score = dotProduct / Math.sqrt(D_HEAD)

    const qLabel = POSITION_LABELS[sel.qPos] ?? `pos ${sel.qPos}`
    const kLabel = POSITION_LABELS[sel.kPos] ?? `pos ${sel.kPos}`

    // Causal mask: weights run 0..qPos.
    const rowWeights: number[] = []
    for (let j = 0; j <= sel.qPos; j++) rowWeights.push(am.data[sel.qPos * T_len + j])

    const headOutput: number[] = new Array(D_HEAD).fill(0)
    for (let j = 0; j <= sel.qPos; j++) {
      const w = rowWeights[j]
      for (let d = 0; d < D_HEAD; d++) {
        headOutput[d] += w * am.v[j * D_HEAD + d]
      }
    }

    const stripColor = makeDivergingStripColor(maxAbs([...products, ...headOutput]))

    const cellH = this.root.isNarrow ? 18 : 24
    const labelW = this.root.isNarrow ? 100 : 140

    const renderStrip = (vec: number[], label?: string) => {
      const cells = div({ class: 'flex gap-0 flex-1 min-w-0' },
        stripCells(vec, stripColor, cellH, (i, v) => `dim ${i}: ${v.toFixed(3)}`)
      )
      return label
        ? div({ class: 'flex items-center gap-2 mb-1' },
            div({ class: [styles.monoLabelTiny, styles.stripRowLabel], style: { width: labelW + 'px' } }, label),
            cells)
        : div({ class: 'flex mb-1' }, cells)
    }

    const inside = this.root.model.inside[this.activeAttnStep]
    const attnW = this.root.model.getAttnWeights(sel.layer)

    // Per-head slice of the layer's [D_MODEL, D_MODEL] projection matrix.
    // Q/K/V take a column slice → [D_MODEL, D_HEAD]; W_o takes a row slice → [D_HEAD, D_MODEL].
    const colSliceHead = (W: Float32Array): Float32Array => {
      const out = new Float32Array(D_MODEL * D_HEAD)
      for (let i = 0; i < D_MODEL; i++) {
        for (let j = 0; j < D_HEAD; j++) {
          out[i * D_HEAD + j] = W[i * D_MODEL + sel.head * D_HEAD + j]
        }
      }
      return out
    }
    const rowSliceHead = (W: Float32Array): Float32Array => {
      const out = new Float32Array(D_HEAD * D_MODEL)
      for (let i = 0; i < D_HEAD; i++) {
        for (let j = 0; j < D_MODEL; j++) {
          out[i * D_MODEL + j] = W[(sel.head * D_HEAD + i) * D_MODEL + j]
        }
      }
      return out
    }

    // Per-head contribution to residual_out at qPos = head_output @ W_o_slice.
    // The full residual sums all heads' contributions; this drilldown shows only one.
    let headContrib: Float32Array | null = null
    let wOSlice: Float32Array | null = null
    if (attnW) {
      wOSlice = rowSliceHead(attnW.wO)
      headContrib = new Float32Array(D_MODEL)
      for (let i = 0; i < D_MODEL; i++) {
        let s = 0
        for (let d = 0; d < D_HEAD; d++) s += headOutput[d] * wOSlice[d * D_MODEL + i]
        headContrib[i] = s
      }
    }

    const projectSection = inside && attnW
      ? div({ class: 'mb-3' },
          div({ class: [styles.body, 'mb-3 text-center'] },
            `First, project each position's residual to produce its Q, K, V. For this pair: Q from the residual at "${qLabel}", K and V from the residual at "${kLabel}". Residual on top, projected vector below.`
          ),
          this.renderProjectionWires(inside.lattice[sel.layer][sel.qPos].residual, qVec, colSliceHead(attnW.wQ), D_MODEL, D_HEAD, `Residual at "${qLabel}"`, `Query at "${qLabel}"`),
          this.renderProjectionWires(inside.lattice[sel.layer][sel.kPos].residual, kVec, colSliceHead(attnW.wK), D_MODEL, D_HEAD, `Residual at "${kLabel}"`, `Key at "${kLabel}"`),
          this.renderProjectionWires(inside.lattice[sel.layer][sel.kPos].residual, vVec, colSliceHead(attnW.wV), D_MODEL, D_HEAD, `Residual at "${kLabel}" (shared with K)`, `Value at "${kLabel}"`)
        )
      : div()

    const scoreSection = div(
      div({ class: [styles.body, 'mb-2 text-center'] },
        `Next, measure how well every query matches every key (one matmul). Below, we zoom into your selected pair: Q at "${qLabel}" against K at "${kLabel}".`
      ),
      div({ class: [styles.body, 'mb-1.5 text-center'] },
        `The ${D_HEAD} cells below are the element-wise product of Q (at "${qLabel}") and K (at "${kLabel}") — one cell per dimension, each cell = Q[d] × K[d]. Bright = strong agreement (score up); pale cells contribute little either way.`
      ),
      renderStrip(products),
      div({ class: [styles.body, 'mt-2.5 text-center'] },
        `Adding these ${D_HEAD} cells gives ${dotProduct.toFixed(2)}. Divided by √${D_HEAD}, that's this pair's raw score: ${score.toFixed(3)}.`
      ),
      div({ class: [styles.body, 'mt-3.5 mb-2 text-center'] },
        `Softmax across this row's scores (one per attended key — ${sel.qPos + 1} here, due to causal mask) → attention weights. The one for the selected key "${kLabel}" is `,
        span({ style: { color: themeMgr.theme.colors.accent.css } }, '['),
        span({ style: {
          fontWeight: 'bold',
          color: lerpRgb(themeMgr.theme.colors.surface.rawValue, themeMgr.theme.colors.primary.rawValue, rowWeights[sel.kPos])
        } }, `${(rowWeights[sel.kPos] * 100).toFixed(1)}%`),
        span({ style: { color: themeMgr.theme.colors.accent.css } }, ']'),
        ` of query "${qLabel}" 's attention.`
      ),
      this.renderSoftmaxBars(rowWeights, sel.qPos, sel.kPos)
    )

    const outputSection = div(
      div({ class: [styles.body, 'mb-2 text-center'] },
        `Finally, compute the weighted sum of the V vectors using the row's attention weights (one matmul). The result is a ${D_HEAD}-dim vector — this head's output. All heads' outputs are concatenated and projected by W_o into a ${D_MODEL}-dim vector, then added to the residual stream. Below: this head's slice of that projection.`
      ),
      headContrib && wOSlice
        ? this.renderProjectionWires(headOutput, headContrib, wOSlice, D_HEAD, D_MODEL, `V blend at "${qLabel}"`, `This head's output at "${qLabel}"`)
        : renderStrip(headOutput, `This head's output at "${qLabel}"`)
    )

    const tabs: { id: AttnDetailTab; label: string }[] = [
      { id: 'project', label: 'Project →' },
      { id: 'score',   label: 'Score →' },
      { id: 'output',  label: 'Attend' },
    ]
    const tabBar = div({ class: 'flex gap-3 mb-3' },
      tabs.map(t => button({
        class: [styles.subTabBtn, this.attnDetailTab === t.id ? styles.subTabBtnActive : styles.subTabBtnInactive],
        onClick: () => { this.attnDetailTab = t.id; this.update() }
      }, t.label))
    )

    return div({ class: styles.detailPanelTight },
      tabBar,
      this.attnDetailTab === 'project' ? projectSection
        : this.attnDetailTab === 'score' ? scoreSection
        : outputSection
    )
  }

  // Bar chart of the row's softmaxed attention weights. The selected key's bar
  // gets a purple outline matching the heatmap cell highlight.
  renderSoftmaxBars(rowWeights: number[], qPos: number, kPos: number): VElement {
    const textCol = themeMgr.theme.colors.text.rawValue
    const primary = themeMgr.theme.colors.primary.rawValue
    const accentCss = themeMgr.theme.colors.accent.css
    return div({ class: 'flex items-end gap-0.5 justify-center', style: { minHeight: '54px' } },
      range(qPos + 1).map(j => {
        const w = rowWeights[j]
        const isSelKey = j === kPos
        const heightPx = Math.max(2, Math.round(w * 45))
        return div({ class: 'flex flex-col items-center w-8' },
          div({ class: 'box-content', style: {
            width: '22px',
            height: heightPx + 'px',
            backgroundColor: isSelKey ? primary : textCol,
            opacity: String(0.3 + 0.7 * w),
            border: isSelKey ? `2px solid ${accentCss}` : 'none'
          } }),
          div({ class: styles.softmaxPosLabel }, POSITION_LABELS[j] ?? `${j}`)
        )
      })
    )
  }

  // Source-strip → top-N weight fan → dest-strip diagram. Caller pre-slices the
  // per-head weight matrix to [srcN, dstN]; this method is direction-agnostic.
  renderProjectionWires(
    srcVec: ArrayLike<number>,
    dstVec: ArrayLike<number>,
    headW: Float32Array,
    srcN: number,
    dstN: number,
    srcLabel: string,
    dstLabel: string,
  ): VElement {
    const W = 700, H = 110
    const PAD_X = 24
    const STRIP_X0 = PAD_X
    const STRIP_W = W - 2 * PAD_X
    const ROW_H = STRIP_CELL_H
    const Y_TOP = 24
    const Y_BOT = 76

    const xCenter = (j: number, n: number) => STRIP_X0 + STRIP_W * (j + 0.5) / n
    const xEdge = (j: number, n: number) => STRIP_X0 + STRIP_W * j / n

    const TOP_N = 80
    const wTop = pickTopWeights(headW, srcN, dstN, TOP_N)
    const wColor = makeDivergingStripColor(Math.abs(wTop[0]?.w ?? 1e-6))

    const renderRow = (vals: ArrayLike<number>, n: number, y: number): VElement[] => {
      const color = makeDivergingStripColor(maxAbs(vals) || 1)
      const cellW = STRIP_W / n
      const cells: VElement[] = []
      for (let i = 0; i < n; i++) {
        cells.push(rect({
          // +0.5 closes hairline gaps from non-integer cellW.
          x: xEdge(i, n), y, width: cellW + 0.5, height: ROW_H,
          fill: color(vals[i])
        }))
      }
      return cells
    }

    const fanLines = wTop.map(c => line({
      x1: xCenter(c.i, srcN), y1: Y_TOP + ROW_H,
      x2: xCenter(c.j, dstN), y2: Y_BOT,
      stroke: wColor(c.w), strokeWidth: 0.6
    }))

    const topCells = renderRow(srcVec, srcN, Y_TOP)
    const botCells = renderRow(dstVec, dstN, Y_BOT)

    const labelColor = themeMgr.theme.colors.textMuted.css
    const labels: VElement[] = [
      text({ x: PAD_X, y: 16, fill: labelColor, fontSize: '11', textAnchor: 'start' }, srcLabel),
      text({ x: PAD_X, y: Y_BOT + ROW_H + 14, fill: labelColor, fontSize: '11', textAnchor: 'start' }, dstLabel)
    ]

    return svg({
      viewBox: `0 0 ${W} ${H}`,
      width: '100%',
      class: 'block max-w-full mb-6',
      preserveAspectRatio: 'xMidYMid meet'
    }, ...fanLines, ...topCells, ...botCells, ...labels)
  }

  attnGridView() {
    const stepMaps = this.currentStepAttnMaps
    if (stepMaps.length === 0) {
      return div({ class: [styles.body, 'py-6'] }, 'No attention to show yet.')
    }
    return div({ class: styles.attnGrid, style: { gridTemplateColumns: `repeat(${N_HEADS}, minmax(0, 1fr))` } },
      range(N_LAYERS * N_HEADS).map(idx => {
        const layer = Math.floor(idx / N_HEADS)
        const head = idx % N_HEADS
        const attn = stepMaps[idx]
        const T = attn.T
        let bestPos = 0
        let bestWeight = -1
        for (let j = 0; j <= T - 1; j++) {
          const w = attn.data[(T - 1) * T + j]
          if (w > bestWeight) { bestWeight = w; bestPos = j }
        }
        const label = POSITION_LABELS[bestPos] ?? `pos ${bestPos}`
        return div({ class: styles.attnHeadCell },
          div({ class: styles.attnHeadLabel }, `L${layer + 1} H${head + 1} → ${label} (${(bestWeight * 100).toFixed(0)}%)`),
          this.heatmapView(attn, layer, head)
        )
      })
    )
  }

  heatmapView(attn: AttnMap, layer: number, head: number) {
    const T_len = attn.T
    const labels: string[] = promptTokens(this.root.inputPanel.operandA, this.root.inputPanel.operandB)
      .map(t => TOKEN_LABELS[t])
    for (let i = 0; i < this.root.model.predictedDigits.length && labels.length < T_len; i++) {
      labels.push(String(this.root.model.predictedDigits[i]))
    }
    while (labels.length < T_len) labels.push('?')
    const cellSize = Math.max(18, Math.min(32, Math.floor(220 / T_len)))
    const labelSize = 18
    const totalSize = T_len * cellSize + labelSize
    const cellBg = themeMgr.theme.colors.surface.rawValue
    const cellHi = themeMgr.theme.colors.primary.rawValue
    const cellColor = (a: number) => lerpRgb(cellBg, cellHi, a)
    const freshCol = this.freshColIdx
    const freshColor = themeMgr.theme.colors.accent.css  // purple, matches selection ring
    const cells: VElement[] = []
    if (freshCol >= 0) {
      cells.push(rect({
        x: labelSize + freshCol * cellSize - 1, y: labelSize - 1,
        width: cellSize + 1, height: T_len * cellSize + 1,
        fill: freshColor, opacity: '0.18'
      }))
    }
    for (let j = 0; j < T_len; j++) {
      const isFresh = j === freshCol
      cells.push(text({
        x: labelSize + j * cellSize + cellSize / 2,
        y: labelSize - 4,
        textAnchor: 'middle',
        fontSize: '10',
        fontFamily: 'monospace',
        fontWeight: isFresh ? 'bold' as any : 'normal' as any,
        fill: isFresh ? freshColor : themeMgr.theme.colors.textMuted.css
      }, labels[j]))
    }
    for (let i = 0; i < T_len; i++) {
      cells.push(text({
        x: labelSize - 4,
        y: labelSize + i * cellSize + cellSize / 2 + 3,
        textAnchor: 'end',
        fontSize: '10',
        fontFamily: 'monospace',
        fill: themeMgr.theme.colors.textMuted.css
      }, labels[i]))
    }
    const sel = this.selectedAttnCell
    const selHere = sel && sel.layer === layer && sel.head === head
    for (let i = 0; i < T_len; i++) {
      for (let j = 0; j <= i; j++) {
        const a = attn.data[i * T_len + j]
        const isSel = selHere && sel!.qPos === i && sel!.kPos === j
        cells.push(rect({
          x: labelSize + j * cellSize, y: labelSize + i * cellSize,
          width: cellSize - 1, height: cellSize - 1,
          fill: cellColor(a),
          stroke: isSel ? themeMgr.theme.colors.accent.rawValue : 'none',
          strokeWidth: isSel ? '2' : '0',
          class: 'cursor-pointer',
          onClick: () => {
            this.selectedAttnCell = { layer, head, qPos: i, kPos: j }
            this.update()
          }
        }))
      }
    }
    return svg({
      viewBox: `0 0 ${totalSize} ${totalSize}`,
      preserveAspectRatio: 'xMidYMid meet',
      class: 'w-full h-auto block',
      style: { maxWidth: totalSize + 'px' }
    }, cells)
  }
}

// ---------- Inside-the-model panel ----------
class InsidePanel extends Component implements IInsidePanel {
  selectedCell: { layer: number; pos: number } | null = null
  // Sticky across cell selections.
  cellDetailTab: 'stages' | 'wires' | 'vocab' = 'stages'

  wiresPanel = new WiresPanel()

  get root() { return this.ctx.root as any as IRoot }

  // Defaults to bottom-right cell (drives the next token).
  get effectiveSelection(): { layer: number; pos: number } {
    const inside = this.root.model.inside
    const stagesLen = N_LAYERS + 1  // embedding + N_LAYERS
    if (inside.length === 0) return { layer: stagesLen - 1, pos: 0 }
    const step = inside[inside.length - 1]
    const T = step.inputTokens.length
    const sel = this.selectedCell ?? { layer: stagesLen - 1, pos: T - 1 }
    return {
      layer: clamp(sel.layer, 0, stagesLen - 1),
      pos: clamp(sel.pos, 0, T - 1)
    }
  }

  view() {
    if (this.root.model.inside.length === 0) {
      return div({ class: styles.body }, 'Waiting for prediction...')
    }
    // Last step's lattice subsumes all earlier ones — causal masking means
    // residuals at positions 0..N depend only on inputs 0..N.
    const step = this.root.model.inside[this.root.model.inside.length - 1]
    return div(
      div({ class: [styles.body, 'mb-2 text-center'] },
        `One forward pass producing the final result. Each cell shows the model's most-likely next-token prediction at that (depth, position), with brightness = confidence. The three circled digits in the bottom row are the digits the model produced. Click any cell for its full distribution.`
      ),
      this.latticeView(step)
    )
  }

  latticeView(step: InsideStep) {
    const T = step.inputTokens.length
    const stageNames = this.root.isNarrow
      ? ['Emb', ...range(N_LAYERS).map(l => `L${l + 1}`)]
      : ['After embedding', ...range(N_LAYERS).map(l => `After layer ${l + 1}`)]
    const cellW = this.root.isNarrow ? 28 : 38
    const cellH = this.root.isNarrow ? 24 : 28
    const labelColW = this.root.isNarrow ? 44 : 110
    const headerH = 28
    const cellBg = themeMgr.theme.colors.surface.rawValue
    const cellHi = themeMgr.theme.colors.primary.rawValue
    const cellColor = (p: number) => lerpRgb(cellBg, cellHi, p)
    const badgeSize = cellH - 6
    const trueResultDigits = splitResultDigits(this.root.inputPanel.operandA + this.root.inputPanel.operandB)

    const sel = this.selectedCell ?? { layer: stageNames.length - 1, pos: T - 1 }
    const selLayer = clamp(sel.layer, 0, stageNames.length - 1)
    const selPos = clamp(sel.pos, 0, T - 1)

    return div({ class: 'overflow-auto' },
      div({ class: 'mx-auto w-fit' },
        div({ class: 'flex items-end mb-0.5 gap-1', style: { height: headerH + 'px' } },
          div({ style: { width: labelColW + 'px' } }),
          range(T).map(pos => {
            // Positions > T - N_RESULT_DIGITS are the model's own previous outputs fed back.
            const isPredInput = pos > T - N_RESULT_DIGITS
            let colorClass = styles.mutedText
            if (isPredInput) {
              const idx = pos - (T - N_RESULT_DIGITS + 1)
              colorClass = step.inputTokens[pos] === trueResultDigits[idx] ? styles.successHigh : styles.successLow
            }
            return div({
              class: ['text-center text-[11px] font-mono', isPredInput ? 'font-bold' : '', colorClass],
              style: { width: cellW + 'px' }
            }, `${pos}\n` + TOKEN_LABELS[step.inputTokens[pos]])
          })
        ),
        stageNames.map((stageName, layerIdx) => {
          const cells = step.lattice[layerIdx]
          return div({ class: 'flex items-center mb-0.5 gap-1' },
            div({ class: styles.rowLabel, style: { width: labelColW + 'px' } }, stageName),
            range(T).map(pos => {
              const cell = cells[pos]
              const isFinalRow = layerIdx === stageNames.length - 1
              // Final-layer cells at the last N_RESULT_DIGITS positions — each drove a generated digit.
              const isGenerationCell = isFinalRow && pos >= T - N_RESULT_DIGITS
              const isSelected = layerIdx === selLayer && pos === selPos
              const accent = themeMgr.theme.colors.accent.css
              const shadows: string[] = []
              if (isSelected) shadows.push(`inset 0 0 0 3px ${accent}`)
              if (shadows.length === 0) shadows.push(`inset 0 0 0 1px ${themeMgr.theme.colors.border.css}`)
              let cellChild: any = TOKEN_LABELS[cell.topToken]
              if (isGenerationCell) {
                const idx = pos - (T - N_RESULT_DIGITS)
                const isCorrect = cell.topToken === trueResultDigits[idx]
                cellChild = span({
                  class: [styles.badge, isCorrect ? styles.successHigh : styles.successLow],
                  style: { width: badgeSize + 'px', height: badgeSize + 'px' }
                }, TOKEN_LABELS[cell.topToken])
              }
              return div({
                class: styles.latticeCell,
                style: {
                  width: cellW + 'px',
                  height: cellH + 'px',
                  backgroundColor: cellColor(cell.topProb),
                  boxShadow: shadows.join(', ')
                },
                title: `${TOKEN_LABELS[cell.topToken]} ${(cell.topProb * 100).toFixed(0)}%`,
                onClick: () => { this.selectedCell = { layer: layerIdx, pos }; this.update() }
              }, cellChild)
            })
          )
        })
      ),
      this.cellDetailView(step, selLayer, selPos, stageNames)
    )
  }

  cellDetailView(step: InsideStep, layerIdx: number, pos: number, stageNames: string[]) {
    const T = step.inputTokens.length
    // At result positions, inputTokens[pos+1] is the model's own prior prediction
    // (possibly wrong) — use the ground-truth digit instead.
    const correctToken = pos >= T - N_RESULT_DIGITS
      ? splitResultDigits(this.root.inputPanel.operandA + this.root.inputPanel.operandB)[pos - (T - N_RESULT_DIGITS)]
      : step.inputTokens[pos + 1]

    const subTabs: { id: 'stages' | 'wires' | 'vocab'; label: string }[] = [
      { id: 'stages', label: 'Residual' },
      { id: 'wires', label: 'MLP' },
      { id: 'vocab', label: 'Projection' }
    ]
    return div({ class: styles.detailPanel },
      div({ class: [styles.body, 'mb-2 text-center'] },
        `Position ${pos} (token "${TOKEN_LABELS[step.inputTokens[pos]]}") · selected stage: after ${bareStageName(stageNames[layerIdx])} · correct next token at this position is "${TOKEN_LABELS[correctToken]}"`
      ),
      div({ class: 'flex gap-x-3 gap-y-0 mb-3 flex-wrap justify-center' },
        subTabs.map(t => button({
          class: [styles.subTabBtn, this.cellDetailTab === t.id ? styles.subTabBtnActive : styles.subTabBtnInactive],
          onClick: () => { this.cellDetailTab = t.id; this.update() }
        }, t.label))
      ),
      this.cellDetailTab === 'wires' ? this.wiresPanel.view() :
      this.cellDetailTab === 'vocab' ? this.vocabView(step.lattice[layerIdx][pos], correctToken, stageNames[layerIdx]) :
      this.stagesView(step, layerIdx, pos, stageNames)
    )
  }

  // Globally normalized across stages so layer-to-layer magnitude differences are visible.
  stagesView(step: InsideStep, layerIdx: number, pos: number, stageNames: string[]): VElement {
    const allStageResiduals = step.lattice.map(stageCells => stageCells[pos].residual)
    const stageMaxes = allStageResiduals.map(maxAbs)
    const stripColor = makeDivergingStripColor(Math.max(...stageMaxes))
    const stripLabelW = this.root.isNarrow ? 86 : 190
    const primaryCss = themeMgr.theme.colors.primary.css
    return div({ class: 'mb-4' },
      div({ class: [styles.body, 'mb-1.5 text-center'] },
        `Residual stream at this position, across all stages (block boundaries). After attention heads and MLPs have added into it.`
      ),
      allStageResiduals.map((sRes, sIdx) => {
        const isSel = sIdx === layerIdx
        return div({ class: styles.stripRow },
          div({
            class: [styles.stripRowLabel, isSel ? styles.successHigh : styles.mutedText, isSel ? 'font-bold' : ''],
            style: { width: stripLabelW + 'px' }
          }, `${stageNames[sIdx]}  max=${stageMaxes[sIdx].toFixed(2)}`),
          div({
            class: 'flex gap-0 flex-1 min-w-0 box-border',
            style: { boxShadow: isSel ? `0 0 0 2px ${primaryCss}` : 'none' }
          },
            stripCells(sRes, stripColor, STRIP_CELL_H, (i, v) => `${stageNames[sIdx]} · channel ${i}: ${v.toFixed(3)}`)
          )
        )
      })
    )
  }

  vocabView(cell: LatticeCell, correctToken: number, stageName: string): VElement {
    const topIdx = cell.probs.reduce((best, p, i) => p > cell.probs[best] ? i : best, 0)
    return div(
      div({ class: [styles.body, 'mb-1 text-center'] }, `Vocabulary projection after ${bareStageName(stageName)} (logit lens):`),
      probBarsView({
        probs: cell.probs,
        picked: topIdx,
        truth: correctToken,
        labels: range(VOCAB).map(v => TOKEN_LABELS[v]),
        maxBarH: 70
      })
    )
  }
}

// ---------- Embeddings panel ----------
class TokenEmbeddingsPanel extends Component {
  selectedEmbToken: number | null = null
  embeddingScrubIdx: number | null = null  // null = live latest; otherwise index into history
  embeddingPlaying = false
  #playTimer: number | null = null

  get root() { return this.ctx.root as any as IRoot }
  get embeddingHistory() { return this.root.model.embeddingHistory }

  reset() {
    this.embeddingScrubIdx = null
    this.selectedEmbToken = null
    this.embeddingPlaying = false
    if (this.#playTimer != null) { clearInterval(this.#playTimer); this.#playTimer = null }
  }

  // null = live (latest frame).
  get scrubValue(): number {
    const len = this.embeddingHistory.length
    if (len === 0) return 0
    return this.embeddingScrubIdx == null ? len - 1 : this.embeddingScrubIdx
  }
  set scrubValue(v: number) {
    const len = this.embeddingHistory.length
    if (len === 0) return
    this.#stopEmbeddingPlay()
    this.embeddingScrubIdx = v >= len - 1 ? null : Math.max(0, v)
  }

  view() {
    const flat = this.#currentEmbeddingFlat()
    if (!flat || flat.length === 0) {
      return div({ class: styles.body }, 'Waiting for first snapshot...')
    }
    const displayedRows = this.#rowsFromFlat(flat)

    // Axes locked to LATEST snapshot so they don't wobble while scrubbing.
    const latestFlat = this.embeddingHistory.length > 0
      ? this.embeddingHistory[this.embeddingHistory.length - 1].tokEmb
      : flat
    const latestRows = this.#rowsFromFlat(latestFlat)
    const basis = pcaBasis(latestRows)
    const latestPoints = projectThroughBasis(latestRows, basis)
    const allXs = latestPoints.map(p => p[0])
    const allYs = latestPoints.map(p => p[1])
    const xRange = Math.max(...allXs) - Math.min(...allXs)
    const yRange = Math.max(...allYs) - Math.min(...allYs)
    // Pad 10% so early frames (clustered near center) have room.
    const pad = 0.1
    const minX = Math.min(...allXs) - xRange * pad
    const maxX = Math.max(...allXs) + xRange * pad
    const minY = Math.min(...allYs) - yRange * pad
    const maxY = Math.max(...allYs) + yRange * pad

    const points = projectThroughBasis(displayedRows, basis)
    const W = 480, H = 360, plotPad = 30
    const sx = (x: number) => plotPad + (x - minX) / Math.max(1e-9, maxX - minX) * (W - 2 * plotPad)
    const sy = (y: number) => H - plotPad - (y - minY) / Math.max(1e-9, maxY - minY) * (H - 2 * plotPad)
    const labelColor = (v: number) => v < 10 ? themeMgr.theme.colors.primary.rawValue : themeMgr.theme.colors.text.rawValue

    const embStripColor = makeDivergingStripColor(
      Math.max(...displayedRows.map(r => maxAbs(r)))
    )
    const sel = this.selectedEmbToken

    const histLen = this.embeddingHistory.length
    const isLive = this.embeddingScrubIdx == null
    const displayStep = isLive ? this.root.model.trainStep : this.embeddingHistory[this.embeddingScrubIdx!].step
    const sliderMax = histLen > 0 ? histLen - 1 : 0

    return div(
      div({ class: [styles.body, 'mb-2'] }, 'Each token\'s 64-dim embedding, shown in 2D — the plane that spreads the 12 tokens as far apart as possible (PCA). As training progresses, the digit tokens walk into a circle in numerical order — 0, 1, 2, …, 9, with 9 next to 0 — and the model has discovered the cyclic structure of mod-10 arithmetic. The "+" and "=" tokens sit near the center. The axes are locked to the latest snapshot, so when you scrub backward through training history the projection doesn\'t wobble — you watch the circle actually form.'),
      div({ class: [styles.body, 'mb-2'] }, 'Reminiscent of the "grokking" paper (Power et al.): small transformers spontaneously discovering algebraic structure (here, the mod-10 cycle of digits) often after a long memorization plateau before a sudden breakthrough. Click any token to see its raw 64-dim embedding row.'),
      div({ class: styles.chartBox },
        svg({
          viewBox: `0 0 ${W} ${H}`,
          preserveAspectRatio: 'xMidYMid meet',
          class: 'block mx-auto h-auto',
          style: { maxWidth: W + 'px', width: '100%' }
        },
          points.map((p, i) => text({
            x: sx(p[0]), y: sy(p[1]) + 5,
            textAnchor: 'middle',
            fontSize: i < 10 ? '20' : '18',
            fontFamily: 'monospace',
            fontWeight: 'bold',
            fill: labelColor(i),
            stroke: i === sel ? themeMgr.theme.colors.accent.css : 'none',
            strokeWidth: i === sel ? '1' : '0',
            class: 'cursor-pointer',
            onClick: () => {
              this.selectedEmbToken = this.selectedEmbToken === i ? null : i
              this.update()
            }
          }, TOKEN_LABELS[i]))
        )
      ),
      sel != null && displayedRows[sel] ? div({ class: 'mt-3' },
        div({ class: [styles.body, 'mb-1'] },
          `Token "${TOKEN_LABELS[sel]}" — its ${D_MODEL}-dim row from tok_emb. These are the numbers PCA projects to the (x,y) point above. Click another token to switch, or click "${TOKEN_LABELS[sel]}" again to hide.`
        ),
        div({ class: 'flex gap-0' },
          stripCells(displayedRows[sel], embStripColor, STRIP_CELL_H, (i, v) => `dim ${i}: ${v.toFixed(3)}`)
        )
      ) : div(),
      div({ class: 'flex items-center gap-3 mt-2.5' },
        button({
          class: [styles.iconBtn, styles.iconBtnSmall],
          onClick: () => this.#toggleEmbeddingPlay(),
          title: this.embeddingPlaying ? 'Pause' : 'Play formation animation'
        }, this.embeddingPlaying ? iconPause() : iconPlay()),
        button({
          class: styles.btn,
          onClick: () => { this.embeddingScrubIdx = null; this.update() }
        }, isLive ? 'Live' : 'Jump to live'),
        div({ class: 'flex-1' },
          formField({
            target: this,
            prop: () => this.scrubValue,
            inputFn: inputRange,
            label: '',
            inputProps: {
              attrs: { min: 0, max: sliderMax, step: 1, style: { width: '100%' } }
            }
          })
        ),
        div({ class: [styles.monoLabelTiny, 'text-right'], style: { minWidth: '120px' } },
          `step ${displayStep.toLocaleString()} ${isLive ? '(live)' : `· ${this.embeddingScrubIdx! + 1}/${histLen}`}`
        )
      )
    )
  }

  #toggleEmbeddingPlay() {
    if (this.embeddingPlaying) { this.#stopEmbeddingPlay(); return }
    if (this.embeddingHistory.length < 2) return
    this.embeddingPlaying = true
    this.embeddingScrubIdx = 0
    this.#playTimer = (setInterval(() => {
      const len = this.embeddingHistory.length
      if (len === 0) { this.#stopEmbeddingPlay(); return }
      const cur = this.embeddingScrubIdx ?? len - 1
      const next = cur + 1
      if (next >= len) {
        this.embeddingScrubIdx = null
        this.#stopEmbeddingPlay()
        return
      }
      this.embeddingScrubIdx = next
      this.update()
    }, 90) as unknown) as number
    this.update()
  }

  #stopEmbeddingPlay() {
    if (this.#playTimer != null) { clearInterval(this.#playTimer); this.#playTimer = null }
    this.embeddingPlaying = false
    this.update()
  }

  #currentEmbeddingFlat(): Float32Array | null {
    if (this.embeddingScrubIdx != null && this.embeddingHistory[this.embeddingScrubIdx]) {
      return this.embeddingHistory[this.embeddingScrubIdx].tokEmb
    }
    return this.root.model.getTokEmbSnapshot()
  }

  #rowsFromFlat(flat: Float32Array): number[][] {
    const rows: number[][] = []
    for (let i = 0; i < VOCAB; i++) {
      const r: number[] = []
      for (let j = 0; j < D_MODEL; j++) r.push(flat[i * D_MODEL + j])
      rows.push(r)
    }
    return rows
  }
}

// ---------- Input panel ----------
class InputPanel extends Component implements IInputPanel {
  // Setters clamp to 0..99 and trigger a prediction refresh.
  _operandA = 27
  _operandB = 45
  get operandA() { return this._operandA }
  set operandA(v: number) { this._operandA = clamp(Math.floor(v), 0, 99); this.root.model.refreshPrediction() }
  get operandB() { return this._operandB }
  set operandB(v: number) { this._operandB = clamp(Math.floor(v), 0, 99); this.root.model.refreshPrediction() }

  get root() { return this.ctx.root as any as IRoot }
  get model() { return this.root.model }

  setPreset(a: number, b: number) {
    // Set storage directly to avoid two refreshes via the setters.
    this._operandA = clamp(Math.floor(a), 0, 99)
    this._operandB = clamp(Math.floor(b), 0, 99)
    this.model.refreshPrediction()
    this.update()
  }

  pickRandomPair(predicate: (a: number, b: number) => boolean, maxAttempts: number) {
    let a = 0, b = 0
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      a = Math.floor(Math.random() * 100); b = Math.floor(Math.random() * 100)
      if (predicate(a, b)) break
    }
    this.setPreset(a, b)
  }

  pickRandomTraining() { this.pickRandomPair((a, b) => !isTestPair(a, b), 200) }
  pickRandomHeldOut() { this.pickRandomPair(isTestPair, 5000) }

  pickRandomFailure() {
    if (this.model.recentFailures.length === 0) return
    const f = this.model.recentFailures[Math.floor(Math.random() * this.model.recentFailures.length)]
    this.setPreset(f.a, f.b)
  }

  view() {
    const trueDigits = splitResultDigits(this._operandA + this._operandB)
    const predDigits = this.model.predictedDigits
    const ready = predDigits.length === N_RESULT_DIGITS
    // Hide the hundreds slot for sums < 100.
    const showHundreds = ready && (predDigits[2] !== 0 || trueDigits[2] !== 0)
    const places = showHundreds ? [2, 1, 0] : [1, 0]
    const resultChildren = ready
      ? places.map(place => span({
          class: predDigits[place] === trueDigits[place] ? styles.successHigh : styles.successLow
        }, String(predDigits[place])))
      : '?'
    return div({ class: styles.panel },
      div({ class: styles.sectionTitle }, 'Try an addition'),
      div({ class: [styles.body, 'mb-2'] }, 'Type two numbers (0-99). Watch the model predict the sum, digit by digit.'),
      div({ class: styles.additionRow },
        formField({
          target: this,
          prop: () => this.operandA,
          inputFn: inputNumber,
          label: '',
          inputProps: {
            numberParams: { min: 0, max: 99, maximumFractionDigits: 0 },
            attrs: { class: styles.numberInput }
          }
        }),
        span({ class: styles.bigOp }, '+'),
        formField({
          target: this,
          prop: () => this.operandB,
          inputFn: inputNumber,
          label: '',
          inputProps: {
            numberParams: { min: 0, max: 99, maximumFractionDigits: 0 },
            attrs: { class: styles.numberInput }
          }
        }),
        span({ class: styles.bigOp }, '='),
        span({ class: styles.bigResult }, resultChildren)
      ),
      div({ class: [styles.body, 'mb-2'] },
        `True answer: ${this._operandA + this._operandB} ${isTestPair(this._operandA, this._operandB) ? '(held out from training)' : '(seen in training)'}`
      ),
      div({ class: styles.presetRow },
        button({ class: styles.btn, onClick: () => this.pickRandomTraining() }, 'Random in-training'),
        button({ class: styles.btn, onClick: () => this.pickRandomHeldOut() }, 'Random held-out'),
        button({ class: styles.btn, onClick: () => this.pickRandomFailure(), title: this.model.recentFailures.length === 0 ? 'No recent failures yet — train a bit first' : 'Pick a held-out pair the model is currently getting wrong' }, 'Random failure')
      ),
      div({ class: [styles.sectionTitle, 'mt-6'] }, 'Per-digit prediction'),
      this.digitProbsView()
    )
  }

  digitProbsView() {
    if (this.model.digitProbs.length === 0) return div({ class: styles.body }, 'No prediction yet.')
    const trueDigits = splitResultDigits(this._operandA + this._operandB)
    const labels = range(10).map(d => String(d))
    return div({ class: styles.digitProbs },
      this.model.digitProbs.map((probs, step) => {
        const placeName = RESULT_DIGIT_NAMES[step] ?? `pos${step}`
        const picked = this.model.predictedDigits[step]
        return div(
          probBarsView({
            probs,
            picked,
            truth: trueDigits[step],
            labels,
            maxBarH: 50
          }),
          div({ class: [styles.bodyText, 'font-bold text-center mt-0.5'] }, placeName)
        )
      })
    )
  }
}

// ---------- Wires panel ----------
// MLP drilldown for InsidePanel's selected cell: residual-in → hidden ReLU →
// residual-out strips, connected by the strongest mlp.up.W / mlp.down.W weights.
class WiresPanel extends Component {
  get root() { return this.ctx.root as any as IRoot }
  get parent() { return this.ctx.parent as any as IInsidePanel }

  view() {
    const sel = this.parent.effectiveSelection
    if (sel.layer === 0) {
      return div({ class: [styles.body, 'text-center my-6'] },
        'No MLP at the embedding stage — embeddings are just a lookup. Click any cell in the layer 1, 2, or 3 rows above to see that layer\'s MLP wires.'
      )
    }

    const mlpLayerIdx = sel.layer - 1
    const pos = sel.pos

    const weights = this.root.model.getMlpWeights(mlpLayerIdx)
    if (!weights) return div({ class: styles.body }, 'Initializing weights…')
    const { w1, w2, b2 } = weights

    const step = this.root.model.inside.at(-1) ?? null
    const hidden = step?.mlpHiddens[mlpLayerIdx]?.[pos] ?? null
    const afterResidual = step?.lattice[mlpLayerIdx + 1]?.[pos]?.residual ?? null

    // residualOut = mlp.down.W · hidden + b2  (MLP contribution to the residual stream)
    // residualIn  = afterResidual − residualOut
    let residualIn: Float32Array | null = null
    let residualOut: Float32Array | null = null
    if (hidden && afterResidual) {
      residualOut = new Float32Array(D_MODEL)
      for (let i = 0; i < D_MODEL; i++) {
        let s = b2[i]
        for (let j = 0; j < HIDDEN; j++) s += hidden[j] * w2[j * D_MODEL + i]
        residualOut[i] = s
      }
      residualIn = new Float32Array(D_MODEL)
      for (let i = 0; i < D_MODEL; i++) residualIn[i] = afterResidual[i] - residualOut[i]
    }

    const firingNote = hidden
      ? ` (${countFiring(hidden)}/${HIDDEN} firing — color = activation magnitude; never blue, ReLU forbids it)`
      : ''

    return div(
      div({ class: [styles.body, 'mb-1 text-center'] },
        `Inside layer ${mlpLayerIdx + 1}'s MLP at this position. ReLU on the hidden layer is where the transformer gets most of its non-linearity. The hidden strip is never blue — ReLU forbids negatives.`
      ),
      this.renderSvg(w1, w2, residualIn, hidden, residualOut)
    )
  }

  renderSvg(
    w1: Float32Array,
    w2: Float32Array,
    residualIn: ArrayLike<number> | null,
    hidden: ArrayLike<number> | null,
    residualOut: ArrayLike<number> | null
  ) {
    const W = 700, H = 222
    const PAD_X = 24
    const STRIP_X0 = PAD_X
    const STRIP_W = W - 2 * PAD_X
    const ROW_H = STRIP_CELL_H
    const Y_TOP = 20
    const STRIP_Y0 = 100
    const STRIP_Y1 = STRIP_Y0 + ROW_H
    const Y_BOT = 184

    const xEdge = (j: number, n: number) => STRIP_X0 + STRIP_W * j / n
    const xCenter = (j: number, n: number) => STRIP_X0 + STRIP_W * (j + 0.5) / n

    const TOP_N = 300

    const w1Top = pickTopWeights(w1, D_MODEL, HIDDEN, TOP_N)
    const w2Top = pickTopWeights(w2, HIDDEN, D_MODEL, TOP_N)
    // w*Top is sorted by descending |w|, so [0] is the matrix's global max.
    const w1Color = makeDivergingStripColor(Math.abs(w1Top[0]?.w ?? 1e-6))
    const w2Color = makeDivergingStripColor(Math.abs(w2Top[0]?.w ?? 1e-6))

    const cellBg = themeMgr.theme.colors.surface.rawValue
    const renderRow = (vals: ArrayLike<number> | null, n: number, y: number): VElement[] => {
      const color = vals ? makeDivergingStripColor(maxAbs(vals) || 1) : null
      const cellW = STRIP_W / n
      const cells: VElement[] = []
      for (let i = 0; i < n; i++) {
        cells.push(rect({
          // +0.5 prevents hairline gaps between cells at non-integer cellW.
          x: xEdge(i, n), y, width: cellW + 0.5, height: ROW_H,
          fill: color && vals ? color(vals[i]) : cellBg
        }))
      }
      return cells
    }

    const drawFan = (
      top: { i: number; j: number; w: number }[],
      srcN: number, srcY: number, dstN: number, dstY: number,
      color: (w: number) => string
    ): VElement[] => top.map(c => line({
      x1: xCenter(c.i, srcN), y1: srcY,
      x2: xCenter(c.j, dstN), y2: dstY,
      stroke: color(c.w), strokeWidth: 0.6
    }))

    const lines = [
      ...drawFan(w1Top, D_MODEL, Y_TOP + ROW_H, HIDDEN, STRIP_Y0, w1Color),
      ...drawFan(w2Top, HIDDEN, STRIP_Y1, D_MODEL, Y_BOT, w2Color)
    ]
    const topCells = renderRow(residualIn, D_MODEL, Y_TOP)
    const midCells = renderRow(hidden, HIDDEN, STRIP_Y0)
    const botCells = renderRow(residualOut, D_MODEL, Y_BOT)

    const labelColor = themeMgr.theme.colors.textMuted.css
    const textColor = themeMgr.theme.colors.text.css
    const stripCenterY = STRIP_Y0 + ROW_H / 2 + 4  // +4 ≈ baseline offset for 11px text
    const labels: VElement[] = [
      text({ x: PAD_X, y: 14, fill: labelColor, fontSize: '11', textAnchor: 'start' }, `Residual in (${D_MODEL} channels) — flows into MLP`),
      text({ x: W - PAD_X, y: 14, fill: labelColor, fontSize: '11', textAnchor: 'end' }, 'mlp.up.W wires below ↓'),
      text({ x: W / 2, y: stripCenterY, fill: textColor, fontSize: '11', textAnchor: 'middle', fontWeight: 'bold' as any }, `Hidden ReLU (${HIDDEN} neurons)`),
      text({ x: PAD_X, y: Y_BOT + ROW_H + 14, fill: labelColor, fontSize: '11', textAnchor: 'start' }, `MLP output (${D_MODEL} channels) — added into the residual stream`),
      text({ x: W - PAD_X, y: Y_BOT + ROW_H + 14, fill: labelColor, fontSize: '11', textAnchor: 'end' }, '↑ mlp.down.W wires above')
    ]

    return svg({
      viewBox: `0 0 ${W} ${H}`,
      width: '100%',
      class: 'block max-w-full',
      preserveAspectRatio: 'xMidYMid meet'
    }, ...lines, ...topCells, ...midCells, ...botCells, ...labels)
  }
}

// ---------- Explainer panel ----------
type ExplainerTopic = 'training' | 'architecture' | 'notes'

class ExplainerPanel extends Component {
  get root() { return this.ctx.root as any as IRoot }

  selectedTopic: ExplainerTopic = 'training'

  view() {
    const topics: { id: ExplainerTopic; label: string }[] = [
      { id: 'training', label: 'Training' },
      { id: 'architecture', label: 'Architecture' },
      { id: 'notes', label: 'Notes' }
    ]
    return div({ class: styles.explainerContainer },
      div({ class: 'flex gap-x-3 gap-y-0 mb-3 flex-wrap' },
        topics.map(t => button({
          class: [styles.subTabBtn, this.selectedTopic === t.id ? styles.subTabBtnActive : styles.subTabBtnInactive],
          onClick: () => { this.selectedTopic = t.id; this.update() }
        }, t.label))
      ),
      this.topicView(),
      this.liveTransformerDiagram()
    )
  }

  topicView() {
    switch (this.selectedTopic) {
      case 'training': return this.trainingView()
      case 'architecture': return this.architectureView()
      case 'notes': return this.notesView()
    }
  }

  trainingView() {
    return div(
      p('A small transformer is being trained from scratch in your browser to add 2-digit numbers. The vocabulary is just digits 0-9, "+", and "=". Each training example is one addition: "27+45=270". (The answer 72 is padded to 3 digits and reversed — the model generates the result one digit at a time, units first, so carries flow naturally left-to-right.)'),
      p('We hold out 20% of (a, b) pairs from training. Held-out accuracy measures real generalization: can the model add numbers it has never seen?'),
      p('At each training step, the model predicts the result digits for a mini-batch of additions, cross-entropy measures how wrong it was, and AdamW (Adam with decoupled weight decay, a widely-used optimizer) updates the parameters to be a bit less wrong next time. Loss is masked to the result-digit positions only — the model is graded on getting the answer right, not on predicting the operands.'),
      p('What backprop and the optimizer step actually do, up close: ',
        a({ href: 'https://typebulb.com/u/samples/xor-x-ray/full', target: '_blank' }, 'XOR X-ray'),
        ' — a 2-2-1 network where you click through forward, chain-rule backward, and the parameter update one phase at a time.')
    )
  }

  architectureView() {
    return div(
      p(`What flows through every channel in the diagram below is a ${D_MODEL}-dim vector — a single point in ${D_MODEL}-dimensional space, where the model encodes information as *directions*. At the bottom, the residual starts where token and position vectors merge at ⊕ (so the model sees order); ${D_MODEL} dimensions has enough room to keep them distinguishable downstream. For the foundations — what a tensor is, what these additions are doing here — see this `, a({ href: 'https://typebulb.com/u/samples/tensors/full', target: '_blank' }, 'interactive Tensors tutorial'), '.'),
      p(`The residual stream is the vertical channel: at every position, it runs upward through all ${N_LAYERS} layers. Every block in every layer reads from it and writes back into it at ⊕. In the diagram, it's the green vertical line at each position. At the top, the same token-embedding matrix transposed turns the final residual into a prediction.`),
      p('The K/V stream is the horizontal channel: at each layer, K and V at every position are made available to all later positions in that same layer. Only attention reads from it; each position\'s MLP reads only its own residual. In the diagram, the K/V bus is the horizontal line under each layer; each purple K/V circle writes to it, each attention block reads from it. Causal flow runs left-to-right, but the work is parallel — what RNNs do step by step, transformers do in one pass.')
    )
  }

  notesView() {
    return div(
      p('In small transformers, MLPs do roughly two-thirds of the FLOPs and supply all the element-wise nonlinearity; attention\'s nonlinearity is via softmax. Most of the rest is the attention heads, which serve as information routers. (In much larger LLMs with long sequences, attention\'s share grows.)'),
      p('See ', a({ href: 'https://tinyurl.com/44ayrzfp', target: '_blank' }, 'this transformer diagrammed by nn-dna'), ', which generates architecture diagrams from plain-English descriptions of neural networks.'),
      p('Co-built with Claude Opus 4.7; inspired by @repligate / j⧉nus\'s "How Information Flows Through Transformers".')
    )
  }

  // In-SVG so it travels with the diagram in screenshots; textLength+lengthAdjust
  // forces consistent inter-item gaps despite per-glyph width variance.
  diagramLegendSvg(W: number, H: number): VElement[] {
    const muted = themeMgr.theme.colors.textMuted.css
    const c = themeMgr.theme.colors
    const y = H - 10
    const swatchGap = 6, itemGap = 20, charW = 6, fontSize = '11'

    type Shape = { width: number; render: (fill: string, x: number) => VElement }
    const SHAPES: Record<string, Shape> = {
      circle: { width: 10, render: (f, x) => circle({ cx: x + 5, cy: y - 4, r: '5', fill: f, stroke: 'none' }) },
      square: { width: 10, render: (f, x) => rect({ x, y: y - 9, width: '10', height: '10', fill: f, stroke: 'none', rx: '1.5' }) },
      line:   { width: 20, render: (f, x) => rect({ x, y: y - 5, width: '20', height: '2', fill: f, stroke: 'none' }) }
    }

    const items: { kind: string; color: string; label: string }[] = [
      { kind: 'circle', color: c.accent.css,   label: 'K/V Computation' },
      { kind: 'square', color: c.attn.css,     label: 'Attention Computation' },
      { kind: 'square', color: c.mlp.css,      label: 'MLP Computation' },
      { kind: 'line',   color: c.primary.css,  label: 'Residual Stream' },
      { kind: 'line',   color: c.accent.css,   label: 'K/V Stream' }
    ]
    const labelW = (s: string) => s.length * charW
    const itemW = (it: typeof items[0]) => SHAPES[it.kind].width + swatchGap + labelW(it.label)
    const totalW = items.reduce((s, it) => s + itemW(it), 0) + itemGap * (items.length - 1)

    const out: VElement[] = []
    let x = (W - totalW) / 2
    for (const it of items) {
      const shape = SHAPES[it.kind]
      out.push(shape.render(it.color, x))
      out.push(text({
        x: x + shape.width + swatchGap, y,
        textAnchor: 'start', fontSize, fontFamily: 'sans-serif',
        fill: muted,
        textLength: String(labelW(it.label)),
        lengthAdjust: 'spacingAndGlyphs' as any
      }, it.label))
      x += itemW(it) + itemGap
    }
    return out
  }

  // Wiring follows j⧉nus's "How Information Flows Through Transformers" — per-node
  // K/V circle + attention block + MLP block, with a horizontal K/V bus per layer.
  liveTransformerDiagram() {
    const a = this.root.inputPanel.operandA
    const b = this.root.inputPanel.operandB
    const predicted = this.root.model.predictedDigits  // [ones, tens, hundreds] when ready
    const trueResultDigits = splitResultDigits(a + b)

    // Position 6 = predicted ones fed back, position 7 = predicted tens fed back.
    const prompt = promptTokens(a, b)
    const inputTok = [...prompt, predicted[0] ?? -1, predicted[1] ?? -1]
    const groundTruthNext = prompt.slice(1)
    const tokLabel = (t: number) => (t < 0 || t >= TOKEN_LABELS.length) ? '?' : TOKEN_LABELS[t]

    const N_POS = 8
    const N_ROWS = N_LAYERS + 2   // head + N_LAYERS layers + embed
    const cellW = 80
    const nodeH = 92        // layer-node height; tuned so the K/V→attn stub reads as a distinct segment
    const embedNodeH = 50   // embedding row: tok + pos lookups feed ⊕
    const headNodeH = 50    // head row: lnf + tied unembed projection (residual → vocab)
    const padL = 80, padR = 20, padT = 14, padB = 32  // stable frame; padB fits the in-SVG legend
    // Content shift inside the stable frame. Reducing padL would shrink W and
    // (with `w-full` sizing) make the whole SVG appear zoomed in; instead, we
    // keep W/H constant and offset content (and the legend, via <g>) here.
    const shiftX = -8
    const shiftY = -10
    const rowHeights = [headNodeH, nodeH, nodeH, nodeH, embedNodeH]
    const W = padL + cellW * N_POS + padR
    const H = padT + rowHeights.reduce((s, h) => s + h, 0) + padB
    const xAt = (p: number) => padL + shiftX + cellW * (p + 0.5)
    const yTop = (rowFromTop: number) =>
      padT + shiftY + rowHeights.slice(0, rowFromTop).reduce((s, h) => s + h, 0)
    // Head at top, embedding at bottom; layers in between.
    const rowHead = 0
    const rowEmbed = N_ROWS - 1
    const layerRowsTopDown = [1, 2, 3]
    const rowLabels = ['Unembed', 'Layer 3', 'Layer 2', 'Layer 1', 'Embed']

    const residualColor = themeMgr.theme.colors.primary.css
    const errorColor = themeMgr.theme.colors.error.css
    const kvColor = themeMgr.theme.colors.accent.css
    const attnColor = themeMgr.theme.colors.attn.css
    const mlpColor = themeMgr.theme.colors.mlp.css
    const mutedCol = themeMgr.theme.colors.textMuted.css
    const textCol = themeMgr.theme.colors.text.css
    const borderCol = themeMgr.theme.colors.border.css
    const nodeFill = themeMgr.theme.colors.surface.css
    const bgColor = themeMgr.theme.colors.background.css

    const elements: VElement[] = []

    // ===== Helpers =====
    // All draw helpers push into `elements`.

    // Marching dashes. Caller must orient the path so draw direction == flow
    // direction (the residual is drawn bottom-to-top because flow is upward).
    // 4 solid / 2 gap, 6 px/sec — short cycle so stubs (~6px) show a full dash.
    const flowPath = (d: string, color: string, strokeWidth: number = 1.5) => {
      elements.push(path({
        d, stroke: color, strokeWidth: String(strokeWidth), fill: 'none',
        strokeDashArray: '4 2'
      }, animate({
        attributeName: 'stroke-dashoffset',
        from: '0', to: '-6',
        dur: '1s',
        repeatCount: 'indefinite'
      })))
    }

    // Triangle arrowhead at (x, y) pointing in `dir`. Axial = wing offset along
    // the arrow axis; perp = wing spread perpendicular to it.
    const arrowhead = (
      x: number, y: number, dir: 'up' | 'down' | 'left' | 'right',
      color: string, axial = 3.5, perp = 2.5,
    ) => {
      const d =
        dir === 'right' ? `M ${x - axial} ${y - perp} L ${x} ${y} L ${x - axial} ${y + perp} Z` :
        dir === 'left'  ? `M ${x + axial} ${y - perp} L ${x} ${y} L ${x + axial} ${y + perp} Z` :
        dir === 'up'    ? `M ${x - perp} ${y + axial} L ${x} ${y} L ${x + perp} ${y + axial} Z` :
                          `M ${x - perp} ${y - axial} L ${x} ${y} L ${x + perp} ${y - axial} Z`
      elements.push(path({ d, fill: color, stroke: 'none' }))
    }

    // Flow line ending in an arrowhead at (x2, y2). Horizontal or vertical only.
    const drawArrow = (x1: number, y1: number, x2: number, y2: number, color: string) => {
      flowPath(`M ${x1} ${y1} L ${x2} ${y2}`, color)
      const dir = y1 === y2 ? (x2 > x1 ? 'right' : 'left') : (y2 > y1 ? 'down' : 'up')
      arrowhead(x2, y2, dir, color)
    }

    // Filled with bg so the residual line is masked inside the circle — otherwise
    // the vertical residual stroke overlaps the cross arm and ⊕ reads as ⊖.
    const drawMerge = (mcx: number, mcy: number) => {
      elements.push(circle({
        cx: mcx, cy: mcy, r: '4',
        fill: bgColor, stroke: residualColor, strokeWidth: '1.5'
      }))
      elements.push(path({
        d: `M ${mcx - 2.5} ${mcy} L ${mcx + 2.5} ${mcy} M ${mcx} ${mcy - 2.5} L ${mcx} ${mcy + 2.5}`,
        stroke: residualColor, strokeWidth: '1.2', fill: 'none'
      }))
    }

    const drawBlock = (cx: number, nodeY: number, topY: number, color: string) => {
      elements.push(rect({
        x: blockX(cx), y: nodeY + topY, width: String(blockW), height: String(blockH),
        fill: color, stroke: 'none', rx: '2'
      }))
    }

    // ===== Drawing =====

    // Residual: drawn bottom-to-top so dashes march upward (flow direction).
    // Starts at the ⊕ inside the embed row (tok + pos merge); ends inside the
    // head row entering the unembed box from below (which then displays the
    // model's prediction at that position).
    const embedMergeY = 7      // ⊕ y-offset within embed row
    const headArrowY = 37      // arrowhead tip y-offset within head row (box bottom)
    for (let p = 0; p < N_POS; p++) {
      flowPath(
        `M ${xAt(p)} ${yTop(rowEmbed) + embedMergeY} L ${xAt(p)} ${yTop(rowHead) + headArrowY + 6}`,
        residualColor, 2
      )
      arrowhead(xAt(p), yTop(rowHead) + headArrowY, 'up', residualColor, 6, 4)
    }

    // Each layer node = attn block (bottom) + MLP block (top), each with a pre-LN
    // side trip: residual splits, branch enters block, block output ⊕ skip = new
    // residual. K/V circle sits in the attn side trip on the K/V bus.
    const blockW = 18
    const blockH = 12
    const blockX = (cx: number) => cx + 20
    const blockCx = (cx: number) => cx + 29  // K/V circle x

    // Per-node vertical layout (y from nodeY = top of row). All other per-node
    // y-coordinates derive from these.
    const layerY = {
      mlpTop: 14,        // ⊕ MLP merge / MLP block top edge
      mlpBottom: 26,     // MLP block bottom = MLP-block split point
      attnTop: 34,       // ⊕ attn merge / attn block top edge
      attnBottom: 46,    // attn block bottom = top of K/V→attn stub
      kvStubBottom: 58,  // bottom of K/V→attn stub
      kvCenter: 66,      // K/V circle center / K/V bus y
      kvBottom: 71,      // K/V circle bottom (kvCenter + r=5)
      attnSplit: 84,     // residual → K/V branch starts below the bus
    }
    const busOffsetY = layerY.kvCenter

    // Bus runs only between K/V_0 and K/V_{N-1} — pos 0 has no left bus neighbor,
    // pos N-1 has no right one. Single mid-row arrowhead shows causal direction.
    for (const rowL of layerRowsTopDown) {
      const busY = yTop(rowL) + busOffsetY
      const bcxFirst = blockCx(xAt(0))
      const bcxLast = blockCx(xAt(N_POS - 1))
      flowPath(
        `M ${bcxFirst} ${busY} L ${bcxLast} ${busY}`,
        kvColor, 2
      )
      // Arrow placed mid-bus (not at endpoint) so it doesn't suggest continuation.
      const arrowTip = (blockCx(xAt(3)) + blockCx(xAt(4))) / 2
      arrowhead(arrowTip, busY, 'right', kvColor, 10, 5)
    }

    // Per-node internals (layer rows only). Two non-obvious geometric facts:
    //   - Residual at cx visibly crosses the bus at kvCenter but doesn't
    //     interact — only the K/V circle (at bcx) is on the bus.
    //   - attnSplit is BELOW the bus so the Manhattan branch up to K/V doesn't
    //     overlap the prior K/V circle's bus segment.
    for (const rowL of layerRowsTopDown) {
      const nodeY = yTop(rowL)
      for (let p = 0; p < N_POS; p++) {
        const cx = xAt(p)
        const bcx = blockCx(cx)

        drawBlock(cx, nodeY, layerY.mlpTop, mlpColor)
        // Block arrows inset 2px from the corners so they don't appear to come from the corner points.
        drawArrow(cx, nodeY + layerY.mlpBottom - 2, blockX(cx), nodeY + layerY.mlpBottom - 2, residualColor)
        drawArrow(blockX(cx), nodeY + layerY.mlpTop + 2, cx + 4, nodeY + layerY.mlpTop + 2, residualColor)
        drawMerge(cx, nodeY + layerY.mlpTop + 2)

        drawBlock(cx, nodeY, layerY.attnTop, attnColor)
        // Attn-input Manhattan branch routes BELOW the bus to avoid overlapping
        // the prior position's bus segment.
        flowPath(
          `M ${cx} ${nodeY + layerY.attnSplit} L ${bcx} ${nodeY + layerY.attnSplit} L ${bcx} ${nodeY + layerY.kvBottom}`,
          residualColor
        )
        arrowhead(bcx, nodeY + layerY.kvBottom, 'up', residualColor)
        drawArrow(bcx, nodeY + layerY.kvStubBottom, bcx, nodeY + layerY.attnBottom, kvColor)
        drawArrow(blockX(cx), nodeY + layerY.attnTop + 2, cx + 4, nodeY + layerY.attnTop + 2, residualColor)
        drawMerge(cx, nodeY + layerY.attnTop + 2)
        // K/V circle drawn last so it sits on top of the arrow tip + bus line.
        elements.push(circle({
          cx: bcx, cy: nodeY + layerY.kvCenter, r: '5',
          fill: kvColor, stroke: 'none'
        }))
      }
    }

    // Boxes for embed (tok + pos) and unembed: same size, hold actual values.
    // Stroke is `border` (low-contrast palette tone) so the cell outlines don't
    // outshine the digits inside — including green digits on correct predictions.
    const lookupW = 24
    const lookupH = 22
    const lookupStroke = borderCol
    const lookupYHead = 15
    // Embed boxes shifted down so the arrow's vertical leg equals its
    // horizontal leg (perfectly square turn into ⊕).
    const lookupYEmbed = embedMergeY + 10  // boxCx (14) − ⊕ radius (4)

    // Embed row: tok and pos lookups (off-axis) feed ⊕ on the residual line.
    // The ⊕ is where positional info enters the model. Positions 6/7 hold the
    // model's prior ones/tens prediction fed back (colored by correctness).
    for (let p = 0; p < N_POS; p++) {
      const nodeY = yTop(rowEmbed)
      const cx = xAt(p)
      const mergeY = nodeY + embedMergeY
      const boxY = nodeY + lookupYEmbed
      const tok = inputTok[p]
      const isResultInput = p >= 6
      const resultInputWrong = isResultInput && tok >= 0 && tok !== trueResultDigits[p - 6]
      const tokColor =
        !isResultInput ? textCol :
        tok < 0 ? mutedCol :
        resultInputWrong ? errorColor :
        residualColor
      // tok box (left of center)
      elements.push(rect({
        x: cx - 26, y: boxY, width: String(lookupW), height: String(lookupH),
        fill: nodeFill, stroke: lookupStroke, strokeWidth: '1.5', rx: '2'
      }))
      elements.push(text({
        x: cx - 14, y: boxY + 16,
        textAnchor: 'middle', fontSize: '15', fontFamily: 'monospace', fontWeight: 'bold',
        fill: tokColor
      }, tokLabel(tok)))
      // pos box (right of center)
      elements.push(rect({
        x: cx + 2, y: boxY, width: String(lookupW), height: String(lookupH),
        fill: nodeFill, stroke: lookupStroke, strokeWidth: '1.5', rx: '2'
      }))
      elements.push(text({
        x: cx + 14, y: boxY + 16,
        textAnchor: 'middle', fontSize: '15', fontFamily: 'monospace',
        fill: mutedCol
      }, String(p)))
      // Branches up to ⊕. Arrowheads end at the circle edge (r=4).
      flowPath(`M ${cx - 14} ${boxY} L ${cx - 14} ${mergeY} L ${cx - 4} ${mergeY}`, residualColor)
      arrowhead(cx - 4, mergeY, 'right', residualColor)
      flowPath(`M ${cx + 14} ${boxY} L ${cx + 14} ${mergeY} L ${cx + 4} ${mergeY}`, residualColor)
      arrowhead(cx + 4, mergeY, 'left', residualColor)
      drawMerge(cx, mergeY)
    }

    // Head row: residual is consumed by the unembed projection (matmul with
    // tok_emb^T, preceded by final LN). Box shows the model's prediction at
    // each position — positions 5/6/7 are the ones/tens/hundreds result
    // digits (green/red by correctness); 0–4 are untrained next-operand
    // guesses, shown muted.
    for (let p = 0; p < N_POS; p++) {
      const nodeY = yTop(rowHead)
      const cx = xAt(p)
      const boxY = nodeY + lookupYHead
      let label = '?'
      let isPred = false
      let isWrong = false
      if (p < 5) {
        label = tokLabel(groundTruthNext[p])
      } else {
        const idx = p - 5
        const t = predicted[idx]
        if (t !== undefined) {
          label = tokLabel(t)
          isPred = true
          isWrong = t !== trueResultDigits[idx]
        }
      }
      const fillColor = isPred ? (isWrong ? errorColor : residualColor) : mutedCol
      elements.push(rect({
        x: cx - 12, y: boxY, width: String(lookupW), height: String(lookupH),
        fill: nodeFill, stroke: lookupStroke, strokeWidth: '1.5', rx: '2'
      }))
      elements.push(text({
        x: cx, y: boxY + 16,
        textAnchor: 'middle', fontSize: '15', fontFamily: 'monospace', fontWeight: 'bold',
        fill: fillColor
      }, label))
    }

    for (let i = 0; i < N_ROWS; i++) {
      // Embed row's boxes are shifted down 2px (square-arrow alignment);
      // keep the row label tracking with them.
      const extra = i === rowEmbed ? (lookupYEmbed - lookupYHead) : 0
      elements.push(text({
        x: padL + shiftX - 4, y: yTop(i) + rowHeights[i] / 2 + 4 + extra,
        textAnchor: 'end', fontSize: '11', fontFamily: 'monospace',
        fill: mutedCol
      }, rowLabels[i]))
    }

    elements.push(g({ transform: `translate(${shiftX} ${shiftY})` }, this.diagramLegendSvg(W, H)))

    return svg({
      viewBox: `0 0 ${W} ${H}`,
      preserveAspectRatio: 'xMidYMid meet',
      class: styles.chartBox
    }, elements)
  }
}

// ---------- Root ----------
class Root extends Component implements IRoot {
  activeTab: 'explainer' | 'attention' | 'inside' | 'embeddings' = 'explainer'

  // Child components must be public fields — domeleon enumerates them; # fields are invisible.
  model = new Model()
  inputPanel = new InputPanel()
  attentionPanel = new AttentionPanel()
  insidePanel = new InsidePanel()
  tokenEmbeddingsPanel = new TokenEmbeddingsPanel()
  explainerPanel = new ExplainerPanel()

  #layout: 'wide' | 'narrow' = window.innerWidth >= 900 ? 'wide' : 'narrow'
  get isNarrow() { return this.#layout === 'narrow' }

  constructor() {
    super()
    new MutationObserver(() => {
      const t = document.documentElement.getAttribute('data-theme')
      if ((t === 'light' || t === 'dark') && t !== themeMgr.themeName) {
        themeMgr.themeName = t
        this.update()
      }
    }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    window.addEventListener('resize', () => {
      this.#layout = window.innerWidth >= 900 ? 'wide' : 'narrow'
      this.update()
    })
  }

  reset() {
    this.tokenEmbeddingsPanel.reset()
    this.model.reset()
  }

  view() {
    if (!this.model.isReady) {
      return div({ class: styles.waitContainer },
        div({ class: styles.waitText }, this.model.status),
        div({ class: styles.waitSpinner }, '⏳')
      )
    }
    return div({ class: styles.layout },
      div({ class: styles.header },
        div({ class: styles.headerContainer },
          h1({ class: styles.headerTitle }, 'Transformer learns addition'),
          p({ class: [styles.body, 'mt-0 mb-5'] }, `A small transformer (${N_LAYERS} layers, ${N_HEADS} heads, ~${Math.round(this.model.paramCount / 1000)}K params) is being trained from scratch in your browser to add 2-digit numbers.`)
        )
      ),
      div({ class: styles.controlStripBar },
        div({ class: styles.controlStripContent },
          this.controlsContent()
        )
      ),
      div({ class: styles.content },
        div({ class: styles.mainContainer },
          this.mainContent()
        )
      )
    )
  }

  controlsContent() {
    const stat = (label: string, body: VElement) =>
      div({ class: styles.statCol },
        div({ class: [styles.labelText, 'whitespace-nowrap'] }, label),
        body
      )
    const valueText = (s: string) => div({ class: styles.valueText }, s)
    const withSpark = (body: VElement, spark: VElement) =>
      div({ class: styles.statValueRow }, body, spark)
    return div({ class: styles.controlsRow },
      div({ class: styles.statsRow },
        div({ class: styles.controlGroup },
          button({
            onClick: () => this.model.toggleRun(),
            class: [styles.iconBtn, styles.iconBtnPlay]
          }, this.model.isRunning ? iconPause() : iconPlay()),
          button({
            onClick: () => this.reset(),
            class: [styles.iconBtn, styles.iconBtnSmall],
            title: 'Reset model'
          }, iconReset())
        ),
        stat('Steps', valueText(this.model.trainStep.toLocaleString())),
        stat('Examples', valueText(this.model.examplesSeen.toLocaleString())),
        stat('Ex/s', valueText(this.model.examplesPerSec.toFixed(0))),
        stat('Batch', valueText(String(BATCH_SIZE))),
        div({ class: 'flex items-center gap-x-3 sm:gap-x-8' },
          stat('Masked loss', withSpark(
            valueText(this.model.avgLoss.toFixed(3)),
            this.sparkline(this.model.lossHistory, themeMgr.theme.colors.text.rawValue, 'Loss', Math.log(VOCAB))
          )),
          stat('Held-out acc', withSpark(
            div({ class: styles.valueText },
              span({ class: this.model.testAcc > 0.9 ? styles.successHigh : styles.successLow }, (this.model.testAcc * 100).toFixed(0) + '%')
            ),
            this.sparkline(this.model.accHistory, themeMgr.theme.colors.primary.rawValue, 'Held-out acc', 1)
          ))
        )
      )
    )
  }

  mainContent() {
    const gridTemplateColumns = this.isNarrow ? '1fr' : 'minmax(360px, 1fr) 2fr'
    return div({ class: styles.mainGrid, style: { gridTemplateColumns } }, this.inputPanel.view(), this.tabsPanel())
  }

  tabsPanel() {
    const tabs: { id: 'explainer' | 'attention' | 'inside' | 'embeddings'; label: string }[] = [
      { id: 'explainer', label: 'Explainer' },
      { id: 'attention', label: 'Attention' },
      { id: 'inside', label: 'Blocks' },
      { id: 'embeddings', label: 'Token embeddings' }
    ]
    return div({ class: styles.panel },
      div({ class: 'flex gap-1.5 mb-3 flex-wrap' },
        tabs.map(t => button({
          class: [styles.tabBtn, this.activeTab === t.id ? styles.tabBtnActive : styles.tabBtnInactive],
          onClick: () => { this.activeTab = t.id; this.update() }
        }, t.label))
      ),
      this.activeTab === 'attention' ? this.attentionPanel.view() :
      this.activeTab === 'inside' ? this.insidePanel.view() :
      this.activeTab === 'embeddings' ? this.tokenEmbeddingsPanel.view() :
      this.explainerPanel.view()
    )
  }

  sparkline(data: { step: number; value: number }[], color: string, label: string, maxV: number) {
    const W = 80, H = 14, pad = 1
    if (data.length < 2) {
      return svg({ viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', class: styles.sparklineBox })
    }
    const minS = data[0].step
    const maxS = data[data.length - 1].step
    const sx = (s: number) => (s - minS) / Math.max(1, maxS - minS) * W
    const sy = (v: number) => (H - pad) - Math.max(0, Math.min(1, v / maxV)) * (H - 2 * pad)
    const d = data.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.step).toFixed(1)},${sy(p.value).toFixed(1)}`).join(' ')
    return svg({
      viewBox: `0 0 ${W} ${H}`,
      preserveAspectRatio: 'none',
      class: styles.sparklineBox,
      title: `${label} · y: 0–${maxV.toFixed(2)} · steps ${minS.toLocaleString()}–${maxS.toLocaleString()}`
    },
      path({ d, stroke: color, strokeWidth: '1.5', fill: 'none' })
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
    "tensorgrad": "^0.1.8"
  },
  "description": "Watch a transformer learn 2-digit addition from scratch in your browser. Type two numbers and see it predict the sum digit by digit. Built with tensorgrad (autograd + WebGPU)."
}
```