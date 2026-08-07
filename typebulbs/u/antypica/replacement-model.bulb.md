---
format: typebulb/v1
name: "Replacement Model"
---

**code.tsx**

```tsx
import { App, Component, div, h1, h3, p, span, button, input, svg, polyline, line, g, rect, text } from "domeleon"
import {
  Module, Linear, RMSNorm, compile, lr,
  add, mul, sub, sum, div as tdiv, matmul, swapAxes, reshape, embedding,
  splitHeads, mergeHeads, rope, softmaxCausal, categorical, relu,
  abs, mean, square, capture,
  crossEntropy, checkWebGPU, init,
  type Tensor,
} from "tensorgrad"

// Rung three. The Pokéformer SAE bulb re-encodes the MESSAGES between
// blocks (autoencoders on the residual stream; all machinery intact).
// This bulb replaces MACHINERY: each block's MLP is removed from the
// spliced model and a sparse transcoder, trained to imitate that MLP's
// input-to-output behavior, is seated in its socket. Attention and norms
// stay original (a per-letter module cannot imitate attention, which moves
// information between letters).

// === Transformer hyperparams (same architecture as the Pokéformer bulb) ===
const V = 27               // vocab: a–z + EOS
const EOS = 26
const T = 16               // 1 BOS + up-to-14 chars + EOS
const D = 64               // model dim
const N_LAYERS = 2
const N_HEADS = 4
const D_HEAD = D / N_HEADS
const FFN = 4 * D
const B_TRAIN = 32
const B_INFER = 1
const LM_STEPS = 1500
const LR = lr.linear({ peak: 0.005, final: 0.0005, steps: LM_STEPS })
const SCALE_QK = 1 / Math.sqrt(D_HEAD)

// === Transcoder hyperparams ===
// One transcoder per block, each trained on (MLP input, MLP output) pairs
// from the trained model. Same sparse architecture as the sibling bulb's
// SAEs (F = 8x overcomplete, L1 on the hidden), different objective:
// predict the output, not reconstruct the input.
const F = 512
const TC_B = 256           // rows per transcoder training batch
const TC_STEPS = 2500
const TC_LR = 1e-3
const DEFAULT_L1 = 2e-3    // λ; lower than the SAE bulb: imitation fidelity
                           // compounds letter-by-letter at greedy, so it is
                           // worth buying (L0 rises from ~20 toward ~35)
const B_EVAL = 32          // harvest / CE-eval batch (names per batch)
const N_PAIRS = 8          // aligned prompt-completion pairs per batch

// === Corpus (same as the Pokéformer bulb) ===

const FALLBACK_NAMES: readonly string[] = [
  "pikachu", "charizard", "bulbasaur", "squirtle", "mewtwo", "mew",
  "articuno", "zapdos", "moltres", "lugia", "hooh", "rayquaza",
  "kyogre", "groudon", "dialga", "palkia", "giratina", "arceus",
  "eevee", "vaporeon", "jolteon", "flareon", "espeon", "umbreon",
  "leafeon", "glaceon", "sylveon", "snorlax", "gengar", "alakazam",
  "machamp", "lucario", "garchomp", "dragonite", "tyranitar", "metagross",
  "salamence", "blaziken", "swampert", "sceptile",
]

function cleanCorpus(raw: readonly string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const name of raw) {
    const cleaned = name.toLowerCase().normalize("NFD").replace(/[^a-z]/g, "")
    if (cleaned.length < 2 || cleaned.length > T - 2) continue
    if (seen.has(cleaned)) continue
    seen.add(cleaned)
    out.push(cleaned)
  }
  return out
}

let NAMES: string[] = cleanCorpus(FALLBACK_NAMES)
let NAMES_SET: Set<string> = new Set(NAMES)
let corpusSource: "pokeapi" | "fallback" = "fallback"

async function loadCorpus(): Promise<void> {
  try {
    const r = await fetch("https://pokeapi.co/api/v2/pokemon-species?limit=1025")
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    const data = await r.json() as { results: { name: string }[] }
    const cleaned = cleanCorpus(data.results.map(x => x.name))
    if (cleaned.length < 100) throw new Error(`only ${cleaned.length} usable names`)
    NAMES = cleaned
    NAMES_SET = new Set(NAMES)
    corpusSource = "pokeapi"
  } catch (e) {
    console.warn("PokéAPI fetch failed, using fallback corpus:", e)
  }
}

// === Transformer model ===

class Attention extends Module {
  q = new Linear(D, D, { bias: false })
  k = new Linear(D, D, { bias: false })
  v = new Linear(D, D, { bias: false })
  o = new Linear(D, D, { bias: false })
}

class MLP extends Module {
  up   = new Linear(D, FFN)
  down = new Linear(FFN, D)
}

class Block extends Module {
  n1   = new RMSNorm(D)
  attn = new Attention()
  n2   = new RMSNorm(D)
  mlp  = new MLP()
}

class NameLM extends Module {
  tok_emb: Tensor
  layers: Block[]
  nf: RMSNorm
  constructor() {
    super()
    this.tok_emb = this.param([V, D])
    this.layers = []
    for (let i = 0; i < N_LAYERS; i++) this.layers.push(new Block())
    this.nf = new RMSNorm(D)
  }
}

function attnFwd(p: Attention, x: Tensor): Tensor {
  const q = splitHeads(p.q.fwd(x), N_HEADS)
  const k = splitHeads(p.k.fwd(x), N_HEADS)
  const v = splitHeads(p.v.fwd(x), N_HEADS)
  const [qr, kr] = rope(q, k)
  const scores = mul(matmul(qr, swapAxes(kr, -1, -2)), SCALE_QK)
  return p.o.fwd(mergeHeads(matmul(softmaxCausal(scores), v)))
}

function mlpFwd(p: MLP, x: Tensor): Tensor {
  return p.down.fwd(relu(p.up.fwd(x)))
}

function blockFwd(p: Block, x: Tensor): Tensor {
  const a = attnFwd(p.attn, p.n1.fwd(x))
  const x1 = add(x, a)
  return add(x1, mlpFwd(p.mlp, p.n2.fwd(x1)))
}

function headOut(m: NameLM, x: Tensor): Tensor {
  return matmul(m.nf.fwd(x), swapAxes(m.tok_emb, -1, -2))   // tied LM head
}

// Plain forward: all blocks intact.
function logitsFwd(m: NameLM, tokens: Tensor): Tensor {
  let x = embedding(m.tok_emb, tokens)
  for (const block of m.layers) x = blockFwd(block, x)
  return headOut(m, x)
}

// A transcoder standing in for one MLP, weights fed as plain input tensors
// (frozen idiom). `si` / `so` are the input / output RMS scales its
// training data was normalized by.
type TCW = { We: Tensor; be: Tensor; Wd: Tensor; bd: Tensor; si: Tensor; so: Tensor }
function tcFwd(w: TCW, x: Tensor): Tensor {
  const xn = tdiv(x, w.si)
  const h = relu(add(matmul(xn, w.We), w.be))
  const y = add(matmul(h, w.Wd), w.bd)
  return mul(y, w.so)
}

// The machinery swap: each block runs with its ORIGINAL attention but its
// MLP replaced by the transcoder. The original MLP weights never execute
// in this forward.
function splicedLogitsFwd(m: NameLM, tokens: Tensor, ws: TCW[]): Tensor {
  let x = embedding(m.tok_emb, tokens)
  for (let i = 0; i < N_LAYERS; i++) {
    const b = m.layers[i]!
    const x1 = add(x, attnFwd(b.attn, b.n1.fwd(x)))
    x = add(x1, tcFwd(ws[i]!, b.n2.fwd(x1)))
  }
  return headOut(m, x)
}

// Flat inputs record (attach inputs are flat tensors) → per-layer bundles.
type SpliceInputs = {
  We0: Tensor; be0: Tensor; Wd0: Tensor; bd0: Tensor; si0: Tensor; so0: Tensor
  We1: Tensor; be1: Tensor; Wd1: Tensor; bd1: Tensor; si1: Tensor; so1: Tensor
}
function bundles(i: SpliceInputs): TCW[] {
  return [
    { We: i.We0, be: i.be0, Wd: i.Wd0, bd: i.bd0, si: i.si0, so: i.so0 },
    { We: i.We1, be: i.be1, Wd: i.Wd1, bd: i.bd1, si: i.si1, so: i.so1 },
  ]
}

// Masked per-valid-token CE (mask pre-normalized to sum to 1).
function maskedCE(logits: Tensor, targets: Tensor, mask: Tensor): Tensor {
  const ce = crossEntropy(logits, targets, { reduction: "none" })
  return sum(mul(ce, mask))
}

function lossFn(m: NameLM, { tokens, targets, mask }: { tokens: Tensor; targets: Tensor; mask: Tensor }): Tensor {
  return maskedCE(logitsFwd(m, tokens), targets, mask)
}

// Sampling forwards: in-graph categorical with runtime temperature.
function sampleOrigFn(m: NameLM, { tokens, temperature }: { tokens: Tensor; temperature: Tensor }): Tensor {
  return categorical(tdiv(logitsFwd(m, tokens), temperature))
}
function sampleSplicedFn(m: NameLM, i: { tokens: Tensor; temperature: Tensor } & SpliceInputs): Tensor {
  return categorical(tdiv(splicedLogitsFwd(m, i.tokens, bundles(i)), i.temperature))
}

// Harvest forward: each MLP's input and output, from the intact model.
// Output is block 1's MLP output; the other three ride along as captures.
function harvestFn(m: NameLM, { tokens }: { tokens: Tensor }): Tensor {
  let x = embedding(m.tok_emb, tokens)
  const b0 = m.layers[0]!
  const x1a = add(x, attnFwd(b0.attn, b0.n1.fwd(x)))
  const in0 = capture("in0", b0.n2.fwd(x1a))
  const out0 = capture("out0", mlpFwd(b0.mlp, in0))
  x = add(x1a, out0)
  const b1 = m.layers[1]!
  const x1b = add(x, attnFwd(b1.attn, b1.n1.fwd(x)))
  const in1 = capture("in1", b1.n2.fwd(x1b))
  return mlpFwd(b1.mlp, in1)
}

// CE-eval forwards.
function evalOrigFn(m: NameLM, i: { tokens: Tensor; targets: Tensor; mask: Tensor }): Tensor {
  return maskedCE(logitsFwd(m, i.tokens), i.targets, i.mask)
}
function evalSplicedFn(m: NameLM, i: { tokens: Tensor; targets: Tensor; mask: Tensor } & SpliceInputs): Tensor {
  return maskedCE(splicedLogitsFwd(m, i.tokens, bundles(i)), i.targets, i.mask)
}

// === Transcoder model ===

class Transcoder extends Module {
  We: Tensor; be: Tensor; Wd: Tensor; bd: Tensor
  constructor() {
    super()
    this.We = this.param([D, F])
    this.be = this.param([F], { init: init.zeros() })
    this.Wd = this.param([F, D])
    this.bd = this.param([D], { init: init.zeros() })
  }
}

// loss = MSE(prediction, true MLP output) + λ · mean-over-rows
// sum-over-features |h|. Both x and y arrive pre-normalized to unit RMS.
function tcLossFn(m: Transcoder, { x, y, l1 }: { x: Tensor; y: Tensor; l1: Tensor }): Tensor {
  const h = capture("h", relu(add(matmul(x, m.We), m.be)))
  const pred = add(matmul(h, m.Wd), m.bd)
  const mse = mean(square(sub(pred, y)))
  const sparsity = mul(mean(abs(h)), F)
  return add(mse, mul(sum(l1), sparsity))
}

// === Data prep ===

const charToId = (ch: string) => ch.charCodeAt(0) - 97
const idToChar = (id: number) => String.fromCharCode(97 + id)

function encodeName(name: string): { tokens: number[]; targets: number[]; validCount: number } {
  const seq: number[] = [EOS]
  for (const ch of name) seq.push(charToId(ch))
  while (seq.length < T + 1) seq.push(EOS)
  return {
    tokens:  seq.slice(0, T),
    targets: seq.slice(1, T + 1),
    validCount: Math.min(name.length + 1, T),
  }
}

function makeTrainingBatch(): { tokens: Int32Array; targets: Int32Array; mask: Float32Array } {
  const tokens  = new Int32Array(B_TRAIN * T)
  const targets = new Int32Array(B_TRAIN * T)
  const mask    = new Float32Array(B_TRAIN * T)
  let totalValid = 0
  for (let i = 0; i < B_TRAIN; i++) {
    const name = NAMES[Math.floor(Math.random() * NAMES.length)]!
    const ex = encodeName(name)
    for (let t = 0; t < T; t++) {
      tokens[i * T + t]  = ex.tokens[t]!
      targets[i * T + t] = ex.targets[t]!
      if (t < ex.validCount) { mask[i * T + t] = 1; totalValid++ }
    }
  }
  const inv = 1 / totalValid
  for (let i = 0; i < mask.length; i++) mask[i] = mask[i]! * inv
  return { tokens, targets, mask }
}

function makeFixedBatch(names: string[], from: number): { tokens: Int32Array; targets: Int32Array; mask: Float32Array; valid: number[] } {
  const tokens  = new Int32Array(B_EVAL * T)
  const targets = new Int32Array(B_EVAL * T)
  const mask    = new Float32Array(B_EVAL * T)
  const valid: number[] = []
  let totalValid = 0
  for (let i = 0; i < B_EVAL; i++) {
    const name = names[(from + i) % names.length]!
    const ex = encodeName(name)
    valid.push(ex.validCount)
    for (let t = 0; t < T; t++) {
      tokens[i * T + t]  = ex.tokens[t]!
      targets[i * T + t] = ex.targets[t]!
      if (t < ex.validCount) { mask[i * T + t] = 1; totalValid++ }
    }
  }
  const inv = 1 / totalValid
  for (let i = 0; i < mask.length; i++) mask[i] = mask[i]! * inv
  return { tokens, targets, mask, valid }
}

// === Pipeline state ===

type Phase = "boot" | "train-lm" | "harvest" | "train-tc" | "explore" | "no-webgpu" | "error"

// One aligned comparison: both models complete the same stem. `id` is the
// row's slot, so a refresh reuses the same DOM row instead of rebuilding it;
// `pending` marks a row whose replacement is still being sampled.
type Pair = { stem: string; orig: string; spliced: string; id: number; pending: boolean }

class Demo {
  phase: Phase = "boot"
  errorMsg = ""
  onUpdate: (() => void) | null = null

  // LM training
  lmTrainer: any = null
  lmStep = 0
  lmLoss = NaN
  lmLossHistory: number[] = []

  // Harvest: per-depth (MLP input, MLP output) rows, each normalized to
  // unit RMS with its own scale.
  rowsIn:  (Float32Array | null)[] = [null, null]
  rowsOut: (Float32Array | null)[] = [null, null]
  nRows = 0
  scaleIn  = [1, 1]
  scaleOut = [1, 1]

  // Transcoder training (two, sequential)
  tcTrainer: any = null
  tcIdx = 0
  tcStep = 0
  tcMse = NaN
  tcL0 = NaN
  tcMseHistory: number[] = []
  l1 = DEFAULT_L1
  tcL0s: number[] = []
  // Real decoder rows with the largest norms, per depth, for the anatomy strip.
  topWaves: { feat: number; curve: number[] }[][] = []

  // Per-depth transcoder weights: trained, and untrained random inits (the
  // control: identical sockets, nothing learned).
  tcW: Record<string, Float32Array>[] = []
  rndW: Record<string, Float32Array>[] = []

  // Eval
  ceOrig = NaN
  ceSpliced = NaN
  ceRandom = NaN

  // Attached forwards
  fSampleOrig: any = null
  fSampleSpliced: any = null
  fHarvest: any = null
  fEvalOrig: any = null
  fEvalSpliced: any = null

  // Explore-phase generation: a batch of aligned pairs, refreshed manually.
  pairs: Pair[] = []
  // Session-lifetime unison tally: the per-batch count swings with the
  // stems drawn; this converges to the true agreement rate.
  uniTotal = 0
  pairTotal = 0
  // Per generated word: each depth's MLP input and true output per letter,
  // so a click can compare the real MLP's move with the glass one's.
  wordActs: Map<string, { in0: Float32Array; out0: Float32Array; in1: Float32Array; out1: Float32Array; len: number }> = new Map()
  generating = false
  evalNames: string[] = []
  gpuMsg = ""

  async run() {
    try {
      const gpu = await checkWebGPU()
      if (!gpu.ok) { this.gpuMsg = gpu.message; this.phase = "no-webgpu"; this.onUpdate?.(); return }
      await this.boot()
      await this.trainLM()
      await this.harvest()
      await this.trainTC()
      await this.finalize()
      this.phase = "explore"
      tb.log(`[phase] explore  ce orig ${this.ceOrig.toFixed(3)}  spliced ${this.ceSpliced.toFixed(3)}  random ${this.ceRandom.toFixed(3)}`)
      this.onUpdate?.()
    } catch (e) {
      this.phase = "error"
      this.errorMsg = String((e as { message?: string })?.message ?? e)
      tb.log(`[error] ${this.errorMsg}`)
      this.onUpdate?.()
    }
  }

  private async boot() {
    const corpusPromise = loadCorpus()
    const t0 = performance.now()
    this.lmTrainer = await compile({
      model: new NameLM(),
      loss: lossFn,
      inputs: {
        tokens:  { shape: [B_TRAIN, T], dtype: "i32" },
        targets: { shape: [B_TRAIN, T], dtype: "i32" },
        mask:    [B_TRAIN, T],
      },
      optimizer: { kind: "adamw", lr: LR, weightDecay: 0.01, clipGradNorm: 1.0 },
    })
    const tcIn = {
      We0: [D, F] as [number, number], be0: [F] as [number],
      Wd0: [F, D] as [number, number], bd0: [D] as [number],
      si0: [1] as [number], so0: [1] as [number],
      We1: [D, F] as [number, number], be1: [F] as [number],
      Wd1: [F, D] as [number, number], bd1: [D] as [number],
      si1: [1] as [number], so1: [1] as [number],
    }
    this.fSampleOrig = await this.lmTrainer.attach({
      forward: sampleOrigFn,
      inputs: { tokens: { shape: [B_INFER, T], dtype: "i32" }, temperature: { shape: [1], dtype: "f32" } },
      output: "i32",
    })
    this.fSampleSpliced = await this.lmTrainer.attach({
      forward: sampleSplicedFn,
      inputs: { tokens: { shape: [B_INFER, T], dtype: "i32" }, temperature: { shape: [1], dtype: "f32" }, ...tcIn },
      output: "i32",
    })
    this.fHarvest = await this.lmTrainer.attach({
      forward: harvestFn,
      inputs: { tokens: { shape: [B_EVAL, T], dtype: "i32" } },
    })
    const evalIn = {
      tokens: { shape: [B_EVAL, T], dtype: "i32" as const },
      targets: { shape: [B_EVAL, T], dtype: "i32" as const },
      mask: [B_EVAL, T] as [number, number],
    }
    this.fEvalOrig    = await this.lmTrainer.attach({ forward: evalOrigFn,    inputs: evalIn })
    this.fEvalSpliced = await this.lmTrainer.attach({ forward: evalSplicedFn, inputs: { ...evalIn, ...tcIn } })
    await corpusPromise
    this.evalNames = NAMES.filter((_, i) => i % 8 === 0)
    tb.log(`[boot] compiled in ${(performance.now() - t0).toFixed(0)}ms, corpus ${NAMES.length} (${corpusSource})`)
  }

  private async trainLM() {
    this.phase = "train-lm"
    this.onUpdate?.()
    tb.log(`[phase] train-lm`)
    while (this.lmStep < LM_STEPS) {
      const r = await this.lmTrainer.step(makeTrainingBatch())
      if (r.kind === "failed") throw new Error(`LM step failed: ${r.error}`)
      if (r.kind !== "completed") continue
      this.lmStep++
      this.lmLoss = r.loss
      this.lmLossHistory.push(r.loss)
      if (this.lmLossHistory.length > 300) this.lmLossHistory.shift()
      if (this.lmStep === 1 || this.lmStep % 250 === 0) tb.log(`[lm] step ${this.lmStep} loss ${r.loss.toFixed(4)}`)
      if (this.lmStep % 5 === 0) this.onUpdate?.()
    }
  }

  private async harvest() {
    this.phase = "harvest"
    this.onUpdate?.()
    tb.log(`[phase] harvest`)
    const nBatches = Math.ceil(NAMES.length / B_EVAL)
    const raw: { arrs: Float32Array[]; count: number }[] = []
    let total = 0
    for (let b = 0; b < nBatches; b++) {
      const from = b * B_EVAL
      const batch = makeFixedBatch(NAMES, from)
      const r = await this.fHarvest.run({ tokens: batch.tokens })
      if (r.kind === "failed") throw new Error(`harvest failed: ${r.error}`)
      if (r.kind !== "completed") { b--; continue }
      const srcs = [
        r.captures.get("in0") as Float32Array,
        r.captures.get("out0") as Float32Array,
        r.captures.get("in1") as Float32Array,
        r.output as Float32Array,               // out1
      ]
      const keep: number[] = []
      for (let i = 0; i < B_EVAL; i++) {
        if (from + i >= NAMES.length) break   // wrap rows are duplicates; drop
        for (let t = 0; t < batch.valid[i]!; t++) keep.push(i * T + t)
      }
      const arrs = srcs.map(src => {
        const out = new Float32Array(keep.length * D)
        keep.forEach((s, k) => out.set(src.subarray(s * D, s * D + D), k * D))
        return out
      })
      raw.push({ arrs, count: keep.length })
      total += keep.length
    }
    // Concatenate the four streams, normalize each to unit RMS.
    const streams: Float32Array[] = []
    const scales: number[] = []
    for (let s = 0; s < 4; s++) {
      const rows = new Float32Array(total * D)
      let off = 0
      for (const c of raw) { rows.set(c.arrs[s]!, off * D); off += c.count }
      let sq = 0
      for (let i = 0; i < rows.length; i++) sq += rows[i]! * rows[i]!
      const scale = Math.sqrt(sq / rows.length) || 1
      for (let i = 0; i < rows.length; i++) rows[i]! /= scale
      streams.push(rows)
      scales.push(scale)
    }
    this.rowsIn  = [streams[0]!, streams[2]!]
    this.rowsOut = [streams[1]!, streams[3]!]
    this.scaleIn  = [scales[0]!, scales[2]!]
    this.scaleOut = [scales[1]!, scales[3]!]
    this.nRows = total
    tb.log(`[harvest] ${total} (in, out) pairs per depth, out rms scales ${this.scaleOut.map(s => s.toFixed(3)).join(" / ")}`)
  }

  private makeTCBatch(depth: number): { x: Float32Array; y: Float32Array; l1: Float32Array } {
    const x = new Float32Array(TC_B * D)
    const y = new Float32Array(TC_B * D)
    const rin = this.rowsIn[depth]!, rout = this.rowsOut[depth]!
    for (let i = 0; i < TC_B; i++) {
      const r = Math.floor(Math.random() * this.nRows)
      x.set(rin.subarray(r * D, r * D + D), i * D)
      y.set(rout.subarray(r * D, r * D + D), i * D)
    }
    return { x, y, l1: new Float32Array([this.l1]) }
  }

  private async trainTC() {
    this.phase = "train-tc"
    this.onUpdate?.()
    for (let depth = 0; depth < N_LAYERS; depth++) {
      this.tcIdx = depth
      this.tcStep = 0
      this.tcMseHistory = []
      this.onUpdate?.()
      tb.log(`[phase] train-tc ${depth + 1}/${N_LAYERS}`)
      if (!this.tcTrainer) {
        this.tcTrainer = await compile({
          model: new Transcoder(),
          loss: tcLossFn,
          inputs: { x: [TC_B, D], y: [TC_B, D], l1: [1] },
          optimizer: { kind: "adam", lr: TC_LR },
        })
      } else {
        await this.tcTrainer.replaceModel(new Transcoder())
      }
      // Snapshot the untrained init: the random-transcoder control.
      this.rndW.push(await this.tcTrainer.downloadParams())
      while (this.tcStep < TC_STEPS) {
        const r = await this.tcTrainer.step(this.makeTCBatch(depth))
        if (r.kind === "failed") throw new Error(`transcoder step failed: ${r.error}`)
        if (r.kind !== "completed") continue
        this.tcStep++
        if (this.tcStep % 5 === 0 || this.tcStep === 1) {
          const h = r.captures.get("h") as Float32Array
          let active = 0
          for (let i = 0; i < h.length; i++) if (h[i]! > 1e-6) active++
          this.tcL0 = active / TC_B
          this.tcMse = r.loss
          this.tcMseHistory.push(r.loss)
          if (this.tcMseHistory.length > 300) this.tcMseHistory.shift()
          this.onUpdate?.()
        }
        if (this.tcStep === 1 || this.tcStep % 250 === 0) {
          tb.log(`[tc${depth}] step ${this.tcStep} loss ${r.loss.toFixed(4)} L0 ${this.tcL0.toFixed(1)}`)
        }
      }
      const trained = await this.tcTrainer.downloadParams()
      for (const k of ["We", "be", "Wd", "bd"]) {
        if (!trained[k]) throw new Error(`transcoder param '${k}' missing; have: ${Object.keys(trained).join(", ")}`)
      }
      this.tcW.push(trained)
      this.tcL0s.push(this.tcL0)
    }
  }

  // Flat inputs record for the spliced forwards, from per-depth weights.
  private spliceInputs(w: Record<string, Float32Array>[]) {
    return {
      We0: w[0]!.We!, be0: w[0]!.be!, Wd0: w[0]!.Wd!, bd0: w[0]!.bd!,
      si0: new Float32Array([this.scaleIn[0]!]), so0: new Float32Array([this.scaleOut[0]!]),
      We1: w[1]!.We!, be1: w[1]!.be!, Wd1: w[1]!.Wd!, bd1: w[1]!.bd!,
      si1: new Float32Array([this.scaleIn[1]!]), so1: new Float32Array([this.scaleOut[1]!]),
    }
  }

  private async finalize() {
    const nBatches = Math.floor(this.evalNames.length / B_EVAL) || 1
    const spliceIn = this.spliceInputs(this.tcW)
    const randIn = this.spliceInputs(this.rndW)
    let o = 0, s = 0, a = 0
    for (let b = 0; b < nBatches; b++) {
      const batch = makeFixedBatch(this.evalNames, b * B_EVAL)
      const io = { tokens: batch.tokens, targets: batch.targets, mask: batch.mask }
      const ro = await this.fEvalOrig.run(io)
      const rs = await this.fEvalSpliced.run({ ...io, ...spliceIn })
      const ra = await this.fEvalSpliced.run({ ...io, ...randIn })
      for (const [r, label] of [[ro, "orig"], [rs, "spliced"], [ra, "random"]] as const) {
        if (r.kind !== "completed") throw new Error(`eval ${label} did not complete`)
      }
      o += (ro.output as Float32Array)[0]!
      s += (rs.output as Float32Array)[0]!
      a += (ra.output as Float32Array)[0]!
    }
    this.ceOrig = o / nBatches
    this.ceSpliced = s / nBatches
    this.ceRandom = a / nBatches

    // Each depth's three biggest library waves for the anatomy strip.
    this.topWaves = this.tcW.map(w => {
      const Wd = w.Wd!
      const ranked: { feat: number; norm: number }[] = []
      for (let f = 0; f < F; f++) {
        let n = 0
        for (let d2 = 0; d2 < D; d2++) n += Wd[f * D + d2]! * Wd[f * D + d2]!
        ranked.push({ feat: f, norm: n })
      }
      ranked.sort((p, q) => q.norm - p.norm)
      return ranked.slice(0, 3).map(r => ({
        feat: r.feat,
        curve: Array.from({ length: D }, (_, d2) => Wd[r.feat * D + d2]!),
      }))
    })
  }

  // Autoregressive sampling through either forward.
  private async sampleOne(spliced: boolean, temperature: number, prefix: string): Promise<string> {
    const f = spliced ? this.fSampleSpliced : this.fSampleOrig
    const tokens = new Int32Array(B_INFER * T)
    tokens.fill(EOS)
    for (let i = 0; i < prefix.length && i < T - 2; i++) tokens[1 + i] = charToId(prefix[i]!)
    const temp = new Float32Array([Math.max(0.05, temperature)])
    const extra = spliced ? this.spliceInputs(this.tcW) : {}
    let p = 1 + Math.min(prefix.length, T - 2)
    while (p < T) {
      const r = await f.run({ tokens, temperature: temp, ...extra })
      if (r.kind === "failed") { tb.log(`[gen] failed: ${r.error}`); break }
      if (r.kind !== "completed") break
      const nextId = (r.output as Int32Array)[p - 1]!
      if (nextId === EOS || nextId < 0 || nextId >= 26) break
      tokens[p] = nextId
      p++
    }
    let name = ""
    for (let t = 1; t < p; t++) {
      const id = tokens[t]!
      if (id === EOS) break
      name += idToChar(id)
    }
    return name
  }

  async generateBatch(temperature: number, prefix: string) {
    if (this.phase !== "explore" || this.generating) return
    this.generating = true
    const clean = prefix.toLowerCase().replace(/[^a-z]/g, "").slice(0, T - 3)
    const stems: string[] = []
    if (clean.length > 0) {
      for (let i = 0; i < N_PAIRS; i++) stems.push(clean)
    } else {
      const used = new Set<string>()
      let guard = 0
      while (stems.length < N_PAIRS && guard++ < 300) {
        const name = NAMES[Math.floor(Math.random() * NAMES.length)]!
        const stem = name.slice(0, 2)
        if (used.has(stem)) continue
        used.add(stem)
        stems.push(stem)
      }
    }
    // Seat the new batch into the standing rows instead of clearing them:
    // every row keeps its slot and its height while its replacement is
    // sampled, so nothing below collapses and regrows eight times.
    this.pairs = stems.map((_, i) => {
      const prev = this.pairs[i]
      return prev ? { ...prev, pending: true } : { stem: "", orig: "", spliced: "", id: i, pending: true }
    })
    this.onUpdate?.()
    for (let i = 0; i < stems.length; i++) {
      const stem = stems[i]!
      const [orig, spliced] = await Promise.all([
        this.sampleOne(false, temperature, stem),
        this.sampleOne(true, temperature, stem),
      ])
      const rows = [...this.pairs]
      rows[i] = { stem, orig, spliced, id: i, pending: false }
      this.pairs = rows
      this.pairTotal++
      if (orig === spliced) this.uniTotal++
      await this.analyzeBatch()
      this.onUpdate?.()
    }
    this.generating = false
    this.onUpdate?.()
  }

  // One harvest run over the batch's words: keep each depth's MLP input and
  // true output for every letter, for the click-a-letter comparison.
  private async analyzeBatch() {
    if (!this.fHarvest || this.tcW.length < N_LAYERS) return
    const words = [...new Set(this.pairs.flatMap(pr => [pr.orig, pr.spliced]))]
      .filter(w => w.length > 0 && w.length <= T - 2)
      .slice(0, B_EVAL)
    if (words.length === 0) return
    const tokens = new Int32Array(B_EVAL * T)
    tokens.fill(EOS)
    words.forEach((w, i) => {
      for (let j = 0; j < w.length; j++) tokens[i * T + 1 + j] = charToId(w[j]!)
    })
    const r = await this.fHarvest.run({ tokens })
    if (r.kind !== "completed") return
    const srcs = [
      r.captures.get("in0") as Float32Array,
      r.captures.get("out0") as Float32Array,
      r.captures.get("in1") as Float32Array,
      r.output as Float32Array,
    ]
    const m = new Map<string, { in0: Float32Array; out0: Float32Array; in1: Float32Array; out1: Float32Array; len: number }>()
    words.forEach((w, i) => {
      const slices = srcs.map(src => {
        const out = new Float32Array(w.length * D)
        for (let p = 1; p <= w.length; p++) {
          const s = (i * T + p) * D
          out.set(src.subarray(s, s + D), (p - 1) * D)
        }
        return out
      })
      m.set(w, { in0: slices[0]!, out0: slices[1]!, in1: slices[2]!, out1: slices[3]!, len: w.length })
    })
    this.wordActs = m
  }

  // JS-side transcoder pass on one letter: what the glass MLP outputs for
  // this input, and which library waves built it.
  decompose(depth: number, xIn: Float32Array) {
    const w = this.tcW[depth]!
    const si = this.scaleIn[depth]!, so = this.scaleOut[depth]!
    const We = w.We!, be = w.be!, Wd = w.Wd!, bd = w.bd!
    const h = new Float32Array(F)
    for (let f = 0; f < F; f++) {
      let acc = be[f]!
      for (let d = 0; d < D; d++) acc += (xIn[d]! / si) * We[d * F + f]!
      h[f] = acc > 0 ? acc : 0
    }
    const pred = new Float32Array(D)
    for (let d = 0; d < D; d++) {
      let acc = bd[d]!
      for (let f = 0; f < F; f++) if (h[f]! > 0) acc += h[f]! * Wd[f * D + d]!
      pred[d] = acc * so
    }
    const atoms: { feat: number; weight: number; mag: number; curve: Float32Array }[] = []
    for (let f = 0; f < F; f++) {
      if (h[f]! <= 1e-6) continue
      const curve = new Float32Array(D)
      let mag = 0
      for (let d = 0; d < D; d++) {
        const v = h[f]! * Wd[f * D + d]! * so
        curve[d] = v
        mag += v * v
      }
      atoms.push({ feat: f, weight: h[f]!, mag: Math.sqrt(mag), curve })
    }
    atoms.sort((p, q) => q.mag - p.mag)
    return { pred, atoms }
  }
}

// === UI ===

const PHASES: { key: Phase; label: string }[] = [
  { key: "train-lm",  label: "1 · Train the transformer" },
  { key: "harvest",   label: "2 · Record each MLP's moves" },
  { key: "train-tc",  label: "3 · Train the transcoders" },
  { key: "explore",   label: "4 · Replace and explore" },
]

class Root extends Component {
  demo = new Demo()
  temperature = 0.05
  sel: { word: string; pos: number } | null = null
  private kicked = false

  async init() {
    this.demo.onUpdate = () => {
      if (this.demo.phase === "explore" && !this.kicked) {
        this.kicked = true
        this.demo.generateBatch(this.temperature, "")
      }
      if (this.demo.phase === "explore" && !this.selValid()) this.autoSelect()
      this.update()
    }
    tb.onMessage((m: unknown) => {
      if (m === "selftest") {
        const d = this.demo
        const payload = {
          phase: d.phase,
          lmStep: d.lmStep, lmLoss: d.lmLoss,
          nRows: d.nRows,
          tcStep: d.tcStep, tcLoss: d.tcMse, tcL0s: d.tcL0s.map(v => +v.toFixed(1)),
          ceOrig: d.ceOrig, ceSpliced: d.ceSpliced, ceRandom: d.ceRandom,
          unison: d.pairs.filter(pr => !pr.pending && pr.orig === pr.spliced).length,
          uniTotal: d.uniTotal, pairTotal: d.pairTotal,
          pairs: d.pairs.map(pr => ({ stem: pr.stem, orig: pr.orig, spliced: pr.spliced })),
          waves: (() => {
            const pr = d.pairs[0]
            if (!pr) return null
            const acts = d.wordActs.get(pr.orig)
            if (!acts) return null
            const d0 = d.decompose(0, acts.in0.subarray(0, D))
            const d1 = d.decompose(1, acts.in1.subarray(0, D))
            return {
              word: pr.orig,
              active: [d0.atoms.length, d1.atoms.length],
              top0: d0.atoms.slice(0, 3).map(a => ({ feat: a.feat, weight: +a.weight.toFixed(3) })),
            }
          })(),
          error: d.errorMsg || undefined,
        }
        tb.log(`[selftest] ${JSON.stringify(payload)}`)
        return payload
      }
    })
    this.demo.run()
  }

  setTemperature(v: number) { this.temperature = v; this.update() }

  // A selection survives until its word leaves the batch, so a refresh keeps
  // the wave view populated and swaps its contents once rather than tearing
  // the panel out and putting it back.
  private selValid(): boolean {
    if (!this.sel) return false
    const acts = this.demo.wordActs.get(this.sel.word)
    return !!acts && this.sel.pos < acts.len
  }

  // Open one letter's wave view on its own, so the two MLPs' moves are on
  // screen before anyone discovers the letters are clickable. Prefers the
  // first letter where the two models parted ways; falls back to the middle.
  private autoSelect() {
    for (const pr of this.demo.pairs) {
      if (pr.pending || pr.orig.length === 0) continue
      const acts = this.demo.wordActs.get(pr.orig)
      if (!acts || acts.len === 0) continue
      let k = 0
      while (k < pr.orig.length && k < pr.spliced.length && pr.orig[k] === pr.spliced[k]) k++
      this.sel = { word: pr.orig, pos: k < acts.len ? k : Math.floor(acts.len / 2) }
      return
    }
  }

  view() {
    const d = this.demo
    return div({ class: "app" },
      div({ class: "header" },
        h1("Replacement Model"),
        p({},
          "A tiny transformer learns to invent Pokémon names. Then its MLPs are removed outright and two ",
          span({ class: "kw" }, "sparse transcoders"),
          ", trained only to imitate each MLP's input-to-output behavior, are seated in the empty sockets. Attention stays original. If the model still speaks, the machinery itself, not just its messages, has an interpretable stand-in."),
      ),
      d.phase === "no-webgpu" ? div({ class: ["panel", "boot"] }, d.gpuMsg)
      : d.phase === "error" ? div({ class: ["panel", "boot"] }, `Something broke: ${d.errorMsg}`)
      : d.phase === "boot" ? div({ class: ["panel", "boot"] }, "Compiling training + replacement graphs to WebGPU…")
      : div({},
          this.stepper(),
          d.phase === "explore" ? this.exploreView() : this.progressView(),
        ),
    )
  }

  stepper() {
    const cur = this.demo.phase
    const curIdx = PHASES.findIndex(p => p.key === cur)
    return div({ class: "stepper" },
      PHASES.map((p, i) =>
        span({ class: i < curIdx ? "step done" : i === curIdx ? "step current" : "step" }, p.label)),
    )
  }

  progressView() {
    const d = this.demo
    if (d.phase === "train-lm") {
      return div({ class: ["panel", "progress"] },
        h3(`Training the transformer  ·  step ${d.lmStep} / ${LM_STEPS}  ·  loss ${isFinite(d.lmLoss) ? d.lmLoss.toFixed(3) : "…"}`),
        this.chart(d.lmLossHistory, Math.log(V) * 1.05),
        p({ class: "hint" }, "Cross-entropy per character. Chance is ln 27 ≈ 3.30; a trained model lands around 1.6."),
      )
    }
    if (d.phase === "harvest") {
      return div({ class: ["panel", "progress"] }, h3("Recording each MLP's moves…"),
        p({ class: "hint" }, "Every name flows through the trained model; for each character, each MLP's input and output are saved as one training example of the function to imitate."))
    }
    return div({ class: ["panel", "progress"] },
      h3(`Training transcoder ${d.tcIdx + 1} of ${N_LAYERS}  ·  step ${d.tcStep} / ${TC_STEPS}  ·  loss ${isFinite(d.tcMse) ? d.tcMse.toFixed(4) : "…"}  ·  L0 ≈ ${isFinite(d.tcL0) ? d.tcL0.toFixed(1) : "…"}`),
      this.chart(d.tcMseHistory, undefined),
      p({ class: "hint" }, `Imitation + sparsity loss: predict the MLP's output from its input, using as few of the ${F} features as possible per character.`),
    )
  }

  exploreView() {
    const d = this.demo
    return div({},
      this.anatomy(),
      div({ class: ["panel", "controls"] },
        div({ class: "control-row" },
          span({ class: "control-label" }, "temp"),
          input({
            class: "temp-slider", type: "range", min: "0.05", max: "1.5", step: "0.05",
            value: String(this.temperature),
            onInput: (e: any) => this.setTemperature(parseFloat(e.target.value)),
          }),
          span({ class: "temp-value" }, this.temperature <= 0.06 ? "greedy" : this.temperature.toFixed(2)),
          button({
            class: "btn",
            disabled: d.generating,
            onClick: () => d.generateBatch(this.temperature, ""),
          }, d.generating ? "generating…" : "new words"),
        ),
      ),
      this.pairsPanel(),
      this.cePanel(),
    )
  }

  // Compact anatomy strip: blocks opened, MLPs replaced, real numbers.
  anatomy() {
    const d = this.demo
    const recovery = Math.round(100 * (d.ceRandom - d.ceSpliced) / (d.ceRandom - d.ceOrig))
    const AW = 880, AH = 128, SY = 56
    const seg = (x1: number, x2: number, spliced: boolean) =>
      line({ x1, y1: SY, x2, y2: SY, class: spliced ? "an-spine spliced-line march" : "an-spine orig-line march" })
    const box = (x: number, w: number, label: string) => g({},
      rect({ x, y: SY - 20, width: w, height: 40, rx: 7, class: "an-box" }),
      text({ x: x + w / 2, y: SY + 4, class: "an-label" }, label),
    )
    const openBlock = (x: number, depth: number, label: string) => {
      const waves = d.topWaves[depth] ?? []
      const gx = x + 96
      const pts = (curve: number[], x0: number) => {
        let m = 1e-6
        for (const v of curve) if (Math.abs(v) > m) m = Math.abs(v)
        return curve.map((v, i) => `${(x0 + (i / (curve.length - 1)) * 22).toFixed(1)},${(SY - (v / m) * 11).toFixed(1)}`).join(" ")
      }
      return g({},
        rect({ x, y: SY - 30, width: 200, height: 60, rx: 9, class: "an-open" }),
        text({ x: x + 10, y: SY - 36, class: "an-sub an-left" }, label),
        rect({ x: x + 10, y: SY - 16, width: 76, height: 32, rx: 6, class: "an-box" }),
        text({ x: x + 48, y: SY + 4, class: "an-label" }, "attn"),
        rect({ x: gx, y: SY - 19, width: 92, height: 38, rx: 6, class: "an-codec" }),
        waves.map((wv, i) =>
          polyline({ key: wv.feat, points: pts(wv.curve, gx + 8 + i * 28), style: { stroke: this.atomColor(wv.feat), fill: "none", strokeWidth: "1.3" } })),
        text({ x: x + 100, y: SY + 34, class: "an-sub" },
          `glass MLP · keep ~${isFinite(d.tcL0s[depth] ?? NaN) ? d.tcL0s[depth]!.toFixed(0) : "…"} of ${F}`),
      )
    }
    return div({ class: ["panel", "anatomy-panel"] },
      svg({ viewBox: `0 0 ${AW} ${AH}`, width: "100%", class: "anatomy" },
        text({ x: 26, y: SY + 4, class: "an-sub" }, "tokens"),
        seg(52, 70, false),
        box(70, 62, "embed"),
        seg(132, 162, false),
        openBlock(162, 0, "block 1: original MLP removed"),
        seg(362, 400, true),
        openBlock(400, 1, "block 2: original MLP removed"),
        seg(600, 640, true),
        box(640, 58, "head"),
        seg(698, 726, true),
        text({ x: 754, y: SY + 4, class: "an-label" }, "logits"),
        text({ x: AW - 8, y: 16, class: "an-intact" },
          isFinite(recovery) ? `signal ≈ ${recovery}% intact` : ""),
      ),
      p({ class: "hint" },
        "The replaced model's anatomy. Attention runs original; each block's MLP is gone, its socket filled by a sparse transcoder that imitates it (the squiggles are its real biggest library waves). The random control seats untrained transcoders in the same sockets."),
    )
  }

  cePanel() {
    const d = this.demo
    const delta = d.ceSpliced - d.ceOrig
    return div({ class: ["panel", "ce-panel"] },
      h3(`Cross-entropy per character  ·  lower is better  ·  chance is ${Math.log(V).toFixed(2)}`),
      div({ class: "ce-row" },
        div({ class: "ce-cell" }, span({ class: "ce-label" }, "original model"), span({ class: "ce-val" }, d.ceOrig.toFixed(3)), span({ class: "ce-sub" }, "both MLPs intact")),
        div({ class: "ce-cell" }, span({ class: "ce-label" }, "MLPs replaced by transcoders"), span({ class: "ce-val" }, d.ceSpliced.toFixed(3)), span({ class: "ce-sub" }, `+${delta.toFixed(3)} vs original`)),
        div({ class: "ce-cell" }, span({ class: "ce-label" }, "untrained transcoders"), span({ class: "ce-val" }, d.ceRandom.toFixed(3)), span({ class: "ce-sub" }, "the control")),
      ),
      div({ class: "ce-aside" },
        "Sparsity: the two glass MLPs keep ",
        span({ class: "ce-aside-val" },
          d.tcL0s.length > 0 ? d.tcL0s.map(v => v.toFixed(0)).join(" and ") : "…"),
        ` of ${F} features active per letter.`),
      p({ class: "hint" },
        "The original MLPs never execute in the replaced model; trained imitations compute in their place. The claim: learned transcoders barely hurt, while untrained ones in the same sockets do worse than guessing. That gap is the machinery the imitations learned."),
    )
  }

  pairsPanel() {
    const d = this.demo
    const settled = d.pairs.filter(pr => !pr.pending)
    const uni = settled.filter(pr => pr.orig === pr.spliced).length
    const tail = d.generating ? "  ·  generating…" : ""
    return div({ class: ["panel", "pairs-panel"] },
      h3(settled.length > 0
        ? `Same prompt, both brains  ·  ${uni}/${settled.length} in unison${d.pairTotal > N_PAIRS ? `  ·  ${d.uniTotal}/${d.pairTotal} overall` : ""}${tail}`
        : `Same prompt, both brains${tail}`),
      p({ class: "hint" },
        "Each row gives both models the same starting letters (dimmed). At greedy temperature every choice is the model's single best guess, so a matching row means the replacement machinery preserved the model's decisions letter for letter; highlighted letters are where it changed its mind. Raise the temperature to let both improvise. Click any letter to compare the real MLP's move with the glass one's."),
      div({ class: "pairs-head" }, span({}, "original"), span({}, "MLPs replaced")),
      div({ class: "pairs" }, d.pairs.map(pr => this.pairRow(pr))),
      this.wavePanel(),
    )
  }

  pairRow(pr: Pair) {
    const a = pr.orig, b = pr.spliced
    let k = 0
    while (k < a.length && k < b.length && a[k] === b[k]) k++
    const stemLen = Math.min(pr.stem.length, k)
    const word = (w: string) =>
      span({ class: "pair-word", title: NAMES_SET.has(w) ? "matches a real name in the corpus" : undefined },
        ...w.split("").map((ch, i2) => {
          const picked = this.sel !== null && this.sel.word === w && this.sel.pos === i2
          const cls = "ltr" + (i2 < stemLen ? " stem" : "") + (i2 >= k ? " diff" : "") + (picked ? " picked" : "")
          return span({
            class: cls,
            onClick: () => { this.sel = { word: w, pos: i2 }; this.update() },
          }, ch)
        }),
      )
    return div({ key: pr.id, class: pr.pending ? "pair-row pending" : "pair-row" },
      word(a),
      a === b && a.length > 0
        ? span({ class: "pair-uni" }, word(b), span({ class: "unison" }, "✓ unison"))
        : word(b),
    )
  }

  // Stable per-atom color, same formula as the sibling bulbs.
  private atomColor(feat: number): string {
    return `hsl(${((feat * 137.508) % 360).toFixed(1)}, 65%, 52%)`
  }

  private wavePoints(curve: ArrayLike<number>, w: number, h: number, yMax: number): string {
    const pts: string[] = []
    for (let i = 0; i < D; i++) {
      const px = (i / (D - 1)) * w
      const py = h / 2 - ((curve[i] as number) / yMax) * (h / 2 - 2)
      pts.push(`${px.toFixed(1)},${py.toFixed(1)}`)
    }
    return pts.join(" ")
  }

  wavePanel() {
    const d = this.demo
    if (!this.sel) return null
    const acts = d.wordActs.get(this.sel.word)
    if (!acts || this.sel.pos >= acts.len) return null
    const w = this.sel.word, pos = this.sel.pos
    return div({ class: "waves" },
      h3({}, "Both MLPs' moves while reading \"", span({ class: "wave-hi" }, w[pos]!), `" in ${w}`),
      div({ class: "wave-cols" },
        this.depthWaves(0, acts.in0.subarray(pos * D, pos * D + D), acts.out0.subarray(pos * D, pos * D + D)),
        this.depthWaves(1, acts.in1.subarray(pos * D, pos * D + D), acts.out1.subarray(pos * D, pos * D + D)),
      ),
      p({ class: "hint" },
        "Gray is the real MLP's output for this letter (its 64 channels in a fixed order; the shape is a fingerprint, not a function). Teal is the glass MLP's imitation, built from the library waves below with their weights. The real MLP computed this with 16,384 tangled weights; the glass one with a few dozen nameable waves."),
    )
  }

  depthWaves(depth: number, xIn: Float32Array, yTrue: Float32Array) {
    const dec = this.demo.decompose(depth, xIn)
    const W = 300, H = 72, AH = 26
    let yMax = 1e-6
    for (let i = 0; i < D; i++) {
      const a = Math.abs(yTrue[i]!), b = Math.abs(dec.pred[i]!)
      if (a > yMax) yMax = a
      if (b > yMax) yMax = b
    }
    const shown = dec.atoms.slice(0, 6)
    return div({ class: "wave-col" },
      div({ class: "wave-title" }, `glass MLP ${depth + 1}  ·  ${dec.atoms.length} waves active`),
      svg({ viewBox: `0 0 ${W} ${H}`, width: "100%", class: "wave-main" },
        line({ x1: 0, y1: H / 2, x2: W, y2: H / 2, class: "wave-axis" }),
        polyline({ points: this.wavePoints(yTrue, W, H, yMax), class: "wave-orig" }),
        polyline({ points: this.wavePoints(dec.pred, W, H, yMax), class: "wave-rebuilt" }),
      ),
      div({ class: "wave-atoms" },
        shown.map(a =>
          div({ class: "wave-atom", key: a.feat },
            svg({ viewBox: `0 0 ${W} ${AH}`, width: "100%" },
              polyline({ points: this.wavePoints(a.curve, W, AH, yMax), style: { stroke: this.atomColor(a.feat), fill: "none", strokeWidth: "1.4" } }),
            ),
            span({ class: "wave-label" }, `#${a.feat} × ${a.weight.toFixed(2)}`),
          )),
        dec.atoms.length > shown.length
          ? div({ class: "wave-more" }, `+${dec.atoms.length - shown.length} smaller waves`)
          : null,
      ),
    )
  }

  chart(hist: number[], ceil: number | undefined) {
    if (hist.length < 2) return div({ class: "chart-empty" }, "(warming up…)")
    const w = 600, h = 40
    const padL = 4, padR = 4, padT = 3, padB = 3
    const innerW = w - padL - padR
    const innerH = h - padT - padB
    const maxL = Math.max(ceil ?? 0, ...hist)
    const xAt = (i: number) => padL + (i / Math.max(1, hist.length - 1)) * innerW
    const yAt = (l: number) => padT + (1 - l / maxL) * innerH
    const pts = hist.map((l, i) => `${xAt(i).toFixed(1)},${yAt(l).toFixed(1)}`).join(" ")
    return svg({ viewBox: `0 0 ${w} ${h}`, width: "100%", class: "loss-chart" },
      polyline({ points: pts, class: "chart-line" }),
    )
  }
}

const root = new Root()
new App({ root, id: "app" })
root.init()
```

**styles.css**

```css
:root {
  color-scheme: light;
  --bg-page:        #ffffff;
  --bg-panel:       #ffffff;
  --bg-subpanel:    #f6f7f9;
  --text-primary:   #1a1a1a;
  --text-muted:     #666666;
  --border:         #dddddd;
  --border-strong:  #888888;
  --accent:         #0d9488;
  --chart-line:     #0d9488;
  --name-bg:        #f3f4f6;
  --kw:             #0f766e;
  --hi-bg:          #ccfbf1;
  --font-mono:      ui-monospace, Menlo, monospace;
}
html[data-theme="dark"] {
  color-scheme: dark;
  --bg-page:        #0c0c0e;
  --bg-panel:       #18181b;
  --bg-subpanel:    #1a1a1d;
  --text-primary:   #e5e5e5;
  --text-muted:     #9ca3af;
  --border:         #2e2e34;
  --border-strong:  #555560;
  --accent:         #2dd4bf;
  --chart-line:     #2dd4bf;
  --name-bg:        #232328;
  --kw:             #5eead4;
  --hi-bg:          #134e4a;
}

body {
  font-family: ui-sans-serif, system-ui, sans-serif;
  margin: 0;
  padding: 0 1rem 0.5rem;
  background: var(--bg-page);
  color: var(--text-primary);
  -webkit-font-smoothing: antialiased;
}

/* Padding, not margin: the last panel's bottom margin collapses straight out
   of .app, so a margin here buys no gap at the foot of the page. */
.app { max-width: 1180px; margin: 0 auto; padding-bottom: 0.75rem; }

.header { margin: 1.2rem 0 0.4rem; }
.header h1 { font-size: 1.4rem; margin: 0; font-weight: 600; text-align: center; }
.header p {
  color: var(--text-muted);
  margin: 1rem 0 0;
  line-height: 1.55;
  font-size: 0.9rem;
}
.kw { color: var(--kw); font-weight: 600; }

.panel {
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-panel);
  padding: 0.85rem 1rem;
  margin-bottom: 1rem;
}
.panel h3 { font-size: 0.9rem; font-weight: 600; margin: 0 0 0.6rem; }

.boot {
  margin-top: 1.6rem;
  color: var(--text-muted);
  font-style: italic;
  text-align: center;
  padding: 2rem;
}
.hint { color: var(--text-muted); font-size: 0.82rem; line-height: 1.5; margin: 0.5rem 0 0; }

/* The transient training panels are only a few lines tall; the default panel
   rhythm reads cramped there, so open up the vertical spacing. */
.progress { padding: 1.25rem 1.2rem 1.15rem; }
.progress h3 { margin-bottom: 0.9rem; }
.progress .hint { margin-top: 0.85rem; }

.stepper {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.4rem 1.2rem;
  margin: 1.6rem 0 1.7rem;
  font-size: 0.82rem;
}
.step { color: var(--text-muted); }
.step.current { color: var(--accent); font-weight: 600; }
.step.done { color: var(--text-primary); }
.step.done::after { content: " ✓"; }

.controls .control-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  align-items: center;
  justify-content: center;
}
.control-label { font-size: 0.82rem; color: var(--text-muted); }
.temp-slider { width: 8rem; accent-color: var(--accent); }
.temp-value {
  font-family: var(--font-mono);
  font-size: 0.82rem;
  color: var(--text-muted);
  min-width: 2.5rem;
}
.btn {
  font: inherit;
  font-size: 0.82rem;
  padding: 0.35rem 0.9rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-panel);
  color: var(--text-primary);
  cursor: pointer;
}
.btn:hover { border-color: var(--border-strong); }
.btn:disabled { opacity: 0.55; cursor: default; }

.ce-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}
.ce-cell {
  flex: 1 1 8rem;
  background: var(--bg-subpanel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.55rem 0.8rem;
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}
.ce-label { font-size: 0.75rem; color: var(--text-muted); }
.ce-val { font-family: var(--font-mono); font-size: 1.25rem; font-weight: 600; }
.ce-sub { font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); }
.ce-aside {
  margin-top: 0.6rem;
  font-size: 0.82rem;
  color: var(--text-muted);
}
.ce-aside-val {
  font-family: var(--font-mono);
  font-weight: 600;
  color: var(--text-primary);
}

.pairs-head {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
  font-size: 0.75rem;
  color: var(--text-muted);
  margin: 0.7rem 0 0.3rem;
}
.pairs { display: flex; flex-direction: column; }
.pair-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
  padding: 0.4rem 0;
  min-height: 1.3rem;
  border-top: 1px solid var(--border);
  align-items: baseline;
  transition: opacity 0.28s ease;
  animation: pair-fade-in 0.35s ease-out backwards;
}
.pair-row.pending { opacity: 0.3; }
@keyframes pair-fade-in {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: none; }
}
.pair-word {
  font-family: var(--font-mono);
  font-size: 1.05rem;
  font-weight: 600;
  text-transform: capitalize;
}
.pair-word .stem { opacity: 0.5; }
.pair-word .diff {
  background: var(--hi-bg);
  color: var(--accent);
  border-radius: 3px;
  padding: 0 2px;
}
.unison {
  color: var(--text-muted);
  font-size: 0.82rem;
  font-style: italic;
}
.pair-uni { display: flex; gap: 0.5rem; align-items: baseline; }

.ltr { cursor: pointer; border-radius: 3px; }
.ltr:hover { background: var(--hi-bg); }
.ltr.picked { background: var(--accent); color: #fff; }

.waves {
  margin-top: 0.9rem;
  border-top: 1px solid var(--border);
  padding-top: 0.7rem;
}
.waves h3 { margin-bottom: 0.5rem; }
.wave-hi {
  background: var(--hi-bg);
  color: var(--accent);
  border-radius: 3px;
  padding: 0 3px;
}
.wave-cols {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
}
@media (min-width: 720px) {
  .wave-cols { grid-template-columns: 1fr 1fr; }
}
.wave-title {
  font-size: 0.78rem;
  color: var(--text-muted);
  margin-bottom: 0.45rem;
  line-height: 1.4;
}
.wave-main {
  display: block;
  background: var(--bg-subpanel);
  border: 1px solid var(--border);
  border-radius: 6px;
}
.wave-axis { stroke: var(--border); stroke-width: 1; }
.wave-orig { fill: none; stroke: var(--text-muted); stroke-width: 1.4; opacity: 0.75; }
.wave-rebuilt { fill: none; stroke: var(--accent); stroke-width: 1.6; }
.wave-atoms { margin-top: 0.35rem; display: flex; flex-direction: column; gap: 0.15rem; }
.wave-atom {
  display: grid;
  grid-template-columns: 1fr 5.5rem;
  gap: 0.5rem;
  align-items: center;
}
.wave-atom svg { display: block; }
.wave-label {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-muted);
  text-align: right;
}
.wave-more {
  font-size: 0.75rem;
  color: var(--text-muted);
  font-style: italic;
  margin-top: 0.15rem;
}

.anatomy-panel { padding-bottom: 0.6rem; }
.anatomy { display: block; }
.an-spine { stroke-width: 2; stroke-dasharray: 5 5; }
.orig-line { stroke: #9ca3af; }
html[data-theme="dark"] .orig-line { stroke: #6b7280; }
.spliced-line { stroke: var(--accent); }
.march { animation: march 0.7s linear infinite; }
@keyframes march { to { stroke-dashoffset: -10; } }
.an-box {
  fill: var(--bg-subpanel);
  stroke: var(--border-strong);
  stroke-width: 1;
}
.an-open {
  fill: none;
  stroke: var(--border);
  stroke-width: 1;
}
.an-codec {
  fill: var(--bg-panel);
  stroke: var(--accent);
  stroke-width: 1.6;
}
.an-label { fill: var(--text-primary); font-size: 13px; text-anchor: middle; font-weight: 600; }
.an-sub { fill: var(--text-muted); font-size: 11px; text-anchor: middle; }
.an-sub.an-left { text-anchor: start; }
.an-intact { fill: var(--accent); font-size: 12px; font-weight: 600; text-anchor: end; }

.loss-chart { display: block; width: 100%; height: auto; }
.loss-chart .chart-line { fill: none; stroke: var(--chart-line); stroke-width: 1.5; }
.chart-empty {
  color: var(--text-muted);
  font-size: 0.85rem;
  font-style: italic;
  text-align: center;
  padding: 1rem 0;
}
```

**index.html**

```html
<div id="app"></div>
```

**config.json**

```json
{
  "description": "A tiny transformer invents Pokémon names. Then its MLPs are removed and sparse transcoders imitate them in place. It still speaks. Live on WebGPU.",
  "dependencies": {
    "domeleon": "^0.6.3",
    "tensorgrad": "^0.4.4"
  }
}
```
