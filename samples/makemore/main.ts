// Char-level name generator. Same transformer as samples/transformer/, but
// the task here genuinely requires learning a distribution from data: given
// ~32K real first names, learn to emit plausible *new* names (Adley, Vondelle,
// Maryon...). Unlike the addition demo, there's no underlying algorithm to
// imitate — name-shaped strings are only defined by reference to the training
// set. Karpathy's makemore tutorial in TypeScript form.
//
// Model: 3-layer pre-LN decoder transformer, 64 dim, 4 heads. Vocab = 27
// (26 letters + '.' as BOS/EOS/pad). Loss = mean cross-entropy on next-char
// prediction; pad positions train the model to keep emitting '.' after EOS,
// which is exactly what we want at sampling time. Sampling is autoregressive
// with multinomial temperature-1 sampling, stops on '.' or SEQ_LEN.

import {
  Module, compile, spec, isWebGPUAvailable, lr, nn,
  add, mul, sum, swapAxes,
  relu, matmul, embedding, arange,
  softmaxCausal, splitHeads, mergeHeads,
  type Tensor,
} from 'tensorgrad'

// ---------- Hyperparameters -------------------------------------------------
const VOCAB = 27                              // 26 letters + '.' (BOS/EOS/pad)
const D = 64
const N_LAYERS = 3
const N_HEADS = 4
const D_HEAD = D / N_HEADS
const SEQ_LEN = 17                            // longest name is 15 chars: '.' + 15 + '.'
const T = SEQ_LEN - 1                         // 16 input positions; targets shifted by 1
const B = 32
const LR = lr.linear({ peak: 0.005, final: 0.0005, steps: 1500 })
const SCALE_QK = 1 / Math.sqrt(D_HEAD)

const PAD = 0                                 // '.' = 0; 'a'..'z' = 1..26
const encodeChar = (ch: string) => ch === '.' ? PAD : (ch.charCodeAt(0) - 97 + 1)
const decodeChar = (idx: number) => idx === PAD ? '.' : String.fromCharCode(idx - 1 + 97)

// ---------- Modules: structure of the param tree ---------------------------

class Attention extends Module {
  q = new nn.Linear(D, D, { bias: false })
  k = new nn.Linear(D, D, { bias: false })
  v = new nn.Linear(D, D, { bias: false })
  o = new nn.Linear(D, D, { bias: false })
}

class MLP extends Module {
  up   = new nn.Linear(D, 4 * D)
  down = new nn.Linear(4 * D, D)
}

class Block extends Module {
  ln1  = new nn.LayerNorm(D)
  attn = new Attention()
  ln2  = new nn.LayerNorm(D)
  mlp  = new MLP()
}

class Transformer extends Module {
  tok_emb: Tensor; pos_emb: Tensor
  layers: Block[]
  lnf: nn.LayerNorm
  constructor() {
    super()
    this.tok_emb = this.param([VOCAB, D])
    this.pos_emb = this.param([SEQ_LEN, D])
    this.layers = []
    for (let i = 0; i < N_LAYERS; i++) this.layers.push(new Block())
    this.lnf = new nn.LayerNorm(D)
  }
}

// ---------- Forward functions ----------------------------------------------

function attentionFwd(p: Attention, x: Tensor): Tensor {
  const q = splitHeads(p.q.fwd(x), N_HEADS)
  const k = splitHeads(p.k.fwd(x), N_HEADS)
  const v = splitHeads(p.v.fwd(x), N_HEADS)
  const scores = mul(matmul(q, swapAxes(k, -1, -2)), SCALE_QK)
  return p.o.fwd(mergeHeads(matmul(softmaxCausal(scores), v)))
}

function mlpFwd(p: MLP, x: Tensor): Tensor {
  return p.down.fwd(relu(p.up.fwd(x)))
}

function blockFwd(p: Block, x: Tensor): Tensor {
  const a = attentionFwd(p.attn, p.ln1.fwd(x))
  const x1 = add(x, a)
  return add(x1, mlpFwd(p.mlp, p.ln2.fwd(x1)))
}

function modelFwd(p: Transformer, tokens: Tensor): Tensor {
  const tokE = embedding(tokens, p.tok_emb)
  const posE = embedding(arange(T), p.pos_emb)
  let x = add(tokE, posE)
  for (let i = 0; i < p.layers.length; i++) x = blockFwd(p.layers[i]!, x)
  return matmul(p.lnf.fwd(x), swapAxes(p.tok_emb, -1, -2))   // tied output head
}

// Loss is mean cross-entropy over *valid* positions only — those predicting the
// first char of the name through the EOS '.'. The mask is normalized by
// makeBatch so sum(mask) = 1 over the whole batch, which makes sum(ce*mask)
// the mean and lets the reported number compare directly with makemore
// references (~2.0 at convergence) instead of being diluted ~2x by trivial
// '.'-from-'.' pad positions.
function lossFn(p: Transformer, { tokens, targets, mask }: { tokens: Tensor; targets: Tensor; mask: Tensor }): Tensor {
  return sum(mul(nn.crossEntropy(modelFwd(p, tokens), targets, { reduction: 'none' }), mask))
}

function predictFwd(p: Transformer, { tokens }: { tokens: Tensor }): Tensor {
  return modelFwd(p, tokens)
}

// ---------- Sampling helpers ------------------------------------------------

function softmaxRow(out: Float32Array, start: number, len: number): Float32Array {
  let max = -Infinity
  for (let i = 0; i < len; i++) max = Math.max(max, out[start + i]!)
  const probs = new Float32Array(len)
  let sum = 0
  for (let i = 0; i < len; i++) { probs[i] = Math.exp(out[start + i]! - max); sum += probs[i]! }
  for (let i = 0; i < len; i++) probs[i] = probs[i]! / sum
  return probs
}

function sampleFromProbs(probs: Float32Array): number {
  const r = Math.random()
  let acc = 0
  for (let i = 0; i < probs.length; i++) { acc += probs[i]!; if (r < acc) return i }
  return probs.length - 1
}

// ---------- Data ------------------------------------------------------------

// Karpathy's makemore dataset — ~32K lowercase first names, one per line.
const NAMES_URL = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'

async function loadNames(): Promise<string[]> {
  const r = await fetch(NAMES_URL)
  if (!r.ok) throw new Error(`fetch names.txt: ${r.status}`)
  return (await r.text()).split('\n').map(s => s.trim()).filter(s => /^[a-z]+$/.test(s))
}

let TRAIN: string[] = []                          // ~90% of names; training pool
let TEST: string[] = []                           // ~10% holdout for val loss
let TRAIN_SET = new Set<string>()                 // for novelty check at sample time

// Stable hash → split. Same name always goes to the same side, even across
// reloads — so val loss is measuring true held-out generalization.
function isTestName(name: string): boolean {
  let h = 0
  for (let i = 0; i < name.length; i++) h = ((h * 31) + name.charCodeAt(i)) | 0
  return (((h % 10) + 10) % 10) === 0             // ~10% test
}

function makeBatch(useTest: boolean): { tokens: Int32Array; targets: Int32Array; mask: Float32Array } {
  const pool = useTest ? TEST : TRAIN
  const tokens = new Int32Array(B * T)
  const targets = new Int32Array(B * T)
  const mask = new Float32Array(B * T)
  // full = '.' + name + '.' + pad; tokens = full[0..T), targets = full[1..T+1).
  // Valid prediction positions: 0..name.length (BOS → first char, then each
  // char → next char, ending with last char → EOS '.'). Beyond that the
  // target is '.' predicting '.' from a context of all '.' — trivially zero
  // loss after a couple of steps; including those positions in the mean is
  // what makes the reported loss look ~2x better than it actually is.
  const full = new Int32Array(SEQ_LEN)
  let validCount = 0
  for (let bi = 0; bi < B; bi++) {
    const name = pool[Math.floor(Math.random() * pool.length)]!
    full.fill(PAD)
    for (let i = 0; i < name.length && i + 1 < SEQ_LEN; i++) full[i + 1] = encodeChar(name[i]!)
    const lastValid = Math.min(name.length, T - 1)
    for (let t = 0; t < T; t++) {
      tokens[bi * T + t] = full[t]!
      targets[bi * T + t] = full[t + 1]!
      if (t <= lastValid) { mask[bi * T + t] = 1; validCount++ }
    }
  }
  // Normalize so sum(mask) === 1; then sum(ce * mask) is the per-valid-token mean.
  const inv = 1 / validCount
  for (let i = 0; i < mask.length; i++) mask[i] = mask[i]! * inv
  return { tokens, targets, mask }
}

// ---------- Logging UI ------------------------------------------------------

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

stopBtn.onclick = () => { stopRequested = true }

// ---------- Main ------------------------------------------------------------

async function run() {
  if (!isWebGPUAvailable()) { log('WebGPU not available. Use Chrome 113+ or Safari 17.4+.', 'err'); return }

  runBtn.disabled = true; stopBtn.disabled = false; stopRequested = false; logEl.innerHTML = ''

  log('Loading names corpus...')
  const all = await loadNames()
  TRAIN = []; TEST = []
  for (const n of all) (isTestName(n) ? TEST : TRAIN).push(n)
  TRAIN_SET = new Set(TRAIN)
  const avgLen = all.reduce((s, n) => s + n.length, 0) / all.length
  log(`  ${all.length.toLocaleString()} names loaded — ${TRAIN.length.toLocaleString()} train / ${TEST.length.toLocaleString()} val (avg length ${avgLen.toFixed(1)})`, 'ok')

  log('Building model + compiling...')
  const t0 = performance.now()
  const model = new Transformer()
  const train = await compile(spec({
    model,
    loss: lossFn,
    optimizer: { kind: 'adamw', lr: LR, weightDecay: 0.01 },
    inputs: {
      tokens:  { shape: [B, T], dtype: 'i32' },
      targets: { shape: [B, T], dtype: 'i32' },
      mask:    [B, T],
    },
  }))
  log(`  ${train.paramNames.length} params, ${train.kernels.length} kernels, compile ${(performance.now() - t0).toFixed(0)} ms`, 'ok')

  log('Compiling inference + val-loss graphs...')
  const tInfer = performance.now()
  const infer = await compile(spec({
    model,
    forward: predictFwd,
    inputs: { tokens: { shape: [1, T], dtype: 'i32' } },
  }), { shareWith: train })
  // Forward-only loss graph at full batch shape — feeds the periodic val probe.
  // Shares the param buffers via { shareWith }, so val loss reflects the
  // latest training state.
  const valLossFwd = await compile(spec({
    model,
    forward: lossFn,
    inputs: {
      tokens:  { shape: [B, T], dtype: 'i32' },
      targets: { shape: [B, T], dtype: 'i32' },
      mask:    [B, T],
    },
  }), { shareWith: train })
  log(`  compile ${(performance.now() - tInfer).toFixed(0)} ms`, 'ok')

  const tokensBuf = new Int32Array(T)
  async function sampleName(): Promise<string> {
    const generated: number[] = []
    while (generated.length + 1 < T) {
      tokensBuf.fill(PAD)                       // tokens[0] = '.' (BOS)
      for (let i = 0; i < generated.length; i++) tokensBuf[i + 1] = generated[i]!
      const r = await infer.run({ tokens: tokensBuf })
      if (r.kind === 'aborted') return generated.map(decodeChar).join('')
      const readPos = generated.length          // next-token logits at the last written position
      const probs = softmaxRow(r.output, readPos * VOCAB, VOCAB)
      const next = sampleFromProbs(probs)
      if (next === PAD) break
      generated.push(next)
    }
    return generated.map(decodeChar).join('')
  }

  log('Training...')
  let step = 0
  let stepStart = performance.now()
  while (!stopRequested) {
    step++
    const sr = await train.step(makeBatch(false))
    if (sr.kind === 'aborted') break
    const lossVal = sr.loss
    if (step === 1 || step % 20 === 0) {
      const interval = step === 1 ? 1 : 20
      const dt = (performance.now() - stepStart) / Math.max(1, interval)
      stepStart = performance.now()
      const exPerSec = dt > 0 ? Math.round(B * 1000 / dt) : 0
      log(`  step ${step.toString().padStart(4)}  loss ${lossVal.toFixed(4)}  (${exPerSec.toLocaleString()} ex/s)`)
    }
    if (step === 1 || step % 100 === 0) {
      const vr = await valLossFwd.run(makeBatch(true))
      if (vr.kind === 'aborted') break
      const valLoss = vr.output[0]!
      const samples: string[] = []
      for (let i = 0; i < 8; i++) samples.push(await sampleName())
      const novel = samples.filter(s => !TRAIN_SET.has(s) && s.length > 0).length
      log(`  [val@${step}] loss ${valLoss.toFixed(4)}   [samples] ${samples.join(', ')}   (novel ${novel}/8)`)
    }
    if (step % 5 === 0) await new Promise(r => setTimeout(r, 0))
  }
  log(`Stopped at step ${step}.`, 'ok')
  infer.destroy(); valLossFwd.destroy(); train.destroy()
  runBtn.disabled = false; stopBtn.disabled = true
}

runBtn.onclick = () => { run().catch(e => log(`error: ${e?.message ?? e}\n${e?.stack ?? ''}`, 'err')) }
