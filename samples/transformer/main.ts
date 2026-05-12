// Transformer training in the browser, using tensorgrad's Module abstraction.
// Compare to the previous version of this file (or to transformer-jax.bulb.md):
// the model is now defined as nested classes; param names are auto-derived
// from property paths; no string-keyed dictionaries; no manual paramInput
// boilerplate. Forward functions stay pure — JAX-style separation of state
// and computation.

import {
  Module, compileModule, lr, nn, capture,
  add, mul, sum, swapAxes,
  relu, matmul, embedding, arange,
  softmaxCausal,
  type Tensor,
} from 'tensorgrad'

// ---------- Hyperparameters ------------------------------------------------
const VOCAB = 12
const D = 64
const N_LAYERS = 3
const N_HEADS = 4
const D_HEAD = D / N_HEADS
const SEQ_LEN = 9
const T = SEQ_LEN - 1
const RESULT_START = 6
const N_RESULT_DIGITS = 3
const TOK_PLUS = 10
const TOK_EQ = 11
const B = 128
// Linear LR decay: peak at step 1, finalLr by `decaySteps`, flat after.
// Letting bigger-batch take a bigger initial step then anneal — the recipe
// the JS bulb uses to recover small-batch generalization at higher throughput.
const LR = lr.linearDecay({ peak: 0.005, final: 0.0005, steps: 1500 })
const SCALE_QK = 1 / Math.sqrt(D_HEAD)

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

// ---------- View functions: pure forward computation -----------------------

function attentionFwd(p: Attention, x: Tensor, layerIdx: number): Tensor {
  // `splitHeads(p.q.fwd(x), H)` does the multi-head reshape+permute pattern
  // ([B, T, D] → [B, H, T, D/H]) in one call. `mergeHeads` is its inverse.
  const q = nn.splitHeads(p.q.fwd(x), N_HEADS)
  const k = nn.splitHeads(p.k.fwd(x), N_HEADS)
  const v = nn.splitHeads(p.v.fwd(x), N_HEADS)
  const scores = mul(matmul(q, swapAxes(k, -1, -2)), SCALE_QK)
  const attn = capture(`attn.${layerIdx}`, softmaxCausal(scores))
  return p.o.fwd(nn.mergeHeads(matmul(attn, v)))
}

function mlpFwd(p: MLP, x: Tensor): Tensor {
  return p.down.fwd(relu(p.up.fwd(x)))
}

function blockFwd(p: Block, x: Tensor, layerIdx: number): Tensor {
  const a = attentionFwd(p.attn, p.ln1.fwd(x), layerIdx)
  const x1 = add(x, a)
  return add(x1, mlpFwd(p.mlp, p.ln2.fwd(x1)))
}

function modelFwd(p: Transformer, tokens: Tensor): Tensor {
  const tokE = embedding(tokens, p.tok_emb)
  const posE = embedding(arange(T), p.pos_emb)
  let x = add(tokE, posE)
  for (let i = 0; i < p.layers.length; i++) {
    x = capture(`residual.${i}`, x)
    x = blockFwd(p.layers[i]!, x, i)
  }
  const xn = p.lnf.fwd(x)
  return matmul(xn, swapAxes(p.tok_emb, -1, -2))
}

function lossFn(p: Transformer, { tokens, targets, mask }: { tokens: Tensor; targets: Tensor; mask: Tensor }): Tensor {
  const ce = nn.crossEntropy(modelFwd(p, tokens), targets)   // [B, T] of -log p(target)
  return mul(sum(mul(ce, mask)), 1 / (B * N_RESULT_DIGITS))
}

function predictFwd(p: Transformer, { tokens }: { tokens: Tensor }): Tensor {
  return modelFwd(p, tokens)
}

// ---------- CPU-side: batch generation -------------------------------------

function isTestPair(a: number, b: number): boolean { return ((a * 100 + b) * 31 + 7) % 5 === 0 }

function makeBatch(): { tokens: Int32Array; targets: Int32Array } {
  const tokens = new Int32Array(B * T)
  const targets = new Int32Array(B * T)
  for (let bi = 0; bi < B; bi++) {
    let a, c
    do { a = Math.floor(Math.random() * 100); c = Math.floor(Math.random() * 100) } while (isTestPair(a, c))
    const sum = a + c
    const seq = [
      Math.floor(a / 10), a % 10, TOK_PLUS,
      Math.floor(c / 10), c % 10, TOK_EQ,
      sum % 10, Math.floor(sum / 10) % 10, Math.floor(sum / 100),
    ]
    for (let t = 0; t < T; t++) { tokens[bi * T + t] = seq[t]!; targets[bi * T + t] = seq[t + 1]! }
  }
  return { tokens, targets }
}

const RESULT_MASK = new Float32Array(T)
for (let t = RESULT_START - 1; t < T; t++) RESULT_MASK[t] = 1

// ---------- Logging UI ----------------------------------------------------

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
const origConsoleError = console.error.bind(console)
console.error = (...args: unknown[]) => {
  origConsoleError(...args)
  const text = args.map(a => typeof a === 'string' ? a : (a instanceof Error ? a.stack ?? a.message : JSON.stringify(a))).join(' ')
  log(`[console.error] ${text}`, 'err')
}

stopBtn.onclick = () => { stopRequested = true }

// ---------- Main ----------------------------------------------------------

async function run() {
  runBtn.disabled = true
  stopBtn.disabled = false
  stopRequested = false
  logEl.innerHTML = ''
  log('Building model + compiling...')
  const t0 = performance.now()
  const compiled = await compileModule({
    factory: () => new Transformer(),
    loss: lossFn,
    adam: { lr: LR, weightDecay: 0.01 },
    inputs: {
      tokens:  { shape: [B, T], dtype: 'i32' },
      targets: { shape: [B, T], dtype: 'i32' },
      mask:    [T],
    },
  })
  const compileMs = performance.now() - t0

  log(`  ${compiled.paramNames.length} params, ${compiled.kernelCount} kernels, compile ${compileMs.toFixed(0)} ms`, 'ok')

  log('Compiling inference graph (B=1)...')
  const tInfer = performance.now()
  const predict = await compiled.compileForward({
    forward: predictFwd,
    inputs: { tokens: { shape: [1, T], dtype: 'i32' } },
  })
  log(`  compile ${(performance.now() - tInfer).toFixed(0)} ms`, 'ok')

  // One-row test input for the per-100-step shape check.
  const inferTokens = new Int32Array(T)
  {
    const b0 = makeBatch()
    for (let t = 0; t < T; t++) inferTokens[t] = b0.tokens[t]!
  }

  // Held-out accuracy check: 3-step autoregressive prediction through the
  // inference graph on pairs the training set excludes (isTestPair === true).
  // Pad to T with 0s; causal masking makes positions ≥ realLen invisible to
  // the position we read from. Mirrors what the bulb's predictAddition does,
  // minus the per-head attn-map reshaping (sample is library-test, not viz).
  const tokensBuf = new Int32Array(T)
  async function predictPair(a: number, b: number): Promise<number[]> {
    const prefix = [Math.floor(a / 10), a % 10, TOK_PLUS, Math.floor(b / 10), b % 10, TOK_EQ]
    const generated: number[] = []
    for (let s = 0; s < N_RESULT_DIGITS; s++) {
      const realLen = prefix.length + generated.length
      tokensBuf.fill(0)
      for (let i = 0; i < prefix.length; i++) tokensBuf[i] = prefix[i]!
      for (let i = 0; i < generated.length; i++) tokensBuf[prefix.length + i] = generated[i]!
      const output = await predict.run({ tokens: tokensBuf })
      const lastStart = (realLen - 1) * VOCAB
      let best = 0
      let bestL = output[lastStart]!
      for (let v = 1; v < 10; v++) {
        const l = output[lastStart + v]!
        if (l > bestL) { bestL = l; best = v }
      }
      generated.push(best)
    }
    return generated
  }
  async function evalAccuracy(nPairs: number, useTestPairs: boolean): Promise<number> {
    let correct = 0
    let evaluated = 0
    let attempts = 0
    while (evaluated < nPairs && attempts < nPairs * 50) {
      attempts++
      const a = Math.floor(Math.random() * 100), b = Math.floor(Math.random() * 100)
      if (isTestPair(a, b) !== useTestPairs) continue
      const generated = await predictPair(a, b)
      const got = generated[0]! + generated[1]! * 10 + generated[2]! * 100
      if (got === a + b) correct++
      evaluated++
    }
    return evaluated === 0 ? 0 : correct / evaluated
  }

  // Worker-architecture smoke: exercise reset() (re-init main-thread →
  // uploadParams → resetOptimizer round-trips through the worker) before any
  // training, then a downloadParamGrads round-trip. Both should be silent;
  // any error here is a worker-protocol bug.
  log('Worker smoke: reset() + downloadParamGrads()...')
  await compiled.reset()
  const grads = await compiled.downloadParamGrads()
  const gradEntries = Object.keys(grads).length
  log(`  ${gradEntries} param grads (expect zeros after reset)`, 'ok')

  log('Training...')
  let step = 0
  let stepStart = performance.now()
  while (!stopRequested) {
    step++
    const { tokens, targets } = makeBatch()
    const lossVal = await compiled.step({ tokens, targets, mask: RESULT_MASK })
    if (step === 1 || step % 20 === 0) {
      const interval = step === 1 ? 1 : 20
      const dt = (performance.now() - stepStart) / Math.max(1, interval)
      stepStart = performance.now()
      const exPerSec = dt > 0 ? Math.round(B * 1000 / dt) : 0
      const examples = (step * B).toLocaleString()
      log(`  step ${step.toString().padStart(4)}  loss ${lossVal.toFixed(4)}  examples ${examples}  (${exPerSec.toLocaleString()} ex/s)`)
    }
    // Periodic inference checks: the cheap one runs every 100 steps to
    // exercise run() + withCaptures + sharedParams; the held-out accuracy
    // probe runs every 100 steps too, verifying autoregressive prediction.
    if (step === 1 || step % 100 === 0) {
      const { output, captures } = await predict.run({ tokens: inferTokens }, { withCaptures: true })
      const expectOutput = 1 * T * VOCAB
      const expectAttn = 1 * N_HEADS * T * T
      const attn0 = captures.get('attn.0')
      const captureOk = attn0.length === expectAttn
      const finiteOk = Number.isFinite(output[0])
      log(`  [infer] output=${output.length} (expect ${expectOutput}), captures.attn.0=${attn0.length} (expect ${expectAttn}), output[0]=${output[0]?.toFixed(4)}, ok=${captureOk && finiteOk}`)
      const trainAcc = await evalAccuracy(100, false)
      const testAcc = await evalAccuracy(100, true)
      // Sample one prediction so we can see what the model is actually saying.
      const sampleA = 23, sampleB = 45
      const samplePred = await predictPair(sampleA, sampleB)
      const sampleGot = samplePred[0]! + samplePred[1]! * 10 + samplePred[2]! * 100
      log(`  [acc] step ${step} train=${(trainAcc * 100).toFixed(1)}% test=${(testAcc * 100).toFixed(1)}%  sample ${sampleA}+${sampleB}=${sampleGot} (expect ${sampleA + sampleB})`)
    }
    if (step % 5 === 0) await new Promise(r => setTimeout(r, 0))
  }
  log(`Stopped at step ${step}.`, 'ok')
  predict.destroy()
  compiled.destroy()
  runBtn.disabled = false; stopBtn.disabled = true
}

runBtn.onclick = () => { run().catch(e => log(`error: ${e?.message ?? e}\n${e?.stack ?? ''}`, 'err')) }
