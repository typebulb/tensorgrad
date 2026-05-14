// Transformer training in the browser, using tensorgrad's Module abstraction.
// Compare to the previous version of this file (or to transformer-jax.bulb.md):
// the model is now defined as nested classes; param names are auto-derived
// from property paths; no string-keyed dictionaries; no manual paramInput
// boilerplate. Forward functions stay pure — JAX-style separation of state
// and computation.
//
// File layout: ML + training above, UI below. The ML section exposes a small
// set of entry points (startTraining, stopTraining) and emits updates via the
// `onStatus` hook the UI registers at boot. DOM access lives entirely in the
// UI section.

import type { CompiledTraining, CompiledForward } from 'tensorgrad'
import {
  Transformer, irSpec, compileTraining,
  B, T, VOCAB, N_HEADS, RESULT_START, N_RESULT_DIGITS, TOK_PLUS, TOK_EQ,
} from './spec.ts'

// ========== MODEL / TRAINING ==========

// Model + loss + predict live in ./spec.ts so the IR viewer can import them
// without triggering this file's boot side effects.

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

// ---------- Training lifecycle ----------------------------------------------

let train: CompiledTraining<Transformer> | null = null
let infer: CompiledForward<Transformer> | null = null
let running = false
let step = 0

// UI-supplied sink; assigned in the UI section so the ML side has zero DOM
// dependencies. Default no-op lets this section behave in isolation.
let onStatus: (msg: string, cls?: 'err' | 'ok') => void = () => {}

async function buildGraphs(): Promise<void> {
  onStatus('Building model + compiling...')
  const t0 = performance.now()
  train = await compileTraining()
  onStatus(`  ${train.paramNames.length} params, ${train.kernels.length} kernels, compile ${(performance.now() - t0).toFixed(0)} ms`, 'ok')

  onStatus('Compiling inference graph (B=1)...')
  const tInfer = performance.now()
  infer = await train.attach({
    forward: irSpec.predict,
    inputs: { tokens: { shape: [1, T], dtype: 'i32' } },
  })
  onStatus(`  compile ${(performance.now() - tInfer).toFixed(0)} ms`, 'ok')
}

// Held-out accuracy check: 3-step autoregressive prediction through the
// inference graph on pairs the training set excludes (isTestPair === true).
// Pad to T with 0s; causal masking makes positions ≥ realLen invisible to
// the position we read from.
const tokensBuf = new Int32Array(T)
async function predictPair(a: number, b: number): Promise<number[]> {
  if (!infer) return []
  const prefix = [Math.floor(a / 10), a % 10, TOK_PLUS, Math.floor(b / 10), b % 10, TOK_EQ]
  const generated: number[] = []
  for (let s = 0; s < N_RESULT_DIGITS; s++) {
    const realLen = prefix.length + generated.length
    tokensBuf.fill(0)
    for (let i = 0; i < prefix.length; i++) tokensBuf[i] = prefix[i]!
    for (let i = 0; i < generated.length; i++) tokensBuf[prefix.length + i] = generated[i]!
    const r = await infer.run({ tokens: tokensBuf })
    if (r.kind === 'aborted') return generated
    const output = r.output
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

const inferTokens = new Int32Array(T)

async function startTraining(): Promise<void> {
  if (running) return
  running = true
  step = 0
  try {
    if (!train || !infer) await buildGraphs()
    if (!train || !infer) return

    // Seed inferTokens from a fresh batch so the per-100-step shape check has
    // a deterministic input to feed.
    const b0 = makeBatch()
    for (let t = 0; t < T; t++) inferTokens[t] = b0.tokens[t]!

    // Worker-architecture smoke: exercise reset() (re-init main-thread →
    // uploadParams → resetOptimizer round-trips through the worker) plus a
    // params download. Both should be silent; any error here is a worker-
    // protocol bug.
    onStatus('Worker smoke: reset() + downloadParams()...')
    await train.reset()
    const params = await train.downloadParams()
    onStatus(`  ${Object.keys(params).length} params re-initialized`, 'ok')

    await runTraining()
  } catch (e) {
    running = false
    onStatus(`error: ${(e as { message?: string })?.message ?? e}`, 'err')
    throw e
  }
}

function stopTraining(): void {
  running = false
}

async function runTraining(): Promise<void> {
  if (!train || !infer) return
  onStatus('Training...')
  let stepStart = performance.now()
  while (running) {
    step++
    const sr = await train.step({ ...makeBatch(), mask: RESULT_MASK })
    if (sr.kind === 'aborted') break
    const lossVal = sr.loss
    if (step === 1 || step % 20 === 0) {
      const interval = step === 1 ? 1 : 20
      const dt = (performance.now() - stepStart) / Math.max(1, interval)
      stepStart = performance.now()
      const exPerSec = dt > 0 ? Math.round(B * 1000 / dt) : 0
      const examples = (step * B).toLocaleString()
      onStatus(`  step ${step.toString().padStart(4)}  loss ${lossVal.toFixed(4)}  examples ${examples}  (${exPerSec.toLocaleString()} ex/s)`)
    }
    // Periodic inference checks: cheap shape/capture check + held-out accuracy
    // probe (autoregressive prediction on excluded test pairs).
    if (step === 1 || step % 100 === 0) {
      const rr = await infer.run({ tokens: inferTokens })
      if (rr.kind === 'aborted') break
      const { output, captures } = rr
      const expectOutput = 1 * T * VOCAB
      const expectAttn = 1 * N_HEADS * T * T
      const attn0 = captures.get('attn.0')
      const captureOk = attn0.length === expectAttn
      const finiteOk = Number.isFinite(output[0])
      onStatus(`  [infer] output=${output.length} (expect ${expectOutput}), captures.attn.0=${attn0.length} (expect ${expectAttn}), output[0]=${output[0]?.toFixed(4)}, ok=${captureOk && finiteOk}`)
      const trainAcc = await evalAccuracy(100, false)
      const testAcc = await evalAccuracy(100, true)
      const sampleA = 23, sampleB = 45
      const samplePred = await predictPair(sampleA, sampleB)
      const sampleGot = samplePred[0]! + samplePred[1]! * 10 + samplePred[2]! * 100
      onStatus(`  [acc] step ${step} train=${(trainAcc * 100).toFixed(1)}% test=${(testAcc * 100).toFixed(1)}%  sample ${sampleA}+${sampleB}=${sampleGot} (expect ${sampleA + sampleB})`)
    }
    if (step % 5 === 0) await new Promise(r => setTimeout(r, 0))
  }
  onStatus(`Stopped at step ${step}.`, 'ok')
}

// ========== UI ==========

const logEl = document.getElementById('log')!
const runBtn = document.getElementById('run') as HTMLButtonElement
const stopBtn = document.getElementById('stop') as HTMLButtonElement

function log(msg: string, cls?: 'err' | 'ok'): void {
  const line = document.createElement('div')
  if (cls) line.className = cls
  line.textContent = msg
  logEl.appendChild(line)
  logEl.scrollTop = logEl.scrollHeight
  console.log(msg)
  fetch('/__log', { method: 'POST', body: JSON.stringify({ msg }) }).catch(() => {})
}

onStatus = log

window.addEventListener('error', e => log(`[error] ${e.message}`, 'err'))
window.addEventListener('unhandledrejection', e => log(`[promise] ${String((e as any).reason?.message ?? (e as any).reason)}`, 'err'))
const origConsoleError = console.error.bind(console)
console.error = (...args: unknown[]) => {
  origConsoleError(...args)
  const text = args.map(a => typeof a === 'string' ? a : (a instanceof Error ? a.stack ?? a.message : JSON.stringify(a))).join(' ')
  log(`[console.error] ${text}`, 'err')
}

runBtn.addEventListener('click', () => {
  runBtn.disabled = true
  stopBtn.disabled = false
  logEl.innerHTML = ''
  startTraining().finally(() => {
    runBtn.disabled = false
    stopBtn.disabled = true
  })
})

stopBtn.addEventListener('click', () => { stopTraining() })
