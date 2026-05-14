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
//
// File layout: ML + training above, UI below. The ML section exposes a small
// set of entry points (startTraining, stopTraining) and emits updates via the
// `onStatus` hook the UI registers at boot. DOM access lives entirely in the
// UI section.

import {
  isWebGPUAvailable,
  type CompiledTraining, type CompiledForward,
} from 'tensorgrad'
import {
  Transformer, lossFn, irSpec, compileTraining,
  B, T, VOCAB, SEQ_LEN,
} from './spec.ts'

// ========== MODEL / TRAINING ==========

const PAD = 0                                 // '.' = 0; 'a'..'z' = 1..26
const encodeChar = (ch: string) => ch === '.' ? PAD : (ch.charCodeAt(0) - 97 + 1)
const decodeChar = (idx: number) => idx === PAD ? '.' : String.fromCharCode(idx - 1 + 97)

// Model + loss + predict live in ./spec.ts so the IR viewer can import them
// without triggering this file's boot side effects.

// ---------- Sampling helpers (CPU-side, on raw logits from infer.run) -------

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

// ---------- Training lifecycle ----------------------------------------------

let train: CompiledTraining<Transformer> | null = null
let infer: CompiledForward<Transformer> | null = null
let valLossFwd: CompiledForward<Transformer> | null = null
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

  onStatus('Compiling inference + val-loss graphs...')
  const tInfer = performance.now()
  infer = await train.attach({
    forward: irSpec.predict,
    inputs: { tokens: { shape: [1, T], dtype: 'i32' } },
  })
  // Forward-only loss graph at full batch shape — feeds the periodic val probe.
  // Attached to `train`, so val loss reflects the latest training state.
  valLossFwd = await train.attach({
    forward: lossFn,
    inputs: {
      tokens:  { shape: [B, T], dtype: 'i32' },
      targets: { shape: [B, T], dtype: 'i32' },
      mask:    [B, T],
    },
  })
  onStatus(`  compile ${(performance.now() - tInfer).toFixed(0)} ms`, 'ok')
}

const tokensBuf = new Int32Array(T)
async function sampleName(): Promise<string> {
  if (!infer) return ''
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

async function startTraining(): Promise<void> {
  if (running) return
  running = true
  step = 0
  try {
    if (!isWebGPUAvailable()) {
      onStatus('WebGPU not available. Use Chrome 113+ or Safari 17.4+.', 'err')
      running = false
      return
    }
    if (TRAIN.length === 0) {
      onStatus('Loading names corpus...')
      const all = await loadNames()
      TRAIN = []; TEST = []
      for (const n of all) (isTestName(n) ? TEST : TRAIN).push(n)
      TRAIN_SET = new Set(TRAIN)
      const avgLen = all.reduce((s, n) => s + n.length, 0) / all.length
      onStatus(`  ${all.length.toLocaleString()} names loaded — ${TRAIN.length.toLocaleString()} train / ${TEST.length.toLocaleString()} val (avg length ${avgLen.toFixed(1)})`, 'ok')
    }
    if (!train || !infer || !valLossFwd) await buildGraphs()
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
  if (!train || !infer || !valLossFwd) return
  onStatus('Training...')
  let stepStart = performance.now()
  while (running) {
    step++
    const sr = await train.step(makeBatch(false))
    if (sr.kind === 'aborted') break
    const lossVal = sr.loss
    if (step === 1 || step % 20 === 0) {
      const interval = step === 1 ? 1 : 20
      const dt = (performance.now() - stepStart) / Math.max(1, interval)
      stepStart = performance.now()
      const exPerSec = dt > 0 ? Math.round(B * 1000 / dt) : 0
      onStatus(`  step ${step.toString().padStart(4)}  loss ${lossVal.toFixed(4)}  (${exPerSec.toLocaleString()} ex/s)`)
    }
    if (step === 1 || step % 100 === 0) {
      const vr = await valLossFwd.run(makeBatch(true))
      if (vr.kind === 'aborted') break
      const valLoss = vr.output[0]!
      const samples: string[] = []
      for (let i = 0; i < 8; i++) samples.push(await sampleName())
      const novel = samples.filter(s => !TRAIN_SET.has(s) && s.length > 0).length
      onStatus(`  [val@${step}] loss ${valLoss.toFixed(4)}   [samples] ${samples.join(', ')}   (novel ${novel}/8)`)
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
