// Pokéformer sample. Tiny Llama-style transformer learns to invent new
// Pokémon names from PokéAPI's ~1024-name corpus. Doubles as the canonical
// runtime smoke for in-graph stochastic sampling + i32 readback: the bugs
// in 0.1.8 (worker prng) and 0.1.9 (output dtype) would both have been
// caught by running this sample against the live workspace tensorgrad.

import {
  isWebGPUAvailable,
  type CompiledTraining, type CompiledForward,
} from 'tensorgrad'
import {
  NameLM, irSpec, compileTraining,
  B, T, VOCAB, EOS, B_INFER,
} from './spec.ts'

// ========== DATA ==========

const FALLBACK_NAMES: readonly string[] = [
  'pikachu', 'charizard', 'bulbasaur', 'squirtle', 'mewtwo', 'mew',
  'articuno', 'zapdos', 'moltres', 'lugia', 'hooh', 'rayquaza',
  'kyogre', 'groudon', 'dialga', 'palkia', 'giratina', 'arceus',
  'eevee', 'vaporeon', 'jolteon', 'flareon', 'espeon', 'umbreon',
  'leafeon', 'glaceon', 'sylveon', 'snorlax', 'gengar', 'alakazam',
  'machamp', 'lucario', 'garchomp', 'dragonite', 'tyranitar', 'metagross',
  'salamence', 'blaziken', 'swampert', 'sceptile',
]

function cleanCorpus(raw: readonly string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const name of raw) {
    const cleaned = name.toLowerCase().normalize('NFD').replace(/[^a-z]/g, '')
    if (cleaned.length < 2 || cleaned.length > T - 2) continue
    if (seen.has(cleaned)) continue
    seen.add(cleaned)
    out.push(cleaned)
  }
  return out
}

let NAMES: string[] = cleanCorpus(FALLBACK_NAMES)
let NAMES_SET = new Set(NAMES)
let corpusSource: 'pokeapi' | 'fallback' = 'fallback'

async function loadCorpus(): Promise<void> {
  try {
    const r = await fetch('https://pokeapi.co/api/v2/pokemon-species?limit=1025')
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    const data = await r.json() as { results: { name: string }[] }
    const cleaned = cleanCorpus(data.results.map(x => x.name))
    if (cleaned.length < 100) throw new Error(`only ${cleaned.length} usable names`)
    NAMES = cleaned
    NAMES_SET = new Set(NAMES)
    corpusSource = 'pokeapi'
  } catch (e) {
    console.warn('PokéAPI fetch failed, using fallback corpus:', e)
  }
}

const charToId = (ch: string) => ch.charCodeAt(0) - 97
const idToChar = (id: number) => String.fromCharCode(97 + id)

// Training layout: [EOS, c0, c1, ..., c_{n-1}, EOS, EOS, ...]
// Input = positions 0..T-1, target = positions 1..T.
// Mask = 1 for positions 0..nameLength (BOS→c0, c_i→c_{i+1}, c_last→EOS); 0
// for post-EOS padding (trivial EOS→EOS predictions). Mask normalized so
// sum(mask) === 1 per batch, making the masked sum a per-valid-token mean.
function encodeName(name: string): { tokens: number[]; targets: number[]; validCount: number } {
  const seq: number[] = [EOS]
  for (const ch of name) seq.push(charToId(ch))
  while (seq.length < T + 1) seq.push(EOS)
  return {
    tokens:  seq.slice(0, T),
    targets: seq.slice(1, T + 1),
    validCount: Math.min(name.length + 1, T),  // BOS→c0 through c_last→EOS
  }
}

function makeBatch(): { tokens: Int32Array; targets: Int32Array; mask: Float32Array } {
  const tokens  = new Int32Array(B * T)
  const targets = new Int32Array(B * T)
  const mask    = new Float32Array(B * T)
  let totalValid = 0
  for (let i = 0; i < B; i++) {
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

// ========== TRAINING LIFECYCLE ==========

let train: CompiledTraining<NameLM> | null = null
let infer: CompiledForward<NameLM, typeof irSpec.predictInputs, 'i32'> | null = null
let running = false
let step = 0

let onStatus: (msg: string, cls?: 'err' | 'ok') => void = () => {}

async function buildGraphs(): Promise<void> {
  onStatus('Building model + compiling...')
  const t0 = performance.now()
  train = await compileTraining()
  onStatus(`  ${train.paramNames.length} params, ${train.kernels.length} kernels, compile ${(performance.now() - t0).toFixed(0)} ms`, 'ok')

  onStatus('Attaching inference graph (categorical baked in)...')
  const tInfer = performance.now()
  infer = await train.attach({
    forward: irSpec.predict,
    inputs: irSpec.predictInputs,
    output: irSpec.predictOutput,
  })
  onStatus(`  compile ${(performance.now() - tInfer).toFixed(0)} ms`, 'ok')
}

// Autoregressive sampling. Each `infer.run` returns Int32Array sampled
// tokens [B_INFER, T] from the in-graph categorical. We extract the sample
// at position (p - 1), write it into position p, loop.
const tokensBuf = new Int32Array(B_INFER * T)
const tempBuf = new Float32Array(1)
async function sampleName(temperature: number): Promise<string> {
  if (!infer) return ''
  tokensBuf.fill(EOS)
  tempBuf[0] = Math.max(0.05, temperature)
  let p = 1
  while (p < T) {
    const r = await infer.run({ tokens: tokensBuf, temperature: tempBuf })
    if (r.kind !== 'completed') return decodeFrom(tokensBuf, p)
    // r.output is Int32Array — declared on the spec via `output: 'i32'`.
    const nextId = r.output[p - 1]!
    if (nextId === EOS) break
    if (nextId < 0 || nextId >= 26) break
    tokensBuf[p] = nextId
    p++
  }
  return decodeFrom(tokensBuf, p)
}

function decodeFrom(tokens: Int32Array, lengthFilled: number): string {
  let name = ''
  for (let t = 1; t < lengthFilled; t++) {
    const id = tokens[t]!
    if (id === EOS) break
    if (id >= 0 && id < 26) name += idToChar(id)
  }
  return name
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
    onStatus('Loading Pokémon corpus...')
    await loadCorpus()
    onStatus(`  ${NAMES.length} names (${corpusSource})`, 'ok')
    if (!train || !infer) await buildGraphs()
    await runTraining()
  } catch (e) {
    running = false
    onStatus(`error: ${(e as { message?: string })?.message ?? e}`, 'err')
    throw e
  }
}

function stopTraining(): void { running = false }

async function runTraining(): Promise<void> {
  if (!train || !infer) return
  onStatus('Training...')
  let stepStart = performance.now()
  while (running) {
    step++
    const sr = await train.step(makeBatch())
    if (sr.kind !== 'completed') break
    const lossVal = sr.loss
    if (step === 1 || step % 20 === 0) {
      const interval = step === 1 ? 1 : 20
      const dt = (performance.now() - stepStart) / Math.max(1, interval)
      stepStart = performance.now()
      const exPerSec = dt > 0 ? Math.round(B * 1000 / dt) : 0
      onStatus(`  step ${step.toString().padStart(4)}  loss ${lossVal.toFixed(4)}  (${exPerSec.toLocaleString()} ex/s)`)
    }
    if (step === 1 || step % 100 === 0) {
      const samples: string[] = []
      for (let i = 0; i < 8; i++) samples.push(await sampleName(0.8))
      const novel = samples.filter(s => !NAMES_SET.has(s) && s.length > 0).length
      onStatus(`  [samples@${step}] ${samples.join(', ')}   (novel ${novel}/8)`)
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
