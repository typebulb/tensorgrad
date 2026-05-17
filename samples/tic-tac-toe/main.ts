// Tic-tac-toe self-play. A small policy MLP plays itself; the gradient
// signal is the terminal outcome of each game (+1 / 0 / -1 from each player's
// perspective), assigned to every move that player made. After enough
// rollouts the network learns optimal tic-tac-toe — every game vs another
// optimal player draws.
//
// The fun part: you can play the model while it trains. It starts random
// and gets harder. Once it's converged, every game ends in a draw assuming
// you play optimally too. Beating the model means the model isn't done
// learning yet.
//
// State encoding: current-player perspective, 3 binary channels × 9 cells.
// Channel 0 = empty, channel 1 = my pieces, channel 2 = opponent pieces.
// The same network plays both X and O — symmetry handled by the encoding.
//
// Illegal moves: masked out before sampling. Training also masks out moves
// past the actual end of each game (games end after 5–9 plies, but the
// rollout tensor has a fixed 9-ply length).
//
// File layout: ML + app logic at the top, UI at the bottom. Same convention
// as the other samples.

import {
  isWebGPUAvailable,
  type CompiledTraining, type CompiledForward,
} from 'tensorgrad'
import {
  Policy, irSpec, compileTraining,
  K, MAX_MOVES, N_SLOTS, STATE_DIM,
} from './spec.ts'

// ========== MODEL / TRAINING ==========

// ---------------------------------------------------------------------------
// Game engine. Pure CPU, board stored as `cells[0..8]` where 0 = empty,
// 1 = player 0, 2 = player 1.
// ---------------------------------------------------------------------------

// 8 winning triples (3 rows + 3 cols + 2 diagonals).
const LINES: readonly [number, number, number][] = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6],
]

interface Game {
  cells: Int8Array          // 0 / 1 / 2
  turn: 0 | 1
  done: boolean
  /** -1 = draw, 0 = player 0 won, 1 = player 1 won. Only valid when `done`. */
  winner: -1 | 0 | 1
}

function newGame(): Game {
  return { cells: new Int8Array(9), turn: 0, done: false, winner: -1 }
}

function checkWin(cells: Int8Array, p: 1 | 2): boolean {
  for (const [a, b, c] of LINES) {
    if (cells[a] === p && cells[b] === p && cells[c] === p) return true
  }
  return false
}

function applyMove(g: Game, cell: number): Game {
  if (g.done) return g
  const cells = new Int8Array(g.cells)
  const piece: 1 | 2 = g.turn === 0 ? 1 : 2
  cells[cell] = piece
  let done = false
  let winner: -1 | 0 | 1 = -1
  if (checkWin(cells, piece)) {
    done = true
    winner = g.turn
  } else if (cells.every(c => c !== 0)) {
    done = true
    winner = -1
  }
  return { cells, turn: (g.turn ^ 1) as 0 | 1, done, winner }
}

// Pack game state from `g.turn`'s perspective into 27 binary features.
function packState(g: Game, out: Float32Array, offset: number): void {
  const me: 1 | 2 = g.turn === 0 ? 1 : 2
  for (let i = 0; i < 9; i++) {
    out[offset + i]      = g.cells[i] === 0  ? 1 : 0     // empty
    out[offset + 9 + i]  = g.cells[i] === me ? 1 : 0     // mine
    out[offset + 18 + i] = (g.cells[i] !== 0 && g.cells[i] !== me) ? 1 : 0  // theirs
  }
}

// Model + loss + predict live in ./spec.ts.

// ---------------------------------------------------------------------------
// State + lifecycle
// ---------------------------------------------------------------------------

let train: CompiledTraining<Policy> | null = null
let infer: CompiledForward<Policy> | null = null
let running = false
let rolloutCount = 0
let lastWinRate = 0

let onStatus: (msg: string) => void = () => {}
let onPolicyUpdated: () => void = () => {}  // fires after every train step

// Sample an action from probabilities, restricted to legal cells.
function sampleMaskedAction(probs: Float32Array, offset: number, legal: Int8Array): number {
  let total = 0
  for (let i = 0; i < 9; i++) if (legal[i] === 0) total += probs[offset + i]!
  if (total < 1e-12) {
    // Numerical degeneracy — fall back to uniform over legal moves.
    const empties: number[] = []
    for (let i = 0; i < 9; i++) if (legal[i] === 0) empties.push(i)
    return empties[Math.floor(Math.random() * empties.length)]!
  }
  let u = Math.random() * total
  for (let i = 0; i < 9; i++) {
    if (legal[i] !== 0) continue
    u -= probs[offset + i]!
    if (u < 0) return i
  }
  // Floating-point fallback.
  for (let i = 8; i >= 0; i--) if (legal[i] === 0) return i
  return 0
}

// One K-parallel self-play rollout.
async function rollout(): Promise<{
  states: Float32Array; actions: Int32Array; outcomes: Float32Array; mask: Float32Array
}> {
  if (!infer)throw new Error('rollout: no inference graph')
  const games: Game[] = Array.from({ length: K }, newGame)
  const statesAll  = new Float32Array(N_SLOTS * STATE_DIM)
  const actionsAll = new Int32Array(N_SLOTS)
  const playerAll  = new Int8Array(N_SLOTS)
  const maskAll    = new Float32Array(N_SLOTS)
  const outcomesAll = new Float32Array(N_SLOTS)
  const stateBatch = new Float32Array(K * STATE_DIM)

  for (let move = 0; move < MAX_MOVES; move++) {
    // Pack states for live games. Done games contribute zero gradient
    // (mask = 0 keeps their slots from leaking signal).
    for (let k = 0; k < K; k++) {
      if (!games[k]!.done) packState(games[k]!, stateBatch, k * STATE_DIM)
    }
    const probsR = await infer.run({ state: stateBatch })  // [K, 9]
    if (probsR.kind !== 'completed') break
    const probs = probsR.output

    for (let k = 0; k < K; k++) {
      const g = games[k]!
      if (g.done) continue
      const idx = move * K + k
      // Legality mask: 0 = legal (empty), 1 = illegal.
      const legal = new Int8Array(9)
      for (let i = 0; i < 9; i++) legal[i] = g.cells[i] === 0 ? 0 : 1
      const a = sampleMaskedAction(probs, k * 9, legal)
      statesAll.set(stateBatch.subarray(k * STATE_DIM, (k + 1) * STATE_DIM), idx * STATE_DIM)
      actionsAll[idx] = a
      playerAll[idx]  = g.turn
      maskAll[idx]    = 1
      games[k] = applyMove(g, a)
    }
  }

  // Assign per-move outcomes from each player's perspective. winner=-1 means
  // a draw (zero return); player==winner gets +1, the loser gets -1.
  for (let k = 0; k < K; k++) {
    const w = games[k]!.winner
    for (let move = 0; move < MAX_MOVES; move++) {
      const idx = move * K + k
      if (maskAll[idx] === 0) continue
      if (w === -1) outcomesAll[idx] = 0
      else outcomesAll[idx] = playerAll[idx] === w ? 1 : -1
    }
  }

  // Normalize outcomes over unmasked entries for variance reduction.
  let count = 0, s1 = 0
  for (let i = 0; i < N_SLOTS; i++) if (maskAll[i]) { s1 += outcomesAll[i]!; count++ }
  const meanO = count > 0 ? s1 / count : 0
  let vs = 0
  for (let i = 0; i < N_SLOTS; i++) if (maskAll[i]) vs += (outcomesAll[i]! - meanO) ** 2
  const stdO = Math.sqrt(vs / Math.max(1, count)) + 1e-8
  for (let i = 0; i < N_SLOTS; i++) if (maskAll[i]) outcomesAll[i] = (outcomesAll[i]! - meanO) / stdO

  return { states: statesAll, actions: actionsAll, outcomes: outcomesAll, mask: maskAll }
}

// Evaluate current policy vs random opponent. Plays NEVAL games as each side
// and reports overall win+draw rate from the trained model's perspective.
async function evalVsRandom(): Promise<number> {
  if (!infer)return 0
  const NEVAL = 30
  let wins = 0, draws = 0
  const stateBatch = new Float32Array(STATE_DIM)
  for (let gi = 0; gi < NEVAL * 2; gi++) {
    let g = newGame()
    const modelIsPlayer: 0 | 1 = gi < NEVAL ? 0 : 1
    while (!g.done) {
      if (g.turn === modelIsPlayer) {
        packState(g, stateBatch, 0)
        const pr = await infer.run({ state: stateBatch })
        if (pr.kind !== 'completed') return 0
        const probs = pr.output
        const legal = new Int8Array(9)
        for (let i = 0; i < 9; i++) legal[i] = g.cells[i] === 0 ? 0 : 1
        // Greedy at eval time.
        let bestIdx = -1, bestVal = -Infinity
        for (let i = 0; i < 9; i++) {
          if (legal[i] !== 0) continue
          if (probs[i]! > bestVal) { bestVal = probs[i]!; bestIdx = i }
        }
        g = applyMove(g, bestIdx)
      } else {
        const empties: number[] = []
        for (let i = 0; i < 9; i++) if (g.cells[i] === 0) empties.push(i)
        g = applyMove(g, empties[Math.floor(Math.random() * empties.length)]!)
      }
    }
    if (g.winner === modelIsPlayer) wins++
    else if (g.winner === -1) draws++
  }
  return (wins + draws) / (NEVAL * 2)
}

async function runTraining(): Promise<void> {
  while (running && train) {
    const r = await rollout()
    const sr = await train.step({
      states: r.states, actions: r.actions, outcomes: r.outcomes, mask: r.mask,
    })
    if (sr.kind !== 'completed') return
    const loss = sr.loss
    if (!Number.isFinite(loss)) {
      onStatus(`rollout ${rolloutCount}: loss is ${loss} — NaN, aborting.`)
      running = false
      return
    }
    rolloutCount++
    onPolicyUpdated()
    if (rolloutCount % 10 === 0) {
      lastWinRate = await evalVsRandom()
    }
    onStatus(`rollout ${rolloutCount}  loss ${loss.toFixed(4)}  win+draw vs random ${(lastWinRate * 100).toFixed(1)}%`)
    await new Promise(r => setTimeout(r, 0))
  }
}

// Pick the model's best legal move from the given game state.
async function modelMove(g: Game): Promise<number> {
  if (!infer)throw new Error('modelMove: no inference graph')
  const stateBuf = new Float32Array(STATE_DIM)
  packState(g, stateBuf, 0)
  const pr = await infer.run({ state: stateBuf })
  if (pr.kind !== 'completed') return -1
  const probs = pr.output
  let bestIdx = -1, bestVal = -Infinity
  for (let i = 0; i < 9; i++) {
    if (g.cells[i] !== 0) continue
    if (probs[i]! > bestVal) { bestVal = probs[i]!; bestIdx = i }
  }
  return bestIdx
}

async function buildGraphs(): Promise<void> {
  onStatus('compiling…')
  const t0 = performance.now()
  train = await compileTraining()
  // Polymorphic batch dim: K=16 during rollouts, B=1 for human-vs-AI moves.
  infer = await train.attach({
    forward: irSpec.predict,
    inputs: { state: [null, STATE_DIM] },
  })
  rolloutCount = 0
  lastWinRate = 0
  onStatus(`compiled (${train.kernels.length} kernels, ${(performance.now() - t0).toFixed(0)} ms)`)
}

function startTraining(): void {
  if (running) return
  running = true
  void runTraining()
}

function stopTraining(): void {
  running = false
}

async function resetWeights(): Promise<void> {
  if (!train) return
  const wasRunning = running
  running = false
  await new Promise<void>(r => setTimeout(r, 0))
  await train.reset()
  rolloutCount = 0
  lastWinRate = 0
  onStatus(`weights re-initialized (seed ${train.seed})`)
  if (wasRunning) { running = true; void runTraining() }
}

// ========== UI ==========

const statusEl     = document.getElementById('status')   as HTMLDivElement
const trainBtn     = document.getElementById('train')    as HTMLButtonElement
const stopBtn      = document.getElementById('stop')     as HTMLButtonElement
const resetBtn     = document.getElementById('reset')    as HTMLButtonElement
const newGameBtn   = document.getElementById('new-game') as HTMLButtonElement
const sideSelect   = document.getElementById('side')     as HTMLSelectElement
const boardEl      = document.getElementById('board')    as HTMLDivElement
const verdictEl    = document.getElementById('verdict')  as HTMLDivElement

const cellEls: HTMLDivElement[] = []
for (let i = 0; i < 9; i++) {
  const c = document.createElement('div')
  c.className = 'cell'
  c.dataset.index = String(i)
  c.addEventListener('click', () => { void handleCellClick(i) })
  boardEl.appendChild(c)
  cellEls.push(c)
}

let humanGame: Game = newGame()
let humanPlayer: 0 | 1 = 0   // 0 = X (moves first), 1 = O

function renderBoard(): void {
  for (let i = 0; i < 9; i++) {
    const v = humanGame.cells[i]
    cellEls[i]!.textContent = v === 1 ? 'X' : v === 2 ? 'O' : ''
    cellEls[i]!.classList.toggle('filled', v !== 0)
  }
  if (humanGame.done) {
    if (humanGame.winner === -1) verdictEl.textContent = 'draw'
    else if (humanGame.winner === humanPlayer) verdictEl.textContent = 'you win'
    else verdictEl.textContent = 'model wins'
  } else if (humanGame.turn === humanPlayer) {
    verdictEl.textContent = 'your move'
  } else {
    verdictEl.textContent = 'model thinking…'
  }
}

async function handleCellClick(idx: number): Promise<void> {
  if (humanGame.done) return
  if (humanGame.turn !== humanPlayer) return
  if (humanGame.cells[idx] !== 0) return
  humanGame = applyMove(humanGame, idx)
  renderBoard()
  if (!humanGame.done) await modelRespond()
}

async function modelRespond(): Promise<void> {
  const m = await modelMove(humanGame)
  humanGame = applyMove(humanGame, m)
  renderBoard()
}

async function startNewGame(): Promise<void> {
  humanGame = newGame()
  humanPlayer = sideSelect.value === 'O' ? 1 : 0
  renderBoard()
  // If model goes first, get its move now.
  if (humanGame.turn !== humanPlayer && infer) await modelRespond()
}

onStatus = (msg) => { statusEl.textContent = msg }
onPolicyUpdated = () => { /* hook for future "model just changed" effects */ }

trainBtn.addEventListener('click', () => {
  trainBtn.disabled = true
  stopBtn.disabled = false
  startTraining()
})

stopBtn.addEventListener('click', () => {
  stopTraining()
  trainBtn.disabled = false
  stopBtn.disabled = true
})

resetBtn.addEventListener('click', () => { void resetWeights() })
newGameBtn.addEventListener('click', () => { void startNewGame() })
sideSelect.addEventListener('change', () => { void startNewGame() })

renderBoard()

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  if (!isWebGPUAvailable()) {
    onStatus('WebGPU not available. Try Chrome 113+ or Safari 17.4+.')
    return
  }
  await buildGraphs()
  trainBtn.disabled = false
  resetBtn.disabled = false
  newGameBtn.disabled = false
  await startNewGame()
}

boot().catch((e: unknown) => {
  const msg = (e as { message?: string })?.message ?? String(e)
  onStatus(`error: ${msg}`)
  console.error(e)
})
