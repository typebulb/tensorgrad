---
format: typebulb/v1
name: KataGo
---

**code.tsx**

```tsx
import {
  Module, Conv2d, compileForward, checkWebGPU,
  add, mul, matmul, relu, reshape, concat, mean, maxPool2d,
  type Tensor,
} from 'tensorgrad'
import {
  App, Component, div, section, h1, p, span, strong, em,
  button, label, canvas, inputSelect,
} from 'domeleon'

// ==========================================================================
// Go tactics
// ==========================================================================

// The two Go algorithms KataGo feeds its net as inputs, because a 6-block
// convnet cannot derive either one: ladder search (spatial features 14-17) and
// Benson's unconditional life (features 18,19).
//
// Ported from lightvector/KataGo cpp/game/board.cpp —
//   Board::searchIsLadderCaptured / searchIsLadderCapturedAttackerFirst2Libs
//   Board::calculateArea / calculateAreaForPla   (https://senseis.xmp.net/?BensonsAlgorithm)
//
// Fidelity matters more than elegance here: the net was trained against these
// exact outputs, including their approximations. Where KataGo gives up early or
// answers conservatively, so must we, or the planes disagree in precisely the
// tactical positions they exist to describe.

const EMPTY = 0
const BLACK = 1
const WHITE = -1

const MAX_LADDER_SEARCH_NODE_BUDGET = 25000   // board.cpp, same constant

type Color = number

// Orthogonal neighbours of a point on a `size` x `size` board.
function neighbors(loc: number, size: number): number[] {
  const y = (loc / size) | 0, x = loc % size
  const out: number[] = []
  if (y > 0) out.push(loc - size)
  if (y < size - 1) out.push(loc + size)
  if (x > 0) out.push(loc - 1)
  if (x < size - 1) out.push(loc + 1)
  return out
}

interface Group { stones: number[]; liberties: number[] }

// The chain at `start` and its liberties; both empty for an empty point.
function group(board: Int8Array, size: number, start: number): Group {
  const color = board[start]!
  const stones: number[] = []
  const liberties: number[] = []
  if (color === EMPTY) return { stones, liberties }
  const seen = new Set<number>([start])
  const libSeen = new Set<number>()
  const stack = [start]
  while (stack.length) {
    const cur = stack.pop()!
    stones.push(cur)
    for (const n of neighbors(cur, size)) {
      const c = board[n]!
      if (c === EMPTY) {
        if (!libSeen.has(n)) { libSeen.add(n); liberties.push(n) }
      } else if (c === color && !seen.has(n)) {
        seen.add(n); stack.push(n)
      }
    }
  }
  return { stones, liberties }
}

// Simple ko: exactly one stone captured by a lone stone left with exactly one
// liberty — the only shape whose recapture would immediately repeat the position.
const koPoint = (placed: Group, captured: number[]): number =>
  captured.length === 1 && placed.stones.length === 1 && placed.liberties.length === 1
    ? captured[0]! : -1

// ---------------------------------------------------------------------------
//  A mutable board supporting play/undo, for the ladder search
// ---------------------------------------------------------------------------

interface PlayRecord {
  loc: number
  captured: number[]
  prevKo: number
}

class TacticalBoard {
  readonly size: number
  readonly colors: Int8Array
  koLoc: number

  constructor(colors: Int8Array, size: number, koLoc = -1) {
    this.size = size
    this.colors = colors.slice()
    this.koLoc = koLoc
  }

  chain(loc: number): Group { return group(this.colors, this.size, loc) }

  numLiberties(loc: number): number { return this.chain(loc).liberties.length }

  // Empty neighbours of a point (KataGo's getNumImmediateLiberties).
  immediateLiberties(loc: number): number {
    let n = 0
    for (const a of neighbors(loc, this.size)) if (this.colors[a] === EMPTY) n++
    return n
  }

  // Play without legality checks beyond suicide; returns null if the move is
  // suicidal (and therefore not a candidate).
  play(loc: number, pla: Color): PlayRecord | null {
    const rec: PlayRecord = { loc, captured: [], prevKo: this.koLoc }
    this.colors[loc] = pla
    for (const n of neighbors(loc, this.size)) {
      if (this.colors[n] !== -pla) continue
      const c = this.chain(n)
      if (c.liberties.length > 0) continue
      for (const s of c.stones) { this.colors[s] = EMPTY; rec.captured.push(s) }
    }
    const placed = this.chain(loc)
    if (placed.liberties.length === 0) {
      // Undo the suicide before reporting it.
      for (const s of rec.captured) this.colors[s] = -pla
      this.colors[loc] = EMPTY
      return null
    }
    this.koLoc = koPoint(placed, rec.captured)
    return rec
  }

  undo(rec: PlayRecord): void {
    const pla = this.colors[rec.loc]!
    this.colors[rec.loc] = EMPTY
    for (const s of rec.captured) this.colors[s] = -pla
    this.koLoc = rec.prevKo
  }

  // Capturing moves that would gain the chain at loc liberties: precisely the
  // lone liberties of adjacent opponent chains in atari.
  libertyGainingCaptures(loc: number): number[] {
    const opp = -this.colors[loc]!
    const out: number[] = []
    const done = new Set<number>()
    for (const stone of this.chain(loc).stones) {
      for (const a of neighbors(stone, this.size)) {
        if (this.colors[a] !== opp || done.has(a)) continue
        const c = this.chain(a)
        if (c.liberties.length !== 1) continue
        for (const s of c.stones) done.add(s)
        for (const lib of c.liberties) if (!out.includes(lib)) out.push(lib)
      }
    }
    return out
  }

  // Move-ordering heuristic: for each adjacent friendly chain, libs*2 - 3.
  heuristicConnectionLibertiesX2(loc: number, pla: Color): number {
    let total = 0
    for (const a of neighbors(loc, this.size)) {
      if (this.colors[a] !== pla) continue
      const libs = this.numLiberties(a)
      if (libs > 1) total += libs * 2 - 3
    }
    return total
  }
}

const isAdjacent = (a: number, b: number, size: number): boolean => {
  const d = Math.abs(a - b)
  if (d === size) return true
  return d === 1 && ((a / size) | 0) === ((b / size) | 0)
}

// ---------------------------------------------------------------------------
//  Ladder search
// ---------------------------------------------------------------------------

interface LadderState { nodes: number; exhausted: boolean }

// Is the chain at `loc` captured, assuming both sides play the ladder out?
// `defenderFirst` = the defender moves next (used when the group is in atari).
function ladderSearch(b: TacticalBoard, loc: number, isDefender: boolean, st: LadderState): boolean {
  if (st.exhausted) return false
  if (st.nodes++ >= MAX_LADDER_SEARCH_NODE_BUDGET) { st.exhausted = true; return false }

  const pla = b.colors[loc]!            // defender's colour
  const opp = -pla
  const chain = b.chain(loc)
  const libs = chain.liberties.length

  // Base cases, straight from board.cpp.
  if (!isDefender && libs <= 1) return true       // attacker to move, already atari -> dead
  if (!isDefender && libs >= 3) return false      // attacker to move, too many libs -> alive
  if (isDefender && libs >= 2) return false       // defender got out
  // A defender facing a ko point is treated as escaping: KataGo deliberately
  // refuses to claim ladders that depend on ko, and this also rules out loops.
  if (isDefender && b.koLoc >= 0) return false

  let moves: number[]
  if (isDefender) {
    moves = [...b.libertyGainingCaptures(loc), ...chain.liberties]
  } else {
    moves = [...chain.liberties]
    if (moves.length !== 2) return false          // caller guarantees 2 libs here
    const l0 = b.immediateLiberties(moves[0]!)
    const l1 = b.immediateLiberties(moves[1]!)

    // If the two liberties aren't adjacent, filling one can't reduce the other,
    // so an escape giving 3+ liberties settles that branch immediately.
    if (!isAdjacent(moves[0]!, moves[1]!, b.size)) {
      // l0 is how many liberties the DEFENDER would gain by escaping at
      // liberty 0. If that escape wins, the attacker is forced to occupy
      // liberty 0 itself — keep that move, not the other one.
      if (l0 >= 3 && l1 >= 3) return false
      else if (l0 >= 3) moves = [moves[0]!]
      else if (l1 >= 3) moves = [moves[1]!]
    }
    // Order by liberties gained plus potential connections. With a finite node
    // budget, ordering can change the answer, so it is not merely a speedup.
    if (moves.length > 1) {
      const s0 = l0 * 2 + b.heuristicConnectionLibertiesX2(moves[0]!, pla)
      const s1 = l1 * 2 + b.heuristicConnectionLibertiesX2(moves[1]!, pla)
      if (s1 > s0) moves = [moves[1]!, moves[0]!]
    }
  }

  for (const move of moves) {
    if (b.colors[move] !== EMPTY) continue
    if (move === b.koLoc) continue
    const rec = b.play(move, isDefender ? pla : opp)
    if (!rec) continue                            // suicide, not a candidate
    // The defender's stone may have been captured outright.
    const gone = b.colors[loc] !== pla
    const captured = gone ? true : ladderSearch(b, loc, !isDefender, st)
    b.undo(rec)
    if (isDefender && !captured) return false     // defender found an escape
    if (!isDefender && captured) return true      // attacker found a kill
  }
  // Defender exhausted every try -> caught. Attacker exhausted every try -> alive.
  return isDefender
}

// For a chain with exactly 2 liberties: does an attacker move first work, and
// which first moves are the working ones? (feature 17)
function ladderAttackerFirst(b: TacticalBoard, loc: number, workingMoves: number[]): boolean {
  const pla = b.colors[loc]!
  const opp = -pla
  const libs = b.chain(loc).liberties
  if (libs.length !== 2) return false

  // Try BOTH liberties unconditionally. The attacker heuristics inside
  // ladderSearch (adjacency quitouts, move ordering) belong to deeper attacker
  // nodes, not here — applying them at the root prunes moves KataGo always
  // tries, and silently loses ladders.
  const works = [false, false]
  for (let i = 0; i < 2; i++) {
    const move = libs[i]!
    if (move === b.koLoc) continue                 // isLegal(move, opp) rejects the ko point
    const rec = b.play(move, opp)                  // suicide is never relevant to ladders
    if (!rec) continue
    // The recursive call is defenderFirst, which clears the ko at its root:
    // KataGo assumes every ko works for the defender.
    b.koLoc = -1
    works[i] = b.colors[loc] !== pla
      ? true
      : ladderSearch(b, loc, true, { nodes: 0, exhausted: false })
    b.undo(rec)                                    // restores the previous ko
  }

  if (!works[0] && !works[1]) return false
  workingMoves.length = 0
  if (works[0]) workingMoves.push(libs[0]!)
  if (works[1]) workingMoves.push(libs[1]!)
  return true
}

// Visit every stone whose chain is in a losing ladder, solving once per chain.
// Mirrors KataGo's iterLadders: only chains with 1 or 2 liberties are candidates.
function iterLadders(
  colors: Int8Array,
  size: number,
  koLoc: number,
  visit: (loc: number, workingMoves: number[]) => void,
): void {
  const b = new TacticalBoard(colors, size, koLoc)
  const solved = new Map<number, { caught: boolean; workingMoves: number[] }>()

  for (let loc = 0; loc < size * size; loc++) {
    const color = colors[loc]!
    if (color === EMPTY) continue
    const chain = b.chain(loc)
    const libs = chain.liberties.length
    if (libs !== 1 && libs !== 2) continue

    const head = Math.min(...chain.stones)          // canonical id for the chain
    let entry = solved.get(head)
    if (!entry) {
      const workingMoves: number[] = []
      let caught: boolean
      if (libs === 1) {
        // In atari: the defender replies first, and all kos are assumed to work
        // for the defender, so clear the ko point at the root.
        const saved = b.koLoc
        b.koLoc = -1
        caught = ladderSearch(b, loc, true, { nodes: 0, exhausted: false })
        b.koLoc = saved
      } else {
        caught = ladderAttackerFirst(b, loc, workingMoves)
      }
      entry = { caught, workingMoves }
      solved.set(head, entry)
    }
    if (entry.caught) visit(loc, entry.workingMoves)
  }
}

// ---------------------------------------------------------------------------
//  Benson's algorithm — unconditional life
// ---------------------------------------------------------------------------

interface Region {
  locs: number[]
  vitalFor: Set<number>          // pla chain ids this region is vital for
  internalSpaces: number         // points not adjacent to any pla stone, capped at 2
  containsOpp: boolean
  bordersNonPassAlive: boolean
}

// Marks `result` with points owned by `pla` under Benson's, with
// safeBigTerritories and unsafeBigTerritories both enabled (KataGo's setting
// for area scoring with no tax — nninputs.cpp fillRowV7).
function calculateAreaForPla(
  colors: Int8Array, size: number, pla: Color, suicideLegal: boolean, result: Int8Array,
): void {
  const area = size * size
  const opp = -pla

  // 1. Chains of pla.
  const chainOf = new Int32Array(area).fill(-1)
  const chains: number[][] = []
  let atLeastOnePla = false
  for (let loc = 0; loc < area; loc++) {
    if (colors[loc] !== pla) continue
    atLeastOnePla = true
    if (chainOf[loc] >= 0) continue
    const id = chains.length
    const stones: number[] = []
    const stack = [loc]
    chainOf[loc] = id
    while (stack.length) {
      const cur = stack.pop()!
      stones.push(cur)
      for (const n of neighbors(cur, size)) {
        if (colors[n] === pla && chainOf[n] < 0) { chainOf[n] = id; stack.push(n) }
      }
    }
    chains.push(stones)
  }
  if (!atLeastOnePla) return

  // 2. Maximal regions of empty-or-opponent points.
  const regionOf = new Int32Array(area).fill(-1)
  const regions: Region[] = []
  for (let loc = 0; loc < area; loc++) {
    if (colors[loc] === pla || regionOf[loc] >= 0) continue
    const id = regions.length
    const locs: number[] = []
    const stack = [loc]
    regionOf[loc] = id
    while (stack.length) {
      const cur = stack.pop()!
      locs.push(cur)
      for (const n of neighbors(cur, size)) {
        if (colors[n] !== pla && regionOf[n] < 0) { regionOf[n] = id; stack.push(n) }
      }
    }

    // 3. Vitality: a region is vital to a chain when every qualifying point of
    // the region touches it. With suicide illegal only empty points qualify.
    let vital: Set<number> | null = null
    let internalSpaces = 0
    let containsOpp = false
    for (const p of locs) {
      if (colors[p] === opp) containsOpp = true
      const adjacentChains = new Set<number>()
      for (const n of neighbors(p, size)) if (colors[n] === pla) adjacentChains.add(chainOf[n]!)
      if (adjacentChains.size === 0 && internalSpaces < 2) internalSpaces++
      if (!suicideLegal && colors[p] !== EMPTY) continue
      if (vital === null) vital = adjacentChains
      else for (const c of [...vital]) if (!adjacentChains.has(c)) vital.delete(c)
    }
    regions.push({
      locs, vitalFor: vital ?? new Set(), internalSpaces, containsOpp, bordersNonPassAlive: false,
    })
  }

  // 4. Eliminate: a chain needs >= 2 vital regions to survive; killing a chain
  // spoils every region it borders, which may kill further chains.
  const vitalCount = new Int32Array(chains.length)
  for (const r of regions) for (const c of r.vitalFor) vitalCount[c]!++
  const killed = new Array<boolean>(chains.length).fill(false)
  for (;;) {
    let killedAnything = false
    for (let id = 0; id < chains.length; id++) {
      if (killed[id] || vitalCount[id]! >= 2) continue
      killed[id] = true
      killedAnything = true
      for (const stone of chains[id]!) {
        for (const n of neighbors(stone, size)) {
          const r = regionOf[n]!
          if (r < 0 || regions[r]!.bordersNonPassAlive) continue
          regions[r]!.bordersNonPassAlive = true
          for (const c of regions[r]!.vitalFor) vitalCount[c]!--
        }
      }
    }
    if (!killedAnything) break
  }

  // 5. Mark surviving chains, then their territory.
  for (let id = 0; id < chains.length; id++) {
    if (!killed[id]) for (const s of chains[id]!) result[s] = pla
  }
  for (const r of regions) {
    const shouldMark =
      (r.internalSpaces <= 1 && !r.bordersNonPassAlive) ||
      (!r.containsOpp && !r.bordersNonPassAlive)            // safeBigTerritories
    if (shouldMark) {
      for (const p of r.locs) result[p] = pla               // unconditional, may overwrite
    } else if (!r.containsOpp) {                            // unsafeBigTerritories
      for (const p of r.locs) if (result[p] === EMPTY) result[p] = pla
    }
  }
}

// Ownership of every point: pass-alive territory for both colours, then any
// remaining point takes the colour of the stone sitting on it.
function calculateArea(colors: Int8Array, size: number, suicideLegal = false): Int8Array {
  const result = new Int8Array(size * size)
  calculateAreaForPla(colors, size, BLACK, suicideLegal, result)
  calculateAreaForPla(colors, size, WHITE, suicideLegal, result)
  for (let loc = 0; loc < size * size; loc++) {
    if (result[loc] === EMPTY) result[loc] = colors[loc]!
  }
  return result
}

// ==========================================================================
// Positions and input encoding
// ==========================================================================

// KataGo V7 input encoding (22 spatial + 19 global), NCHW.
//
// Ported from lightvector/KataGo cpp/neuralnet/nninputs.cpp `NNInputs::fillRowV7`.
// Model version 8 consumes input version 7 (nninputs.h:85-86).
//
// The planes that need real Go algorithms — ladders (14-17) and pass-alive
// territory (18,19) — come from the tactics above.
//
// Scope note: this targets `rules: chinese` — SCORING_AREA, KO_SIMPLE, TAX_NONE,
// no button, suicide illegal (rules.cpp:202-211). Under those rules globals
// 6-13 and 15-17 are all identically zero, and feature 6 is the simple ko point
// only, with no superko history to track.

const SPATIAL_FEATURES = 22
const GLOBAL_FEATURES = 19
const KOMI_CLIP_RADIUS = 20      // nninputs.h:21

interface Move {
  pla: Color
  loc: number                    // -1 = pass
}

interface Position {
  size: number
  board: Int8Array               // size*size, row-major from the top (KataGo's y=0 is the top row)
  toPlay: Color
  koLoc: number                  // simple-ko forbidden point, -1 if none
  history: Move[]
  prevBoards: Int8Array[]        // most recent first; ladder planes 15,16 need them
  komi: number                   // points to White
}

const PASS = -1

const newPosition = (size: number, komi = 7.5): Position => ({
  size,
  board: new Int8Array(size * size),
  toPlay: BLACK,
  koLoc: -1,
  history: [],
  prevBoards: [],
  komi,
})

// The boards KataGo runs the ladder search over for planes 15 and 16. It falls
// back to the current board when that much history isn't available, so an
// opening position reports the same ladders on all three planes rather than none.
function historyBoards(pos: Position, included: number): [Int8Array, Int8Array] {
  const prev = included < 1 ? pos.board : (pos.prevBoards[0] ?? pos.board)
  const prevPrev = included < 2 ? prev : (pos.prevBoards[1] ?? prev)
  return [prev, prevPrev]
}

// ---------------------------------------------------------------------------
//  Board mechanics
// ---------------------------------------------------------------------------

// Play assuming legality has been checked. Returns the successor position,
// leaving `pos` untouched — the undo stack and the search tree both keep
// references to it.
function playAssumeLegal(pos: Position, loc: number): Position {
  const next: Position = {
    size: pos.size,
    board: pos.board.slice(),
    toPlay: -pos.toPlay,
    koLoc: -1,
    history: [...pos.history, { pla: pos.toPlay, loc }],
    prevBoards: [pos.board, ...pos.prevBoards].slice(0, 2),
    komi: pos.komi,
  }
  if (loc === PASS) return next

  next.board[loc] = pos.toPlay
  const captured: number[] = []
  for (const n of neighbors(loc, pos.size)) {
    if (next.board[n] !== -pos.toPlay) continue
    const g = group(next.board, pos.size, n)
    if (g.liberties.length > 0) continue
    for (const s of g.stones) { next.board[s] = EMPTY; captured.push(s) }
  }
  next.koLoc = koPoint(group(next.board, pos.size, loc), captured)
  return next
}

// Suicide is illegal under chinese rules (multiStoneSuicideLegal = false), so a
// move is legal exactly when it leaves the played stone with a liberty: an empty
// neighbour, a friendly chain with a liberty to spare, or an enemy chain whose
// last liberty is this point and which therefore comes off the board.
function isLegal(pos: Position, loc: number): boolean {
  if (loc === PASS) return true
  if (pos.board[loc] !== EMPTY) return false
  if (loc === pos.koLoc) return false
  for (const n of neighbors(loc, pos.size)) {
    const c = pos.board[n]!
    if (c === EMPTY) return true
    const libs = group(pos.board, pos.size, n).liberties.length
    if (c === pos.toPlay ? libs > 1 : libs === 1) return true
  }
  return false
}

// ---------------------------------------------------------------------------
//  Encoding
// ---------------------------------------------------------------------------

interface Encoded {
  spatial: Float32Array          // [22, size, size] NCHW
  global: Float32Array           // [19]
}

function encode(pos: Position): Encoded {
  const { size, board, toPlay } = pos
  const area = size * size
  const spatial = new Float32Array(SPATIAL_FEATURES * area)
  const global = new Float32Array(GLOBAL_FEATURES)
  const opp = -toPlay
  const set = (feature: number, loc: number, value = 1) => { spatial[feature * area + loc] = value }

  // ch 0: on-board. ch 1,2: own/opponent stone. ch 3,4,5: exactly 1/2/3 liberties.
  const libCache = new Map<number, number>()
  for (let loc = 0; loc < area; loc++) {
    set(0, loc)
    const stone = board[loc]!
    if (stone === EMPTY) continue
    set(stone === toPlay ? 1 : 2, loc)
    let libs = libCache.get(loc)
    if (libs === undefined) {
      const g = group(board, size, loc)
      libs = g.liberties.length
      for (const s of g.stones) libCache.set(s, libs)
    }
    if (libs === 1) set(3, loc)
    else if (libs === 2) set(4, loc)
    else if (libs === 3) set(5, loc)
  }

  // ch 6: ko ban. Under KO_SIMPLE with no encore this is just the ko point;
  // ch 7,8 are encore-only and stay zero.
  if (pos.koLoc >= 0) set(6, pos.koLoc)

  // ch 9-13 and globals 0-4: the last five moves, most recent first.
  // KataGo walks the chain only while players strictly alternate and stops at
  // the first break — so a doubled move truncates history rather than shifting it.
  const hist = pos.history
  const expected = [opp, toPlay, opp, toPlay, opp]
  let included = 0
  for (let back = 1; back <= 5; back++) {
    const move = hist[hist.length - back]
    if (!move || move.pla !== expected[back - 1]) break
    included = back
    if (move.loc === PASS) global[back - 1] = 1
    else set(8 + back, move.loc)
  }

  // ch 14-17: ladders, on the current board and the two before it. ch 17 marks
  // the attacker's working first moves, and only for opponent chains that still
  // have more than one liberty.
  iterLadders(board, size, pos.koLoc, (loc, workingMoves) => {
    set(14, loc)
    if (board[loc] === opp && (libCache.get(loc) ?? group(board, size, loc).liberties.length) > 1) {
      for (const m of workingMoves) set(17, m)
    }
  })
  const prevBoards = historyBoards(pos, included)
  iterLadders(prevBoards[0]!, size, -1, loc => set(15, loc))
  iterLadders(prevBoards[1]!, size, -1, loc => set(16, loc))

  // ch 18,19: current territory under Benson's, from the mover's perspective.
  const areaOwner = calculateArea(board, size, false)
  for (let loc = 0; loc < area; loc++) {
    if (areaOwner[loc] === toPlay) set(18, loc)
    else if (areaOwner[loc] === opp) set(19, loc)
  }

  // ch 20,21 are second-encore starting stones — always zero in normal play.

  // Globals. Under chinese rules 6-13 and 15-17 are all zero:
  //   6,7  KO_SIMPLE writes nothing      9     SCORING_AREA writes nothing
  //   8    suicide illegal               10,11 TAX_NONE writes nothing
  //   12,13 encorePhase 0                15,16 no playoutDoublingAdvantage
  //   17   hasButton false
  let selfKomi = toPlay === WHITE ? pos.komi : -pos.komi
  const clip = area + KOMI_CLIP_RADIUS
  selfKomi = Math.min(clip, Math.max(-clip, selfKomi))
  global[5] = selfKomi / 20

  // global 14: would a pass end the phase? Under area scoring in the main phase
  // that means the previous move was already a pass.
  const last = hist[hist.length - 1]
  global[14] = last && last.loc === PASS ? 1 : 0

  // global 18: triangular komi-parity wave (nninputs.cpp, fillRowV7 tail).
  // Only written for area scoring, which is our case.
  const drawableKomisAreEven = area % 2 === 0
  const komiFloor = drawableKomisAreEven
    ? Math.floor(selfKomi / 2) * 2
    : Math.floor((selfKomi - 1) / 2) * 2 + 1
  const delta = Math.min(2, Math.max(0, selfKomi - komiFloor))
  global[18] = delta < 0.5 ? delta : delta < 1.5 ? 1 - delta : delta - 2

  return { spatial, global }
}

// ==========================================================================
// The network
// ==========================================================================

// KataGo b6c96 (model version 8) as a tensorgrad graph.
//
// Reference: lightvector/KataGo — python/katago/train/model_pytorch.py for the
// forward pass and global-pool formulas, cpp/neuralnet/desc.cpp for the weight
// format. Version 8 predates configurable activations, so every actv is ReLU
// (desc.cpp:382-401). The exported batchnorms all carry mean=0, var=1, so each
// "norm" is a plain per-channel affine — and since KataGo's convs have no bias,
// these affines ARE the biases.

const TRUNK = 96
const MID = 96
const REG = 64          // regular path width inside a gpool block
const GPOOL = 32        // gpool path width
const PHEAD = 32        // policy head channels
const VHEAD = 32        // value head channels
const V2 = 64

// ---------------------------------------------------------------------------
//  Modules
// ---------------------------------------------------------------------------

// Per-channel affine, shaped [C,1,1] so trailing-axis broadcast lines it up
// against [B,C,H,W] without a reshape at every call site.
class Norm extends Module {
  scale: Tensor
  bias: Tensor
  constructor(channels: number) {
    super()
    this.scale = this.param([channels, 1, 1])
    this.bias = this.param([channels, 1, 1])
  }
}

const norm = (n: Norm, x: Tensor): Tensor => add(mul(x, n.scale), n.bias)

class OrdinaryBlock extends Module {
  norm1: Norm
  w1: Conv2d
  norm2: Norm
  w2: Conv2d
  constructor() {
    super()
    this.norm1 = new Norm(TRUNK)
    this.w1 = new Conv2d(TRUNK, MID, 3, { padding: 1, bias: false })
    this.norm2 = new Norm(MID)
    this.w2 = new Conv2d(MID, TRUNK, 3, { padding: 1, bias: false })
  }
}

class GpoolBlock extends Module {
  norm1: Norm
  w1a: Conv2d          // regular path
  w1b: Conv2d          // pooled path
  norm1b: Norm
  w1r: Tensor          // [3*GPOOL, REG] — pooled stats back into the regular path
  norm2: Norm
  w2: Conv2d
  constructor() {
    super()
    this.norm1 = new Norm(TRUNK)
    this.w1a = new Conv2d(TRUNK, REG, 3, { padding: 1, bias: false })
    this.w1b = new Conv2d(TRUNK, GPOOL, 3, { padding: 1, bias: false })
    this.norm1b = new Norm(GPOOL)
    this.w1r = this.param([3 * GPOOL, REG])
    this.norm2 = new Norm(REG)
    this.w2 = new Conv2d(REG, TRUNK, 3, { padding: 1, bias: false })
  }
}

type Block = OrdinaryBlock | GpoolBlock

class KataGo extends Module {
  conv1 = new Conv2d(SPATIAL_FEATURES, TRUNK, 5, { padding: 2, bias: false })
  ginputw = this.param([GLOBAL_FEATURES, TRUNK])
  blocks: Block[]
  trunkNorm = new Norm(TRUNK)

  // policy head
  p1 = new Conv2d(TRUNK, PHEAD, 1, { bias: false })
  g1 = new Conv2d(TRUNK, PHEAD, 1, { bias: false })
  g1norm = new Norm(PHEAD)
  matmulg2w = this.param([3 * PHEAD, PHEAD])
  p1norm = new Norm(PHEAD)
  p2 = new Conv2d(PHEAD, 1, 1, { bias: false })
  matmulpass = this.param([3 * PHEAD, 1])

  // value head
  v1 = new Conv2d(TRUNK, VHEAD, 1, { bias: false })
  v1norm = new Norm(VHEAD)
  v2w = this.param([3 * VHEAD, V2])
  v2b = this.param([V2])
  v3w = this.param([V2, 3])            // win / loss / draw logits
  v3b = this.param([3])
  sv3w = this.param([V2, 4])           // score-related outputs
  sv3b = this.param([4])
  ownership = new Conv2d(VHEAD, 1, 1, { bias: false })

  // `kinds` mirrors the manifest's block list so the tree matches the checkpoint.
  constructor(kinds: readonly ('ordinary' | 'gpool')[]) {
    super()
    this.blocks = kinds.map(k => (k === 'gpool' ? new GpoolBlock() : new OrdinaryBlock()))
  }
}

// ---------------------------------------------------------------------------
//  Forward
// ---------------------------------------------------------------------------

// KataGo scales its pooled means by board size so one net serves every board.
// model_pytorch.py:505-514 (trunk/policy) and :534-540 (value head).
// On a full board the mask is all ones, so the mask terms drop out and the
// max reduces to a plain global max — which maxPool2d over the whole board is.
const poolConstants = (boardSize: number) => {
  const area = boardSize * boardSize
  const s = Math.sqrt(area) - 14
  return { area, scale2: s / 10, scale3: (s * s) / 100 - 0.1 }
}

const makeForward = (boardSize: number) => {
  const { area, scale2, scale3 } = poolConstants(boardSize)

  // [B,C,H,W] -> [B,3C] : mean, size-scaled mean, max
  const gpool = (x: Tensor, channels: number): Tensor => {
    const avg = mean(reshape(x, [-1, channels, area]), -1)
    const peak = reshape(maxPool2d(x, boardSize), [-1, channels])
    return concat([avg, mul(avg, scale2), peak], -1)
  }

  // The value head pools three mean-based stats — no max term. Easy to get
  // wrong by assuming symmetry with the trunk pool; it isn't symmetric.
  const vgpool = (x: Tensor, channels: number): Tensor => {
    const avg = mean(reshape(x, [-1, channels, area]), -1)
    return concat([avg, mul(avg, scale2), mul(avg, scale3)], -1)
  }

  // Broadcast a per-channel vector back across the board.
  const spread = (v: Tensor, channels: number): Tensor => reshape(v, [-1, channels, 1, 1])

  return (m: KataGo, { spatial, global: globalIn }: { spatial: Tensor; global: Tensor }): Tensor => {
    let h = add(m.conv1.fwd(spatial), spread(matmul(globalIn, m.ginputw), TRUNK))

    for (const block of m.blocks) {
      const pre = relu(norm(block.norm1, h))
      if (block instanceof GpoolBlock) {
        const regular = block.w1a.fwd(pre)
        const pooled = gpool(relu(norm(block.norm1b, block.w1b.fwd(pre))), GPOOL)
        const biased = add(regular, spread(matmul(pooled, block.w1r), REG))
        h = add(h, block.w2.fwd(relu(norm(block.norm2, biased))))
      } else {
        const hidden = relu(norm(block.norm2, block.w1.fwd(pre)))
        h = add(h, block.w2.fwd(hidden))
      }
    }

    const trunk = relu(norm(m.trunkNorm, h))

    // policy: p1 gets a bias derived from g1's pooled stats; pass comes off the
    // same pooled vector, which is why the pass logit needs no spatial path.
    const g1 = relu(norm(m.g1norm, m.g1.fwd(trunk)))
    const g1pool = gpool(g1, PHEAD)
    const p1 = add(m.p1.fwd(trunk), spread(matmul(g1pool, m.matmulg2w), PHEAD))
    const points = reshape(m.p2.fwd(relu(norm(m.p1norm, p1))), [-1, area])
    const pass = matmul(g1pool, m.matmulpass)

    // value: raw logits and raw ownership — KataGo applies softmax/tanh
    // downstream (model_pytorch.py:2844-2847), so we keep the model faithful
    // and post-process in JS.
    const v1 = relu(norm(m.v1norm, m.v1.fwd(trunk)))
    const v2 = relu(add(matmul(vgpool(v1, VHEAD), m.v2w), m.v2b))
    const value = add(matmul(v2, m.v3w), m.v3b)
    const score = add(matmul(v2, m.sv3w), m.sv3b)
    const ownership = reshape(m.ownership.fwd(v1), [-1, area])

    // One output, one readback: [points | pass | value(3) | score(4) | ownership]
    return concat([points, pass, value, score, ownership], -1)
  }
}

// Where each head starts inside the packed output row.
const outputLayout = (boardSize: number) => {
  const area = boardSize * boardSize
  return {
    points: 0,
    pass: area,
    value: area + 1,
    score: area + 4,
    ownership: area + 8,
    width: area * 2 + 8,
  } as const
}

// ---------------------------------------------------------------------------
//  Weight import
// ---------------------------------------------------------------------------

interface TensorSpec {
  shape: number[]
  dtype: 'float32' | 'int8'
  offset: number
  scaleOffset?: number            // int8 only: one f32 scale per groupSize weights
  groupSize?: number
}

interface Manifest {
  format: string
  architecture: { numBlocks: number; blocks: ('ordinary' | 'gpool')[] }
  tensors: Record<string, TensorSpec>
  totalBytes: number
}

// convert.mjs emits KataGo's own layer names; the Module tree derives different
// ones from its field paths. This is the only place the two vocabularies meet.
function toParamNames(manifest: Manifest, blob: ArrayBuffer): Record<string, Float32Array> {
  const params: Record<string, Float32Array> = {}
  // INT8 is a shipping format, not a compute format: tensorgrad is f32-only, so
  // everything expands here. The quantization buys download size (1.05 MB vs
  // 4.11 MB), nothing at inference time.
  const read = (katagoName: string) => {
    const t = manifest.tensors[katagoName]
    if (!t) throw new Error(`checkpoint is missing ${katagoName}`)
    const count = t.shape.reduce((a, b) => a * b, 1)
    if (t.dtype === 'float32') return new Float32Array(blob, t.offset, count)
    if (t.scaleOffset === undefined) throw new Error(`${katagoName}: int8 tensor with no scales`)

    const groupSize = t.groupSize ?? count
    const groups = Math.ceil(count / groupSize)
    const q = new Int8Array(blob, t.offset, count)
    const scales = new Float32Array(blob, t.scaleOffset, groups)
    const out = new Float32Array(count)
    for (let i = 0; i < count; i++) out[i] = q[i]! * scales[(i / groupSize) | 0]!
    return out
  }
  const copy = (dst: string, src: string) => { params[dst] = read(src) }
  const copyNorm = (dst: string, src: string) => {
    copy(`${dst}.scale`, `${src}.scale`)
    copy(`${dst}.bias`, `${src}.bias`)
  }

  copy('conv1.W', 'conv1')
  copy('ginputw', 'ginputw')

  manifest.architecture.blocks.forEach((kind, i) => {
    const src = `rconv${i + 1}`
    const dst = `blocks.${i}`
    copyNorm(`${dst}.norm1`, `${src}/norm1`)
    copyNorm(`${dst}.norm2`, `${src}/norm2`)
    copy(`${dst}.w2.W`, `${src}/w2`)
    if (kind === 'gpool') {
      copy(`${dst}.w1a.W`, `${src}/w1a`)
      copy(`${dst}.w1b.W`, `${src}/w1b`)
      copyNorm(`${dst}.norm1b`, `${src}/norm1b`)
      copy(`${dst}.w1r`, `${src}/w1r`)
    } else {
      copy(`${dst}.w1.W`, `${src}/w1`)
    }
  })

  copyNorm('trunkNorm', 'trunk/norm')

  copy('p1.W', 'p1/intermediate_conv/w')
  copy('g1.W', 'g1/w')
  copyNorm('g1norm', 'g1/norm')
  copy('matmulg2w', 'matmulg2w')
  copyNorm('p1norm', 'p1/norm')
  copy('p2.W', 'p2/w')
  copy('matmulpass', 'matmulpass')

  copy('v1.W', 'v1/w')
  copyNorm('v1norm', 'v1/norm')
  copy('v2w', 'v2/w')
  copy('v2b', 'v2/b')
  copy('v3w', 'v3/w')
  copy('v3b', 'v3/b')
  copy('sv3w', 'sv3/w')
  copy('sv3b', 'sv3/b')
  copy('ownership.W', 'vownership/w')

  return params
}

// Fetch once from the network, thereafter from CacheStorage. The checkpoint is
// 1.16 MB and immutable, so re-downloading it on every open is pure latency.
// Falls back to a plain fetch wherever the Cache API isn't available.
async function cachedFetch(url: string): Promise<ArrayBuffer> {
  const download = async (): Promise<ArrayBuffer> => {
    // One retry after a beat: a transient error on the first visit otherwise kills the
    // page for good, when the file is immutable and cached forever after one success.
    for (let attempt = 0; ; attempt++) {
      try {
        const res = await fetch(url)
        if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`)
        return await res.arrayBuffer()
      } catch (e) {
        if (attempt >= 1) throw e
        await new Promise(r => setTimeout(r, 1000))
      }
    }
  }
  try {
    const cache = await caches.open('katago-b6c96')
    const hit = await cache.match(url)
    if (hit) return hit.arrayBuffer()
    const buf = await download()
    try { await cache.put(url, new Response(buf)) } catch { /* caching is an optimisation */ }
    return buf
  } catch {
    return download()
  }
}

async function loadKataGo(baseUrl: string, boardSize: number) {
  const [manifestBuf, blob] = await Promise.all([
    cachedFetch(`${baseUrl}/katago-b6c96.json`),
    cachedFetch(`${baseUrl}/katago-b6c96.bin`),
  ])
  const manifest: Manifest = JSON.parse(new TextDecoder().decode(manifestBuf)) as Manifest
  if (blob.byteLength !== manifest.totalBytes) throw new Error('truncated checkpoint')

  const net = await compileForward({
    model: new KataGo(manifest.architecture.blocks),
    forward: makeForward(boardSize),
    inputs: {
      spatial: [null, SPATIAL_FEATURES, boardSize, boardSize],
      global: [null, GLOBAL_FEATURES],
    },
  })

  // uploadParams is strict both ways, so surface a mismatch as a readable diff
  // rather than whatever error the library raises first.
  const params = toParamNames(manifest, blob)
  const expected = new Set(net.paramNames as readonly string[])
  const missing = [...expected].filter(n => !(n in params))
  const extra = Object.keys(params).filter(n => !expected.has(n))
  if (missing.length || extra.length) {
    throw new Error(`param mismatch — missing: ${missing.join(', ') || 'none'}; unexpected: ${extra.join(', ') || 'none'}`)
  }

  await net.uploadParams(params)
  return { net, layout: outputLayout(boardSize) }
}

// ==========================================================================
// play
// ==========================================================================

const BOARD_SIZE = 19
const AREA = BOARD_SIZE * BOARD_SIZE
const KOMI = 7.5
const PAD_RATIO = 22 / 30
const HOSHI = [3, 9, 15]

// Low: on a 361-point board even a modest temperature spreads enough weight
// onto bad moves that the games fall apart (winrate swinging 50% -> 99% in
// three plies). This keeps them recognisably Go while still diverging run to run.
const DEMO_TEMP = 0.15
// Only the attract loop's opening is sampled, so the page doesn't replay one
// identical game forever. A Swap continues a real game and plays properly from
// the first move.
const DEMO_OPENING_PLIES = 8
const DEMO_MOVE_MS = 350
// KataGo resigns rather than grinding out a lost game — Go players resign far
// more often than chess players, and a bot that never does is a chore. There is
// no matching button for you: with no rating and no opponent waiting, resigning
// is just New Game with extra steps.
const RESIGN_WINRATE = 0.04
const RESIGN_MIN_PLY = 30
const ATTRACT_STATUS = 'KataGo is playing itself. Click the board to play.'
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))
// A frame — or 50ms, whichever comes first. requestAnimationFrame never fires
// at all in a backgrounded tab, and awaiting one there parks startup forever.
const nextFrame = () => new Promise(r => {
  const go = () => r(null)
  requestAnimationFrame(go)
  setTimeout(go, 50)
})

// ---------------------------------------------------------------------------
//  Strength
// ---------------------------------------------------------------------------

// Two knobs, not one. Search only ever makes the net stronger, and its floor is
// raw policy: the whole network, playing its single best guess, which is what
// this bulb used to be. So everything below that floor has to come from move
// *selection*: price a shortlist of the net's own candidates and deliberately
// take one that gives up a target number of points. That loses coherently.
// Temperature, the other obvious lever, instead plays well and then randomly
// self-destructs.
//
// The names are a plain relative ladder rather than kyu/dan or 'club', which
// would be claims about human rank that nothing here measures.
interface Skill {
  visits: number          // 0 = no search
  candidates: number      // soft play: how many policy moves to price
  targetLoss: number      // soft play: points to hand over
}

const SKILLS = {
  easiest: { visits: 0, candidates: 24, targetLoss: 8 },
  easy: { visits: 0, candidates: 16, targetLoss: 3 },
  medium: { visits: 0, candidates: 0, targetLoss: 0 },
  hard: { visits: 400, candidates: 0, targetLoss: 0 },
  // 1200 is the budget KataGo's own published ratings are measured at. Costs
  // ~1.5 ms/visit here, so ~1.8 s a move, which is well inside what a Go player
  // expects a bot to take. Search is bounded by the JS encoder rather than the
  // GPU, so this scales with the CPU, not the graphics card.
  hardest: { visits: 1200, candidates: 0, targetLoss: 0 },
} satisfies Record<string, Skill>

type SkillName = keyof typeof SKILLS
const SKILL_NAMES = ['easiest', 'easy', 'medium', 'hard', 'hardest'] as const

type OverlayKind = 'ownership' | 'policy' | 'none'
const OVERLAY_NAMES = ['ownership', 'policy', 'none'] as const

interface Evaluation {
  policy: Float32Array          // legal-masked probabilities, length AREA+1
  ownership: Float32Array       // tanh'd, Black's perspective
  moverWin: number              // the net's own frame: P(win) for the side to move
  blackWinrate: number
  blackLead: number
  ms: number
}

interface Choice { move: number; e: Evaluation; ms: number; visits: number }

// Greedy self-play is deterministic, so the same game would replay forever.
// A little temperature makes each run different without wrecking the play.
function sampleFrom(probs: Float32Array, temperature: number): number {
  const idx: number[] = []
  const w: number[] = []
  let total = 0
  for (let i = 0; i <= AREA; i++) {
    if (probs[i]! < 0) continue
    const v = Math.pow(probs[i]!, 1 / temperature)
    if (v <= 0) continue
    idx.push(i); w.push(v); total += v
  }
  if (!idx.length) return AREA
  let r = Math.random() * total
  for (let i = 0; i < idx.length; i++) { r -= w[i]!; if (r <= 0) return idx[i]! }
  return idx[idx.length - 1]!
}

function argmaxPolicy(probs: Float32Array): number {
  let best = AREA, bestP = -Infinity
  for (let i = 0; i <= AREA; i++) if (probs[i]! > bestP) { bestP = probs[i]!; best = i }
  return best
}

// ---------------------------------------------------------------------------
//  Search tree
// ---------------------------------------------------------------------------

// PUCT with virtual loss, so several leaves can be gathered into one batched
// dispatch instead of paying batch-1 latency per visit. A node's statistics are
// stored in ITS OWN mover's frame, so a parent reads a child as `1 - Q` — get
// that backwards and the search confidently picks the worst move available.
//
// Everything in this block is tree arithmetic and touches no net; the descent
// that feeds it is Engine.search below.
const CPUCT = 1.1
const FPU = 0.2                  // first-play urgency reduction for unvisited children
const MAX_DEPTH = 60

interface MctsNode {
  pos: Position
  n: number
  w: number                      // summed value, in pos.toPlay's frame
  vl: number                     // descents currently in flight
  prior: Float32Array | null     // null until expanded
  value: number | null           // net value, cached so terminals re-use it
  terminal: boolean
  children: Map<number, MctsNode>
}

const endsGame = (p: Position): boolean => {
  const h = p.history
  return h.length >= 2 && h[h.length - 1]!.loc === PASS && h[h.length - 2]!.loc === PASS
}

const makeNode = (p: Position): MctsNode => ({
  pos: p, n: 0, w: 0, vl: 0, prior: null, value: null,
  terminal: endsGame(p), children: new Map(),
})

function selectChild(node: MctsNode): number {
  const prior = node.prior!
  const parentQ = node.n > 0 ? node.w / node.n : 0.5
  const sqrtN = Math.sqrt(Math.max(1, node.n + node.vl))
  let best = -1
  let bestScore = -Infinity
  for (let i = 0; i <= AREA; i++) {
    const p = prior[i]!
    if (p < 0) continue                       // illegal
    const child = node.children.get(i)
    const cn = child ? child.n + child.vl : 0
    // The virtual loss counts as a virtual WIN for the child, which is what
    // makes an in-flight branch unattractive to whoever is choosing above it.
    const q = child && cn > 0 ? 1 - (child.w + child.vl) / cn : parentQ - FPU
    const score = q + CPUCT * p * sqrtN / (1 + cn)
    if (score > bestScore) { bestScore = score; best = i }
  }
  return best
}

function descend(root: MctsNode): { path: MctsNode[]; leaf: MctsNode } {
  const path: MctsNode[] = []
  let node = root
  for (;;) {
    path.push(node)
    node.vl += 1
    if (node.terminal || node.prior === null || path.length > MAX_DEPTH) break
    const idx = selectChild(node)
    if (idx < 0) break
    let child = node.children.get(idx)
    if (!child) {
      child = makeNode(playAssumeLegal(node.pos, idx === AREA ? PASS : idx))
      node.children.set(idx, child)
    }
    node = child
  }
  return { path, leaf: node }
}

// Walk back to the root, flipping perspective at every ply and releasing the
// virtual loss laid down on the way in.
function backup(path: MctsNode[], leafValue: number): void {
  let v = leafValue
  for (let i = path.length - 1; i >= 0; i--) {
    const node = path[i]!
    node.vl -= 1
    node.n += 1
    node.w += v
    v = 1 - v
  }
}

const bestRootMove = (root: MctsNode): number => {
  let best = -1, bestN = -1
  for (const [move, child] of root.children) {
    if (child.n > bestN) { bestN = child.n; best = move }
  }
  return best
}

// ---------------------------------------------------------------------------
//  Engine — everything that touches the net
// ---------------------------------------------------------------------------

// The batch axis is dynamic, so N positions cost one dispatch — but tensorgrad
// specialises a pipeline per batch SHAPE, and each new shape costs seconds to
// compile. Measured cold, a lone batch-2 call took 1957 ms. So every batched run
// here fills exactly BATCH slots and pads: only two pipelines ever exist (1 and
// BATCH) and each compile is paid once. Letting the batch size follow the work
// instead stalls for ~2 s every time a new size shows up.
//
// Encoding does not batch. That stays ~1 ms of CPU per position, and it is what
// actually bounds search — not the net.
const BATCH = 8
const SPATIAL_STRIDE = SPATIAL_FEATURES * AREA

type Compiled = Awaited<ReturnType<typeof loadKataGo>>

// One object for the compiled net, the layout of the row it returns, and every
// question you can ask it: evaluate a position, evaluate a batch, search, choose
// a move. It only exists once it is usable — `load` doesn't resolve until the
// pipeline it needs is compiled — so nothing inside ever re-checks whether the
// net is up. `Game` holds the single nullable reference and answers that
// question for the whole page, which is why the word `null` appears nowhere
// below.
class Engine {
  #net: Compiled['net']
  #layout: Compiled['layout']

  private constructor(c: Compiled) {
    this.#net = c.net
    this.#layout = c.layout
  }

  static async load(baseUrl: string, boardSize: number): Promise<Engine> {
    const engine = new Engine(await loadKataGo(baseUrl, boardSize))
    // Only the single-position shape is needed to start playing, so that is all
    // `load` waits behind. The batched shape costs seconds to compile and
    // nothing on screen needs it yet — `warm` takes it in the background.
    await engine.#runRows([newPosition(boardSize, KOMI)], 1)
    return engine
  }

  // The batched pipeline is only reached by search and by the weakened settings.
  // Compiling it off to one side rather than on first use puts that stall
  // wherever the caller chooses to put it — in practice the attract loop, where
  // a skipped beat is invisible — instead of mid-game on the move after you
  // change the strength.
  async warm(): Promise<void> {
    await sleep(1200)                       // let the page settle first
    const t0 = performance.now()
    await this.#runRows([newPosition(BOARD_SIZE, KOMI)], BATCH)
    tb.log(`batch pipeline warmed in ${(performance.now() - t0).toFixed(0)} ms`)
  }

  // ---- asking the net ----

  // One row of the packed output. `base` is the row offset, so this reads a
  // batched run as readily as a single position — search needs the batched form
  // and the UI needs the single one, and the two must not drift apart.
  #decodeRow(p: Position, out: Float32Array, base: number, ms: number): Evaluation {
    const layout = this.#layout
    const black = p.toPlay === BLACK
    const logitAt = (i: number) =>
      i === AREA ? out[base + layout.pass]! : out[base + layout.points + i]!

    // Mask illegal moves, then softmax over what remains — KataGo's convention.
    const probs = new Float32Array(AREA + 1)
    const legal: boolean[] = []
    for (let i = 0; i < AREA; i++) legal.push(isLegal(p, i))
    legal.push(true)
    let max = -Infinity
    for (let i = 0; i <= AREA; i++) if (legal[i] && logitAt(i) > max) max = logitAt(i)
    let total = 0
    for (let i = 0; i <= AREA; i++) {
      if (!legal[i]) { probs[i] = -1; continue }
      const v = Math.exp(logitAt(i) - max)
      probs[i] = v
      total += v
    }
    for (let i = 0; i <= AREA; i++) if (probs[i]! >= 0) probs[i]! /= total

    const moverWin = 1 / (1 + Math.exp(-(out[base + layout.value]! - out[base + layout.value + 1]!)))

    const own = new Float32Array(AREA)
    const sign = black ? 1 : -1
    for (let i = 0; i < AREA; i++) own[i] = sign * Math.tanh(out[base + layout.ownership + i]!)

    // sv3 is [scoreMean, scoreMeanSq, lead, varTimeLeft], and KataGo scales the
    // score outputs by 20 before reporting. Calibrated against the reference
    // rather than assumed: -0.7701 -> -15.4430, -0.7458 -> -14.9224.
    const lead = out[base + layout.score + 2]! * 20 * sign

    return {
      policy: probs,
      ownership: own,
      moverWin,
      blackWinrate: black ? moverWin : 1 - moverWin,
      blackLead: lead,
      ms,
    }
  }

  async #runRows(list: Position[], slots: number): Promise<Evaluation[]> {
    if (list.length === 0) return []
    const spatial = new Float32Array(slots * SPATIAL_STRIDE)
    const global = new Float32Array(slots * GLOBAL_FEATURES)
    for (let i = 0; i < list.length; i++) {
      const enc = encode(list[i]!)
      spatial.set(enc.spatial, i * SPATIAL_STRIDE)
      global.set(enc.global, i * GLOBAL_FEATURES)
    }
    // Padding slots repeat the last real position rather than being re-encoded:
    // their output is discarded, and the encoder is the expensive half.
    const last = list.length - 1
    for (let i = list.length; i < slots; i++) {
      spatial.copyWithin(i * SPATIAL_STRIDE, last * SPATIAL_STRIDE, (last + 1) * SPATIAL_STRIDE)
      global.copyWithin(i * GLOBAL_FEATURES, last * GLOBAL_FEATURES, (last + 1) * GLOBAL_FEATURES)
    }
    const t0 = performance.now()
    const r = await this.#net.run({ spatial, global })
    const ms = performance.now() - t0
    if (r.kind !== 'completed') return []
    const width = this.#layout.width
    return list.map((p, i) => this.#decodeRow(p, r.output, i * width, ms / list.length))
  }

  async evaluateBatch(list: Position[]): Promise<Evaluation[]> {
    if (list.length === 1) return this.#runRows(list, 1)
    const out: Evaluation[] = []
    for (let i = 0; i < list.length; i += BATCH) {
      const chunk = list.slice(i, i + BATCH)
      const rows = await this.#runRows(chunk, BATCH)
      if (rows.length !== chunk.length) return out
      out.push(...rows)
    }
    return out
  }

  async evaluate(p: Position): Promise<Evaluation | null> {
    return (await this.evaluateBatch([p]))[0] ?? null
  }

  // ---- searching ----

  async search(rootPos: Position, visits: number): Promise<{ root: MctsNode; rootEval: Evaluation } | null> {
    const rootEval = await this.evaluate(rootPos)
    if (!rootEval) return null
    const root = makeNode(rootPos)
    root.prior = rootEval.policy
    root.value = rootEval.moverWin
    root.n = 1
    root.w = rootEval.moverWin

    let done = 1
    while (done < visits) {
      const want = Math.min(BATCH, visits - done)
      const pending: { path: MctsNode[]; leaf: MctsNode }[] = []
      for (let i = 0; i < want; i++) {
        const d = descend(root)
        // A depth cut-off or a revisited terminal already has a value; those cost
        // nothing and must not go to the GPU.
        if (d.leaf.value !== null) { backup(d.path, d.leaf.value); done += 1 }
        else pending.push(d)
      }
      if (pending.length === 0) continue
      const evals = await this.evaluateBatch(pending.map(d => d.leaf.pos))
      if (evals.length !== pending.length) {
        for (const d of pending) backup(d.path, 0.5)   // release the virtual losses
        return { root, rootEval }
      }
      for (let i = 0; i < pending.length; i++) {
        const { path, leaf } = pending[i]!
        const ev = evals[i]!
        leaf.value = ev.moverWin
        if (!leaf.terminal) leaf.prior = ev.policy
        backup(path, ev.moverWin)
        done += 1
      }
    }
    return { root, rootEval }
  }

  // ---- choosing a move ----

  // Price the net's own top candidates and take the one closest to a target
  // score loss. The shortlist is policy-ordered, so even the discarded moves are
  // moves the net considered — the result plays sensible-looking Go and is simply
  // behind, rather than playing well and then throwing a group away.
  async #softMove(p: Position, skill: Skill): Promise<{ move: number; e: Evaluation } | null> {
    const e = await this.evaluate(p)
    if (!e) return null
    const order: number[] = []
    for (let i = 0; i <= AREA; i++) if (e.policy[i]! >= 0) order.push(i)
    order.sort((a, b) => e.policy[b]! - e.policy[a]!)
    // Pass survives the shortlist only when the net genuinely wants it. A weak
    // setting that stumbles into a pass hands over the game for nothing.
    const top = order.filter((m, i) => m !== AREA || i === 0).slice(0, skill.candidates)
    if (top.length <= 1) return { move: top[0] ?? AREA, e }

    const evs = await this.evaluateBatch(top.map(m => playAssumeLegal(p, m === AREA ? PASS : m)))
    if (evs.length !== top.length) return { move: top[0]!, e }
    const sign = p.toPlay === BLACK ? 1 : -1
    const leads = evs.map(x => x.blackLead * sign)      // onto the mover's frame
    const target = Math.max(...leads) - skill.targetLoss
    let pick = 0
    let bestDist = Infinity
    for (let i = 0; i < leads.length; i++) {
      const d = Math.abs(leads[i]! - target)
      if (d < bestDist) { bestDist = d; pick = i }
    }
    return { move: top[pick]!, e }
  }

  async chooseMove(p: Position, skill: Skill): Promise<Choice | null> {
    const t0 = performance.now()
    if (skill.visits > 0) {
      const r = await this.search(p, skill.visits)
      if (!r) return null
      const move = bestRootMove(r.root)
      return {
        move: move < 0 ? argmaxPolicy(r.rootEval.policy) : move,
        e: r.rootEval, ms: performance.now() - t0, visits: skill.visits,
      }
    }
    if (skill.candidates > 0) {
      const r = await this.#softMove(p, skill)
      if (!r) return null
      return { move: r.move, e: r.e, ms: performance.now() - t0, visits: 0 }
    }
    const e = await this.evaluate(p)
    if (!e) return null
    return { move: argmaxPolicy(e.policy), e, ms: performance.now() - t0, visits: 0 }
  }
}

// ---------------------------------------------------------------------------
//  Ending a game
// ---------------------------------------------------------------------------

// Area score straight off the net's ownership head. At a finished position
// every point is claimed by someone and a dead group sits inside enemy
// territory, so thresholding ownership does dead-stone removal for free — which
// is the hard part of scoring Go and the reason most engines punt on it.
//
// This is the net's count, not a rules-exact one: there is no dead-stone
// negotiation, and a game stopped early will be scored on an unsettled board.
function scoreFromOwnership(own: Float32Array): string {
  let black = 0
  for (let i = 0; i < AREA; i++) if (own[i]! > 0) black++
  const lead = black - (AREA - black) - KOMI
  return lead > 0 ? `B+${lead.toFixed(1)}` : `W+${(-lead).toFixed(1)}`
}

// ---------------------------------------------------------------------------
//  Game record
// ---------------------------------------------------------------------------

interface GameRecord {
  pos: Position
  mode: 'auto' | 'play'
  humanColor: Color
  swapped: boolean
  result: string
}

// SGF FF[4]. A point is its column letter then its row letter, both counting
// from 'a' with row 'a' at the top — which is exactly how `loc` is laid out, so
// nothing flips. A pass is an empty value.
function toSgf(r: GameRecord): string {
  const letter = (n: number) => String.fromCharCode(97 + n)
  const point = (loc: number) =>
    loc === PASS ? '' : letter(loc % BOARD_SIZE) + letter(Math.floor(loc / BOARD_SIZE))
  // After a Swap neither side held one colour for the whole game, so naming the
  // players would be a fiction. The attract loop is KataGo on both sides.
  const names =
    r.mode === 'auto' ? 'PB[KataGo b6c96]PW[KataGo b6c96]'
    : r.swapped ? ''
    : r.humanColor === BLACK ? 'PB[Human]PW[KataGo b6c96]'
    : 'PB[KataGo b6c96]PW[Human]'
  const moves = r.pos.history
    .map(m => `;${m.pla === BLACK ? 'B' : 'W'}[${point(m.loc)}]`)
    .join('')
  return `(;GM[1]FF[4]CA[UTF-8]AP[tensorgrad kata-go]SZ[${BOARD_SIZE}]` +
    `KM[${KOMI}]RU[Chinese]${names}${r.result ? `RE[${r.result}]` : ''}${moves})`
}

function downloadSgf(sgf: string, plies: number): void {
  const url = URL.createObjectURL(new Blob([sgf], { type: 'application/x-go-sgf' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `katago-${plies}-moves.sgf`
  a.click()
  // The click is already dispatched, but revoking in the same tick has raced in
  // some browsers, so let the frame finish first.
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

// ==========================================================================
// The page
// ==========================================================================
//
// Four components, split by what they own rather than by where they appear.
// `Game` is the model: the position, the loop that drives the engine, and the
// line of text under the board. It has no view of its own. `Board` is the
// canvas — painted rather than rendered, so it owns a drawing context and a
// geometry no view can express. `Controls` is the row above it. `Root` is the
// page, and holds the three.
//
// Nothing here keeps a copy of anything another one owns: the buttons read
// `game`, the two dropdowns bind straight onto `game.skill` and
// `board.overlay`, and the board paints from `game`. So there is no state to
// push between them and nothing that can disagree — an update anywhere
// re-renders the tree from whatever the owners now say.

// What a child may reach for. Deliberately the two components and nothing
// else — a child reads the game or the board, never the page.
interface IRoot {
  game: Game
  board: Board
}

// ---------------------------------------------------------------------------
//  Game — the position, the engine loop, the status line
// ---------------------------------------------------------------------------

class Game extends Component {
  // Bound by the strength dropdown; see the setter below.
  #skill: SkillName = 'medium'

  // The one nullable reference to the engine on this page. Every entry point
  // below resolves it once and hands the non-null `Engine` down, so the "is the
  // net up" question is asked at the door and nowhere after it.
  #engine?: Engine

  pos = newPosition(BOARD_SIZE, KOMI)
  lastMove?: number

  // What the readouts under the board report. One field replaces the pair of
  // DOM writes the old showEval() made, so the bar and the score line can't
  // disagree with each other or go stale.
  shown?: Evaluation
  // What the board may paint. Cleared independently of `shown`, because your
  // own move makes the policy heat wrong immediately while the numbers under
  // the board are still the last true reading.
  policy?: Float32Array
  ownership?: Float32Array

  status = 'starting…'
  // `waiting` shimmers the line. Reserve it for states where the page owes you
  // something and can't say when: everything else is a result, and shimmering a
  // result reads as though it were still settling.
  waiting = true
  // Until the engine is up there is nothing on the board to read, so the status
  // line reads on the board rather than under it, where the eye already is.
  loading = true
  busy = false

  #stack: Position[] = []
  // Two states. 'auto' is the engine driving both colours — used as an attract
  // loop on load, and again for Swap, where it takes over your side from the
  // current position. `autoGen` is the cancellation token: the loop re-checks it
  // after every await and bails if it has been superseded.
  #mode: 'auto' | 'play' = 'auto'
  #autoGen = 0
  #gameOver = false
  // Which colour you hold. Swap trades it with KataGo mid-game, so nothing may
  // assume the human is Black.
  #humanColor: Color = BLACK
  // Record-keeping for the SGF header: the result once the game has an outcome,
  // and whether a Swap means neither player held one colour throughout.
  #gameResult = ''
  #swapped = false
  #started = false

  get #board() { return (this.ctx.root as unknown as IRoot).board }

  // ---- what the views read ----

  // Whether the engine is up. Gates the one control that is live before a game
  // exists, and stands behind `idle`.
  get ready(): boolean { return this.#engine !== undefined }

  // The controls are live only between the moves of a human game — not while
  // the attract loop is running and not after the game has been scored.
  get idle(): boolean { return this.ready && this.#mode === 'play' && !this.#gameOver }
  get plies(): number { return this.pos.history.length }
  // Undo goes two plies back, so it needs two to go back to.
  get canUndo(): boolean { return this.#stack.length >= 2 }

  scoreLine(): string {
    const e = this.shown
    if (!e) return ''
    const lead = e.blackLead
    return `black ${(e.blackWinrate * 100).toFixed(1)}%   ·   ` +
      `${lead >= 0 ? 'B' : 'W'}+${Math.abs(lead).toFixed(1)}   ·   ${e.ms.toFixed(1)} ms`
  }

  // A setter, so the dropdown can bind to it directly and the side effect still
  // happens: changing strength mid-game says so, rather than silently altering
  // who you are playing.
  get skill(): SkillName { return this.#skill }
  set skill(v: SkillName) {
    this.#skill = v
    if (this.#mode === 'play' && !this.#gameOver) this.#setStatus(`Strength: ${v}. Your move.`)
  }

  // ---- state ----

  #setStatus(s: string, waiting = false): void {
    this.status = s
    this.waiting = waiting
    this.update()
  }

  // Moves are carried as policy indices throughout: 0..AREA-1 are points, AREA
  // is a pass. Only here does that turn back into a board position.
  #commitMove(move: number): void {
    this.#stack.push(this.pos)
    this.pos = playAssumeLegal(this.pos, move === AREA ? PASS : move)
    this.lastMove = move === AREA ? undefined : move
    this.update()
  }

  #resetGame(): void {
    this.pos = newPosition(BOARD_SIZE, KOMI)
    this.#stack = []
    this.lastMove = undefined
    this.#gameResult = ''
    this.#swapped = false
  }

  // Adopt an evaluation as what the overlays and the score line are showing.
  #applyEval(e: Evaluation | null): void {
    this.policy = e?.policy
    this.ownership = e?.ownership
    if (e) this.shown = e
    this.update()
  }

  async #refresh(engine: Engine): Promise<void> {
    this.#applyEval(await engine.evaluate(this.pos))
  }

  // ---- playing ----

  async #reply(engine: Engine): Promise<void> {
    const c = await engine.chooseMove(this.pos, SKILLS[this.#skill])
    if (!c) return
    const engineWin = this.#humanColor === BLACK ? 1 - c.e.blackWinrate : c.e.blackWinrate
    if (engineWin < RESIGN_WINRATE && this.pos.history.length >= RESIGN_MIN_PLY) {
      this.#gameResult = `${this.#humanColor === BLACK ? 'B' : 'W'}+R`
      await this.#finishGame(engine, 'KataGo resigns. You win.', false)
      return
    }
    this.#commitMove(c.move)
    if (endsGame(this.pos)) { await this.#finishGame(engine, 'Both players passed.', true); return }
    // The move is already ringed on the board and its cost is already in the
    // score line, so naming it again is noise. Two exceptions: a pass leaves no
    // mark, so it has to be said; and search time is not the score line's `ms`,
    // which times one evaluation rather than the whole search.
    const cost = c.visits ? `  ·  ${c.visits} visits in ${c.ms.toFixed(0)} ms` : ''
    this.#setStatus(c.move === AREA ? `KataGo passes. Your move.${cost}` : `Your move.${cost}`)
    await this.#refresh(engine)
  }

  async #finishGame(engine: Engine, headline: string, withScore: boolean): Promise<void> {
    this.#autoGen++                       // stop any auto loop still in flight
    this.#mode = 'play'
    this.#gameOver = true
    const e = await engine.evaluate(this.pos)
    if (e) {
      this.shown = e
      this.policy = undefined
      this.ownership = e.ownership
      this.#board.overlay = 'ownership'   // the shading IS the explanation of the score
    }
    if (withScore && e) this.#gameResult = scoreFromOwnership(e.ownership)
    this.#setStatus(withScore && e
      ? `${headline}  ·  ${scoreFromOwnership(e.ownership)} by the net's count.`
      : headline)
  }

  // KataGo against itself, restarting forever — the attract loop, so the page
  // isn't a dead board before you touch it. Every await is followed by a
  // generation check so a click mid-move never lands after you've taken over.
  async #runAttract(engine: Engine): Promise<void> {
    const gen = ++this.#autoGen
    this.#mode = 'auto'
    this.#gameOver = false
    this.#resetGame()
    // Set once, not per move. The board and the score line already show what is
    // happening; the only thing this line has to say is how to get in.
    this.#setStatus(ATTRACT_STATUS)

    let ply = 0
    while (this.#mode === 'auto' && gen === this.#autoGen) {
      // Two consecutive passes ends the game; Position tracks history, not a
      // pass counter, so read it off the tail.
      if (endsGame(this.pos) || ply >= 400) {
        this.#setStatus('Self-play finished. Starting again…')
        await sleep(2500)
        if (gen !== this.#autoGen) return
        this.#resetGame()
        ply = 0
        this.#setStatus(ATTRACT_STATUS)
        continue
      }

      const t0 = performance.now()
      let e: Evaluation | null
      let move: number
      if (ply < DEMO_OPENING_PLIES) {
        e = await engine.evaluate(this.pos)
        if (!e || gen !== this.#autoGen) return
        move = sampleFrom(e.policy, DEMO_TEMP)
      } else {
        const c = await engine.chooseMove(this.pos, SKILLS[this.#skill])
        if (!c || gen !== this.#autoGen) return
        e = c.e
        move = c.move
      }
      this.#applyEval(e)
      this.#commitMove(move)
      ply++

      // Pace it. An unsearched move is ~6 ms, so unpaced it plays a whole game
      // in a few seconds — far too fast to watch, and the sleep yields the
      // thread for input. Searching settings already spend longer than the
      // pacing, so only the remainder is waited out.
      const rest = DEMO_MOVE_MS - (performance.now() - t0)
      if (rest > 0) await sleep(rest)
      if (gen !== this.#autoGen) return
    }
  }

  #stopAuto(): void {
    this.#autoGen++
    this.#mode = 'play'
  }

  // ---- what the controls call ----

  async newGame(): Promise<void> {
    const engine = this.#engine
    if (this.busy || !engine) return
    this.#stopAuto()
    this.busy = true
    this.#gameOver = false
    this.#humanColor = BLACK
    this.#resetGame()
    this.#setStatus('Your move. You are Black.')
    await this.#refresh(engine)
    this.busy = false
    this.update()
  }

  async pass(): Promise<void> {
    const engine = this.#engine
    if (this.busy || !this.idle || !engine || this.pos.toPlay !== this.#humanColor) return
    this.busy = true
    this.#commitMove(AREA)
    // Your pass may itself have ended the game, before KataGo is asked anything.
    if (endsGame(this.pos)) { await this.#finishGame(engine, 'Both players passed.', true); this.busy = false; return }
    this.#setStatus('You passed. KataGo is thinking…', true)
    await this.#reply(engine)
    this.busy = false
    this.update()
  }

  async undo(): Promise<void> {
    const engine = this.#engine
    if (this.busy || !this.idle || !engine || !this.canUndo) return
    this.busy = true
    this.#gameOver = false
    // Two plies back, so it stays your turn.
    this.#stack.pop()
    this.pos = this.#stack.pop() ?? newPosition(BOARD_SIZE, KOMI)
    this.lastMove = undefined
    this.#setStatus('Took back a move.')
    await this.#refresh(engine)
    this.busy = false
    this.update()
  }

  // Trade colours with KataGo. It answers immediately if the swap has left it on
  // move, which is the usual case — you swap instead of playing, so the move you
  // were about to make becomes its move.
  async swap(): Promise<void> {
    const engine = this.#engine
    if (this.busy || !this.idle || !engine) return
    this.#humanColor = -this.#humanColor
    this.#swapped = true
    this.#setStatus(`Swapped. You are ${this.#humanColor === BLACK ? 'Black' : 'White'} now.`)
    if (this.pos.toPlay !== this.#humanColor) {
      this.busy = true
      await this.#reply(engine)
      this.busy = false
      this.update()
    } else {
      await this.#refresh(engine)
    }
  }

  sgfText(): string {
    return toSgf({
      pos: this.pos, mode: this.#mode, humanColor: this.#humanColor,
      swapped: this.#swapped, result: this.#gameResult,
    })
  }

  saveSgf(): void {
    if (this.plies > 0) downloadSgf(this.sgfText(), this.plies)
  }

  async playPoint(row: number, col: number): Promise<void> {
    const engine = this.#engine
    if (this.busy || !engine || this.#gameOver) return
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return

    // Clicking during the attract loop starts a fresh game and becomes your
    // first move: inheriting a board you didn't choose is worse than starting
    // over. Swap is the deliberate version of that, and has its own button.
    if (this.#mode === 'auto') {
      this.#stopAuto()
      this.#humanColor = BLACK
      this.#resetGame()
    } else if (this.pos.toPlay !== this.#humanColor) return

    const loc = row * BOARD_SIZE + col
    if (!isLegal(this.pos, loc)) {
      this.#setStatus('Illegal there: occupied, ko, or self-capture.')
      return
    }

    this.busy = true
    this.#commitMove(loc)
    this.policy = undefined
    this.#setStatus('KataGo is thinking…', true)
    await this.#reply(engine)
    this.busy = false
    this.update()
  }

  // A raw pass by whoever is to move, with no engine reply: state with no
  // control behind it. The only way to reach the two-passes-end-the-game path,
  // since the net won't volunteer a pass this early.
  forcePass(): void { this.#commitMove(AREA) }

  // Everything a probe would otherwise have to scrape out of the DOM, reported
  // from the state itself.
  report() {
    return {
      status: this.status,
      score: this.scoreLine(),
      toPlay: this.pos.toPlay === BLACK ? 'B' : 'W',
      human: this.#humanColor === BLACK ? 'B' : 'W',
      ply: this.plies,
      stack: this.#stack.length,
      stones: this.pos.board.reduce((n, c) => n + (c !== EMPTY ? 1 : 0), 0),
      mode: this.#mode,
      gameOver: this.#gameOver,
    }
  }

  // ---- startup ----

  // Driven from the board's onMounted rather than onAttached, because the board
  // existing is the condition the load waits behind. A remount (the About tab
  // and back) must not start it a second time.
  async start(): Promise<void> {
    if (this.#started) return
    this.#started = true
    try {
      await this.#boot()
    } catch (e: unknown) {
      this.#setStatus(`error: ${(e as { message?: string })?.message ?? String(e)}`)
    }
  }

  async #boot(): Promise<void> {
    // A frame before anything else. The board is already mounted and drawn by
    // the time this runs, and this is where it gets painted — everything below
    // holds the main thread, since compiling WGSL is synchronous, so a board
    // that has not reached the screen by here would not appear until the engine
    // is up.
    await nextFrame()
    const gpu = await checkWebGPU()
    if (!gpu.ok) {
      this.#setStatus(gpu.message)
      return
    }
    this.#setStatus('Compiling the graph and uploading the weights…', true)
    const t0 = performance.now()
    const engine = this.#engine = await Engine.load('assets', BOARD_SIZE)
    // The board is live from here on and says more than any message can.
    this.loading = false
    tb.log(`ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`)
    this.update()
    // Off to one side, so the batched pipeline's compile lands inside the
    // attract loop rather than on the move after you change the strength.
    void engine.warm()
    await this.#runAttract(engine)
  }
}

// ---------------------------------------------------------------------------
//  Board — the canvas
// ---------------------------------------------------------------------------

class Board extends Component {
  // Bound by the overlay dropdown. It lives here rather than on the game
  // because it changes what is painted and nothing about what is played.
  overlay: OverlayKind = 'ownership'

  // The canvas is the one element any of this holds onto. It is painted rather
  // than rendered, so it needs a context, a box to hit-test against, and a
  // resize signal — none of which a view can express. Everything else about the
  // layout is the stylesheet's, including the board's size.
  #canvasEl?: HTMLCanvasElement
  #ctx2d?: CanvasRenderingContext2D

  // Board geometry in CSS pixels, re-derived whenever the canvas box changes.
  #boardPx = 0
  #cell = 0
  #pad = 0
  #lastLayoutWidth = -1

  get #game() { return (this.ctx.root as unknown as IRoot).game }

  view() {
    const game = this.#game
    return div({ class: 'board-wrap' },
      canvas({
        // Keyed so the diff can only ever patch this element. A recreated
        // canvas is a blank backing store, and the board would vanish.
        key: 'board',
        onClick: (e: Event) => this.#click(e as MouseEvent),
        onMounted: el => {
          const c = el as HTMLCanvasElement
          this.#canvasEl = c
          this.#ctx2d = c.getContext('2d') ?? undefined
          const ro = new ResizeObserver(() => { if (this.#layout()) this.draw() })
          ro.observe(c)
          this.#layout()
          this.draw()
          void game.start()
          return () => {
            ro.disconnect()
            this.#canvasEl = undefined
            this.#ctx2d = undefined
            // A remount is a fresh element with a fresh backing store, so the
            // cached width must not suppress the rebuild.
            this.#lastLayoutWidth = -1
          }
        },
      }),
      game.loading
        ? div({ id: 'loading' },
            p({ class: 'board-note' },
              span({ id: 'loading-text', class: game.waiting ? 'waiting' : '' }, game.status)))
        : null,
    )
  }

  // Drawing happens in CSS pixels; the backing store carries the device pixel
  // ratio, which also stops the whole board being a 1x bitmap upscaled on a
  // retina display. The size it reads is whatever the stylesheet settled on.
  #layout(): boolean {
    const canvasEl = this.#canvasEl, ctx2d = this.#ctx2d
    if (!canvasEl || !ctx2d) return false
    // clientWidth, not the bounding rect: the canvas is border-box, so the rect
    // is 2px wider than the surface actually drawn into.
    const raw = canvasEl.clientWidth
    if (raw < 1) return false
    const width = Math.max(220, Math.round(raw))
    if (width === this.#lastLayoutWidth) return false
    this.#lastLayoutWidth = width
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    // Assigning width resets the context transform, so the scale goes back after.
    canvasEl.width = canvasEl.height = Math.round(width * dpr)
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0)
    this.#boardPx = width
    this.#cell = width / (BOARD_SIZE - 1 + 2 * PAD_RATIO)
    this.#pad = this.#cell * PAD_RATIO
    return true
  }

  #px(i: number): number { return this.#pad + i * this.#cell }

  draw(): void {
    const ctx2d = this.#ctx2d
    if (!ctx2d || this.#cell <= 0) return
    const game = this.#game
    const cell = this.#cell
    const css = getComputedStyle(document.documentElement)
    const v = (n: string) => css.getPropertyValue(n).trim()
    ctx2d.fillStyle = v('--board')
    ctx2d.fillRect(0, 0, this.#boardPx, this.#boardPx)

    // Ownership shading sits under the grid so it reads as territory, not marks.
    if (this.overlay === 'ownership' && game.ownership) {
      for (let i = 0; i < AREA; i++) {
        const o = game.ownership[i]!
        if (Math.abs(o) < 0.06) continue
        ctx2d.fillStyle = o > 0
          ? `rgba(20,20,20,${Math.abs(o) * 0.42})`
          : `rgba(250,248,242,${Math.abs(o) * 0.52})`
        const x = this.#px(i % BOARD_SIZE), y = this.#px(Math.floor(i / BOARD_SIZE))
        ctx2d.fillRect(x - cell / 2, y - cell / 2, cell, cell)
      }
    }

    ctx2d.strokeStyle = v('--line')
    ctx2d.lineWidth = 1
    for (let i = 0; i < BOARD_SIZE; i++) {
      ctx2d.beginPath(); ctx2d.moveTo(this.#px(i), this.#px(0)); ctx2d.lineTo(this.#px(i), this.#px(BOARD_SIZE - 1)); ctx2d.stroke()
      ctx2d.beginPath(); ctx2d.moveTo(this.#px(0), this.#px(i)); ctx2d.lineTo(this.#px(BOARD_SIZE - 1), this.#px(i)); ctx2d.stroke()
    }
    ctx2d.fillStyle = v('--line')
    for (const r of HOSHI) for (const c of HOSHI) {
      ctx2d.beginPath(); ctx2d.arc(this.#px(c), this.#px(r), Math.max(2, cell * 0.085), 0, Math.PI * 2); ctx2d.fill()
    }

    if (this.overlay === 'policy' && game.policy) {
      for (let i = 0; i < AREA; i++) {
        const p = game.policy[i]!
        if (p < 0.004) continue
        ctx2d.fillStyle = v('--heat')
        ctx2d.globalAlpha = Math.min(0.8, 0.12 + p * 2)
        ctx2d.beginPath()
        ctx2d.arc(this.#px(i % BOARD_SIZE), this.#px(Math.floor(i / BOARD_SIZE)), cell * 0.34, 0, Math.PI * 2)
        ctx2d.fill()
        ctx2d.globalAlpha = 1
      }
    }

    for (let i = 0; i < AREA; i++) {
      const c = game.pos.board[i]!
      if (c === EMPTY) continue
      const x = this.#px(i % BOARD_SIZE), y = this.#px(Math.floor(i / BOARD_SIZE))
      ctx2d.beginPath(); ctx2d.arc(x, y, cell * 0.46, 0, Math.PI * 2)
      ctx2d.fillStyle = c === BLACK ? '#151515' : '#f6f4ee'
      ctx2d.fill()
      ctx2d.strokeStyle = c === BLACK ? '#000' : '#b5b0a4'
      ctx2d.stroke()
      if (i === game.lastMove) {
        ctx2d.beginPath(); ctx2d.arc(x, y, cell * 0.15, 0, Math.PI * 2)
        ctx2d.strokeStyle = c === BLACK ? '#f6f4ee' : '#151515'
        ctx2d.lineWidth = 2; ctx2d.stroke(); ctx2d.lineWidth = 1
      }
    }
  }

  // Hit-testing runs in CSS pixels, the same space the board is drawn in.
  #click(ev: MouseEvent): void {
    const canvasEl = this.#canvasEl
    if (!canvasEl) return
    const rect = canvasEl.getBoundingClientRect()
    void this.#game.playPoint(
      Math.round((ev.clientY - rect.top - this.#pad) / this.#cell),
      Math.round((ev.clientX - rect.left - this.#pad) / this.#cell),
    )
  }

  // A board click has no named control for `typebulb send` to aim at, so the
  // probe synthesises the event the canvas handler actually reads. The setup is
  // faked; the trigger stays real.
  clickPoint(row: number, col: number): void {
    const canvasEl = this.#canvasEl
    if (!canvasEl) return
    const rect = canvasEl.getBoundingClientRect()
    canvasEl.dispatchEvent(new MouseEvent('click', {
      clientX: rect.left + this.#px(col), clientY: rect.top + this.#px(row),
    }))
  }
}

// ---------------------------------------------------------------------------
//  Controls — the row above the board
// ---------------------------------------------------------------------------

class Controls extends Component {
  get #root() { return this.ctx.root as unknown as IRoot }

  view() {
    const game = this.#root.game
    const idle = game.idle
    // Two groups, not seven controls: the row's spacing is then one decision
    // about two boxes, taken entirely in the stylesheet.
    return div({ class: 'controls' },
      div({ class: 'btn-group' },
        button({ class: 'primary', disabled: !game.ready, onClick: () => void game.newGame() }, 'New Game'),
        button({ disabled: !idle, onClick: () => void game.pass() }, 'Pass'),
        button({ disabled: !idle || !game.canUndo, onClick: () => void game.undo() }, 'Undo'),
        button({ disabled: !idle, onClick: () => void game.swap() }, 'Swap'),
        // Deliberately not gated on `idle`: a self-play game and a finished
        // game are both worth saving. The only thing that makes the record
        // meaningless is having no moves in it.
        button({ disabled: game.plies === 0, onClick: () => game.saveSgf() }, 'Download SGF'),
      ),
      div({ class: 'selects' },
        this.#picker('strength', game, () => game.skill, SKILL_NAMES),
        this.#picker('overlay', this.#root.board, () => this.#root.board.overlay, OVERLAY_NAMES),
      ),
    )
  }

  // A caption over a bound dropdown. The label wraps the select, which is what
  // names it — for a screen reader, and for `tb:set combobox "strength" = hard`
  // alike. Binding writes straight back onto the owning component, so there is
  // no local copy of either setting and no handler to keep in step.
  #picker<T extends string>(caption: string, target: Component, prop: () => T, options: readonly T[]) {
    return label({ class: 'sel' }, caption,
      inputSelect({ target, prop, options: options.map(o => ({ value: o, label: o })) }),
    )
  }
}

// ---------------------------------------------------------------------------
//  Root — the page
// ---------------------------------------------------------------------------

class Root extends Component implements IRoot {
  game = new Game()
  board = new Board()
  controls = new Controls()

  // Two tabs: the game, and everything there is to say about it. The prose was
  // pushing the board below the fold.
  tab: 'play' | 'about' = 'play'

  // The canvas is the one part of the page the framework doesn't render: it is
  // painted, not diffed. `onUpdated` here catches every update in the tree — a
  // move, a dropdown, the status line — so the repaint rides all of them and
  // the board simply can't fall out of step with the state.
  onUpdated(): void {
    this.board.draw()
  }

  view() {
    return div({ class: 'page' },
      h1('KataGo'),
      p({ class: 'subtitle' },
        'A real KataGo network (b6c96) playing full-size Go in the browser. ',
        strong('1.16 MB'), ' of weights, compiled to WGSL by tensorgrad.',
      ),
      div({ class: 'tabs' }, this.#tabBtn('play', 'Play'), this.#tabBtn('about', 'About')),
      this.tab === 'play' ? this.#playView() : this.#aboutView(),
    )
  }

  #tabBtn(tab: 'play' | 'about', label: string) {
    return button({
      class: 'tab',
      ariaSelected: this.tab === tab,
      onClick: () => { if (this.tab !== tab) { this.tab = tab; this.update() } },
    }, label)
  }

  #playView() {
    const game = this.game
    return section({ id: 'tab-play', class: game.loading ? 'loading' : '' },
      this.controls.view(),
      this.board.view(),
      div({ class: 'bar-track' },
        div({ id: 'bar', style: { width: `${((game.shown?.blackWinrate ?? 0.5) * 100).toFixed(1)}%` } })),
      div({ id: 'score' }, game.scoreLine()),
      // Hidden rather than emptied while the note is on the board, so its
      // disappearance moves nothing.
      div({ id: 'status', class: game.waiting ? 'waiting' : '' }, game.status),
    )
  }

  #aboutView() {
    return section({ id: 'tab-about' },
      p({ class: 'note' },
        strong('Rules.'), ' Chinese: area scoring, komi 7.5, simple ko (no superko), suicide ',
        'illegal, no handicap. The input encoder is written for exactly those, so they are ',
        'fixed here rather than a setting. Counting is the net’s own, off the ownership ',
        'head, so there is no dead-stone negotiation and a game stopped early is scored on an ',
        'unsettled board.',
      ),
      p({ class: 'note' },
        'It opens playing itself; click the board to take over as Black. No ONNX Runtime and ',
        'no WASM blob.',
      ),
      p({ class: 'note' },
        'The ownership overlay is the net’s own territory estimate, an auxiliary head that ',
        'comes free with every evaluation. The whole port was verified against the KataGo C++ ',
        'engine on 24 reference positions, then quantised to INT8 for download.',
      ),
      p({ class: 'note' },
        strong('Pass'), ' twice and the game is scored: the ownership head decides who owns ',
        'each point, which also settles which stones are dead, so the board shades into its ',
        'own result. ', strong('Swap'), ' trades colours with KataGo: it plays the move you ',
        'were about to make, and you carry on with the other side. It resigns when its winrate ',
        'collapses, rather than playing a lost game out. ', strong('Download SGF'), ' saves the ',
        'game record at any point, with the rules and the result in its header.',
      ),
      p({ class: 'note' },
        strong('Strength'), ' works two ways, because search only ever makes the net stronger, ',
        'and its floor (', em('medium'), ') is the raw policy: the whole network, playing its ',
        'single best guess. ', em('hard'), ' and ', em('hardest'), ' add 400 and 1200 visits of ',
        'tree search. ', em('easy'), ' and ', em('easiest'), ' instead price the net’s own ',
        'shortlist and take a move that gives away roughly 3 or 8 points, so they lose ',
        'coherently rather than blundering at random.',
      ),
    )
  }
}

const root = new Root()
new App({ root, id: 'app' })

// Probe hook for `typebulb send`. Dev-only, and deliberately narrow: it holds
// only what the built-in verbs cannot reach. Every button and both dropdowns
// are named controls, so `tb:click button "Pass"` and `tb:set combobox
// "strength" = hard` already drive them — and a disabled target is an error
// there rather than a silent no-op, which is the whole point of using them.
tb.onMessage(async (msg: unknown) => {
  const [cmd, a, b] = String(msg).trim().split(/\s+/)
  if (cmd === 'play') {
    // A point on the canvas has no role or name, so `tb:click` can't aim at it.
    root.board.clickPoint(Number(a), Number(b))
  } else if (cmd === 'forcepass') {
    // A move no control offers — the only way to reach the two-passes ending.
    root.game.forcePass()
  } else if (cmd === 'sgf') {
    // The record is a string with no home in the DOM, so a snapshot cannot read
    // it, and the button only hands it to the browser's downloader.
    return root.game.sgfText()
  }
  // Any other message just settles and reports. The built-in verbs reply with
  // the immediate frame, which is too early for a searching move.
  for (let i = 0; i < 600 && root.game.busy; i++) await sleep(50)
  return root.game.report()
})
```

**index.html**

```html
<div id="app"></div>
```

**styles.css**

```css
:root {
  color-scheme: light;
  --bg: #fff; --fg: #1a1a1a; --muted: #6b6b6b; --border: #ddd;
  --board: #e3c489; --line: #6f5a37; --heat: #1d7fd6;
  --accent: #2d6a4f; --accent-fg: #fff; --btn-bg: #fff; --btn-border: #bdbdbd;
  --bar-bg: #e6e6e6; --bar-fg: #2b2b2b;
  --shim-base: #b0b0b0; --shim-hot: #101010;
}
html[data-theme="dark"] {
  color-scheme: dark;
  --bg: #1b1b1b; --fg: #ededed; --muted: #9d9d9d; --border: #333;
  --board: #b9955e; --line: #46381f; --heat: #5eb0ff;
  --accent: #52b788; --accent-fg: #0d1f16; --btn-bg: #2a2a2a; --btn-border: #454545;
  --bar-bg: #333; --bar-fg: #ededed;
  --shim-base: #5a5a5a; --shim-hot: #ffffff;
}
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
}
/* The page is exactly the viewport, and everything inside it is a fixed-height
   row except the board, which takes what is left. The vertical space is padding
   rather than margin: a margin sits outside the box, so whether it counts
   toward the document's height depends on the host, and it can't be part of a
   height that has to add up exactly. The width is wide enough that the five
   buttons and both dropdowns share one line — they come to roughly 600px
   together, and 660 left no room. */
.page {
  max-width: 780px; margin: 0 auto; padding: 2rem 1rem;
  box-sizing: border-box; height: 100dvh;
  display: flex; flex-direction: column;
}
#tab-play { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
/* The prose is shorter than the viewport on most screens, so this scrollbar is
   a fallback for a short window rather than the normal state. */
#tab-about { flex: 1 1 auto; min-height: 0; overflow-y: auto; }
/* Zeroed because the page's top space is its padding now: a default heading
   margin no longer collapses into it, it stacks on top of it. */
h1 { font-size: 1.35rem; margin-top: 0; margin-bottom: 0.3rem; }
.subtitle { color: var(--muted); margin-top: 0; margin-bottom: 1.1rem; line-height: 1.55; }
.tabs {
  display: flex; gap: 0.25rem;
  border-bottom: 1px solid var(--border); margin-bottom: 1rem;
}
.tab {
  padding: 0.45rem 0.9rem; font-size: 0.9rem; cursor: pointer;
  background: none; border: none; border-radius: 0; color: var(--muted);
  border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.tab[aria-selected="true"] { color: var(--fg); border-bottom-color: var(--accent); }
.controls {
  display: flex; gap: 0.5rem; align-items: flex-end; flex-wrap: wrap;
  margin-bottom: 0.9rem;
}
/* Auto margins on both sides of each group, rather than a justify-content that
   would have to change on wrap. Sharing a line, the free space splits four ways
   and the two groups sit evenly inset; wrapped, each line holds one group whose
   own two margins centre it. That is both behaviours at every width, with no
   breakpoint to guess — `space-between` gets the first but pins a wrapped line
   to the start, which is why this used to be measured in script. */
.btn-group, .selects { margin-inline: auto; }
.btn-group { display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: center; }
/* The dropdowns carry their captions above them, so their column is taller than
   a button. Bottom-aligning the row plus a shared control height is what keeps
   the select boxes on the same line as the buttons rather than sagging below. */
.controls button, .controls select { height: 2.15rem; }
button {
  padding: 0 0.9rem; font-size: 0.9rem; cursor: pointer;
  background: var(--btn-bg); color: var(--fg);
  border: 1px solid var(--btn-border); border-radius: 4px;
}
button.primary { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
button:disabled { opacity: 0.45; cursor: default; }
.sel {
  display: flex; flex-direction: column; gap: 0.25rem;
  font-size: 0.72rem; color: var(--muted); letter-spacing: 0.02em;
}
/* One flex item holding both dropdowns, so they ride the button row when it
   fits and wrap together onto their own line when it doesn't — rather than the
   overlay dropping alone and leaving strength orphaned beside the buttons. */
.selects { display: flex; gap: 0.75rem; align-items: flex-end; }
select {
  font-family: inherit; font-size: 0.85rem;
  padding: 0 0.4rem; background: var(--btn-bg); color: var(--fg);
  border: 1px solid var(--btn-border); border-radius: 4px;
}
/* The board takes whichever runs out first: the width of its column, or the
   height the rest of the page leaves. That is what `min(100cqw, 100cqh)` says,
   against a board-wrap that is sized by the flex column and declared a size
   container — so the two candidate sizes are both facts the layout already
   knows, and neither has to be measured or re-applied. It is also monotonic in
   the window by construction: no breakpoint to cross, so a middling window can
   never end up with a bigger board than a large one. The floor is the one case
   where the page is allowed to scroll instead: below it a 19x19 board stops
   being worth playing on.

   border-box so the width covers the 1px border too — the height it is racing
   against is a border box, and 2px is all it takes to bring back a scrollbar. */
.board-wrap {
  flex: 1 1 auto; min-height: 0; container-type: size;
  display: flex; align-items: center; justify-content: center;
}
canvas {
  display: block; aspect-ratio: 1 / 1;
  width: max(320px, min(100cqw, 100cqh));
  box-sizing: border-box; cursor: pointer;
  border: 1px solid var(--border); border-radius: 6px;
  /* The same board, in CSS, so it is on screen before any script has run —
     nothing about wood and 19 lines needs a WGSL compiler to arrive from a CDN
     first, which is what a static `import from 'tensorgrad'` otherwise makes it
     wait for. draw() paints over all of it opaquely on its first pass.

     The two constants are PAD_RATIO's, in percentages of the board:
       span = 18 + 2 * 22/30 = 19.4667 cells   ->  cell = 100/span = 5.137%
       pad  = 22/30 * cell                     ->          3.767%
     A tiled background's percentage offset is measured against the box LESS one
     tile, so the offset that lands the first line on the pad is
     3.767 / (1 - 0.05137) = 3.9711%, not 3.767%. */
  background-color: var(--board);
  background-image:
    linear-gradient(to right, var(--line) 0 1px, transparent 1px),
    linear-gradient(to bottom, var(--line) 0 1px, transparent 1px);
  background-size: 5.137% 5.137%;
  background-position: 3.9711% 3.9711%;
}
/* The overlay covers the wrapper rather than the board, which comes to the same
   place: the board is centred in it, so the note is centred on the board. A
   size container is already a containing block, so it needs no `position`. */
#loading {
  position: absolute; inset: 0; display: grid; place-items: center;
  pointer-events: none; padding: 0 1rem;
}
/* An opaque card, not bare text: the board is a mid-tone wood in both themes,
   and the shimmer's stops are chosen against the page background. On the wood
   the bright end of the sweep all but disappears. */
.board-note {
  margin: 0; padding: 0.5rem 0.85rem; border-radius: 8px;
  background: var(--bg); border: 1px solid var(--border);
  font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.85rem;
  color: var(--muted); line-height: 1.5; text-align: center; max-width: 34ch;
}
/* Hidden, not emptied: the line keeps its box, so the layout is the same
   before and after the overlay goes, and the board never resizes underneath it. */
#tab-play.loading #status { visibility: hidden; }
.bar-track {
  height: 6px; background: var(--bar-bg); border-radius: 3px;
  overflow: hidden; margin-top: 0.9rem;
}
#bar { height: 100%; width: 50%; background: var(--bar-fg); transition: width 0.25s ease; }
#score, #status {
  font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.82rem;
  color: var(--muted); margin-top: 0.5rem; min-height: 1.3em;
}
/* A highlight band swept across the glyphs themselves, via background-clip, so
   the waiting reads as the text working rather than as a spinner parked beside
   it. Both stops are themed, so it stays legible in light and dark, and with
   the animation suppressed it degrades to a static gradient rather than to
   invisible text. */
/* Scoped to #status, not bare .waiting: `#score, #status` sets `color` at ID
   specificity, which beats a class, so `color: transparent` silently loses and
   the glyphs stay painted over the gradient. */
#status.waiting, #loading-text.waiting {
  /* inline-block so the gradient box hugs the glyphs. As a full-width block the
     band spends most of its sweep over empty space to the right of the text,
     which is how a technically-correct shimmer ends up looking like nothing. */
  display: inline-block;
  background: linear-gradient(90deg,
    var(--shim-base) 35%, var(--shim-hot) 50%, var(--shim-base) 65%);
  background-size: 200% 100%;
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: shimmer 1.5s linear infinite;
}
/* 100% -> 0% walks the highlight from the left edge to the right edge and no
   further. Sweeping to a negative position parks it off-screen for half the
   cycle, so the line sits dead between glints. */
@keyframes shimmer {
  from { background-position: 100% 0; }
  to { background-position: 0% 0; }
}
@media (prefers-reduced-motion: reduce) {
  #status.waiting { animation: none; }
}
.note {
  font-size: 0.95rem; color: var(--muted); line-height: 1.65;
  margin-top: 0; margin-bottom: 1.1rem;
}
```

**config.json**

```json
{
  "description": "A real KataGo network (b6c96) playing full-size Go in the browser. 1.16 MB of weights, compiled to WGSL by tensorgrad.",
  "dependencies": {
    "domeleon": "^0.6.3",
    "tensorgrad": "^0.5.1"
  }
}
```