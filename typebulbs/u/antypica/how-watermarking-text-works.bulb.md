---
format: typebulb/v1
name: How Watermarking Text Works
---

**code.tsx**

```tsx
import {
  Module, Linear, LayerNorm, compileForward, checkWebGPU,
  add, mul, matmul, sum, reshape, swapAxes,
  splitHeads, mergeHeads, softmaxCausal, gelu,
  type Tensor,
} from 'tensorgrad'
import {
  App, Component, a, div, h1, h2, h3, p, span, strong, em, button, inputRange, inputTextArea,
  table, thead, tbody, tr, th, td,
  type VElement,
} from 'domeleon'

// ============================================================================
//  The language model
// ============================================================================

// TinyStories-1M (roneneldan/TinyStories-1M), a GPT-Neo decoder trained only on
// synthetic three-year-old-vocabulary stories. 3.75M parameters, of which the 50257-row
// embedding table — also the tied output head — is 3.2M and the entire eight-layer
// transformer stack only 0.4M. The copy shipped here is 3.63M: the exporter keeps the
// first 256 rows of the 2048-row position table, which is all a 256-token window reaches.
const D = 64, L = 8, HEADS = 16, VOCAB = 50257
const HEAD_DIM = D / HEADS

// The compiled graph is one fixed length. 256 is also GPT-Neo's local-attention window:
// this checkpoint alternates global and local(256) attention layers, and below 256 tokens
// the two are the same function, so one causal attention is exact. Above it they diverge.
const CTX = 256

const INPUTS = { embed: [1, CTX, D], sel: [1, CTX] } as const

class Block extends Module {
  ln1 = new LayerNorm(D)
  q = new Linear(D, D, { bias: false })
  k = new Linear(D, D, { bias: false })
  v = new Linear(D, D, { bias: false })
  attnOut = new Linear(D, D)
  ln2 = new LayerNorm(D)
  fc = new Linear(D, 4 * D)
  proj = new Linear(4 * D, D)
}

class TinyStories extends Module {
  // Present as a parameter for the output head only. The input-side lookup happens on the
  // CPU: tensorgrad's `embedding` composes to `oneHot @ table`, and a [1,256,50257] one-hot
  // is 51MB of tensor per step to express what is really a gather of 256 rows.
  wte = this.param([VOCAB, D])
  blocks: Block[]
  lnf = new LayerNorm(D)
  constructor() {
    super()
    this.blocks = Array.from({ length: L }, () => new Block())
  }
}

// GPT-Neo is PRE-norm, and its attention does NOT divide qk by sqrt(headDim). That missing
// scale is a Mesh-Tensorflow inheritance and the single easiest way to get this port wrong:
// adding it still yields fluent stories, drawn from a distribution ~45% off the real one.
// scripts/pack-tinystories.mjs checks the unscaled form against HuggingFace's own logits on
// every build — though against its own CPU reference pass, so what that proves is the
// weights and the arithmetic, not this graph: the two are separate implementations that
// share the checkpoint and a 1e-5 layernorm epsilon.
function block(b: Block, h: Tensor): Tensor {
  const a = b.ln1.fwd(h)
  const q = splitHeads(b.q.fwd(a), HEADS)
  const k = splitHeads(b.k.fwd(a), HEADS)
  const v = splitHeads(b.v.fwd(a), HEADS)
  const attn = softmaxCausal(matmul(q, swapAxes(k, -1, -2)), -1)
  // 16 batched [256,256] @ [256,4] products — head dim 4, because the checkpoint runs 16
  // heads over 64 channels. This is the shape that spent this page's first month broken on
  // every Android phone: tensorgrad emitted a kernel Qualcomm's Adreno driver miscompiled,
  // so the stories came out as one word repeated and the detector read a mark that was not
  // there. Fixed in the library as of 0.4.7, which this bulb requires.
  const ctx = mergeHeads(matmul(attn, v))
  const h2 = add(h, b.attnOut.fwd(ctx))
  const m = b.ln2.fwd(h2)
  return add(h2, b.proj.fwd(gelu(b.fc.fwd(m), { approximate: 'tanh' })))
}

// `sel` is one-hot over positions: it picks the last real token's row out of the padded
// window. Padding needs no mask of its own — attention is causal, so a real position can
// never see a later pad.
function forward(m: TinyStories, { embed, sel }: { embed: Tensor; sel: Tensor }): Tensor {
  let h = embed
  for (const b of m.blocks) h = block(b, h)
  h = m.lnf.fwd(h)
  const last = sum(mul(h, reshape(sel, [1, CTX, 1])), 1)
  return matmul(last, swapAxes(m.wte, -1, -2))
}

// ============================================================================
//  The watermark: tournament sampling
// ============================================================================

// Dathathri et al., "Scalable watermarking for identifying large language model outputs",
// Nature 634 (2024) — the SynthID-Text scheme, deployed in Gemini and open-sourced in
// HuggingFace Transformers as SynthIDTextWatermarkLogitsProcessor.
//
// The construction in one paragraph. At each step the model produces its distribution over
// the vocabulary. Draw N^m candidate words from THAT distribution, independently and with
// replacement, and run them through m knockout layers: in each layer the survivors are split
// into matches of N, and the winner of a match is the candidate whose coin came up heads,
// ties broken uniformly at random. The coins come from the key. The last word standing is
// emitted.
//
// What that buys, and it is the whole reassurance: every candidate in the bracket was drawn
// from the model's own distribution before the key saw it, so the key can never promote a
// word the model would not have said. It only chooses among words the model was already
// willing to say. At N = 2 the choosing is exactly fair on average over keys — proved in the
// paper, derived and checked by brute force in scripts/check-tournament-closed-form.mjs.
//
// The value of the key is arbitrary; this one is fixed so runs are reproducible.
const KEY = 0x7E1C4A93

/** 32-bit integer hash (murmur3 finalizer over a mixed pair), lifted from the two sibling
 *  watermarking bulbs. Called about V*m times per generated token here, so its cost is the
 *  cost of the whole scheme. */
function hash32(a: number, b: number): number {
  let x = (a ^ Math.imul(b, 0x9E3779B1)) >>> 0
  x = Math.imul(x ^ (x >>> 16), 0x85EBCA6B) >>> 0
  x = Math.imul(x ^ (x >>> 13), 0xC2B2AE35) >>> 0
  return (x ^ (x >>> 16)) >>> 0
}

/** H in the paper: how many preceding tokens are hashed with the key to seed a position.
 *
 *  This is the difference that makes the whole scheme cheap to detect. The distortion-free
 *  bulb's key is one long sequence and a text may start anywhere inside it, which forces the
 *  detector to search every alignment and leaves no formula to write the null down with. Here
 *  the seed travels with the text: any reader holding the key can recompute it from the four
 *  words in front of them, at any position, with nothing to align and nothing to search. */
const SEED_CONTEXT = 4

/** The layer count the paper runs its experiments at, and the ceiling on the slider.
 *
 *  NOT "the deployed value". The paper says "Unless otherwise mentioned, for all SynthID-Text
 *  experiments, we use m = 30 tournament layers", and separately that the non-distortionary
 *  configuration is productionized in Gemini. It never joins the two, and never publishes
 *  Gemini's own m, H or scorer. Copy on this page kept being regenerated from this comment, so
 *  the wording here is load-bearing: say published, never deployed. See specs/watermarking.md
 *  A.5. What the paper DOES pin about the deployment is the non-distortion level, single
 *  sequence at K = 1, so "runs in Gemini" about the scheme itself is fair and stays. */
const MAX_LAYERS = 30

/** r_t: the seed for the position that follows `ids[t-SEED_CONTEXT .. t-1]`. */
function contextSeed(ids: readonly number[], t: number): number {
  let s = KEY
  for (let i = t - SEED_CONTEXT; i < t; i++) s = hash32(s, ids[i]!)
  return s
}

/** g_l(x, r): the key's coin for word x in layer l at seed r, Bernoulli(0.5).
 *
 *  The paper writes it as F_g^-1(h(x, l, r) / 2^n_sec) with F_g = Bernoulli(0.5), which is
 *  one hashed bit per (word, layer, position). That is what this is: the top bit of a hash
 *  of the word under a per-layer seed. Reading the m layers off m separate bits of ONE hash
 *  would be about twice as fast and is a common shortcut; it is not taken, because then the
 *  layers' independence rests on the hash's avalanche rather than on the construction, and
 *  the whole test assumes those coins are independent. `selftest` measures the per-layer
 *  heads rate on unmarked text either way. */
const layerSeed = (r: number, layer: number) => hash32(r, layer)
const coinAt = (seed: number, tok: number) => hash32(seed, tok) >>> 31

/** How many of this word's m coins came up heads at this position. The detector's entire
 *  arithmetic, and the number the story lights words by. */
function headsFor(r: number, tok: number, layers: number): number {
  let h = 0
  for (let l = 0; l < layers; l++) h += coinAt(layerSeed(r, l), tok)
  return h
}

/** Every coin of one word at one position, for the strip in the inspector. */
function coinsFor(r: number, tok: number, layers: number): Uint8Array {
  const out = new Uint8Array(layers)
  for (let l = 0; l < layers; l++) out[l] = coinAt(layerSeed(r, l), tok)
  return out
}

// ---- the closed form -------------------------------------------------------

/** One tournament layer, applied to a whole distribution at once.
 *
 *  m = 30 layers means 2^30 candidates, and nobody draws a billion words per step. They do
 *  not have to: the tournament's effect on the distribution has an exact closed form costing
 *  one pass over the vocabulary per layer. With m1 the probability mass sitting on heads,
 *
 *      heads word x:   p'(x) = p(x) * [1 - (1 - m1)^N] / m1
 *      tails word x:   p'(x) = p(x) * (1 - m1)^(N - 1)
 *
 *  and at the default N = 2 that is simply p(x)*(2 - m1) for heads and p(x)*(1 - m1) for
 *  tails. Each layer's survivors are i.i.d. draws from the previous layer's distribution,
 *  because the matches are disjoint groups of independent draws, so applying this m times
 *  with a fresh coin vector each time is the whole sampler.
 *
 *  Derived rather than quoted, and checked two ways: scripts/check-tournament-closed-form.mjs
 *  enumerates every coin assignment over a five-word distribution against exact enumeration
 *  of all V^N candidate tuples (agreement to 1e-15), and the `agreement` probe here plays a
 *  hundred thousand literal brackets against it on the model's own distribution. Ties are
 *  broken uniformly over SLOTS rather than over distinct words, which matters because draws
 *  are with replacement and one word can hold several slots.
 *
 *  `ids` names the word behind each slot of `q`, for the fairness check, which runs the same
 *  rule over a few thousand words rather than the whole vocabulary. Without it a slot is its own
 *  word. One implementation either way: the page's central claim is that this formula and the
 *  literal bracket agree, and a second copy of it would make that comparison worth less.
 *
 *  `competitors` is 2 everywhere the page can reach. It stays a parameter because the paper's
 *  distortionary setting is the only positive control this check has: a measurement that has only
 *  ever printed "nothing here" cannot be told apart from one that always would, so `selftest`
 *  runs it at 3 and requires a bias to show. Nothing on the page offers the choice. */
function tournamentLayer(
  q: Float64Array, seed: number, coin: Uint8Array, ids?: Int32Array, competitors = 2,
): void {
  const n = q.length
  let m1 = 0
  for (let v = 0; v < n; v++) {
    const c = coinAt(seed, ids ? ids[v]! : v)
    coin[v] = c
    if (c) m1 += q[v]!
  }
  // Every word with any mass sits on the same side. The layer is then a formality: whichever
  // side that is, its multiplier works out to 1 and the other side has nothing to scale.
  if (m1 <= 0 || m1 >= 1) return
  const heads = (1 - (1 - m1) ** competitors) / m1
  const tails = (1 - m1) ** (competitors - 1)
  for (let v = 0; v < n; v++) q[v] = q[v]! * (coin[v] ? heads : tails)
}

/** The distribution tournament sampling actually emits from, at one position under one key.
 *  Writes into `out` and returns it. */
function tournamentDistribution(
  probs: Float64Array, r: number, layers: number, out: Float64Array, coin: Uint8Array,
): Float64Array {
  out.set(probs)
  for (let l = 0; l < layers; l++) tournamentLayer(out, layerSeed(r, l), coin)
  return out
}

// ---- the literal bracket ---------------------------------------------------

interface BracketSlot { id: number; text: string; heads: number }
interface BracketRound {
  /** The entrants to this layer, in match order: consecutive pairs. */
  slots: BracketSlot[]
  /** Per match, which of the pair won, 0 or 1. */
  wonAt: number[]
}
interface Bracket {
  rounds: BracketRound[]
  /** Distinct words among the drawn candidates. One means the position was never in doubt. */
  distinct: number
}

/** Running totals of `p` into `cdf`, the form `pickFrom` searches. Returns `cdf`. */
function cumulate(p: Float64Array, cdf: Float64Array): Float64Array {
  let acc = 0
  for (let v = 0; v < p.length; v++) { acc += p[v]!; cdf[v] = acc }
  return cdf
}

/** Sample one index from a cumulative distribution. */
function pickFrom(cdf: Float64Array, u: number): number {
  const target = u * cdf[cdf.length - 1]!
  let lo = 0, hi = cdf.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (cdf[mid]! < target) lo = mid + 1
    else hi = mid
  }
  return lo
}

/** The tournament played out for real: 2^m words drawn from the model, then m knockout
 *  layers. Only reachable for small brackets, which is the point of drawing it — a reader
 *  understands a knockout bracket before they have finished looking at it. Above sixteen
 *  slots the closed form above emits the same distribution without the draws. */
function playBracket(
  cdf: Float64Array, r: number, layers: number, rand: () => number, label: (id: number) => string,
): { id: number; bracket: Bracket } {
  let alive: number[] = []
  for (let i = 0; i < 2 ** layers; i++) alive.push(pickFrom(cdf, rand()))
  const distinct = new Set(alive).size
  const rounds: BracketRound[] = []
  for (let l = 0; l < layers; l++) {
    const seed = layerSeed(r, l)
    const coins = alive.map(id => coinAt(seed, id))
    const slots = alive.map((id, i) => ({ id, text: label(id), heads: coins[i]! }))
    const wonAt: number[] = []
    const next: number[] = []
    for (let i = 0; i < alive.length; i += 2) {
      // Two matching coins single out neither slot, so the match goes to a fair toss. Tossed over
      // SLOTS rather than over words, which matters because draws are with replacement and one
      // word can hold both: that is what makes the closed form come out the way it does.
      const at = coins[i] === coins[i + 1]
        ? (rand() < 0.5 ? 0 : 1)
        : coins[i]! > coins[i + 1]! ? 0 : 1
      wonAt.push(at)
      next.push(alive[i + at]!)
    }
    rounds.push({ slots, wonAt })
    alive = next
  }
  return { id: alive[0]!, bracket: { rounds, distinct } }
}

// ---- detection -------------------------------------------------------------

/** log of the gamma function: Lanczos, g = 5, six coefficients. Here only to build a log
 *  binomial coefficient. */
function lgamma(x: number): number {
  const COF = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
  ]
  let y = x, t = x + 5.5
  t -= (x + 0.5) * Math.log(t)
  let s = 1.000000000190015
  for (let j = 0; j < 6; j++) s += COF[j]! / ++y
  return -t + Math.log(2.5066282746310005 * s / x)
}

/** One-sided p-value: P(Binomial(n, 1/2) >= k), the chance a keyless writer's coins run this
 *  far above half.
 *
 *  Summed exactly rather than through the normal approximation. At one half the binomial is
 *  symmetric and the Gaussian is a far better fit than it was on the green list bulb's skewed
 *  quarter, but n here runs to a few thousand terms and summing them is free, so there is no
 *  reason to quote an approximation as an odds. The z the meter reports is untouched: it is
 *  the paper's statistic. */
function binomialTail(k: number, n: number): number {
  const from = Math.ceil(k)
  if (n <= 0 || from <= 0) return 1
  if (from > n) return 0
  let term = Math.exp(lgamma(n + 1) - lgamma(from + 1) - lgamma(n - from + 1) - n * Math.LN2)
  let s = term
  for (let i = from; i < n; i++) {
    term *= (n - i) / (i + 1)
    s += term
  }
  return Math.min(1, s)
}

/** The chance a word lights by luck alone: P(Binomial(layers, 1/2) > layers/2). Half for an
 *  odd number of layers, less for an even one, since a level split is not a majority. */
function litChance(layers: number): number {
  return binomialTail(Math.floor(layers / 2) + 1, layers)
}

interface Score {
  /** Positions the test counted. */
  counted: number
  /** Coins behind those positions: counted * layers. */
  coins: number
  heads: number
  z: number
  p: number
  /** Per token: heads out of `layers`, or -1 for a position the test could not count. */
  marks: Int8Array
  /** Heads per layer, over the counted positions. Flat across layers is what independent
   *  coins look like; a layer standing out would mean the per-layer seeds are correlated,
   *  which is the one way the g-value construction here could be quietly wrong. Measured from
   *  the text rather than from the generator, so it reads unmarked text too, which is the case
   *  that would expose it. */
  perLayer: Int32Array
  /** Positions skipped because their four-word context had already been seen. */
  repeats: number
  /** Positions with no full context window in front of them, so at most SEED_CONTEXT. */
  contextless: number
  /** Words past the cap, read by nobody. */
  dropped: number
}

/** Whether a word's coins came up heads often enough to light it. A strict majority is the one
 *  threshold statable in a legend without a table, and `-1` is a position the test could not
 *  count. Every mark on both tabs and the `lit` figure in `stats` read from here. */
const isLit = (heads: number, layers: number) => heads >= 0 && heads * 2 > layers

/** How many of a scored document's words light. */
const litCount = (s: Score, layers: number) => [...s.marks].filter(m => isLit(m, layers)).length

/** The figures any scored text has, so the `stats` and `read` probes never report one quantity
 *  two ways. Rounded here because these go to a terminal, not into more arithmetic. */
function scoreSummary(s: Score) {
  return {
    counted: s.counted, coins: s.coins, heads: s.heads,
    headRate: +(s.heads / Math.max(1, s.coins)).toFixed(3),
    z: +s.z.toFixed(2), p: s.p, repeats: s.repeats,
  }
}

/** The longest text the detector will score. Generous, because unlike the distortion-free
 *  bulb's permutation test this is one pass of a few hashes per word. */
const TEST_MAX = 4000

/** Detection, which needs no model, no prompt and no record of how the text was written: only
 *  the key, the text and the same tokenizer. Walk the words, recompute each position's seed
 *  from the four words before it, look up the coins of the word that is actually there, and
 *  count the heads. Under the null every coin is an independent fair coin, so the heads are
 *  Binomial(coins, 1/2) and the z-score follows from a formula.
 *
 *      Score(x) = (1 / mT) * sum_t sum_l g_l(x_t, r_t)
 *
 *  Repeated context masking, Algorithm 3 of the paper. A position whose four-word context has
 *  already been seen is skipped, because reusing a seed reuses the coins and breaks the
 *  independence the test assumes. The generator applies the same rule and leaves those
 *  positions unwatermarked, so the two agree exactly on which positions carry a mark. On text
 *  as loopy as a tiny model's this matters more than it would on a large one. */
function scoreTokens(ids: readonly number[], layers: number): Score {
  const dropped = Math.max(0, ids.length - TEST_MAX)
  const n = ids.length - dropped
  const marks = new Int8Array(n).fill(-1)
  const perLayer = new Int32Array(layers)
  const seen = new Set<number>()
  let counted = 0, heads = 0, repeats = 0
  for (let t = SEED_CONTEXT; t < n; t++) {
    const r = contextSeed(ids, t)
    if (seen.has(r)) { repeats++; continue }
    seen.add(r)
    let h = 0
    for (let l = 0; l < layers; l++) {
      const c = coinAt(layerSeed(r, l), ids[t]!)
      h += c
      perLayer[l] += c
    }
    marks[t] = h
    counted++
    heads += h
  }
  const coins = counted * layers
  const z = coins > 0 ? (heads - coins / 2) / Math.sqrt(coins / 4) : 0
  return {
    counted, coins, heads, z, p: binomialTail(heads, coins), marks, perLayer, repeats,
    contextless: Math.min(SEED_CONTEXT, n), dropped,
  }
}

/** Deterministic PRNG. Tournament sampling draws its candidates for real, so a fixed key
 *  leaves plenty of randomness in the loop and there is no determinism trap of the kind the
 *  distortion-free bulb has to work around with a random key offset. Fixing the seed is what
 *  makes a session reproducible; only "write another" steps it. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6D2B79F5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Hand the event loop a turn between chunks, without going through a timer.
 *
 *  `setTimeout(0)` is the obvious way to write this and it is a trap. A tab that has been in the
 *  background for five minutes has its timers throttled to roughly one call a minute, so a loop
 *  yielding between chunks turns half a second of work into a quarter of an hour: switch away
 *  mid-check, come back, and the count has not moved. Observed on the distortion-free bulb,
 *  where this helper was written, not theorised.
 *
 *  A MessageChannel post is a macrotask that is not throttled that way. The page still gets to
 *  paint between chunks while it is visible, and the work still finishes at full speed while it
 *  is not. The queue is there because two checks can briefly overlap, one of them being
 *  cancelled, and a single resolver slot would strand whichever promise it overwrote. */
const yieldToLoop = (() => {
  const channel = new MessageChannel()
  const waiting: (() => void)[] = []
  channel.port1.onmessage = () => waiting.shift()?.()
  return () => new Promise<void>(done => { waiting.push(done); channel.port2.postMessage(0) })
})()

/** Where the meter stops hedging, and where it starts.
 *
 *  Deliberately stricter than the conventional two standard deviations. Under no watermark the
 *  score is a fair coin count, so a loose cutoff hands a "probably watermarked" to a share of
 *  keyless texts, and on a page built to defuse an argument about marked text that is the
 *  expensive direction to be wrong in. It is also the error a reader meets personally, by
 *  pasting their own writing into the Detect tab. At z = 2.5 that costs about one keyless text
 *  in 160, and at z = 4 about one in 32,000.
 *
 *  Both the meter's word and the paragraph under it read from these, because when they were two
 *  separate numbers they drifted and contradicted each other on screen. */
const Z_MARKED = 4
const Z_MAYBE = 2.5

/** Where the fairness check stops calling a distance roughness. Set well clear of the fair case's
 *  own spread rather than tight to it: repeated runs of the pairing the page ships land between
 *  0.3 and 1.6 times the roughness, and the paper's distortionary setting reads about 9. Three
 *  sits in that gap, so the reading never hedges about the only configuration on offer. Read by
 *  the card's verdict and by `selftest`, both sides of the same judgement. */
const FAIR_RATIO = 3

/** Counted positions below which the meter says nothing at all. Read by the headline, the
 *  verdict beside it and the paragraph under it, for the same reason the two z thresholds are. */
const MIN_COUNTED = 15

// ============================================================================
//  Byte-level BPE (GPT-2 / GPT-Neo)
// ============================================================================

/** GPT-2's byte-to-printable-character map: every one of the 256 byte values gets a
 *  distinct visible codepoint, so a token string never contains whitespace or control
 *  characters and merges can be done with plain string operations. */
function byteEncoder(): string[] {
  const bs: number[] = []
  for (let i = 0x21; i <= 0x7E; i++) bs.push(i)
  for (let i = 0xA1; i <= 0xAC; i++) bs.push(i)
  for (let i = 0xAE; i <= 0xFF; i++) bs.push(i)
  const cs = bs.slice()
  let n = 0
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++ }
  const out = new Array<string>(256)
  for (let i = 0; i < bs.length; i++) out[bs[i]!] = String.fromCodePoint(cs[i]!)
  return out
}

const SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

class Tokenizer {
  #vocab: string[]
  #ids = new Map<string, number>()
  #ranks = new Map<string, number>()
  #b2c = byteEncoder()
  #c2b = new Map<string, number>()
  #cache = new Map<string, number[]>()
  #one = new Map<number, string>()
  readonly eos: number

  constructor(spec: { vocab: string[]; merges: string[]; eos: number }) {
    this.#vocab = spec.vocab
    spec.vocab.forEach((t, i) => this.#ids.set(t, i))
    spec.merges.forEach((m, i) => this.#ranks.set(m, i))
    this.#b2c.forEach((c, b) => this.#c2b.set(c, b))
    this.eos = spec.eos
  }

  /** Greedy rank-ordered merge over one pre-split piece, already byte-encoded. */
  #bpe(piece: string): string[] {
    let symbols = Array.from(piece)
    if (symbols.length < 2) return symbols
    for (;;) {
      let bestRank = Infinity, bestAt = -1
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.#ranks.get(`${symbols[i]} ${symbols[i + 1]}`)
        if (rank !== undefined && rank < bestRank) { bestRank = rank; bestAt = i }
      }
      if (bestAt < 0) return symbols
      symbols = [
        ...symbols.slice(0, bestAt),
        symbols[bestAt]! + symbols[bestAt + 1]!,
        ...symbols.slice(bestAt + 2),
      ]
    }
  }

  encode(text: string): number[] {
    const out: number[] = []
    for (const [piece] of text.matchAll(SPLIT)) {
      const hit = this.#cache.get(piece)
      if (hit) { out.push(...hit); continue }
      let encoded = ''
      for (const byte of new TextEncoder().encode(piece)) encoded += this.#b2c[byte]!
      const ids: number[] = []
      for (const sym of this.#bpe(encoded)) {
        const id = this.#ids.get(sym)
        if (id !== undefined) ids.push(id)
      }
      this.#cache.set(piece, ids)
      out.push(...ids)
    }
    return out
  }

  decode(ids: readonly number[]): string {
    const bytes: number[] = []
    for (const id of ids) {
      for (const ch of this.#vocab[id] ?? '') bytes.push(this.#c2b.get(ch) ?? 0)
    }
    return new TextDecoder().decode(Uint8Array.from(bytes))
  }

  /** One token's text, memoised. A bracket labels up to sixteen slots per step and the
   *  inspector relabels them on every hover, so this is asked far more often than it changes. */
  one(id: number): string {
    let hit = this.#one.get(id)
    if (hit === undefined) { hit = this.decode([id]); this.#one.set(id, hit) }
    return hit
  }

  /** Per-token strings, for showing text a token at a time. A token can end mid-character, so
   *  this streams one decoder across the sequence rather than decoding each token alone: the
   *  character lands on the token that completes it, instead of becoming a replacement mark on
   *  both. Matters the moment anyone pastes anything that is not ASCII. */
  pieces(ids: readonly number[]): string[] {
    const dec = new TextDecoder()
    return ids.map(id => {
      const bytes: number[] = []
      for (const ch of this.#vocab[id] ?? '') bytes.push(this.#c2b.get(ch) ?? 0)
      return dec.decode(Uint8Array.from(bytes), { stream: true })
    })
  }
}

// ============================================================================
//  The checkpoint
// ============================================================================

interface QTensor {
  name: string; shape: number[]; kind: string; file: string
  dataOffset: number; scalesOffset?: number; rowStart?: number; rowCount?: number
}
interface Manifest {
  groupSize: number; bits: number; files: string[]
  dims: { D: number; L: number; H: number; VOCAB: number; CTX: number }
  tensors: QTensor[]
}

// Mirror of the unpack in scripts/pack-tinystories.mjs, which runs it against the
// HuggingFace reference logits on every build. `wte` arrives as row-aligned shards, each
// its own file, because no single asset may pass 2MB; they stitch back by row offset.
function unpackParams(manifest: Manifest, bins: Record<string, ArrayBuffer>) {
  const out: Record<string, Float32Array> = {}
  for (const t of manifest.tensors) {
    const buf = bins[t.file]!
    const total = t.shape.reduce((a, b) => a * b, 1)
    if (t.kind === 'f32') { out[t.name] = new Float32Array(buf, t.dataOffset, total); continue }
    const bits = Number(t.kind.slice(1))
    const cols = t.shape.length > 1 ? t.shape[t.shape.length - 1]! : total
    const n = t.rowCount === undefined ? total : t.rowCount * cols
    const at = t.rowStart === undefined ? 0 : t.rowStart * cols
    const packed = new Uint8Array(buf, t.dataOffset, bits === 4 ? n / 2 : n)
    const scales = new Float32Array(buf, t.scalesOffset!, n / manifest.groupSize)
    const v = out[t.name] ?? (out[t.name] = new Float32Array(total))
    const bias = 1 << (bits - 1)
    for (let i = 0; i < n; i++) {
      const s = scales[(i / manifest.groupSize) | 0]!
      const q = bits === 4 ? (packed[i >> 1]! >> ((i & 1) * 4)) & 15 : packed[i]!
      v[at + i] = (q - bias) * s
    }
  }
  return out
}

// Its own cache, so the three watermarking bulbs never evict each other's copy of the same
// checkpoint. The prefix is what the sweep below deletes stale versions by, and it collides
// with neither the green list bulb's `watermark-` nor the distortion-free bulb's
// `distortion-free-`.
const CACHE = 'tournament-weights-v1'

/** Fetch once, thereafter from CacheStorage — the checkpoint is immutable, so paying 6MB
 *  on every open is pure latency. Falls back to a plain fetch where the Cache API isn't
 *  there (and the whole path is best-effort: caching failing must not stop the bulb). */
async function cachedFetch(url: string, onProgress?: (got: number, total: number) => void) {
  const download = async (): Promise<ArrayBuffer> => {
    const res = await fetch(url)
    if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`)
    const total = Number(res.headers.get('content-length')) || 0
    if (!res.body || !onProgress || !total) return await res.arrayBuffer()
    const reader = res.body.getReader()
    const chunks: Uint8Array[] = []
    let got = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      got += value.byteLength
      onProgress(got, total)
    }
    const bytes = new Uint8Array(got)
    let off = 0
    for (const c of chunks) { bytes.set(c, off); off += c.byteLength }
    return bytes.buffer as ArrayBuffer
  }
  try {
    for (const name of await caches.keys()) {
      if (name.startsWith('tournament-') && name !== CACHE) await caches.delete(name)
    }
    const cache = await caches.open(CACHE)
    const hit = await cache.match(url)
    if (hit) {
      const buf = await hit.arrayBuffer()
      onProgress?.(buf.byteLength, buf.byteLength)
      return buf
    }
    const buf = await download()
    try { await cache.put(url, new Response(buf)) } catch { /* an optimisation, not a step */ }
    return buf
  } catch {
    return download()
  }
}

// ============================================================================
//  Generation
// ============================================================================

/** One candidate at one step, for the inspector when the bracket is too large to draw. */
interface Candidate { id: number; text: string; p: number; heads: number }

interface Step {
  id: number
  text: string
  /** The model's own probability for the word that came out. */
  p: number
  /** Shannon entropy of the model's distribution here, in bits: how much freedom this
   *  position had, and so how much mark it could possibly carry. A position with one possible
   *  word carries none, under this scheme or any other. */
  entropy: number
  /** The position's seed, and the emitted word's coins under it. Undefined where the position
   *  was not watermarked. */
  seed?: number
  coins?: Uint8Array
  heads?: number
  /** The tournament as it was actually played, when it was small enough to play. */
  bracket?: Bracket
  /** The words the model rated highest, with the coins they would have carried. What the
   *  inspector shows once the bracket is being computed rather than played out. */
  top: Candidate[]
  /** Why this position carries no mark. A repeated context is the paper's own masking rule;
   *  `opening` is a position with fewer than four words in front of it. */
  masked?: 'repeat' | 'opening'
  /** The word the model rated highest, when it is not the one that came out. Sampling, not
   *  the key, is what a reader misreads as a malfunction, and every candidate in the bracket
   *  was the model's own draw. */
  likeliest?: { text: string; p: number }
}

interface Settings {
  temperature: number
  watermark: boolean
  layers: number
  maxTokens: number
  seed: number
}

// Four layers rather than three, measured like for like on six seeds: four scores z = 4.68 to
// 5.85 where three scores 4.54 to 5.37, better at both ends because every layer is another coin
// the test counts. Both clear the threshold, so the margin is the whole of the case for four,
// along with sixteen slots still being a bracket worth drawing. An earlier note here claimed
// three layers "mostly reads probably watermarked" off three seeds; it does not, and the numbers
// above replace it.
//
// The seed is chosen over twelve of them, on two jobs: score well enough that the default reads
// "watermarked" rather than "probably", and open on a line that cannot be read as damage. A tiny
// model asked for one word at a time will say "her dolls and her dolls", and a reader meeting
// that in the WATERMARKED sample blames the watermark. Seed 6 scores z = 5.5 with nothing masked
// and carries all three teachable brackets. Seeds 1 and 2 score higher, 5.8 and 5.85, and open on
// "sunny day at night" and "hot winter day": contradictions in the second sentence, which is
// exactly what this criterion is here to catch.
const DEFAULTS: Settings = {
  temperature: 0.85, watermark: true, layers: 4, maxTokens: 110, seed: 6,
}

/** Above this many slots the bracket stops being a picture and starts being a wall, so the
 *  closed form takes over and the inspector says so. Sixteen is N = 2 at four layers. */
const BRACKET_MAX = 16

const SHOW_K = 6          // candidates listed when there is no bracket to draw
const CHECK_ROWS = 4      // words tabled by the fairness check

const DEFAULT_PROMPT = 'Once upon a time, there was a little girl named Lily.'

/** The Detect tab's human writing: Tolstoy, "What Is Art?" (1897), Maude translation.
 *
 *  A scheme that finds a mark in this has a bug, and it is the kind of bug that looks like a
 *  spectacular result rather than a bug, so it is worth a button of its own.
 *
 *  The em-dashes are Tolstoy's and stay. Section 8's sweep bans them from anything a reader
 *  sees, and this is the one exception the rule has to carry: repunctuating a quotation to
 *  satisfy a house style is misquoting, and this string is the only quoted text on the page. */
const HUMAN_WRITING =
  'I know that most men—not only those considered clever, but even those who are very clever, ' +
  'and capable of understanding most difficult scientific, mathematical, or philosophic ' +
  'problems—can very seldom discern even the simplest and most obvious truth if it be such as ' +
  'to oblige them to admit the falsity of conclusions they have formed, perhaps with much ' +
  'difficulty—conclusions of which they are proud, which they have taught to others, and on ' +
  'which they have built their lives.'

/** What the fairness check produces: the model's distribution at one position beside the one
 *  the scheme actually emits from, averaged over many keys. */
interface Fairness {
  keys: number
  /** Candidates per match this reading was taken at. Two everywhere the page can reach; the
   *  selftest is the only caller that asks for anything else, as its control. */
  competitors: number
  layers: number
  rows: { text: string; p: number; keyed: number }[]
  /** Total variation distance from the model's own distribution. */
  tv: number
  /** The distance a finite number of keys would leave behind even if the scheme were exactly
   *  fair, worked out from how much the per-key distributions actually vary. This is what makes
   *  the measurement readable: `tv` alone is a number with no scale, and comparing it against
   *  the same run at fewer keys turned out to be two noisy draws whose ratio was noisier still,
   *  because the distance is dominated by whichever word the model is most confident about. */
  tvNoise: number
  /** tv over tvNoise. About one is a distance made of nothing but roughness; well above one is
   *  a bias that no number of keys will remove. */
  ratio: number
  /** Mass of the words the check ran over. Always under one, and how far under depends on the
   *  temperature rather than being the millionth the support was meant to leave: see
   *  `checkFairness`. */
  covers: number
}

class Engine extends Component {
  status = 'starting…'
  progress = 0
  failed = false
  ready = false

  // Flat fields rather than a nested settings object: domeleon's inputs bind a target
  // component and a property path, and a flat path is the one these controls can write to.
  temperature = DEFAULTS.temperature
  watermark = DEFAULTS.watermark
  layers = DEFAULTS.layers
  maxTokens = DEFAULTS.maxTokens
  seed = DEFAULTS.seed

  prompt = DEFAULT_PROMPT
  steps: Step[] = []
  promptIds: number[] = []
  generating = false
  /** Whether the last run stopped because it ran out of length rather than because the model
   *  finished. A tiny model rarely reaches its end token inside a hundred tokens, so the story
   *  usually stops mid-phrase, and a sentence cut in half reads as the page having broken rather
   *  than as a budget being spent. */
  truncated = false
  selected = -1
  /** Token under the pointer, or -1. Preview; `selected` is the pinned one that outlives it. */
  hovered = -1
  /** Text in the detector pane. Diverges from the generated text as soon as it is edited,
   *  which is the point of having it be editable. */
  detectorText = ''

  fairness?: Fairness
  fairBusy = false
  fairDone = 0

  #model?: Awaited<ReturnType<typeof compileForward<TinyStories, typeof INPUTS>>>
  #tok?: Tokenizer
  #wte?: Float32Array
  #wpe?: Float32Array
  #started = false
  #run = 0
  #fairRun = 0
  /** Settles when any generation in flight has finished. The check needs a forward pass of its
   *  own whenever the prompt has changed under it, and two overlapping runs of one compiled
   *  graph write the same input buffers. */
  #idle: Promise<unknown> = Promise.resolve()
  /** The prompt's own logits, which is all the check ever needs from the model. Filled for
   *  free by the first step of every generation, so moving a slider re-measures without
   *  touching the GPU at all. That is what makes the reading cheap enough to follow a knob. */
  #promptLogits?: { prompt: string; logits: Float32Array }
  #fairTimer?: number
  #fairWanted = false
  /** The inputs behind the reading on screen. The check is a pure function of these, so when
   *  they have not moved there is nothing to recompute and reopening the tab is free. */
  #fairKey = ''
  #ready: Promise<boolean> = Promise.resolve(false)
  #scoreCache = new Map<string, Score>()
  // Scratch for the per-step closed form, which touches the whole vocabulary once per layer.
  #probs = new Float64Array(VOCAB)
  #emit = new Float64Array(VOCAB)
  #cdf = new Float64Array(VOCAB)
  #coin = new Uint8Array(VOCAB)

  override onAttached() {
    if (this.#started) return
    this.#started = true
    this.#ready = this.start()
  }

  whenReady() { return this.#ready }
  get tokenizer() { return this.#tok }

  get settings(): Settings {
    return {
      temperature: this.temperature, watermark: this.watermark, layers: this.layers,
      maxTokens: this.maxTokens, seed: this.seed,
    }
  }

  /** Back to the defaults with a patch on top. Distinct from the `set` probe, which patches
   *  whatever is already there: a selftest step wants a known starting point, not a delta. */
  resetSettings(patch: Partial<Settings> = {}) { Object.assign(this, DEFAULTS, patch) }

  setStatus(s: string, progress?: number) {
    this.status = s
    if (progress !== undefined) this.progress = progress
    tb.log(s)
    this.update()
  }

  async start(): Promise<boolean> {
    try {
      const gpu = await checkWebGPU()
      if (!gpu.ok) { this.failed = true; this.setStatus(gpu.message); return false }

      this.setStatus('Downloading the model…', 0.02)
      const manifest = JSON.parse(new TextDecoder().decode(
        await cachedFetch('assets/tinystories-1m.json'))) as Manifest

      const files = [...manifest.files, 'tokenizer.json']
      const got = new Array(files.length).fill(0)
      const size = new Array(files.length).fill(0)
      let shown = -1
      const onBytes = (i: number) => (received: number, total: number) => {
        got[i] = received; size[i] = total
        if (!size.every(s => s > 0)) return
        const done = got.reduce((a, b) => a + b, 0), all = size.reduce((a, b) => a + b, 0)
        const pct = Math.round((done / all) * 100)
        if (pct === shown) return
        shown = pct
        this.setStatus(`Downloading the model (${(done / 1e6).toFixed(1)} of ${(all / 1e6).toFixed(1)} MB)`,
          0.02 + 0.68 * (done / all))
      }
      const buffers = await Promise.all(
        files.map((f, i) => cachedFetch(`assets/${f}`, onBytes(i))))

      const bins: Record<string, ArrayBuffer> = {}
      manifest.files.forEach((f, i) => { bins[f] = buffers[i]! })
      this.#tok = new Tokenizer(JSON.parse(new TextDecoder().decode(buffers[files.length - 1]!)))

      this.setStatus('Unpacking the weights…', 0.74)
      await new Promise(r => setTimeout(r, 0))
      const params = unpackParams(manifest, bins)
      this.#wte = params['wte']!
      this.#wpe = params['wpe']!

      this.setStatus('Compiling the graph…', 0.82)
      this.#model = await compileForward({ model: new TinyStories(), forward, inputs: INPUTS })

      // wpe is deliberately not a graph parameter: positions are added on the CPU alongside
      // the token lookup, so uploadParams gets exactly the tensors the graph declares.
      const wanted = new Set(this.#model.paramNames as readonly string[])
      const own: Record<string, Float32Array> = {}
      for (const [name, value] of Object.entries(params)) if (wanted.has(name)) own[name] = value
      const missing = [...wanted].filter(n => !(n in own))
      if (missing.length) throw new Error(`checkpoint is missing: ${missing.join(', ')}`)

      this.setStatus('Uploading the weights to the GPU…', 0.92)
      await this.#model.uploadParams(own)

      this.progress = 1
      this.ready = true
      this.status = ''
      tb.log('Ready.')
      this.update()
      await this.generate()
      // A reader who opened the explainer while this was loading asked for a reading and got
      // "needs the model". Now they get it as soon as there is a model.
      if (this.#fairWanted) this.scheduleFairness(0)
      return true
    } catch (err) {
      this.failed = true
      this.setStatus(`Something broke while starting: ${(err as Error).message}`)
      return false
    }
  }

  /** Raw next-token logits for a token sequence. The embedding lookup and the position
   *  add both happen here, on the CPU, and the result is fed in as the graph's first
   *  hidden state. */
  async logitsFor(ids: readonly number[]): Promise<Float32Array> {
    const n = Math.min(ids.length, CTX)
    const from = ids.length - n
    const embed = new Float32Array(CTX * D)
    for (let t = 0; t < n; t++) {
      const row = ids[from + t]! * D
      for (let i = 0; i < D; i++) embed[t * D + i] = this.#wte![row + i]! + this.#wpe![t * D + i]!
    }
    const sel = new Float32Array(CTX)
    sel[n - 1] = 1
    const r = await this.#model!.run({ embed, sel })
    if (r.kind !== 'completed') throw new Error(`forward run ${r.kind}`)
    return r.output as Float32Array
  }

  /** The model's distribution over the WHOLE vocabulary at one position, after temperature.
   *
   *  No top-k cut, unlike the two sibling bulbs. Algorithm 2 draws its candidates from the
   *  model's distribution, and the closed form runs over the vocabulary in about a
   *  millisecond a layer, so there is no reason to hand the tournament a truncated one and
   *  then have to say so. Writes into the shared scratch buffer and returns it. */
  distribution(logits: Float32Array, temperature: number): Float64Array {
    const q = this.#probs
    const temp = Math.max(0.05, temperature)
    let max = -Infinity
    for (let v = 0; v < VOCAB; v++) if (logits[v]! > max) max = logits[v]!
    let total = 0
    for (let v = 0; v < VOCAB; v++) { const e = Math.exp((logits[v]! - max) / temp); q[v] = e; total += e }
    for (let v = 0; v < VOCAB; v++) q[v] = q[v]! / total
    return q
  }

  /** Kept separate from the body so the in-flight run is holdable: `#idle` is what stops the
   *  check from driving the compiled graph while generation is already driving it. */
  generate(): Promise<void> {
    const running = this.#generate()
    this.#idle = running.catch(() => {})
    return running
  }

  async #generate() {
    if (!this.ready || this.generating) return
    const run = ++this.#run
    const s = this.settings
    const tok = this.#tok!
    const rand = mulberry32(s.seed)

    this.generating = true
    this.steps = []
    this.selected = -1
    this.hovered = -1
    this.promptIds = tok.encode(this.prompt)
    this.detectorText = ''
    this.update()

    const ids = [...this.promptIds]
    // The masking rule, seeded with the prompt's own contexts so that generation and
    // detection agree exactly on which positions carry a mark. The detector walks the whole
    // document, prompt included, and a context first seen inside the prompt is one it will
    // skip later; leaving the generator's set empty would have it mark a position the
    // detector then declines to read.
    //
    // Strictly the positions BEFORE the first generated one, and `sampleStep` records each
    // context as it uses it. Running this bound one past the end instead files the first
    // generated position's own context before that position is reached, so every step in the
    // story reads as a repeat of itself and the whole run comes out unmarked at z = 0.
    const seen = new Set<number>()
    for (let t = SEED_CONTEXT; t < ids.length; t++) seen.add(contextSeed(ids, t))

    const budget = Math.min(s.maxTokens, CTX - this.promptIds.length)

    try {
      let finished = false
      for (let n = 0; n < budget; n++) {
        if (run !== this.#run) return
        const logits = await this.logitsFor(ids)
        // The first step's logits ARE the distribution the fairness check measures, so it is
        // handed a copy rather than left to compute its own. Copied because the runner may
        // reuse its output buffer between runs.
        if (n === 0) this.#promptLogits = { prompt: this.prompt, logits: Float32Array.from(logits) }
        const step = this.sampleStep(logits, ids, seen, s, rand, tok)
        if (step.id === tok.eos) { finished = true; break }
        this.steps.push(step)
        ids.push(step.id)
        this.update()
      }
      this.truncated = !finished
    } catch (err) {
      tb.log(`generation failed: ${(err as Error).message}`)
    } finally {
      if (run === this.#run) {
        this.detectorText = this.fullText()
        this.generating = false
        this.update()
      }
    }
  }

  /** One decoding step.
   *
   *  Three paths, and only the first is the scheme: a tournament small enough to play out
   *  slot by slot, a tournament computed in closed form because it is a billion candidates
   *  wide, and ordinary sampling for a position the scheme leaves alone. The last covers both
   *  the watermark being off and the paper's masking rule, which declines to re-mark a
   *  context it has already used. */
  sampleStep(
    logits: Float32Array, ids: readonly number[], seen: Set<number>,
    s: Settings, rand: () => number, tok: Tokenizer,
  ): Step {
    const probs = this.distribution(logits, s.temperature)
    const cdf = cumulate(probs, this.#cdf)

    const opening = ids.length < SEED_CONTEXT
    const seed = opening ? 0 : contextSeed(ids, ids.length)
    const masked: Step['masked'] = opening ? 'opening' : seen.has(seed) ? 'repeat' : undefined
    // Recorded as it is used, whether or not this position ends up carrying a mark, which is
    // what makes a later repeat a repeat. Exactly what the detector does as it walks the text.
    if (!opening) seen.add(seed)
    const marking = s.watermark && !masked

    let id: number
    let bracket: Bracket | undefined
    if (marking && 2 ** s.layers <= BRACKET_MAX) {
      const played = playBracket(cdf, seed, s.layers, rand, t => tok.one(t))
      id = played.id
      bracket = played.bracket
    } else if (marking) {
      const emit = tournamentDistribution(probs, seed, s.layers, this.#emit, this.#coin)
      id = pickFrom(cumulate(emit, cdf), rand())
    } else {
      id = pickFrom(cdf, rand())
    }

    let entropy = 0
    for (let v = 0; v < VOCAB; v++) { const q = probs[v]!; if (q > 1e-12) entropy -= q * Math.log2(q) }

    // `topIndices` hands them back in order, so the head of the list is the argmax and the
    // likeliest word costs no scan of its own.
    const top = topIndices(probs, SHOW_K)
    const mode = top[0]!
    if (!top.includes(id)) top.push(id)

    return {
      id,
      text: tok.one(id),
      p: probs[id]!,
      entropy,
      ...(masked ? { masked } : {}),
      ...(marking
        ? { seed, coins: coinsFor(seed, id, s.layers), heads: headsFor(seed, id, s.layers) }
        : {}),
      ...(bracket ? { bracket } : {}),
      ...(id === mode ? {} : { likeliest: { text: tok.one(mode), p: probs[mode]! } }),
      top: top.map(v => ({
        id: v,
        text: tok.one(v),
        p: probs[v]!,
        heads: marking ? headsFor(seed, v, s.layers) : 0,
      })),
    }
  }

  /** The generated continuation, without the prompt. */
  text(): string {
    return this.#tok?.decode(this.steps.map(s => s.id)) ?? ''
  }

  /** The whole document: the prompt as it was typed, then what the model wrote. What a
   *  detector is actually handed, because a finished text carries no mark saying where the
   *  prompt stopped. It is also the honest case: the opening line was written by a person,
   *  carries nothing, and dilutes the score. */
  fullText(): string {
    return this.prompt + this.text()
  }

  /** The document as generated, in tokens, without a round trip through text. The Generate
   *  tab needs the token stream to line its marks up with the spans it drew, and re-tokenizing
   *  prose can merge across the seam between the prompt and the first generated word. The
   *  Detect tab does re-tokenize, because a detector must. */
  documentIds(): number[] {
    return [...this.promptIds, ...this.steps.map(s => s.id)]
  }

  /** Memoised: the score is asked for on every render, and the detector's box is typed into. */
  scoreOf(ids: readonly number[]): Score {
    const key = `${this.layers}:${ids.length}:${ids.join(',')}`
    const hit = this.#scoreCache.get(key)
    if (hit) return hit
    const s = scoreTokens(ids, this.layers)
    if (this.#scoreCache.size > 8) this.#scoreCache.clear()
    this.#scoreCache.set(key, s)
    return s
  }

  /** What the Generate tab reads: the document it just wrote, scored with the key alone and no
   *  record of how it was written.
   *
   *  Not quite what a stranger would compute, and the difference is worth naming because both
   *  numbers can be on screen at once. This scores `documentIds`, the stream as generated, so
   *  the marks line up with the spans that were drawn. A stranger has to tokenize the text
   *  first, which is what the Detect tab does, and BPE can merge across the seam between the
   *  prompt and the first generated word. `selftest`'s `tokenization` row measures the gap. */
  liveScore(): Score {
    return this.scoreOf(this.documentIds())
  }

  /** What the Detect tab reads: whatever is in the box, tokenized from scratch. */
  detectScore(): { score: Score; ids: number[] } | undefined {
    if (!this.#tok || !this.detectorText.trim()) return undefined
    const ids = this.#tok.encode(this.detectorText)
    return { score: this.scoreOf(ids), ids }
  }

  /** The prompt's logits, from cache where generation has already paid for them. Only a prompt
   *  edited without a regeneration reaches the graph, and that waits for an idle one. */
  async promptLogits(): Promise<Float32Array> {
    const prompt = this.prompt
    if (this.#promptLogits?.prompt === prompt) return this.#promptLogits.logits
    await this.#idle
    if (this.#promptLogits?.prompt === prompt) return this.#promptLogits.logits
    const logits = Float32Array.from(await this.logitsFor(this.#tok!.encode(prompt)))
    this.#promptLogits = { prompt, logits }
    return logits
  }

  /** Re-measure once the reader stops moving a slider.
   *
   *  There was a button here, and it could not do anything. The check walks a fixed sequence of
   *  keys, so at unchanged settings it recomputes the same number to the last decimal: pressing
   *  it was indistinguishable from not pressing it, and it sat far enough above the figures
   *  that its one real effect, changing a setting first, was invisible. A reading that follows
   *  the knob is what was wanted, and at half a second a run it can. */
  scheduleFairness(delay = 300) {
    this.#fairWanted = true
    clearTimeout(this.#fairTimer)
    if (!this.ready) return
    this.#fairTimer = window.setTimeout(() => {
      const key = `${this.prompt}|${this.temperature}|${this.layers}`
      if (key === this.#fairKey && this.fairness) return
      // Recorded on success only: a run retired by a later one must not book the settings it
      // never finished measuring.
      void this.checkFairness().then(out => { if (out) this.#fairKey = key })
    }, delay)
  }

  // ---- the fairness check ---------------------------------------------------

  /** The claim the whole page rests on, measured rather than asserted.
   *
   *  Take one position and one distribution, and average the distribution tournament sampling
   *  emits from over thousands of random keys. Not sampled draws: the closed form gives the
   *  EXACT emission distribution per key, so the only error left in the average is the finite
   *  number of keys, and what is on screen is the scheme's own bias rather than the reader's
   *  patience.
   *
   *  Two readings, at a quarter of the keys and at all of them, because a single distance is
   *  unreadable on its own. So the check also works out, from how much the per-key
   *  distributions actually vary, how far off the average would land at this many keys if the
   *  scheme were exactly fair. Measured distance against that expectation is the whole reading:
   *  about one means there is nothing there but roughness, and several times one is a bias.
   *
   *  The earlier version compared the distance at a quarter of the keys against the distance at
   *  all of them, expecting a halving. That reads well and does not work: this model puts 88%
   *  of its mass on one word at the position being measured, so the distance is essentially one
   *  random number, and the ratio of two of those swung between 1.2 and 3.0 across runs of the
   *  same fair setting.
   *
   *  Run over the words carrying all but a millionth of the model's mass rather than all 50257,
   *  which buys the twenty-fold speedup that lets the check run at the published thirty layers.
   *  Two bounds sit on that support and which one binds depends on the temperature: at ordinary
   *  settings the millionth binds, and at the top of the temperature slider the 4096-word cap
   *  binds first instead, with `covers` reading about 0.998 at temperature 1.4.
   *
   *  **`base` is renormalised by `covers`, and it has to be.** Left unnormalised it sums to
   *  `covers` while `tournamentLayer` computes m1 over the same short support, so both
   *  multipliers come out slightly too large and the surviving mass creeps back toward 1 layer by
   *  layer. One layer leaves m1(2-m1) + (c-m1)(1-m1) = 1 - (1-c)(1-m1), so the deficit halves per
   *  layer and is gone well before thirty. The check then scored a distribution summing to 1
   *  against a `base` summing to 0.998 and called the difference watermark bias: an additive
   *  0.0008 that, unlike roughness, did NOT shrink as more keys were averaged, so the ratio
   *  stopped converging and turned back up. `verify` at 8000 keys sat right where it turns.
   *  Reproduced outside the bulb in scripts/check-fairness-support-bias.mjs.
   *
   *  Renormalising means both columns of the table describe the truncated distribution rather
   *  than the model's, which is the honest thing anyway: that IS what this check tests. They stay
   *  on one scale, so the comparison the card exists to make is unaffected, and at default
   *  settings `covers` is a millionth off 1 and nothing moves at all.
   *
   *  Read the ratio's TREND, never its level. The fair-case ratio is not 1 and is not constant:
   *  about 1.2 at thirty layers, 0.76 at four. `tvNoise` assumes the per-word mean is normal and
   *  the per-key values span many orders of magnitude, so it has no business being exact either
   *  way. FAIR_RATIO is set well clear of that spread for exactly this reason. */
  async checkFairness(keys = 2000, competitors = 2): Promise<Fairness | undefined> {
    // Deliberately not guarded on fairBusy: a reader dragging a slider supersedes their own
    // measurement, and the run counter below already retires the one they left behind.
    if (!this.ready) return this.fairness
    const run = ++this.#fairRun
    // Captured rather than read live inside the loop. The sliders write straight into these
    // fields, so a reader dragging one mid-run would otherwise have the layer count change half
    // way through a single measurement, and the row it files stamped with a setting it never
    // measured. The run counter retires the stale reading, but only once the next one starts.
    const layers = this.layers, temperature = this.temperature
    this.fairBusy = true
    this.fairDone = 0
    this.update()
    try {
      const tok = this.#tok!
      const probs = this.distribution(await this.promptLogits(), temperature)
      if (run !== this.#fairRun) return undefined
      const order = topIndices(probs, 4096)
      let covers = 0
      const support: number[] = []
      for (const v of order) {
        support.push(v)
        covers += probs[v]!
        if (covers >= 1 - 1e-6) break
      }

      const S = support.length
      // Renormalised, not raw. See the note above: an unnormalised base makes the tournament
      // appear to manufacture the mass the truncation removed, and scores it as bias.
      const base = Float64Array.from(support, v => probs[v]! / covers)
      const work = new Float64Array(S)
      const coin = new Uint8Array(S)
      const acc = new Float64Array(S)
      const accSq = new Float64Array(S)
      // Coins are looked up by word id and this runs over a subset of the vocabulary, so
      // `tournamentLayer` is handed the word behind each slot rather than the slot's index.
      const idsOf = Int32Array.from(support)

      for (let k = 0; k < keys; k++) {
        if (run !== this.#fairRun) return undefined
        work.set(base)
        const r = hash32(0x51ED270B, k)
        for (let l = 0; l < layers; l++) {
          tournamentLayer(work, layerSeed(r, l), coin, idsOf, competitors)
        }
        for (let i = 0; i < S; i++) acc[i] = acc[i]! + work[i]!
        for (let i = 0; i < S; i++) accSq[i] = accSq[i]! + work[i]! * work[i]!
        if ((k & 63) === 63) {
          this.fairDone = k + 1
          this.update()
          await yieldToLoop()
        }
      }

      // E|mean - truth| for a mean of `keys` draws is its standard error times root two over
      // pi, so the expected distance is that summed over the words and halved like any other
      // total variation distance.
      let tv = 0, tvNoise = 0
      for (let i = 0; i < S; i++) {
        const mean = acc[i]! / keys
        tv += Math.abs(mean - base[i]!)
        const variance = Math.max(0, accSq[i]! / keys - mean * mean)
        tvNoise += Math.sqrt(variance / keys) * Math.SQRT2 / Math.sqrt(Math.PI)
      }
      const shown = support.slice(0, CHECK_ROWS)
      const out: Fairness = {
        keys, competitors, layers,
        tv: tv / 2, tvNoise: tvNoise / 2, ratio: (tv / 2) / Math.max(1e-12, tvNoise / 2), covers,
        // Read from `base` rather than from `probs`, which is the shared scratch buffer: by the
        // time this line runs the check has yielded many times and a generation may have
        // written its own step into it. `base` is this run's own copy, taken before any yield.
        rows: shown.map((v, i) => ({
          text: tok.one(v),
          p: base[i]!,
          keyed: acc[i]! / keys,
        })),
      }
      this.fairness = out
      return out
    } finally {
      if (run === this.#fairRun) {
        this.fairBusy = false
        this.update()
      }
    }
  }

  /** Two independent implementations of the same rule, asked to agree.
   *
   *  Play the literal bracket many thousands of times at one position under one key and
   *  compare where the winners landed against the closed form's distribution at that same key.
   *  The control is the same draws measured against the model's own distribution, which the
   *  tournament is NOT emitting from at a fixed key: without it a small distance would only
   *  prove that both numbers were roughly the model's. */
  async checkAgreement(draws = 100000) {
    if (!this.ready) return undefined
    const tok = this.#tok!
    const ids = tok.encode(this.prompt)
    const probs = this.distribution(await this.logitsFor(ids), this.temperature)
    const cdf = cumulate(probs, this.#cdf)
    const model = Float64Array.from(probs)

    const r = contextSeed(ids, ids.length)
    const emit = tournamentDistribution(probs, r, this.layers, this.#emit, this.#coin)
    const closed = Float64Array.from(emit)

    const rand = mulberry32(0x9F1C)
    const counts = new Float64Array(VOCAB)
    for (let i = 0; i < draws; i++) {
      counts[playBracket(cdf, r, this.layers, rand, () => '').id]++
    }

    let tvClosed = 0, tvModel = 0
    for (let v = 0; v < VOCAB; v++) {
      const f = counts[v]! / draws
      tvClosed += Math.abs(f - closed[v]!)
      tvModel += Math.abs(f - model[v]!)
    }
    return {
      draws, layers: this.layers,
      tvClosedForm: +(tvClosed / 2).toFixed(4),
      tvModel: +(tvModel / 2).toFixed(4),
      // The bracket must land far nearer the closed form than the un-tournamented model.
      agree: tvClosed / 2 < 0.02 && tvClosed < tvModel / 2,
    }
  }

  // ---- terminal probes (typebulb send) --------------------------------------

  /** Numbers for one run, the way a sweep wants them. */
  runStats() {
    const s = this.liveScore()
    return {
      tokens: this.steps.length,
      ...scoreSummary(s),
      lit: litCount(s, this.layers),
      masked: this.steps.filter(st => st.masked === 'repeat').length,
      entropy: +(this.steps.reduce((a, st) => a + st.entropy, 0) / Math.max(1, this.steps.length)).toFixed(2),
      // Taken from the detector's own walk, so it reads unmarked text as readily as marked.
      // Computing it from the generator's steps instead measured nothing at all with the
      // watermark off, because with the key idle no step carries coins.
      perLayerRate: [...s.perLayer].map(n => +(n / Math.max(1, s.counted)).toFixed(3)),
      // The three cases the bracket exists to show, located rather than hoped for.
      bracketCases: this.bracketCases(),
    }
  }

  /** Where in the current story each teachable bracket lives: a position with many different
   *  words in it, a bracket that came out unanimous, and a position where one word held several
   *  slots and won for that reason.
   *
   *  The middle key is `unanimous`, not `forced`, and the distinction is the whole of P1 in
   *  specs/watermarking-corrections.md. What `distinct === 1` detects is that every slot drew
   *  the same word, which is a fact about the DRAW. It is not evidence that only one word was
   *  possible: slots are filled with replacement, so P(all identical) is about p(top)^slots and
   *  runs at 44% for p = 0.95. Measured on the shipped default, this case lands on a word the
   *  model gave 95%, and the popover prints that number. Naming it `forced` is how the false
   *  version got into four strings of copy, so the name stays descriptive of what was seen. */
  bracketCases() {
    let spread = -1, unanimous = -1, repeated = -1
    let widest = 0
    for (let i = 0; i < this.steps.length; i++) {
      const b = this.steps[i]!.bracket
      if (!b) continue
      const slots = b.rounds[0]!.slots
      if (b.distinct === 1 && unanimous < 0) unanimous = i
      if (b.distinct > widest) { widest = b.distinct; spread = i }
      if (repeated < 0 && b.distinct > 1 && b.distinct < slots.length) {
        const winner = this.steps[i]!.id
        if (slots.filter(s => s.id === winner).length > 1) repeated = i
      }
    }
    return {
      spread: spread < 0 ? undefined : { at: spread, distinct: widest, text: this.steps[spread]!.text },
      unanimous: unanimous < 0 ? undefined
        : { at: unanimous, text: this.steps[unanimous]!.text },
      repeated: repeated < 0 ? undefined : { at: repeated, text: this.steps[repeated]!.text },
    }
  }

  async selftest() {
    if (!await this.#ready) return { error: this.status }
    const out: Record<string, unknown> = {}

    this.resetSettings({ watermark: true })
    this.prompt = DEFAULT_PROMPT
    await this.generate()
    out['watermarked'] = this.runStats()

    // The Generate tab scores the stream it generated; a detector handed the same text has to
    // tokenize it first, and BPE can merge across the seam between prompt and story. A reader
    // can put both numbers side by side by pressing "Restore the story", so the gap is worth
    // measuring rather than assuming. A merge perturbs the four following context windows
    // before the seeds resynchronise, so a small gap is expected. Reported rather than
    // asserted: what the right tolerance is has never been measured, and a threshold guessed
    // here would be a red test nobody trusts rather than a fact anybody knows.
    const genIds = this.documentIds()
    const reIds = this.#tok!.encode(this.fullText())
    const zGen = this.scoreOf(genIds).z, zRe = this.scoreOf(reIds).z
    out['tokenization'] = {
      sameStream: genIds.length === reIds.length && genIds.every((v, i) => v === reIds[i]),
      generatedTokens: genIds.length, retokenizedTokens: reIds.length,
      zGenerated: +zGen.toFixed(2), zRetokenized: +zRe.toFixed(2),
      dz: +(zGen - zRe).toFixed(2),
    }

    // No determinism trap here, unlike the distortion-free bulb: the candidates are drawn for
    // real, so a fixed key leaves the loop plenty of randomness of its own. Confirmed rather
    // than assumed, because "the same story every time" is a failure that looks like a feature.
    const first = this.text()
    await this.generate()
    const again = this.text()
    this.seed = (this.seed + 1) >>> 0
    await this.generate()
    out['determinism'] = { sameSeedRepeats: first === again, steppedSeedDiffers: this.text() !== first }

    this.resetSettings({ watermark: false })
    this.prompt = DEFAULT_PROMPT
    await this.generate()
    const off = this.runStats()
    out['unwatermarked'] = off
    // The null, from the other side: with the key off, the coins of the words that came out
    // are fair coins and every layer's rate should sit on a half.
    out['nullFlat'] = off.perLayerRate.every(r => Math.abs(r - 0.5) < 0.12)

    // Human writing, against the same key and the same test. A scheme that finds a mark in
    // human writing has a bug, and it will look like a spectacular result rather than a bug.
    this.detectorText = HUMAN_WRITING
    const human = this.detectScore()
    out['human'] = human && {
      counted: human.score.counted,
      z: +human.score.z.toFixed(2),
      p: +human.score.p.toFixed(3),
      notSignificant: human.score.p > 0.05,
    }

    this.resetSettings({ watermark: true })
    await this.generate()
    // Step 5 of the plan: the bracket and the closed form are two implementations of one rule,
    // and their agreeing is the strongest evidence available cheaply that either is right.
    out['agreement'] = await this.checkAgreement(60000)

    // Non-distortion, measured, and measured against a control. The pairing the page runs must sit
    // where the measurement's own roughness puts it. The paper's distortionary setting, which the
    // page does not offer, must stand well clear of it: without that row a check that has only
    // ever read "nothing here" cannot be told from one that always would.
    const fair2 = await this.checkFairness(1200)
    const fair3 = await this.checkFairness(1200, 3)
    out['fairness'] = {
      two: fair2 && { tv: +fair2.tv.toFixed(4), noise: +fair2.tvNoise.toFixed(4), ratio: +fair2.ratio.toFixed(2) },
      three: fair3 && { tv: +fair3.tv.toFixed(4), noise: +fair3.tvNoise.toFixed(4), ratio: +fair3.ratio.toFixed(2) },
      twoIsNoiseOnly: !!fair2 && fair2.ratio < FAIR_RATIO,
      threeIsBiasedVsNoise: !!fair3 && fair3.ratio > FAIR_RATIO,
      threeIsBiased: !!fair2 && !!fair3 && fair3.tv > 2 * fair2.tv,
    }

    this.resetSettings()
    this.prompt = DEFAULT_PROMPT
    await this.generate()
    return out
  }
}

/** The k highest-scoring indices, by partial insertion rather than a sort: ordering all
 *  50257 entries would cost more than the forward pass that produced them. */
function topIndices(score: Float64Array, k: number): number[] {
  const kept: number[] = []
  let floor = -Infinity
  for (let v = 0; v < score.length; v++) {
    const x = score[v]!
    if (kept.length === k && x <= floor) continue
    let at = kept.length
    while (at > 0 && score[kept[at - 1]!]! < x) at--
    kept.splice(at, 0, v)
    if (kept.length > k) kept.pop()
    floor = score[kept[kept.length - 1]!]!
  }
  return kept
}

// ============================================================================
//  UI
// ============================================================================

const pct = (x: number) => `${(x * 100).toFixed(1)}%`

/** Compact percent for the inspector's number column, where 67% and 0.02% both have to fit in
 *  the same space. The small end matters: it is where "the model rated this at one in five
 *  thousand" lives, and rounding it to 0.0% throws away the point of showing it. */
const pctTight = (x: number) =>
  x >= 0.095 ? `${Math.round(x * 100)}%`
    : x >= 0.0005 ? `${(x * 100).toFixed(1)}%`
      : x >= 0.00005 ? `${(x * 100).toFixed(2)}%`
        : '~0%'

/** How many candidates a bracket holds, written the way it can be read at that size: a drawable
 *  bracket is a count the reader can check against the picture, and a thirty-layer one is an
 *  order of magnitude and nothing more. 3^30 is fifteen digits, which is a number nobody reads.
 *
 *  Shared by the layers slider and the inspector, which have to agree: the inspector used to
 *  announce "Thirty layers" over whatever count the slider was actually on. */
const slotCount = (layers: number) => {
  const slots = 2 ** layers
  if (slots <= BRACKET_MAX) return `${slots}`
  // Exact below a million, since 2^n costs nothing to state exactly and a reader can check it
  // against the layer count. Above one the word carries it better than ten digits do.
  return magnitude(slots) ?? slots.toLocaleString()
}

/** A magnitude word above a million, where nine digits are harder to hold than the word, and
 *  nothing below one: "250 thousand" reads as an approximation of something, where 250,000 reads
 *  as the number it is. Undefined below a million so each caller keeps its own precision. */
function magnitude(n: number): string | undefined {
  for (const [scale, word] of [[1e9, 'billion'], [1e6, 'million']] as const) {
    if (n >= scale) {
      const v = n / scale
      return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${word}`
    }
  }
  return undefined
}

/** Odds at two significant figures. Six seeds at one setting on the green list bulb scored
 *  anywhere from z = 4.97 to z = 7.24, which is one keyless writer in 600,000 at one end and one
 *  in 74 billion at the other. Every digit past the second reports that spread as though it were
 *  a measurement, so under a million the zeros carry the rounding on their face. */
function oddsText(odds: number): string {
  // Rounded before the magnitude test, not after, or 999,999 rounds up into a plain "1,000,000"
  // sitting next to a "1.0 million" for the value one above it.
  const n = Number(odds.toPrecision(2))
  return magnitude(n) ?? n.toLocaleString()
}

/** The meter's word and the tone it is set in, from one place. Two parallel ternaries over the
 *  same thresholds is exactly how a verdict and its colour drift apart. */
function verdictOf(s: Score): { word: string; tone: string } {
  // Silent below the count the headline already calls too little text: a verdict beside "too
  // little text" would be claiming what the headline just declined to.
  if (s.counted < MIN_COUNTED) return { word: '', tone: 'none' }
  if (s.z >= Z_MARKED) return { word: 'watermarked', tone: 'strong' }
  if (s.z >= Z_MAYBE) return { word: 'probably watermarked', tone: 'weak' }
  return { word: 'no signal', tone: 'none' }
}

/** The meter's headline, in the one unit everybody already has intuitions for.
 *
 *  It used to read `z = 4.13`, which is exact, standard, and means nothing at all to a reader
 *  without a statistics background. Those are the readers this page is for. Odds against chance
 *  say the same thing and need no glossary, and the z is still there in the small print for
 *  anyone who wants it. */
function oddsHeadline(s: Score): string {
  if (s.counted < MIN_COUNTED) return 'too little text'
  const odds = s.p > 0 ? 1 / s.p : Infinity
  // One over p runs to hundreds of digits by z = 30, and a number that long is not an odds.
  if (odds > 1e12) return 'under 1 in a trillion'
  return `1 in ${oddsText(Math.max(2, odds))}`
}

/** The gloss under the meter. It no longer repeats the odds, which are now the headline right
 *  above it, and does the one job the headline cannot: reconciling two numbers that look like
 *  they contradict each other. */
function chanceText(s: Score): string {
  if (s.counted < MIN_COUNTED) return 'Too little text here to say anything either way.'
  if (s.z < Z_MAYBE) {
    return 'A writer without the key scores this well often enough to mean nothing. That is not ' +
      'proof the text is unmarked: a short or heavily rewritten passage can carry a mark that no ' +
      'longer counts high enough to see.'
  }
  // A heads rate of 59% against a chance rate of 50% is a modest-looking gap, and the odds
  // above it are astronomical. Nothing bridges them but the count, and a reader who cannot
  // make that leap reads one of the two as a mistake.
  return 'A single coin proves nothing, since every one of them comes up heads half the time ' +
    `by luck. The odds above are what it takes for ${s.coins.toLocaleString()} of them to run ` +
    'this far above that.'
}

/** Token text made visible: a leading space is information here, and a newline has to stop
 *  being one. */
const visible = (t: string) => t.replace(/ /g, '·').replace(/\n/g, '⏎')

/** Token text for running prose, where the word rather than the token is the subject. */
const inWords = (t: string) => t.trim() ? t.trim().replace(/\n/g, '⏎') : visible(t)

/** Sources open in a new tab. The model is sitting in this page's memory, and navigating away
 *  costs a 6MB reload to come back to. */
const source = (href: string, label: VElement | string) =>
  a({ href, target: '_blank', rel: 'noopener noreferrer' }, label)

const PAPER_SYNTHID = 'https://www.nature.com/articles/s41586-024-08025-4'
const PAPER_ATTACK = 'https://arxiv.org/abs/2603.03410'
/** Anthropic's own account, 14 August 2026. It is what lets this page say Claude's watermark
 *  shares the method: "Claude's text watermark is a version of the SynthID-Text approach
 *  published by Google DeepMind in a Nature paper in 2024." It publishes no parameters, so the
 *  copy stops at the method and does not claim the configuration. Text in
 *  specs/watermarking-anthropic-source.txt. */
const PAPER_ANTHROPIC = 'https://www.anthropic.com/news/claude-text-watermark'

/** Which edge the inspector hangs from, so a word near either margin doesn't push it out of
 *  the story block. Measured once per hover rather than tracked, since the token doesn't move. */
function alignFor(el: HTMLElement | null): 'start' | 'center' | 'end' {
  const box = el?.closest('.story') as HTMLElement | null
  if (!el || !box) return 'center'
  const tok = el.getBoundingClientRect(), story = box.getBoundingClientRect()
  const at = (tok.left + tok.width / 2 - story.left) / (story.width || 1)
  return at < 0.24 ? 'start' : at > 0.76 ? 'end' : 'center'
}

type Tab = 'write' | 'detect' | 'about'

// ---- the explainer's toy match ---------------------------------------------

/** Three tickets, two of them cat. The model wanting cat two thirds of the time is expressed
 *  ONLY by how many tickets cat holds, which is the whole point: by the time two tickets are
 *  drawn, that two thirds has already been spent and there is nothing left for a coin to bend. */
const TOY_BAG = [0, 0, 1] as const
const TOY_NAME = ['cat', 'dog'] as const

interface ToyDraw {
  /** Which tickets came out. Two draws can land on the same ticket, since the first goes back
   *  in before the second, and seeing that happen is half of what the card is for. */
  picked: [number, number]
  /** The key: a coin for cat and a coin for dog, redrawn every time. */
  key: [number, number]
  winner: 0 | 1
  how: 'walkover' | 'coin' | 'toss'
}

/** One run of the scheme with the model removed. Same three cases as `sampleStep`, small
 *  enough to be exhausted by hand. */
function playToy(): ToyDraw {
  const i = Math.floor(Math.random() * TOY_BAG.length)
  const j = Math.floor(Math.random() * TOY_BAG.length)
  const key: [number, number] = [Math.random() < 0.5 ? 1 : 0, Math.random() < 0.5 ? 1 : 0]
  const a = TOY_BAG[i]!, b = TOY_BAG[j]!
  const ca = key[a]!, cb = key[b]!
  if (a === b) return { picked: [i, j], key, winner: a, how: 'walkover' }
  if (ca !== cb) return { picked: [i, j], key, winner: ca > cb ? a : b, how: 'coin' }
  return { picked: [i, j], key, winner: Math.random() < 0.5 ? a : b, how: 'toss' }
}

class Root extends Component {
  engine = new Engine()
  tab: Tab = 'write'
  advanced = false
  popAlign: 'start' | 'center' | 'end' = 'center'

  toy?: ToyDraw
  /** 0 idle, 1 first ticket out, 2 second ticket out, 3 match resolved. Staged rather than
   *  shown at once because separating the draw from the match in time is what separates them
   *  in the reader's head, and conflating the two is the misreading the card exists to fix. */
  toyStage = 0
  #toyTimers: number[] = []

  toyDraw(): ToyDraw {
    for (const t of this.#toyTimers) clearTimeout(t)
    this.#toyTimers = []
    const d = playToy()
    this.toy = d
    this.toyStage = 1
    this.update()
    this.#toyTimers.push(window.setTimeout(() => { this.toyStage = 2; this.update() }, 380))
    this.#toyTimers.push(window.setTimeout(() => { this.toyStage = 3; this.update() }, 820))
    return d
  }

  view() {
    return div({ class: 'app' },
      div({ class: 'header' },
        h1('How Watermarking Text Works'),
        p({ class: 'tagline' },
          'Learn how text watermarking works on a toy language model, locally in your browser. ' +
          'Before each word a secret key runs a tournament between words the model already ' +
          'drew, so the winner is always one it was willing to say.'),
      ),
      div({ class: 'tabs' },
        this.tabButton('write', 'Generate'),
        this.tabButton('detect', 'Detect'),
        this.tabButton('about', 'How it works'),
      ),
      this.tab === 'about' ? this.about()
        : !this.engine.ready ? this.loading()
        : this.tab === 'write' ? this.write()
        : this.detector(),
    )
  }

  tabButton(id: Tab, label: string) {
    return button({
      class: this.tab === id ? 'tab active' : 'tab',
      // Opening the explainer asks for a reading. It is a no-op until the model is there, and
      // `start` picks it up again once it is.
      onClick: () => {
        this.tab = id
        if (id === 'about') this.engine.scheduleFairness(0)
        this.update()
      },
    }, label)
  }

  write() {
    return div({ class: 'stage' }, this.controls(), this.output())
  }

  /** Hand the finished text across to the reading half. The text is the only thing that
   *  crosses: no seeds, no probabilities, no record of which words the model considered.
   *  Switching tabs is the gesture, and the gesture is the claim.
   *
   *  The prompt crosses with it. Sending the continuation alone opens the detector
   *  mid-sentence and claims a document boundary no reader of the finished text could see. */
  handoff() {
    this.engine.detectorText = this.engine.fullText()
    this.tab = 'detect'
    this.update()
  }

  loading() {
    const e = this.engine
    if (e.failed) return div({ class: 'loading' }, p({ class: 'loading-lead' }, e.status))
    const p100 = Math.round(e.progress * 100)
    return div({ class: 'loading' },
      p({ class: 'loading-lead' },
        'Loading a 3.7 million parameter language model into your browser to run locally.'),
      div({ class: 'progress' },
        div({ class: 'progress-track' }, div({ class: 'progress-fill', style: { width: `${p100}%` } })),
        div({ class: 'progress-row' },
          span({ class: 'progress-step' }, e.status || 'Warming up…'),
          span({ class: 'progress-pct' }, `${p100}%`)),
      ),
    )
  }

  controls() {
    const e = this.engine
    const s = e.settings
    return div({ class: 'panel controls' },
      div({ class: 'panel-head' },
        h3('Start the story:'),
        div({ class: 'head-actions' },
          button({
            class: 'chip', disabled: e.generating,
            // Stepping the seed rather than randomising it keeps a session reproducible: from
            // a fresh load the stories come in the same order.
            onClick: () => { e.seed = (e.seed + 1) >>> 0; void e.generate() },
          }, e.generating ? 'Writing…' : e.steps.length ? 'Regenerate' : 'Write a story'),
          span({ class: 'hint' }, `${e.steps.length} tokens`),
        ),
      ),
      inputTextArea({
        target: e,
        prop: () => e.prompt,
        attrs: { class: 'prompt-input', rows: 2, spellCheck: false },
      }),
      div({ class: 'mainbar' },
        button({
          class: s.watermark ? 'toggle on' : 'toggle',
          // Its label is the word beside it, which a pointer can see and a screen reader cannot.
          // Naming it also puts it in the accessibility outline, which is what `tb:click` reads.
          ariaLabel: 'Watermark',
          onClick: () => { e.watermark = !e.watermark; e.update(); void e.generate() },
        }, span({ class: 'toggle-knob' })),
        span({ class: 'mainbar-label' }, 'Watermark'),
        span({ class: 'mainbar-value' }, s.watermark ? 'on' : 'off'),
        this.disclose(),
      ),
      p({ class: 'mainbar-help' },
        'Whether the key runs the tournament. Off is the same model drawing one word instead ' +
        'of several and keeping it. The story changes, because a different draw is a different ' +
        'story, and that is the only difference there is.'),
      this.advanced ? this.knobs() : null,
    )
  }

  disclose() {
    return button({
      class: this.advanced ? 'disclose open' : 'disclose',
      onClick: () => { this.advanced = !this.advanced; this.update() },
    }, 'Advanced')
  }

  /** The instruments. All four rewrite the story, so all four live with the tab that writes
   *  one; the two that belong to the scheme rather than to the decoder appear again beside the
   *  check that measures what they cost. */
  knobs() {
    const e = this.engine
    const s = e.settings
    return div({ class: 'knobs' },
      this.knob('Temperature', s.temperature.toFixed(2),
        inputRange({
          target: e, prop: () => e.temperature,
          attrs: {
            min: 0.3, max: 1.4, step: 0.05, ariaLabel: 'Temperature',
            onChange: () => void e.generate(),
          },
        }),
        'How adventurously the model samples. The mark rides on the freedom the model already ' +
        'has: a position with one possible word carries none, whatever the key says.'),
      this.layersKnob(),
      this.knob('Length', `${s.maxTokens} tokens`,
        inputRange({
          target: e, prop: () => e.maxTokens,
          attrs: {
            min: 30, max: 200, step: 10, ariaLabel: 'Length',
            onChange: () => void e.generate(),
          },
        }),
        'How much the model writes. Evidence accumulates a coin at a time, so length is the ' +
        'single biggest thing separating a mark that shows from one that does not.'),
    )
  }

  layersKnob(compact = false) {
    const e = this.engine
    const slots = 2 ** e.layers
    const count = slotCount(e.layers)
    // Only two settings on this slider are real: the largest bracket that can be drawn for a
    // reader, and the one the paper publishes. Everything between them is interpolation, and
    // naming the two stops is what stops a continuous slider implying m is a per-message dial.
    const anchor = slots <= BRACKET_MAX ? ', drawable'
      : e.layers === MAX_LAYERS ? ', as published' : ''
    return this.knob('Tournament layers',
      `${e.layers}, ${count} candidates${anchor}`,
      inputRange({
        target: e, prop: () => e.layers,
        attrs: {
          min: 1, max: MAX_LAYERS, step: 1, ariaLabel: 'Tournament layers',
          onChange: () => { void e.generate(); if (compact) e.scheduleFairness() },
        },
      }),
      compact ? undefined
      : 'Rounds of the tournament, and so coins per word. Every layer is another coin the test ' +
      `counts, so the mark sharpens as this rises. Two settings on here are real: small enough ` +
      `to draw, which stops at ${BRACKET_MAX} candidates, and thirty, which is the number the ` +
      'paper publishes. A deployment picks one and keeps it, because the detector has to know ' +
      'how many coins to count.')
  }

  /** Help is optional: beside the check that measures it, the layers slider appears a second time
   *  and its paragraph would push the number it moves off the bottom of the screen. */
  knob(label: string, value: string, control: VElement, help?: string) {
    return div({ class: 'knob' },
      div({ class: 'knob-head' },
        span({ class: 'knob-label' }, label),
        span({ class: 'knob-value' }, value)),
      control,
      help ? p({ class: 'knob-help' }, help) : null,
    )
  }

  // ---- the story ------------------------------------------------------------

  /** The legend, naming what each mark MEANS rather than what it looks like.
   *
   *  One channel, because there is one question: how many of this word's coins came up heads.
   *  That is not a proxy for the evidence, it IS the evidence, and the lit share of the
   *  paragraph is the numerator of the number in the meter. Both sibling bulbs had to hedge
   *  that relationship; this one does not.
   *
   *  What it also has to say is an absence. There is no mark for "the key changed this word",
   *  because the key changes no words, and a reader arriving from a green list explainer is
   *  expecting one. */
  storyKey(s: Score) {
    const e = this.engine
    const layers = e.layers
    const lit = litCount(s, layers)
    const uncounted = s.marks.length - s.counted
    const chance = Math.round(litChance(layers) * 100)
    const handoff = [
      ', or ',
      button({
        class: 'inline-link', disabled: !e.steps.length,
        onClick: () => this.handoff(),
      }, 'take it to the Detect tab'),
      ' and see what counting alone makes of it.',
    ]
    const marks =
      `A word is highlighted when more than half of its ${layers} coins came up heads: ${lit} ` +
      `of the ${s.counted} words the test counts, against the ${chance}% luck alone would ` +
      `give. Your opening line is read and marked exactly like the rest, and being your own ` +
      `writing it lights at about that chance rate: the gap between it and the story is the ` +
      `watermark. Greyed out means not counted at all, either a repeat of a four-word context ` +
      `the test has already read or one of the first ${SEED_CONTEXT} words, which have nothing ` +
      `in front of them to seed a coin with, ${uncounted} of them. Point at any word of the ` +
      'story for the tournament it came out of'
    if (!e.settings.watermark) {
      return ['Nothing is steering this run: the words were drawn with ordinary randomness and ' +
        'the key is only being read against them afterwards, so the coins are fair ones. ' +
        marks, handoff]
    }
    return [marks, handoff]
  }

  /** The shell both tabs put a result in: the heading with the meter on its right, the marked
   *  text, and a foot facing what the run came to. The detector's panel being the story's panel
   *  is deliberate, so the shape lives here rather than in two copies kept in step by hand. */
  resultPanel(title: string, s: Score, body: VElement | null, foot: (VElement | null)[]) {
    return div({ class: 'panel output' },
      div({ class: 'output-head' }, h3(title), this.meter(s)),
      body,
      div({ class: 'panel-foot' }, foot),
    )
  }

  /** The document, one span per token, carrying one mark.
   *
   *  The prompt is painted too, unlike either sibling bulb, because the test reads it and a
   *  legend quoting counts over words the reader cannot see marked is a legend that does not
   *  add up. It also teaches the thing the page most wants taught, for free: the human opening
   *  lights at the chance rate and the story does not. */
  output() {
    const e = this.engine
    const s = e.liveScore()
    const from = e.promptIds.length
    const tok = e.tokenizer
    const layers = e.layers
    const litAt = (i: number) => isLit(s.marks[i] ?? -1, layers)
    return this.resultPanel(
      e.settings.watermark ? 'The story (watermarked)' : 'The story', s,
      div({ class: 'story' },
        tok ? tok.pieces(e.promptIds).map((t, i) => span({
          class: ['tok', 'echo', litAt(i) ? 'lit' : '', s.marks[i] === -1 ? 'unscored' : ''],
        }, t)) : null,
        e.steps.map((st, i) => span({
          class: [
            'tok',
            litAt(from + i) ? 'lit' : '',
            s.marks[from + i] === -1 ? 'unscored' : '',
            e.selected === i ? 'selected' : '',
          ],
          onMouseEnter: (ev: MouseEvent) => {
            this.popAlign = alignFor(ev.currentTarget as HTMLElement)
            e.hovered = i
            this.update()
          },
          onMouseLeave: () => { if (e.hovered === i) { e.hovered = -1; this.update() } },
          onClick: () => { e.selected = e.selected === i ? -1 : i; e.update() },
        }, st.text, this.inspector(i))),
        e.generating ? span({ class: 'caret' }) : null,
        // Not a token, and deliberately outside the marked spans: the test never saw it, so it
        // carries no mark and cannot be pointed at.
        !e.generating && e.truncated && e.steps.length
          ? span({ class: 'text-more' }, ' …') : null,
      ),
      [
        p({ class: 'story-key' }, this.storyKey(s)),
        e.settings.watermark
          ? p({ class: 'story-key' },
              'No word here was promoted. Every candidate in every bracket was drawn from the ' +
              'model\'s own distribution before the key saw it, so the key can never reach for ' +
              'a word the model would not have said. It only decides which of them wins.')
          : null,
      ],
    )
  }

  /** The counting test, as it stands. z is the distance from what a keyless writer would
   *  average, in standard deviations, and it is the paper's own statistic. */
  meter(s: Score) {
    const cap = 8
    const frac = Math.max(0, Math.min(1, s.z / cap))
    const verdict = verdictOf(s)
    // Named so its rendered width can be measured from the terminal with `tb:rect`, which is
    // how the wrapping above was diagnosed rather than guessed at.
    return div({ class: 'meter', role: 'group', ariaLabel: 'evidence' },
      div({ class: 'meter-row' },
        span({ class: 'meter-odds' }, oddsHeadline(s)),
        span({ class: ['meter-verdict', verdict.tone] }, verdict.word),
      ),
      div({ class: 'meter-sub' }, 'the chance a writer without the key scores this high'),
      div({ class: 'meter-track' },
        div({ class: 'meter-fill', style: { width: `${frac * 100}%` } }),
        div({ class: 'meter-tick', style: { left: `${(Z_MARKED / cap) * 100}%` } }),
      ),
      div({ class: 'meter-foot' },
        span(s.coins
          ? `${s.heads.toLocaleString()} of ${s.coins.toLocaleString()} coins came up heads ` +
            `(${pct(s.heads / s.coins)} against 50% by chance)`
          : ''),
        span({ class: 'meter-z' }, `z = ${s.z.toFixed(2)}`),
      ),
    )
  }

  // ---- the inspector --------------------------------------------------------

  /** One decoding step, opened at the word it produced.
   *
   *  Where the bracket is small enough it is drawn slot by slot, which is the reason this bulb
   *  exists: a knockout tournament is understood before a reader has finished looking at it,
   *  and three things become visible at once that a paragraph struggles with. A position with
   *  eight different words is the mechanism. A likely word holding several slots at once is WHY
   *  the scheme is fair rather than merely a claim that it is. And a bracket where that word
   *  holds EVERY slot is the same thing taken to its limit: nothing to decide, no say for the
   *  key.
   *
   *  That last case is not evidence of a position where only one word would do, and must not be
   *  described as one. Slots are filled with replacement, so P(all identical) is about
   *  p(top)^slots: 44% at p = 0.95, 18% at 0.9, 7% at 0.85, and at one layer it is simply
   *  sum p(x)^2, the common case at any peaked position. The popover prints the drawn word's
   *  own probability two lines above this text, so calling it forced contradicts a number the
   *  reader can already see. See specs/watermarking-corrections.md P1. */
  inspector(i: number) {
    const e = this.engine
    const active = e.hovered >= 0 ? e.hovered : e.selected
    if (active !== i) return null
    const step = e.steps[i]
    if (!step) return null
    return div({ class: ['pop', this.popAlign] },
      div({ class: 'pop-card' },
        div({ class: 'pop-head' },
          span('drew ', strong(visible(step.text))),
          span(`${pctTight(step.p)} chance` +
            (step.heads === undefined ? '' : ` · ${step.heads} of ${e.layers} heads`)),
        ),
        step.masked ? this.maskedNote(step)
          : !step.coins ? this.plainNote()
            : step.bracket ? this.bracketView(step.bracket)
              : this.computedView(step),
        step.coins ? this.coinStrip(step.coins) : null,
        this.inspectorNote(step),
      ),
    )
  }

  maskedNote(step: Step) {
    return div({ class: 'pop-note lead' },
      step.masked === 'repeat'
        ? ['These four words had already come up once, so this position was left alone. ' +
            'Marking it again would reuse the same coins, and the test counts coins it assumes ' +
            'are independent. Both halves of the scheme skip it.']
        : ['There were fewer than four words in front of this one, so there was nothing to ' +
            'seed the coins with. The opening of any text is unmarkable and unreadable alike.'],
    )
  }

  plainNote() {
    return div({ class: 'pop-note lead' },
      'The key is off, so the model drew one word and kept it. Turn the watermark on and this ' +
      'same position becomes a bracket of candidates from this same distribution.')
  }

  /** The tournament as it was actually played. Columns are layers, rows are slots, and a cell
   *  is one candidate with the coin it carried into that match. */
  bracketView(b: Bracket) {
    const rows = b.rounds[0]!.slots.length
    const cells: VElement[] = []
    b.rounds.forEach((round, r) => {
      const span0 = 2 ** r
      round.slots.forEach((slot, i) => {
        const won = round.wonAt[i >> 1] === (i & 1)
        cells.push(div({
          class: ['slot', won ? 'won' : 'out'],
          style: { gridColumn: `${r + 1}`, gridRow: `${i * span0 + 1} / span ${span0}` },
        },
          span({ class: slot.heads ? 'coin heads' : 'coin' }),
          span({ class: 'slot-tok' }, inWords(slot.text)),
        ))
      })
    })
    const last = b.rounds[b.rounds.length - 1]!
    const winner = last.slots[last.wonAt[0]!]!
    cells.push(div({
      class: 'slot final',
      style: { gridColumn: `${b.rounds.length + 1}`, gridRow: `1 / span ${rows}` },
    }, span({ class: 'slot-tok' }, inWords(winner.text))))
    return [
      div({ class: 'bracket-wrap' },
        div({
          class: 'bracket',
          style: {
            gridTemplateColumns: `repeat(${b.rounds.length + 1}, minmax(0, 1fr))`,
            // A sixteen-slot bracket is twice the height of an eight-slot one, and the card hangs
            // above the word it belongs to, so the taller one earns a tighter row.
            gridTemplateRows: `repeat(${rows}, ${rows > 8 ? '0.95rem' : '1.15rem'})`,
          },
        }, cells),
      ),
      // Outside the scrolling wrap, so a wide bracket does not carry its own legend off the edge.
      p({ class: 'bracket-key' },
        'A filled circle is a coin that came up heads, a hollow one tails. Faded words lost ' +
        'their match.'),
    ]
  }

  /** What replaces the bracket once it is a billion candidates wide: the words the model
   *  rated highest, with the coins each was carrying. The winner tends to be one the model
   *  already liked whose coins ran heads, which is the same story the bracket tells. */
  computedView(step: Step) {
    const e = this.engine
    const rows = step.top.slice(0, SHOW_K)
    if (!rows.some(c => c.id === step.id)) {
      const chosen = step.top.find(c => c.id === step.id)
      if (chosen) rows.push(chosen)
    }
    return div({ class: 'bars' },
      div({ class: 'bar-head' },
        span(''),
        div({ class: 'axis' }, span('0'), span('50%'), span('100%')),
        span('heads')),
      rows.map(c => div({ class: ['bar-row', c.id === step.id ? 'chosen' : ''] },
        span({ class: 'bar-tok' }, visible(c.text)),
        div({ class: 'bar-track' }, div({ class: 'bar-p', style: { width: `${c.p * 100}%` } })),
        span({ class: 'bar-heads' }, `${c.heads}/${e.layers}`),
      )),
    )
  }

  /** The emitted word's coins, which are exactly what the test counts at this position. */
  coinStrip(coins: Uint8Array) {
    return div({ class: 'coins' }, Array.from(coins, c =>
      span({ class: c ? 'coin heads' : 'coin' })))
  }

  /** The card's one claim, and under it the one thing a reader misreads as a malfunction. */
  inspectorNote(step: Step) {
    const lead = this.inspectorLead(step)
    return [
      lead ? div({ class: 'pop-note lead' }, lead) : null,
      step.likeliest
        ? div({ class: 'pop-note' },
            'The draw is random, so the likeliest word usually wins but not always. Here ',
            strong(inWords(step.likeliest.text)), ' was likeliest and did not come up.')
        : null,
    ]
  }

  /** What the tournament above the note did, or nothing at a position that had no tournament,
   *  where `maskedNote` and `plainNote` have already said why. */
  inspectorLead(step: Step): string | undefined {
    if (step.masked || !step.coins) return undefined
    const e = this.engine
    const b = step.bracket
    if (b && b.distinct === 1) {
      return 'Every slot here drew the same word. Slots are filled with replacement, so a word ' +
        'the model rates highly often takes all of them at once, and then there is nothing to ' +
        'decide and the key has no say. Its coins are still counted, and being fair coins they ' +
        'carry no evidence either way.'
    }
    if (b) {
      return 'Each match is won by the candidate whose coin came up heads, and two matching ' +
        'coins are settled at random. Every one of these words was the model\'s own draw.'
    }
    return `${e.layers} layers is ${slotCount(e.layers)} candidates, so the ` +
      'tournament is computed rather than played out. The distribution it produces has an ' +
      'exact formula costing one pass over the vocabulary per layer, which is what lets the ' +
      'published thirty layers, a billion candidates wide, run at all.'
  }

  // ---- the detector ---------------------------------------------------------

  detector() {
    const e = this.engine
    const found = e.detectScore()
    return div({ class: 'stage' },
      div({ class: 'panel controls' },
        div({ class: 'panel-head' },
          h3('Test for watermark:'),
          div({ class: 'head-actions' },
            button({
              class: 'chip', disabled: !e.steps.length,
              onClick: () => { e.detectorText = e.fullText(); e.update() },
            }, 'Restore the story'),
            button({
              class: 'chip',
              onClick: () => { e.detectorText = HUMAN_WRITING; e.update() },
            }, 'Try human writing'),
          ),
        ),
        inputTextArea({
          target: e,
          prop: () => e.detectorText,
          attrs: {
            class: 'detector-input', rows: 6, spellCheck: false,
            placeholder: 'Paste any text here to test it against the key…',
          },
        }),
      ),
      found ? this.reading(found)
        : div({ class: 'panel output' }, p({ class: 'detector-stats' }, 'Nothing to read yet.')),
    )
  }

  reading(found: { score: Score; ids: number[] }) {
    return this.resultPanel('The text (as the key reads it)', found.score,
      this.readBack(found),
      [this.readKey(found.score), p({ class: 'detector-odds' }, chanceText(found.score))],
    )
  }

  readBack(found: { score: Score; ids: number[] }) {
    const tok = this.engine.tokenizer
    if (!tok) return null
    const layers = this.engine.layers
    const s = found.score
    return div({ class: 'read-back' },
      tok.pieces(found.ids.slice(0, s.marks.length)).map((t, i) => {
        const m = s.marks[i]!
        return span({ class: ['rtok', m < 0 ? 'unscored' : '', isLit(m, layers) ? 'lit' : ''] }, t)
      }),
      s.dropped
        ? span({ class: 'text-more' },
            ` … and ${s.dropped} more words, past the ${TEST_MAX.toLocaleString()} this test reads`)
        : null,
    )
  }

  readKey(s: Score) {
    const layers = this.engine.layers
    const lit = litCount(s, layers)
    return p({ class: 'read-key' },
      'Recomputed from the text alone, with no model and no record of how it was written. ' +
      `A word is highlighted when more than half of its ${layers} coins came up heads, ` +
      `${lit} of the ${s.counted} words counted. Greyed out are the ${s.contextless} opening ` +
      `words with nothing in front of them, and ${s.repeats} whose four-word context had ` +
      'already been read. Everything else counted and came up short of a majority.')
  }

  // ---- the check ------------------------------------------------------------

  /** The claim the whole page rests on, turned into an instrument.
   *
   *  A reader who moves a slider and watches a measured number respond has understood the claim
   *  better than any paragraph of mine will manage, which only works if the knob and the number
   *  it moves are on screen together. An earlier version ran head to verdict past a full page, so
   *  pressing the button changed a figure the reader could not see. Everything here is ordered by
   *  that constraint: the three figures sit directly under the slider, the prose that used to
   *  explain them is a sentence, and the table is below the fold as corroboration.
   *
   *  The layers slider here is the same setting as the one on the Generate tab, not a copy, so
   *  moving it rewrites the story too. That has to be true: the detector counts a document at the
   *  current layer count, and a story marked at four read back at thirty would score against coins
   *  it never carried. */
  fairnessCard() {
    const e = this.engine
    const f = e.fairness
    return div({ class: 'check' },
      div({ class: 'check-head' },
        h3('Measure the distortion'),
        span({ class: 'check-status' },
          e.fairBusy ? `measuring, ${e.fairDone.toLocaleString()} keys…` : ''),
      ),
      p(!e.ready
        ? 'This one needs the model, which loads on the Generate tab.'
        : 'Work out exactly what the tournament emits from at one position, under thousands of ' +
          'keys, and average. Drawing every candidate from the model is not on its own enough ' +
          'to leave its rates alone: the choosing between them has to be even handed too. It ' +
          'is, and the distance below is what that looks like measured rather than asserted.'),
      div({ class: 'check-knobs' }, this.layersKnob(true)),
      f
        ? [
          this.fairnessFigures(f, e.fairBusy),
          p({ class: 'check-foot' }, this.fairnessVerdict(f)),
          table({ class: 'check-table' },
            thead(tr(
              th('word'),
              th({ class: 'num' }, 'model'),
              th({ class: 'num' }, `over ${f.keys.toLocaleString()} keys`))),
            tbody(f.rows.map(row => tr(
              td({ class: 'mono' }, visible(row.text)),
              td({ class: 'num' }, pct(row.p)),
              td({ class: 'num' }, pct(row.keyed)),
            ))),
          ),
        ]
        : null,
    )
  }

  /** The whole reading, in one line and in the order it is thought about: what was measured,
   *  what a fair scheme would have left behind anyway, and the one over the other. Sat under a
   *  paragraph before, where pressing the button appeared to do nothing. */
  fairnessFigures(f: Fairness, busy: boolean) {
    return div({ class: ['check-figures', busy ? 'busy' : ''], role: 'group', ariaLabel: 'reading' },
      this.figure(f.tv.toFixed(4), 'measured distance'),
      this.figure(f.tvNoise.toFixed(4), 'roughness at these keys'),
      this.figure(`${f.ratio.toFixed(1)}x`, 'one over the other'),
    )
  }

  figure(value: string, label: string) {
    return div({ class: 'figure' },
      span({ class: 'figure-value' }, value),
      span({ class: 'figure-label' }, label),
    )
  }

  /** What the distance means, which is not its size but how it compares against the distance a
   *  finite number of keys would have left behind anyway. A reader handed one number has no way
   *  to judge it; a reader handed it beside what fairness itself would produce has. The figures
   *  above carry the numbers, so this says only what they mean. */
  fairnessVerdict(f: Fairness) {
    if (f.ratio < FAIR_RATIO) {
      return 'At or under what a perfectly fair scheme leaves behind at this many keys, so that ' +
        'distance is the measurement\'s own roughness rather than anything the watermark did. ' +
        'More keys pushes it towards zero.'
    }
    // Unreachable at a fair tournament, which is the only one on offer here, so it reads as the
    // guard it is rather than as a second verdict a reader is meant to go looking for.
    return 'Further out than the roughness of this many keys explains, which should not happen ' +
      'here: a tournament between pairs is fair whatever the layers are set to. Something is ' +
      'wrong with this reading rather than with the watermark.'
  }

  // ---- how it works ---------------------------------------------------------

  /** The scheme with the model taken out of it: three tickets, one match, one coin each.
   *
   *  It sits inside the explainer because this is the paragraph a reader stops believing, and
   *  the objection always has one shape: if cat is worth two thirds and dog one third, a coin
   *  that settles between them must flatten the two towards each other. The answer is that the
   *  coin never meets those numbers. They are spent drawing the tickets, and by the time two
   *  tickets face each other it is one slip against one slip. Watching the same word take both
   *  slots, which happens five draws in nine here, is what makes that land.
   *
   *  Deliberately not the Generate tab's bracket. That one is real, sixteen slots wide over a
   *  fifty thousand word vocabulary, and it shows the scheme working. This one is small enough
   *  to exhaust, and it shows why the scheme works. */
  toyCard() {
    const t = this.toy
    const shown = !!t && this.toyStage >= 3
    // Both draws landing on one ticket is the case worth seeing, since it is what the whole
    // card is for, so that ticket is drawn as two stacked cards rather than as one lit twice.
    const twice = !!t && this.toyStage >= 2 && t.picked[0] === t.picked[1]
    return div({ class: 'check' },
      div({ class: 'check-head' },
        h3('Watch one match'),
        button({ class: 'chip', onClick: () => { this.toyDraw() } }, 'Draw two tickets'),
      ),
      p('Two words and no model. Cat is worth two thirds, so cat holds two of the three ' +
        'tickets, and that is the only place its two thirds lives. Two tickets come out, the ' +
        'first going back in before the second, so the same word often takes both slots.'),
      // Named so the two rows' rects can be read back with `tb:rect`, which is how their
      // centring was confirmed rather than eyeballed: same centre, or it is not centred.
      div({ class: 'toy-bag', role: 'group', ariaLabel: 'the bag' }, TOY_BAG.map((w, i) => span({
        class: ['toy-ticket',
          t && this.toyStage >= 1 && t.picked[0] === i ? 'out' : '',
          t && this.toyStage >= 2 && t.picked[1] === i ? 'out' : '',
          twice && t!.picked[0] === i ? 'twice' : ''],
      }, TOY_NAME[w]))),
      div({ class: 'toy-match', role: 'group', ariaLabel: 'the match' },
        this.toySlot(0),
        span({ class: 'toy-vs' }, 'vs'),
        this.toySlot(1),
      ),
      // The winner sits on its own line rather than trailing the pair. Hung off the right, it
      // pushed the two slots left of the axis the bag above them centres on, so the match read
      // as lopsided. Bare, with no arrow in front of it: an arrow is one glyph wide and shifts
      // the word off the centre line every other row is built on.
      div({ class: ['toy-outcome', shown ? 'shown' : ''], role: 'group', ariaLabel: 'the winner' },
        span({ class: 'toy-winner' },
          span({ class: 'toy-card-word' }, shown ? TOY_NAME[t!.winner] : ''),
          span({ class: 'toy-card-coin' },
            shown ? (t!.key[t!.winner] ? 'heads' : 'tails') : ''),
        ),
      ),
      p({ class: ['toy-note', shown ? 'shown' : ''] }, shown ? this.toyVerdict(t!) : ' '),
      this.toyTable(),
    )
  }

  /** A slot is a ticket's place in the match and reads as one. Empty, it holds a single centred
   *  placeholder like the tickets in the bag above it: a lone glyph sitting high because a blank
   *  coin line is reserved under it reads as badly set rather than as waiting. Once a word lands
   *  the coin line is reserved by CSS instead of by a space, so revealing the coin moves nothing. */
  toySlot(k: 0 | 1) {
    const t = this.toy
    const filled = !!t && this.toyStage >= k + 1
    if (!filled) return span({ class: 'toy-slot' }, span({ class: 'toy-card-word' }, '?'))
    const w = TOY_BAG[t!.picked[k]]!
    return span({ class: 'toy-slot filled' },
      span({ class: 'toy-card-word' }, TOY_NAME[w]),
      span({ class: 'toy-card-coin' }, this.toyStage >= 3 ? (t!.key[w]! ? 'heads' : 'tails') : ''),
    )
  }

  toyVerdict(t: ToyDraw): (VElement | string)[] {
    if (t.how === 'walkover') {
      return ['Both tickets say ', strong(TOY_NAME[TOY_BAG[t.picked[0]]!]),
        ', so there was nothing to decide and the coins were never looked at. That is five ' +
        'draws in nine here, with two different words in the bag: drawing with replacement is ' +
        'what makes it so common.']
    }
    if (t.how === 'coin') {
      return ['One of each. The coins disagree, so the key settles it and ',
        strong(TOY_NAME[t.winner]), ' wins on heads. One ticket against one ticket, which is ' +
        'the only situation the key ever gets to touch.']
    }
    return ['One of each, but both coins came up ', strong(t.key[0] ? 'heads' : 'tails'),
      '. Coins that agree single out neither word, so the match goes to a fair toss, exactly ' +
      'as it would with no key at all.']
  }

  /** Every pair of coins the key can deal here, four of them, equally likely, with what each does
   *  to cat worked out exactly rather than sampled. A pair giving both words the same coin decides
   *  nothing, so it leaves the bag's own rate; otherwise the heads word takes every mixed draw
   *  outright, and mixed draws are 2pq of them.
   *
   *  It says "the key's two coins" rather than "the key", and the caption says the key deals a
   *  fresh pair at every position. The key itself is one fixed secret, which is the whole reason
   *  a detector can recompute anything; what moves down the text is the pair it deals. */
  toyTable() {
    const pCat = 2 / 3, q = 1 / 3
    const keys: [number, number][] = [[1, 1], [1, 0], [0, 1], [0, 0]]
    const rates = keys.map(([kc, kd]) =>
      kc === kd ? pCat : kc === 1 ? pCat * (1 + q) : pCat * pCat)
    const t = this.toy
    return [
      table({ class: 'check-table' },
        thead(tr(th('the key\'s two coins'), th({ class: 'num' }, 'how often cat comes out'))),
        tbody(
          keys.map(([kc, kd], i) => {
            const used = !!t && this.toyStage >= 3 && t.key[0] === kc && t.key[1] === kd
            return tr({ class: used ? 'toy-used' : '' },
              td(`cat ${kc ? 'heads' : 'tails'}, dog ${kd ? 'heads' : 'tails'}`,
                used ? span({ class: 'toy-used-tag' }, ' this run') : ''),
              td({ class: 'num' }, pct(rates[i]!)))
          }),
          tr({ class: 'toy-avg' },
            td('average over the four'),
            td({ class: 'num' }, pct(rates.reduce((a, b) => a + b, 0) / 4))),
        ),
      ),
      p({ class: 'check-foot' },
        `Hold the second row still and cat really does come out ${pct(rates[1]!)} of the time ` +
        `instead of ${pct(pCat)}. That is a real bend, and a bend held still is what a ` +
        `watermark is. The third row bends it the same distance the other way, the first and ` +
        `last do not bend it at all, and the key deals a fresh pair at every position in the ` +
        `text, so what a reader without it gets is the average: the bag's own number, exactly.`),
    ]
  }

  /** The explainer, and it is deliberately short.
   *
   *  Its reader has just played with the other two tabs and has four or five questions. It
   *  answers those and stops. An earlier draft ran to nine sections and set the scheme against
   *  the two other watermarking families for contrast, which is a paper's shape rather than a
   *  page's: it made the reader learn two schemes they had not asked about in order to
   *  understand the one in front of them, and every extra claim was another thing that could be
   *  wrong. What survives is what a curious reader of THIS page needs, standing on its own. */
  about() {
    return div({ class: 'panel prose' },
      h2('How it works'),
      p({ class: 'aside' },
        'Everything here is tournament sampling, published as SynthID-Text by ',
        source(PAPER_SYNTHID, 'Dathathri and colleagues in Nature (2024)'),
        ' and running in Gemini. ',
        source(PAPER_ANTHROPIC, 'Anthropic says'),
        ' Claude\'s watermark is a version of the same approach, without publishing its ' +
        'settings, so this is the method rather than the exact configuration.'),

      h3('A tournament among the model\'s own words'),
      p('A language model does not choose the next word. It works out a probability for every ' +
        'word in its vocabulary and draws one. Often dozens of words would have served; ' +
        'sometimes only one can possibly come next.'),
      p('Instead of drawing once, draw several times from those same probabilities and hold a ' +
        'tournament between them. The key gives every word a coin, redrawn at every position ' +
        'and unguessable without it. In each match the word whose coin came up heads wins, and ' +
        'two matching coins are settled at random. One word survives, and that is the one ' +
        'written down.'),
      this.toyCard(),
      p('The key never proposes a word. Every candidate was the model\'s own draw before the ' +
        'key saw it, so the key cannot reach for a word the model would not have said. It only ' +
        'decides which of them wins.'),

      h3('Does it change what the model writes?'),
      p('No. Over the randomness of the key the words come out at exactly the rates the model ' +
        'asked for: not approximately, exactly, and for any number of rounds. That is a claim ' +
        'you can measure here rather than take on trust.'),
      this.fairnessCard(),
      p('Point at a word on the Generate tab and you will regularly find every slot in its ' +
        'bracket holding the same word. That is the draw rather than the position: slots are ' +
        'filled with replacement, so a word the model likes takes all of them and wins for that ' +
        'reason. Nothing was decided there and the key had no say.'),
      p('Separately, there are positions where only one word is possible at all, and those ' +
        'carry no evidence in either direction. The watermark rides entirely on freedom the ' +
        'model already had.'),
      p({ class: 'aside' },
        'Google ran an experiment spanning about 20 million Gemini responses, half of them ' +
        'watermarked. The rate at which people gave a response a thumbs up differed by 0.01% ' +
        'from unwatermarked, which they report as not statistically significant.'),

      h3('Finding it again'),
      p('The coins are seeded from the four words in front of each position, so anyone holding ' +
        'the key can work them out from the text alone, with no model and no record of how it ' +
        'was written. Walk the words, look up the coins belonging to the word that is actually ' +
        'there, and count the heads. Someone writing without the key gets heads half the time, ' +
        'so text running far enough above half is the evidence, and every extra round of the ' +
        'tournament is another coin to count. The published configuration runs thirty of them.'),
      p('One rule keeps the count honest. If the same four words have already come up, that ' +
        'position is skipped, because reusing a context would reuse the coins. Those are the ' +
        'greyed out words, along with the opening few that have nothing in front of them.'),

      h3('What a result does and does not tell you'),
      p('It says the text probably came out of a watermarked model. It does not say who wrote ' +
        'it. People use models to proofread, translate, summarise and reformat, and the words ' +
        'handed back carry the mark whoever thought of them first.'),
      p('It carries one bit and nothing else: marked, or not. Nor is it something a stranger ' +
        'can do to you. Without the key every coin is unpredictable, and the test is exactly as ' +
        'informative as drawing lots.'),
      p('A low score proves nothing either. A short passage, or a heavily rewritten one, can ' +
        'carry a mark that no longer counts high enough to see, which is why the meter reads ',
        em('no signal'), ' rather than ', em('not watermarked'), '.'),

      h3('The model here, and what this page simplifies'),
      p('The writing is done by TinyStories-1M: 3.7 million parameters, trained only on simple ' +
        'synthetic stories, which is why it writes like a children\'s book. It runs entirely ' +
        'on your device. Nothing about the scheme needs the model to be small or good, because ' +
        'the tournament acts on the probabilities, and every language model produces those.'),
      p('Two simplifications worth naming. The score here is the plain mean of the coins, while ' +
        'the detector the paper publishes uses a learned Bayesian score that ',
        source(PAPER_ATTACK, 'later work'),
        ' finds more robust. And nothing here is built to survive editing, which the paper ' +
        'does measure.'),
      p('The paper also defines a variant that runs more than two candidates in each match. It ' +
        'needs less text to detect, and it gets there by bending the model\'s rates, so it ' +
        'is neither what Gemini runs nor what is here.'),
      p('Most of the copy in this explainer was, ironically, generated by Opus 5 (though with ' +
        'considerable feedback to streamline the explanations).'),

      p({ class: 'aside' },
        'Word is a convenient lie throughout this page. The real unit is a token, a word piece, ' +
        'which is why the inspector writes a leading space as · rather than dropping it: for ' +
        'the model that space is part of the token.'),
    )
  }
}
const root = new Root()
new App({ root, id: 'app' })

tb.onMessage((m: unknown) => {
  if (m === 'selftest') return root.engine.selftest()
  // The document as the detector receives it, prompt and all. Paired with `read`, it is what
  // lets a probe hold one marked text still and vary what surrounds it.
  if (m === 'story') return root.engine.fullText()
  if (m === 'stats') return root.engine.runStats()
  if (m === 'state') return {
    ready: root.engine.ready, status: root.engine.status,
    tab: root.tab,
    tokens: root.engine.steps.length,
    generating: root.engine.generating,
    settings: root.engine.settings,
  }
  // Whether this client actually has a rendered page, asked of the client itself. `tb:snapshot`
  // reads the accessibility outline of whichever page the CLI is talking to, and when several
  // clients are attached to one bulb that is not necessarily the window a human is looking at.
  // A live app answering probes from a document with an empty mount is the case that reads as a
  // catastrophic render bug and is not one.
  if (m === 'dom') {
    const el = document.getElementById('app')
    return {
      mount: !!el,
      children: el?.childElementCount ?? 0,
      html: el?.innerHTML.length ?? 0,
      // Rendered truth for the toy card's marks, which `tb:snapshot` cannot see because they
      // are classes rather than text: `twice` is the stacked pair, and its absence when two
      // different tickets carry the same word is correct rather than a miss.
      tickets: [...document.querySelectorAll('.toy-ticket')].map(t => t.className),
      title: document.title,
      url: location.href,
      hidden: document.hidden,
      size: `${window.innerWidth}x${window.innerHeight}`,
    }
  }
  // The two measurements the page rests on, at counts too slow for a button.
  if (m === 'verify') return root.engine.checkFairness(8000)
  if (m === 'agreement') return root.engine.checkAgreement(200000)
  // The explainer's toy match. `toy` plays one and leaves it on screen; {"toy": n} runs n
  // silently and returns the shares, which is how the five-draws-in-nine in its copy was
  // measured rather than asserted.
  if (m === 'toy') {
    const d = root.toyDraw()
    return {
      words: [TOY_NAME[TOY_BAG[d.picked[0]]!], TOY_NAME[TOY_BAG[d.picked[1]]!]],
      key: { cat: d.key[0] ? 'heads' : 'tails', dog: d.key[1] ? 'heads' : 'tails' },
      winner: TOY_NAME[d.winner], how: d.how,
      // Which physical tickets, not just which words: drawing the SAME ticket twice is the
      // case that renders as a stacked pair, and two different cat tickets is not that case.
      picked: d.picked, sameTicket: d.picked[0] === d.picked[1],
    }
  }
  if (typeof m === 'object' && m !== null && 'toy' in m) {
    const n = Math.max(1, Number((m as { toy: unknown }).toy))
    let cat = 0, walkover = 0, coin = 0, toss = 0
    for (let i = 0; i < n; i++) {
      const d = playToy()
      if (d.winner === 0) cat++
      if (d.how === 'walkover') walkover++
      else if (d.how === 'coin') coin++
      else toss++
    }
    return { runs: n, cat: cat / n, walkover: walkover / n, coin: coin / n, toss: toss / n }
  }
  // Pin a word's bracket open. The hover path has no terminal equivalent, so this is how the
  // rendered card gets read back (pin, then tb:snapshot); -1 dismisses it.
  if (typeof m === 'object' && m !== null && 'pin' in m) {
    const i = Number((m as { pin: unknown }).pin)
    root.engine.hovered = -1
    root.engine.selected = i
    root.tab = 'write'
    root.update()
    const s = root.engine.steps[i]
    return {
      pinned: i, of: root.engine.steps.length,
      token: s?.text ?? null, heads: s?.heads ?? null,
      distinct: s?.bracket?.distinct ?? null, masked: s?.masked ?? null,
    }
  }
  // Score arbitrary text against the key without touching the model. The `set` probe cannot
  // answer this: it regenerates, and regenerating replaces the detector's text with the new
  // story. Anything asking what happens to ONE marked text as its surroundings change has to
  // hold the mark still and vary only what is read with it.
  if (typeof m === 'object' && m !== null && 'read' in m) {
    root.engine.detectorText = String((m as { read: unknown }).read)
    root.engine.update()
    const found = root.engine.detectScore()
    if (!found) return undefined
    return { ...scoreSummary(found.score), dropped: found.score.dropped }
  }
  // Drive the settings from the terminal and get the run's numbers back. The toggles are
  // unlabelled buttons, so `tb:click` cannot reach them, and every question about how the
  // scheme behaves at some temperature or layer count otherwise needs a human to drag a slider.
  if (typeof m === 'object' && m !== null && 'set' in m) {
    Object.assign(root.engine, (m as { set: Partial<Settings> }).set)
    root.engine.update()
    return root.engine.generate().then(() => ({
      settings: root.engine.settings,
      // The story itself, so one call answers both what a setting scores and what it reads
      // like. Choosing a default needs the two together and a second probe can race a reload.
      text: root.engine.text(),
      ...root.engine.runStats(),
    }))
  }
})
```

**index.html**

```html
<div id="app"></div>
```

**styles.css**

```css
/* One hue carries the key's evidence, in the two places that evidence appears: the coins in
   the inspector, and the wash on a word whose coins mostly came up heads. Same meaning, same
   colour, and nowhere else on the page — the accent blue already means "you can interact with
   this", so spending it here would read as a link. Only the lit side is painted: a wash on the
   rest would leave the eye judging a ratio between two tints instead of reading the lit share,
   and it would read as a verdict on words that did nothing wrong.

   Violet rather than the amber this started as, which went muddy at a 20% tint: a wash is a
   thin slice of a hue, so it wants a hue with enough chroma to survive being thinned. Checked
   against the accent blue, which it has to stay distinct from, and kept off the red-green axis
   so it never reads as a verdict on the words it marks.

   There is no second mark, and its absence is deliberate. The green list bulb underlines the
   words its key changed. This key changes no words, and a channel for an event that never
   happens teaches the opposite of the scheme. */
:root {
  --bg: #F4F4F5;
  --fg: #1C1C1C;
  --muted: #5A5A5A;
  --panel: #FFFFFF;
  --raised: #F6F6F7;
  --border: #E3E3E6;
  --hairline: #EDEDEF;
  /* "You can interact with this", and nothing else on the page: tabs, links, the slider thumbs,
     the meter's fill.

     Quiet on purpose. This page has one colour that carries meaning, and it is the violet below;
     the accent is chrome. The stock blue it started as, and the cyan tried after it, both ran at
     around 80% saturation against the violet's 83, so two equally loud voices sat next to each
     other and neither read as chosen. This runs at 34%, a third of the violet's, which leaves
     exactly one saturated thing on the page: the evidence. Contrast came along for free, 7.86
     against the page and 8.64 on a panel, where the original blue failed AA for link text at 3.62.

     Amber is the textbook complement to violet and was measured: 124 degrees off, and rejected at
     90% saturation. It would be the loudest thing here, and it paints the meter's fill, where a
     warm bar lengthening as the evidence mounts reads as a warning about the text. Same reason
     teal lost despite the widest gap of all: green there reads as a verdict. */
  --accent: #334E68;
  --card-shadow: 0 1px 2px rgba(18, 18, 23, .05), 0 1px 1px rgba(18, 18, 23, .03);
  --marked: #7C3AED;
  --tint-marked: 20%;
  --tint-marked-hover: 32%;

  /* The type scale, and the whole of it: every size on the page names a role here rather than
     picking a number at the point of use. Ten of them used to sit inside a quarter of a rem of
     each other, which reads as unfinished rather than as considered. Adding a twelfth should feel
     like a decision. */
  --text-title: 1.6rem;     /* the page's name */
  --text-figure: 1.35rem;   /* a number that is the answer */
  --text-section: 1.15rem;  /* a section's name */
  --text-lead: 1.05rem;     /* a headline reading, and the line that opens the page */
  --text-body: 1rem;        /* the marked text itself */
  --text-head: .95rem;      /* a card's name, and the words on the toy's cards */
  --text-label: .85rem;     /* a control's name, a column head */
  --text-small: .8rem;      /* a reading beside a label, a chip, a hint, a verdict */
  --text-fine: .74rem;      /* under a figure or a meter, and inside the inspector card */
  --text-token: .7rem;      /* a word shown as a token */
  --text-axis: .64rem;      /* an axis or a unit label inside the inspector */

  --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  /* The ground a bar runs along. Written out twice before this, which is how two things that mean
     one thing start to drift. */
  --track: color-mix(in srgb, var(--fg) 10%, transparent);
}

html[data-theme="dark"] {
  --bg: #161616;
  --fg: #E8E8E8;
  --muted: #979797;
  --panel: #1E1E1E;
  --raised: #262626;
  --border: #2E2E2E;
  --hairline: #272727;
  --card-shadow: none;
  /* Its own value here, which it did not have before: one accent across both themes left the dark
     page at 4.55, where this reads 8.29. Lifted rather than merely lightened, because a slate this
     desaturated goes to near-white on a dark ground and stops reading as interactive at all. */
  --accent: #8FB3D9;
  --marked: #A78BFA;
  --tint-marked: 26%;
  --tint-marked-hover: 38%;
}

body { background: var(--bg); color: var(--fg); }

.app {
  max-width: 62rem;
  margin: 0 auto;
  padding: 2rem 1.25rem 4rem;
  font: 15px/1.55 var(--sans);
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.header h1 { font-size: var(--text-title); margin: 0 0 .35rem; letter-spacing: -.01em; }
/* No reading measure of its own. Every other block on the page runs to the app's width, and a
   standfirst set narrower than everything under it reads as a wrapping accident rather than as
   a measure. */
.tagline { margin: 0; color: var(--muted); }

.tabs { display: flex; gap: .25rem; border-bottom: 1px solid var(--border); }
.tab {
  font: inherit; padding: .5rem .9rem; background: transparent; color: var(--muted);
  border: none; border-bottom: 2px solid transparent; cursor: pointer;
}
.tab:hover { color: var(--fg); }
.tab.active { color: var(--fg); border-bottom-color: var(--accent); }

.stage { display: flex; flex-direction: column; gap: 1rem; }

.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: var(--card-shadow);
  padding: 1.1rem 1.2rem;
}
.panel h3 { font-size: var(--text-head); margin: 0; letter-spacing: .01em; }

/* ---- loading ------------------------------------------------------------- */

.loading {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 1rem; padding: 3.5rem 1.5rem; text-align: center;
  background: var(--panel); border: 1px solid var(--border); box-shadow: var(--card-shadow);
}
.loading-lead { margin: 0; font-size: var(--text-lead); max-width: 32rem; }
.progress { width: 100%; max-width: 32rem; display: flex; flex-direction: column; gap: .55rem; }
.progress-track { height: 6px; overflow: hidden; background: var(--track); }
.progress-fill {
  height: 100%; background-color: var(--accent);
  background-image: linear-gradient(90deg, transparent, rgba(255,255,255,.38), transparent);
  background-size: 45% 100%; background-repeat: no-repeat;
  animation: sheen 1.6s linear infinite; transition: width .35s ease;
}
@keyframes sheen { from { background-position: -60% 0; } to { background-position: 160% 0; } }
.progress-row { display: flex; justify-content: space-between; font-size: var(--text-label); color: var(--muted); }
.progress-pct { font-variant-numeric: tabular-nums; }
@media (prefers-reduced-motion: reduce) { .progress-fill { animation: none; background-image: none; } }

/* ---- controls ------------------------------------------------------------ */

/* Both input panels head their box the same way: the label on the left, what acts on the box on
   the right, wrapping under when the row cannot hold both. */
.panel-head {
  display: flex; justify-content: space-between; align-items: center;
  gap: .4rem 1rem; flex-wrap: wrap; margin-bottom: .65rem;
}
.head-actions { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
.chip {
  font: inherit; font-size: var(--text-small); padding: .25rem .65rem; cursor: pointer;
  background: var(--raised); color: var(--fg); border: 1px solid var(--border); border-radius: 999px;
}
.chip[disabled] { opacity: .45; cursor: default; }
/* Beats the shared hover below on specificity, which is what stops a control that cannot be
   pressed from answering the pointer as though it could. */
.chip[disabled]:hover { color: var(--fg); border-color: var(--border); }

.prompt-input, .detector-input {
  width: 100%; font: inherit; padding: .6rem .7rem; resize: vertical;
  background: var(--raised); color: var(--fg); border: 1px solid var(--border);
  line-height: 1.5;
}
.prompt-input:focus, .detector-input:focus { outline: none; border-color: var(--accent); }

/* Head, slider and help are three bands shared across the row rather than three boxes each knob
   stacks for itself. A label wrapping to two lines in one column used to push that column's
   slider a line below its neighbours', because every knob measured its own head and the grid only
   ever agreed on the outer box. Subgrid makes the bands line up whatever any one of them wraps to.
   A knob without a help paragraph, which is how the check card shows these two, spans two. */
.knobs {
  display: grid; gap: .9rem 1.1rem; margin: .3rem 0 .4rem;
  grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
}
.knob { display: grid; grid-template-rows: subgrid; grid-row: span 2; row-gap: .3rem; }
.knob:has(.knob-help) { grid-row: span 3; }
.mainbar { display: flex; align-items: center; gap: .55rem; flex-wrap: wrap; margin: .85rem 0 .3rem; }
/* One label-and-value pair, wherever a control names itself and reports where it stands: the name
   in the text colour, the reading beside it muted but at the same weight, so the two read as one
   line rather than as a heading with a footnote. The main toggle and the sliders used to differ by
   .05rem on each half, which is the kind of gap nobody can see and everybody can feel. */
.mainbar-label, .knob-label { font-size: var(--text-label); font-weight: 600; }
.mainbar-value, .knob-value {
  font-size: var(--text-small); font-weight: 600; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.mainbar-value { min-width: 1.5rem; }
/* Body size, like every other line of prose on the page. In an explainer this short there is no
   tier of writing worth demoting to fine print. Labels, values and the instruments keep their
   smaller sizes, being furniture rather than prose. */
.mainbar-help { margin: 0 0 .5rem; color: var(--muted); }
.disclose {
  font: inherit; font-size: var(--text-small); cursor: pointer; margin-left: .5rem;
  background: var(--raised); color: var(--muted);
  border: 1px solid var(--border); border-radius: 999px; padding: .25rem .75rem;
  display: inline-flex; align-items: center; gap: .45rem;
}
.disclose.open { color: var(--fg); }
/* Drawn rather than typed: a glyph triangle renders at whatever size the font feels like, and at
   this size that was about six pixels. Borders give an exact one, and it can rotate. */
.disclose::before {
  content: ''; width: 0; height: 0; transition: transform .15s ease;
  border-left: 7px solid currentColor;
  border-top: 5px solid transparent;
  border-bottom: 5px solid transparent;
}
.disclose.open::before { transform: rotate(90deg); }
/* One hover for every pill button, and it is the accent, because the accent is this page's word
   for "answers to the pointer" and these were the only controls not saying it. Text and edge
   together: a border shifting on its own is a change a reader notices without reading. Set after
   `.disclose.open`, which ties it on specificity, so hovering an open disclosure still answers.
   The drawn triangle rides `currentColor` and comes along for free. */
.chip:hover, .disclose:hover { color: var(--accent); border-color: var(--accent); }
.knob-head { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.knob input[type="range"] { width: 100%; accent-color: var(--accent); }
.knob-help { margin: 0; color: var(--muted); line-height: 1.45; }

.toggle {
  width: 2.4rem; height: 1.3rem; border-radius: 999px; padding: 0; cursor: pointer;
  background: color-mix(in srgb, var(--fg) 18%, transparent);
  border: 1px solid var(--border); position: relative; transition: background .15s;
}
.toggle.on { background: var(--accent); border-color: var(--accent); }
.toggle-knob {
  position: absolute; top: 50%; left: 2px; transform: translateY(-50%);
  width: 1rem; height: 1rem; border-radius: 50%; background: #fff; transition: left .15s;
}
.toggle.on .toggle-knob { left: calc(100% - 1rem - 2px); }

.hint { font-size: var(--text-small); color: var(--muted); }

/* ---- story --------------------------------------------------------------- */

.output-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 1.5rem; flex-wrap: wrap; }
/* Both tabs put their marked text on the same surface, and the surface is what separates it from
   the legend under it. No overflow on the story, unlike the read-back: the brackets hang out of
   its box. */
.story, .read-back {
  margin-top: .8rem; padding: .7rem .8rem; line-height: 1.9; font-size: var(--text-body);
  background: var(--raised); border: 1px solid var(--border);
  white-space: pre-wrap; overflow-wrap: break-word;
}
.panel-foot { display: flex; flex-direction: column; gap: .55rem; margin-top: .9rem; }
/* A button that reads as a word, because it is one: it sits inside a sentence and its job is the
   verb of that sentence. */
.inline-link {
  font: inherit; background: none; border: none; padding: 0; cursor: pointer;
  color: var(--accent); text-decoration: underline;
  text-decoration-color: color-mix(in srgb, var(--accent) 40%, transparent);
  text-underline-offset: 2px;
}
.inline-link:hover { text-decoration-color: var(--accent); }
.inline-link[disabled] { color: var(--muted); text-decoration: none; cursor: default; }
/* One voice for every line that sits under a result, on both tabs. Not fine print in either
   case: a legend nobody can read without leaning in is a legend nobody reads. */
.story-key, .read-key, .detector-odds, .detector-stats {
  margin: 0; line-height: 1.5; color: var(--fg);
}
/* Plain ground. Painting every token faintly to advertise that they are all clickable would
   saturate the one channel the mark has: a wash reads as a mark against unpainted text and as
   nothing against a page already washed. Clickability is a hover affordance instead.

   Square corners, and no gap between them. Tokens tile the text with nothing in between (a
   leading space belongs to the token that follows it), so a radius would round only the ends
   that happen to sit where two classes meet, and a gap would open holes inside words. Runs of
   one colour are meant to merge: what the eye should pick up is how much of the paragraph is
   lit, not where the token boundaries fell. */
.tok { padding: .1em 0; transition: background .12s; position: relative; }
/* The one mark on the page, and the same one on both tabs: a word whose coins mostly came up
   heads. Paired with `.rtok`, the detector's read-back, so the two can never drift. */
.tok.lit, .rtok.lit { background: color-mix(in srgb, var(--marked) var(--tint-marked), transparent); }
/* The prompt, which the test reads exactly as it reads the story and which is where a reader
   can see for themselves what unmarked text looks like under the same key. Muted, because it is
   the part a person wrote, and still marked, because a legend quoting counts over words the
   reader cannot see marked is a legend that does not add up. */
.tok.echo { color: var(--muted); }
/* Positions the test could not count. Faded is what uncounted looks like, and every one of them
   is named in the legend rather than only the interesting ones. */
.tok.unscored, .rtok.unscored { opacity: .45; }
.tok:not(.echo) { cursor: pointer; }
.tok:not(.echo):hover { background: color-mix(in srgb, var(--fg) 9%, transparent); }
.tok.lit:not(.echo):hover { background: color-mix(in srgb, var(--marked) var(--tint-marked-hover), transparent); }
.tok.selected { outline: 2px solid var(--accent); outline-offset: 1px; }
.caret {
  display: inline-block; width: .5em; height: 1.1em; vertical-align: text-bottom;
  background: var(--accent); animation: blink 1s steps(2) infinite;
}
@keyframes blink { 50% { opacity: 0; } }
@media (prefers-reduced-motion: reduce) { .caret { animation: none; } }

/* ---- meter --------------------------------------------------------------- */

/* Wide enough that the foot's count line sits on one line at any ordinary window width. It was
   22rem, which wrapped "283 of 476 coins came up heads (59.5% against 50% by chance)" onto two.
   Still shrinkable, so a narrow window wraps it rather than overflowing, and the heading beside
   it has room to spare either way. */
.meter { min-width: 17rem; flex: 0 1 29rem; display: flex; flex-direction: column; gap: .3rem; }
.meter-row { display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; }
/* Odds against chance, because that is the one unit a reader needs no statistics for. The
   z-score is still reported, in the small print, for the readers who prefer it. */
.meter-odds { font-size: var(--text-lead); font-weight: 600; font-variant-numeric: tabular-nums; }
.meter-sub { font-size: var(--text-fine); color: var(--muted); margin-top: -.1rem; }
.meter-z { font-variant-numeric: tabular-nums; white-space: nowrap; }
.meter-verdict { font-size: var(--text-small); text-transform: uppercase; letter-spacing: .04em; }
.meter-verdict.strong { color: var(--fg); font-weight: 600; }
.meter-verdict.weak, .meter-verdict.none { color: var(--muted); }
.meter-track { position: relative; height: 8px; background: var(--track); }
.meter-fill { height: 100%; background: var(--accent); transition: width .2s ease; }
.meter-tick { position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--fg); opacity: .45; }
.meter-foot { display: flex; justify-content: space-between; gap: .8rem; font-size: var(--text-fine); color: var(--muted); }

/* ---- the inspector -------------------------------------------------------- */

/* Anchored to the token's own box, so it needs no measurement to follow the word. `bottom: 100%`
   with the gap as transparent padding *inside* the popover keeps the whole path from word to card
   within the token's element: leaving for the card never fires its mouseleave. */
.pop {
  position: absolute; bottom: 100%; z-index: 5;
  width: 23rem; max-width: calc(100vw - 2rem); padding-bottom: 7px;
}
.pop.center { left: 50%; transform: translateX(-50%); }
.pop.start { left: 0; }
.pop.end { right: 0; }
.pop-card {
  background: var(--panel); border: 1px solid var(--border); padding: .5rem .6rem;
  box-shadow: 0 4px 16px rgba(18, 18, 23, .16);
  /* The story sets pre-wrap and a 1.9 line-height for prose; neither suits a chart. */
  white-space: normal; cursor: default;
  font: 400 var(--text-fine)/1.45 var(--sans);
}
html[data-theme="dark"] .pop-card { box-shadow: 0 4px 16px rgba(0, 0, 0, .55); }
.pop-head {
  display: flex; justify-content: space-between; gap: .6rem;
  color: var(--muted); margin-bottom: .35rem;
}
.pop-head strong {
  color: var(--fg); font-family: var(--mono);
}
.pop-note {
  margin-top: .5rem; padding-top: .45rem; border-top: 1px solid var(--hairline);
  color: var(--muted);
}
.pop-note strong {
  color: var(--fg); font-family: var(--mono);
}
/* The card's one claim, rather than a description of its chart. */
.pop-note.lead { color: var(--fg); }

/* ---- the bracket ---------------------------------------------------------- */

/* Columns are knockout rounds and rows are slots, so a candidate's cell spans as many rows as
   the sub-bracket it won. That spanning IS the bracket: each survivor sits centred against the
   two cells it came from, which is the shape a reader recognises without being told. */
/* Horizontal only, and explicitly so. `overflow-x: auto` on its own computes overflow-y to `auto`
   as well, and a sixteen-slot bracket sets rows a fraction of a pixel shorter than the line box
   they hold, so the card came up with a vertical scrollbar that had nothing to scroll to. The row
   height below fixes the overflow; this stops any future fraction of a pixel doing it again. */
.bracket-wrap { overflow-x: auto; overflow-y: hidden; }
.bracket { display: grid; gap: 1px 4px; min-width: 15rem; }
.slot {
  display: flex; align-items: center; gap: .25rem; min-width: 0; padding: 0 .2rem;
  border-left: 1px solid var(--hairline);
}
/* Its own line-height, not the card's 1.45: at .68rem that is a .99rem line box, which does not
   fit the .95rem rows a sixteen-slot bracket uses. */
.slot-tok {
  font-family: var(--mono); font-size: var(--text-token);
  line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
/* Knocked out, not wrong: the word fades and the eye follows what survives across the columns.

   The fade sits on the WORD and not on the slot, which is a fix rather than a preference. On the
   slot it took the coin down with it, so a losing heads coin rendered as a pale violet that read
   as a third fill weight carrying some third meaning, and there is no third meaning. It also hid
   the one thing that explains a knockout: a loser's coin is the reason it lost, so it has to stay
   legible. Two channels, and they no longer multiply into four. */
.slot.out .slot-tok { opacity: .4; }
.slot.final {
  border: 1px solid color-mix(in srgb, var(--marked) 45%, transparent);
  background: color-mix(in srgb, var(--marked) var(--tint-marked), transparent);
  justify-content: center;
}
.slot.final .slot-tok { font-weight: 700; }
/* A coin, drawn rather than written: filled for heads, hollow for tails. Two matching coins in
   one match mean the match was settled at random, which the pair of identical marks says on its
   own without a third channel to explain it. */
.coin {
  flex: none; width: .5rem; height: .5rem; border-radius: 50%;
  border: 1px solid color-mix(in srgb, var(--marked) 70%, transparent);
}
.coin.heads { background: var(--marked); }
/* The bracket's two channels, named. They are the only marks on the page that were left to be
   worked out, and a mark that has to be worked out gets worked out wrong: same shape as the
   story's legend, the appearance and then what it means. */
.bracket-key { margin: .4rem 0 0; color: var(--muted); font-size: var(--text-fine); }
/* Every coin of the word that was written, which is exactly what the test counts here. */
.coins {
  display: flex; flex-wrap: wrap; gap: 2px; margin-top: .45rem;
  padding-top: .45rem; border-top: 1px solid var(--hairline);
}

/* ---- the candidate list (no bracket to draw) ------------------------------ */

.bars { display: flex; flex-direction: column; gap: .2rem; }
.bar-head, .bar-row {
  display: grid; grid-template-columns: 3.8rem 1fr 2.2rem; align-items: center; gap: .4rem;
}
.bar-head { color: var(--muted); font-size: var(--text-axis); margin-bottom: .15rem; }
.axis { display: flex; justify-content: space-between; }
.bar-tok {
  font-family: var(--mono); font-size: var(--text-token);
  white-space: pre; overflow: hidden; text-overflow: ellipsis; text-align: right;
}
/* Gridlines every 25%, so a length can be read off the axis instead of off a number beside it. */
.bar-track {
  position: relative; height: 1.1rem;
  border-right: 1px solid color-mix(in srgb, var(--fg) 15%, transparent);
  background-image: repeating-linear-gradient(to right,
    color-mix(in srgb, var(--fg) 15%, transparent) 0 1px, transparent 1px 25%);
}
.bar-p {
  position: absolute; left: 0; top: 30%; bottom: 30%; min-width: 1px;
  background: color-mix(in srgb, var(--fg) 42%, transparent);
}
.bar-heads {
  text-align: right; font-variant-numeric: tabular-nums; color: var(--muted);
  font-size: var(--text-axis);
}
.bar-row.chosen .bar-tok, .bar-row.chosen .bar-heads { font-weight: 700; color: var(--fg); }
.bar-row.chosen {
  background: color-mix(in srgb, var(--fg) 8%, transparent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--fg) 8%, transparent);
}

/* ---- detector ------------------------------------------------------------ */

.read-back { max-height: 17rem; overflow-y: auto; }
.rtok { padding: .1em 0; }
/* A trailing note inside a text surface, in both places one is needed: the story that ran out of
   length, and the pasted text that ran past what the detector reads. */
.text-more { color: var(--muted); }

/* ---- prose --------------------------------------------------------------- */

/* The explainer fills the panel like every other section rather than sitting in a narrower
   reading measure of its own. It carries two instruments now, and a card set narrower than the
   story and detector panels reads as a different kind of thing than they are. */
.prose { max-width: none; }
.prose h2 { font-size: var(--text-section); margin: 0 0 .6rem; }
.prose h3 { font-size: var(--text-head); margin: 1.4rem 0 .4rem; }
.prose p { margin: 0 0 .7rem; }
.prose .aside {
  color: var(--muted);
  border-left: 2px solid var(--border); padding-left: .75rem;
}
.prose a {
  color: var(--accent); text-underline-offset: 2px;
  text-decoration-color: color-mix(in srgb, var(--accent) 40%, transparent);
}
.prose a:hover { text-decoration-color: var(--accent); }

/* The claim, measured on the page, with the knobs that move it. A panel inside the prose rather
   than beside it, because it belongs to the paragraph above it. */
.check {
  border: 1px solid var(--border); background: var(--raised);
  padding: .9rem 1rem; margin: .2rem 0 1rem;
}
/* A chip inside a card is the one place the default chip fill disappears: `.chip` paints itself
   `--raised` and so does `.check`, so the control and its container are the same surface and only
   the 1px border separates them. Every other chip on the page sits on a `.panel`, which is
   `--panel`, and reads fine.

   Stepping the fill toward `--fg` rather than picking a lighter or darker value is what makes this
   work in both themes from one rule: `--fg` flips with the theme, so the chip darkens on the light
   page and lightens on the dark one, and in both cases it moves AWAY from the card. Same idiom as
   `.toggle`, which mixes `--fg` into its own track for the same reason.

   The hover is restated rather than inherited. `.chip:hover` and `.check .chip` both score one
   class plus one element, so the later of the two wins outright, and this block sits after it:
   without the line below, giving the chip a resting border here would silently kill its hover
   border everywhere inside a card. Same trap as `.check p` beating `.check-caption`. */
.check .chip {
  background: color-mix(in srgb, var(--fg) 10%, var(--raised));
  border-color: color-mix(in srgb, var(--fg) 20%, transparent);
}
.check .chip:not([disabled]):hover { color: var(--accent); border-color: var(--accent); }
.check-head {
  display: flex; justify-content: space-between; align-items: center;
  gap: .5rem 1rem; flex-wrap: wrap; margin-bottom: .5rem;
}
/* The heading's prose margins do not belong inside a card. 1.4rem above opened a gap under the
   card's own padding, and being asymmetric against .4rem below, on a flex line whose items are
   centred by margin box, it sat the heading half a rem lower than the status beside it. */
.check .check-head h3 { margin: 0; }
.check p { margin: 0 0 .6rem; }
.check-knobs {
  display: grid; gap: .9rem 1.1rem; margin: .2rem 0 .9rem;
  grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
}
/* Qualified with the element, because `.check p` above sets a margin on every paragraph in the
   card and beats a bare class on specificity. Both of these looked like they had spacing and
   had none. */
.check p.check-foot { margin: .6rem 0 0; color: var(--muted); }
/* The reading itself, kept directly under the sliders that move it: a figure a reader has to
   scroll to find is a figure they cannot tell has changed. */
.check-figures {
  display: flex; justify-content: center; gap: 1.8rem; flex-wrap: wrap; margin: .1rem 0 .7rem;
  transition: opacity .15s ease;
}
/* Dimmed rather than blanked while a new reading comes in: the figure a reader was looking at
   stays legible, and its replacement lands in the same place instead of after a gap. */
.check-figures.busy { opacity: .45; }
.check-status { font-size: var(--text-small); color: var(--muted); }
.figure { display: flex; flex-direction: column; align-items: center; }
.figure-value {
  font-size: var(--text-figure); font-weight: 700; font-variant-numeric: tabular-nums;
  letter-spacing: -.01em; line-height: 1.2;
}
.figure-label { font-size: var(--text-fine); color: var(--muted); }
.check-table {
  width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums;
}
/* Equal columns rather than content-sized ones. At auto width the slack in a 100% table lands
   wherever the widest cell puts it, which reads as the whole table having drifted to one edge. */
.check-table { table-layout: fixed; }
/* Column names carry the same weight and colour as a control's label, because that is what they
   are: the name of the thing under them. Muted, they read as a caption on the numbers. */
.check-table th {
  text-align: left; font-size: var(--text-label); font-weight: 600; color: var(--fg);
  border-bottom: 1px solid var(--border); padding: .3rem .4rem;
}
.check-table td { padding: .28rem .4rem; border-bottom: 1px solid var(--hairline); }
/* Centred, not ranged right. Two columns of percentages read against each other are being
   compared, not summed, and there is no column of totals under them for a decimal point to line
   up with. */
.check-table .num, .check-table th.num { text-align: center; }
.check-table .mono { font-family: var(--mono); }

/* ---- the toy match, inside the explainer ---------------------------------- */

/* Its own small vocabulary of marks rather than the story's. These are tickets in a bag, not
   words in a text, and painting a winner with --marked would say the key had reached for it,
   which is the exact misreading this card exists to undo. */
/* The bag and the match it feeds are a diagram, not prose, so they centre while the sentence
   underneath stays ranged left. Every element in the row is min-width'd wide enough for its
   longest state, so the reveal fills the slots without the row shifting under the pointer. */
.toy-bag { display: flex; justify-content: center; gap: .4rem; margin: 0 0 .9rem; }
/* One card, whether it is sitting in the bag or standing in a slot, because it IS one thing in
   two places. Same footprint either way: a ticket that grew on being drawn would read as the
   draw having changed it, which is the whole misreading this card exists to prevent. The slot
   carries a second line for its coin, so both reserve the height of two. */
.toy-ticket, .toy-slot, .toy-winner {
  box-sizing: border-box;
  min-width: 4.8rem; min-height: 3rem; padding: .35rem .5rem;
  border: 1px solid var(--border); background: var(--panel); border-radius: .3rem;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: .05rem; font-size: var(--text-head); line-height: 1.25; text-align: center;
}
.toy-ticket {
  transition: border-color .15s ease, box-shadow .15s ease;
}
/* Drawn tickets are outlined where they sit rather than lifted out of the row. Raising them
   moved the row under the eye at the exact moment the reader was comparing it with the two
   rows below, and the outline already says which two came out. */
.toy-ticket.out { border-color: var(--fg); box-shadow: inset 0 0 0 1px var(--fg); }
/* The same ticket drawn for both slots, shown the way a pair of cards is shown anywhere else:
   a second one behind it, offset down and right. Two shadows do it, painted back to front, so
   the outer draws the ghost card's edge and the inner fills it. Nothing here affects layout,
   which is the point: the row does not move when a draw happens to repeat. */
.toy-ticket.twice {
  border-color: var(--fg);
  box-shadow:
    inset 0 0 0 1px var(--fg),
    .3rem .3rem 0 -1px var(--panel),
    .3rem .3rem 0 0 var(--fg);
}

.toy-match {
  display: flex; justify-content: center; align-items: center; gap: .6rem; flex-wrap: wrap;
  margin-bottom: .35rem;
}
/* An empty slot used to be `background: transparent`, which on a `.check` card means it paints
   nothing and shows `--raised` through itself: identical to its container, with only a `--border`
   dashed outline to say it is there at all. Same failure as the chip above, one step worse,
   because a dashed line at `--border` is fainter than a solid one.

   It is deliberately NOT given the filled slot's `--panel`. Empty and filled should differ in
   three channels rather than two, so the fill steps only a little way off the card while the
   dashed border and the muted `?` carry the rest. Mixing toward `--fg` again, so the step is
   away from the container on both themes rather than only one. Kept below the chip's 10% because
   this is a place waiting to be filled, not a control asking to be pressed. */
.toy-slot:not(.filled) {
  border-style: dashed; color: var(--muted);
  background: color-mix(in srgb, var(--fg) 6%, var(--raised));
  border-color: color-mix(in srgb, var(--fg) 16%, transparent);
}
/* Both lines hold their height with or without text, so a coin appearing at the last stage moves
   nothing under the pointer. Reserved in CSS rather than by a space in the markup, which is what
   the empty slot used to do: there it reserved a coin line the slot did not want, and pushed its
   placeholder off the centre every ticket in the bag sits on. */
.toy-card-word { font-weight: 600; min-height: 1.25em; }
.toy-card-coin { font-size: var(--text-token); color: var(--muted); min-height: 1.25em; }
.toy-vs { color: var(--muted); font-size: var(--text-small); }
.toy-outcome {
  display: flex; justify-content: center; align-items: center;
  margin-bottom: .5rem; opacity: 0; transition: opacity .2s ease;
}
.toy-outcome.shown { opacity: 1; }
/* The word that survived, on the same card it arrived on, outlined like a drawn ticket. It is
   the same object a third time rather than a summary of one, which is what a bracket shows. */
.toy-winner { border-color: var(--fg); }
/* Held at the height of its longest verdict, so resolving a match never shifts the table
   underneath it out from under the pointer. */
.toy-note { min-height: 3.1em; opacity: 0; transition: opacity .2s ease; }
.toy-note.shown { opacity: 1; }
.check-table tr.toy-used td { font-weight: 700; }
.toy-used-tag { color: var(--muted); font-weight: 400; font-size: var(--text-fine); }
.check-table tr.toy-avg td { font-weight: 700; border-top: 1px solid var(--border); }
```

**config.json**

```json
{
  "dependencies": {
    "tensorgrad": "^0.4.7",
    "domeleon": "^0.6.6"
  },
  "description": "Learn how text watermarking works in your browser: a secret key runs a tournament between words the model drew, so the winner is one it would say."
}
```
