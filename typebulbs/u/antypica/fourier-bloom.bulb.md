---
format: typebulb/v1
name: "Fourier Bloom"
---

**code.tsx**

```tsx
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls"
import { App, Component, a, button, canvas, div, h1, inputRange, p, path, polygon, rect, span, svg } from "domeleon"
import {
  Module, compile, checkWebGPU, Linear, RMSNorm, lr,
  add, mul, matmul, swapAxes, softmaxCausal, silu, narrow, reshape,
  embedding, arange, splitHeads, mergeHeads, crossEntropy,
  type Tensor, type CompiledTraining, type CompiledForward,
} from "tensorgrad"

// ============================================================================
// Watch the 3D Fourier knot CRYSTALLIZE out of a real trained model. We train
// the d=384 transformer on c = a·b⁻¹ (mod 97) live on WebGPU (the fixed
// tensorgrad), and every few steps re-read the learned readout head W_U, project
// its 96 class columns onto the K1=11 / K2=17 Fourier directions, and move the
// three.js beads there. Random cloud → cylinder knot as it groks. Then walk the
// learned knot vs. cut a straight linear chord through the void.
// ============================================================================

const P = 97, VOCAB = P + 2, TOK_OP = P, TOK_EQ = P + 1
const N = P - 1                  // 96 invertible classes
const SEQ_LEN = 4, ANS_POS = SEQ_LEN - 1
const D_MODEL = 384, N_LAYERS = 2, N_HEADS = 12, D_FF = 4 * D_MODEL
const BATCH = 512
const LR = lr.warmup({ peak: 1e-3, steps: 10, after: 1e-3 })
const WD = 1.0, BETA1 = 0.9, BETA2 = 0.98
const MAX_STEPS = 900
const POLISH_STEPS = 100         // after first 100/100, train this many more so the knot sharpens, then stop
const VAL_FRAC = 0.5
const KNOT_EVERY = 8             // re-read W_U + reproject every N steps
const ACC_EVERY = 25
const K1 = 11, K2 = 17           // fallback harmonics; per-run we auto-detect the model's own
const SCALE = 1.7
const CLUSTER0 = 0.25            // initial random-init blob, as a fraction of the cylinder radius; it blooms out to ≈ fill the cylinder (tune to taste)

const range = (n: number) => Array.from({ length: n }, (_, i) => i)

// ---- number theory: discrete log mod 97 ----
function modpow(b: number, e: number, m: number) {
  let r = 1
  b %= m
  while (e > 0) {
    if (e & 1) r = (r * b) % m
    b = (b * b) % m
    e >>= 1
  }
  return r
}
const modinv = (b: number) => modpow(b, P - 2, P)

function findGenerator() {
  for (let g = 2; g < P; g++) {
    if (modpow(g, (P - 1) / 2, P) !== 1 && modpow(g, (P - 1) / 3, P) !== 1) return g
  }
  return 5
}
const G = findGenerator()

const DLOG: number[] = (() => {
  const d = new Array(P).fill(0)
  let x = 1
  for (let l = 0; l < N; l++) { d[x] = l; x = (x * G) % P }
  return d
})()
const INVERT = range(N).map(i => i + 1)   // classes 1..96

// ---- dataset ----
type Pair = { a: number; b: number; c: number }
const isVal = (a: number, b: number) => (((a * 131 + b) * 2654435761) >>> 0) % 1000 < VAL_FRAC * 1000
const ALL: Pair[] = []
for (let a = 1; a < P; a++) for (let b = 1; b < P; b++) ALL.push({ a, b, c: (a * modinv(b)) % P })
const TRAIN = ALL.filter(p => !isVal(p.a, p.b)), VAL = ALL.filter(p => isVal(p.a, p.b))
const enc = (p: Pair) => [p.a, TOK_OP, p.b, TOK_EQ]

function makeBatch() {
  const tokens = new Int32Array(BATCH * SEQ_LEN), targets = new Int32Array(BATCH)
  for (let i = 0; i < BATCH; i++) {
    const p = TRAIN[(Math.random() * TRAIN.length) | 0]!
    const s = enc(p)
    for (let t = 0; t < SEQ_LEN; t++) tokens[i * SEQ_LEN + t] = s[t]!
    targets[i] = p.c
  }
  return { tokens, targets }
}
const EVAL = (pairs: Pair[]) => pairs.filter((_, i) => i % 7 === 0).slice(0, 120)
const EV_T = EVAL(TRAIN), EV_V = EVAL(VAL)

function evalTok(pairs: Pair[]) {
  const t = new Int32Array(pairs.length * SEQ_LEN)
  for (let i = 0; i < pairs.length; i++) {
    const s = enc(pairs[i]!)
    for (let k = 0; k < SEQ_LEN; k++) t[i * SEQ_LEN + k] = s[k]!
  }
  return t
}

// ---- model (same arch as fourier-bloom) ----
class Attention extends Module {
  q = new Linear(D_MODEL, D_MODEL, { bias: false })
  k = new Linear(D_MODEL, D_MODEL, { bias: false })
  v = new Linear(D_MODEL, D_MODEL, { bias: false })
  o = new Linear(D_MODEL, D_MODEL, { bias: false })
}
class SwiGLU extends Module {
  gate = new Linear(D_MODEL, D_FF, { bias: false })
  up = new Linear(D_MODEL, D_FF, { bias: false })
  down = new Linear(D_FF, D_MODEL, { bias: false })
}
class Block extends Module {
  n1 = new RMSNorm(D_MODEL)
  attn = new Attention()
  n2 = new RMSNorm(D_MODEL)
  mlp = new SwiGLU()
}
class GrokModel extends Module {
  tok_emb: Tensor
  pos_emb: Tensor
  unembed = new Linear(D_MODEL, P, { bias: false })
  nf = new RMSNorm(D_MODEL)
  layers: Block[]
  constructor() {
    super()
    this.tok_emb = this.param([VOCAB, D_MODEL])
    this.pos_emb = this.param([SEQ_LEN, D_MODEL])
    this.layers = range(N_LAYERS).map(() => new Block())
  }
}

function attnFwd(p: Attention, x: Tensor) {
  const q = splitHeads(p.q.fwd(x), N_HEADS)
  const k = splitHeads(p.k.fwd(x), N_HEADS)
  const v = splitHeads(p.v.fwd(x), N_HEADS)
  const s = mul(matmul(q, swapAxes(k, -1, -2)), 1 / Math.sqrt(D_MODEL / N_HEADS))
  return p.o.fwd(mergeHeads(matmul(softmaxCausal(s), v)))
}
function blockFwd(p: Block, x: Tensor) {
  const x1 = add(x, attnFwd(p.attn, p.n1.fwd(x)))
  return add(
    x1,
    p.mlp.down.fwd(mul(silu(p.mlp.gate.fwd(p.n2.fwd(x1))), p.mlp.up.fwd(p.n2.fwd(x1)))),
  )
}
function modelFwd(p: GrokModel, tokens: Tensor) {
  let x = add(embedding(p.tok_emb, tokens), embedding(p.pos_emb, arange(SEQ_LEN)))
  for (const b of p.layers) x = blockFwd(b, x)
  return p.unembed.fwd(p.nf.fwd(x))
}
function lossFn(p: GrokModel, { tokens, targets }: { tokens: Tensor; targets: Tensor }) {
  const ans = reshape(narrow(modelFwd(p, tokens), 1, ANS_POS, 1), [BATCH, P])
  return crossEntropy(ans, targets, { reduction: "mean" })
}
function predictFn(p: GrokModel, { tokens }: { tokens: Tensor }) {
  return modelFwd(p, tokens)
}

// ---- project learned W_U columns -> 3D knot coords (x,z circle from k1; y from k2) ----
// W_U stored [D_MODEL, P] row-major: element (i, c) at i*P + c. Column c = class-c readout.
// We don't hard-code the harmonics: each run groks on its OWN dominant frequencies
// (chosen by init), so we read the readout's Fourier power spectrum and project onto
// THIS run's two strongest. That's why a grok now blooms reliably instead of only
// when the seed happened to land on 11/17 — we always look through the right window.
// ONE fixed scale per run, captured from the first (random-init) frame: the knot starts as
// a small blob (CLUSTER0 of the cylinder) and blooms outward as rMean grows. Deliberately NOT
// re-locked to an exact fit at grok — that snap made the knot visibly jump in size the instant
// it flipped to walk mode. clusterScale persists across retrains.
let clusterScale: number | null = null

function projectKnot(W_U: Float32Array, refine = false): { pts: [number, number, number][]; cv: number; freqs: [number, number] } {
  const cols = INVERT.map(c => {
    const v = new Float64Array(D_MODEL)
    for (let i = 0; i < D_MODEL; i++) v[i] = W_U[i * P + c]!
    return v
  })
  const mean = new Float64Array(D_MODEL)
  for (const v of cols) for (let i = 0; i < D_MODEL; i++) mean[i]! += v[i]! / N
  const cen = cols.map(v => Float64Array.from(v, (x, i) => x - mean[i]!))
  const theta = INVERT.map(c => (2 * Math.PI * DLOG[c]!) / N)
  // raw (un-normalized) Fourier direction in D_MODEL space at frequency K
  const dirRaw = (K: number, fn: (t: number) => number) => {
    const d = new Float64Array(D_MODEL)
    for (let j = 0; j < N; j++) {
      const w = fn(K * theta[j]!), r = cen[j]!
      for (let i = 0; i < D_MODEL; i++) d[i]! += w * r[i]!
    }
    return d
  }
  const norm2 = (d: Float64Array) => { let s = 0; for (const x of d) s += x * x; return s }
  const unit = (d: Float64Array) => { const n = Math.sqrt(norm2(d)) || 1; return Float64Array.from(d, x => x / n) }
  const dot = (a: Float64Array, b: Float64Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!; return s }
  const angDist = (d: number) => { const m = ((d % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI); return m > Math.PI ? 2 * Math.PI - m : m }
  // cache each frequency's cos/sin Fourier directions (the heavy part) so power,
  // decode, and the final projection all reuse them instead of recomputing
  const dirCache = new Map<number, { c: Float64Array; s: Float64Array }>()
  const dirs = (k: number) => {
    let d = dirCache.get(k)
    if (!d) { d = { c: dirRaw(k, Math.cos), s: dirRaw(k, Math.sin) }; dirCache.set(k, d) }
    return d
  }

  // candidate frequencies: coprime to N so they wind through all 96 (a frequency
  // sharing a factor with N collapses onto fewer angles, not a clean ring)
  const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a)
  const cand = range(N / 2).filter(k => k >= 1 && gcd(k, N) === 1)
  // how well frequency k's circle DECODES division on the val pairs: a real
  // algorithmic harmonic recovers the answer by angle-subtraction; a high-power-
  // but-spurious low frequency (e.g. 1, a smooth drift) does not. Consulted only
  // to refine the FINAL frequency choice (below) — never live.
  const decodeScore = (k: number) => {
    const cxk = unit(dirs(k).c), sxk = unit(dirs(k).s)
    const phiByL = new Float64Array(N)
    for (let i = 0; i < N; i++) phiByL[DLOG[INVERT[i]!]!] = Math.atan2(dot(cen[i]!, sxk), dot(cen[i]!, cxk))
    let ok = 0
    for (const p of EV_V) {
      const tgt = phiByL[DLOG[p.a]!]! - phiByL[DLOG[p.b]!]!, lc = DLOG[p.c]!
      let best = 0, bd = Infinity
      for (let l = 0; l < N; l++) { const d = angDist(phiByL[l]! - tgt); if (d < bd) { bd = d; best = l } }
      if (best === lc) ok++
    }
    return ok / EV_V.length
  }
  // pick by raw POWER for the live animation — power evolves smoothly, so the
  // bloom crystallizes without the knot jumping between projections every update.
  // Only the final locked knot (refine) re-ranks the strongest few by DECODE to
  // evict a spurious low frequency. (Decode is a noisy thresholded count — fine
  // to consult once, jittery to chase frame to frame.)
  const powOf = new Map(cand.map(k => [k, norm2(dirs(k).c) + norm2(dirs(k).s)]))
  const ranked = cand.sort((a, b) => powOf.get(b)! - powOf.get(a)!)
  let k1: number, k2: number
  if (refine) {
    const byDecode = ranked.slice(0, 10).map(k => [k, decodeScore(k)] as [number, number]).sort((a, b) => b[1] - a[1])
    k1 = byDecode[0]?.[0] ?? K1; k2 = byDecode.find(([k]) => k !== k1)?.[0] ?? K2
  } else {
    k1 = ranked[0] ?? K1; k2 = ranked.find(k => k !== k1) ?? K2
  }
  const cx = unit(dirs(k1).c), sx = unit(dirs(k1).s), cy = unit(dirs(k2).s)
  const raw = cen.map(r => [dot(r, cx), dot(r, cy), dot(r, sx)] as [number, number, number])
  const radii = raw.map(([x, , z]) => Math.hypot(x, z))
  const rMean = radii.reduce((a, b) => a + b, 0) / N || 1
  const rStd = Math.sqrt(radii.reduce((a, b) => a + (b - rMean) ** 2, 0) / N)
  // ONE fixed scale per run or the bloom cancels. rMean (the ring radius) GROWS as the harmonic
  // forms — that growth IS the bloom; dividing by it per-frame (the old SCALE / rMean) or by
  // column energy (which grows with it) pins the radius constant, so the ring only hollows out.
  // So fix the scale once from the first random-init frame (a small blob) and let it ride: the
  // knot blooms out and STAYS that size into walk mode, no snap. CLUSTER0 sets the blob size.
  if (clusterScale === null) clusterScale = (CLUSTER0 * SCALE) / rMean
  const s = clusterScale
  return { pts: raw.map(([x, y, z]) => [x * s, y * s, z * s]), cv: rStd / rMean, freqs: [k1, k2] }
}

// ============================================================================
// Trainer — the non-visual component that owns the live training run: the loop,
// its run-id lifecycle, and the status line the UI reads. It holds no three.js; it
// reaches the Stage via this.ctx.root.stage and drives it (applyCoords / beginWalk
// / resetWalk) as the model groks. Calling update() re-renders the Root tree.
// ============================================================================
class Trainer extends Component {
  status = "starting…"
  // Held from the start of a run until the first beads land on the stage. While it holds,
  // `status` is superimposed on the canvas instead of parked in the dock: compiling takes
  // seconds, and a blank scene with a caption at the bottom reads as a bulb that doesn't
  // work. `failed` marks the terminal no-WebGPU case, where the message stays put (and
  // stops shimmering) because nothing will ever render behind it.
  loading = true
  failed = false
  #runId = 0

  get #stage() { return (this.ctx.root as unknown as IRoot).stage }

  retrain() {
    const stage = this.#stage
    stage.resetWalk()
    // retrain implies play: a fresh run blocks at the `while (paused)` gate before
    // its first step, so without resuming the readout would freeze on "compiling…"
    if (stage.paused) stage.togglePause()
    this.run()
  }

  async run() {
    const stage = this.#stage
    const run = ++this.#runId          // invalidates any earlier run still looping
    this.loading = true                // a retrain recompiles too: put the status back on the stage
    const gpu = await checkWebGPU()
    if (!gpu.ok) {
      this.failed = true
      this.#setStatus(gpu.message)
      return
    }
    this.#setStatus("compiling WGSL kernels…")
    const t: CompiledTraining<GrokModel> = await compile({
      model: new GrokModel(),
      loss: lossFn,
      optimizer: { kind: "adamw", lr: LR, weightDecay: WD, beta1: BETA1, beta2: BETA2 },
      inputs: { tokens: { shape: [BATCH, SEQ_LEN], dtype: "i32" }, targets: { shape: [BATCH], dtype: "i32" } },
    })
    if (run !== this.#runId) return
    const infer = await t.attach({ forward: predictFn, inputs: { tokens: { shape: [null, SEQ_LEN], dtype: "i32" } } })
    if (run !== this.#runId) return
    // First beads on screen. The stage speaks for itself from here, so hand the running
    // commentary back to the dock readout.
    stage.applyCoords(projectKnot((await t.downloadParams())["unembed.W"]!))
    this.loading = false
    this.update()
    let step = 0, valAcc = 0, trainAcc = 0, loss = 0, cv = 1, freqs: [number, number] = [K1, K2], grokAt = -1
    while (step < MAX_STEPS) {
      if (run !== this.#runId) return   // a reset (or auto-retry) superseded this run
      while (stage.paused) {            // hold here while paused (still bail on a reset)
        await new Promise(res => setTimeout(res, 100))
        if (run !== this.#runId) return
      }
      const r = await t.step(makeBatch())
      if (r.kind === "completed") { step++; loss = r.loss }
      if (step % KNOT_EVERY === 0) {
        const pr = projectKnot((await t.downloadParams())["unembed.W"]!)
        stage.applyCoords(pr)
        cv = pr.cv; freqs = pr.freqs
      }
      if (step % ACC_EVERY === 0) {
        trainAcc = await this.#accuracy(infer, EV_T)
        valAcc = await this.#accuracy(infer, EV_V)
        if (grokAt < 0 && trainAcc >= 1 && valAcc >= 1) grokAt = step   // first perfect train+val
      }
      const ph = grokAt >= 0 ? "consolidating the knot" : valAcc >= 0.9 ? "grokked" : valAcc >= 0.15 ? "generalizing" : trainAcc >= 0.6 ? "memorizing" : "warming up"
      this.#setStatus(`step ${step}/${MAX_STEPS} · loss ${loss.toFixed(3)} · train ${(trainAcc * 100) | 0}% · val ${(valAcc * 100) | 0}% · knot CV ${cv.toFixed(2)} · K=${freqs[0]}/${freqs[1]} · ${ph}`)
      // early stop: once perfect, train only POLISH_STEPS more so the knot finishes
      // sharpening (CV keeps dropping after accuracy saturates), then lock it in
      if (grokAt >= 0 && step >= grokAt + POLISH_STEPS) break
      await new Promise(res => setTimeout(res, 0))   // yield so the render loop runs
    }
    valAcc = await this.#accuracy(infer, EV_V)
    if (run !== this.#runId) return
    if (valAcc < 0.9) {
      // unlucky init — never generalized; reroll fresh weights and try again
      this.#setStatus(`unlucky init — didn't grok (val ${(valAcc * 100) | 0}%). retraining…`)
      this.retrain()
      return
    }
    // grokked: read W_U once more, lock the learned knot, switch to walk/chord
    const fin = projectKnot((await t.downloadParams())["unembed.W"]!, true)   // refine by decode
    stage.applyCoords(fin)
    stage.beginWalk()
    this.#setStatus(`grokked on frequencies ${fin.freqs[0]} & ${fin.freqs[1]} — walk the LEARNED knot (rainbow) vs. cut the chord (blue) · CV ${fin.cv.toFixed(2)}`)
  }

  #setStatus(s: string) { this.status = s; this.update() }

  async #accuracy(infer: CompiledForward, pairs: Pair[]) {
    const r = await infer.run({ tokens: evalTok(pairs) })
    if (r.kind !== "completed") return 0
    const lg = r.output
    let ok = 0
    for (let i = 0; i < pairs.length; i++) {
      const off = (i * SEQ_LEN + ANS_POS) * P
      let best = 1
      for (let c = 2; c < P; c++) if (lg[off + c]! > lg[off + best]!) best = c
      if (best === pairs[i]!.c) ok++
    }
    return ok / pairs.length
  }
}

// ============================================================================
// Stage — a component whose "view" is the WebGL canvas rather than DOM. It owns
// everything intrinsically imperative (the three.js renderer, camera, controls,
// mesh buffers, RAF loop) as private state, and exposes the verbs the rest of the
// app drives it with: applyCoords / beginWalk / resetWalk / togglePause. morph +
// paused live here because they ARE scene state — the slider and pause button in
// Root bind straight to them.
// ============================================================================
const hue = (l: number) => new THREE.Color().setHSL(l / N, 0.7, 0.55)

class Stage extends Component {
  #morph = 0
  #paused = false
  #phase: "train" | "walk" = "train"

  #container!: HTMLElement
  #renderer!: THREE.WebGLRenderer
  #scene = new THREE.Scene()
  #camera!: THREE.PerspectiveCamera
  #controls!: OrbitControls
  #world = new THREE.Group()
  #beads: THREE.Mesh[] = []
  #knotLine!: THREE.LineLoop
  #shellMat!: THREE.MeshBasicMaterial
  #ringMat!: THREE.LineBasicMaterial
  #walkMat!: THREE.MeshStandardMaterial
  #walkMesh: THREE.Mesh | null = null
  #chord!: THREE.Line
  #startDot!: THREE.Mesh
  #endDot!: THREE.Mesh
  // learned ("real") coords by discrete-log position l, plus the platonic ideal:
  // the pure (k1,k2) torus knot the same projection would yield if W_U were perfectly
  // sinusoidal. morph (0 = real/messy, 1 = platonic/ideal) lerps between them.
  #PReal: [number, number, number][] = range(N).map(() => [0, 0, 0])
  #PPlat: [number, number, number][] = range(N).map(() => [0, 0, 0])
  #delta = 1
  #walkT = 0
  #walkDone = false        // walk has swept to full and stopped (no loop); the ✕ then appears
  #cleared = false         // ✕ pressed: path wiped, idling — ▶ play redraws it from the start
  #clock = new THREE.Clock()

  get paused() { return this.#paused }
  get walkDone() { return this.#walkDone }
  get cleared() { return this.#cleared }

  // the slider binds to `morph`; writing it re-places the beads + knot line, and
  // (so a drag while paused isn't half-applied) rebuilds the frozen walk too
  get morph() { return this.#morph }
  set morph(v: number) {
    this.#morph = v
    if (!this.#renderer) return           // slider moved before the canvas mounted
    this.#refresh()
    if (this.#phase === "walk") this.#rebuildWalk(this.#delta)
  }

  boot(canvas: HTMLCanvasElement, container: HTMLElement) {
    if (this.#renderer) return            // canvas onMounted can fire more than once
    this.#container = container
    const renderer = this.#renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    const camera = this.#camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)
    camera.position.set(5.5, 3.125, 5.5)   // viewing angle only; #resize() fits the distance
    const controls = this.#controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.7

    // lighting — a soft fill + a strong key + a cool rim so the lit materials below
    // pick up gradients and specular highlights as the scene auto-rotates (depth)
    const scene = this.#scene
    scene.add(new THREE.AmbientLight(0xffffff, 0.35))
    const key = new THREE.DirectionalLight(0xfff1e0, 2.2)
    key.position.set(5, 6, 4)
    scene.add(key)
    const rim = new THREE.DirectionalLight(0x88aaff, 0.9)
    rim.position.set(-5, -2, -4)
    scene.add(rim)
    const fill = new THREE.PointLight(0xffffff, 0.6, 0, 1.5)
    camera.add(fill)              // headlight: travels with the camera
    scene.add(camera)

    // everything geometric mounts on `world`, laid on its side so the cylinder
    // axis is horizontal — like a car tire. Auto-rotate (about the vertical) then
    // turns it like a car on a mall turntable, showing the knot's 3D form, instead
    // of spinning it flat about its own axis like a lazy susan. Lights stay in the
    // scene (not the group) so the surface shading shifts as the knot turns.
    const world = this.#world
    world.rotation.z = Math.PI / 2
    scene.add(world)

    // a faint translucent cylinder shell whispers the manifold's form.
    // depthWrite:false keeps this scaffold from occluding the knot chords behind
    // its far wall (pure overlay).
    const shellMat = this.#shellMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false })
    world.add(new THREE.Mesh(new THREE.CylinderGeometry(SCALE, SCALE, 2 * SCALE, 64, 1, true), shellMat))
    // horizontal rings (not vertical lines): they read as a round tube from any
    // angle, but run perpendicular to the bead arrangement so they can't be
    // mistaken for one-slot-per-number the way the old vertical grid was.
    const RINGS = 5
    const ringGeo = new THREE.BufferGeometry().setFromPoints(
      range(64).map(i => {
        const a = (2 * Math.PI * i) / 64
        return new THREE.Vector3(Math.cos(a) * SCALE, 0, Math.sin(a) * SCALE)
      }),
    )
    const ringMat = this.#ringMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.22, depthWrite: false })
    range(RINGS).forEach(i => {
      const ring = new THREE.LineLoop(ringGeo, ringMat)
      ring.position.y = -SCALE + (2 * SCALE * i) / (RINGS - 1)
      world.add(ring)
    })
    this.#applyScaffoldTheme()
    new MutationObserver(() => this.#applyScaffoldTheme()).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] })
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.#applyScaffoldTheme())

    // beads, indexed by discrete-log position l (so l-order == walk order)
    const beadGeo = new THREE.SphereGeometry(0.055 * SCALE, 24, 24)
    this.#beads = range(N).map(l => {
      const c = hue(l)
      const m = new THREE.Mesh(beadGeo, new THREE.MeshStandardMaterial({
        color: c,
        emissive: c.clone().multiplyScalar(0.25),   // keep hue readable in shadow
        roughness: 0.35,
        metalness: 0.1,
      }))
      world.add(m)
      return m
    })
    // full knot line (beads in l-order, closed) — the learned manifold. Vertex-
    // colored so each segment is a gradient between the two beads it joins, making
    // the loop the same rainbow as the dots (à la Yudin) rather than a flat grey.
    const knotLine = this.#knotLine = new THREE.LineLoop(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.4 }),
    )
    {
      const cols = new Float32Array(N * 3)
      range(N).forEach(l => { const c = hue(l); cols[l * 3] = c.r; cols[l * 3 + 1] = c.g; cols[l * 3 + 2] = c.b })
      knotLine.geometry.setAttribute("color", new THREE.BufferAttribute(cols, 3))
    }
    world.add(knotLine)

    // walk + chord (revealed post-grok). One reused lit material for the walk tube;
    // per-segment vertex colors (set in #rebuildWalk) make it wear the rainbow of
    // the beads it threads through.
    this.#walkMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.3, metalness: 0.25 })
    const chord = this.#chord = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineDashedMaterial({ color: 0x3b82f6, dashSize: 0.18, gapSize: 0.12 }),
    )
    chord.visible = false
    world.add(chord)
    const dot = (c: number) => new THREE.Mesh(
      new THREE.SphereGeometry(0.08 * SCALE, 32, 32),
      new THREE.MeshStandardMaterial({ color: c, emissive: 0x222222, roughness: 0.25, metalness: 0.2 }),
    )
    const startDot = this.#startDot = dot(0xffffff), endDot = this.#endDot = dot(0xffffff)
    startDot.visible = endDot.visible = false
    world.add(startDot, endDot)

    new ResizeObserver(() => this.#resize()).observe(container)
    this.#resize()
    this.#render()
  }

  // ---- public verbs the Trainer + UI drive ----

  applyCoords(pr: { pts: [number, number, number][]; freqs: [number, number] }) {
    // rebuild the platonic ideal for this run's detected (k1,k2) so the morph aligns
    const [k1, k2] = pr.freqs
    for (let l = 0; l < N; l++) {
      const th = (2 * Math.PI * l) / N
      this.#PPlat[l] = [SCALE * Math.cos(k1 * th), SCALE * Math.sin(k2 * th), SCALE * Math.sin(k1 * th)]
    }
    // pr.pts is per-INVERT class (index i -> class INVERT[i]); reindex by discrete-log l
    for (let i = 0; i < N; i++) {
      const c = INVERT[i]!, l = DLOG[c]!
      this.#PReal[l] = pr.pts[i]!
    }
    this.#refresh()
  }

  beginWalk() { this.#phase = "walk"; this.#walkDone = false; this.#cleared = false }   // grokked — sweep the walk once

  // the ✕ button: wipe the drawn path and idle, leaving the bare knot to admire. SEPARATE
  // from play/pause (the global pause — see togglePause). Clearing never redraws; ▶ play
  // does. `cleared` (not pause) idles only the sweep, so auto-rotation keeps turning.
  clearWalk() {
    this.#clearPath()
    this.#walkT = 0; this.#delta = 1; this.#walkDone = false; this.#cleared = true
    this.update()
  }

  // ▶ play from the cleared state: redraw the walk from the first connection.
  playWalk() {
    this.#cleared = false
    if (this.#paused) this.togglePause()   // unpause so the redraw (and rotation) actually runs
    this.update()
  }

  #clearPath() {
    if (this.#walkMesh) { this.#walkMesh.geometry.dispose(); this.#world.remove(this.#walkMesh); this.#walkMesh = null }
    this.#chord.visible = this.#startDot.visible = this.#endDot.visible = false
  }

  resetWalk() {
    this.#phase = "train"
    this.#clearPath()
    this.#delta = 1; this.#walkT = 0; this.#walkDone = false; this.#cleared = false
  }

  // the global pause/play. #paused gates the training loop, the walk sweep, AND auto-rotation,
  // so toggling it freezes or resumes the whole scene at whatever stage it's currently in.
  togglePause() {
    this.#paused = !this.#paused
    if (this.#controls) this.#controls.autoRotate = !this.#paused
    this.update()                          // re-render Root so the pause/play button flips
    return this.#paused
  }

  // ---- private rendering internals ----

  #disp(l: number): [number, number, number] {
    const a = this.#PReal[l]!, b = this.#PPlat[l]!, m = this.#morph
    return [a[0] + (b[0] - a[0]) * m, a[1] + (b[1] - a[1]) * m, a[2] + (b[2] - a[2]) * m]
  }

  #refresh() {
    for (let l = 0; l < N; l++) this.#beads[l]!.position.set(...this.#disp(l))
    this.#knotLine.geometry.setFromPoints(range(N).map(l => new THREE.Vector3(...this.#disp(l))))
    // fade the connecting loop with the morph: barely there when it's a noisy
    // tangle (real), brighter as it resolves into a clean knot (platonic)
    ;(this.#knotLine.material as THREE.LineBasicMaterial).opacity = 0.12 + 0.43 * this.#morph
  }

  // the scaffold is the one scene element that needs theming: its grey-blues are
  // tuned for the dark void and wash out on the light bg at this opacity, so darken
  // them there. Mirrors the CSS tokens, and tracks live theme flips.
  #applyScaffoldTheme() {
    const t = document.documentElement.getAttribute("data-theme")
    const light = t === "light" || (t !== "dark" && !matchMedia("(prefers-color-scheme: dark)").matches)
    this.#shellMat.color.setHex(light ? 0x435a74 : 0x8899aa)
    this.#ringMat.color.setHex(light ? 0x32475f : 0x556677)
  }

  // sweep a growing arc from bead 0 out to bead `delta` (the walk), plus the
  // straight chord across those same two endpoints. delta is always 1…95 and the
  // walk always starts at bead 0, so there's no signed/arbitrary-origin case.
  #rebuildWalk(delta: number) {
    const arc = range(delta + 1).map(i => new THREE.Vector3(...this.#disp(i)))
    if (this.#walkMesh) { this.#walkMesh.geometry.dispose(); this.#world.remove(this.#walkMesh) }
    if (arc.length >= 2) {
      const curve = new THREE.CatmullRomCurve3(arc)
      // TubeGeometry samples the spline by ARC LENGTH, which starves exactly the hairpins:
      // a hairpin is short in space but the sharpest bend, so it receives the FEWEST rings
      // right where it needs the most, and the straight segments between rings then chord
      // across the apex and skirt the bead ("almost round the pole, but disqualified").
      // Arc-length sampling means no ring count is clever enough to land one on each bead —
      // the only real lever is raw density: enough rings that even an under-sampled hairpin
      // has several to bend through. 12/span clears it; bump higher if any remain.
      const RADIAL = 16, tubular = delta * 12
      const geo = new THREE.TubeGeometry(curve, tubular, 0.02 * SCALE, RADIAL, false)
      // paint each tube ring the hue of the bead it sits on, interpolating between
      // consecutive beads — the walk wears the rainbow of the path it traverses
      const arcCol = range(delta + 1).map(k => hue(k))
      const colors = new Float32Array((tubular + 1) * (RADIAL + 1) * 3)
      let p = 0
      for (let i = 0; i <= tubular; i++) {
        const u = (i / tubular) * delta, k = Math.min(Math.floor(u), delta - 1), f = u - k
        const ca = arcCol[k]!, cb = arcCol[k + 1]!
        const cr = ca.r + (cb.r - ca.r) * f, cg = ca.g + (cb.g - ca.g) * f, cbl = ca.b + (cb.b - ca.b) * f
        for (let j = 0; j <= RADIAL; j++) { colors[p++] = cr; colors[p++] = cg; colors[p++] = cbl }
      }
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3))
      this.#walkMesh = new THREE.Mesh(geo, this.#walkMat)
      this.#walkMesh.visible = true
      this.#world.add(this.#walkMesh)
    }
    const a = new THREE.Vector3(...this.#disp(0)), b = new THREE.Vector3(...this.#disp(delta))
    this.#chord.geometry.dispose()
    this.#chord.geometry = new THREE.BufferGeometry().setFromPoints([a, b])
    this.#chord.computeLineDistances()
    this.#startDot.position.copy(a)
    this.#endDot.position.copy(b)
    this.#chord.visible = this.#startDot.visible = this.#endDot.visible = true
  }

  #resize() {
    const w = this.#container.clientWidth, h = this.#container.clientHeight
    this.#renderer.setSize(w, h, false)
    const camera = this.#camera
    camera.aspect = w / h
    // Fit + center the knot in the vertical BAND between the top/bottom overlays,
    // not the whole viewport, so it never crowds the header/dock text. The band is
    // measured live (text wraps differently per width) and drives framing two ways.
    const ovH = (sel: string) => (this.#container.querySelector(sel) as HTMLElement | null)?.offsetHeight ?? 0
    const headerH = ovH(".header"), dockH = ovH(".dock")
    const bandH = Math.max(1, h - headerH - dockH)
    // 1) position: shift to the band's center (down is +). A view-offset keeps the
    //    knot the orbit center so it doesn't wobble while auto-rotating — and it
    //    auto-generalizes the old hand-tuned downward nudge to every screen size.
    camera.setViewOffset(w, h, 0, -(headerH - dockH) / 2, w, h)
    // 2) size: FILL and the band cap together set the on-screen size; whichever is smaller
    //    binds. Both bumped ×1.2 (from 0.68 / 0.9) for a 20% zoom-in — the knot only reaches
    //    the cylinder, so there was headroom — so the scene scales up uniformly whichever
    //    term binds. 1.08 lets the cylinder edge sit right up to / just behind the overlays.
    const KNOT_R = SCALE * 1.55, FILL = 0.82   // fraction of the viewport the scene fills
    const halfFov = (camera.fov / 2) * Math.PI / 180
    const fit = Math.min(FILL, (bandH / h) * 1.08, FILL * (w / h))
    const dist = KNOT_R / (Math.tan(halfFov) * fit)
    const dir = camera.position.clone().sub(this.#controls.target).normalize()
    camera.position.copy(this.#controls.target).add(dir.multiplyScalar(dist))
    camera.updateProjectionMatrix()
    // repaint immediately: setSize() just resized (and cleared) the drawing buffer, and
    // waiting for the next animation frame leaves a blank frame visible — that's the blink.
    this.#renderer.render(this.#scene, this.#camera)
  }

  // render loop — always smooth, independent of training
  #render() {
    requestAnimationFrame(() => this.#render())
    const dt = this.#clock.getDelta()
    if (this.#phase === "walk" && !this.#paused && !this.#walkDone && !this.#cleared) {
      this.#walkT += dt * 7
      const d = 1 + Math.floor(this.#walkT)                 // sweep Δ = 1 … 95, no wrap
      if (d >= N - 1) {
        this.#delta = N - 1                                 // last bead reached: all dots connected
        this.#rebuildWalk(this.#delta)
        this.#walkDone = true
        this.update()                                      // re-render so the button shows ▶ play ✕
      } else {
        this.#delta = d
        this.#rebuildWalk(this.#delta)
      }
    }
    this.#controls.update()   // always — keeps damping + manual drag alive even when paused
    this.#renderer.render(this.#scene, this.#camera)
  }
}

// ============================================================================
// Root — the only component with a DOM view: the header, the canvas the Stage
// mounts on, and the dock (legend + morph slider + retrain/pause + status). It
// owns the two logic components as public fields, so domeleon wires their ctx and
// re-renders this tree whenever either calls update(). The UI binds straight to
// their state — no status callback, no scene handle, no mirrored flags.
// ============================================================================
// ---- inline SVG icons: real geometry, not font glyphs, so they center like any box and
// inherit the button's color via currentColor (no second font's metrics to fight) ----
const icon = (...kids: any[]) => svg({ class: "ico", viewBox: "0 0 24 24", fill: "currentColor" }, ...kids)
const ICON = {
  play: () => icon(polygon({ points: "8,5 8,19 19,12" })),
  pause: () => icon(rect({ x: 7, y: 5, width: 3.5, height: 14, rx: 1 }), rect({ x: 13.5, y: 5, width: 3.5, height: 14, rx: 1 })),
  retrain: () => icon(path({ d: "M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" })),
  clear: () => svg({ class: "ico", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2.4, strokeLineCap: "round" }, path({ d: "M6 6L18 18M6 18L18 6" })),
}

interface IRoot { trainer: Trainer; stage: Stage }

class Root extends Component implements IRoot {
  trainer = new Trainer()
  stage = new Stage()

  #boot(el: Element) {
    const cv = el as HTMLCanvasElement
    this.stage.boot(cv, cv.parentElement as HTMLElement)
    this.trainer.run()
  }

  view() {
    const t = this.trainer
    return div({ class: t.loading ? ["stage", "busy"] : "stage" },
      canvas({ class: "scene", key: "scene", onMounted: (el: Element) => this.#boot(el) }),
      // The compile/reload status, superimposed over the middle of the scene (the same move
      // kata-go makes over its board) rather than left in the dock where it's easy to miss.
      t.loading
        ? div({ class: "note-overlay" },
            p({ class: "stage-note" },
              t.failed ? t.status : span({ class: "shimmer" }, t.status)))
        : null,
      div({ class: "header" },
        h1({ class: "title" }, "Fourier Bloom"),
        p({ class: "tag" }, "a tiny transformer learns clock math in your browser"),
        p({ class: "cap" }, "A tiny transformer learns to divide on a 97-hour clock (c = a ÷ b, mod 97). Plot each answer as a dot and you can watch it think: they start as a random cloud and bloom into a clean knot as it “groks” the rule."),
        p({ class: "credit" },
          "Based on Nikolay Yudin's ",
          a({ class: "credit-link", href: "https://nick-yudin.github.io/manifold_features/fourier-bloom/", target: "_blank", rel: "noopener noreferrer" }, "work you can learn about here"),
          ".",
        ),
      ),
      div({ class: "dock" },
        div({ class: "legend" },
          span({ class: ["k", "walk"] }, "● walk the curve"),
          span({ class: ["k", "chord"] }, "▦ cut straight across"),
        ),
        div({ class: "morph" },
          span({ class: "morph-lbl" }, "real"),
          inputRange({
            target: this.stage,
            prop: () => this.stage.morph,
            attrs: { min: 0, max: 1, step: 0.01, title: "Drag from real to platonic to straighten the messy learned knot into its ideal shape" },
          }),
          span({ class: "morph-lbl" }, "platonic"),
          span({ class: "morph-sep" }),
          button({ class: "icon-btn", title: "retrain the model", onClick: () => this.trainer.retrain() }, ICON.retrain()),
          // transport: icon-only circles. play/pause is the global pause — it freezes/resumes
          // training, the walk sweep, and rotation together (and redraws from the cleared
          // state). The clear ✕ is its own circle, shown only once the walk is drawn.
          button(
            {
              class: "icon-btn",
              title: this.stage.paused || this.stage.cleared ? "play" : "pause",
              onClick: () => this.stage.cleared ? this.stage.playWalk() : this.stage.togglePause(),
            },
            this.stage.paused || this.stage.cleared ? ICON.play() : ICON.pause(),
          ),
          this.stage.walkDone
            ? button({ class: "icon-btn", title: "clear the path", onClick: () => this.stage.clearWalk() }, ICON.clear())
            : null,
        ),
        div({ class: "readout" }, this.trainer.status),
      ),
    )
  }
}

new App({ root: new Root(), id: "app" })
```

**index.html**

```html
<div id="app"></div>
```

**styles.css**

```css
/* Theme tokens — light defaults on :root; dark overrides on
   html[data-theme="dark"]. The host doesn't always set the attribute, so :root
   must be a fully usable light theme on its own. The WebGL canvas is alpha:true
   (transparent), so the 3D knot sits straight on --bg and adapts for free —
   only this DOM overlay needs theming. */
:root {
  color-scheme: light;
  /* one sans for all UI text (the inherited default), one mono reserved for live data
     (the status readout). Everything else should match by inheriting — don't re-declare
     a font on a rule unless it is genuinely a different KIND of text. */
  --font: ui-sans-serif, system-ui, sans-serif;
  --font-mono: ui-monospace, Menlo, Consolas, monospace;
  /* ONE body/UI text size, inherited by everything. Only deliberate headings (title, tag)
     opt out. Don't add a font-size to a rule unless it's a true heading — every control,
     label, legend and readout should be this size, so they stay consistent by default. */
  --text: 0.9rem;
  --accent: #ff7a18;      /* brand orange — constant across themes (slider fill + focus ring) */
  --bg: #eef1f6;          /* soft cool grey, not stark white, so the rainbow knot still pops */
  --fg: #1a1d24;
  --muted: #5a6273;
  --readout: #3a4250;
  --tag: #c2620e;         /* darkened orange — the dark-theme #ff9a4d is too light on a pale bg */
  --panel: #ffffffcc;
  --panel-border: #cdd3de;
  --btn-bg: #ffffff;
  --btn-fg-hover: #11141c;
  --btn-border-hover: #aab2c0;
  --credit-link: #3a5da8;
  --credit-link-hover: #1f3d80;
  --scrim-top: linear-gradient(#eef1f6e6, #eef1f600);
  --scrim-bottom: linear-gradient(#eef1f600, #eef1f6f2);
}
html[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0b0d12;
  --fg: #e7e9ee;
  --muted: #aeb6c4;
  --readout: #cdd3dd;
  --tag: #ff9a4d;
  --panel: #11141ccc;
  --panel-border: #2a2f3a;
  --btn-bg: #1b1f2a;
  --btn-fg-hover: #ffffff;
  --btn-border-hover: #3a4250;
  --credit-link: #9bb3d6;
  --credit-link-hover: #cfe0ff;
  --scrim-top: linear-gradient(#0b0d12e6, #0b0d1200);
  --scrim-bottom: linear-gradient(#0b0d1200, #0b0d12f2);
}
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg); font-family: var(--font); font-size: var(--text); }
#app { position: relative; width: 100%; height: 100dvh; min-height: 580px; overflow: hidden; --gap: 14px; }
.stage { position: absolute; inset: 0; }
.scene { position: absolute; inset: 0; width: 100%; height: 100%; display: block; user-select: none; }
/* header + footer are identical flex columns; one --gap drives every vertical space.
   pointer-events:auto so the text bands CAPTURE drags instead of passing them through
   to the OrbitControls canvas underneath — otherwise a drag that starts even slightly
   off the glyphs (gaps, padding, empty sides) falls through and rotates the scene,
   eating the text selection. The knot sits in the central area, which stays draggable;
   only the thin top/bottom bands stop rotating-by-drag (auto-rotate is unaffected). */
.header, .dock { position: absolute; left: 0; right: 0; display: flex; flex-direction: column; align-items: center; gap: var(--gap); padding: var(--gap) 18px; text-align: center; pointer-events: auto; }
.header { top: 0; background: var(--scrim-top); }
/* bump bottom padding a touch: the top gap sits above a tall 1.5rem h1 whose
   line-box leading reads as extra space, so match it at the small readout line */
.dock { bottom: 0; background: var(--scrim-bottom); padding-bottom: calc(var(--gap) + 4px); }
.header > *, .dock > * { margin: 0; }   /* kill default h1/p margins — gap is the only spacer */
.header > *, .header a, .dock .morph, .dock .legend, .dock .readout { pointer-events: auto; }
/* prose reads as selectable text (cursor + explicit user-select, in case anything upstream opts out) */
.title, .tag, .cap, .credit, .readout { user-select: text; cursor: text; }
.title { font-size: 1.5rem; font-weight: 680; letter-spacing: .04em; text-transform: uppercase; color: var(--fg); }
.tag { font-size: 1.25rem; font-style: italic; color: var(--tag); }   /* a clear step above the prose, not level with it */
.cap { max-width: 680px; line-height: 1.5; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 16px; }
.legend .walk { background: linear-gradient(90deg, #ff5a5a, #ffd24d, #5ad6a0, #5aa0ff, #c77dff); -webkit-background-clip: text; background-clip: text; color: transparent; }
.legend .chord { color: #3b82f6; }
.readout { font-family: var(--font-mono); color: var(--readout); }   /* mono: live data, the one place it's warranted */
.morph { display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 8px; max-width: calc(100vw - 28px); padding: 7px 12px; border-radius: 18px; background: var(--panel); border: 1px solid var(--panel-border); backdrop-filter: blur(6px); }
.morph input[type=range] { width: 130px; flex: none; accent-color: var(--accent); cursor: pointer; }
.morph .icon-btn, .morph .morph-lbl { white-space: nowrap; flex: none; }
/* the only text left in the dock controls is the slider's real/platonic labels; like all bare
   text they center on the font baseline, a hair below the slider track's geometric centerline,
   so lift them to match. (The icon circles removed every other piece of control text.) */
.morph-lbl { color: var(--muted); transform: translateY(-2px); }
.credit { line-height: 1.5; color: var(--muted); }
.credit-link { color: var(--credit-link); }
.credit-link:hover { color: var(--credit-link-hover); }
/* every control is an icon-only circle — no text inside, so the in-button text-alignment
   problem is gone entirely. Fixed square = never resizes; the icon centers as plain geometry. */
.icon-btn { cursor: pointer; box-sizing: border-box; display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; color: var(--readout); background: var(--btn-bg); border: 1px solid var(--panel-border); border-radius: 50%; }
.icon-btn:hover { color: var(--btn-fg-hover); border-color: var(--btn-border-hover); }
/* one consistent focus ring instead of the UA default (which differs per color-scheme) */
.icon-btn:focus-visible, .morph input[type=range]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.ico { width: 1.45em; height: 1.45em; flex: none; }   /* the icon glyph inside a button */
.morph-sep { width: 1px; align-self: stretch; background: var(--panel-border); }
/* Boot status, superimposed on the scene. Until the first knot lands the canvas is empty,
   so a status parked at the bottom of the dock reads as "this bulb is broken" rather than
   "this bulb is loading". pointer-events:none so the card never swallows a drag headed for
   the OrbitControls canvas underneath. */
.note-overlay { position: absolute; inset: 0; display: grid; place-items: center; padding: 0 18px; pointer-events: none; }
/* A card, not bare text: the beads fade in underneath the moment coords land, and loose
   glyphs over a rotating rainbow knot are unreadable. Matches the .morph dock panel. */
.stage-note { margin: 0; padding: 8px 14px; max-width: 34ch; border-radius: 12px; background: var(--panel); border: 1px solid var(--panel-border); backdrop-filter: blur(6px); font-family: var(--font-mono); color: var(--readout); line-height: 1.5; text-align: center; }
/* Sweep a highlight band across the glyphs themselves (background-clip:text) so the wait
   reads as the text working, rather than as a spinner parked beside it. Both stops are
   themed tokens, so it stays legible either way; with motion suppressed it degrades to a
   static gradient rather than to invisible text. */
.shimmer { display: inline-block; background: linear-gradient(90deg, var(--readout) 35%, var(--accent) 50%, var(--readout) 65%); background-size: 200% 100%; -webkit-background-clip: text; background-clip: text; color: transparent; animation: shimmer 1.6s linear infinite; }
@keyframes shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }
@media (prefers-reduced-motion: reduce) { .shimmer { animation: none; } }
/* Hidden, not emptied: the readout keeps its box, so the dock height is identical before
   and after the overlay goes and #resize()'s band fit doesn't jump when training starts. */
.stage.busy .readout { visibility: hidden; }
/* short viewports: the overlays don't shrink with height, so reclaim band space
   for the knot — drop the descriptive caption and tighten the vertical rhythm */
@media (max-height: 620px) {
  #app { --gap: 9px; }
  .cap { display: none; }
  .title { font-size: 1.3rem; }
}
```

**config.json**

```json
{
  "description": "A tiny transformer learns clock math in your browser.",
  "dependencies": {
    "three": "^0.160.0",
    "tensorgrad": "^0.4.6",
    "domeleon": "^0.6.3"
  }
}
```
