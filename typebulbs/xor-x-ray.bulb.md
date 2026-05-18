---
format: typebulb/v1
name: "XOR X-ray"
---

**code.tsx**

```tsx
import { App, Component, div, h1, h2, h3, p, pre, span, button, a, svg, g, circle, line, path, rect, text } from "domeleon"

type Params = {
  W1: number[][]  // [hidden][input] — W1[j][i] is the weight from input i to hidden j
  b1: number[]
  W2: number[]    // length 2 — single output, so this is a vector not a matrix
  b2: number
}

type AdamState = {
  mW1: number[][]; vW1: number[][]
  mb1: number[];   vb1: number[]
  mW2: number[];   vW2: number[]
  mb2: number;     vb2: number
}

type Forward = {
  x: number[]; z1: number[]; h: number[]; z2: number; y: number; target: number; loss: number
}

type Backward = {
  dy: number; dz2: number; dh: number[]; dz1: number[]
  dW1: number[][]; db1: number[]; dW2: number[]; db2: number
}

type Deltas = {
  dW1: number[][]; db1: number[]; dW2: number[]; db2: number
}

// Mulberry32 — small reproducible PRNG.
function makeRng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6D2B79F5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function initParams(seed: number): Params {
  const rand = makeRng(seed)
  const r = () => (rand() * 2 - 1) * 0.8
  return {
    W1: [[r(), r()], [r(), r()]],
    b1: [r() * 0.1, r() * 0.1],
    W2: [r(), r()],
    b2: r() * 0.1,
  }
}

function initAdam(): AdamState {
  return {
    mW1: [[0, 0], [0, 0]], vW1: [[0, 0], [0, 0]],
    mb1: [0, 0], vb1: [0, 0],
    mW2: [0, 0], vW2: [0, 0],
    mb2: 0, vb2: 0,
  }
}

type HiddenAct = "tanh" | "relu"

// Output activation is hardcoded to tanh — only the hidden choice toggles.
// Output needs a bounded range for MSE against {0,1} targets.
function hAct(z: number, kind: HiddenAct): number {
  return kind === "tanh" ? Math.tanh(z) : Math.max(0, z)
}
function hActGrad(h: number, z: number, kind: HiddenAct): number {
  return kind === "tanh" ? (1 - h * h) : (z > 0 ? 1 : 0)
}

// Forward: x -> W1·x + b1 -> hAct -> W2·h + b2 -> tanh -> (y - target)²
function forward(p: Params, x: number[], target: number, hk: HiddenAct): Forward {
  const z1 = [
    p.W1[0][0] * x[0] + p.W1[0][1] * x[1] + p.b1[0],
    p.W1[1][0] * x[0] + p.W1[1][1] * x[1] + p.b1[1],
  ]
  const hVals = [hAct(z1[0], hk), hAct(z1[1], hk)]
  const z2 = p.W2[0] * hVals[0] + p.W2[1] * hVals[1] + p.b2
  const y = Math.tanh(z2)
  const loss = (y - target) ** 2
  return { x, z1, h: hVals, z2, y, target, loss }
}

// Backward: chain rule op by op; each line below labels which adjoint it computes.
function backward(p: Params, f: Forward, hk: HiddenAct): Backward {
  const dy  = 2 * (f.y - f.target)                       // ∂L/∂y
  const dz2 = dy * (1 - f.y * f.y)                       // through output tanh
  const dh  = [dz2 * p.W2[0], dz2 * p.W2[1]]             // through W2
  const dz1 = [dh[0] * hActGrad(f.h[0], f.z1[0], hk),    // through hidden activation
               dh[1] * hActGrad(f.h[1], f.z1[1], hk)]
  const dW1 = [                                          // dW1[j][i] = dz1[j] · x[i]
    [dz1[0] * f.x[0], dz1[0] * f.x[1]],
    [dz1[1] * f.x[0], dz1[1] * f.x[1]],
  ]
  const db1 = [dz1[0], dz1[1]]
  const dW2 = [dz2 * f.h[0], dz2 * f.h[1]]
  const db2 = dz2
  return { dy, dz2, dh, dz1, dW1, db1, dW2, db2 }
}

function applySGD(p: Params, g: Backward, lr: number): Deltas {
  const dW1: number[][] = [[0, 0], [0, 0]]
  const db1: number[]   = [0, 0]
  const dW2: number[]   = [0, 0]
  for (let j = 0; j < 2; j++) {
    for (let i = 0; i < 2; i++) {
      const d = -lr * g.dW1[j][i]; dW1[j][i] = d; p.W1[j][i] += d
    }
    db1[j] = -lr * g.db1[j]; p.b1[j] += db1[j]
    dW2[j] = -lr * g.dW2[j]; p.W2[j] += dW2[j]
  }
  const db2 = -lr * g.db2; p.b2 += db2
  return { dW1, db1, dW2, db2 }
}

// Adam: per-parameter running averages of gradient (m) and squared gradient (v),
// bias-corrected, step = -lr · m̂ / (√v̂ + ε). Every parameter has its own m, v.
function applyAdam(p: Params, g: Backward, s: AdamState, t: number, lr: number): Deltas {
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8
  const c1 = 1 - Math.pow(beta1, t)
  const c2 = 1 - Math.pow(beta2, t)
  const upd = (grad: number, m: number, v: number) => {
    const newM = beta1 * m + (1 - beta1) * grad
    const newV = beta2 * v + (1 - beta2) * grad * grad
    const mHat = newM / c1
    const vHat = newV / c2
    return { delta: -lr * mHat / (Math.sqrt(vHat) + eps), m: newM, v: newV }
  }
  const dW1: number[][] = [[0, 0], [0, 0]]
  const db1: number[]   = [0, 0]
  const dW2: number[]   = [0, 0]
  for (let j = 0; j < 2; j++) {
    for (let i = 0; i < 2; i++) {
      const u = upd(g.dW1[j][i], s.mW1[j][i], s.vW1[j][i])
      dW1[j][i] = u.delta; p.W1[j][i] += u.delta
      s.mW1[j][i] = u.m; s.vW1[j][i] = u.v
    }
    const u1 = upd(g.db1[j], s.mb1[j], s.vb1[j])
    db1[j] = u1.delta; p.b1[j] += u1.delta
    s.mb1[j] = u1.m; s.vb1[j] = u1.v
    const u2 = upd(g.dW2[j], s.mW2[j], s.vW2[j])
    dW2[j] = u2.delta; p.W2[j] += u2.delta
    s.mW2[j] = u2.m; s.vW2[j] = u2.v
  }
  const u3 = upd(g.db2, s.mb2, s.vb2)
  const db2 = u3.delta; p.b2 += db2
  s.mb2 = u3.m; s.vb2 = u3.v
  return { dW1, db1, dW2, db2 }
}

const EXAMPLES = [
  { x: [0, 0], y: 0 },
  { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 },
  { x: [1, 1], y: 0 },
]

// 'idle' → first click computes forward.
// forward → backward → updated → (next click) advances exIdx and forwards next example.
type Phase = "idle" | "forward" | "backward" | "updated"

// Granularity of one "advance" — orthogonal from "advance once vs run":
//   phase = one sub-step (forward, backward, or updated)
//   step  = one full training step (3 phases, one example)
//   epoch = one full pass through all 4 examples (12 phases)
type StepUnit = "phase" | "step" | "epoch"

class Demo {
  p!: Params
  s!: AdamState
  exIdx = 0
  phase: Phase = "idle"
  optimizer: "sgd" | "adam" = "sgd"
  hiddenAct: HiddenAct = "tanh"
  stepUnit: StepUnit = "phase"
  lr = 0.1
  t = 0                  // Adam bias-correction counter (only increments on Adam updates, so it's distinct from trainStep)
  trainStep = 0
  seed = 7
  // fwd is always populated (constructor → reset() → forward on example 0);
  // diagram never looks naked at idle.
  fwd!: Forward
  bwd: Backward | null = null
  deltas: Deltas | null = null
  lossHistory: { loss: number, exIdx: number }[] = []
  // All-time max loss for chart scaling. Monotonically non-decreasing so the
  // y-axis doesn't rescale upward as old high-loss bars age out of the visible
  // window (which would create a false "loss going back up" appearance).
  lossMax = 0

  constructor() { this.reset() }

  reset() {
    this.p = initParams(this.seed)
    this.s = initAdam()
    this.exIdx = 0
    this.phase = "idle"
    this.t = 0
    this.trainStep = 0
    this.fwd = forward(this.p, EXAMPLES[0].x, EXAMPLES[0].y, this.hiddenAct)
    this.bwd = null; this.deltas = null
    this.lossHistory = []
    this.lossMax = 0
  }

  reseed() {
    this.seed = Math.floor(Math.random() * 100000)
    this.reset()
  }

  step() {
    if (this.phase === "idle" || this.phase === "updated") {
      if (this.phase === "updated") this.exIdx = (this.exIdx + 1) % EXAMPLES.length
      const ex = EXAMPLES[this.exIdx]
      this.fwd = forward(this.p, ex.x, ex.y, this.hiddenAct)
      this.bwd = null; this.deltas = null
      this.phase = "forward"
    } else if (this.phase === "forward") {
      this.bwd = backward(this.p, this.fwd, this.hiddenAct)
      this.phase = "backward"
    } else if (this.phase === "backward") {
      if (this.optimizer === "adam") {
        this.t++
        this.deltas = applyAdam(this.p, this.bwd!, this.s, this.t, this.lr)
      } else {
        this.deltas = applySGD(this.p, this.bwd!, this.lr)
      }
      this.lossHistory.push({ loss: this.fwd.loss, exIdx: this.exIdx })
      if (this.lossHistory.length > 200) this.lossHistory.shift()
      if (this.fwd.loss > this.lossMax) this.lossMax = this.fwd.loss
      this.trainStep++
      this.phase = "updated"
    }
  }

  fullStep() {
    do { this.step() } while (this.phase !== "updated")
  }

  // One pass through all 4 examples. Always lands on "updated" so the
  // user sees the post-update state — a partial start (mid-step) gets
  // completed by the first fullStep, then 3 more full steps follow.
  cycleStep() {
    for (let i = 0; i < 4; i++) this.fullStep()
  }
}

// Inline SVG icons. Font glyphs (▶ ⏸ ↺) center unpredictably across
// platforms; SVG with a fixed viewBox is reliable. Shapes use currentColor
// so the button's `color` flows through (e.g. `.accent` for purple).
function btnIcon(...shapes: any[]) {
  return svg({ viewBox: "0 0 16 16", width: "14", height: "14", class: "btn-icon" }, ...shapes)
}
const iconPlay  = () => btnIcon(path({ d: "M3 2 L13 8 L3 14 Z", fill: "currentColor" }))
const iconPause = () => btnIcon(
  rect({ x: 3,  y: 3, width: 3, height: 10, fill: "currentColor" }),
  rect({ x: 10, y: 3, width: 3, height: 10, fill: "currentColor" }),
)
const iconStep = () => btnIcon(
  path({ d: "M3 3 L11 8 L3 13 Z", fill: "currentColor" }),
  rect({ x: 12, y: 3, width: 2, height: 10, fill: "currentColor" }),
)

function fmt(x: number, d = 2): string {
  if (Number.isNaN(x)) return "NaN"
  if (Math.abs(x) < 1e-6) return "0"
  return x.toFixed(d)
}

function fmtSign(x: number, d = 3): string {
  if (Number.isNaN(x)) return "NaN"
  const s = x.toFixed(d)
  return x >= 0 ? "+" + s : s
}

// Uses --w-pos / --w-neg so neuron tints stay consistent with edge colors
// across themes. Hardcoded rgba() would lock one palette in for both.
function neuronFill(v: number): string {
  const pct = (0.12 + 0.4 * Math.min(Math.abs(v), 1)) * 100
  const token = v >= 0 ? "var(--w-pos)" : "var(--w-neg)"
  return `color-mix(in srgb, ${token} ${pct}%, transparent)`
}

const X_IN = 80, X_HID = 410, X_OUT = 740
const Y_TOP = 47, Y_BOT = 217, Y_OUT = 132
const R = 24
const yFor = (j: number) => j === 0 ? Y_TOP : Y_BOT

interface IRoot {
  demo: Demo
  inspected: string | null
  inspect(key: string | null): void
  running: boolean
}

// Discriminated form of an inspected element. Click sites pass a string key
// (e.g. "W1.0.1"); parseSel turns it into a typed object so dispatch is one
// switch instead of six startsWith branches.
type Selection =
  | { kind: "x",  i: number }
  | { kind: "h",  j: number }
  | { kind: "y" }
  | { kind: "b1", j: number }
  | { kind: "b2" }
  | { kind: "W1", j: number, i: number }
  | { kind: "W2", j: number }

function parseSel(key: string): Selection {
  if (key === "y")  return { kind: "y" }
  if (key === "b2") return { kind: "b2" }
  if (key.startsWith("x."))  return { kind: "x",  i: parseInt(key.slice(2)) }
  if (key.startsWith("h."))  return { kind: "h",  j: parseInt(key.slice(2)) }
  if (key.startsWith("b1.")) return { kind: "b1", j: parseInt(key.slice(3)) }
  if (key.startsWith("W2.")) return { kind: "W2", j: parseInt(key.slice(3)) }
  if (key.startsWith("W1.")) { const [j, i] = key.slice(3).split(".").map(Number); return { kind: "W1", j, i } }
  throw new Error("parseSel: unknown key " + key)
}

// Unified accessor for a learnable parameter: name, read/write into Params,
// and the current ∂/Δ. Collapses what would otherwise be five separate
// switches inside inlineChart (one per quantity).
type ParamAccess = {
  name:  string
  read:  () => number
  write: (v: number) => void
  grad:  number | null
  delta: number | null
}
function paramAccess(d: Demo, sel: Selection): ParamAccess {
  const b = d.bwd, dt = d.deltas
  switch (sel.kind) {
    case "W1": {
      const { j, i } = sel
      return {
        name:  `W1[${j}][${i}]`,
        read:  () => d.p.W1[j][i],
        write: v  => { d.p.W1[j][i] = v },
        grad:  b  ? b.dW1[j][i]  : null,
        delta: dt ? dt.dW1[j][i] : null,
      }
    }
    case "W2": {
      const { j } = sel
      return {
        name:  `W2[${j}]`,
        read:  () => d.p.W2[j],
        write: v  => { d.p.W2[j] = v },
        grad:  b  ? b.dW2[j]  : null,
        delta: dt ? dt.dW2[j] : null,
      }
    }
    case "b1": {
      const { j } = sel
      return {
        name:  `b1[${j}]`,
        read:  () => d.p.b1[j],
        write: v  => { d.p.b1[j] = v },
        grad:  b  ? b.db1[j]  : null,
        delta: dt ? dt.db1[j] : null,
      }
    }
    case "b2":
      return {
        name:  "b2",
        read:  () => d.p.b2,
        write: v  => { d.p.b2 = v },
        grad:  b  ? b.db2  : null,
        delta: dt ? dt.db2 : null,
      }
    default:
      throw new Error("paramAccess: not a learnable param")
  }
}

class Diagram extends Component {
  get root() { return this.ctx.root as any as IRoot }
  get demo() { return this.root.demo }

  view() {
    return svg({
      viewBox: "0 0 820 264",
      width: "100%",
      preserveAspectRatio: "xMidYMid meet",
      class: "diagram-svg",
      // Click on empty SVG background deselects. Child elements (neurons,
      // edges, charts) bubble up here too, but their event.target points
      // back to themselves; only a true background click has target === svg.
      onClick: (e: any) => {
        if (e.target === e.currentTarget) this.root.inspect(null)
      },
    },
      this.edges(),
      this.neurons(),
      this.paramCharts(),
    )
  }

  edges() {
    const d = this.demo
    const out: any[] = []
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 2; i++) {
        out.push(this.edge(X_IN, yFor(i), X_HID, yFor(j), d.p.W1[j][i], `W1.${j}.${i}`))
      }
    }
    for (let j = 0; j < 2; j++) {
      out.push(this.edge(X_HID, yFor(j), X_OUT, Y_OUT, d.p.W2[j], `W2.${j}`))
    }
    return out
  }

  selectionHalo(cx: number, cy: number) {
    return circle({ cx, cy, r: R + 5, fill: "none", stroke: "var(--accent)", strokeWidth: "2", strokeOpacity: "0.6" })
  }

  // Edge is just the connection line; the value, ∂, and Δ are shown inside
  // the in-situ parameter chart positioned at the edge's midpoint (see
  // paramCharts).
  edge(x1: number, ya: number, x2: number, yb: number, w: number, key: string) {
    const isSel = this.root.inspected === key
    const thickness = Math.min(0.8 + Math.abs(w) * 2.5, 5)
    const stroke = isSel ? "var(--accent)" : (w >= 0 ? "var(--w-pos)" : "var(--w-neg)")
    return g({ style: { cursor: "pointer" }, onClick: () => this.root.inspect(key) },
      // Wide transparent hit area so thin lines are still easy to click.
      line({ x1, y1: ya, x2, y2: yb, stroke: "transparent", strokeWidth: "12" }),
      line({ x1, y1: ya, x2, y2: yb, stroke, strokeWidth: String(isSel ? Math.max(thickness + 1, 3) : thickness), strokeOpacity: isSel ? "1" : "0.65" }),
    )
  }

  // Two-circle pattern: opaque base under the rgba tint, so edges passing
  // through the circle's footprint don't bleed through the semi-transparent
  // fill. `mirror=true` puts label-and-z below instead of above (bottom-row
  // hidden neuron and the output). Label right-anchored at cx, z= left-anchored
  // at cx+3 — keeps them visually adjacent regardless of value width.
  bodyNeuron(cx: number, cy: number, label: string, post: number, pre: number, neuronKey: string, mirror = false) {
    const root = this.root
    const isNeuronSel = root.inspected === neuronKey
    const labelY = mirror ? cy + R + 17 : cy - R - 11
    return g(
      circle({ cx, cy, r: R, fill: "var(--neuron-bg)" }),
      isNeuronSel ? this.selectionHalo(cx, cy) : null,
      circle({
        cx, cy, r: R,
        fill: neuronFill(post),
        stroke: "var(--border-strong)",
        strokeWidth: "1.5",
        style: { cursor: "pointer" },
        onClick: () => root.inspect(neuronKey),
      }),
      // pointer-events:none only on the value text — it sits over the
      // clickable circle and would otherwise steal the click.
      text({ x: cx, y: cy + 5, textAnchor: "middle", class: "n-val", style: { pointerEvents: "none" } }, fmt(post, 2)),
      text({ x: cx, y: labelY, textAnchor: "end", class: "n-label" }, label),
      text({ x: cx + 3, y: labelY, textAnchor: "start", class: "n-sub" }, `z=${fmt(pre, 2)}`),
    )
  }

  neurons() {
    const d = this.demo
    const f = d.fwd
    const out: any[] = []
    const root = this.root

    for (let i = 0; i < 2; i++) {
      const y = yFor(i)
      const key = `x.${i}`
      const isSel = root.inspected === key
      // Bottom-row input: label below the circle (mirrored), so the diagram
      // is symmetric across the horizontal midline.
      const labelY = i === 1 ? y + R + 17 : y - R - 11
      out.push(g(
        isSel ? this.selectionHalo(X_IN, y) : null,
        circle({
          cx: X_IN, cy: y, r: R,
          fill: "var(--neuron-bg)",
          stroke: "var(--border-strong)",
          strokeWidth: "1.5",
          style: { cursor: "pointer" },
          onClick: () => root.inspect(key),
        }),
        text({ x: X_IN, y: y + 5, textAnchor: "middle", class: "n-val", style: { pointerEvents: "none" } }, fmt(f.x[i], 1)),
        text({ x: X_IN, y: labelY, textAnchor: "middle", class: "n-label" }, `x${i+1}`),
      ))
    }

    for (let j = 0; j < 2; j++) {
      out.push(this.bodyNeuron(
        X_HID, yFor(j), `h${j+1}`,
        f.h[j],
        f.z1[j],
        `h.${j}`,
        j === 1, // mirror the bottom-row neuron
      ))
    }

    out.push(this.bodyNeuron(
      X_OUT, Y_OUT, "y",
      f.y,
      f.z2,
      "y",
      true, // label + z below — keeps the space above y clear for the b2 chart
    ))

    return out
  }

  // All 9 learnable parameters get an in-situ mini-chart. Layout:
  //   - W1 charts: vertically stacked column at x=245 (diagonals would
  //     otherwise overlap at the edge crossing).
  //   - b1 charts: between h1 and h2 at equal thirds.
  //   - W2 charts: at the hidden→output edge midpoints.
  //   - b2 chart: above the output neuron.
  paramCharts() {
    const out: any[] = []
    const W1_X = 245
    const w1Stack: { j: 0|1, i: 0|1, cy: number }[] = [
      { j: 0, i: 0, cy:  42 },
      { j: 1, i: 0, cy: 102 },
      { j: 0, i: 1, cy: 162 },
      { j: 1, i: 1, cy: 222 },
    ]
    for (const p of w1Stack) {
      out.push(this.inlineChart({ kind: "W1", j: p.j, i: p.i }, W1_X, p.cy, `W1.${p.j}.${p.i}`))
    }
    const b1Step = (Y_BOT - Y_TOP) / 3
    out.push(this.inlineChart({ kind: "b1", j: 0 }, X_HID, Y_TOP + b1Step,                   "b1.0"))
    out.push(this.inlineChart({ kind: "b1", j: 1 }, X_HID, Y_BOT - b1Step,                   "b1.1"))
    out.push(this.inlineChart({ kind: "W2", j: 0 }, (X_HID + X_OUT) / 2, (Y_TOP + Y_OUT) / 2, "W2.0"))
    out.push(this.inlineChart({ kind: "W2", j: 1 }, (X_HID + X_OUT) / 2, (Y_BOT + Y_OUT) / 2, "W2.1"))
    out.push(this.inlineChart({ kind: "b2"        }, X_OUT, Y_OUT - 58,                       "b2"))
    return out
  }

  // The actual mini-chart for one parameter. Sweeps the parameter through a
  // range (everything else fixed), plots loss as a curve, marks the current
  // value, and overlays a tangent whose slope IS ∂. Self-contained with
  // overlaid name + value + ∂ + Δ.
  inlineChart(sel: Selection, cx: number, cy: number, key: string) {
    const d = this.demo
    const acc = paramAccess(d, sel)
    const cur = acc.read()
    const ex = EXAMPLES[d.exIdx]
    const curLoss = forward(d.p, ex.x, ex.y, d.hiddenAct).loss
    const spread = Math.max(1.0, 2 * Math.abs(cur))
    const N = 24
    const pts: { v: number, loss: number }[] = []
    for (let i = 0; i <= N; i++) {
      const v = cur - spread + (2 * spread * i / N)
      acc.write(v)
      pts.push({ v, loss: forward(d.p, ex.x, ex.y, d.hiddenAct).loss })
    }
    acc.write(cur)

    const grad        = (d.phase === "backward" || d.phase === "updated") ? acc.grad : null
    const delta       = d.phase === "updated" ? acc.delta : null
    // Tangent shows whenever a gradient is available — same condition as
    // ∂. On `updated` the slope is one step stale (∂ was computed at the
    // pre-update position), but the line is drawn through (cur, curLoss),
    // so the dot still sits on it — only the slope direction is nudged.
    const tangentGrad = grad

    const W = 108, H = 46
    const PLOT_LEFT = 4, PLOT_RIGHT = W - 4
    const PLOT_TOP = 13, PLOT_BOT = H - 13
    const xMin = pts[0].v, xMax = pts[pts.length - 1].v
    const yMax = Math.max(0.001, ...pts.map(p => p.loss))
    const sx = (v: number) => PLOT_LEFT + ((v - xMin) / (xMax - xMin)) * (PLOT_RIGHT - PLOT_LEFT)
    const sy = (l: number) => PLOT_BOT - (l / yMax) * (PLOT_BOT - PLOT_TOP)

    const pathD = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.v).toFixed(1)} ${sy(p.loss).toFixed(1)}`).join(' ')
    const cxPt = sx(cur), cyPt = sy(curLoss)

    const tDx = (xMax - xMin) * 0.22
    const tangentEl = tangentGrad !== null ? line({
      x1: sx(cur - tDx), y1: sy(curLoss - tangentGrad * tDx),
      x2: sx(cur + tDx), y2: sy(curLoss + tangentGrad * tDx),
      stroke: "var(--grad-color)",
      strokeWidth: "1.5",
      strokeLineCap: "round",
    }) : null

    const isSel = this.root.inspected === key
    return g({
      transform: `translate(${(cx - W / 2).toFixed(1)}, ${(cy - H / 2).toFixed(1)})`,
      style: { cursor: "pointer" },
      onClick: () => this.root.inspect(key),
    },
      rect({
        x: 0, y: 0, width: W, height: H,
        fill: "var(--chart-tint)",
        stroke: isSel ? "var(--accent)" : "var(--border)",
        strokeWidth: isSel ? "1.5" : "0.6",
        rx: "3",
      }),
      text({ x: 4,     y: 9, class: "il-name" }, acc.name),
      text({ x: W - 4, y: 9, textAnchor: "end", class: "il-val" }, fmt(cur, 2)),
      line({ x1: PLOT_LEFT, y1: PLOT_BOT, x2: PLOT_RIGHT, y2: PLOT_BOT, stroke: "var(--border)", strokeWidth: "0.4" }),
      path({ d: pathD, stroke: "var(--text-muted)", strokeWidth: "1", fill: "none" }),
      tangentEl,
      circle({ cx: cxPt, cy: cyPt, r: "1.8", fill: "var(--accent)" }),
      grad  !== null ? text({ x: 4,     y: H - 4, class: "il-grad"  }, `∂${fmtSign(grad,  2)}`) : null,
      delta !== null ? text({ x: W - 4, y: H - 4, textAnchor: "end", class: "il-delta" }, `Δ${fmtSign(delta, 3)}`) : null,
    )
  }
}

class Root extends Component implements IRoot {
  demo = new Demo()
  diagram = new Diagram()
  runTimer: number | null = null
  activeTab: "demo" | "code" = "demo"
  inspected: string | null = null

  // Advance by one of the currently-selected StepUnit. Used by both the
  // Advance button and the Run autoplay loop, so "advance once" and "play"
  // share the same granularity setting.
  advanceOnce() {
    switch (this.demo.stepUnit) {
      case "phase": this.demo.step(); return
      case "step":  this.demo.fullStep(); return
      case "epoch": this.demo.cycleStep(); return
    }
  }
  step()          { this.advanceOnce(); this.update() }
  stopRun()       { if (this.runTimer !== null) { clearInterval(this.runTimer); this.runTimer = null } }
  reset()         { this.stopRun(); this.demo.reset(); this.update() }
  reseed()        { this.stopRun(); this.demo.reseed(); this.update() }
  setOptimizer(o: "sgd" | "adam") { this.stopRun(); this.demo.optimizer = o; this.demo.reset(); this.inspected = null; this.update() }
  setHiddenAct(a: HiddenAct)      { this.stopRun(); this.demo.hiddenAct = a; this.demo.reset(); this.inspected = null; this.update() }
  setStepUnit(u: StepUnit)        { this.demo.stepUnit = u; this.update() }
  setLr(lr: number) { this.demo.lr = lr; this.update() }
  inspect(key: string | null) {
    // Same key = deselect; otherwise select. Lets callers just pass their key.
    this.inspected = (key !== null && this.inspected === key) ? null : key
    this.update()
  }

  get running() { return this.runTimer !== null }

  toggleRun() {
    if (this.running) {
      this.stopRun()
    } else {
      this.runTimer = window.setInterval(() => {
        this.advanceOnce()
        this.update()
      }, 80)
    }
    this.update()
  }

  view() {
    const d = this.demo
    const info = this.phaseInfo()
    const showDemo = this.activeTab === "demo"
    const showCode = this.activeTab === "code"
    return div({ class: "app" },
      div({ class: "header" },
        h1("XOR X-ray"),
        p({}, "A 2-2-1 neural network learns XOR. Step through the full training loop: forward arithmetic, chain-rule backward, optimizer update."),
      ),

      div({ class: "tabs" },
        this.tabBtn("demo", "Step through"),
        this.tabBtn("code", "In tensorgrad"),
      ),

      div({ class: "tab-content", style: { display: showDemo ? "block" : "none" } },
        div({ class: "controls" },
          div({ class: "btn-group" },
            button({ onClick: () => this.toggleRun(), class: ["icon", "accent"] }, this.running ? iconPause() : iconPlay()),
            button({ onClick: () => this.step(), class: ["icon", "accent"], title: "Advance" }, iconStep()),
            button({ onClick: () => this.reset(),     class: "icon"             }, "↺"),
            button({ onClick: () => this.reseed()                               }, "New seed"),
          ),

          this.toggleGroup<StepUnit>("Advance by",
            [{ value: "phase", label: "phase" }, { value: "step", label: "step" }, { value: "epoch", label: "epoch" }],
            d.stepUnit, u => this.setStepUnit(u)),

          this.toggleGroup<HiddenAct>("Activation",
            [{ value: "tanh", label: "tanh" }, { value: "relu", label: "ReLU" }],
            d.hiddenAct, a => this.setHiddenAct(a)),

          this.toggleGroup<"sgd" | "adam">("Optimizer",
            [{ value: "sgd", label: "SGD" }, { value: "adam", label: "Adam" }],
            d.optimizer, o => this.setOptimizer(o)),

          this.toggleGroup<number>("LR",
            [0.01, 0.1, 0.5].map(lr => ({ value: lr, label: String(lr) })),
            d.lr, lr => this.setLr(lr)),

          div({ class: "stat" }, `step ${d.trainStep}`),
        ),

        div({ class: ["panel", "phase-banner", `phase-${d.phase}`] },
          div({ class: "phase-title" }, info.title),
          div({ class: "phase-desc"  }, info.desc),
        ),

        div({ class: ["panel", "diagram-panel"] },
          this.diagram.view(),
          this.inspector(),
        ),

        div({ class: "subpanel-row" },
          this.predictionsPanel(),
          this.lossChart(),
        ),
      ),

      div({ class: "tab-content", style: { display: showCode ? "block" : "none" } },
        this.handoffPanel(),
      ),
    )
  }

  tabBtn(key: "demo" | "code", label: string) {
    return button({
      class: ["tab-btn", this.activeTab === key && "active"],
      onClick: () => {
        if (this.activeTab === key) return
        this.activeTab = key
        this.update()
      },
    }, label)
  }

  toggleGroup<T>(label: string, choices: { value: T, label: string }[], current: T, onSelect: (v: T) => void) {
    return div({ class: "toggle-group" },
      div({ class: "toggle-label" }, label),
      choices.map(c => button({
        onClick: () => onSelect(c.value),
        class: ["toggle", current === c.value && "active"],
      }, c.label)),
    )
  }

  inspector() {
    const sel = this.inspected
    if (!sel) {
      return div({ class: "inspector inspector-empty" },
        "Click any neuron or weight in the diagram to see its math here.",
      )
    }
    return div({ class: "inspector" },
      div({ class: "inspector-head" },
        div({ class: "inspector-label" }, this.inspectorTitle(sel)),
        button({ class: "inspector-close", onClick: () => this.inspect(null) }, "✕"),
      ),
      div({ class: "inspector-body" }, this.inspectorBody(sel)),
    )
  }

  inspectorTitle(sel: string): string {
    const p = parseSel(sel)
    switch (p.kind) {
      case "x":  return `Input x${p.i + 1}`
      case "h":  return `Hidden neuron h${p.j + 1}`
      case "y":  return "Output neuron y"
      case "b1": return `Hidden bias b1[${p.j}]`
      case "b2": return "Output bias b2"
      case "W1": return `Weight W1[${p.j}][${p.i}] — input x${p.i + 1} → hidden h${p.j + 1}`
      case "W2": return `Weight W2[${p.j}] — hidden h${p.j + 1} → output y`
    }
  }

  // Phase-accumulating: forward shows the element's activation; backward adds
  // its gradient; updated adds the optimizer step. By 'updated' the user has
  // seen the full chain for this element.
  inspectorBody(sel: string) {
    const d = this.demo
    const f = d.fwd
    const b = d.bwd
    const deltas = d.deltas
    const lines: string[] = []
    const p = parseSel(sel)

    switch (p.kind) {
      case "x": {
        lines.push(`x${p.i + 1} = ${fmt(f.x[p.i])}   (this step's input)`)
        break
      }
      case "h": {
        const j = p.j
        lines.push(`h${j + 1} = ${d.hiddenAct}(W1[${j}]·x + b1[${j}])`)
        lines.push(`   = ${d.hiddenAct}(${fmt(d.p.W1[j][0])}·${fmt(f.x[0])} + ${fmt(d.p.W1[j][1])}·${fmt(f.x[1])} + ${fmt(d.p.b1[j])})`)
        lines.push(`   = ${d.hiddenAct}(${fmt(f.z1[j])})  =  ${fmt(f.h[j], 3)}`)
        if (b) {
          lines.push("")
          const actDeriv = d.hiddenAct === "tanh" ? `(1 − h²) = ${fmt(1 - f.h[j] ** 2, 3)}` : `(z > 0) = ${f.z1[j] > 0 ? 1 : 0}`
          lines.push(`∂h${j + 1} = ∂z2 · W2[${j}]  =  ${fmt(b.dz2, 3)} · ${fmt(d.p.W2[j])}  =  ${fmt(b.dh[j], 3)}`)
          lines.push(`∂z${j + 1} = ∂h${j + 1} · ${actDeriv}  =  ${fmt(b.dz1[j], 3)}`)
        }
        break
      }
      case "y": {
        lines.push(`y = tanh(W2·h + b2)`)
        lines.push(`  = tanh(${fmt(d.p.W2[0])}·${fmt(f.h[0])} + ${fmt(d.p.W2[1])}·${fmt(f.h[1])} + ${fmt(d.p.b2)})`)
        lines.push(`  = tanh(${fmt(f.z2)})  =  ${fmt(f.y, 3)}`)
        lines.push("")
        lines.push(`loss = (y − target)² = (${fmt(f.y, 3)} − ${f.target})² = ${fmt(f.loss, 4)}`)
        if (b) {
          lines.push("")
          lines.push(`∂y  = 2·(y − target)  =  2·(${fmt(f.y, 3)} − ${f.target})  =  ${fmt(b.dy, 3)}`)
          lines.push(`∂z2 = ∂y · (1 − y²)  =  ${fmt(b.dy, 3)} · ${fmt(1 - f.y ** 2, 3)}  =  ${fmt(b.dz2, 3)}`)
        }
        break
      }
      case "W1": {
        const { j, i } = p
        lines.push(`current value: ${fmt(d.p.W1[j][i], 3)}`)
        lines.push(`forward contribution: w·x${i + 1} = ${fmt(d.p.W1[j][i])} · ${fmt(f.x[i])} = ${fmt(d.p.W1[j][i] * f.x[i], 3)}`)
        if (b) {
          lines.push("")
          lines.push(`∂W1[${j}][${i}] = ∂z${j + 1} · x${i + 1}  =  ${fmt(b.dz1[j], 3)} · ${fmt(f.x[i])}  =  ${fmt(b.dW1[j][i], 3)}`)
        }
        if (deltas) this.appendOptimizerLines(lines, b!.dW1[j][i], deltas.dW1[j][i], d.s.mW1[j][i], d.s.vW1[j][i])
        break
      }
      case "W2": {
        const j = p.j
        lines.push(`current value: ${fmt(d.p.W2[j], 3)}`)
        lines.push(`forward contribution: w·h${j + 1} = ${fmt(d.p.W2[j])} · ${fmt(f.h[j])} = ${fmt(d.p.W2[j] * f.h[j], 3)}`)
        if (b) {
          lines.push("")
          lines.push(`∂W2[${j}] = ∂z2 · h${j + 1}  =  ${fmt(b.dz2, 3)} · ${fmt(f.h[j])}  =  ${fmt(b.dW2[j], 3)}`)
        }
        if (deltas) this.appendOptimizerLines(lines, b!.dW2[j], deltas.dW2[j], d.s.mW2[j], d.s.vW2[j])
        break
      }
      case "b1": {
        const j = p.j
        lines.push(`current value: ${fmt(d.p.b1[j], 3)}`)
        if (b) {
          lines.push("")
          lines.push(`∂b1[${j}] = ∂z${j + 1}  =  ${fmt(b.dz1[j], 3)}    (bias gradient = pre-activation gradient)`)
        }
        if (deltas) this.appendOptimizerLines(lines, b!.db1[j], deltas.db1[j], d.s.mb1[j], d.s.vb1[j])
        break
      }
      case "b2": {
        lines.push(`current value: ${fmt(d.p.b2, 3)}`)
        if (b) {
          lines.push("")
          lines.push(`∂b2 = ∂z2  =  ${fmt(b.dz2, 3)}`)
        }
        if (deltas) this.appendOptimizerLines(lines, b!.db2, deltas.db2, d.s.mb2, d.s.vb2)
        break
      }
    }

    const showChartNote = p.kind === "W1" || p.kind === "W2" || p.kind === "b1" || p.kind === "b2"

    return [
      ...lines.map(l => div({ class: l === "" ? "inspector-gap" : "inspector-line" }, l)),
      showChartNote ? div({ class: "inspector-note" },
        "The optimizer here merely sees the orange tangent line for this parameter, generated by backprop. Purely for you, the reader, we create a mini-chart computing what the loss would be for nearby parameter values, if all other parameters were held constant. This would be far too expensive in a real model, and furthermore, the 1-dimensional axis here is just one of n parameter axes that together define the real loss function.",
      ) : null,
    ]
  }

  // The optimizer math is the same shape for every parameter — factor it
  // here so each parameter's inspector body just delegates.
  appendOptimizerLines(lines: string[], grad: number, delta: number, m: number, v: number) {
    const d = this.demo
    lines.push("")
    if (d.optimizer === "sgd") {
      lines.push(`Δ = −lr · ∂  =  −${d.lr} · ${fmt(grad, 3)}  =  ${fmt(delta, 4)}    (current shown above is post-update)`)
      return
    }
    const c1 = 1 - Math.pow(0.9,   d.t)
    const c2 = 1 - Math.pow(0.999, d.t)
    lines.push(`Adam (t=${d.t}, β₁=0.9, β₂=0.999):`)
    lines.push(`  m̂ = m / (1 − 0.9^t)   =  ${fmt(m, 4)} / ${fmt(c1, 3)}  =  ${fmt(m / c1, 4)}`)
    lines.push(`  v̂ = v / (1 − 0.999^t) =  ${fmt(v, 5)} / ${fmt(c2, 5)}  =  ${fmt(v / c2, 5)}`)
    lines.push(`  Δ = −lr · m̂ / √v̂      =  ${fmt(delta, 4)}    (current shown above is post-update)`)
  }

  phaseInfo(): { title: string; desc: string } {
    const d = this.demo
    switch (d.phase) {
      case "idle":
        return {
          title: "Ready",
          desc:  'Tiny network: 2 inputs, 2 hidden neurons (tanh), 1 output (tanh). Random weights. Click "Advance" to start training on the four XOR examples, one at a time.',
        }
      case "forward": {
        const opSentence = d.hiddenAct === "tanh"
          ? "Each neuron multiplies its incoming values by the weight of their connections, adds its bias (learned value above each circle), then passes the result through tanh, squashing it to between −1 and 1."
          : "Each neuron multiplies its incoming values by the weight of their connections, adds its bias (learned value above each circle). Hidden neurons then pass the result through ReLU (negative becomes 0, positive passes through); the output uses tanh, squashing to between −1 and 1."
        return {
          title: "Forward done",
          desc:  `Walked the network forward. ${opSentence} The result appears inside. We compare the output to the target (the answer we want); loss = (output − target)².`,
        }
      }
      case "backward": {
        const actDeriv = d.hiddenAct === "tanh"
          ? "With tanh, the hidden derivative is 1 − h² (smooth, always nonzero)."
          : "With ReLU, the hidden derivative is a 0/1 mask — ∂z is zero at any 'dead' neuron (z ≤ 0)."
        return {
          title: "Backward done",
          desc:  `Walked the network in reverse via the chain rule. ∂z (say "partial z") at each neuron and ∂ on each edge is the local gradient — how much the loss changes per unit change there. ${actDeriv}`,
        }
      }
      case "updated": {
        const opt = d.optimizer === "adam" ? "Adam" : "SGD"
        const desc = d.optimizer === "adam"
          ? "Walked the network in reverse again (in practice this step can interleave with backprop). As each weight was traversed, it moved by −lr · m̂ / (√v̂ + ε), where m̂ and v̂ are running averages of the gradient and its square (one m, one v per parameter — that's the 'adaptive per-parameter' part). After a few steps the Δ values stop being proportional to ∂. Toggle to SGD to see what proportional-to-∂ updates look like."
          : "Walked the network in reverse again (in practice this step can interleave with backprop). As each weight was traversed, it moved by −lr · ∂ — every edge gets the same lr factor, so Δ is directly proportional to gradient magnitude. Toggle to Adam to see per-parameter adaptive step sizes instead."
        return {
          title: `${opt} update done`,
          desc,
        }
      }
    }
  }

  predictionsPanel() {
    const d = this.demo
    return div({ class: ["panel", "subpanel"] },
      h3("Network output on all 4 XOR inputs"),
      div({ class: "pred-table" },
        EXAMPLES.map((ex, i) => {
          const f = forward(d.p, ex.x, ex.y, d.hiddenAct)
          const correct = Math.abs(f.y - ex.y) < 0.3
          const isCurrent = i === d.exIdx
          return div({ class: ["pred-row", correct ? "correct" : "incorrect", isCurrent && "current"] },
            span({ class: "ex-dot", style: { background: `var(--ex-color-${i})` } }),
            span({ class: "pred-in"    }, `(${ex.x[0]}, ${ex.x[1]})`),
            span({ class: "pred-arrow" }, "→"),
            span({ class: "pred-out"   }, fmt(f.y, 3)),
            span({ class: "pred-tgt"   }, `(want ${ex.y})`),
            span({ class: "pred-mark"  }, correct ? "✓" : "·"),
          )
        }),
      ),
      div({ class: "pred-note" }, "The outlined row is the example currently being trained on. All four rows are re-evaluated after every optimizer step — even rows we're not training on shift, because the weights are shared across all inputs. The network has learned XOR when all four are ✓."),
    )
  }

  lossChart() {
    const d = this.demo
    const hist = d.lossHistory
    const maxL = d.lossMax > 0 ? d.lossMax : null
    return div({ class: ["panel", "subpanel"] },
      h3({},
        `Loss per step (last ${hist.length})`,
        span({ class: "chart-current" }, ` · loss ${fmt(d.fwd.loss, 4)}`),
        maxL !== null ? span({ class: "chart-current" }, ` · max ${fmt(maxL, 3)}`) : null,
      ),
      div({ class: "loss-legend" },
        EXAMPLES.map((ex, i) => span({ class: "ex-item" },
          span({ class: "ex-swatch", style: { background: `var(--ex-color-${i})` } }),
          `(${ex.x[0]}, ${ex.x[1]})`,
        )),
      ),
      hist.length < 1
        ? div({ class: "chart-empty" }, "(no training steps yet)")
        : this.renderLossChart(hist, maxL ?? 0.01),
    )
  }

  // Bars not a line: each step is a discrete event measuring one specific
  // example — no meaningful interpolation between samples.
  renderLossChart(hist: { loss: number, exIdx: number }[], maxL: number) {
    const w = 320, ht = 100
    const innerHt = ht - 6
    const barW = w / hist.length
    const bars = hist.map((h, i) => {
      const barHt = (h.loss / maxL) * innerHt
      return rect({
        x: i * barW,
        y: ht - barHt - 3,
        width: Math.max(barW, 0.5),
        height: barHt,
        fill: `var(--ex-color-${h.exIdx})`,
      })
    })
    return svg({ viewBox: `0 0 ${w} ${ht}`, width: "100%", preserveAspectRatio: "none", class: "loss-chart" },
      bars,
    )
  }

  handoffPanel() {
    return div({ class: ["panel", "handoff"] },
      h2("The same network and training, in tensorgrad"),
      p({}, "This is how to write the same logic in tensorgrad. It's a tiny library that compiles neural networks to WebGPU, good for in-browser demos and visualisations."),
      p({}, "Visualize ", a({ href: 'https://tinyurl.com/mtca47db', target: '_blank' }, 'this network as a dataflow diagram'), ", drawn by nn-dna, a tool that turns plain-English descriptions of neural networks into architecture diagrams."),
      pre({ class: "code" }, this.handoffCode()),
    )
  }

  handoffCode() {
    return `import {
  Module, compile, Linear, tanh, sub, square, mean,
  type Tensor,
} from 'tensorgrad'

class XORNet extends Module {
  l1 = new Linear(2, 2)   // input  → hidden
  l2 = new Linear(2, 1)   // hidden → output
}

function netFwd(m: XORNet, x: Tensor): Tensor {
  return tanh(m.l2.fwd(tanh(m.l1.fwd(x))))
}

function lossFn(m: XORNet, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  return mean(square(sub(netFwd(m, x), y)))
}

const train = await compile({
  model: new XORNet(),
  loss: lossFn,
  inputs: { x: [1, 2], y: [1, 1] },           // batch size 1
  optimizer: { kind: 'adam', lr: 0.1 },
})

const examples = [
  { x: [0, 0], y: [0] },
  { x: [0, 1], y: [1] },
  { x: [1, 0], y: [1] },
  { x: [1, 1], y: [0] },
]

for (let step = 0; step < 200; step++) {
  const ex = examples[step % 4]
  await train.step({
    x: new Float32Array(ex.x),
    y: new Float32Array(ex.y),
  })
}`
  }
}

new App({ root: new Root(), id: "app" })
```

**styles.css**

```css
/* Theme tokens — light defaults on :root; dark overrides on
   html[data-theme="dark"]. The host doesn't always set the attribute,
   so :root must be a fully usable light theme on its own. */
:root {
  color-scheme: light;
  --bg-page:         #ffffff;
  --bg-panel:        #ffffff;
  --bg-canvas:       #fafbfc;
  --bg-subpanel:     #f6f7f9;
  --text-primary:    #1a1a1a;
  --text-muted:      #666666;
  --text-on-accent:  #ffffff;
  --border:          #dddddd;
  --border-strong:   #888888;
  --accent:          #6366f1;
  --accent-hover:    #4f46e5;
  --neuron-bg:       #ffffff;
  --chart-tint:      #fff4d6;
  --grad-color:      #d97706;
  --w-pos:           rgb(34, 139, 34);
  --w-neg:           rgb(220, 38, 38);
  --phase-forward:   #2563eb;
  --phase-backward:  #dc2626;
  --phase-optimizer: #16a34a;
  /* Per-example identity colors. Tableau10 picks that avoid the correct/
     incorrect green/red and the indigo accent. */
  --ex-color-0:      #4e79a7;
  --ex-color-1:      #e377c2;
  --ex-color-2:      #76b7b2;
  --ex-color-3:      #9c755f;
  --font-mono:       ui-monospace, Menlo, monospace;
}
html[data-theme="dark"] {
  color-scheme: dark;
  --bg-page:         #0c0c0e;
  --bg-panel:        #18181b;
  --bg-canvas:       #0c0c0e;
  --bg-subpanel:     #1a1a1d;
  --text-primary:    #e5e5e5;
  --text-muted:      #9ca3af;
  --text-on-accent:  #ffffff;
  --border:          #2e2e34;
  --border-strong:   #555560;
  --accent:          #818cf8;
  --accent-hover:    #6366f1;
  --neuron-bg:       #1f1f24;
  --chart-tint:      #1f1d16;
  --grad-color:      #fb923c;
  --w-pos:           rgb(74, 222, 128);
  --w-neg:           rgb(248, 113, 113);
  --phase-forward:   #60a5fa;
  --phase-backward:  #f87171;
  --phase-optimizer: #4ade80;
  --ex-color-0:      #6f9bd1;
  --ex-color-1:      #ee9bd1;
  --ex-color-2:      #a5d4d0;
  --ex-color-3:      #c89e88;
}

body {
  font-family: ui-sans-serif, system-ui, sans-serif;
  margin: 0;
  padding: 0 1rem 2rem;
  background: var(--bg-page);
  color: var(--text-primary);
  -webkit-font-smoothing: antialiased;
}

.app { max-width: 1180px; margin: 0 auto; }

.tabs {
  display: flex;
  gap: 1.2rem;
  margin: 0 0 1rem;
  padding: 0.4rem 0 0;
  border-bottom: 1px solid var(--border);
}
.tab-btn {
  padding: 0.4rem 0.1rem 0.35rem;
  margin-bottom: -1px;
  border: none;
  border-bottom: 2px solid transparent;
  background: transparent;
  font: inherit;
  font-size: 0.9rem;
  cursor: pointer;
  color: var(--text-muted);
  border-radius: 0;
  transition: color 0.15s, border-color 0.15s;
}
.tab-btn:hover { color: var(--text-primary); }
.tab-btn.active { color: var(--text-primary); border-bottom-color: var(--accent); font-weight: 600; }

.header { margin: 1.2rem 0 0.4rem; }
.header h1 { font-size: 1.4rem; margin: 0; font-weight: 600; }
.header p {
  color: var(--text-muted);
  margin: 0.4rem 0 0;
  line-height: 1.55;
  font-size: 0.9rem;
}

/* Base card — composed with phase-banner / subpanel / diagram-panel / handoff. */
.panel {
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-panel);
}

.diagram-panel {
  overflow: hidden;
  background: var(--bg-canvas);
  /* Joins seamlessly with the phase-banner above: square top, rounded
     bottom, no top border (banner's bottom edge is the seam, supplied by
     the bg-panel → bg-canvas tone shift). */
  border-top: 0;
  border-radius: 0 0 8px 8px;
}

.inspector {
  border-top: 1px solid var(--border);
  background: var(--bg-subpanel);
  padding: 0.7rem 1rem;
}
.inspector-empty {
  color: var(--text-muted);
  font-size: 0.85rem;
  font-style: italic;
  text-align: center;
}
.inspector-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}
.inspector-label {
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--text-primary);
}
.inspector-close {
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.95rem;
  padding: 0 0.25rem;
  line-height: 1;
}
.inspector-close:hover { color: var(--text-primary); }
.inspector-body {
  font-family: var(--font-mono);
  font-size: 0.9rem;
  color: var(--text-primary);
  white-space: pre-wrap;
  overflow-x: auto;
}
.inspector-line { padding: 0.05rem 0; }
.inspector-gap  { height: 0.4rem; }
.inspector-note {
  margin-top: 0.7rem;
  padding-top: 0.55rem;
  border-top: 1px dashed var(--border);
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 0.95rem;
  color: var(--text-muted);
  line-height: 1.5;
  white-space: normal;
}
.inspector-note-title {
  font-weight: 600;
  color: var(--text-primary);
}

/* Text styles for the per-parameter mini-charts. */
.diagram-svg .il-name  { font-family: var(--font-mono); font-size: 8px; font-weight: 600; fill: var(--text-primary); }
.diagram-svg .il-val   { font-family: var(--font-mono); font-size: 8px; fill: var(--text-primary); }
.diagram-svg .il-grad  { font-family: var(--font-mono); font-size: 8px; fill: var(--grad-color); }
.diagram-svg .il-delta { font-family: var(--font-mono); font-size: 8px; fill: var(--phase-optimizer); }

.diagram-svg {
  display: block;
  width: 100%;
  padding: 0.25rem 0;
  box-sizing: border-box;
}

.phase-banner {
  margin-bottom: 0;
  padding: 0.85rem 1rem;
  /* Square bottom: joins seamlessly into the diagram-panel below. */
  border-bottom: 0;
  border-radius: 8px 8px 0 0;
  /* Reserved height so the panel doesn't bob between phases. Sized for the
     longest case: forward desc with ReLU selected wraps to 3 lines because
     it has to spell out the asymmetry (hidden uses ReLU, output stays tanh
     — bounded range needed for MSE against {0,1} targets). Px (not rem)
     because some host pages override root font-size, which would balloon
     this reserve out of proportion to the actual text. */
  min-height: 90px;
}

.phase-title {
  font-weight: 600;
  font-size: 0.9rem;
  margin-bottom: 0.3rem;
  text-decoration-line: underline;
  text-decoration-thickness: 3px;
  text-underline-offset: 4px;
  transition: text-decoration-color 0.15s;
}
/* Phase indicator: title underlined in the phase color. Longhand props
   throughout — the `text-decoration` shorthand resets text-decoration-thickness. */
.phase-banner.phase-idle     .phase-title { text-decoration-color: var(--border-strong); }
.phase-banner.phase-forward  .phase-title { text-decoration-color: var(--phase-forward); }
.phase-banner.phase-backward .phase-title { text-decoration-color: var(--phase-backward); }
.phase-banner.phase-updated  .phase-title { text-decoration-color: var(--phase-optimizer); }
.phase-desc {
  color: var(--text-muted);
  font-size: 0.92rem;
  line-height: 1.5;
}

.diagram-svg .n-val   { font-family: var(--font-mono); font-size: 14px; font-weight: 600; fill: var(--text-primary); }
.diagram-svg .n-label { font-size: 12px; fill: var(--text-muted); }
.diagram-svg .n-sub   { font-family: var(--font-mono); font-size: 10px; fill: var(--text-muted); }

.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
  margin-bottom: 0.8rem;
}
.controls button {
  padding: 0.4rem 0.85rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-panel);
  color: var(--text-primary);
  font: inherit;
  font-size: 0.85rem;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
.controls button:hover { border-color: var(--border-strong); }
.btn-group { display: flex; gap: 0.5rem; }
/* Icon-only buttons: fixed width and flex-centered for predictable SVG
   placement. `.accent` paints the icon in the brand color (SVG uses
   currentColor). */
.controls button.icon {
  width: 2.4rem;
  padding-left: 0;
  padding-right: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  line-height: 1;
}
.controls button.icon.accent { color: var(--accent); }
.btn-icon { display: block; }
.controls button.toggle.active {
  background: color-mix(in srgb, var(--accent) 12%, transparent);
  color: var(--accent);
  border-color: var(--accent);
}
.toggle-group {
  display: flex;
  align-items: center;
  margin-left: 0.6rem;
}
.toggle-label {
  font-size: 0.78rem;
  color: var(--text-muted);
  margin-right: 0.4rem;
}
/* Segmented control: only outer corners rounded; adjacent borders overlap
   into one seam via -1px margin; active button's border wins via z-index. */
.toggle-group .toggle {
  position: relative;
  border-radius: 0;
}
.toggle-group .toggle:first-of-type {
  border-top-left-radius: 6px;
  border-bottom-left-radius: 6px;
}
.toggle-group .toggle:last-of-type {
  border-top-right-radius: 6px;
  border-bottom-right-radius: 6px;
}
.toggle-group .toggle + .toggle { margin-left: -1px; }
.toggle-group .toggle.active { z-index: 1; }
.stat {
  margin-left: auto;
  font-size: 0.85rem;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

.subpanel-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-top: 1.2rem;
}
@media (max-width: 760px) {
  .subpanel-row { grid-template-columns: 1fr; }
}

.subpanel {
  padding: 0.85rem 1rem;
  display: flex;
  flex-direction: column;
}
.loss-chart {
  display: block;
  flex: 1;
  min-height: 120px;
}
.subpanel h3 {
  font-size: 0.9rem;
  font-weight: 600;
  margin: 0 0 0.6rem;
  color: var(--text-primary);
}

.pred-table {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-family: var(--font-mono);
  font-size: 0.85rem;
}
.pred-row {
  display: grid;
  grid-template-columns: 0.7rem 4rem 1.2rem 4rem 5rem 1rem;
  gap: 0.4rem;
  align-items: center;
  padding: 0.2rem 0.45rem;
  border-radius: 4px;
}
.chart-current {
  font-family: var(--font-mono);
  font-weight: normal;
  font-size: 0.82rem;
  color: var(--text-muted);
}
.pred-row.correct   { background: rgba(34, 139, 34, 0.12); }
.pred-row.incorrect { background: rgba(220, 38, 38, 0.05); }
/* The example currently being trained on. Outline (not border) so it doesn't
   shift layout as the highlight moves between rows. */
.pred-row.current   { outline: 2px solid var(--accent); outline-offset: -2px; }
.pred-in    { color: var(--text-primary); }
.pred-arrow { color: var(--text-muted); }
.pred-out   { color: var(--text-primary); }
.pred-tgt   { color: var(--text-muted); }
.pred-row.correct   .pred-mark { color: var(--w-pos); font-weight: bold; }
.pred-row.incorrect .pred-mark { color: var(--text-muted); }

.pred-note {
  margin-top: 0.7rem;
  font-size: 0.95rem;
  color: var(--text-muted);
  line-height: 1.5;
}

.chart-empty {
  font-size: 0.85rem;
  color: var(--text-muted);
  padding: 1.5rem 0;
  text-align: center;
}

.loss-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin: -0.2rem 0 0.5rem;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--text-muted);
}
.loss-legend .ex-item { display: inline-flex; align-items: center; gap: 0.3rem; }
.ex-swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
}
.ex-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.handoff {
  padding: 0.85rem 1rem;
}
.handoff > :first-child { margin-top: 0; }
.handoff > :last-child  { margin-bottom: 0; }
.handoff h2 {
  font-size: 1.05rem;
  font-weight: 600;
  margin: 0 0 0.8rem;
}
.handoff p {
  color: var(--text-primary);
  line-height: 1.55;
  margin: 0.7rem 0;
  font-size: 0.9rem;
}
.handoff .code {
  background: var(--bg-subpanel);
  padding: 1rem;
  border-radius: 6px;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  line-height: 1.5;
  overflow-x: auto;
  border: 1px solid var(--border);
  margin: 0.6rem 0 0;
  white-space: pre;
}
```

**index.html**

```html
<div id="app"></div>
```

**config.json**

```json
{
  "description": "A 2-2-1 neural network learns XOR. Step through the full training loop: forward arithmetic, chain-rule backward, optimizer update.",
  "dependencies": {
    "domeleon": "^0.6.0"
  }
}
```