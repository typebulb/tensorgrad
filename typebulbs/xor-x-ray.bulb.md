---
format: typebulb/v1
name: "XOR X-ray"
---

**code.tsx**

```tsx
import { App, Component, div, h1, h2, h3, p, pre, span, strong, button, svg, g, circle, line, path, rect, text } from "domeleon"

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

class Demo {
  p: Params
  s: AdamState
  exIdx = 0
  phase: Phase = "idle"
  optimizer: "sgd" | "adam" = "adam"
  hiddenAct: HiddenAct = "tanh"
  lr = 0.1
  t = 0                  // Adam bias-correction counter (only increments on Adam updates, so it's distinct from trainStep)
  trainStep = 0
  seed = 7
  fwd: Forward  | null = null
  bwd: Backward | null = null
  deltas: Deltas | null = null
  lossHistory: { loss: number, exIdx: number }[] = []

  constructor() {
    this.p = initParams(this.seed)
    this.s = initAdam()
  }

  reset() {
    this.p = initParams(this.seed)
    this.s = initAdam()
    this.exIdx = 0
    this.phase = "idle"
    this.t = 0
    this.trainStep = 0
    this.fwd = null; this.bwd = null; this.deltas = null
    this.lossHistory = []
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
      this.bwd = backward(this.p, this.fwd!, this.hiddenAct)
      this.phase = "backward"
    } else if (this.phase === "backward") {
      if (this.optimizer === "adam") {
        this.t++
        this.deltas = applyAdam(this.p, this.bwd!, this.s, this.t, this.lr)
      } else {
        this.deltas = applySGD(this.p, this.bwd!, this.lr)
      }
      this.lossHistory.push({ loss: this.fwd!.loss, exIdx: this.exIdx })
      if (this.lossHistory.length > 200) this.lossHistory.shift()
      this.trainStep++
      this.phase = "updated"
    }
  }

  fullStep() {
    do { this.step() } while (this.phase !== "updated")
  }
}

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
const Y_TOP = 90, Y_BOT = 230, Y_OUT = 160
const R = 24
const yFor = (j: number) => j === 0 ? Y_TOP : Y_BOT

interface IRoot { demo: Demo }

class Diagram extends Component {
  get root() { return this.ctx.root as any as IRoot }
  get demo() { return this.root.demo }

  view() {
    return svg({
      viewBox: "0 30 820 260",
      width: "100%",
      preserveAspectRatio: "xMidYMid meet",
      class: "diagram-svg",
    },
      this.edges(),
      this.neurons(),
    )
  }

  edges() {
    const d = this.demo
    const out: any[] = []
    // Input → hidden. Label fractions offset crossing diagonals from sharing
    // a midpoint label position.
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 2; i++) {
        const labelT = i === j ? 0.5 : (i === 0 ? 0.35 : 0.65)
        out.push(this.edge(
          X_IN, yFor(i), X_HID, yFor(j),
          d.p.W1[j][i],
          d.bwd?.dW1[j][i] ?? null,
          d.deltas?.dW1[j][i] ?? null,
          labelT,
        ))
      }
    }
    for (let j = 0; j < 2; j++) {
      out.push(this.edge(
        X_HID, yFor(j), X_OUT, Y_OUT,
        d.p.W2[j],
        d.bwd?.dW2[j] ?? null,
        d.deltas?.dW2[j] ?? null,
        0.5,
      ))
    }
    return out
  }

  edge(x1: number, ya: number, x2: number, yb: number, w: number,
       grad: number | null, delta: number | null, labelT: number) {
    const phase = this.demo.phase
    const thickness = Math.min(0.8 + Math.abs(w) * 2.5, 5)
    const stroke = w >= 0 ? "var(--w-pos)" : "var(--w-neg)"
    const lx = x1 + labelT * (x2 - x1)
    const ly = ya + labelT * (yb - ya)
    const showGrad  = (phase === "backward" || phase === "updated") && grad !== null
    const showDelta = phase === "updated" && delta !== null
    return g(
      line({ x1, y1: ya, x2, y2: yb, stroke, strokeWidth: String(thickness), strokeOpacity: "0.65" }),
      text({ x: lx, y: ly - 6,  textAnchor: "middle", class: "w-label" }, fmt(w, 2)),
      showGrad  ? text({ x: lx, y: ly - 22, textAnchor: "middle", class: "grad-label" },  "∂ " + fmtSign(grad!, 3))   : null,
      showDelta ? text({ x: lx, y: ly + 14, textAnchor: "middle", class: "delta-label" }, "Δ " + fmtSign(delta!, 4)) : null,
    )
  }

  // Two-circle pattern: opaque base under the rgba tint, so edges passing
  // through the circle's footprint don't bleed through the semi-transparent fill.
  bodyNeuron(cx: number, cy: number, label: string, post: number | null, bias: number, pre: number | null, dzg: number | null) {
    return g(
      circle({ cx, cy, r: R, fill: "var(--neuron-bg)" }),
      circle({ cx, cy, r: R, fill: post !== null ? neuronFill(post) : "transparent", stroke: "var(--border-strong)", strokeWidth: "1.5" }),
      text({ x: cx, y: cy + 5,      textAnchor: "middle", class: "n-val" },   post !== null ? fmt(post, 2) : label),
      text({ x: cx, y: cy - R - 16, textAnchor: "middle", class: "n-label" }, label),
      text({ x: cx, y: cy - R - 4,  textAnchor: "middle", class: "n-sub" },   `b=${fmt(bias, 2)}`),
      pre !== null ? text({ x: cx, y: cy + R + 14, textAnchor: "middle", class: "n-sub"  }, `z=${fmt(pre, 2)}`)      : null,
      dzg !== null ? text({ x: cx, y: cy + R + 28, textAnchor: "middle", class: "n-grad" }, `∂z=${fmtSign(dzg, 3)}`) : null,
    )
  }

  neurons() {
    const d = this.demo
    const f = d.fwd
    const showFwd  = f !== null && d.phase !== "idle"
    const showGrad = (d.phase === "backward" || d.phase === "updated") && d.bwd !== null
    const out: any[] = []

    for (let i = 0; i < 2; i++) {
      const y = yFor(i)
      const val = showFwd ? f!.x[i] : null
      out.push(g(
        circle({ cx: X_IN, cy: y, r: R, fill: "var(--neuron-bg)", stroke: "var(--border-strong)", strokeWidth: "1.5" }),
        text({ x: X_IN, y: y + 5,     textAnchor: "middle", class: "n-val" },   val !== null ? fmt(val, 1) : `x${i+1}`),
        text({ x: X_IN, y: y - R - 8, textAnchor: "middle", class: "n-label" }, `x${i+1}`),
      ))
    }

    for (let j = 0; j < 2; j++) {
      out.push(this.bodyNeuron(
        X_HID, yFor(j), `h${j+1}`,
        showFwd  ? f!.h[j]       : null,
        d.p.b1[j],
        showFwd  ? f!.z1[j]      : null,
        showGrad ? d.bwd!.dz1[j] : null,
      ))
    }

    out.push(this.bodyNeuron(
      X_OUT, Y_OUT, "y",
      showFwd  ? f!.y    : null,
      d.p.b2,
      showFwd  ? f!.z2   : null,
      showGrad ? d.bwd!.dz2 : null,
    ))

    return out
  }
}

class Root extends Component implements IRoot {
  demo = new Demo()
  diagram = new Diagram()
  runTimer: number | null = null
  activeTab: "demo" | "code" = "demo"

  step()          { this.demo.step(); this.update() }
  stopRun()       { if (this.runTimer !== null) { clearInterval(this.runTimer); this.runTimer = null } }
  reset()         { this.stopRun(); this.demo.reset(); this.update() }
  reseed()        { this.stopRun(); this.demo.reseed(); this.update() }
  setOptimizer(o: "sgd" | "adam") { this.stopRun(); this.demo.optimizer = o; this.demo.reset(); this.update() }
  setHiddenAct(a: HiddenAct)      { this.stopRun(); this.demo.hiddenAct = a; this.demo.reset(); this.update() }
  setLr(lr: number) { this.demo.lr = lr; this.update() }

  get running() { return this.runTimer !== null }

  toggleRun() {
    if (this.running) {
      this.stopRun()
    } else {
      this.runTimer = window.setInterval(() => {
        this.demo.fullStep()
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
        p({}, "A 2-2-1 neural network learning XOR — one example at a time. Step through the full training loop: forward arithmetic, chain-rule backward, optimizer update."),
      ),

      div({ class: "tabs" },
        this.tabBtn("demo", "Step through"),
        this.tabBtn("code", "In tensorgrad"),
      ),

      div({ class: "tab-content", style: { display: showDemo ? "block" : "none" } },
        div({ class: "controls" },
          button({ onClick: () => this.step(),      class: "primary" }, info.btn),
          div({ class: "btn-group" },
            button({ onClick: () => this.toggleRun() }, this.running ? "⏸ Pause" : "▶ Run"),
            button({ onClick: () => this.reset()     }, "Reset"),
            button({ onClick: () => this.reseed()    }, "New seed"),
          ),

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

  phaseInfo(): { title: string; desc: string; btn: string } {
    const d = this.demo
    switch (d.phase) {
      case "idle":
        return {
          title: "Ready",
          desc:  'Tiny network: 2 inputs, 2 hidden neurons (tanh), 1 output (tanh). Random weights. Click "Step" to start training on the four XOR examples, one at a time.',
          btn:   "▷ Step: compute forward",
        }
      case "forward":
        return {
          title: "Forward done",
          desc:  "Computed hidden activations from inputs, then the output from hidden activations — each via tanh(W·x + b). The number inside each neuron is its activation. The target sits next to the output; loss = (output − target)².",
          btn:   "▷ Step: compute backward",
        }
      case "backward": {
        const actDeriv = d.hiddenAct === "tanh"
          ? "Hidden activation is tanh, so its derivative is 1 − h² — smooth, always nonzero."
          : "Hidden activation is ReLU, so its derivative is a 0/1 mask — ∂z at a hidden neuron is zero whenever its pre-activation z was negative (a 'dead' neuron, no gradient flows back)."
        return {
          title: "Backward done",
          desc:  `Walked the network in reverse via the chain rule. ∂z on each neuron and ∂ on each edge are the local gradient — how much the loss would change if that quantity changed by 1. ${actDeriv} Also: ∂ on an input→hidden edge is zero whenever its input is zero (no signal to flow back through).`,
          btn:   "▷ Step: apply optimizer",
        }
      }
      case "updated": {
        const opt = d.optimizer === "adam" ? "Adam" : "SGD"
        const desc = d.optimizer === "adam"
          ? "Each weight moved by −lr · m̂ / (√v̂ + ε), where m̂ and v̂ are running averages of the gradient and its square (one m, one v per parameter — that's the 'adaptive per-parameter' part). After a few steps the Δ values stop being proportional to ∂. Toggle to SGD to see what proportional-to-∂ updates look like."
          : "Each weight moved by −lr · ∂ — every edge gets the same lr factor, so Δ is directly proportional to gradient magnitude. Toggle to Adam to see per-parameter adaptive step sizes instead."
        return {
          title: `${opt} update done`,
          desc,
          btn:   "▷ Step: next example",
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
          const isCurrent = i === d.exIdx && d.fwd !== null
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
    const maxL = hist.length > 0 ? Math.max(...hist.map(h => h.loss), 0.01) : null
    return div({ class: ["panel", "subpanel"] },
      h3({},
        `Loss per step (last ${hist.length})`,
        d.fwd       ? span({ class: "chart-current" }, ` · loss ${fmt(d.fwd.loss, 4)}`) : null,
        maxL !== null ? span({ class: "chart-current" }, ` · max ${fmt(maxL, 3)}`)     : null,
      ),
      div({ class: "loss-legend" },
        EXAMPLES.map((ex, i) => span({ class: "ex-item" },
          span({ class: "ex-swatch", style: { background: `var(--ex-color-${i})` } }),
          `(${ex.x[0]}, ${ex.x[1]})`,
        )),
      ),
      maxL === null
        ? div({ class: "chart-empty" }, "(no training steps yet)")
        : this.renderLossChart(hist, maxL),
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
      p({}, "Everything above — forward arithmetic, chain-rule backward, Adam updates — is exactly what tensorgrad runs for any network, automatically. Here's this XOR training loop, rewritten with tensorgrad:"),
      pre({ class: "code" }, this.handoffCode()),
      p({}, "Two wins, and they're entangled — the abstraction is what gives you the performance."),
      p({}, strong("Abstraction"), ": the hand-coded scalar version above is ~120 lines of math (forward, backward, both optimizers, all the per-parameter Adam state). The tensorgrad version is ~30. You write the forward; the rest is derived."),
      p({}, strong("Performance"), ": that same ~30 lines compiles to GPU — which is why the Neural Network sample can learn to recognize handwritten digits from 60,000 examples in under a second on a modern GPU."),
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
  --w-pos:           rgb(34, 139, 34);
  --w-neg:           rgb(220, 38, 38);
  --phase-forward:   #2563eb;
  --phase-backward:  #dc2626;
  --phase-optimizer: #16a34a;
  /* Per-example identity colors. Tableau10 picks that avoid the correct/
     incorrect green/red and the indigo accent. */
  --ex-color-0:      #4e79a7;
  --ex-color-1:      #f28e2c;
  --ex-color-2:      #76b7b2;
  --ex-color-3:      #af7aa1;
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
  --w-pos:           rgb(74, 222, 128);
  --w-neg:           rgb(248, 113, 113);
  --phase-forward:   #60a5fa;
  --phase-backward:  #f87171;
  --phase-optimizer: #4ade80;
  --ex-color-0:      #6f9bd1;
  --ex-color-1:      #ffb24d;
  --ex-color-2:      #a5d4d0;
  --ex-color-3:      #ce99c6;
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

/* Sticky so the active tab stays visible while scrolling tall tab content. */
.tabs {
  display: flex;
  gap: 1.2rem;
  margin: 0 0 1rem;
  padding: 0.4rem 0 0;
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
  background: var(--bg-page);
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
}

.diagram-svg {
  display: block;
  width: 100%;
  padding: 0.25rem 0;
  box-sizing: border-box;
}

.phase-banner {
  margin-bottom: 0.8rem;
  padding: 0.85rem 1rem;
  border-left: 4px solid var(--border-strong);
  transition: border-left-color 0.15s;
}
.phase-banner.phase-idle     { border-left-color: var(--border-strong); }
.phase-banner.phase-forward  { border-left-color: var(--phase-forward); }
.phase-banner.phase-backward { border-left-color: var(--phase-backward); }
.phase-banner.phase-updated  { border-left-color: var(--phase-optimizer); }

.phase-title {
  font-weight: 600;
  font-size: 0.9rem;
  margin-bottom: 0.3rem;
}
.phase-desc {
  color: var(--text-muted);
  font-size: 0.85rem;
  line-height: 1.5;
}

.diagram-svg .n-val      { font-family: var(--font-mono); font-size: 14px; font-weight: 600; fill: var(--text-primary); }
.diagram-svg .n-label    { font-size: 12px; fill: var(--text-muted); }
.diagram-svg .n-sub      { font-family: var(--font-mono); font-size: 10px; fill: var(--text-muted); }
.diagram-svg .n-grad     { font-family: var(--font-mono); font-size: 11px; fill: var(--phase-backward); }
.diagram-svg .w-label    { font-family: var(--font-mono); font-size: 11px; fill: var(--text-primary); }
.diagram-svg .grad-label { font-family: var(--font-mono); font-size: 11px; fill: var(--phase-backward); }
.diagram-svg .delta-label{ font-family: var(--font-mono); font-size: 11px; fill: var(--phase-optimizer); }

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
.controls button.primary {
  background: var(--accent);
  color: var(--text-on-accent);
  border-color: var(--accent);
  /* Reserve space for the longest label variant ("▷ Step: compute backward")
     so phase changes don't push the rest of the strip sideways. */
  min-width: 14rem;
}
.controls button.primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
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
  font-size: 0.78rem;
  color: var(--text-muted);
  line-height: 1.4;
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
  margin: 0.6rem 0 1rem;
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
  "description": "Step through a tiny neural network learning XOR, one operation at a time — forward arithmetic, chain-rule backward, Adam optimizer update.",
  "dependencies": {
    "domeleon": "^0.6.0"
  }
}
```