---
format: typebulb/v1
name: Chameleon
---

**code.tsx**

```tsx
import { App, Component, a, button, canvas, div, h1, img, p, path, polygon, rect, span, svg } from "domeleon"
import {
  Module, compileForward, checkWebGPU, Conv2d, Linear, init,
  conv2d, maxPool2d, relu, sin, greater, where, mul, add,
  narrow, concat, reshape, permute, randn, ones, zeros,
  type Tensor, type CompiledForward,
} from "tensorgrad"

// ============================================================================
// One organism from "From Cells to Pixels" (Pajouheshgar et al., SIGGRAPH
// 2026), running the authors' own trained weights — zero training, zero local
// files. A neural cellular automaton evolves a 96×96 cell grid, and a
// jointly-trained SIREN field decodes each cell's 32-dim state + sub-cell
// coordinates into 384×384 RGBA: the chameleon. Weights (~280KB, base64
// float32 in JSON) and the target image are fetched from our asset host.
// Scratch it and it regrows. The Menagerie bulb is the 24-organism version.
// ============================================================================

const C = 32, FC = 256, H = 64, OMEGA = 10 // channels, update-MLP width, SIREN width/freq (paper config)
const G = 96                               // cell grid — the paper's training lattice
const S = 4, GH = G * S                    // render scale, render res (384)
const GG = G * G, STATE_LEN = C * GG, GGH = GH * GH
const SIN = 4 + C                          // SIREN input: 4 trig coord features + C state channels
const DAMAGE_R = 7                         // scratch radius in cells

const ASSETS = "https://assets.typebulb.com/regrow"
const MODEL_URL = `${ASSETS}/chameleon.json`
const TARGET_URL = `${ASSETS}/chameleon.png`

const nextFrame = () => new Promise(res => requestAnimationFrame(res))
const clamp01 = (v: number) => v < 0 ? 0 : v > 1 ? 1 : v
const slog = (s: string) => { if (tb.mode === "local") tb.log(s) }

// Fetch once from the network, thereafter from CacheStorage (offline-friendly);
// falls back to a plain fetch if the Cache API is unavailable.
async function cachedFetch(url: string): Promise<ArrayBuffer> {
  const download = async (): Promise<ArrayBuffer> => {
    const res = await fetch(url)
    if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`)
    return res.arrayBuffer()
  }
  try {
    const cache = await caches.open("tensorgrad-assets")
    const hit = await cache.match(url)
    if (hit) return hit.arrayBuffer()
    const buf = await download()
    try { await cache.put(url, new Response(buf)) } catch { /* caching is an optimization */ }
    return buf
  } catch {
    return download()
  }
}

// ============================================================================
// The model — same fields (and so same tensorgrad param names) as the other
// Cells-to-Pixels bulbs. Init is irrelevant: the imported weights are
// uploaded before any run.
// ============================================================================
class NCAModel extends Module {
  w1 = new Conv2d(4 * C, FC, 1, { init: init.zeros() })
  w2 = new Conv2d(FC, C, 1, { bias: false, init: init.zeros() })
  s1 = new Linear(SIN, H, { init: init.zeros() })
  s2 = new Linear(H, H, { init: init.zeros() })
  s3 = new Linear(H, H, { init: init.zeros() })
  sOut = new Linear(H, 4, { init: init.zeros() })
}

// ============================================================================
// Weight importer — the authors' JSON layout → tensorgrad's param record.
//   nca.w1.weight [FC, 4C]: input columns are FEATURE-major (k·C + c, k over
//     identity/sobelX/sobelY/laplacian — the layout their GLSL shader wants);
//     our grouped perception conv emits CHANNEL-major (c·4 + k, the torch
//     training layout) — permute the columns back.
//   nca.w2.weight.T [FC, C]: shipped pre-transposed for the shader; our
//     Conv2d weight is [C, FC] — transpose.
//   lppn.net.*: torch Linear [out, in]; tensorgrad Linear is [in, out] (x@W)
//     — transpose all four.
// ============================================================================
type TheirTensor = { shape: number[]; dtype: string; data64: string }

function importModel(src: Record<string, TheirTensor>): Record<string, Float32Array> {
  const dec = (key: string, ...shape: number[]) => {
    const e = src[key]
    if (!e) throw new Error(`checkpoint is missing ${key}`)
    if (e.shape.length !== shape.length || e.shape.some((v, i) => v !== shape[i]))
      throw new Error(`${key}: shape [${e.shape}] != expected [${shape}]`)
    const bin = atob(e.data64)
    const u8 = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i)
    return new Float32Array(u8.buffer)
  }
  const w1t = dec("nca.w1.weight", FC, 4 * C)
  const w1 = new Float32Array(FC * 4 * C)
  for (let o = 0; o < FC; o++)
    for (let k = 0; k < 4; k++)
      for (let c = 0; c < C; c++)
        w1[o * 4 * C + c * 4 + k] = w1t[o * 4 * C + k * C + c]!
  const w2t = dec("nca.w2.weight.T", FC, C)
  const w2 = new Float32Array(C * FC)
  for (let h = 0; h < FC; h++)
    for (let c = 0; c < C; c++)
      w2[c * FC + h] = w2t[h * C + c]!
  const lin = (key: string, nin: number, nout: number) => {
    const t = dec(key, nout, nin)
    const w = new Float32Array(nin * nout)
    for (let o = 0; o < nout; o++)
      for (let i = 0; i < nin; i++)
        w[i * nout + o] = t[o * nin + i]!
    return w
  }
  return {
    "w1.W": w1, "w1.b": dec("nca.w1.bias", FC), "w2.W": w2,
    "s1.W": lin("lppn.net.0.linear.weight", SIN, H), "s1.b": dec("lppn.net.0.linear.bias", H),
    "s2.W": lin("lppn.net.1.linear.weight", H, H), "s2.b": dec("lppn.net.1.linear.bias", H),
    "s3.W": lin("lppn.net.2.linear.weight", H, H), "s3.b": dec("lppn.net.2.linear.bias", H),
    "sOut.W": lin("lppn.net.3.weight", H, 4), "sOut.b": dec("lppn.net.3.bias", 4),
  }
}

// ============================================================================
// Graphs — the CA step and SIREN decode, verbatim from the reference:
// circular perception padding, Bernoulli-0.5 update mask, pre×post alive
// gating (the torch training semantics).
// ============================================================================
const alive = (s: Tensor) =>
  greater(maxPool2d(narrow(s, 1, 3, 1), 3, { stride: 1, padding: 1 }), 0.1)

const boolToF32 = (m: Tensor, g: number) =>
  where(m, ones([1, 1, g, g]), zeros([1, 1, g, g]))

function circularPad1(s: Tensor): Tensor {
  const px = concat([narrow(s, 3, G - 1, 1), s, narrow(s, 3, 0, 1)], 3)
  return concat([narrow(px, 2, G - 1, 1), px, narrow(px, 2, 0, 1)], 2)
}

function ncaStep(m: NCAModel, s: Tensor, filters: Tensor): Tensor {
  const pre = alive(s)
  const z = conv2d(circularPad1(s), filters, { padding: 0, groups: C })   // [1, 4C, G, G]
  const ds = m.w2.fwd(relu(m.w1.fwd(z)))
  const mask = greater(randn([1, 1, G, G]), 0)          // Bernoulli(0.5)
  const s2 = where(mask, add(s, ds), s)
  const post = alive(s2)
  return mul(s2, mul(boolToF32(pre, G), boolToF32(post, G)))
}

// exact bilinear ×S upsample of `ch` channels via a depthwise tent conv +
// depth-to-space (ch=1 reuses it for the alpha)
function bilinearUp(x: Tensor, w: Tensor, ch: number): Tensor {
  const shifted = conv2d(x, w, { padding: 1, groups: ch })   // [1, ch·S², G, G]
  const t = reshape(shifted, [1, ch, S, S, G, G])
  return reshape(permute(t, [0, 1, 4, 2, 5, 3]), [1, ch, GH, GH])
}

// the Cells2Pixels decoder, alive-gated the AUTHORS' renderer's way: the alpha
// channel is bilinearly upsampled to render res and maxpooled over 3×3 RENDER
// pixels (a smooth, sub-cell-accurate silhouette).
function decode(m: NCAModel, s: Tensor, bilin: Tensor, bilin1: Tensor, coords: Tensor): Tensor {
  const up = bilinearUp(s, bilin, C)
  const feats = concat([coords, up], 1)                          // [1, 4+C, GH, GH]
  const rows = reshape(permute(feats, [0, 2, 3, 1]), [GGH, SIN])
  let h = sin(mul(m.s1.fwd(rows), OMEGA))
  h = sin(mul(m.s2.fwd(h), OMEGA))
  h = sin(mul(m.s3.fwd(h), OMEGA))
  const img = permute(reshape(m.sOut.fwd(h), [1, GH, GH, 4]), [0, 3, 1, 2])
  const aUp = bilinearUp(narrow(s, 1, 3, 1), bilin1, 1)          // [1,1,GH,GH]
  const mask = greater(maxPool2d(aUp, 3, { stride: 1, padding: 1 }), 0.1)
  return mul(img, boolToF32(mask, GH))
}

function demoFn(m: NCAModel, { state, filters }: { state: Tensor; filters: Tensor }) {
  return ncaStep(m, state, filters)
}

function renderFn(m: NCAModel, { state, bilin, bilin1, coords }: { state: Tensor; bilin: Tensor; bilin1: Tensor; coords: Tensor }) {
  return decode(m, state, bilin, bilin1, coords)
}

// ============================================================================
// Host-side constants: perception filters, bilinear tent tables, coord
// features, seed.
// ============================================================================
const F3 = [
  [0, 0, 0, 0, 1, 0, 0, 0, 0],                          // identity
  [-1, 0, 1, -2, 0, 2, -1, 0, 1],                       // sobel_x
  [-1, -2, -1, 0, 0, 0, 1, 2, 1],                       // sobel_y
  [1, 2, 1, 2, -12, 2, 1, 2, 1],                        // laplacian
]
const FILTERS = (() => {
  const a = new Float32Array(4 * C * 9)                 // [4C, 1, 3, 3], channel-major (c·4+k)
  for (let k = 0; k < 4; k++)
    for (let c = 0; c < C; c++)
      a.set(F3[k]!, (c * 4 + k) * 9)
  return a
})()

const W1D = [
  [0.375, 0.625, 0],
  [0.125, 0.875, 0],
  [0, 0.875, 0.125],
  [0, 0.625, 0.375],
]
const makeBilin = (ch: number) => {
  const a = new Float32Array(ch * S * S * 9)            // [ch·S², 1, 3, 3]
  for (let c = 0; c < ch; c++)
    for (let dy = 0; dy < S; dy++)
      for (let dx = 0; dx < S; dx++) {
        const off = (c * S * S + dy * S + dx) * 9
        for (let ky = 0; ky < 3; ky++)
          for (let kx = 0; kx < 3; kx++)
            a[off + ky * 3 + kx] = W1D[dy]![ky]! * W1D[dx]![kx]!
      }
  return a
}
const BILIN = makeBilin(C)
const BILIN1 = makeBilin(1)

// sub-cell coord features [sin(πy), sin(πx), cos(πy), cos(πx)] per render
// pixel — the reference's num_frequencies=1 encoding, periodic per 4×4 tile
const COORDS1 = (() => {
  const a = new Float32Array(4 * GGH)
  const f = [-0.75, -0.25, 0.25, 0.75]
  for (let y = 0; y < GH; y++)
    for (let x = 0; x < GH; x++) {
      const i = y * GH + x
      const cy = f[y % S]! * Math.PI, cx = f[x % S]! * Math.PI
      a[i] = Math.sin(cy)
      a[GGH + i] = Math.sin(cx)
      a[2 * GGH + i] = Math.cos(cy)
      a[3 * GGH + i] = Math.cos(cx)
    }
  return a
})()

// the paper's seed: all zeros except channels 3..C = 1 at the grid center
const SEED = (() => {
  const a = new Float32Array(STATE_LEN)
  const x = G >> 1, y = G >> 1
  for (let c = 3; c < C; c++) a[c * GG + y * G + x] = 1
  return a
})()

function carveDisc(s: Float32Array, cx: number, cy: number, r: number) {
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(G - 1, Math.ceil(cy + r)); y++)
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(G - 1, Math.ceil(cx + r)); x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 > r * r) continue
      for (let c = 0; c < C; c++) s[c * GG + y * G + x] = 0
    }
}

const aliveCount = (s: Float32Array) => {
  let n = 0
  for (let i = 0; i < GG; i++) if (s[3 * GG + i]! > 0.1) n++
  return n
}

// ============================================================================
// Root — the whole app in one component: the two compiled forwards, the
// imported weights, the frame loop, and the view. No training machinery
// anywhere.
// ============================================================================
const icon = (...kids: any[]) => svg({ class: "ico", viewBox: "0 0 24 24", fill: "currentColor" }, ...kids)
const ICON = {
  play: () => icon(polygon({ points: "8,5 8,19 19,12" })),
  pause: () => icon(rect({ x: 7, y: 5, width: 3.5, height: 14, rx: 1 }), rect({ x: 13.5, y: 5, width: 3.5, height: 14, rx: 1 })),
  refresh: () => icon(path({ d: "M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" })),
}

class Root extends Component {
  status = "starting…"
  paused = false
  #step: CompiledForward | null = null
  #render: CompiledForward | null = null
  #state: Float32Array | null = null
  #token = 0
  #canvas: HTMLCanvasElement | null = null
  #ctx: CanvasRenderingContext2D | null = null
  #img: ImageData | null = null
  #cellsCtx: CanvasRenderingContext2D | null = null
  #cellsImg: ImageData | null = null
  #painting = false
  #pendingCarves: [number, number][] = []     // carves racing an in-flight step
  #renderBusy = false

  boot(cv: HTMLCanvasElement) {
    if (this.#canvas) return
    cv.width = cv.height = GH
    this.#canvas = cv
    this.#ctx = cv.getContext("2d")!
    this.#img = new ImageData(GH, GH)
    cv.addEventListener("pointerdown", e => {
      this.#painting = true
      cv.setPointerCapture(e.pointerId)
      this.#damageAt(e)
    })
    cv.addEventListener("pointermove", e => { if (this.#painting) this.#damageAt(e) })
    cv.addEventListener("pointerup", () => { this.#painting = false })
    this.run()
  }

  bootCells(cv: HTMLCanvasElement) {
    if (this.#cellsCtx) return
    cv.width = cv.height = G
    this.#cellsCtx = cv.getContext("2d")!
    this.#cellsImg = new ImageData(G, G)
  }

  async run() {
    const gpu = await checkWebGPU()
    if (!gpu.ok) {
      this.#setStatus(gpu.message)
      return
    }
    this.#setStatus("compiling WGSL kernels…")
    this.#step = await compileForward({
      model: new NCAModel(), forward: demoFn,
      inputs: { state: [1, C, G, G], filters: [4 * C, 1, 3, 3] },
    })
    this.#render = await compileForward({
      model: new NCAModel(), forward: renderFn,
      inputs: { state: [1, C, G, G], bilin: [C * S * S, 1, 3, 3], bilin1: [S * S, 1, 3, 3], coords: [1, 4, GH, GH] },
    })
    this.#setStatus("fetching the authors' trained chameleon…")
    try {
      const buf = await cachedFetch(MODEL_URL)
      const src = JSON.parse(new TextDecoder().decode(buf)) as Record<string, TheirTensor>
      const params = importModel(src)
      await this.#step.uploadParams(params)
      await this.#render.uploadParams(params)
      this.reseed()
      slog(`loaded chameleon (${Object.keys(params).length} tensors)`)
      this.#setStatus("growing the chameleon — scratch it!")
    } catch (e) {
      slog(`load failed: ${(e as Error).message}`)
      this.#setStatus(`couldn't load the chameleon (${(e as Error).message})`)
      return
    }
    this.#loop()
  }

  reseed() {
    this.#state = SEED.slice()
    this.#pendingCarves.length = 0
    if (this.paused) this.togglePause()
  }

  togglePause() {
    this.paused = !this.paused
    this.update()
  }

  async #loop() {
    const token = ++this.#token
    let frames = 0
    while (token === this.#token) {
      if (!this.paused && this.#state) {
        this.#pendingCarves.length = 0                   // from here, carves race the step
        const r = await this.#step!.run({ state: this.#state, filters: FILTERS })
        if (token !== this.#token) return
        if (r.kind === "completed") {
          const next = r.output as Float32Array
          for (const [cx, cy] of this.#pendingCarves) carveDisc(next, cx, cy, DAMAGE_R)
          this.#pendingCarves.length = 0
          this.#state = this.#stabilize(next) ? SEED.slice() : next
        }
        const rr = await this.#render!.run({ state: this.#state!, bilin: BILIN, bilin1: BILIN1, coords: COORDS1 })
        if (token !== this.#token) return
        if (rr.kind === "completed") {
          this.#paintRender(rr.output as Float32Array)
          this.#paintCells(this.#state!)
        }
        frames++
        if (frames % 300 === 0) slog(`t=${frames} alive=${aliveCount(this.#state!)}`)
      }
      await nextFrame()
    }
  }

  // demo-time stabilizers: clamp the state into [-1,1] — the box the
  // training-time overflow penalty (weight 100) kept it in, so this is a
  // near-no-op while healthy — and reseed on death or runaway growth.
  #stabilize(s: Float32Array): boolean {
    for (let i = 0; i < STATE_LEN; i++) {
      const v = s[i]!
      if (v > 1) s[i] = 1
      else if (v < -1) s[i] = -1
    }
    const n = aliveCount(s)
    return n === 0 || n > GG * 0.6
  }

  #damageAt(e: PointerEvent) {
    if (!this.#state) return
    const rect = this.#canvas!.getBoundingClientRect()
    const x = Math.floor((e.clientX - rect.left) / rect.width * G)
    const y = Math.floor((e.clientY - rect.top) / rect.height * G)
    carveDisc(this.#state, x, y, DAMAGE_R)
    this.#pendingCarves.push([x, y])                     // re-applied if a step was in flight
    this.#paintCells(this.#state)
    this.#renderDamaged()
  }

  // paused pokes still repaint: one render pass on demand
  #renderDamaged() {
    if (this.#renderBusy || !this.#render || !this.#state) return
    this.#renderBusy = true
    this.#render.run({ state: this.#state, bilin: BILIN, bilin1: BILIN1, coords: COORDS1 }).then(r => {
      this.#renderBusy = false
      if (r.kind === "completed") this.#paintRender(r.output as Float32Array)
    })
  }

  #paintRender(arr: Float32Array) { if (this.#ctx) this.#blit(arr, this.#img!, this.#ctx, GGH) }
  #paintCells(arr: Float32Array) { if (this.#cellsCtx) this.#blit(arr, this.#cellsImg!, this.#cellsCtx, GG) }

  // premultiplied CHW RGBA → ImageData: un-premultiply RGB with capped gain,
  // straight alpha, composites onto the themed page background
  #blit(arr: Float32Array, img: ImageData, ctx: CanvasRenderingContext2D, n: number) {
    const d = img.data
    for (let i = 0; i < n; i++) {
      const a = clamp01(arr[3 * n + i]!)
      const ia = a > 0.01 ? 1 / Math.max(a, 0.2) : 0
      d[i * 4] = clamp01(arr[i]! * ia) * 255
      d[i * 4 + 1] = clamp01(arr[n + i]! * ia) * 255
      d[i * 4 + 2] = clamp01(arr[2 * n + i]! * ia) * 255
      d[i * 4 + 3] = a * 255
    }
    ctx.putImageData(img, 0, 0)
  }

  #setStatus(s: string) { this.status = s; this.update() }

  // terminal test hooks (`typebulb send`)
  testPoke() {
    if (!this.#state) { slog("poke ignored: no state"); return }
    const before = aliveCount(this.#state)
    carveDisc(this.#state, G >> 1, G >> 1, DAMAGE_R)
    this.#pendingCarves.push([G >> 1, G >> 1])
    slog(`poked center: alive ${before} -> ${aliveCount(this.#state)}`)
  }

  testStats() {
    slog(`alive=${this.#state ? aliveCount(this.#state) : -1} paused=${this.paused}`)
  }

  view() {
    return div({ class: "wrap" },
      div({ class: "header" },
        h1({ class: "title" }, "Regrow the Chameleon"),
        p({ class: "cap" },
          "Every cell runs the same tiny neural network, and from a single seed they cooperate to grow the ",
          "chameleon, then regrow it when you scratch it. The twist: the cells never hold color, they hold a ",
          "latent code that a shared decoder paints into a far higher-resolution image. It runs live on your ",
          "GPU with the weights and methods ",
          a({ class: "link", href: "https://cells2pixels.github.io/", target: "_blank", rel: "noopener" }, "“From Cells to Pixels”"),
          ".",
        ),
      ),
      div({ class: "stage-box" },
        canvas({ class: "grid", key: "grid", onMounted: (el: Element) => this.boot(el as HTMLCanvasElement) }),
        div({ class: "strip" },
          div({ class: "thumb-box" },
            canvas({ class: "thumb cells", key: "cells", onMounted: (el: Element) => this.bootCells(el as HTMLCanvasElement) }),
            span({ class: "thumb-label" }, `cells ${G}²`),
          ),
          span({ class: "strip-arrow" }, "→"),
          div({ class: "thumb-box" },
            img({ class: "thumb", key: "target", src: TARGET_URL, alt: "chameleon target" }),
            span({ class: "thumb-label" }, "target"),
          ),
        ),
      ),
      div({ class: "dock" },
        div({ class: "bar" },
          button(
            { class: "icon-btn", title: this.paused ? "play" : "pause", onClick: () => this.togglePause() },
            this.paused ? ICON.play() : ICON.pause(),
          ),
          button({ class: "icon-btn", title: "regrow from a single seed cell", onClick: () => this.reseed() }, ICON.refresh()),
        ),
        div({ class: "readout" }, this.status),
      ),
    )
  }
}

const root = new Root()
new App({ root, id: "app" })
tb.onMessage((m: unknown) => {
  if (m === "poke") root.testPoke()
  if (m === "stats") root.testStats()
})
```

**index.html**

```html
<div id="app"></div>
```

**styles.css**

```css
/* Theme tokens — same scheme as the other tensorgrad bulbs. The render canvas
   is transparent, so the organism composites straight onto --bg. */
:root {
  color-scheme: light;
  --font: ui-sans-serif, system-ui, sans-serif;
  --font-mono: ui-monospace, Menlo, Consolas, monospace;
  --text: 0.9rem;
  --accent: #ff7a18;
  --bg: #eef1f6;
  --fg: #1a1d24;
  --muted: #5a6273;
  --readout: #3a4250;
  --panel: #ffffffcc;
  --panel-border: #cdd3de;
  --btn-bg: #ffffff;
  --btn-fg-hover: #11141c;
  --btn-border-hover: #aab2c0;
}
html[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0b0d12;
  --fg: #e7e9ee;
  --muted: #aeb6c4;
  --readout: #cdd3dd;
  --panel: #11141ccc;
  --panel-border: #2a2f3a;
  --btn-bg: #1b1f2a;
  --btn-fg-hover: #ffffff;
  --btn-border-hover: #3a4250;
}
html { font-size: 106.25%; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg); font-family: var(--font); }
body { font-size: var(--text); }
#app { height: 100dvh; min-height: 560px; }
.wrap { height: 100%; display: flex; flex-direction: column; --gap: 12px; }
.header, .dock { display: flex; flex-direction: column; align-items: center; gap: var(--gap); padding: var(--gap) 18px; text-align: center; }
.header > *, .dock > * { margin: 0; }
.title { font-size: 1.5rem; font-weight: 680; letter-spacing: .04em; text-transform: uppercase; color: var(--fg); }
.cap { max-width: 720px; line-height: 1.5; color: var(--muted); }
.cap .link { color: var(--accent); text-decoration: none; border-bottom: 1px solid color-mix(in srgb, var(--accent) 45%, transparent); }
.cap .link:hover { border-bottom-color: var(--accent); }
.stage-box { flex: 1 1 auto; min-height: 220px; display: flex; align-items: center; justify-content: center; gap: 18px; padding: 4px 18px; }
.grid {
  flex: 0 1 auto;
  min-height: 120px;
  height: min(64vmin, 560px);
  max-height: 100%;
  aspect-ratio: 1;
  touch-action: none;
  cursor: crosshair;
}
.strip { flex: none; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; }
.strip-arrow { color: var(--muted); font-size: 1.05rem; transform: rotate(90deg); }
.thumb-box { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.thumb { width: 84px; aspect-ratio: 1; border: 1px solid var(--panel-border); border-radius: 6px; object-fit: contain; }
.thumb.cells { image-rendering: pixelated; }
/* the hosted png is the tight 512² target; the paper centers it in a 768²
   frame (the 96² lattice), so pad 1/6 per side to match the cells' framing */
img.thumb { box-sizing: border-box; padding: 16.667%; }
.thumb-label { font-family: var(--font-mono); font-size: 0.72rem; color: var(--muted); }
.dock { padding-bottom: calc(var(--gap) + 4px); }
.bar { display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 8px; }
.icon-btn { cursor: pointer; box-sizing: border-box; display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; color: var(--readout); background: var(--btn-bg); border: 1px solid var(--panel-border); border-radius: 50%; }
.icon-btn:hover { color: var(--btn-fg-hover); border-color: var(--btn-border-hover); }
.icon-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.ico { width: 1.45em; height: 1.45em; flex: none; }
.readout { font-family: var(--font-mono); color: var(--readout); user-select: text; cursor: text; }
@media (max-width: 620px) {
  /* min-height: auto keeps the flex min-content floor, so a squeezed stage-box
     scrolls instead of letting the strip overflow into the dock */
  .stage-box { flex-direction: column; gap: 10px; min-height: auto; }
  .strip { flex-direction: row; gap: 8px; }
  .strip-arrow { transform: none; margin-bottom: 14px; }
  .thumb { width: min(84px, 22vw); }
}
@media (max-height: 640px) {
  .wrap { --gap: 8px; }
  .cap { display: none; }
  .title { font-size: 1.3rem; }
}
```

**config.json**

```json
{
  "description": "A neural cellular automaton grows a chameleon from one cell on your GPU, using weights from the paper “From Cells to Pixels”. Scratch it and it heals.",
  "dependencies": {
    "tensorgrad": "^0.4.6",
    "domeleon": "^0.6.3"
  }
}
```
