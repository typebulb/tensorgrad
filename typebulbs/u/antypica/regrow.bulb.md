---
format: typebulb/v1
name: Regrow
---

**code.tsx**

```tsx
import { App, Component, a, button, canvas, div, h1, p, path, polygon, polyline, rect, span, svg } from "domeleon"
import {
  Module, compile, isWebGPUAvailable, Conv2d, Linear, init, lr, capture, loadSafetensors,
  conv2d, maxPool2d, relu, sin, greater, where, mul, add, sub, div as tdiv, mean, sum, abs, square, sqrt, clamp,
  matmul, narrow, concat, reshape, permute, randn, ones, zeros,
  type Tensor, type CompiledTraining, type CompiledForward, type LR,
} from "tensorgrad"

// ============================================================================
// "From Cells to Pixels" (cells2pixels.github.io) live in the browser on
// WebGPU: a Growing Neural Cellular Automaton grows a donut from one seed cell
// and regrows it when you bite it. Each cell holds a C-dim latent; a shared NCA
// update rule runs the G² lattice, and a SIREN decoder paints it up to a GH²
// RGBA image — the visible donut. The raw cell channels are latent, not color.
//
// Loss is the paper's stack: masked L2 + L1 on the render, a shape loss on the
// alpha/alive mask, LPIPS(VGG16) on a padding crop (random 3-of-4 channels),
// and a ×100 overflow penalty. Two non-obvious pieces:
//   • per-param gradient normalization on the NCA only (not the SIREN) is what
//     makes L1 safe on a CA (tensorgrad's `normalizeGrads`).
//   • LPIPS-VGG16 weights ride in as frozen INPUTS so DCE prunes their gradient
//     kernels; the forward is validated against torch to 8 decimals.
//
// Proof of concept, so it diverges from the paper for browser speed: mini scale
// (G=32/K=24/S=4, not G=96/8×) and a single 1500-step run (not 5×4000 rounds
// with pool flushes) — enough for a stable donut, not their fine detail. The
// donut's center is a HOLE, so the paper's seed-at-center breaks; placeSeed
// relocates the seed onto the ring.
// ============================================================================
const C = 32, FC = 256                     // cell channels, NCA hidden dim
const G = 32                               // cell lattice (paper: 96)
const B = 8                                // batch
const K = 24                               // CA steps unrolled per forward (≈ 2× the donut radius)
const S = 4, GH = G * S                    // render upscale, render resolution
const H = 64, OMEGA = 10                   // SIREN width, frequency
const PAD = 5                              // arena border in cells (room to grow)
const PADH = PAD * S                       // border in render px
const CROP = GH - 2 * PADH                 // LPIPS crop = the image region inside the border
const POOL_SIZE = 1024
const TRAIN_STEPS = 1500
const INJECT_EVERY = 32                    // re-seed batch slot 0 every N steps
const makeLR = () => lr.multiStep({ peak: 1e-3, milestones: [1000], gamma: 0.3 })
const GG = G * G, STATE_LEN = C * GG, GGH = GH * GH
const SIN = 4 + C                          // SIREN input: 4 sub-cell coords + C latent
const DAMAGE_R = 3                         // scratch radius in cells (display only, never trained)
const CIRCULAR = true                      // toroidal perception padding

const R2 = "https://assets.typebulb.com"               // hosted assets (published fallback)
const LPIPS_PATH = "weights/lpips-vgg16.safetensors"   // local repo copy (~59MB)
const LPIPS_URL = `${R2}/weights/lpips-vgg16.safetensors`
const TRAINED_DIR = "typebulbs-tentative/regrow"
const TRAINED_PATH = `${TRAINED_DIR}/donut.safetensors`
const TRAINED_URL = `${R2}/regrow/donut.safetensors`

const sleep = (ms: number) => new Promise(res => setTimeout(res, ms))
const nextFrame = () => new Promise(res => requestAnimationFrame(res))
const clamp01 = (v: number) => v < 0 ? 0 : v > 1 ? 1 : v
const slog = (s: string) => { if (tb.mode === "local") tb.log(s) }

// seeded uniform(-bound, bound) for the SIREN init (paper: uniform draws)
const uniformInit = (n: number, bound: number, seed: number) => {
  const a = new Float32Array(n)
  let h = (seed * 0x9e3779b9) >>> 0
  for (let i = 0; i < n; i++) {
    h = (h * 1664525 + 1013904223) >>> 0
    a[i] = (h / 4294967296 * 2 - 1) * bound
  }
  return a
}

// ============================================================================
// Model — the NCA update net (w1→relu→w2) + the SIREN decoder (s1..sOut).
// w1 small-random, w2 zeros, SIREN uniform init — the paper's scheme.
// ============================================================================
class NCAModel extends Module {
  w1 = new Conv2d(4 * C, FC, 1, { init: init.randn({ scale: 0.02 }) })
  w2 = new Conv2d(FC, C, 1, { bias: false, init: init.zeros() })
  s1 = new Linear(SIN, H, { init: init.literal(uniformInit(SIN * H, 1 / SIN, 1)) })
  s2 = new Linear(H, H, { init: init.literal(uniformInit(H * H, Math.sqrt(6 / H) / OMEGA, 2)) })
  s3 = new Linear(H, H, { init: init.literal(uniformInit(H * H, Math.sqrt(6 / H) / OMEGA, 3)) })
  sOut = new Linear(H, 4, { init: init.literal(uniformInit(H * 4, Math.sqrt(6 / H) / OMEGA, 4)) })
}
// per-param grad normalization applies to the NCA only — the paper's
// normalize_model_grads(model) touches the CA update rule, not the LPPN/SIREN
const NCA_PARAMS = new Set(["w1.W", "w1.b", "w2.W"])

// ============================================================================
// CA graphs: circular perception, Bernoulli-0.5 stochastic update, pre×post
// alive gating (torch-parity semantics).
// ============================================================================
const alive = (s: Tensor) =>
  greater(maxPool2d(narrow(s, 1, 3, 1), 3, { stride: 1, padding: 1 }), 0.1)

const boolToF32 = (m: Tensor, b: number, g: number) =>
  where(m, ones([b, 1, g, g]), zeros([b, 1, g, g]))

function circularPad1(s: Tensor): Tensor {
  const px = concat([narrow(s, 3, G - 1, 1), s, narrow(s, 3, 0, 1)], 3)
  return concat([narrow(px, 2, G - 1, 1), px, narrow(px, 2, 0, 1)], 2)
}

function ncaStep(m: NCAModel, s: Tensor, filters: Tensor, b: number): Tensor {
  const pre = alive(s)
  const z = CIRCULAR
    ? conv2d(circularPad1(s), filters, { padding: 0, groups: C })
    : conv2d(s, filters, { padding: 1, groups: C })
  const ds = m.w2.fwd(relu(m.w1.fwd(z)))
  const mask = greater(randn([b, 1, G, G]), 0)
  const s2 = where(mask, add(s, ds), s)
  const post = alive(s2)
  return mul(s2, mul(boolToF32(pre, b, G), boolToF32(post, b, G)))
}

// exact bilinear ×S upsample of `ch` channels (tent conv + depth-to-space)
function bilinearUp(x: Tensor, w: Tensor, ch: number, b: number): Tensor {
  const shifted = conv2d(x, w, { padding: 1, groups: ch })
  const t = reshape(shifted, [b, ch, S, S, G, G])
  return reshape(permute(t, [0, 1, 4, 2, 5, 3]), [b, ch, GH, GH])
}

// the paper's renderer: bilinear state + trig sub-cell coords → SIREN → RGBA,
// gated by the RENDER-RES living mask (bilinear alpha, 3×3 maxpool over
// render pixels, >0.1) — their get_living_mask(x_up). Also returns the raw
// bilinear alpha (their x_up[:,3:4]) for the shape loss.
function decode(m: NCAModel, s: Tensor, bilin: Tensor, bilin1: Tensor, coords: Tensor, b: number): { rendered: Tensor; aUp: Tensor } {
  const up = bilinearUp(s, bilin, C, b)
  const feats = concat([coords, up], 1)
  const rows = reshape(permute(feats, [0, 2, 3, 1]), [b * GGH, SIN])
  let h = sin(mul(m.s1.fwd(rows), OMEGA))
  h = sin(mul(m.s2.fwd(h), OMEGA))
  h = sin(mul(m.s3.fwd(h), OMEGA))
  const img = permute(reshape(m.sOut.fwd(h), [b, GH, GH, 4]), [0, 3, 1, 2])
  const aUp = bilinearUp(narrow(s, 1, 3, 1), bilin1, 1, b)
  const mask = greater(maxPool2d(aUp, 3, { stride: 1, padding: 1 }), 0.1)
  return { rendered: mul(img, where(mask, ones([b, 1, GH, GH]), zeros([b, 1, GH, GH]))), aUp }
}

// ============================================================================
// LPIPS(VGG16) in-graph — validated against torch to 8 decimals (see header).
// Weights ride in as frozen INPUTS (DCE prunes their gradient kernels); the
// input transform (2x−1 then ScalingLayer) is applied in-graph via lp_in_*
// (folding it into conv1_1 is inexact at zero-padded borders — measured 2.4%).
// ============================================================================
const VGG_PLAN: [string, number, boolean][] = [   // (name, outC, maxpool-before)
  ["conv1_1", 64, false], ["conv1_2", 64, false],
  ["conv2_1", 128, true], ["conv2_2", 128, false],
  ["conv3_1", 256, true], ["conv3_2", 256, false], ["conv3_3", 256, false],
  ["conv4_1", 512, true], ["conv4_2", 512, false], ["conv4_3", 512, false],
  ["conv5_1", 512, true], ["conv5_2", 512, false], ["conv5_3", 512, false],
]
const VGG_TAPS = new Set(["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3"])
const TAP_C = [64, 128, 256, 512, 512]

const LP_SPEC: Record<string, number[]> = { lp_in_gain: [1, 3, 1, 1], lp_in_offset: [1, 3, 1, 1] }
{
  let prevC = 3
  for (const [name, c] of VGG_PLAN) {
    LP_SPEC[`lp_${name}_W`] = [c, prevC, 3, 3]
    LP_SPEC[`lp_${name}_b`] = [1, c, 1, 1]
    prevC = c
  }
  for (let k = 0; k < 5; k++) LP_SPEC[`lp_lin${k}`] = [1, TAP_C[k]!, 1, 1]
}

function vggTaps(rgb01: Tensor, inp: Record<string, Tensor>): Tensor[] {
  let h = sub(mul(rgb01, inp["lp_in_gain"]!), inp["lp_in_offset"]!)
  const taps: Tensor[] = []
  for (const [name, , pool] of VGG_PLAN) {
    if (pool) h = maxPool2d(h, 2)
    h = relu(add(conv2d(h, inp[`lp_${name}_W`]!, { padding: 1 }), inp[`lp_${name}_b`]!))
    if (VGG_TAPS.has(name)) taps.push(h)
  }
  return taps
}

// LPIPS v0.1: unit-normalize each tap across channels (eps outside the sqrt),
// squared diff, weight by lin, spatial mean, sum taps
function lpips(x: Tensor, y: Tensor, inp: Record<string, Tensor>): Tensor {
  const fx = vggTaps(x, inp), fy = vggTaps(y, inp)
  const unit = (f: Tensor) => tdiv(f, add(sqrt(sum(mul(f, f), 1, { keepDims: true })), 1e-10))
  let total: Tensor | null = null
  for (let k = 0; k < 5; k++) {
    const d = sub(unit(fx[k]!), unit(fy[k]!))
    const w = mul(mul(d, d), inp[`lp_lin${k}`]!)
    const s = mul(mean(w), TAP_C[k]!)      // mean over BCHW × C = channel-sum, batch+spatial mean
    total = total === null ? s : add(total, s)
  }
  return total!
}

// pick 3 of the 4 RGBA channels (random triplet per batch element, host-fed
// one-hot `sel` [B,3,4]) from a padding-cropped image — the paper's LPIPS diet
function cropSelect(img: Tensor, sel: Tensor): Tensor {
  const c = narrow(narrow(img, 2, PADH, CROP), 3, PADH, CROP)     // [B,4,CROP,CROP]
  const flat = reshape(c, [B, 4, CROP * CROP])
  return reshape(matmul(sel, flat), [B, 3, CROP, CROP])           // batched [B,3,4]·[B,4,·]
}

// ============================================================================
// The paper's loss, term for term (image_loss.py + loss.py):
//   L2 + L1 (full canvas, masked render) + shape L2 + shape L1 (target alpha
//   vs 2·aUp, unweighted) + LPIPS (crop, random channel triplet) + 100·overflow
// ============================================================================
function lossFn(m: NCAModel, inp: {
  state: Tensor; filters: Tensor; bilin: Tensor; bilin1: Tensor; coords: Tensor
  targetHi: Tensor; chnSel: Tensor
} & Record<string, Tensor>) {
  const { state, filters, bilin, bilin1, coords, targetHi, chnSel } = inp
  let s = state
  for (let k = 0; k < K; k++) s = ncaStep(m, s, filters, B)
  s = capture("final", s)
  const { rendered, aUp } = decode(m, s, bilin, bilin1, coords, B)
  const diff = sub(rendered, targetHi)
  const l2 = mean(square(diff))
  const l1 = mean(abs(diff))
  const shapeDiff = sub(narrow(targetHi, 1, 3, 1), mul(aUp, 2))   // paper: alpha × 2.0
  const shapeL2 = mean(square(shapeDiff))
  const shapeL1 = mean(abs(shapeDiff))
  const lp = lpips(cropSelect(rendered, chnSel), cropSelect(targetHi, chnSel), inp)
  const over = mul(mean(abs(sub(s, clamp(s, -1, 1)))), 100)
  return add(add(add(add(add(l2, l1), shapeL2), shapeL1), lp), over)
}

// Pretrained mode never trains, but demo/render must attach to a CompiledTraining
// to share its param buffers — and the only way to get one is compile(), which
// requires a loss. Compiling the real lossFn (VGG16-LPIPS twice + a K=24 rollout,
// plus the whole backward pass) purely to throw it away is what pinned the GPU
// process on load and froze even the host page. This stand-in touches every param
// (one CA step + one decode) so none is DCE'd out of the shared buffer set, but
// never goes near VGG16. It is only ever compiled, never stepped.
function lossFnLight(m: NCAModel, { state, filters, bilin, bilin1, coords }: {
  state: Tensor; filters: Tensor; bilin: Tensor; bilin1: Tensor; coords: Tensor
}) {
  const s = ncaStep(m, state, filters, B)
  const { rendered } = decode(m, s, bilin, bilin1, coords, B)
  return mean(square(rendered))
}

function demoFn(m: NCAModel, { state, filters }: { state: Tensor; filters: Tensor }) {
  return ncaStep(m, state, filters, 1)
}

function renderFn(m: NCAModel, { state, bilin, bilin1, coords }: { state: Tensor; bilin: Tensor; bilin1: Tensor; coords: Tensor }) {
  return decode(m, state, bilin, bilin1, coords, 1).rendered
}

// ============================================================================
// Host-side constant tables: perception filters, ×S bilinear upsample kernels,
// sub-cell trig coords, seeds.
// ============================================================================
const F3 = [
  [0, 0, 0, 0, 1, 0, 0, 0, 0],
  [-1, 0, 1, -2, 0, 2, -1, 0, 1],
  [-1, -2, -1, 0, 0, 0, 1, 2, 1],
  [1, 2, 1, 2, -12, 2, 1, 2, 1],
]
const FILTERS = (() => {
  const a = new Float32Array(4 * C * 9)
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
  const a = new Float32Array(ch * S * S * 9)
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
const COORDS_B = (() => {
  const a = new Float32Array(B * 4 * GGH)
  for (let b = 0; b < B; b++) a.set(COORDS1, b * 4 * GGH)
  return a
})()

const makeSeedAt = (x: number, y: number) => {
  const a = new Float32Array(STATE_LEN)
  for (let c = 3; c < C; c++) a[c * GG + y * G + x] = 1
  return a
}
let SEED = makeSeedAt(G >> 1, G >> 1)
let SEED_POS: [number, number] = [G >> 1, G >> 1]
let LO_TARGET: Float32Array | null = null   // S×S box-averaged target (drift probe only — no loss term)

// the paper seeds dead center, but the donut is a hole there. If the seed pixel
// is transparent, death is absorbing (a lethal rule), so relocate the seed to
// the nearest solid pixel and log it.
function placeSeed(lo: Float32Array) {
  const cx = G >> 1, cy = G >> 1
  if (lo[3 * GG + cy * G + cx]! >= 0.95) { SEED = makeSeedAt(cx, cy); SEED_POS = [cx, cy]; return }
  let sx = cx, sy = cy, bestD = Infinity
  for (let y = 0; y < G; y++)
    for (let x = 0; x < G; x++)
      if (lo[3 * GG + y * G + x]! >= 0.95) {
        const d = (x - cx) ** 2 + (y - cy) ** 2
        if (d < bestD) { bestD = d; sx = x; sy = y }
      }
  slog(`WARNING: target center is transparent — seed moved to (${sx},${sy}) (paper seeds dead center)`)
  SEED = makeSeedAt(sx, sy)
  SEED_POS = [sx, sy]
}

// random 3-of-4 channel one-hots, [B,3,4], fresh per training step
function fillChnSel(sel: Float32Array) {
  sel.fill(0)
  for (let b = 0; b < B; b++) {
    const perm = [0, 1, 2, 3]
    for (let i = 3; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0
      const t = perm[i]!; perm[i] = perm[j]!; perm[j] = t
    }
    for (let i = 0; i < 3; i++) sel[b * 12 + i * 4 + perm[i]!] = 1
  }
}

function sampleDistinct(out: Int32Array, n: number) {
  for (let i = 0; i < out.length; i++) {
    let v: number
    do { v = (Math.random() * n) | 0 } while (out.subarray(0, i).includes(v))
    out[i] = v
  }
}

function carveDisc(s: Float32Array, off: number, cx: number, cy: number, r: number) {
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(G - 1, Math.ceil(cy + r)); y++)
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(G - 1, Math.ceil(cx + r)); x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 > r * r) continue
      for (let c = 0; c < C; c++) s[off + c * GG + y * G + x] = 0
    }
}

function sliceL2(state: Float32Array, slice: number, tgt: Float32Array) {
  const off = slice * STATE_LEN
  let s = 0
  for (let i = 0; i < 4 * GG; i++) { const d = state[off + i]! - tgt[i]!; s += d * d }
  return s / (4 * GG)
}

const aliveCount = (s: Float32Array, off = 0) => {
  let n = 0
  for (let i = 0; i < GG; i++) if (s[off + 3 * GG + i]! > 0.1) n++
  return n
}
const deadOrExploded = (alive: number, maxAbs: number) =>
  alive === 0 || alive > GG * 0.6 || maxAbs > 3

// Procedurally generated target: a genuinely 3D donut, RAY-MARCHED — a real
// torus SDF tilted back in Z, orthographic camera down -Z, so the hole reads
// as an ellipse and the near rim actually OCCLUDES the far one. Shaded with a
// warm key light (soft self-shadow + ambient occlusion so the hole reads deep),
// a broad glaze sheen plus a tight specular glint, and a cool rim. Fully
// procedural: no image, no three.js. Emits the same premultiplied CHW RGBA hi +
// S×S box-averaged lo the PNG loader did. The donut's center is a HOLE
// (transparent), so the paper's seed-at-center assumption breaks; placeSeed's
// tripwire relocates the seed onto the ring.
async function loadTarget(): Promise<{ hi: Float32Array; lo: Float32Array }> {
  const INNER = GH - 2 * PADH
  const cx = GH / 2, cy = GH / 2
  const R = INNER * 0.30                      // major (ring) radius, px
  const r = INNER * 0.15                      // minor (tube) radius, px
  const THETA = 0.98                          // tilt angle (rad) ≈ 56°; near rim
  const cT = Math.cos(THETA), sT = -Math.sin(THETA)   // sits low+front, far rim high+back
  const EXTENT = R + r
  const nrm = (x: number, y: number, z: number) => {
    const l = Math.hypot(x, y, z) || 1
    return [x / l, y / l, z / l] as const
  }
  const L = nrm(-0.45, 0.72, 0.6)             // key light: upper-left, toward viewer
  const V = [0, 0, 1] as const                // view direction (camera sits at +Z)
  const Hlf = nrm(L[0] + V[0], L[1] + V[1], L[2] + V[2])   // Blinn half-vector
  const ALBEDO = [0.96, 0.34, 0.52] as const  // berry-pink glaze
  const KEY = [1.0, 0.94, 0.88] as const      // warm key light
  const RIM = [0.6, 0.72, 1.0] as const       // cool rim light

  // signed distance to the Z-axis torus, tilted back by THETA around X
  const map = (x: number, y: number, z: number) => {
    const oy = y * cT + z * sT
    const oz = -y * sT + z * cT
    const qx = Math.hypot(x, oy) - R
    return Math.hypot(qx, oz) - r
  }
  const normalAt = (x: number, y: number, z: number) => {
    const e = 0.3
    return nrm(
      map(x + e, y, z) - map(x - e, y, z),
      map(x, y + e, z) - map(x, y - e, z),
      map(x, y, z + e) - map(x, y, z - e),
    )
  }
  // Quilez soft shadow toward the light (the near tube shadows the far one)
  const softShadow = (x: number, y: number, z: number) => {
    let res = 1, t = 1.2
    for (let i = 0; i < 32 && t < EXTENT * 2.5; i++) {
      const h = map(x + L[0] * t, y + L[1] * t, z + L[2] * t)
      if (h < 0.001) return 0
      if (12 * h / t < res) res = 12 * h / t
      t += Math.max(0.4, h)
    }
    return clamp01(res)
  }
  // Quilez 5-tap ambient occlusion along the surface normal
  const ambientOcc = (x: number, y: number, z: number, nx: number, ny: number, nz: number) => {
    let occ = 0, sca = 1
    for (let i = 1; i <= 5; i++) {
      const d = 0.5 * i
      const h = map(x + nx * d, y + ny * d, z + nz * d)
      occ += (d - h) * sca
      sca *= 0.7
    }
    return clamp01(1 - 1.5 * occ)
  }

  const hi = new Float32Array(4 * GGH)
  const SS = 3                                // subsamples/axis (silhouette AA)
  const roZ = EXTENT + 4                       // ortho ray start, just outside
  const tMax = 2 * EXTENT + 8
  for (let y = 0; y < GH; y++) {
    for (let x = 0; x < GH; x++) {
      let cr = 0, cg = 0, cb = 0, cov = 0
      for (let sy = 0; sy < SS; sy++)
        for (let sx = 0; sx < SS; sx++) {
          const wx = x + (sx + 0.5) / SS - cx
          const wy = -(y + (sy + 0.5) / SS - cy)
          let t = 0, pz = 0, hit = false
          for (let it = 0; it < 96; it++) {
            const zz = roZ - t
            const d = map(wx, wy, zz)
            if (d < 0.02) { pz = zz; hit = true; break }
            t += d
            if (t > tMax) break
          }
          if (!hit) continue
          cov++
          const [nx, ny, nz] = normalAt(wx, wy, pz)
          const diff = Math.max(0, nx * L[0] + ny * L[1] + nz * L[2])
          const sh = softShadow(wx, wy, pz)
          const occ = ambientOcc(wx, wy, pz, nx, ny, nz)
          const ndh = Math.max(0, nx * Hlf[0] + ny * Hlf[1] + nz * Hlf[2])
          const spec = (Math.pow(ndh, 64) * 0.9 + Math.pow(ndh, 12) * 0.22) * sh
          const ndv = Math.max(0, nz)          // N·V, V = +Z
          const rim = Math.pow(1 - ndv, 3) * 0.35
          const key = diff * sh
          const amb = 0.3 * occ
          cr += clamp01(ALBEDO[0] * (amb + key * KEY[0]) + spec * KEY[0] + rim * RIM[0])
          cg += clamp01(ALBEDO[1] * (amb + key * KEY[1]) + spec * KEY[1] + rim * RIM[1])
          cb += clamp01(ALBEDO[2] * (amb + key * KEY[2]) + spec * KEY[2] + rim * RIM[2])
        }
      const n = SS * SS
      const a = cov / n
      if (a <= 0) continue
      const i = y * GH + x
      hi[i] = cr / n                            // premultiplied (Σ over hits / n = color × coverage)
      hi[GGH + i] = cg / n
      hi[2 * GGH + i] = cb / n
      hi[3 * GGH + i] = a
    }
    // the ray-march is pure CPU; yield every few rows so the event loop
    // (status paint, worker messages) isn't starved and the page can't look hung
    if ((y & 7) === 0) await sleep(0)
  }
  const lo = new Float32Array(4 * GG)
  for (let c = 0; c < 4; c++)
    for (let y = 0; y < G; y++)
      for (let x = 0; x < G; x++) {
        let s = 0
        for (let dy = 0; dy < S; dy++)
          for (let dx = 0; dx < S; dx++)
            s += hi[c * GGH + (y * S + dy) * GH + x * S + dx]!
        lo[c * GG + y * G + x] = s / (S * S)
      }
  return { hi, lo }
}

const tile = (t: Float32Array, n: number) => {
  const out = new Float32Array(n * t.length)
  for (let b = 0; b < n; b++) out.set(t, b * t.length)
  return out
}

function saveSafetensors(record: Record<string, Float32Array>): Uint8Array {
  const names = Object.keys(record).sort()
  const header: Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }> = {}
  let off = 0
  for (const n of names) {
    const len = record[n]!.length
    header[n] = { dtype: "F32", shape: [len], data_offsets: [off, off + len * 4] }
    off += len * 4
  }
  const hj = new TextEncoder().encode(JSON.stringify(header))
  const buf = new Uint8Array(8 + hj.length + off)
  new DataView(buf.buffer).setBigUint64(0, BigInt(hj.length), true)
  buf.set(hj, 8)
  let p = 8 + hj.length
  for (const n of names) {
    const f = record[n]!
    buf.set(new Uint8Array(f.buffer, f.byteOffset, f.byteLength), p)
    p += f.byteLength
  }
  return buf
}

// ============================================================================
// LPIPS-VGG16 weights (frozen). Local repo file for dev; the hosted copy
// (streamed with a byte counter) for a published train, where there is no fs.
// ============================================================================
type Progress = (received: number, total: number) => void

// stream a fetch body, reporting bytes-so-far against Content-Length (0 if the
// server omits it), so a big download shows progress instead of looking hung.
async function readWithProgress(res: Response, onProgress?: Progress): Promise<ArrayBuffer> {
  if (!onProgress || !res.body) return res.arrayBuffer()
  const total = Number(res.headers.get("content-length")) || 0
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    onProgress(received, total)
  }
  const out = new Uint8Array(received)
  let off = 0
  for (const c of chunks) { out.set(c, off); off += c.length }
  return out.buffer as ArrayBuffer
}

// Fetch an asset once from the network, thereafter from CacheStorage — which
// survives refreshes and (unlike the plain HTTP cache) isn't evicted for being
// large. Without this, a refresh re-downloaded the ~59MB LPIPS every time.
// Falls back to a plain download if the Cache API is unavailable.
async function cachedFetch(url: string, onProgress?: Progress): Promise<ArrayBuffer> {
  const download = async (): Promise<ArrayBuffer> => {
    const res = await fetch(url)
    if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`)
    return readWithProgress(res, onProgress)
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

let LP: Record<string, Float32Array> = {}
let lpipsPromise: Promise<void> | null = null
function ensureLpips(onProgress?: Progress): Promise<void> {
  if (!lpipsPromise) lpipsPromise = (async () => {
    let buf: ArrayBuffer
    try {
      const bytes = new Uint8Array(await tb.fs.readBytes(LPIPS_PATH))   // local repo copy (dev)
      buf = bytes.buffer as ArrayBuffer
    } catch {
      // hosted copy (published), cached in CacheStorage so a refresh never
      // re-downloads the ~59MB — the VGG16 LPIPS weights never change.
      buf = await cachedFetch(LPIPS_URL, onProgress)
    }
    const { tensors } = loadSafetensors(buf)
    const out: Record<string, Float32Array> = {
      lp_in_gain: tensors["in.gain"]!, lp_in_offset: tensors["in.offset"]!,
    }
    for (const [name] of VGG_PLAN) {
      out[`lp_${name}_W`] = tensors[`${name}.weight`]!
      out[`lp_${name}_b`] = tensors[`${name}.bias`]!
    }
    for (let k = 0; k < 5; k++) out[`lp_lin${k}`] = tensors[`lin${k}.weight`]!
    LP = out
    slog(`lpips: ${Object.keys(out).length} tensors ready`)
  })()
  return lpipsPromise
}

// ============================================================================
// Trainer — owns the compile and the training loop. `pretrained` loads the
// saved checkpoint; `train` runs a fresh 1500-step run.
// ============================================================================
class Trainer extends Component {
  status = "starting…"
  mode: "pretrained" | "train" = "train"
  lossHist: number[] = []
  #runId = 0
  #train: CompiledTraining<NCAModel> | null = null
  #infer: CompiledForward | null = null
  #render: CompiledForward | null = null

  get #stage() { return (this.ctx.root as unknown as IRoot).stage }

  retrain() {
    const stage = this.#stage
    stage.stopDemo()
    if (stage.paused) stage.togglePause()
    this.run()
  }

  setMode(m: "pretrained" | "train") {
    if (m === this.mode) return
    this.mode = m
    this.retrain()
  }

  // one full (re)compile: fresh model, optimizer moments, and LR schedule.
  // Pretrained mode compiles a LIGHT training graph (no VGG16-LPIPS, no K=24
  // rollout) whose only job is to own the shared param buffers the demo/render
  // forwards attach to. The heavy lossFn graph is built only when the user
  // actually trains — otherwise every load paid to compile a VGG16 graph it
  // never ran, saturating the GPU and freezing the page (and the host UI).
  async #compileFresh(): Promise<void> {
    this.#train?.destroy()
    this.#train = null; this.#infer = null; this.#render = null
    const full = this.mode === "train"
    const t = full
      ? await compile({
          model: new NCAModel(),
          loss: lossFn,
          optimizer: {
            kind: "adam", lr: makeLR(),
            // the paper's normalize_model_grads: per-param grad normalization on
            // the NCA only (the SIREN's grads are left raw). No global clipping.
            normalizeGrads: (n) => NCA_PARAMS.has(n),
          },
          inputs: {
            state: [B, C, G, G], filters: [4 * C, 1, 3, 3],
            bilin: [C * S * S, 1, 3, 3], bilin1: [S * S, 1, 3, 3],
            coords: [B, 4, GH, GH], targetHi: [B, 4, GH, GH], chnSel: [B, 3, 4],
            ...LP_SPEC,
          },
        })
      : await compile({
          model: new NCAModel(),
          loss: lossFnLight,
          optimizer: { kind: "adam", lr: makeLR() },
          inputs: {
            state: [B, C, G, G], filters: [4 * C, 1, 3, 3],
            bilin: [C * S * S, 1, 3, 3], bilin1: [S * S, 1, 3, 3],
            coords: [B, 4, GH, GH],
          },
        })
    this.#infer = await t.attach({
      forward: demoFn,
      inputs: { state: [1, C, G, G], filters: [4 * C, 1, 3, 3] },
    })
    this.#render = await t.attach({
      forward: renderFn,
      inputs: { state: [1, C, G, G], bilin: [C * S * S, 1, 3, 3], bilin1: [S * S, 1, 3, 3], coords: [1, 4, GH, GH] },
    })
    this.#train = t
  }

  // load the trained checkpoint: local repo copy first (dev), hosted R2 copy
  // second (published, where there is no local filesystem).
  async #loadWeights(): Promise<boolean> {
    const upload = async (buf: ArrayBuffer) => this.#train!.uploadParams(loadSafetensors(buf).tensors)
    try {
      const bytes = new Uint8Array(await tb.fs.readBytes(TRAINED_PATH))
      await upload(bytes.buffer as ArrayBuffer)
      slog(`loaded ${TRAINED_PATH}`)
      return true
    } catch { /* absent locally — fall through to the hosted copy */ }
    try {
      const res = await fetch(TRAINED_URL)
      if (!res.ok) throw new Error(`${res.status}`)
      await upload(await res.arrayBuffer())
      slog(`loaded ${TRAINED_URL}`)
      return true
    } catch { return false }
  }

  async run() {
    const stage = this.#stage
    const run = ++this.#runId
    if (!isWebGPUAvailable()) {
      this.#setStatus("This needs WebGPU (recent Chrome, Edge, or Safari).")
      return
    }

    this.#setStatus(`generating the donut…`)
    let target: { hi: Float32Array; lo: Float32Array }
    try { target = await loadTarget() }
    catch (e) {
      this.#setStatus(`target generation failed (${(e as Error).message})`)
      slog(`target load failed: ${(e as Error).message}`)
      return
    }
    if (run !== this.#runId) return
    const { hi, lo } = target
    placeSeed(lo)
    LO_TARGET = lo
    stage.showTarget(hi)
    const targetHi = tile(hi, B)

    this.#setStatus(`compiling WGSL kernels (K=${K} rollout + SIREN + VGG16-LPIPS)…`)
    await this.#compileFresh()
    if (run !== this.#runId) return

    // pretrained mode never auto-trains: load the checkpoint, or say there
    // isn't one. Training only ever runs on an explicit `train` click.
    if (this.mode === "pretrained") {
      if (await this.#loadWeights()) {
        if (run !== this.#runId) return
        this.lossHist = []
        this.#setStatus(`ready: bite the donut and watch it regrow`)
        stage.startDemo(this.#infer!, this.#render!)
      } else {
        this.#setStatus(`no pretrained weights found. click “train” to grow one`)
      }
      return
    }

    this.#setStatus("loading LPIPS-VGG16 (~59MB, first time only)…")
    await ensureLpips((received, total) => {
      const mb = (received / 1e6).toFixed(0)
      const totMb = total ? (total / 1e6).toFixed(0) : "?"
      this.#setStatus(`downloading LPIPS-VGG16 (${mb} of ${totMb} MB)…`)
    })
    if (run !== this.#runId) return

    // ------------------------------------------------------------------
    // The training loop: one flat TRAIN_STEPS run of the paper's recipe. A
    // seed is injected into batch slot 0 every INJECT_EVERY steps; uniform
    // pool sampling, full-batch writeback. No damage, no grading, no rounds —
    // one 1500-step pass gives a stable donut (more rounds only buy fine
    // detail this proof of concept doesn't chase).
    // ------------------------------------------------------------------
    const state = new Float32Array(B * STATE_LEN)
    const chnSel = new Float32Array(B * 3 * 4)
    const idx = new Int32Array(B)
    // the on-screen view during training: the LAST batch slice after its
    // K-step rollout, straight from the step's 'final' capture, rendered
    // through the SIREN — you watch exactly what the optimizer just saw.
    // (Slot 0 is the seed-injection slot; the last slice never is.)
    const viewState = new Float32Array(STATE_LEN)
    // pause = playground: scratches carve the view state, and while paused it
    // advances at frame rate under the current weights so wounds heal live.
    // Carves that race an in-flight infer step are re-applied to its output.
    const viewCarves: [number, number][] = []
    stage.onTrainCarve = (x, y) => {
      if (!stage.paused) stage.togglePause()   // scratching mid-train drops into the paused playground
      stage.carveInto(viewState, x, y)
      viewCarves.push([x, y])
      stage.paintCells(viewState)
    }
    let loss = NaN, ema = NaN
    let stepMsAcc = 0, stepMsN = 0
    this.lossHist = []
    slog(`training start: donut, G=${G} K=${K} B=${B} S=${S} pool=${POOL_SIZE} steps=${TRAIN_STEPS}`)

    const pool: Float32Array[] = Array.from({ length: POOL_SIZE }, () => SEED.slice())
    let step = 0
    while (step < TRAIN_STEPS) {
      if (run !== this.#runId) return
      if (stage.paused) {
        this.#setStatus("paused: bite the donut, it regrows with the weights so far")
        while (stage.paused) {
          if (run !== this.#runId) return
          viewCarves.length = 0
          const dr = await this.#infer!.run({ state: viewState, filters: FILTERS })
          if (dr.kind === "completed") {
            const nx = dr.output as Float32Array
            for (const [cx, cy] of viewCarves) stage.carveInto(nx, cx, cy)
            viewState.set(nx)
          }
          if (aliveCount(viewState) === 0) viewState.set(SEED)   // fully scrubbed: regrow
          stage.paintCells(viewState)
          const rr = await this.#render!.run({ state: viewState, bilin: BILIN, bilin1: BILIN1, coords: COORDS1 })
          if (rr.kind === "completed") stage.paintRender(rr.output as Float32Array)
          await nextFrame()
        }
      }
      sampleDistinct(idx, POOL_SIZE)
      for (let b = 0; b < B; b++) state.set(pool[idx[b]!]!, b * STATE_LEN)
      if (step % INJECT_EVERY === 0) state.set(SEED, 0)     // paper: x[:1] = seed
      fillChnSel(chnSel)
      const t0 = performance.now()
      const r = await this.#train!.step({
        state, filters: FILTERS, bilin: BILIN, bilin1: BILIN1,
        coords: COORDS_B, targetHi, chnSel, ...LP,
      })
      stepMsAcc += performance.now() - t0; stepMsN++
      if (r.kind === "failed") {
        if (run !== this.#runId) return
        slog(`step ${step} FAILED: ${r.error}`)
        this.#setStatus(`training failed (${r.error}) — hit “train” again for a fresh init`)
        return
      }
      if (r.kind === "completed") {
        step++
        loss = r.loss
        const fin = r.captures.get("final") as Float32Array
        for (let b = 0; b < B; b++) pool[idx[b]!]!.set(fin.subarray(b * STATE_LEN, (b + 1) * STATE_LEN))
        viewState.set(fin.subarray((B - 1) * STATE_LEN, B * STATE_LEN))
        // a scratch that raced this in-flight step (auto-pausing) would be
        // clobbered by the writeback above — re-apply it before it shows.
        for (const [cx, cy] of viewCarves) stage.carveInto(viewState, cx, cy)
        viewCarves.length = 0
        stage.paintCells(viewState)
        const rr = await this.#render!.run({ state: viewState, bilin: BILIN, bilin1: BILIN1, coords: COORDS1 })
        if (rr.kind === "completed") stage.paintRender(rr.output as Float32Array)
        if (step % 10 === 0 || step === 1) {
          let mx = 0, alive0 = 0
          for (let i = 0; i < STATE_LEN; i++) { const v = Math.abs(fin[i]!); if (v > mx) mx = v }
          for (let i = 0; i < GG; i++) if (fin[3 * GG + i]! > 0.1) alive0++
          slog(`step ${step} loss ${loss.toExponential(4)} aliveSeed ${alive0} maxAbs ${mx.toFixed(3)} ms/step ${(stepMsAcc / Math.max(1, stepMsN)).toFixed(0)}`)
          stepMsAcc = 0; stepMsN = 0
          if (!Number.isNaN(ema)) this.lossHist.push(ema)
        }
        ema = Number.isNaN(ema) ? loss : 0.9 * ema + 0.1 * loss
        this.#setStatus(`step ${step}/${TRAIN_STEPS} · loss ${ema.toFixed(4)}`)
      }
      await sleep(0)
    }
    if (run !== this.#runId) return
    slog(`TRAINING COMPLETE loss ${loss.toFixed(5)}`)
    await this.#saveWeights()
    this.#setStatus(`trained live on your GPU. bite the donut and watch it regrow`)
    stage.startDemo(this.#infer!, this.#render!)
  }

  // download the current params and write them to the local checkpoint.
  async #saveWeights(): Promise<boolean> {
    if (!this.#train) return false
    try {
      const params = await this.#train.downloadParams()
      await tb.fs.write(TRAINED_PATH, saveSafetensors(params))
      slog(`saved weights → ${TRAINED_PATH}`)
      return true
    } catch (e) { slog(`weight save failed: ${(e as Error).message}`); return false }
  }

  #setStatus(s: string) { this.status = s; this.update() }
}

// ============================================================================
// Stage — the three canvases (cells, painted hero, target) + the demo loop.
// ============================================================================
class Stage extends Component {
  paused = false
  demoActive = false
  #live = false                               // has either left box painted yet? gates the "loading…" overlay
  #canvas: HTMLCanvasElement | null = null
  #ctx: CanvasRenderingContext2D | null = null
  #img: ImageData | null = null
  #cellsCtx: CanvasRenderingContext2D | null = null
  #cellsImg: ImageData | null = null
  #demoToken = 0
  #state: Float32Array | null = null
  #painting = false
  #tctx: CanvasRenderingContext2D | null = null
  #timg: ImageData | null = null
  #target: Float32Array | null = null
  #renderDamaged: (() => void) | null = null
  onTrainCarve: ((x: number, y: number) => void) | null = null
  #pendingCarves: [number, number][] = []

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
  }

  bootCells(cv: HTMLCanvasElement) {
    if (this.#cellsCtx) return
    cv.width = cv.height = G
    this.#cellsCtx = cv.getContext("2d")!
    this.#cellsImg = new ImageData(G, G)
  }

  togglePause() {
    this.paused = !this.paused
    this.update()
    return this.paused
  }

  // the two left boxes are blank until the first frame paints; `live` gates a
  // centered "loading…" overlay on them, removed the moment either one paints.
  get live() { return this.#live }
  #markLive() { if (!this.#live) { this.#live = true; this.update() } }

  paintRender(arr: Float32Array) {
    if (!this.#ctx) return
    this.#fill(arr, this.#img!, GGH)
    this.#ctx.putImageData(this.#img!, 0, 0)
    this.#markLive()
  }

  paintCells(arr: Float32Array) {
    if (this.#cellsCtx) this.#blit(arr, this.#cellsImg!, this.#cellsCtx, GG)
    this.#markLive()
  }

  bootTarget(cv: HTMLCanvasElement) {
    if (this.#tctx) return
    cv.width = cv.height = GH
    this.#tctx = cv.getContext("2d")!
    this.#timg = new ImageData(GH, GH)
    if (this.#target) this.showTarget(this.#target)
  }

  showTarget(arr: Float32Array) {
    this.#target = arr
    if (this.#tctx) this.#blit(arr, this.#timg!, this.#tctx, GGH)
  }

  #blit(arr: Float32Array, img: ImageData, ctx: CanvasRenderingContext2D, n: number) {
    this.#fill(arr, img, n)
    ctx.putImageData(img, 0, 0)
  }

  #fill(arr: Float32Array, img: ImageData, n: number) {
    const d = img.data
    for (let i = 0; i < n; i++) {
      const a = clamp01(arr[3 * n + i]!)
      const ia = a > 0.01 ? 1 / Math.max(a, 0.2) : 0
      d[i * 4] = clamp01(arr[i]! * ia) * 255
      d[i * 4 + 1] = clamp01(arr[n + i]! * ia) * 255
      d[i * 4 + 2] = clamp01(arr[2 * n + i]! * ia) * 255
      d[i * 4 + 3] = a * 255
    }
  }

  async startDemo(infer: CompiledForward, render: CompiledForward) {
    const token = ++this.#demoToken
    this.demoActive = true
    this.#state = SEED.slice()
    let renderBusy = false
    this.#renderDamaged = () => {
      if (renderBusy || token !== this.#demoToken) return
      renderBusy = true
      render.run({ state: this.#state!, bilin: BILIN, bilin1: BILIN1, coords: COORDS1 }).then(r => {
        renderBusy = false
        if (token === this.#demoToken && r.kind === "completed") this.paintRender(r.output as Float32Array)
      })
    }
    this.update()
    let frames = 0
    while (token === this.#demoToken) {
      if (!this.paused) {
        this.#pendingCarves.length = 0
        const r = await infer.run({ state: this.#state!, filters: FILTERS })
        if (token !== this.#demoToken) return
        if (r.kind === "completed") {
          const next = r.output as Float32Array
          for (const [cx, cy] of this.#pendingCarves) this.carveInto(next, cx, cy)
          this.#pendingCarves.length = 0
          this.#state = this.#stabilize(next) ? SEED.slice() : next
        }
        const rr = await render.run({ state: this.#state!, bilin: BILIN, bilin1: BILIN1, coords: COORDS1 })
        if (token !== this.#demoToken) return
        if (rr.kind === "completed") {
          this.paintRender(rr.output as Float32Array)
          this.paintCells(this.#state!)
        }
        frames++
        if (frames % 60 === 0 && LO_TARGET) slog(`drift t=${frames} mse=${sliceL2(this.#state!, 0, LO_TARGET).toFixed(5)}`)
      }
      await nextFrame()
    }
  }

  stopDemo() {
    this.#demoToken++
    this.demoActive = false
    this.#renderDamaged = null
    this.#live = false            // a fresh (re)compile is coming — show the loading overlay again
    this.update()
  }

  reseed() {
    if (!this.demoActive || !this.#state) return
    this.#state.set(SEED)
    if (this.paused) this.togglePause()
  }

  #aliveCount() { return this.#state ? aliveCount(this.#state) : 0 }

  #stabilize(s: Float32Array): boolean {
    let maxAbs = 0
    for (let i = 0; i < STATE_LEN; i++) {
      const v = s[i]!
      const av = v < 0 ? -v : v
      if (av > maxAbs) maxAbs = av
      if (av > 1) s[i] = v < 0 ? -1 : 1
    }
    return deadOrExploded(aliveCount(s), maxAbs)
  }

  testPoke() {
    if (!this.demoActive || !this.#state) {
      if (this.onTrainCarve) { this.onTrainCarve(...SEED_POS); slog("poked the training display lineage"); return }
      slog("poke ignored: demo not active")
      return
    }
    const before = this.#aliveCount()
    const [cx, cy] = SEED_POS
    this.#carve(cx, cy)
    slog(`poked at (${cx},${cy}): alive ${before} -> ${this.#aliveCount()}`)
  }

  testStats() { slog(`demo alive ${this.#aliveCount()} (demoActive ${this.demoActive})`) }

  #carve(x: number, y: number) {
    this.carveInto(this.#state!, x, y)
    this.#pendingCarves.push([x, y])
  }

  carveInto(s: Float32Array, x: number, y: number) { carveDisc(s, 0, x, y, DAMAGE_R) }

  #damageAt(e: PointerEvent) {
    const rect = this.#canvas!.getBoundingClientRect()
    const x = Math.floor((e.clientX - rect.left) / rect.width * G)
    const y = Math.floor((e.clientY - rect.top) / rect.height * G)
    if (!this.demoActive || !this.#state) { this.onTrainCarve?.(x, y); return }
    this.#carve(x, y)
    this.paintCells(this.#state!)
    this.#renderDamaged?.()
  }
}

// ============================================================================
// Root — header, canvases, dock.
// ============================================================================
const icon = (...kids: any[]) => svg({ class: "ico", viewBox: "0 0 24 24", fill: "currentColor" }, ...kids)
const ICON = {
  play: () => icon(polygon({ points: "8,5 8,19 19,12" })),
  pause: () => icon(rect({ x: 7, y: 5, width: 3.5, height: 14, rx: 1 }), rect({ x: 13.5, y: 5, width: 3.5, height: 14, rx: 1 })),
  refresh: () => icon(path({ d: "M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" })),
}

interface IRoot { trainer: Trainer; stage: Stage }

class Root extends Component implements IRoot {
  trainer = new Trainer()
  stage = new Stage()

  #boot(el: Element) {
    this.stage.boot(el as HTMLCanvasElement)
    this.trainer.run()
  }

  view() {
    return div({ class: "wrap" },
      div({ class: "header" },
        h1({ class: "title" }, "Regrow the Donut"),
        p({ class: "cap" },
          "Every cell runs the same tiny neural network, and from a single seed they cooperate to grow the ",
          "donut, then regrow it when you bite it. The twist: the cells never hold color, they hold a ",
          "latent code that a shared decoder paints into a far higher-resolution image. It learns all of ",
          "this live on your GPU, using the method from ",
          a({ class: "link", href: "https://cells2pixels.github.io/", target: "_blank", rel: "noopener" }, "“From Cells to Pixels”"),
          ".",
        ),
      ),
      div({ class: "row" },
        div({ class: "panel" },
          div({ class: "pane-wrap" },
            canvas({ class: "pane cells", key: "cells", onMounted: (el: Element) => this.stage.bootCells(el as HTMLCanvasElement) }),
            this.stage.live ? null : span({ class: "pane-loading" }, "loading…"),
          ),
          span({ class: "pane-label" }, `cells ${G}² · raw lattice`),
        ),
        div({ class: "panel" },
          div({ class: "pane-wrap" },
            canvas({ class: "pane grid", key: "grid", onMounted: (el: Element) => this.#boot(el) }),
            this.stage.live ? null : span({ class: "pane-loading" }, "loading…"),
          ),
          span({ class: "pane-label" }, `painted ${GH}² · bite me`),
        ),
        div({ class: "panel" },
          canvas({ class: "pane", key: "target", onMounted: (el: Element) => this.stage.bootTarget(el as HTMLCanvasElement) }),
          span({ class: "pane-label" }, "target · the donut"),
        ),
      ),
      div({ class: "dock" },
        div({ class: "controls" },
          div({ class: "modes" },
            button({ class: ["mode-btn", this.trainer.mode === "pretrained" ? "active" : ""], title: "load the pretrained donut and watch it regrow", onClick: () => this.trainer.setMode("pretrained") }, "pretrained"),
            button({ class: ["mode-btn", this.trainer.mode === "train" ? "active" : ""], title: "train a fresh donut from one seed cell", onClick: () => this.trainer.setMode("train") }, "train"),
          ),
          div({ class: "bar" },
            button(
              { class: "icon-btn", title: this.stage.paused ? "play" : "pause", onClick: () => this.stage.togglePause() },
              this.stage.paused ? ICON.play() : ICON.pause(),
            ),
            this.stage.demoActive
              ? button({ class: "icon-btn", title: "restart from a single seed cell", onClick: () => this.stage.reseed() }, ICON.refresh())
              : null,
          ),
        ),
        this.trainer.lossHist.length > 1 ? this.lossChart() : null,
        div({ class: "readout" }, this.trainer.status),
      ),
    )
  }

  lossChart() {
    const h = this.trainer.lossHist
    if (h.length < 2) return null
    const W = 240, Hh = 34
    const max = Math.max(...h), min = Math.min(...h), rng = max - min || 1
    const x = (i: number) => i / (h.length - 1) * W
    const y = (l: number) => 2 + (max - l) / rng * (Hh - 4)
    const line = h.map((l, i) => `${x(i).toFixed(1)},${y(l).toFixed(1)}`).join(" ")
    return svg({ class: "loss-chart", viewBox: `0 0 ${W} ${Hh}`, preserveAspectRatio: "none" },
      polyline({ points: line, fill: "none", stroke: "var(--accent)", strokeWidth: 1.5, vectorEffect: "non-scaling-stroke" }),
    )
  }
}

const root = new Root()
new App({ root, id: "app" })
tb.onMessage((m: unknown) => {
  if (m === "poke") root.stage.testPoke()
  if (m === "stats") root.stage.testStats()
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
  --font: ui-sans-serif, system-ui, sans-serif;
  --font-mono: ui-monospace, Menlo, Consolas, monospace;
  --text: 0.9rem;
  --accent: #9a794b;
  --bg: #f1f1f1;
  --fg: #1c1c1c;
  --muted: #626262;
  --readout: #414141;
  --panel: #ffffffcc;
  --panel-border: #d4d4d4;
  --btn-bg: #ffffff;
  --btn-fg-hover: #121212;
  --btn-border-hover: #b6b6b6;
}
html[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0e0e0e;
  --fg: #eaeaea;
  --muted: #b0b0b0;
  --readout: #d2d2d2;
  --panel: #161616cc;
  --panel-border: #2e2e2e;
  --btn-bg: #1e1e1e;
  --btn-fg-hover: #ffffff;
  --btn-border-hover: #454545;
}
html { font-size: 106.25%; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg); font-family: var(--font); }
body { font-size: var(--text); }
#app { height: 100dvh; min-height: 520px; }
.wrap { height: 100%; display: flex; flex-direction: column; --gap: 14px; }
.header, .dock { display: flex; flex-direction: column; align-items: center; gap: var(--gap); padding: var(--gap) 18px; text-align: center; }
.header > *, .dock > * { margin: 0; }
.title { font-size: 1.5rem; font-weight: 680; letter-spacing: .04em; text-transform: uppercase; color: var(--accent); }
.cap { max-width: 720px; line-height: 1.5; color: var(--muted); }
.cap .link { color: var(--accent); text-decoration: none; border-bottom: 1px solid color-mix(in srgb, var(--accent) 45%, transparent); }
.cap .link:hover { border-bottom-color: var(--accent); }
/* the triptych: three equal panels, full width, pipeline order */
.row { flex: 1 1 auto; min-height: 200px; max-width: 900px; margin: 0 auto; display: flex; align-items: center; justify-content: center; gap: 14px; padding: 4px 18px; }
.panel { flex: 0 1 auto; display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 0; }
.pane {
  width: min(28vw, 46vmin, 260px);
  aspect-ratio: 1;
  border: 1px solid var(--panel-border);
  border-radius: 8px;
}
.pane.cells { image-rendering: pixelated; }
/* the hero canvas: quiet prominence via a soft warm elevation shadow + a muted
   accent-tinted edge, not a hard gold outline (which read as a stark alert). a
   gentle hover deepens the tint — the "you can touch this one" cue on approach. */
.pane.grid {
  touch-action: none;
  cursor: crosshair;
  border-color: color-mix(in srgb, var(--accent) 38%, var(--panel-border));
  box-shadow: 0 10px 28px -14px color-mix(in srgb, var(--accent) 50%, transparent);
}
.pane.grid:hover { border-color: color-mix(in srgb, var(--accent) 62%, var(--panel-border)); }
.pane-label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--muted); }
.pane-wrap { position: relative; line-height: 0; }
.pane-loading {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  font-family: var(--font-mono); font-size: 0.78rem; letter-spacing: .04em; color: var(--muted);
  pointer-events: none; animation: paneLoad 1.4s ease-in-out infinite;
}
@keyframes paneLoad { 0%, 100% { opacity: .4 } 50% { opacity: .85 } }
@media (max-width: 620px) {
  /* stack the triptych so each pane gets real width and its label sits centered
     under it, instead of three squished panes with colliding mono captions */
  #app { height: auto; min-height: 100dvh; }
  .wrap { height: auto; }
  .row { flex-direction: column; gap: 16px; padding: 12px 18px; }
  .pane { width: min(72vw, 300px); }
}
.dock { padding-bottom: calc(var(--gap) + 4px); }
.controls { display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 10px; }
.bar { display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 8px; }
.icon-btn { cursor: pointer; box-sizing: border-box; display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; color: var(--readout); background: var(--btn-bg); border: 1px solid var(--panel-border); border-radius: 50%; }
.icon-btn:hover { color: var(--btn-fg-hover); border-color: var(--btn-border-hover); }
.icon-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.ico { width: 1.45em; height: 1.45em; flex: none; }
.readout { font-family: var(--font-mono); color: var(--readout); user-select: text; cursor: text; }
.modes { display: inline-flex; gap: 2px; padding: 2px; background: var(--btn-bg); border: 1px solid var(--panel-border); border-radius: 10px; }
.mode-btn { cursor: pointer; font: inherit; font-size: 0.8rem; color: var(--readout); background: transparent; border: none; border-radius: 0; padding: 4px 12px; }
.mode-btn:first-child { border-radius: 8px 0 0 8px; }
.mode-btn:last-child { border-radius: 0 8px 8px 0; }
.mode-btn.active { background: var(--accent); color: #fff; }
.loss-chart { width: 240px; max-width: 100%; height: 34px; display: block; opacity: 0.85; }
@media (max-height: 620px) {
  .wrap { --gap: 9px; }
  .cap { display: none; }
  .title { font-size: 1.3rem; }
}
```

**config.json**

```json
{
  "description": "A neural cellular automaton grows a 3D donut from one cell, trained live on WebGPU with the Cells-to-Pixels recipe.",
  "dependencies": {
    "tensorgrad": "^0.4.3",
    "domeleon": "^0.6.3"
  }
}
```
