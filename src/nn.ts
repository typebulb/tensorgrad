// Standard "batteries-included" Module subclasses for the most common layers.
//
// Each class declares its params and a `.fwd(x)` method that runs the forward
// computation. Forward methods are pure tensorgrad ops — autograd traces
// through them just like any other call.
//
//   import { nn } from 'tensorgrad'
//   class Block extends Module {
//     ln  = new nn.LayerNorm(D)
//     ffn = new nn.Linear(D, 4 * D)
//   }
//   const y = p.ffn.fwd(p.ln.fwd(x))

import { Module } from './module.js'
import type { Tensor } from './ir.js'
import { add, matmul, sub, mul, div, sqrt, mean, sum, reshape, swapAxes, oneHot, logSoftmaxLast, embedding } from './ops.js'
import { ShapeError } from './shape.js'
import { captureSite } from './ir.js'
import type { Captures } from './runtime.js'

// ----------------------------------------------------------------------------
// Embedding: integer indices → row lookup. Like `nn.Embedding` in PyTorch.
// ----------------------------------------------------------------------------

export class Embedding extends Module {
  /** Embedding table, shape `[vocab, dim]`. Default init is `randn` with
   *  PyTorch's default std (0.02 in tensorgrad). For Llama-style scaled
   *  init, pass `{ init: init.randn({ scale: 1 / Math.sqrt(dim) }) }`. */
  W: Tensor
  constructor(public readonly vocab: number, public readonly dim: number) {
    super()
    this.W = this.param([vocab, dim])
  }
  /** Lookup: `idx` is `[...]` of i32, returns `[..., dim]` f32. */
  fwd(idx: Tensor): Tensor {
    return embedding(this.W, idx)
  }
}

// ----------------------------------------------------------------------------
// Linear: y = x @ W (+ b)
// ----------------------------------------------------------------------------

export interface LinearOptions {
  /** Include a bias term (default true). */
  bias?: boolean
}

export class Linear extends Module {
  /** Weight matrix, shape `[inDim, outDim]`. Applied as `x @ W` so input
   *  features are along the input axis and rows are not transposed. */
  W: Tensor
  /** Bias, shape `[outDim]`, or null if `bias: false`. Broadcasts over the
   *  leading axes of `x @ W`. */
  b: Tensor | null
  constructor(public readonly inDim: number, public readonly outDim: number, opts: LinearOptions = {}) {
    super()
    this.W = this.param([inDim, outDim])
    this.b = opts.bias === false ? null : this.param([outDim], { init: 'zeros' })
  }
  fwd(x: Tensor): Tensor {
    const out = matmul(x, this.W)
    return this.b ? add(out, this.b) : out
  }
}

// ----------------------------------------------------------------------------
// LayerNorm — normalizes over the last axis. eps defaults to 1e-5.
// ----------------------------------------------------------------------------

export class LayerNorm extends Module {
  /** Gain (gamma), shape `[d]`, init `ones`. Scales the normalized output. */
  g: Tensor
  /** Bias (beta), shape `[d]`, init `zeros`. Shifts the normalized output. */
  b: Tensor
  constructor(public readonly d: number, public readonly eps: number = 1e-5) {
    super()
    this.g = this.param([d], { init: 'ones' })
    this.b = this.param([d], { init: 'zeros' })
  }
  fwd(x: Tensor): Tensor {
    const m = mean(x, -1, { keepDims: true })
    const c = sub(x, m)
    const v = mean(mul(c, c), -1, { keepDims: true })
    const stdev = sqrt(add(v, this.eps))
    return add(mul(div(c, stdev), this.g), this.b)
  }
}

// ----------------------------------------------------------------------------
// Multi-head attention shape helpers — split the last (model) axis into
// [nHeads, headDim] and bring heads ahead of the sequence axis.
// ----------------------------------------------------------------------------

/** [..., T, D] → [..., H, T, D/H]. Folds the standard
 *  `transpose(reshape(x, [..., T, H, d]), [..., H, T, d])` pattern into one
 *  call. Last dim of `x` must divide evenly by `nHeads`. */
export function splitHeads(x: Tensor, nHeads: number): Tensor {
  const site = captureSite('splitHeads')
  const r = x.shape.length
  if (r < 2) throw new ShapeError(`splitHeads: requires rank >= 2, got ${r}`, site)
  const T = x.shape[r - 2]!
  const D = x.shape[r - 1]!
  if (D % nHeads !== 0) {
    throw new ShapeError(`splitHeads: last dim ${D} not divisible by nHeads ${nHeads}`, site)
  }
  const lead = x.shape.slice(0, r - 2)
  const reshaped = reshape(x, [...lead, T, nHeads, D / nHeads])
  // Swap T (axis lead.length) with H (axis lead.length + 1).
  return swapAxes(reshaped, lead.length, lead.length + 1)
}

/** Inverse of `splitHeads`: [..., H, T, d] → [..., T, H*d]. */
export function mergeHeads(x: Tensor): Tensor {
  const site = captureSite('mergeHeads')
  const r = x.shape.length
  if (r < 3) throw new ShapeError(`mergeHeads: requires rank >= 3, got ${r}`, site)
  const H = x.shape[r - 3]!
  const T = x.shape[r - 2]!
  const d = x.shape[r - 1]!
  const lead = x.shape.slice(0, r - 3)
  // Swap H (axis r-3) and T (axis r-2): [..., H, T, d] → [..., T, H, d]
  const swapped = swapAxes(x, r - 3, r - 2)
  return reshape(swapped, [...lead, T, H * d])
}

/** Slice a captured tensor named `name` into one Float32Array per head, using
 *  the static shape registered at compile time. The leading axis is treated as
 *  heads (matching `splitHeads` layout at B=1); a leading singleton batch is
 *  stripped if present so callers can pass capture names directly. Throws if
 *  the capture isn't registered or wasn't read back this call. */
export function unsplitHeads(captures: Captures, name: string): Float32Array[] {
  const flat = captures.get(name)
  const shape = captures.shapeOf(name)
  if (shape.length < 2) {
    throw new Error(`unsplitHeads: '${name}' shape needs >= 2 dims, got [${shape.join(', ')}]`)
  }
  // For inference graphs at B=1, captures have shape [1, H, ..., ...]. Strip
  // the leading 1 if present so the next axis is heads.
  const s = shape[0] === 1 ? shape.slice(1) : shape
  const H = s[0]!
  let stride = 1
  for (let i = 1; i < s.length; i++) stride *= s[i]!
  const expected = H * stride
  if (flat.length !== expected) {
    throw new Error(`unsplitHeads: '${name}' length ${flat.length} doesn't match shape product ${expected}`)
  }
  return Array.from({ length: H }, (_, h) => flat.slice(h * stride, (h + 1) * stride))
}

// ----------------------------------------------------------------------------
// Loss helpers
// ----------------------------------------------------------------------------

/** Per-position negative log-likelihood along the last axis: returns
 *  `-logProbs[target]` at each position. `logProbs` is `[..., V]` (already
 *  log-softmaxed); `targets` is `[...]` of i32; result is `[...]` (one
 *  rank less than `logProbs`).
 *
 *  Mirrors PyTorch's `F.nll_loss` *before* reduction. Pair with
 *  `logSoftmaxLast` when you need the log-probability intermediate visible
 *  (e.g. to `capture` it for inspection). Otherwise prefer
 *  `crossEntropyLast(logits, targets)` which takes raw logits and fuses
 *  log-softmax + NLL — same numerics, fewer ops, no risk of accidentally
 *  passing logits twice (a silent miscompose if you also wrote
 *  `log_softmax` upstream). */
export function nllLoss(logProbs: Tensor, targets: Tensor): Tensor {
  const site = captureSite('nllLoss')
  if (targets.dtype !== 'i32') {
    throw new ShapeError(`nllLoss: targets must be i32, got ${targets.dtype}`, site)
  }
  const vocab = logProbs.shape[logProbs.shape.length - 1]!
  const targetLp = sum(mul(logProbs, oneHot(targets, vocab, 'f32')), -1)   // [...]
  return mul(targetLp, -1)
}

/** Per-position cross-entropy along the last (vocab) axis: returns
 *  `-log p(target)` at each position. `logits` is `[..., V]` (raw, NOT
 *  pre-log-softmaxed); `targets` is `[...]` of i32; result is `[...]`
 *  (one rank less than logits). The user applies their own masking +
 *  reduction downstream — useful when only some positions contribute
 *  (e.g. result-digit masking) or for label smoothing.
 *
 *  Fused log-softmax + NLL. Pass raw logits — don't apply `logSoftmaxLast`
 *  yourself first or the model silently double-log-softmaxes (a common
 *  miscompose when porting PyTorch code that uses `log_softmax` in the
 *  model and `F.nll_loss` in the loss). If you need the log-probability
 *  intermediate visible, use `logSoftmaxLast` + `nllLoss` instead. */
export function crossEntropyLast(logits: Tensor, targets: Tensor): Tensor {
  const site = captureSite('crossEntropyLast')
  if (targets.dtype !== 'i32') {
    throw new ShapeError(`crossEntropyLast: targets must be i32, got ${targets.dtype}`, site)
  }
  return nllLoss(logSoftmaxLast(logits), targets)
}
