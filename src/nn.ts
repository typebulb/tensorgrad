// Standard "batteries-included" Module subclasses for the most common layers.
//
// JAX-style: each class declares its params (and their init); the forward is a
// plain function the user calls with `(module, x)`. No subclassing, no method
// dispatch — keeps the autograd-traced computation visible at the call site.
//
// Import as a namespace:
//
//   import { nn } from 'tensorgrad'
//   class Block extends Module {
//     ln  = new nn.LayerNorm(D)
//     ffn = new nn.Linear(D, 4 * D)
//   }
//   const y = nn.linearFwd(p.ffn, nn.layerNormFwd(p.ln, x))

import { Module } from './module.js'
import type { Tensor } from './ir.js'
import { add, matmul, sub, mul, div, sqrt, meanLast, sumLast, reshape, swapAxes, oneHot, logSoftmaxLast } from './ops.js'
import { ShapeError } from './shape.js'
import { captureSite } from './ir.js'

// ----------------------------------------------------------------------------
// Linear: y = x @ W (+ b)
// ----------------------------------------------------------------------------

export class Linear extends Module {
  W: Tensor
  b: Tensor | null
  constructor(public readonly inDim: number, public readonly outDim: number, withBias = true) {
    super()
    this.W = this.param([inDim, outDim])                      // randn, scale 0.02
    this.b = withBias ? this.param([outDim], { init: 'zeros' }) : null
  }
}

export function linearFwd(p: Linear, x: Tensor): Tensor {
  const out = matmul(x, p.W)
  return p.b ? add(out, p.b) : out
}

// ----------------------------------------------------------------------------
// LayerNorm — normalizes over the last axis. eps defaults to 1e-5.
// ----------------------------------------------------------------------------

export class LayerNorm extends Module {
  g: Tensor
  b: Tensor
  constructor(public readonly d: number, public readonly eps: number = 1e-5) {
    super()
    this.g = this.param([d], { init: 'ones' })
    this.b = this.param([d], { init: 'zeros' })
  }
}

export function layerNormFwd(p: LayerNorm, x: Tensor): Tensor {
  const m = meanLast(x)
  const c = sub(x, m)
  const v = meanLast(mul(c, c))
  const stdev = sqrt(add(v, p.eps))
  return add(mul(div(c, stdev), p.g), p.b)
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

// ----------------------------------------------------------------------------
// Loss helpers
// ----------------------------------------------------------------------------

/** Per-position cross-entropy along the last (vocab) axis: returns
 *  `-log p(target)` at each position. `logits` is `[..., V]`; `targets` is
 *  `[...]` of i32; result is `[...]` (one rank less than logits). The user
 *  applies their own masking + reduction downstream — useful when only some
 *  positions contribute (e.g. result-digit masking) or for label smoothing. */
export function crossEntropyLast(logits: Tensor, targets: Tensor): Tensor {
  const site = captureSite('crossEntropyLast')
  if (targets.dtype !== 'i32') {
    throw new ShapeError(`crossEntropyLast: targets must be i32, got ${targets.dtype}`, site)
  }
  const vocab = logits.shape[logits.shape.length - 1]!
  const lp = logSoftmaxLast(logits)                                   // [..., V]
  const targetLp = sumLast(mul(lp, oneHot(targets, vocab, 'f32')))    // [...]
  return mul(targetLp, -1)
}
