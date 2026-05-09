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
import { add, matmul, sub, mul, div, sqrt, meanLast } from './ops.js'

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
