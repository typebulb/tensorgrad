// Batteries-included `Module` subclasses for the most common layers. Each
// declares its params and exposes `.fwd(x)`; the forward is regular ops so
// autograd traces through it like any other call.

import { Module, init } from './module.js'
import type { Tensor } from './ir.js'
import { add, matmul, sub, mul, div, sqrt, mean, sum, reshape, oneHot, logSoftmax, embedding, conv2d, pairOpt } from './ops.js'
import type { Conv2dOptions } from './ops.js'
import { ShapeError } from './shape.js'
import { captureSite } from './ir.js'

/** 2D convolution layer (NCHW). Shape and option names match PyTorch's
 *  `nn.Conv2d` so 1-shot ports don't need transposes. Wraps the pure
 *  `conv2d` op plus an optional broadcast-add bias. */
export class Conv2d extends Module {
  /** Weight, shape `[outC, inC, kH, kW]`. Default init is `randn` with
   *  scale 0.02 (the tensorgrad default). For Kaiming init, pass
   *  `{ init: init.kaiming() }` to a custom param() override — or rely on
   *  the loss landscape's tolerance for the default. */
  W: Tensor
  /** Bias, shape `[outC]`, or null if `bias: false`. Broadcasts across the
   *  H_out, W_out axes via reshape. */
  b: Tensor | null
  readonly strideH: number; readonly strideW: number
  readonly padH: number;    readonly padW: number
  constructor(
    public readonly inC: number,
    public readonly outC: number,
    kernelSize: number | readonly [number, number],
    opts: Conv2dOptions & { bias?: boolean } = {},
  ) {
    super()
    const [kH, kW] = pairOpt(kernelSize, 1)
    const [sH, sW] = pairOpt(opts.stride, 1)
    const [pH, pW] = pairOpt(opts.padding, 0)
    this.strideH = sH; this.strideW = sW
    this.padH = pH; this.padW = pW
    this.W = this.param([outC, inC, kH, kW])
    this.b = opts.bias === false ? null : this.param([outC], { init: init.zeros() })
  }
  /** Apply this conv to `x`: `[B, inC, H, W] → [B, outC, H', W']`. Adds
   *  bias (broadcast over the spatial axes) when present. */
  fwd(x: Tensor): Tensor {
    const y = conv2d(x, this.W, { stride: [this.strideH, this.strideW], padding: [this.padH, this.padW] })
    if (!this.b) return y
    const bShaped = reshape(this.b, [1, this.outC, 1, 1])
    return add(y, bShaped)
  }
}

/** Index → row lookup. Matches PyTorch's `nn.Embedding(vocab, dim)`.
 *  Differentiable via the matmul adjoint — no custom scatter-with-atomic-add
 *  backward needed; see `ops.ts`'s `embedding` for the decomposition. */
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
    return embedding(idx, this.W)
  }
}

export interface LinearOptions {
  /** Include a bias term (default true). */
  bias?: boolean
}

/** Affine layer `y = x @ W + b`. Matches PyTorch's `nn.Linear` but the
 *  weight orientation is `[inDim, outDim]` (so `matmul(x, W)` is direct,
 *  no transpose at call site — PyTorch stores `[outDim, inDim]` and uses
 *  `x @ W^T` internally). Bias is optional. */
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
    this.b = opts.bias === false ? null : this.param([outDim], { init: init.zeros() })
  }
  /** Apply: `x @ W` (+ `b` when present). Broadcasts bias over leading axes. */
  fwd(x: Tensor): Tensor {
    const out = matmul(x, this.W)
    return this.b ? add(out, this.b) : out
  }
}

/** Layer normalization over the last axis. Matches PyTorch's
 *  `nn.LayerNorm(d, eps=1e-5)`. Subtract mean, divide by stddev (with `eps`
 *  for stability), then affine-scale by `g` and shift by `b`. */
export class LayerNorm extends Module {
  /** Gain (gamma), shape `[d]`, init `ones`. Scales the normalized output. */
  g: Tensor
  /** Bias (beta), shape `[d]`, init `zeros`. Shifts the normalized output. */
  b: Tensor
  constructor(public readonly d: number, public readonly eps: number = 1e-5) {
    super()
    this.g = this.param([d], { init: init.ones() })
    this.b = this.param([d], { init: init.zeros() })
  }
  /** Normalize over the last axis; affine-scale and shift. Shape preserved. */
  fwd(x: Tensor): Tensor {
    const m = mean(x, -1, { keepDims: true })
    const c = sub(x, m)
    const v = mean(mul(c, c), -1, { keepDims: true })
    const stdev = sqrt(add(v, this.eps))
    return add(mul(div(c, stdev), this.g), this.b)
  }
}

/** Llama-style RMS normalization over the last axis. Scale-only (no
 *  mean-subtraction, no bias): `y = x / sqrt(mean(x², -1) + eps) * g`.
 *  Cheaper than `LayerNorm`; stable enough for modern transformers. */
export class RMSNorm extends Module {
  /** Gain (gamma), shape `[d]`, init `ones`. Scales the RMS-normalized output. */
  g: Tensor
  constructor(public readonly d: number, public readonly eps: number = 1e-6) {
    super()
    this.g = this.param([d], { init: init.ones() })
  }
  /** RMS-normalize over the last axis; affine-scale by `g`. Shape preserved. */
  fwd(x: Tensor): Tensor {
    const ms = mean(mul(x, x), -1, { keepDims: true })
    const rstd = sqrt(add(ms, this.eps))
    return mul(div(x, rstd), this.g)
  }
}

// ---- Loss helpers --------------------------------------------------------

/** How a loss reduces across leading axes:
 *  - `'mean'` (default) — scalar; mean over all leading positions.
 *  - `'sum'` — scalar; sum over all leading positions.
 *  - `'none'` — per-position tensor (one rank less than the input logits).
 *
 *  Matches PyTorch's `F.cross_entropy(..., reduction=...)`. Pass `'none'`
 *  when you want to mask or weight positions yourself before reducing. */
export interface LossOptions {
  reduction?: 'mean' | 'sum' | 'none'
}

function reduceLoss(t: Tensor, reduction: 'mean' | 'sum' | 'none'): Tensor {
  if (reduction === 'none') return t
  return reduction === 'mean' ? mean(t) : sum(t)
}

/** Negative log-likelihood along the last (vocab) axis. `logProbs` is
 *  `[..., V]` (already log-softmaxed); `targets` is `[...]` of i32.
 *  Default reduction is `'mean'` (scalar). Pass `{ reduction: 'none' }` for
 *  a per-position tensor (`[...]`, one rank less than `logProbs`) — use
 *  this when masking or weighting positions before reducing yourself.
 *
 *  Pair with `logSoftmax` only when you need the log-probability intermediate
 *  visible (e.g. to `capture` it for inspection). Otherwise prefer
 *  `crossEntropy(logits, targets)` which takes raw logits and fuses
 *  log-softmax + NLL — same numerics, fewer ops, no risk of accidentally
 *  passing logits twice (a silent miscompose if you also wrote `logSoftmax`
 *  upstream). */
export function nllLoss(logProbs: Tensor, targets: Tensor, opts: LossOptions = {}): Tensor {
  const site = captureSite('nllLoss')
  if (targets.dtype !== 'i32') {
    throw new ShapeError(`nllLoss: targets must be i32, got ${targets.dtype}`, site)
  }
  const vocab = logProbs.shape[logProbs.shape.length - 1]!
  const targetLp = sum(mul(logProbs, oneHot(targets, vocab, 'f32')), -1)
  const perPos = mul(targetLp, -1)
  return reduceLoss(perPos, opts.reduction ?? 'mean')
}

/** Cross-entropy along the last (vocab) axis. `logits` is `[..., V]` (raw,
 *  NOT pre-log-softmaxed); `targets` is `[...]` of i32. Default reduction
 *  is `'mean'` (scalar) — matches PyTorch's `F.cross_entropy`. Pass
 *  `{ reduction: 'none' }` for a per-position tensor (`[...]`, one rank less
 *  than `logits`) when masking positions yourself.
 *
 *  Fused log-softmax + NLL. Pass raw logits — don't apply `logSoftmax`
 *  yourself first or the model silently double-log-softmaxes (a common
 *  miscompose when porting PyTorch code that uses `log_softmax` in the
 *  model and `F.nll_loss` in the loss). If you need the log-probability
 *  intermediate visible, use `logSoftmax` + `nllLoss` instead. */
export function crossEntropy(logits: Tensor, targets: Tensor, opts: LossOptions = {}): Tensor {
  const site = captureSite('crossEntropy')
  if (targets.dtype !== 'i32') {
    throw new ShapeError(`crossEntropy: targets must be i32, got ${targets.dtype}`, site)
  }
  return nllLoss(logSoftmax(logits), targets, opts)
}
