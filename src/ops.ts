// User-facing op surface. Each function captures its call site, validates
// shapes (via src/shape.ts), and appends an op to the current graph. No
// numeric work — these calls just build the IR.

import type { Tensor, Shape, Dtype, OpNode, Graph } from './ir.js'
import { addOp, captureSite } from './ir.js'
import { currentGraph, tensorInput } from './trace.js'
import {
  inferElementwiseBinop, inferUnary, inferMeanLast, inferSumLast, inferArgmaxLast,
  inferReshape, inferPermute, inferMatmul, inferMatmulBatched,
  inferOneHot, inferWhereCausal, inferSliceRange, inferScatterAxis, inferConcat,
  inferBroadcastTo, inferSumToShape, inferReluGrad, inferWhere,
  inferConv2d, inferMaxPool2d,
  ShapeError, showShape,
} from './shape.js'

// ---- Element-wise binops --------------------------------------------------

// Shared helper for arithmetic ops (outDtype = input dtype) and comparisons
// (outDtype = bool).
function binopOp(
  name: string,
  kind: OpNode['kind'],
  a: Tensor, b: Tensor,
  outDtype: Dtype = a.dtype,
): Tensor {
  const site = captureSite(name)
  if (a.dtype !== b.dtype) throw new ShapeError(`${name}: dtype mismatch (${a.dtype} vs ${b.dtype})`, site)
  const outShape = inferElementwiseBinop(name, a.shape, b.shape, site)
  return addOp(currentGraph(), kind, outShape, outDtype, site, { a: a.id, b: b.id })
}

/** Element-wise addition. Trailing-axis broadcast (smaller operand's shape
 *  is a suffix of the larger's). Scalar form: `add(x, 2)` adds 2 to every
 *  element via a fused scalar kernel. */
export function add(a: Tensor, b: Tensor | number): Tensor {
  return typeof b === 'number' ? addScalar(a, b) : binopOp('add', 'add', a, b)
}
/** Element-wise subtraction. Same broadcast rules as `add`; scalar form
 *  subtracts a JS number. */
export function sub(a: Tensor, b: Tensor | number): Tensor {
  return typeof b === 'number' ? addScalar(a, -b) : binopOp('sub', 'sub', a, b)
}
/** Element-wise multiplication. Same broadcast rules as `add`; scalar form
 *  multiplies by a JS number via a fused scalar kernel. */
export function mul(a: Tensor, b: Tensor | number): Tensor {
  return typeof b === 'number' ? mulScalar(a, b) : binopOp('mul', 'mul', a, b)
}
/** Element-wise division. Same broadcast rules as `add`. Scalar form
 *  `div(x, k)` is rewritten as `mulScalar(x, 1/k)` (throws on `k === 0`). */
export function div(a: Tensor, b: Tensor | number): Tensor {
  if (typeof b === 'number') {
    if (b === 0) throw new ShapeError(`div: scalar divisor cannot be zero`, captureSite('div'))
    return mulScalar(a, 1 / b)
  }
  return binopOp('div', 'div', a, b)
}

/** Element-wise minimum. Scalar form clamps from above. Backward routes
 *  the gradient to whichever side won; ties go to `b`. */
export function min(a: Tensor, b: Tensor | number): Tensor {
  const rhs = typeof b === 'number' ? constScalar(b, a.dtype) : b
  return binopOp('min', 'min', a, rhs)
}
/** Element-wise maximum. Scalar form clamps from below. Backward routes
 *  the gradient to whichever side won; ties go to `b`. */
export function max(a: Tensor, b: Tensor | number): Tensor {
  const rhs = typeof b === 'number' ? constScalar(b, a.dtype) : b
  return binopOp('max', 'max', a, rhs)
}

/** Element-wise clamp: `min(max(a, lo), hi)`. Both bounds are scalars. */
export function clamp(a: Tensor, lo: number, hi: number): Tensor {
  return min(max(a, lo), hi)
}

// ---- Element-wise scalar binops -------------------------------------------
// Fused mul/add by a JS number. The Tensor-overload variants of `mul`/`add`
// dispatch here when the rhs is a number, avoiding a 0-d constant allocation.

export function mulScalar(a: Tensor, scalar: number): Tensor {
  const site = captureSite('mulScalar')
  return addOp(currentGraph(), 'mul_scalar', a.shape, a.dtype, site, { a: a.id, scalar })
}

export function addScalar(a: Tensor, scalar: number): Tensor {
  const site = captureSite('addScalar')
  return addOp(currentGraph(), 'add_scalar', a.shape, a.dtype, site, { a: a.id, scalar })
}

// ----------------------------------------------------------------------------
// Unary ops.
// ----------------------------------------------------------------------------

type UnaryKind = 'sqrt' | 'rsqrt' | 'log' | 'exp' | 'relu' | 'neg' | 'abs' | 'tanh' | 'sigmoid' | 'sin' | 'cos'

function unary(name: UnaryKind, a: Tensor): Tensor {
  const site = captureSite(name)
  if (a.dtype !== 'f32') throw new ShapeError(`${name}: requires f32, got ${a.dtype}`, site)
  return addOp(currentGraph(), name, inferUnary(name, a.shape, site), 'f32', site, { a: a.id })
}

/** Element-wise square root. Requires `f32`. */
export const sqrt    = (a: Tensor): Tensor => unary('sqrt',    a)
/** Element-wise reciprocal square root (`1 / sqrt(x)`). Requires `f32`.
 *  One fused kernel — faster and more numerically stable than `1/sqrt(x)`. */
export const rsqrt   = (a: Tensor): Tensor => unary('rsqrt',   a)
/** Element-wise natural log. Requires `f32`. */
export const log     = (a: Tensor): Tensor => unary('log',     a)
/** Element-wise exponential (`e^x`). Requires `f32`. */
export const exp     = (a: Tensor): Tensor => unary('exp',     a)
/** Element-wise ReLU (`max(0, x)`). Requires `f32`. Backward uses a fused
 *  `relu_grad` op that passes `dy` through wherever `x > 0`. */
export const relu    = (a: Tensor): Tensor => unary('relu',    a)
/** Element-wise negation (`-x`). Requires `f32`. */
export const neg     = (a: Tensor): Tensor => unary('neg',     a)
/** Element-wise absolute value. Requires `f32`. Subgradient is 0 at `x = 0`. */
export const abs     = (a: Tensor): Tensor => unary('abs',     a)
/** Element-wise hyperbolic tangent. Requires `f32`. */
export const tanh    = (a: Tensor): Tensor => unary('tanh',    a)
/** Element-wise logistic sigmoid (`1 / (1 + e^-x)`). Requires `f32`. */
export const sigmoid = (a: Tensor): Tensor => unary('sigmoid', a)
/** Element-wise sine. Requires `f32`. */
export const sin     = (a: Tensor): Tensor => unary('sin',     a)
/** Element-wise cosine. Requires `f32`. */
export const cos     = (a: Tensor): Tensor => unary('cos',     a)

/** Element-wise square (`x * x`). Reads better than `mul(x, x)` for the
 *  common MSE / L2-regularization / variance patterns. Pure sugar — no new
 *  kernel, identical IR to `mul(x, x)`. */
export function square(a: Tensor): Tensor {
  return mul(a, a)
}

/** SiLU / Swish: `x * sigmoid(x)`. Composed from primitives. */
export function silu(a: Tensor): Tensor {
  return mul(a, sigmoid(a))
}

/** Hidden tensor_input name for the per-step PRNG seed shared by `dropout`
 *  and `randn`. The runtime auto-injects this scalar before each
 *  `step()`/`run()` when the compiled graph contains any stochastic op;
 *  users do not pass it. Exposed as a named constant so worker + proxy
 *  agree on the convention. */
export const PRNG_SEED_INPUT = '__prngSeed'

/** Inverted dropout: with probability `p`, zero an element; otherwise scale
 *  it by `1 / (1 - p)`. `p` is a compile-time constant in `[0, 1)`. Calling
 *  `dropout(x, 0)` is a no-op (returns `x` directly). The mask is
 *  reproducible from `(per-step seed, this-call salt, thread id)` via a PCG
 *  hash inside the kernel — backward recomputes the same mask, no buffer
 *  capture needed.
 *
 *  Train-vs-eval handling: free-function form, no mode flag. Call `dropout`
 *  inside your training forward (lossFn); omit it from your inference
 *  forward (predictFn). The two are compiled separately by the
 *  `compileModule` / `compileForward` pair.
 *
 *  **Salt ordering.** Salts are assigned by graph-construction order across
 *  `dropout` and `randn` calls combined. Adding or removing a stochastic op
 *  upstream shifts the salts (and therefore the random streams) of every
 *  later stochastic call. The forward / backward pair of a single dropout
 *  always shares its salt, so masks line up correctly. Refactor-induced
 *  stream shifts don't break correctness, just reproducibility across
 *  refactors. */
export function dropout(x: Tensor, p: number): Tensor {
  if (p === 0) return x
  const site = captureSite('dropout')
  if (p < 0 || p >= 1) throw new ShapeError(`dropout: p must be in [0, 1), got ${p}`, site)
  if (x.dtype !== 'f32') throw new ShapeError(`dropout: requires f32, got ${x.dtype}`, site)
  const g = currentGraph()
  const seed = findOrCreatePrngSeed(g)
  // Salt counts both dropout and randn ops so independent stochastic sites
  // get independent PCG streams. Forward/backward of a single dropout share
  // their salt; see `dropoutWithSalt`.
  const salt = countStochasticOps(g)
  return addOp(g, 'dropout', inferUnary('dropout', x.shape, site), 'f32', site, { a: x.id, seed: seed.id, p, salt })
}

/** Sample a tensor of the given shape from N(0, 1). Reuses the shared
 *  per-step PRNG seed — each `step()`/`run()` advances the seed so each call
 *  sees a fresh draw. The output has zero gradient (sampling from a fixed
 *  distribution carries no gradient information), so this is the right
 *  primitive for VAE reparameterization, DDPM training noise, or any
 *  stochastic regularization beyond dropout.
 *
 *  **Salt ordering.** Salts are assigned by graph-construction order across
 *  `randn` and `dropout` combined. Adding or removing a stochastic op
 *  upstream shifts the random streams of every later stochastic call — not
 *  a correctness issue, but a reproducibility-across-refactors gotcha to
 *  know about. */
export function randn(shape: Shape): Tensor {
  const site = captureSite('randn')
  for (const d of shape) {
    if (!Number.isInteger(d) || d <= 0) {
      throw new ShapeError(`randn: shape must be positive integers, got ${showShape(shape)}`, site)
    }
  }
  const g = currentGraph()
  const seed = findOrCreatePrngSeed(g)
  const salt = countStochasticOps(g)
  return addOp(g, 'randn', shape, 'f32', site, { seed: seed.id, salt, shape })
}

/** Internal: emit a `dropout` op with an explicit salt. Used by grad.ts to
 *  emit the backward kernel using the same (seed, salt, p) as the forward,
 *  so the mask matches. */
export function dropoutWithSalt(dy: Tensor, p: number, salt: number, seedId: number): Tensor {
  const site = captureSite('dropout')
  return addOp(currentGraph(), 'dropout', dy.shape, 'f32', site, { a: dy.id, seed: seedId, p, salt })
}

function findOrCreatePrngSeed(g: Graph): Tensor {
  for (const op of g.ops) {
    if (op.kind === 'tensor_input' && op.name === PRNG_SEED_INPUT) {
      return g.tensors[op.out]!
    }
  }
  return tensorInput(PRNG_SEED_INPUT, [], 'i32')
}

// Counts every stochastic op in the graph so each new dropout / randn site
// receives a unique salt and an independent PCG stream.
function countStochasticOps(g: Graph): number {
  let n = 0
  for (const op of g.ops) if (op.kind === 'dropout' || op.kind === 'randn') n++
  return n
}

/** GELU using the GPT-2 tanh approximation:
 *  `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`. Composed. */
export function gelu(a: Tensor): Tensor {
  const c = 0.7978845608028654 // sqrt(2 / π)
  const x3 = mul(mul(a, a), a)
  const inner = mul(add(a, mul(x3, 0.044715)), c)
  return mul(mul(a, 0.5), add(tanh(inner), 1))
}

// ---- Reductions ------------------------------------------------------------
// IR kernels (`mean_last`, `sum_last`, `argmax_last`) are last-axis only.
// Other axes compose as `permute-axis-to-end` + `*_last` + reshape/back-perm,
// so there's no new codegen needed for arbitrary-axis reduction.

export interface ReduceOpts {
  /** Preserve the reduced axis as size 1 (PyTorch's `keepdim=True`).
   *  Default false (axis is removed from the output shape). Ignored when
   *  `axis` is omitted — the no-axis form always returns a 0-d scalar. */
  keepDims?: boolean
}

function meanLastIR(a: Tensor): Tensor {
  const site = captureSite('mean')
  if (a.dtype !== 'f32') throw new ShapeError(`mean: requires f32, got ${a.dtype}`, site)
  const outShape = inferMeanLast('mean', a.shape, site)
  return addOp(currentGraph(), 'mean_last', outShape, a.dtype, site, { a: a.id })
}

function sumLastIR(a: Tensor): Tensor {
  const site = captureSite('sum')
  if (a.dtype !== 'f32') throw new ShapeError(`sum: requires f32, got ${a.dtype}`, site)
  const outShape = inferSumLast('sum', a.shape, site)
  return addOp(currentGraph(), 'sum_last', outShape, a.dtype, site, { a: a.id })
}

function argmaxLastIR(a: Tensor): Tensor {
  const site = captureSite('argmax')
  if (a.dtype !== 'f32') throw new ShapeError(`argmax: requires f32, got ${a.dtype}`, site)
  const outShape = inferArgmaxLast('argmax', a.shape, site)
  return addOp(currentGraph(), 'argmax_last', outShape, 'i32', site, { a: a.id })
}

/** Index of the maximum value along an axis. Returns `i32`, one rank less
 *  than input (no keepDims option — argmax of a multi-axis reduction is
 *  meaningless). Negative axis counts from the end; default is `-1`. With
 *  no `axis`, argmax over the flattened tensor (0-d i32 result).
 *  Non-differentiable: gradients do not flow back through this op. */
export function argmax(a: Tensor, axis?: number): Tensor {
  if (axis === undefined) return argmaxLastIR(reshape(a, [-1]))
  const r = a.shape.length
  const k = axis < 0 ? r + axis : axis
  if (k < 0 || k >= r) {
    throw new ShapeError(`argmax: axis ${axis} out of range for shape [${a.shape.join(',')}]`, captureSite('argmax'))
  }
  if (k === r - 1) return argmaxLastIR(a)
  const perm = [...Array(r).keys()].filter(i => i !== k).concat(k)
  return argmaxLastIR(permute(a, perm))
}

/** Index of the minimum value along an axis. Mirrors `argmax` exactly;
 *  implemented as `argmax(neg(x), axis)` so no new kernel is needed. */
export function argmin(a: Tensor, axis?: number): Tensor {
  return argmax(neg(a), axis)
}

/** Mean along an axis (or all axes). Negative axis counts from the end.
 *  `keepDims` (default false) preserves the reduced axis as size 1.
 *  With no `axis`, reduces all elements to a 0-d scalar.
 *
 *  ```
 *  mean(x)                          // 0-d scalar
 *  mean(x, -1)                      // PyTorch's x.mean(dim=-1)
 *  mean(x, -1, { keepDims: true })  // preserve the trailing axis as size 1
 *  mean(x, 1)                       // reduce middle axis (permutes internally)
 *  ``` */
export function mean(a: Tensor, axis?: number, opts?: ReduceOpts): Tensor {
  if (axis === undefined) {
    const n = a.shape.reduce((p, d) => p * d, 1)
    if (n === 0) throw new ShapeError(`mean: cannot mean over zero elements`, captureSite('mean'))
    return mulScalar(sumLastIR(reshape(a, [-1])), 1 / n)
  }
  return reduceAxis(a, axis, !!opts?.keepDims, 'mean')
}

/** Sum along an axis (or all axes). Negative axis counts from the end.
 *  `keepDims` (default false) preserves the reduced axis as size 1.
 *  With no `axis`, reduces all elements to a 0-d scalar.
 *
 *  ```
 *  sum(x)                          // 0-d scalar
 *  sum(x, -1)                      // PyTorch's x.sum(dim=-1)
 *  sum(x, -1, { keepDims: true })  // preserve the trailing axis as size 1
 *  ``` */
export function sum(a: Tensor, axis?: number, opts?: ReduceOpts): Tensor {
  if (axis === undefined) return sumLastIR(reshape(a, [-1]))
  return reduceAxis(a, axis, !!opts?.keepDims, 'sum')
}

function reduceAxis(a: Tensor, axisArg: number, keepDims: boolean, kind: 'mean' | 'sum'): Tensor {
  const r = a.shape.length
  const k = axisArg < 0 ? r + axisArg : axisArg
  if (k < 0 || k >= r) {
    throw new ShapeError(`${kind}: axis ${axisArg} out of range for shape [${a.shape.join(',')}]`, captureSite(kind))
  }
  const isLast = k === r - 1
  // Move axis k to the trailing position so the last-axis kernel applies.
  const input = isLast ? a : permute(a, [...Array(r).keys()].filter(i => i !== k).concat(k))
  if (kind === 'mean') {
    const reduced = meanLastIR(input)  // [...others, 1]
    if (isLast) return keepDims ? reduced : reshape(reduced, reduced.shape.slice(0, -1))
    if (!keepDims) return reshape(reduced, reduced.shape.slice(0, -1))
    return permute(reduced, backPerm(k, r))
  }
  const dropped = sumLastIR(input)  // [...others]
  if (isLast) return keepDims ? reshape(dropped, [...dropped.shape, 1]) : dropped
  if (!keepDims) return dropped
  return permute(reshape(dropped, [...dropped.shape, 1]), backPerm(k, r))
}

/** Inverse of `[everyone-but-k, k]`: moves the trailing axis back to
 *  position k. Used for `keepDims=true` along a non-last axis. */
function backPerm(k: number, r: number): number[] {
  const out: number[] = []
  for (let i = 0; i < k; i++) out.push(i)
  out.push(r - 1)
  for (let i = k; i < r - 1; i++) out.push(i)
  return out
}

// ----------------------------------------------------------------------------
// Shape ops.
// ----------------------------------------------------------------------------

/** Reshape a tensor to `newShape`. Total element count must match (a single
 *  `-1` is allowed and inferred from the others). Equivalent to PyTorch's
 *  `x.view(...)` / `x.reshape(...)` and NumPy's `np.reshape`. Pure metadata
 *  in the IR — no kernel is emitted when the underlying memory layout is
 *  contiguous. */
export function reshape(a: Tensor, newShape: Shape): Tensor {
  const site = captureSite('reshape')
  const outShape = inferReshape('reshape', a.shape, newShape, site)
  return addOp(currentGraph(), 'reshape', outShape, a.dtype, site, { a: a.id, newShape: outShape })
}

/** Flatten axes `[startAxis, end)` into a single trailing axis. Pure
 *  reshape; no new IR op.
 *
 *  **Default differs from PyTorch.** Tensorgrad defaults to `startAxis=1`
 *  (preserve the batch dim — the canonical CNN classifier-head transition
 *  from `[B, C, H, W]` to `[B, C*H*W]`). PyTorch's `torch.flatten` defaults
 *  to `start_dim=0` (collapse everything to 1-d). Pass `0` explicitly if
 *  you want full flattening. */
export function flatten(a: Tensor, startAxis: number = 1): Tensor {
  const r = a.shape.length
  const s = startAxis < 0 ? r + startAxis : startAxis
  const site = captureSite('flatten')
  if (s < 0 || s > r) {
    throw new ShapeError(`flatten: startAxis ${startAxis} out of range for rank-${r}`, site)
  }
  if (s === r) return a
  return reshape(a, [...a.shape.slice(0, s), -1])
}

/** Permute the axes of a tensor by `perm`. Matches PyTorch's `x.permute(*dims)`
 *  / JAX's `jnp.transpose(x, axes)` / NumPy's `np.transpose(x, axes)`. For the
 *  common case of swapping two axes (PyTorch's `x.transpose(a, b)`), use
 *  `swapAxes`. */
export function permute(a: Tensor, perm: readonly number[]): Tensor {
  const site = captureSite('permute')
  const outShape = inferPermute('permute', a.shape, perm, site)
  return addOp(currentGraph(), 'permute', outShape, a.dtype, site, { a: a.id, perm })
}

/** Swap two axes of a tensor. Negative indices count from the end (so
 *  `swapAxes(x, -1, -2)` swaps the last two — the common attention pattern).
 *  All other axes keep their position. Implemented as `permute` with the
 *  permutation `[0, 1, ..., axis2, ..., axis1, ..., n-1]`. */
export function swapAxes(a: Tensor, axis1: number, axis2: number): Tensor {
  const r = a.shape.length
  const norm = (axis: number): number => axis < 0 ? r + axis : axis
  const i1 = norm(axis1)
  const i2 = norm(axis2)
  const site = captureSite('swapAxes')
  if (i1 < 0 || i1 >= r || i2 < 0 || i2 >= r) {
    throw new ShapeError(`swapAxes: axis out of range — got (${axis1}, ${axis2}) for rank-${r} tensor`, site)
  }
  if (i1 === i2) return a
  const perm = Array.from({ length: r }, (_, k) => k)
  perm[i1] = i2
  perm[i2] = i1
  return permute(a, perm)
}

// ---- Linear algebra -------------------------------------------------------

/** Matrix multiplication. Dispatches on input rank:
 *  - `a [..., M, K] · b [K, N] → [..., M, N]`   (rhs rank 2: broadcast lhs batch)
 *  - `a [..., M, K] · b [..., K, N] → [..., M, N]`  (both batched, same rank)
 *
 *  Broadcasting beyond these two cases (e.g. mismatched batch ranks) isn't
 *  supported — reshape explicitly. The two cases lower to different fused
 *  kernels under the hood; the public surface is one function. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const site = captureSite('matmul')
  if (a.dtype !== 'f32' || b.dtype !== 'f32') {
    throw new ShapeError(`matmul: requires f32, got ${a.dtype} and ${b.dtype}`, site)
  }
  if (a.shape.length < 2) {
    throw new ShapeError(`matmul: lhs must have rank >= 2, got ${showShape(a.shape)}`, site)
  }
  if (b.shape.length < 2) {
    throw new ShapeError(`matmul: rhs must have rank >= 2, got ${showShape(b.shape)}`, site)
  }
  // rhs rank 2: lhs's leading axes broadcast as batch.
  if (b.shape.length === 2) {
    const outShape = inferMatmul('matmul', a.shape, b.shape, site)
    return addOp(currentGraph(), 'matmul', outShape, 'f32', site, { a: a.id, b: b.id })
  }
  if (a.shape.length !== b.shape.length) {
    throw new ShapeError(
      `matmul: rank mismatch — lhs ${showShape(a.shape)} vs rhs ${showShape(b.shape)}. ` +
      `When rhs is batched (rank > 2), lhs must have the same rank. Reshape explicitly if needed.`,
      site,
    )
  }
  const outShape = inferMatmulBatched('matmul', a.shape, b.shape, site)
  return addOp(currentGraph(), 'matmul_batched', outShape, 'f32', site, { a: a.id, b: b.id })
}

// ---- Indexing / casting ---------------------------------------------------

/** One-hot encode an `i32` index tensor of shape `[...]` into shape
 *  `[..., depth]`. Result dtype defaults to `f32`. Used internally by
 *  `embedding`; pair with `matmul` for end-to-end differentiable lookup. */
export function oneHot(indices: Tensor, depth: number, dtype: Dtype = 'f32'): Tensor {
  const site = captureSite('oneHot')
  if (indices.dtype !== 'i32') {
    throw new ShapeError(`oneHot: indices must be i32, got ${indices.dtype}`, site)
  }
  const outShape = inferOneHot('oneHot', indices.shape, depth, site)
  return addOp(currentGraph(), 'one_hot', outShape, dtype, site, { indices: indices.id, depth, dtype })
}

/** Gather from a 1-D table by index. `table` is `[V]` f32; `indices` is any
 *  shape `[...]` of i32; result is `[...]` (same shape as indices). Used for
 *  per-sample lookups into precomputed schedules (DDPM α-schedule, RL value
 *  tables indexed along a single axis, etc.).
 *
 *  Composes from `oneHot` + `matmul` (the same trick `embedding` uses for
 *  `[V, D]` tables) — autograd flows through the matmul adjoint, no new
 *  kernel. For 2-D table lookups (`embedding` semantics) use `embedding`. */
export function take(table: Tensor, indices: Tensor): Tensor {
  const site = captureSite('take')
  if (table.shape.length !== 1) {
    throw new ShapeError(`take: table must be 1-d [V], got ${showShape(table.shape)}`, site)
  }
  if (table.dtype !== 'f32') {
    throw new ShapeError(`take: table must be f32, got ${table.dtype}`, site)
  }
  if (indices.dtype !== 'i32') {
    throw new ShapeError(`take: indices must be i32, got ${indices.dtype}`, site)
  }
  const V = table.shape[0]!
  const oh = oneHot(indices, V, 'f32')       // [...indices.shape, V]
  const t2d = reshape(table, [V, 1])         // [V, 1]
  const result = matmul(oh, t2d)             // [...indices.shape, 1]
  return reshape(result, indices.shape)      // [...indices.shape]
}

/** Embedding lookup: pull rows from `table` indexed by `indices`. Decomposes
 *  to `oneHot(indices, vocab) @ table` so autograd works without a dedicated
 *  scatter-with-atomic-add backward — the matmul adjoint rule handles it.
 *  `table` is `[vocab, dim]`; `indices` is any shape `[...]` of i32; result
 *  is `[..., dim]`. The vocab size is taken from `table.shape[0]`. */
export function embedding(indices: Tensor, table: Tensor): Tensor {
  const site = captureSite('embedding')
  if (table.shape.length !== 2) {
    throw new ShapeError(`embedding: table must be 2-d [vocab, dim], got ${showShape(table.shape)}`, site)
  }
  if (indices.dtype !== 'i32') {
    throw new ShapeError(`embedding: indices must be i32, got ${indices.dtype}`, site)
  }
  return matmul(oneHot(indices, table.shape[0]!, 'f32'), table)
}

/** `arange(n)` → `[n]` tensor of values `[0, 1, ..., n-1]`. Default dtype
 *  is `i32` (the index dtype). Used for position embeddings and similar
 *  index generation. */
export function arange(n: number, dtype: Dtype = 'i32'): Tensor {
  const site = captureSite('arange')
  if (n <= 0 || !Number.isInteger(n)) {
    throw new ShapeError(`arange: n must be a positive integer, got ${n}`, site)
  }
  return addOp(currentGraph(), 'arange', [n], dtype, site, { n, dtype })
}

// ---- ML primitives (fused for cleaner autograd + hand-tuned kernels) ------

function softmaxCausalLastIR(a: Tensor): Tensor {
  const site = captureSite('softmaxCausal')
  if (a.dtype !== 'f32') throw new ShapeError(`softmaxCausal: requires f32, got ${a.dtype}`, site)
  inferWhereCausal('softmaxCausal', a.shape, site)
  return addOp(currentGraph(), 'softmax_causal_last', a.shape, 'f32', site, { a: a.id })
}

/** Causal-masked softmax along an axis (fused mask + softmax). Shape
 *  preserved. The last two axes (after permuting to put `axis` last) must be
 *  square (T×T attention scores). Default `axis = -1`; pass an explicit
 *  axis for non-trailing layouts. Prefer this over composing
 *  `whereCausal` + `softmax` yourself. */
export function softmaxCausal(a: Tensor, axis: number = -1): Tensor {
  return axisPreserving(a, axis, 'softmaxCausal', softmaxCausalLastIR)
}

function logSoftmaxLastIR(a: Tensor): Tensor {
  const site = captureSite('logSoftmax')
  if (a.dtype !== 'f32') throw new ShapeError(`logSoftmax: requires f32, got ${a.dtype}`, site)
  return addOp(currentGraph(), 'log_softmax_last', a.shape, 'f32', site, { a: a.id })
}

/** Numerically-stable log-softmax along an axis. Shape preserved.
 *  Negative axis counts from the end; default is `-1`. Non-last axes
 *  permute internally. */
export function logSoftmax(a: Tensor, axis: number = -1): Tensor {
  return axisPreserving(a, axis, 'logSoftmax', logSoftmaxLastIR)
}

/** Numerically-stable softmax along an axis. Composes `exp(logSoftmax(...))`
 *  — the stabilization happens inside the fused log-softmax kernel.
 *  Negative axis counts from the end; default is `-1`. For classifiers
 *  that want explicit class probabilities and for attention-distribution
 *  visualization. Use `nn.crossEntropy` for the training loss. */
export function softmax(a: Tensor, axis: number = -1): Tensor {
  return exp(logSoftmax(a, axis))
}

// Apply a last-axis-only IR op along an arbitrary axis by permuting in,
// applying, permuting back. Output shape matches input.
function axisPreserving(
  a: Tensor, axisArg: number, opName: string,
  applyLast: (t: Tensor) => Tensor,
): Tensor {
  const r = a.shape.length
  if (r === 0) {
    throw new ShapeError(`${opName}: cannot apply to 0-d tensor`, captureSite(opName))
  }
  const k = axisArg < 0 ? r + axisArg : axisArg
  if (k < 0 || k >= r) {
    throw new ShapeError(`${opName}: axis ${axisArg} out of range for shape [${a.shape.join(',')}]`, captureSite(opName))
  }
  if (k === r - 1) return applyLast(a)
  const perm = [...Array(r).keys()].filter(i => i !== k).concat(k)
  return permute(applyLast(permute(a, perm)), backPerm(k, r))
}

/** Pre-softmax causal mask. Sets cells where `i < j` on the last two axes
 *  to `fillValue` (typically `-1e30`); lower-triangle entries pass through.
 *  Use when you want the masked scores explicitly (e.g. to `capture` them);
 *  for the common case, prefer `softmaxCausal` which fuses the mask + softmax. */
export function whereCausal(a: Tensor, fillValue: number): Tensor {
  const site = captureSite('whereCausal')
  if (a.dtype !== 'f32') throw new ShapeError(`whereCausal: requires f32, got ${a.dtype}`, site)
  inferWhereCausal('whereCausal', a.shape, site)
  return addOp(currentGraph(), 'where_causal', a.shape, 'f32', site, { a: a.id, fillValue })
}

// ---- Slicing --------------------------------------------------------------

/** General-axis slice: take elements `[start, end)` along `axis`. Negative
 *  `axis` indexes from the end (Python convention). */
export function sliceRange(a: Tensor, axis: number, start: number, end: number): Tensor {
  const site = captureSite('sliceRange')
  const ax = axis < 0 ? a.shape.length + axis : axis
  const outShape = inferSliceRange('sliceRange', a.shape, ax, start, end, site)
  return addOp(currentGraph(), 'slice_range', outShape, a.dtype, site, { a: a.id, axis: ax, start, end })
}

/** `sliceRange`'s adjoint: scatter `a` into `[start, end)` along `axis` of
 *  an otherwise-zero tensor of `outShape`. Emitted by autograd; users go
 *  through `sliceRange`. `axis` must be non-negative. */
export function scatterAxis(a: Tensor, outShape: Shape, axis: number, start: number, end: number): Tensor {
  const site = captureSite('scatterAxis')
  const out = inferScatterAxis('scatterAxis', a.shape, outShape, axis, start, end, site)
  return addOp(currentGraph(), 'scatter_axis', out, a.dtype, site, { a: a.id, outShape, axis, start, end })
}

/** Concatenate two or more tensors along `axis`. All inputs must share
 *  shape except along `axis`; output's size on `axis` is the sum. Negative
 *  `axis` indexes from the end. Capped at 7 inputs (WebGPU bind-group
 *  limit) — chain calls for more. */
const CONCAT_INPUT_CAP = 7
export function concat(tensors: readonly Tensor[], axis: number): Tensor {
  const site = captureSite('concat')
  if (tensors.length === 0) throw new ShapeError(`concat: needs at least one input`, site)
  if (tensors.length === 1) return tensors[0]!
  if (tensors.length > CONCAT_INPUT_CAP) {
    throw new ShapeError(
      `concat: ${tensors.length} inputs exceeds the bind-group cap of ${CONCAT_INPUT_CAP}. ` +
      `Chain two concats, or restructure the call site.`, site,
    )
  }
  const dtype = tensors[0]!.dtype
  for (const t of tensors) {
    if (t.dtype !== dtype) throw new ShapeError(`concat: dtype mismatch (${dtype} vs ${t.dtype})`, site)
  }
  const ax = axis < 0 ? tensors[0]!.shape.length + axis : axis
  const outShape = inferConcat('concat', tensors.map(t => t.shape), ax, site)
  return addOp(currentGraph(), 'concat', outShape, dtype, site, { inputs: tensors.map(t => t.id), axis: ax })
}

/** Stack `tensors` along a *new* axis at position `axis`. Each input gets
 *  an `unsqueeze` first, then concat. All inputs must share shape. */
export function stack(tensors: readonly Tensor[], axis: number): Tensor {
  if (tensors.length === 0) throw new ShapeError(`stack: needs at least one input`, captureSite('stack'))
  const t0 = tensors[0]!
  const ax = axis < 0 ? t0.shape.length + 1 + axis : axis
  const newShape = [...t0.shape.slice(0, ax), 1, ...t0.shape.slice(ax)]
  const expanded = tensors.map(t => reshape(t, newShape))
  return concat(expanded, ax)
}

/** `[..., T, D] → [..., H, T, D/H]`. Folds the standard
 *  `permute(reshape(x, [..., T, H, d]), [..., H, T, d])` pattern into one
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
  return swapAxes(reshaped, lead.length, lead.length + 1)
}

/** Inverse of `splitHeads`: `[..., H, T, d] → [..., T, H*d]`. */
export function mergeHeads(x: Tensor): Tensor {
  const site = captureSite('mergeHeads')
  const r = x.shape.length
  if (r < 3) throw new ShapeError(`mergeHeads: requires rank >= 3, got ${r}`, site)
  const H = x.shape[r - 3]!
  const T = x.shape[r - 2]!
  const d = x.shape[r - 1]!
  const lead = x.shape.slice(0, r - 3)
  const swapped = swapAxes(x, r - 3, r - 2)
  return reshape(swapped, [...lead, T, H * d])
}

/** Inverse of `concat`: split into pieces along `axis` with the given
 *  per-piece sizes (must sum to the axis's size). Composes from
 *  `sliceRange` — no new IR. */
export function split(t: Tensor, sizes: readonly number[], axis: number): Tensor[] {
  const site = captureSite('split')
  const ax = axis < 0 ? t.shape.length + axis : axis
  if (ax < 0 || ax >= t.shape.length) throw new ShapeError(`split: axis ${axis} out of range`, site)
  const total = sizes.reduce((s, x) => s + x, 0)
  if (total !== t.shape[ax]) {
    throw new ShapeError(`split: sizes sum to ${total}, but axis ${ax} has size ${t.shape[ax]}`, site)
  }
  const out: Tensor[] = []
  let cursor = 0
  for (const s of sizes) {
    out.push(sliceRange(t, ax, cursor, cursor + s))
    cursor += s
  }
  return out
}

// ---- Broadcast / un-broadcast (mostly used by autograd) -------------------

export function broadcastTo(a: Tensor, targetShape: Shape): Tensor {
  const site = captureSite('broadcastTo')
  inferBroadcastTo('broadcastTo', a.shape, targetShape, site)
  return addOp(currentGraph(), 'broadcast_to', targetShape, a.dtype, site, { a: a.id, targetShape })
}

export function sumToShape(a: Tensor, targetShape: Shape): Tensor {
  const site = captureSite('sumToShape')
  inferSumToShape('sumToShape', a.shape, targetShape, site)
  return addOp(currentGraph(), 'sum_to_shape', targetShape, a.dtype, site, { a: a.id, targetShape })
}

// ---- Constants ------------------------------------------------------------

/** 0-d tensor with a constant value. Used by autograd to seed the loss
 *  cotangent and by comparisons/min/max for their scalar overloads. */
export function constScalar(value: number, dtype: Dtype = 'f32'): Tensor {
  const site = captureSite('constScalar')
  return addOp(currentGraph(), 'const_scalar', [], dtype, site, { value, dtype })
}

// ---- Comparisons and selection --------------------------------------------

/** Element-wise `a < b`. Same broadcast rules as `add`. Returns `bool` —
 *  pair with `where` to select. Non-differentiable. */
export function less(a: Tensor, b: Tensor | number): Tensor {
  const rhs = typeof b === 'number' ? constScalar(b, a.dtype) : b
  return binopOp('less', 'less', a, rhs, 'bool')
}
/** Element-wise `a > b`. Same broadcast rules as `add`. Returns `bool` —
 *  pair with `where` to select. Non-differentiable. */
export function greater(a: Tensor, b: Tensor | number): Tensor {
  const rhs = typeof b === 'number' ? constScalar(b, a.dtype) : b
  return binopOp('greater', 'greater', a, rhs, 'bool')
}

/** Element-wise select: `out[i] = cond[i] ? a[i] : b[i]`. `cond` must be
 *  `bool` (produced by `less`/`greater`); `a` and `b` must share dtype.
 *  All three operands broadcast-compatible to the output shape.
 *  Gradient flows to `a` where `cond` is true, to `b` where false. */
export function where(cond: Tensor, a: Tensor, b: Tensor): Tensor {
  const site = captureSite('where')
  if (cond.dtype !== 'bool') throw new ShapeError(`where: cond must be bool, got ${cond.dtype}`, site)
  if (a.dtype !== b.dtype) throw new ShapeError(`where: a/b dtype mismatch (${a.dtype} vs ${b.dtype})`, site)
  const outShape = inferWhere('where', cond.shape, a.shape, b.shape, site)
  return addOp(currentGraph(), 'where', outShape, a.dtype, site, { cond: cond.id, a: a.id, b: b.id })
}

/** ReLU's adjoint: `dy` where `x > 0`, else 0. Same shape as `x`. Exposed
 *  so codegen can emit the fused kernel; users go through `relu`. */
export function reluGrad(x: Tensor, dy: Tensor): Tensor {
  const site = captureSite('reluGrad')
  if (x.dtype !== 'f32' || dy.dtype !== 'f32') {
    throw new ShapeError(`reluGrad: requires f32, got ${x.dtype} and ${dy.dtype}`, site)
  }
  const outShape = inferReluGrad('reluGrad', x.shape, dy.shape, site)
  return addOp(currentGraph(), 'relu_grad', outShape, 'f32', site, { x: x.id, dy: dy.id })
}

// ----------------------------------------------------------------------------
// Adam-fused ops. Each does its full per-element update in one kernel.
// ----------------------------------------------------------------------------

export function adamUpdateM(m: Tensor, g: Tensor, b1: number): Tensor {
  const site = captureSite('adamUpdateM')
  if (m.dtype !== 'f32' || g.dtype !== 'f32') throw new ShapeError(`adamUpdateM: requires f32`, site)
  if (m.shape.length !== g.shape.length || m.shape.some((d, i) => d !== g.shape[i])) {
    throw new ShapeError(`adamUpdateM: shape mismatch`, site)
  }
  return addOp(currentGraph(), 'adam_update_m', m.shape, 'f32', site, { m: m.id, g: g.id, b1 })
}

export function adamUpdateV(v: Tensor, g: Tensor, b2: number): Tensor {
  const site = captureSite('adamUpdateV')
  if (v.dtype !== 'f32' || g.dtype !== 'f32') throw new ShapeError(`adamUpdateV: requires f32`, site)
  if (v.shape.length !== g.shape.length || v.shape.some((d, i) => d !== g.shape[i])) {
    throw new ShapeError(`adamUpdateV: shape mismatch`, site)
  }
  return addOp(currentGraph(), 'adam_update_v', v.shape, 'f32', site, { v: v.id, g: g.id, b2 })
}

export function adamUpdateP(
  p: Tensor,
  mNew: Tensor,
  vNew: Tensor,
  lrt: Tensor,
  eps: number,
  decayShrink: number | Tensor = 1,
): Tensor {
  const site = captureSite('adamUpdateP')
  if (p.dtype !== 'f32') throw new ShapeError(`adamUpdateP: requires f32`, site)
  if (lrt.dtype !== 'f32' || lrt.shape.length !== 0) {
    throw new ShapeError(`adamUpdateP: lrt must be a 0-d f32 scalar`, site)
  }
  if (p.shape.length !== mNew.shape.length || p.shape.some((d, i) => d !== mNew.shape[i])) {
    throw new ShapeError(`adamUpdateP: p/mNew shape mismatch`, site)
  }
  // Literal bakes into the kernel; tensor input is updated per step. The
  // kernel binds at most one, chosen by whichever the caller provided.
  const isTensor = typeof decayShrink === 'object'
  if (isTensor) {
    if (decayShrink.dtype !== 'f32' || decayShrink.shape.length !== 0) {
      throw new ShapeError(`adamUpdateP: decayShrink tensor must be a 0-d f32 scalar`, site)
    }
  }
  return addOp(currentGraph(), 'adam_update_p', p.shape, 'f32', site, {
    p: p.id,
    mNew: mNew.id,
    vNew: vNew.id,
    lrt: lrt.id,
    eps,
    decayShrink: isTensor ? 1 : decayShrink,
    decayShrinkTensor: isTensor ? decayShrink.id : null,
  })
}

// ---- 2D convolution and pooling (NCHW; layout matches PyTorch) ------------

export interface Conv2dOptions {
  /** Strides along H and W. Number = same for both. Default 1. */
  stride?: number | readonly [number, number]
  /** Per-side padding along H and W (zero-padding). Default 0. */
  padding?: number | readonly [number, number]
}

/** Normalize a number-or-pair to a concrete `[number, number]`, applying
 *  `defaultVal` to both dims when omitted. Shared by conv2d / maxPool2d /
 *  nn.Conv2d for stride/padding/kernel handling. */
export function pairOpt(v: number | readonly [number, number] | undefined, defaultVal: number): [number, number] {
  if (v === undefined) return [defaultVal, defaultVal]
  if (typeof v === 'number') return [v, v]
  return [v[0], v[1]]
}

/** 2D convolution. Input [B, C_in, H, W] · weight [C_out, C_in, K_h, K_w]
 *  -> [B, C_out, H_out, W_out]. Bias is added separately via `add`; see
 *  `nn.Conv2d` for the canonical layer wrapper. */
export function conv2d(input: Tensor, weight: Tensor, opts: Conv2dOptions = {}): Tensor {
  const site = captureSite('conv2d')
  if (input.dtype !== 'f32') throw new ShapeError(`conv2d: input must be f32, got ${input.dtype}`, site)
  if (weight.dtype !== 'f32') throw new ShapeError(`conv2d: weight must be f32, got ${weight.dtype}`, site)
  const [sH, sW] = pairOpt(opts.stride, 1)
  const [pH, pW] = pairOpt(opts.padding, 0)
  const outShape = inferConv2d('conv2d', input.shape, weight.shape, sH, sW, pH, pW, site)
  return addOp(currentGraph(), 'conv2d', outShape, 'f32', site, {
    input: input.id, weight: weight.id,
    strideH: sH, strideW: sW, padH: pH, padW: pW,
  })
}

/** conv2d input-gradient op. Internal: emitted by autograd. */
export function conv2dInputGrad(
  weight: Tensor, dy: Tensor,
  inH: number, inW: number,
  strideH: number, strideW: number, padH: number, padW: number,
): Tensor {
  const site = captureSite('conv2dInputGrad')
  const [cOut, cIn] = [weight.shape[0]!, weight.shape[1]!]
  const B = dy.shape[0]!
  const targetShape: Shape = [B, cIn, inH, inW]
  if (dy.shape[1] !== cOut) {
    throw new ShapeError(`conv2dInputGrad: dy's C=${dy.shape[1]} doesn't match weight C_out=${cOut}`, site)
  }
  return addOp(currentGraph(), 'conv2d_input_grad', targetShape, 'f32', site, {
    weight: weight.id, dy: dy.id, inH, inW, strideH, strideW, padH, padW,
  })
}

/** conv2d weight-gradient op. Internal: emitted by autograd. */
export function conv2dWeightGrad(
  input: Tensor, dy: Tensor,
  kH: number, kW: number,
  strideH: number, strideW: number, padH: number, padW: number,
): Tensor {
  const site = captureSite('conv2dWeightGrad')
  const cIn = input.shape[1]!
  const cOut = dy.shape[1]!
  const targetShape: Shape = [cOut, cIn, kH, kW]
  return addOp(currentGraph(), 'conv2d_weight_grad', targetShape, 'f32', site, {
    input: input.id, dy: dy.id, kH, kW, strideH, strideW, padH, padW,
  })
}

export interface MaxPool2dOptions {
  /** Strides along H and W. Default = kernel size (non-overlapping). */
  stride?: number | readonly [number, number]
  /** Per-side padding along H and W. Default 0. */
  padding?: number | readonly [number, number]
}

/** 2D max pooling. Input [B, C, H, W] -> [B, C, H_out, W_out]. Padded regions
 *  are treated as -inf for argmax. `kernelSize` is `[K_h, K_w]`; pass a number
 *  for a square kernel. Default stride equals kernel size (non-overlapping).
 *  PyTorch's `F.max_pool2d` convention. */
export function maxPool2d(input: Tensor, kernelSize: number | readonly [number, number], opts: MaxPool2dOptions = {}): Tensor {
  const site = captureSite('maxPool2d')
  if (input.dtype !== 'f32') throw new ShapeError(`maxPool2d: input must be f32, got ${input.dtype}`, site)
  const [kH, kW] = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : [kernelSize[0], kernelSize[1]]
  const [strideH, strideW] = opts.stride === undefined
    ? [kH, kW]
    : typeof opts.stride === 'number' ? [opts.stride, opts.stride]
    : [opts.stride[0], opts.stride[1]]
  const [pH, pW] = pairOpt(opts.padding, 0)
  const outShape = inferMaxPool2d('maxPool2d', input.shape, kH, kW, strideH, strideW, pH, pW, site)
  return addOp(currentGraph(), 'max_pool_2d', outShape, 'f32', site, {
    input: input.id, kH, kW, strideH, strideW, padH: pH, padW: pW,
  })
}

/** Nearest-neighbor 2D upsample by an integer factor. Input `[B, C, H, W]`
 *  → `[B, C, H*factor, W*factor]`. Each input pixel is replicated to a
 *  `factor × factor` block. Composes from reshape + broadcast + reshape —
 *  no new IR kernel. Pair with `conv2d` for upsample-then-conv (the modern
 *  alternative to transposed convolution in image-generation U-Nets). */
export function nearestUpsample2d(x: Tensor, factor: number): Tensor {
  const site = captureSite('nearestUpsample2d')
  if (x.shape.length !== 4) {
    throw new ShapeError(`nearestUpsample2d: input must be rank-4 [B, C, H, W], got ${showShape(x.shape)}`, site)
  }
  if (!Number.isInteger(factor) || factor < 1) {
    throw new ShapeError(`nearestUpsample2d: factor must be a positive integer, got ${factor}`, site)
  }
  if (factor === 1) return x
  const [B, C, H, W] = x.shape as [number, number, number, number]
  const expanded = broadcastTo(reshape(x, [B, C, H, 1, W, 1]), [B, C, H, factor, W, factor])
  return reshape(expanded, [B, C, H * factor, W * factor])
}

/** max_pool_2d backward op. Internal: emitted by autograd. Recomputes the
 *  argmax on the fly to avoid needing a saved-indices buffer. */
export function maxPool2dGrad(
  input: Tensor, dy: Tensor,
  kH: number, kW: number, strideH: number, strideW: number, padH: number, padW: number,
): Tensor {
  const site = captureSite('maxPool2dGrad')
  return addOp(currentGraph(), 'max_pool_2d_grad', input.shape, 'f32', site, {
    input: input.id, dy: dy.id, kH, kW, strideH, strideW, padH, padW,
  })
}
