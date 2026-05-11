// User-facing op surface.
//
// Each function here is a thin wrapper:
//   1. capture the call site (for error attribution)
//   2. validate input shapes via src/shape.ts (which throws on mismatch)
//   3. compute the output shape and dtype
//   4. append the op to the current Graph (held in module state by src/trace.ts)
//   5. return the produced Tensor handle
//
// No actual numeric work happens here. These calls just build the IR.

import type { Tensor, Shape, Dtype, OpNode, Graph } from './ir.js'
import { addOp, captureSite } from './ir.js'
import { currentGraph, tensorInput } from './trace.js'
import {
  inferElementwiseBinop, inferUnary, inferMeanLast, inferSumLast, inferArgmaxLast,
  inferReshape, inferTranspose, inferMatmul, inferMatmulBatched,
  inferOneHot, inferWhereCausal, inferSliceRange, inferConcat,
  inferBroadcastTo, inferSumToShape, inferReluGrad, inferWhere,
  inferConv2d, inferMaxPool2d,
  ShapeError, showShape,
} from './shape.js'

// ----------------------------------------------------------------------------
// Element-wise binops (add/sub/mul/div). Trailing-suffix broadcast.
// ----------------------------------------------------------------------------

/**
 * Build an element-wise binop op (forward declaration only — appends to the
 * graph). Used by both arithmetic ops (add/sub/mul/div, output dtype = input
 * dtype) and comparisons (less/greater, output dtype = bool).
 */
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

// Element-wise binops. Second arg can be a Tensor or a JS number; the latter
// dispatches to scalar-fused IR ops internally. `mul(x, 2)` and `mul(x, y)`
// both work — matches every NumPy-shaped library.
export function add(a: Tensor, b: Tensor | number): Tensor {
  return typeof b === 'number' ? addScalar(a, b) : binopOp('add', 'add', a, b)
}
export function sub(a: Tensor, b: Tensor | number): Tensor {
  return typeof b === 'number' ? addScalar(a, -b) : binopOp('sub', 'sub', a, b)
}
export function mul(a: Tensor, b: Tensor | number): Tensor {
  return typeof b === 'number' ? mulScalar(a, b) : binopOp('mul', 'mul', a, b)
}
export function div(a: Tensor, b: Tensor | number): Tensor {
  if (typeof b === 'number') {
    if (b === 0) throw new ShapeError(`div: scalar divisor cannot be zero`, captureSite('div'))
    return mulScalar(a, 1 / b)
  }
  return binopOp('div', 'div', a, b)
}

export function min(a: Tensor, b: Tensor | number): Tensor {
  const rhs = typeof b === 'number' ? constScalar(b, a.dtype) : b
  return binopOp('min', 'min', a, rhs)
}
export function max(a: Tensor, b: Tensor | number): Tensor {
  const rhs = typeof b === 'number' ? constScalar(b, a.dtype) : b
  return binopOp('max', 'max', a, rhs)
}

/** Element-wise clamp: `min(max(a, lo), hi)`. Both bounds are scalars. */
export function clamp(a: Tensor, lo: number, hi: number): Tensor {
  return min(max(a, lo), hi)
}

// ----------------------------------------------------------------------------
// Element-wise scalar binops (mul/add by JS number). Used for things like
// `scores * (1/sqrt(d))` and `logits + 1e-5` where allocating a 0-d tensor
// for the scalar is wasteful.
// ----------------------------------------------------------------------------

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

type UnaryKind = 'sqrt' | 'rsqrt' | 'log' | 'exp' | 'relu' | 'neg' | 'abs' | 'tanh' | 'sigmoid'

function unary(name: UnaryKind, a: Tensor): Tensor {
  const site = captureSite(name)
  if (a.dtype !== 'f32') throw new ShapeError(`${name}: requires f32, got ${a.dtype}`, site)
  return addOp(currentGraph(), name, inferUnary(name, a.shape, site), 'f32', site, { a: a.id })
}

export const sqrt    = (a: Tensor): Tensor => unary('sqrt',    a)
export const rsqrt   = (a: Tensor): Tensor => unary('rsqrt',   a)
export const log     = (a: Tensor): Tensor => unary('log',     a)
export const exp     = (a: Tensor): Tensor => unary('exp',     a)
export const relu    = (a: Tensor): Tensor => unary('relu',    a)
export const neg     = (a: Tensor): Tensor => unary('neg',     a)
export const abs     = (a: Tensor): Tensor => unary('abs',     a)
export const tanh    = (a: Tensor): Tensor => unary('tanh',    a)
export const sigmoid = (a: Tensor): Tensor => unary('sigmoid', a)

/** SiLU / Swish: `x * sigmoid(x)`. Composed from primitives. */
export function silu(a: Tensor): Tensor {
  return mul(a, sigmoid(a))
}

/** Hidden tensor_input name for the per-step dropout RNG. The runtime
 *  auto-injects this scalar before each step()/run() when the compiled
 *  graph contains any `dropout` op; users do not pass it. Exposed as a
 *  named constant so worker + proxy agree on the convention. */
export const DROPOUT_SEED_INPUT = '__dropoutSeed'

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
 *  `compileModule` / `compileForward` pair. */
export function dropout(x: Tensor, p: number): Tensor {
  if (p === 0) return x
  const site = captureSite('dropout')
  if (p < 0 || p >= 1) throw new ShapeError(`dropout: p must be in [0, 1), got ${p}`, site)
  if (x.dtype !== 'f32') throw new ShapeError(`dropout: requires f32, got ${x.dtype}`, site)
  const g = currentGraph()
  const seed = findOrCreateDropoutSeed(g)
  // Salt = count of existing dropout ops in this graph. Stable per call —
  // doesn't shift when unrelated ops get added/removed elsewhere, since
  // we count only `dropout` kinds.
  const salt = countDropoutOps(g)
  return addOp(g, 'dropout', inferUnary('dropout', x.shape, site), 'f32', site, { a: x.id, seed: seed.id, p, salt })
}

/** Internal: emit a `dropout` op with an explicit salt. Used by grad.ts to
 *  emit the backward kernel using the same (seed, salt, p) as the forward,
 *  so the mask matches. */
export function dropoutWithSalt(dy: Tensor, p: number, salt: number, seedId: number): Tensor {
  const site = captureSite('dropout')
  return addOp(currentGraph(), 'dropout', dy.shape, 'f32', site, { a: dy.id, seed: seedId, p, salt })
}

function findOrCreateDropoutSeed(g: Graph): Tensor {
  for (const op of g.ops) {
    if (op.kind === 'tensor_input' && op.name === DROPOUT_SEED_INPUT) {
      return g.tensors[op.out]!
    }
  }
  return tensorInput(DROPOUT_SEED_INPUT, [], 'i32')
}

function countDropoutOps(g: Graph): number {
  let n = 0
  for (const op of g.ops) if (op.kind === 'dropout') n++
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

// ----------------------------------------------------------------------------
// Reductions.
//
// Public surface: `mean(x, axis?, opts?)` and `sum(x, axis?, opts?)`. Both
// follow PyTorch's `dim=` convention: negative indices count from the end,
// `axis` omitted reduces all axes to a 0-d scalar, and `keepDims=false` is
// the default (the axis is removed from the output shape).
//
// The IR-level kernels (`mean_last`, `sum_last`) are last-axis only. Other
// axes compose as `transpose-axis-to-end` + `*_last` + (reshape or
// transpose-back), so there's no new codegen for arbitrary-axis reduction.
// `meanLastIR` / `sumLastIR` are local helpers — not part of the public
// API; consumers always go through `mean`/`sum`.
// ----------------------------------------------------------------------------

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
  return argmaxLastIR(transpose(a, perm))
}

/** Mean along an axis (or all axes). Negative axis counts from the end.
 *  `keepDims` (default false) preserves the reduced axis as size 1.
 *  With no `axis`, reduces all elements to a 0-d scalar.
 *
 *  ```
 *  mean(x)                          // 0-d scalar
 *  mean(x, -1)                      // PyTorch's x.mean(dim=-1)
 *  mean(x, -1, { keepDims: true })  // preserve the trailing axis as size 1
 *  mean(x, 1)                       // reduce middle axis (transposes internally)
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
  // Reduce: produce a tensor with the reduced axis at position r-1 (size 1
  // for mean, dropped for sum). Transpose first when k isn't already last.
  const input = isLast ? a : transpose(a, [...Array(r).keys()].filter(i => i !== k).concat(k))
  if (kind === 'mean') {
    const reduced = meanLastIR(input)  // shape: [...others, 1]
    if (isLast) return keepDims ? reduced : reshape(reduced, reduced.shape.slice(0, -1))
    if (!keepDims) return reshape(reduced, reduced.shape.slice(0, -1))
    return transpose(reduced, backPerm(k, r))
  }
  // sum
  const dropped = sumLastIR(input)  // shape: [...others]
  if (isLast) return keepDims ? reshape(dropped, [...dropped.shape, 1]) : dropped
  if (!keepDims) return dropped
  return transpose(reshape(dropped, [...dropped.shape, 1]), backPerm(k, r))
}

/** Inverse of `[everyone-but-k, k]`: the perm that moves the trailing axis
 *  back to original position k. Used for `keepDims=true` with non-last axis. */
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
  if (s === r) return a  // no axes to collapse
  return reshape(a, [...a.shape.slice(0, s), -1])
}

export function transpose(a: Tensor, perm: readonly number[]): Tensor {
  const site = captureSite('transpose')
  const outShape = inferTranspose('transpose', a.shape, perm, site)
  return addOp(currentGraph(), 'transpose', outShape, a.dtype, site, { a: a.id, perm })
}

/** Swap two axes of a tensor. Negative indices count from the end (so
 *  `swapAxes(x, -1, -2)` swaps the last two — the common attention pattern).
 *  All other axes keep their position. Implemented as `transpose` with the
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
  return transpose(a, perm)
}

// ----------------------------------------------------------------------------
// Linear algebra.
// ----------------------------------------------------------------------------

export function matmul(a: Tensor, b: Tensor): Tensor {
  const site = captureSite('matmul')
  if (a.dtype !== 'f32' || b.dtype !== 'f32') {
    throw new ShapeError(`matmul: requires f32, got ${a.dtype} and ${b.dtype}`, site)
  }
  const outShape = inferMatmul('matmul', a.shape, b.shape, site)
  return addOp(currentGraph(), 'matmul', outShape, 'f32', site, { a: a.id, b: b.id })
}

export function matmulBatched(a: Tensor, b: Tensor): Tensor {
  const site = captureSite('matmulBatched')
  if (a.dtype !== 'f32' || b.dtype !== 'f32') {
    throw new ShapeError(`matmulBatched: requires f32, got ${a.dtype} and ${b.dtype}`, site)
  }
  const outShape = inferMatmulBatched('matmulBatched', a.shape, b.shape, site)
  return addOp(currentGraph(), 'matmul_batched', outShape, 'f32', site, { a: a.id, b: b.id })
}

// ----------------------------------------------------------------------------
// Indexing / casting.
// ----------------------------------------------------------------------------

export function oneHot(indices: Tensor, depth: number, dtype: Dtype = 'f32'): Tensor {
  const site = captureSite('oneHot')
  if (indices.dtype !== 'i32') {
    throw new ShapeError(`oneHot: indices must be i32, got ${indices.dtype}`, site)
  }
  const outShape = inferOneHot('oneHot', indices.shape, depth, site)
  return addOp(currentGraph(), 'one_hot', outShape, dtype, site, { indices: indices.id, depth, dtype })
}

/** Embedding lookup: pull rows from `table` indexed by `indices`. Decomposes
 *  to `oneHot(indices, vocab) @ table` so autograd works without a dedicated
 *  scatter-with-atomic-add backward — the matmul transpose rule handles it.
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

// arange(n) → [n] of values [0, 1, ..., n-1]. Used for position embeddings.
export function arange(n: number, dtype: Dtype = 'i32'): Tensor {
  const site = captureSite('arange')
  if (n <= 0 || !Number.isInteger(n)) {
    throw new ShapeError(`arange: n must be a positive integer, got ${n}`, site)
  }
  return addOp(currentGraph(), 'arange', [n], dtype, site, { n, dtype })
}

// ----------------------------------------------------------------------------
// ML primitives. Fused so autograd's transpose rule is straightforward and the
// kernels can be hand-tuned for our specific shapes.
// ----------------------------------------------------------------------------

// Causal-masked softmax along the last axis. Shape preserved. Last two
// axes must be square (TxT attention scores). Always last-axis by
// construction — the causal mask is over a sequence-by-sequence matrix.
export function softmaxCausal(a: Tensor): Tensor {
  const site = captureSite('softmaxCausal')
  if (a.dtype !== 'f32') throw new ShapeError(`softmaxCausal: requires f32, got ${a.dtype}`, site)
  inferWhereCausal('softmaxCausal', a.shape, site)
  return addOp(currentGraph(), 'softmax_causal_last', a.shape, 'f32', site, { a: a.id })
}

function logSoftmaxLastIR(a: Tensor): Tensor {
  const site = captureSite('logSoftmax')
  if (a.dtype !== 'f32') throw new ShapeError(`logSoftmax: requires f32, got ${a.dtype}`, site)
  return addOp(currentGraph(), 'log_softmax_last', a.shape, 'f32', site, { a: a.id })
}

/** Numerically-stable log-softmax along an axis. Shape preserved.
 *  Negative axis counts from the end; default is `-1`. Non-last axes
 *  transpose internally. */
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

/** Internal helper: apply a last-axis op `f` along an arbitrary axis by
 *  transposing the axis to last, applying `f`, then transposing back.
 *  Output shape matches input (axis-preserving op). */
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
  return transpose(applyLast(transpose(a, perm)), backPerm(k, r))
}

// Pre-softmax causal mask. Sets cells where (i < j) on the last two axes to
// `fillValue` (typically -1e30). Lower-triangle entries pass through.
// Use this when you want the masked scores explicitly (e.g. for capture);
// for the common case, prefer softmaxCausal which fuses both.
export function whereCausal(a: Tensor, fillValue: number): Tensor {
  const site = captureSite('whereCausal')
  if (a.dtype !== 'f32') throw new ShapeError(`whereCausal: requires f32, got ${a.dtype}`, site)
  inferWhereCausal('whereCausal', a.shape, site)
  return addOp(currentGraph(), 'where_causal', a.shape, 'f32', site, { a: a.id, fillValue })
}

// ----------------------------------------------------------------------------
// Slicing.
// ----------------------------------------------------------------------------

/** General-axis slice: take elements `[start, end)` along `axis`. Negative
 *  `axis` indexes from the end (Python convention). */
export function sliceRange(a: Tensor, axis: number, start: number, end: number): Tensor {
  const site = captureSite('sliceRange')
  const ax = axis < 0 ? a.shape.length + axis : axis
  const outShape = inferSliceRange('sliceRange', a.shape, ax, start, end, site)
  return addOp(currentGraph(), 'slice_range', outShape, a.dtype, site, { a: a.id, axis: ax, start, end })
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

// ----------------------------------------------------------------------------
// Broadcast / un-broadcast. Mostly used by autograd, but exposed in case user
// code needs them (e.g. explicit broadcasting for clarity).
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Constants.
// ----------------------------------------------------------------------------

// 0-d tensor with a constant value. Used by autograd to seed the loss cotangent.
export function constScalar(value: number, dtype: Dtype = 'f32'): Tensor {
  const site = captureSite('constScalar')
  return addOp(currentGraph(), 'const_scalar', [], dtype, site, { value, dtype })
}

// ----------------------------------------------------------------------------
// Autograd-internal helpers (exposed for users writing custom transpose rules).
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Comparisons and selection.
// ----------------------------------------------------------------------------

// Comparisons reuse the binop helper but return bool. Second arg accepts a
// Tensor or a JS number — the scalar form composes constScalar with the binop's
// broadcasting, parallel to add/sub/mul/div's scalar overload.
export function less(a: Tensor, b: Tensor | number): Tensor {
  const rhs = typeof b === 'number' ? constScalar(b, a.dtype) : b
  return binopOp('less', 'less', a, rhs, 'bool')
}
export function greater(a: Tensor, b: Tensor | number): Tensor {
  const rhs = typeof b === 'number' ? constScalar(b, a.dtype) : b
  return binopOp('greater', 'greater', a, rhs, 'bool')
}

// where(cond, a, b): elementwise select. cond is bool; a and b can be any matching dtype.
export function where(cond: Tensor, a: Tensor, b: Tensor): Tensor {
  const site = captureSite('where')
  if (cond.dtype !== 'bool') throw new ShapeError(`where: cond must be bool, got ${cond.dtype}`, site)
  if (a.dtype !== b.dtype) throw new ShapeError(`where: a/b dtype mismatch (${a.dtype} vs ${b.dtype})`, site)
  const outShape = inferWhere('where', cond.shape, a.shape, b.shape, site)
  return addOp(currentGraph(), 'where', outShape, a.dtype, site, { cond: cond.id, a: a.id, b: b.id })
}

// reluGrad(x, dy) = dy where x > 0, else 0. Same shape as x. This is the
// transpose rule for relu, exposed as an op so codegen can emit it.
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
  // decayShrink is either a literal (baked into the kernel) or a 0-d scalar
  // tensor input the runtime updates per step. The kernel binds at most one,
  // chosen by whichever the caller provided.
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

// ----------------------------------------------------------------------------
// 2D convolution and pooling (NCHW). Layout matches PyTorch so 1-shot ports
// don't need a transpose. Bias is added separately (via `add` + broadcast) so
// the IR op stays pure; see `nn.Conv2d` for the standard wrapper.
// ----------------------------------------------------------------------------

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
  // Output shape matches the original conv2d input.
  const targetShape: Shape = [B, cIn, inH, inW]
  // Validate dy's channel count matches weight's C_out.
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
  // Output shape matches the original conv2d weight.
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
  // Default stride = kernel size (non-overlapping pooling, matching PyTorch).
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

/** max_pool_2d backward op. Internal: emitted by autograd. Recomputes the
 *  argmax on the fly to avoid needing a saved-indices buffer. */
export function maxPool2dGrad(
  input: Tensor, dy: Tensor,
  kH: number, kW: number, strideH: number, strideW: number, padH: number, padW: number,
): Tensor {
  const site = captureSite('maxPool2dGrad')
  // Output shape matches `input`.
  return addOp(currentGraph(), 'max_pool_2d_grad', input.shape, 'f32', site, {
    input: input.id, dy: dy.id, kH, kW, strideH, strideW, padH, padW,
  })
}
