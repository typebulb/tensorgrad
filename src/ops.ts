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
  inferOneHot, inferWhereCausal, inferSliceLastRange,
  inferBroadcastTo, inferSumToShape, inferReluGrad, inferWhere,
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
// Reductions over the last axis. To reduce along other axes, transpose first.
// (This is intentional — keeps codegen and autograd small.)
// ----------------------------------------------------------------------------

export function meanLast(a: Tensor): Tensor {
  const site = captureSite('meanLast')
  if (a.dtype !== 'f32') throw new ShapeError(`meanLast: requires f32, got ${a.dtype}`, site)
  const outShape = inferMeanLast('meanLast', a.shape, site)
  return addOp(currentGraph(), 'mean_last', outShape, a.dtype, site, { a: a.id })
}

export function sumLast(a: Tensor): Tensor {
  const site = captureSite('sumLast')
  if (a.dtype !== 'f32') throw new ShapeError(`sumLast: requires f32, got ${a.dtype}`, site)
  const outShape = inferSumLast('sumLast', a.shape, site)
  return addOp(currentGraph(), 'sum_last', outShape, a.dtype, site, { a: a.id })
}

/** Reduce all elements to a 0-d scalar. Composes `reshape` + `sumLast`. */
export function sumAll(a: Tensor): Tensor {
  return sumLast(reshape(a, [-1]))
}

/** Index of the maximum value along the last axis. Returns `i32`, one rank
 *  less than input (keepdims=false). Non-differentiable: gradients do not
 *  flow back through this op. Useful for greedy decode, accuracy probes,
 *  and any in-graph classification where you don't want a CPU readback. */
export function argmaxLast(a: Tensor): Tensor {
  const site = captureSite('argmaxLast')
  if (a.dtype !== 'f32') throw new ShapeError(`argmaxLast: requires f32, got ${a.dtype}`, site)
  const outShape = inferArgmaxLast('argmaxLast', a.shape, site)
  return addOp(currentGraph(), 'argmax_last', outShape, 'i32', site, { a: a.id })
}

/** Mean of every element, returning a 0-d scalar. Equivalent to
 *  `mul(sumAll(a), 1 / numel(a))` but spells the intent. The standard tail
 *  of any scalar loss. */
export function meanAll(a: Tensor): Tensor {
  const n = a.shape.reduce((p, d) => p * d, 1)
  if (n === 0) throw new ShapeError(`meanAll: cannot mean over zero elements`, captureSite('meanAll'))
  return mulScalar(sumAll(a), 1 / n)
}

// ----------------------------------------------------------------------------
// Shape ops.
// ----------------------------------------------------------------------------

export function reshape(a: Tensor, newShape: Shape): Tensor {
  const site = captureSite('reshape')
  const outShape = inferReshape('reshape', a.shape, newShape, site)
  return addOp(currentGraph(), 'reshape', outShape, a.dtype, site, { a: a.id, newShape: outShape })
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
export function embedding(table: Tensor, indices: Tensor): Tensor {
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

// Causal-masked softmax along the last axis. Shape preserved. Last two axes
// must be square (TxT attention scores).
export function softmaxCausalLast(a: Tensor): Tensor {
  const site = captureSite('softmaxCausalLast')
  if (a.dtype !== 'f32') throw new ShapeError(`softmaxCausalLast: requires f32, got ${a.dtype}`, site)
  inferWhereCausal('softmaxCausalLast', a.shape, site)  // shape check (square last 2 axes)
  return addOp(currentGraph(), 'softmax_causal_last', a.shape, 'f32', site, { a: a.id })
}

// Numerically-stable log-softmax along the last axis. Shape preserved.
export function logSoftmaxLast(a: Tensor): Tensor {
  const site = captureSite('logSoftmaxLast')
  if (a.dtype !== 'f32') throw new ShapeError(`logSoftmaxLast: requires f32, got ${a.dtype}`, site)
  return addOp(currentGraph(), 'log_softmax_last', a.shape, 'f32', site, { a: a.id })
}

/** Numerically-stable softmax along the last axis. Composes `exp` with
 *  `logSoftmaxLast` — the stabilization happens inside the fused log-softmax
 *  kernel. Output is `[..., V]` of probabilities summing to 1 along the last
 *  axis. For classifiers that want explicit class probabilities, and for
 *  visualization of attention-like distributions. Use `crossEntropyLast` if
 *  you actually want the training loss. */
export function softmaxLast(a: Tensor): Tensor {
  return exp(logSoftmaxLast(a))
}

// Pre-softmax causal mask. Sets cells where (i < j) on the last two axes to
// `fillValue` (typically -1e30). Lower-triangle entries pass through.
// Use this when you want the masked scores explicitly (e.g. for capture);
// for the common case, prefer softmaxCausalLast which fuses both.
export function whereCausal(a: Tensor, fillValue: number): Tensor {
  const site = captureSite('whereCausal')
  if (a.dtype !== 'f32') throw new ShapeError(`whereCausal: requires f32, got ${a.dtype}`, site)
  inferWhereCausal('whereCausal', a.shape, site)
  return addOp(currentGraph(), 'where_causal', a.shape, 'f32', site, { a: a.id, fillValue })
}

// ----------------------------------------------------------------------------
// Slicing.
// ----------------------------------------------------------------------------

// sliceLastRange(a, start, end): slice [start, end) along the last axis.
// Used for splitting Q/K/V from a fused QKV matmul.
export function sliceLastRange(a: Tensor, start: number, end: number): Tensor {
  const site = captureSite('sliceLastRange')
  const outShape = inferSliceLastRange('sliceLastRange', a.shape, start, end, site)
  return addOp(currentGraph(), 'slice_last_range', outShape, a.dtype, site, { a: a.id, start, end })
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
