// Reverse-mode autograd. Walks a traced Graph in reverse and appends backward
// ops in-place via traceInto. Each adjoint rule expresses its contribution as
// regular forward-op calls (e.g. mul(a, b)→c gives da += dc*b, db += dc*a),
// which append to the current graph since we run inside a trace context.
// Cotangents are accumulated as they arrive, so by the time we reach a
// tensor's producer the cotangent sum is complete.

import type { Graph, OpNode, Tensor, Shape } from './ir.js'
import {
  add, sub, mul, div, mulScalar,
  matmul, permute, swapAxes, reshape,
  exp, sin, cos,
  broadcastTo, sumToShape,
  constScalar, reluGrad,
  sum, where, less, greater,
  dropoutWithSalt,
  narrow, scatterAxis, whereCausal,
  conv2dInputGrad, conv2dWeightGrad, maxPool2dGrad,
} from './ops.js'
import { traceInto } from './trace.js'
import { shapesEqual } from './shape.js'

/** Output of `appendGrad`: the graph (now extended with backward ops),
 *  the gradient Tensor for every parameter (keyed by name), and the
 *  scalar loss tensor (echoed from the input graph for convenience). */
export interface GradResult {
  /** The same graph instance passed in, with backward ops appended. */
  readonly graph: Graph
  /** Cotangents (gradients) for each `param_input`, keyed by param name. */
  readonly paramGrads: Record<string, Tensor>
  /** The loss output (unchanged from input). */
  readonly loss: Tensor
}

/**
 * Reverse-mode autograd over a traced `Graph`. Walks the ops in reverse,
 * appending backward ops in-place. The input graph must have a single
 * scalar output (the loss). Returns the same graph (mutated) plus a map
 * from each `param_input` name to its gradient Tensor.
 *
 * Re-enters the graph as the active trace context internally via
 * `traceInto`; callers don't need to manage trace state.
 */
export function appendGrad(graph: Graph): GradResult {
  if (graph.outputs.length !== 1) {
    throw new Error(`autograd: expected graph with exactly 1 output (the loss); got ${graph.outputs.length}`)
  }
  const lossId = graph.outputs[0]!
  const lossTensor = graph.tensors[lossId]!
  if (lossTensor.shape.length !== 0) {
    throw new Error(
      `autograd: loss must be a rank-0 scalar; got shape [${lossTensor.shape.join(', ')}]. ` +
      `Reduce with sum / mulScalar to a scalar before calling appendGrad.`,
    )
  }

  // Snapshot forward ops before emitting backwards so the reverse walk only
  // iterates the original forward subgraph.
  const forwardOpCount = graph.ops.length
  const forwardOps = graph.ops.slice(0, forwardOpCount)

  // tensorId -> the Tensor representing dL/dTensor in the graph.
  const cotangents = new Map<number, Tensor>()

  return traceInto(graph, () => {
    cotangents.set(lossId, constScalar(1.0, 'f32'))

    for (let i = forwardOpCount - 1; i >= 0; i--) {
      const op = forwardOps[i]!
      const outCotan = cotangents.get(op.out)
      if (!outCotan) continue
      runAdjointRule(op, outCotan, graph, cotangents)
    }

    const paramGrads: Record<string, Tensor> = {}
    for (const op of forwardOps) {
      if (op.kind !== 'param_input') continue
      const cotan = cotangents.get(op.out)
      if (!cotan) {
        // No path from this param to the loss — emit explicit zeros so the
        // caller gets a tensor with the right shape.
        const t = graph.tensors[op.out]!
        paramGrads[op.name] = broadcastTo(constScalar(0.0, t.dtype), t.shape)
      } else {
        paramGrads[op.name] = cotan
      }
    }

    return { graph, paramGrads, loss: lossTensor }
  })
}

// Sum into the cotangent of `inputId` (multiple consumers accumulate).
function accumulate(cotangents: Map<number, Tensor>, inputId: number, contribution: Tensor): void {
  const existing = cotangents.get(inputId)
  if (existing) {
    cotangents.set(inputId, add(existing, contribution))
  } else {
    cotangents.set(inputId, contribution)
  }
}

// Reduce a cotangent back to the input's shape, undoing any forward broadcast.
function unbroadcast(cotan: Tensor, toShape: Shape): Tensor {
  if (shapesEqual(cotan.shape, toShape)) return cotan
  return sumToShape(cotan, toShape)
}

// One rule per OpNode kind. Each rule builds backward expressions via the
// ops.ts public functions (which append to the active trace) and accumulates
// cotangent contributions onto each input tensor.

function runAdjointRule(
  op: OpNode,
  outCotan: Tensor,
  graph: Graph,
  cotangents: Map<number, Tensor>,
): void {
  const tensorOf = (id: number) => graph.tensors[id]!

  switch (op.kind) {
    // ---- Leaves: no inputs to accumulate into. -----------------------------
    case 'param_input':
    case 'tensor_input':
    case 'state_input':
    case 'arange':
    case 'const_scalar':
      return

    // ---- Element-wise binops (with broadcast) ------------------------------
    case 'add': {
      const a = tensorOf(op.a), b = tensorOf(op.b)
      accumulate(cotangents, op.a, unbroadcast(outCotan, a.shape))
      accumulate(cotangents, op.b, unbroadcast(outCotan, b.shape))
      return
    }
    case 'sub': {
      const a = tensorOf(op.a), b = tensorOf(op.b)
      accumulate(cotangents, op.a, unbroadcast(outCotan, a.shape))
      accumulate(cotangents, op.b, unbroadcast(mulScalar(outCotan, -1), b.shape))
      return
    }
    case 'mul': {
      // dc/da = b, dc/db = a. Reading the forward tensors via tensorOf is
      // safe — we emit fresh mul() ops, never mutate the originals.
      const a = tensorOf(op.a), b = tensorOf(op.b)
      accumulate(cotangents, op.a, unbroadcast(mul(outCotan, b), a.shape))
      accumulate(cotangents, op.b, unbroadcast(mul(outCotan, a), b.shape))
      return
    }
    case 'div': {
      // dc/da = 1/b. dc/db = -a/b².
      const a = tensorOf(op.a), b = tensorOf(op.b)
      accumulate(cotangents, op.a, unbroadcast(div(outCotan, b), a.shape))
      const numer = mul(outCotan, a)
      const bSq = mul(b, b)
      accumulate(cotangents, op.b, unbroadcast(mulScalar(div(numer, bSq), -1), b.shape))
      return
    }

    // ---- Element-wise scalar binops ---------------------------------------
    case 'mul_scalar': {
      accumulate(cotangents, op.a, mulScalar(outCotan, op.scalar))
      return
    }
    case 'add_scalar': {
      accumulate(cotangents, op.a, outCotan)
      return
    }

    // ---- Unary -------------------------------------------------------------
    case 'sqrt': {
      // dc/da = 1/(2*sqrt(a)) = 1/(2*c).
      const c = tensorOf(op.out)
      accumulate(cotangents, op.a, mulScalar(div(outCotan, c), 0.5))
      return
    }
    case 'rsqrt': {
      // c = a^(-0.5). dc/da = -0.5 * c³.
      const c = tensorOf(op.out)
      const c3 = mul(mul(c, c), c)
      accumulate(cotangents, op.a, mulScalar(mul(outCotan, c3), -0.5))
      return
    }
    case 'log': {
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, div(outCotan, a))
      return
    }
    case 'exp': {
      // dc/da = exp(a) = c.
      const c = tensorOf(op.out)
      accumulate(cotangents, op.a, mul(outCotan, c))
      return
    }
    case 'relu': {
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, reluGrad(a, outCotan))
      return
    }
    case 'neg': {
      accumulate(cotangents, op.a, mulScalar(outCotan, -1))
      return
    }
    case 'abs': {
      // dc/da = sign(a). Subgradient is 0 at a=0 (the where below routes 0
      // there via the outCotan branch — fine in practice).
      const a = tensorOf(op.a)
      const dySigned = where(less(a, constScalar(0, 'f32')), mulScalar(outCotan, -1), outCotan)
      accumulate(cotangents, op.a, dySigned)
      return
    }
    case 'tanh': {
      // dc/da = 1 - c².
      const c = tensorOf(op.out)
      const oneMinusCSq = sub(constScalar(1, 'f32'), mul(c, c))
      accumulate(cotangents, op.a, mul(outCotan, oneMinusCSq))
      return
    }
    case 'sigmoid': {
      // dc/da = c * (1 - c) = c - c².
      const c = tensorOf(op.out)
      const cMinusCSq = sub(c, mul(c, c))
      accumulate(cotangents, op.a, mul(outCotan, cMinusCSq))
      return
    }
    case 'sin': {
      // dc/da = cos(a).
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, mul(outCotan, cos(a)))
      return
    }
    case 'cos': {
      // dc/da = -sin(a).
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, mul(outCotan, mulScalar(sin(a), -1)))
      return
    }
    case 'dropout': {
      // Same kernel, same (seed, salt, p) — the PCG hash reproduces the
      // forward mask. 1/(1-p) scaling is baked into the kernel.
      accumulate(cotangents, op.a, dropoutWithSalt(outCotan, op.p, op.salt, op.seed))
      return
    }
    case 'randn':
      // Samples from a fixed distribution; no differentiable inputs.
      return
    case 'stop_gradient':
      // The whole point: detach the input from the backward pass. Cotangent
      // arrives but doesn't propagate.
      return
    case 'concat': {
      // Slice the gradient back into each input's piece along the concat axis.
      let cursor = 0
      for (const inputId of op.inputs) {
        const inputTensor = tensorOf(inputId)
        const sliceSize = inputTensor.shape[op.axis]!
        accumulate(cotangents, inputId, narrow(outCotan, op.axis, cursor, sliceSize))
        cursor += sliceSize
      }
      return
    }
    case 'slice_range': {
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, scatterAxis(outCotan, a.shape, op.axis, op.start, op.end))
      return
    }
    case 'min': {
      // Pass dy through to whichever side won. Ties go to b (subgradient choice).
      const a = tensorOf(op.a), b = tensorOf(op.b)
      const zero = constScalar(0, 'f32')
      const aWins = less(a, b)
      accumulate(cotangents, op.a, unbroadcast(where(aWins, outCotan, broadcastTo(zero, outCotan.shape)), a.shape))
      accumulate(cotangents, op.b, unbroadcast(where(aWins, broadcastTo(zero, outCotan.shape), outCotan), b.shape))
      return
    }
    case 'max': {
      // Pass dy through to whichever side won. Ties go to b (subgradient choice).
      const a = tensorOf(op.a), b = tensorOf(op.b)
      const zero = constScalar(0, 'f32')
      const aWins = greater(a, b)
      accumulate(cotangents, op.a, unbroadcast(where(aWins, outCotan, broadcastTo(zero, outCotan.shape)), a.shape))
      accumulate(cotangents, op.b, unbroadcast(where(aWins, broadcastTo(zero, outCotan.shape), outCotan), b.shape))
      return
    }

    // ---- Reductions over last axis ---------------------------------------
    case 'mean_last': {
      // outCotan has shape [..., 1]; broadcast to a's shape and divide by D.
      const a = tensorOf(op.a)
      const D = a.shape[a.shape.length - 1]!
      const expanded = broadcastTo(outCotan, a.shape)
      accumulate(cotangents, op.a, mulScalar(expanded, 1 / D))
      return
    }
    case 'sum_last': {
      // sum_last drops the last axis (keepdims=false); add it back as size 1
      // then broadcast to a's shape.
      const a = tensorOf(op.a)
      const withKeep = reshape(outCotan, [...outCotan.shape, 1])
      accumulate(cotangents, op.a, broadcastTo(withKeep, a.shape))
      return
    }

    // ---- Shape ------------------------------------------------------------
    case 'reshape': {
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, reshape(outCotan, a.shape))
      return
    }
    case 'permute': {
      const inv = invertPerm(op.perm)
      accumulate(cotangents, op.a, permute(outCotan, inv))
      return
    }

    // ---- Linear algebra ---------------------------------------------------
    case 'matmul': {
      // a: [..., M, K], b: [K, N]. dA = dC @ B^T, dB = sum_batch(A^T @ dC).
      // The public `matmul` dispatches to the batched kernel for batched
      // operands; sumToShape collapses dB's leading batch dims to [K, N].
      const a = tensorOf(op.a), b = tensorOf(op.b)
      accumulate(cotangents, op.a, matmul(outCotan, swapAxes(b, -1, -2)))
      const aT = swapAxes(a, -1, -2)
      const perBatchDb = matmul(aT, outCotan)
      accumulate(cotangents, op.b, sumToShape(perBatchDb, b.shape))
      return
    }
    case 'matmul_batched': {
      // Per-batch: dA = dC @ B^T, dB = A^T @ dC.
      const a = tensorOf(op.a), b = tensorOf(op.b)
      accumulate(cotangents, op.a, matmul(outCotan, swapAxes(b, -1, -2)))
      accumulate(cotangents, op.b, matmul(swapAxes(a, -1, -2), outCotan))
      return
    }

    // ---- Indexing / casting (no gradient through integer indices) --------
    case 'one_hot':
      return

    // ---- Slicing ---------------------------------------------------------
    case 'slice_last_range': {
      const a = tensorOf(op.a)
      const axis = a.shape.length - 1
      accumulate(cotangents, op.a, scatterAxis(outCotan, a.shape, axis, op.start, op.end))
      return
    }

    // ---- Broadcast / un-broadcast (autograd infrastructure) ---------------
    case 'broadcast_to': {
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, sumToShape(outCotan, a.shape))
      return
    }
    case 'sum_to_shape': {
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, broadcastTo(outCotan, a.shape))
      return
    }

    // ---- ML primitives ---------------------------------------------------
    case 'log_softmax_last': {
      // dL/dA = dL/dC - softmax(a) * sum_last_keepdims(dL/dC).
      const c = tensorOf(op.out)
      const sm = exp(c)
      const sumDcKeep = sum(outCotan, -1, { keepDims: true })
      const term = mul(sm, broadcastTo(sumDcKeep, c.shape))
      accumulate(cotangents, op.a, sub(outCotan, term))
      return
    }
    case 'softmax_causal_last': {
      // c = softmax_causal(a, axis=-1). The causal mask zeros the upper triangle
      // of c; for the backward, the same mask zeros out dx_upper because both
      // paths through softmax depend on c-values that are 0 there.
      // dL/dA = (dL/dC - sum_last_keep(dL/dC * c)) * c
      const c = tensorOf(op.out)
      const dcXc = mul(outCotan, c)
      const sKeep = sum(dcXc, -1, { keepDims: true })
      const inner = sub(outCotan, broadcastTo(sKeep, c.shape))
      accumulate(cotangents, op.a, mul(inner, c))
      return
    }
    // ---- Comparisons + select ---------------------------------------------
    case 'less':
    case 'greater':
      // No gradient flows through bool comparisons. Stop here.
      return

    case 'argmax_last':
      // Non-differentiable: index output is discrete.
      return

    case 'where': {
      // c = where(cond, a, b).
      // dC flows to a where cond is true, to b where cond is false.
      const cond = tensorOf(op.cond)
      const a = tensorOf(op.a)
      const b = tensorOf(op.b)
      const zeroA = broadcastTo(constScalar(0, a.dtype), outCotan.shape)
      const zeroB = broadcastTo(constScalar(0, b.dtype), outCotan.shape)
      accumulate(cotangents, op.a, unbroadcast(where(cond, outCotan, zeroA), a.shape))
      accumulate(cotangents, op.b, unbroadcast(where(cond, zeroB, outCotan), b.shape))
      return
    }

    case 'where_causal':
      // Lower triangle passes through, upper is zeroed — that's whereCausal
      // applied to the cotangent with fillValue=0.
      accumulate(cotangents, op.a, whereCausal(outCotan, 0))
      return

    // ---- Adam ops are post-autograd; no backward through them. ----------
    case 'adam_update_m':
    case 'adam_update_v':
    case 'adam_update_p':
      throw new Error(`autograd: cannot differentiate through ${op.kind}`)

    // ---- Conv2d + MaxPool2d ----------------------------------------------
    case 'conv2d': {
      // dInput = transposed-conv(weight, dy); dWeight = correlation(input, dy).
      // Both are gather kernels reusing the forward's stride/padding params.
      const input = tensorOf(op.input)
      const weight = tensorOf(op.weight)
      const inH = input.shape[2]!
      const inW = input.shape[3]!
      const kH = weight.shape[2]!
      const kW = weight.shape[3]!
      accumulate(cotangents, op.input, conv2dInputGrad(
        weight, outCotan, inH, inW, op.strideH, op.strideW, op.padH, op.padW,
      ))
      accumulate(cotangents, op.weight, conv2dWeightGrad(
        input, outCotan, kH, kW, op.strideH, op.strideW, op.padH, op.padW,
      ))
      return
    }
    case 'max_pool_2d': {
      const input = tensorOf(op.input)
      accumulate(cotangents, op.input, maxPool2dGrad(
        input, outCotan, op.kH, op.kW, op.strideH, op.strideW, op.padH, op.padW,
      ))
      return
    }
    case 'conv2d_input_grad':
    case 'conv2d_weight_grad':
    case 'max_pool_2d_grad':
      throw new Error(`autograd: cannot differentiate through ${op.kind} (it's a backward op)`)

    case 'relu_grad':
      throw new Error(
        `autograd: cannot take second-order gradient through relu_grad — ` +
        `tensorgrad does not support higher-order autodiff.`,
      )

    case 'scatter_axis':
      throw new Error(
        `autograd: cannot differentiate through scatter_axis (it's a backward op)`,
      )

    default: {
      const _exhaustive: never = op
      void _exhaustive
      throw new Error(`autograd: unhandled op kind ${(op as OpNode).kind}`)
    }
  }
}

function invertPerm(perm: readonly number[]): number[] {
  const inv: number[] = new Array(perm.length)
  for (let i = 0; i < perm.length; i++) inv[perm[i]!] = i
  return inv
}
