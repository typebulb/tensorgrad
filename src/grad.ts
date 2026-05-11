// Reverse-mode autograd over a traced Graph.
//
// Given a graph that ends in a scalar loss tensor, this module walks the ops
// in reverse and appends backward ops to the same graph, computing dL/dT for
// every Tensor T that descends from a `param_input`. The final cotangents on
// the param_input tensors are the parameter gradients.
//
// Cotangent accumulation: a tensor with multiple consumers ends up with
// contributions from each. We add them as we encounter them, so by the time
// reverse iteration reaches a tensor's producer op, its cotangent is complete.
//
// Why this works as "more graph nodes": the transpose rule for an op like
// mul(a, b)→c is `da += dc * b; db += dc * a`. The right-hand sides are
// expressible in terms of existing forward ops (mul) plus accumulation (add).
// We just call those op functions, which append nodes to the current graph
// because we run inside an active trace context.

import type { Graph, OpNode, Tensor, Shape } from './ir.js'
import {
  add, sub, mul, div, mulScalar,
  matmul, matmulBatched, transpose, swapAxes, reshape,
  exp,
  broadcastTo, sumToShape,
  constScalar, reluGrad,
  sumLast, where, less, greater,
  dropoutWithSalt,
} from './ops.js'
import { traceInto } from './trace.js'
import { shapesEqual } from './shape.js'

// ============================================================================
// Public API
// ============================================================================

export interface GradResult {
  // The graph, augmented with backward ops.
  readonly graph: Graph
  // Cotangents (gradients) for each param_input, keyed by param name.
  readonly paramGrads: Record<string, Tensor>
  // The loss output (unchanged from input).
  readonly loss: Tensor
}

// `appendGrad(graph)` augments `graph` (which must have already been built by
// `trace(...)` and must have a single scalar output = the loss) with backward
// ops. Returns gradients for every param_input.
//
// Internally re-enters the graph as the active trace context, so backward ops
// emitted by transpose rules append to it. The caller doesn't need to manage
// trace state.
export function appendGrad(graph: Graph): GradResult {
  if (graph.outputs.length !== 1) {
    throw new Error(`autograd: expected graph with exactly 1 output (the loss); got ${graph.outputs.length}`)
  }
  const lossId = graph.outputs[0]!
  const lossTensor = graph.tensors[lossId]!
  if (lossTensor.shape.length !== 0) {
    throw new Error(
      `autograd: loss must be a rank-0 scalar; got shape [${lossTensor.shape.join(', ')}]. ` +
      `Reduce with sumLast / mulScalar to a scalar before calling appendGrad.`,
    )
  }

  // Snapshot the forward portion of the graph before we start emitting backward
  // ops, so the reverse walk only iterates over forward ops.
  const forwardOpCount = graph.ops.length
  const forwardOps = graph.ops.slice(0, forwardOpCount)

  // cotangents: tensorId -> the Tensor representing dL/dTensor in the graph.
  const cotangents = new Map<number, Tensor>()

  return traceInto(graph, () => {
    // Seed: dL/dLoss = 1.0
    cotangents.set(lossId, constScalar(1.0, 'f32'))

    // Reverse walk.
    for (let i = forwardOpCount - 1; i >= 0; i--) {
      const op = forwardOps[i]!
      const outCotan = cotangents.get(op.out)
      if (!outCotan) continue
      runTransposeRule(op, outCotan, graph, cotangents)
    }

    // Collect param gradients by name. Skip non-param leaves.
    const paramGrads: Record<string, Tensor> = {}
    for (const op of forwardOps) {
      if (op.kind !== 'param_input') continue
      // (state_input and tensor_input don't produce gradients we hand back.)
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

// ============================================================================
// Cotangent accumulation
// ============================================================================

// Add `contribution` to the cotangent of tensor `inputId`. If a cotangent
// already exists, sum them (multiple consumers); otherwise initialize.
function accumulate(cotangents: Map<number, Tensor>, inputId: number, contribution: Tensor): void {
  const existing = cotangents.get(inputId)
  if (existing) {
    cotangents.set(inputId, add(existing, contribution))
  } else {
    cotangents.set(inputId, contribution)
  }
}

// Reduce a cotangent to match the input's shape, undoing any broadcast that
// occurred during forward. If `fromShape == toShape`, no-op.
function unbroadcast(cotan: Tensor, toShape: Shape): Tensor {
  if (shapesEqual(cotan.shape, toShape)) return cotan
  return sumToShape(cotan, toShape)
}


// ============================================================================
// Transpose rules
// ============================================================================
//
// One per OpNode kind. Each rule:
//   * receives the forward op + its output cotangent
//   * builds the backward expression(s) in graph terms (calling ops.ts functions)
//   * accumulates cotangent contributions onto each input tensor

function runTransposeRule(
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
    // c = a op b; reduce cotan back to each operand's shape.
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
      const a = tensorOf(op.a), b = tensorOf(op.b)
      // dC/dA = b ; dC/dB = a. Both are forward tensors still alive in the graph.
      // We must NOT consume the forward tensors — they're referenced by id.
      // The mul() helper allocates fresh tensors, so referencing a/b multiple
      // times in different mul() calls is fine: we just emit fresh ops.
      accumulate(cotangents, op.a, unbroadcast(mul(outCotan, b), a.shape))
      accumulate(cotangents, op.b, unbroadcast(mul(outCotan, a), b.shape))
      return
    }
    case 'div': {
      // c = a/b. dc/da = 1/b. dc/db = -a/b^2.
      const a = tensorOf(op.a), b = tensorOf(op.b)
      accumulate(cotangents, op.a, unbroadcast(div(outCotan, b), a.shape))
      // -outCotan * a / (b*b)
      const numer = mul(outCotan, a)
      const bSq = mul(b, b)
      accumulate(cotangents, op.b, unbroadcast(mulScalar(div(numer, bSq), -1), b.shape))
      return
    }

    // ---- Element-wise scalar binops (scalar is a JS number, not a tensor) -
    case 'mul_scalar': {
      // c = a * s. dc/da = s.
      accumulate(cotangents, op.a, mulScalar(outCotan, op.scalar))
      return
    }
    case 'add_scalar': {
      // c = a + s. dc/da = 1.
      accumulate(cotangents, op.a, outCotan)
      return
    }

    // ---- Unary -------------------------------------------------------------
    case 'sqrt': {
      // c = sqrt(a). dc/da = 1/(2*sqrt(a)) = 1/(2*c).
      const c = tensorOf(op.out)
      accumulate(cotangents, op.a, mulScalar(div(outCotan, c), 0.5))
      return
    }
    case 'rsqrt': {
      // c = a^(-0.5). dc/da = -0.5 * a^(-1.5) = -0.5 * c^3.
      const c = tensorOf(op.out)
      const c3 = mul(mul(c, c), c)
      accumulate(cotangents, op.a, mulScalar(mul(outCotan, c3), -0.5))
      return
    }
    case 'log': {
      // c = log(a). dc/da = 1/a.
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, div(outCotan, a))
      return
    }
    case 'exp': {
      // c = exp(a). dc/da = exp(a) = c.
      const c = tensorOf(op.out)
      accumulate(cotangents, op.a, mul(outCotan, c))
      return
    }
    case 'relu': {
      // c = relu(a). dc/da = (a > 0 ? 1 : 0). Use the fused relu_grad op.
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, reluGrad(a, outCotan))
      return
    }
    case 'neg': {
      // c = -a. dc/da = -1.
      accumulate(cotangents, op.a, mulScalar(outCotan, -1))
      return
    }
    case 'abs': {
      // c = |a|. dc/da = sign(a) (subgradient 0 at a=0, fine in practice).
      const a = tensorOf(op.a)
      const dySigned = where(less(a, constScalar(0, 'f32')), mulScalar(outCotan, -1), outCotan)
      accumulate(cotangents, op.a, dySigned)
      return
    }
    case 'tanh': {
      // c = tanh(a). dc/da = 1 - c² = (1 - c) * (1 + c).
      const c = tensorOf(op.out)
      const oneMinusCSq = sub(constScalar(1, 'f32'), mul(c, c))
      accumulate(cotangents, op.a, mul(outCotan, oneMinusCSq))
      return
    }
    case 'sigmoid': {
      // c = sigmoid(a). dc/da = c * (1 - c) = c - c².
      const c = tensorOf(op.out)
      const cMinusCSq = sub(c, mul(c, c))
      accumulate(cotangents, op.a, mul(outCotan, cMinusCSq))
      return
    }
    case 'dropout': {
      // Same kernel applied to dy with the same (seed, salt, p) — the PCG
      // hash reproduces the forward mask. The 1/(1-p) scale is already
      // baked into the dropout kernel.
      accumulate(cotangents, op.a, dropoutWithSalt(outCotan, op.p, op.salt, op.seed))
      return
    }
    case 'min': {
      // c = min(a, b). Pass dy through to whichever side won (a if a <= b).
      const a = tensorOf(op.a), b = tensorOf(op.b)
      const zero = constScalar(0, 'f32')
      const aWins = less(a, b)  // ties go to b; subgradient choice
      accumulate(cotangents, op.a, unbroadcast(where(aWins, outCotan, broadcastTo(zero, outCotan.shape)), a.shape))
      accumulate(cotangents, op.b, unbroadcast(where(aWins, broadcastTo(zero, outCotan.shape), outCotan), b.shape))
      return
    }
    case 'max': {
      // c = max(a, b). Pass dy through to whichever side won (a if a >= b).
      const a = tensorOf(op.a), b = tensorOf(op.b)
      const zero = constScalar(0, 'f32')
      const aWins = greater(a, b)  // ties go to b; subgradient choice
      accumulate(cotangents, op.a, unbroadcast(where(aWins, outCotan, broadcastTo(zero, outCotan.shape)), a.shape))
      accumulate(cotangents, op.b, unbroadcast(where(aWins, broadcastTo(zero, outCotan.shape), outCotan), b.shape))
      return
    }

    // ---- Reductions over last axis ---------------------------------------
    case 'mean_last': {
      // c[..., 1] = mean over last axis of a[..., D]. da[..., d] = dc[..., 0] / D.
      // outCotan has shape [..., 1]; broadcast to a's shape and divide by D.
      const a = tensorOf(op.a)
      const D = a.shape[a.shape.length - 1]!
      const expanded = broadcastTo(outCotan, a.shape)
      accumulate(cotangents, op.a, mulScalar(expanded, 1 / D))
      return
    }
    case 'sum_last': {
      // c[...] = sum over last axis (keepdims=false). da[..., d] = dc[...].
      // outCotan has rank one less than a; broadcast to a's shape (which inserts
      // back the last axis with a's last-axis size).
      const a = tensorOf(op.a)
      // First reshape outCotan to add a trailing 1, then broadcast to a's shape.
      const withKeep = reshape(outCotan, [...outCotan.shape, 1])
      accumulate(cotangents, op.a, broadcastTo(withKeep, a.shape))
      return
    }

    // ---- Shape ------------------------------------------------------------
    case 'reshape': {
      // c = reshape(a, ...). Backward: reshape outCotan back to a's shape.
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, reshape(outCotan, a.shape))
      return
    }
    case 'transpose': {
      // c = transpose(a, perm). Backward: transpose outCotan with inverse perm.
      const inv = invertPerm(op.perm)
      accumulate(cotangents, op.a, transpose(outCotan, inv))
      return
    }

    // ---- Linear algebra ---------------------------------------------------
    case 'matmul': {
      // c = a @ b, where a: [..., M, K], b: [K, N], c: [..., M, N].
      // dA = dC @ B^T  (matmul, since b is unbatched)
      // dB = sum_over_batch( A^T @ dC )
      //
      // Implementation note: dA uses the same `matmul` (a [...,M,N] · b [N,K])
      // because b is rank-2. dB needs A^T which has shape [..., K, M], then
      // matmul with dC ([..., M, N]) gives [..., K, N], which we sum over
      // leading batch dims to get [K, N].
      const a = tensorOf(op.a), b = tensorOf(op.b)
      // dA = dC @ B^T
      accumulate(cotangents, op.a, matmul(outCotan, swapAxes(b, -1, -2)))
      // dB: per-batch A^T @ dC, then sum over batch dims.
      // A is [..., M, K]; transpose last two axes.
      const aT = swapAxes(a, -1, -2)  // [..., K, M]
      // matmul_batched needs same rank on both sides. dC has rank `a.rank`;
      // aT has rank `a.rank`; use matmul_batched if rank > 2, else matmul.
      let perBatchDb: Tensor
      if (a.shape.length > 2) {
        perBatchDb = matmulBatched(aT, outCotan)  // [..., K, N]
      } else {
        perBatchDb = matmul(aT, outCotan)  // [K, N]
      }
      // Sum over leading batch dims to collapse to b's shape [K, N].
      accumulate(cotangents, op.b, sumToShape(perBatchDb, b.shape))
      return
    }
    case 'matmul_batched': {
      // c = a @ b, both [..., M, K] · [..., K, N] -> [..., M, N].
      // dA = dC @ B^T   (per-batch, all batch dims preserved)
      // dB = A^T @ dC   (per-batch)
      const a = tensorOf(op.a), b = tensorOf(op.b)
      accumulate(cotangents, op.a, matmulBatched(outCotan, swapAxes(b, -1, -2)))
      accumulate(cotangents, op.b, matmulBatched(swapAxes(a, -1, -2), outCotan))
      return
    }

    // ---- Indexing / casting (no gradient through integer indices) --------
    case 'one_hot':
      // The output is float, but the input (indices) is integer-valued — no
      // continuous gradient flows through it. Stop here.
      return

    // ---- Slicing ---------------------------------------------------------
    case 'slice_last_range': {
      // c = a[..., start:end]. Backward: pad outCotan with zeros to a's shape.
      // We construct this as: zeros at left, outCotan in middle, zeros at right,
      // concatenated along the last axis. We don't have concat or generic pad
      // ops; the simplest expression here is a sparse expansion via broadcasting
      // and addition of zero tensors. For Phase 2 we punt: slice's autograd is
      // implemented by emitting a single fused op that scatters the cotangent.
      // For now: signal that slice's backward needs a dedicated op kind.
      const a = tensorOf(op.a)
      // Build a zeros tensor of a's shape, then add via... no, we can't do
      // additive scatter without an index_put. Easiest path: add a dedicated
      // backward op kind. For this pass, throw until we extend the IR.
      throw new Error(
        `autograd: slice_last_range backward not implemented yet ` +
        `(would need a scatter-style op or a Concat op). ` +
        `Workaround for now: avoid taking gradients through slices by using ` +
        `separate matmuls for Q/K/V instead of a fused W_qkv. ` +
        `Tensor: ${a.shape} -> ${tensorOf(op.out).shape}`,
      )
    }

    // ---- Broadcast / un-broadcast (autograd infrastructure) ---------------
    case 'broadcast_to': {
      // c = broadcast(a, target). da = sum_to_shape(dc, a.shape).
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, sumToShape(outCotan, a.shape))
      return
    }
    case 'sum_to_shape': {
      // c = sum_to_shape(a, target). da = broadcast_to(dc, a.shape).
      const a = tensorOf(op.a)
      accumulate(cotangents, op.a, broadcastTo(outCotan, a.shape))
      return
    }

    // ---- ML primitives ---------------------------------------------------
    case 'log_softmax_last': {
      // c = log_softmax(a, axis=-1). softmax(a) = exp(c).
      // dL/dA = dL/dC - softmax(a) * sum_last_keepdims(dL/dC)
      const c = tensorOf(op.out)
      const sm = exp(c)  // softmax(a)
      // sum_last with keepdims via reshape: sum_last drops the dim, then
      // reshape to add a trailing 1 back, then broadcast multiplies.
      const sumDc = sumLast(outCotan)            // shape: [..., ] (rank-1 less)
      const sumDcKeep = reshape(sumDc, [...sumDc.shape, 1])
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
      const s = sumLast(dcXc)
      const sKeep = reshape(s, [...s.shape, 1])
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
      // Need broadcast-aware unreduction back to a's and b's original shapes.
      const cond = tensorOf(op.cond)
      const a = tensorOf(op.a)
      const b = tensorOf(op.b)
      // Build zero tensors via broadcasting a 0-d const scalar.
      const zeroA = broadcastTo(constScalar(0, a.dtype), outCotan.shape)
      const zeroB = broadcastTo(constScalar(0, b.dtype), outCotan.shape)
      accumulate(cotangents, op.a, unbroadcast(where(cond, outCotan, zeroA), a.shape))
      accumulate(cotangents, op.b, unbroadcast(where(cond, zeroB, outCotan), b.shape))
      return
    }

    case 'where_causal': {
      // c = where(causal_mask, a, fillValue). Upper triangle becomes constant
      // (no gradient); lower triangle passes a through. So da_lower = dc_lower,
      // da_upper = 0. We can't easily express this with current ops; punt.
      throw new Error(
        `autograd: where_causal backward not yet implemented. ` +
        `Use softmax_causal_last (which fuses the mask + softmax) instead.`,
      )
    }

    // ---- Adam ops are post-autograd; no backward through them. ----------
    case 'adam_update_m':
    case 'adam_update_v':
    case 'adam_update_p':
      throw new Error(`autograd: cannot differentiate through ${op.kind}`)

    // ---- relu_grad has no further backward (autograd-internal) ----------
    case 'relu_grad': {
      // We don't double-differentiate. If someone tries, this will blow up —
      // intentional. Phase 2 doesn't need 2nd-order gradients.
      throw new Error(
        `autograd: cannot take second-order gradient through relu_grad. ` +
        `Phase 2 does not support higher-order autodiff.`,
      )
    }

    default: {
      // Exhaustiveness check at type level.
      const _exhaustive: never = op
      void _exhaustive
      throw new Error(`autograd: unhandled op kind ${(op as OpNode).kind}`)
    }
  }
}

// ============================================================================
// Helpers
// ============================================================================

function invertPerm(perm: readonly number[]): number[] {
  const inv: number[] = new Array(perm.length)
  for (let i = 0; i < perm.length; i++) inv[perm[i]!] = i
  return inv
}
