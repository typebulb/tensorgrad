// Fold a Linear's bias add and a following activation into the GEMM that produced their
// input, so the [rows, N] intermediate is never written to global memory and read back.
//
// Why this and not general elementwise fusion. `specs/Performance.md` §Deferred has the
// general pass measured dead twice, and correctly: merging dispatches buys nothing when
// per-dispatch overhead is ~2 µs and the elementwise kernels already run at bandwidth peak.
// This is a different lever. It does not merge passes, it DELETES TRAFFIC — a SIREN layer
// writes its [147456, 64] activation from the GEMM, reads and rewrites it for the bias, then
// reads and rewrites it again for the sin: 151 MB per layer where a fused store moves 0.
// Measured on the chameleon frame (typebulbs-tentative/chameleon-load.bulb.md, RTX 3080 Ti
// laptop): `add` 0.71 ms / 253.6 MB and `sin` 0.64 ms / 226.5 MB of a 3.35 ms, 966 MB frame.
// Both kinds run AT bandwidth, which is exactly why removing their bytes removes their time.
//
// What each half does on a training graph, decided entirely by consumer counting — no line of
// autograd knows this pass exists, and it reads no gradient.
//
//   Bias folds everywhere, training included. `matmul`'s VJPs are dA = dY·Bᵀ and dB = Aᵀ·dY,
//   and `add`'s just routes the cotangent: none of them reads the matmul's OUTPUT. So that
//   intermediate still has exactly one consumer after `appendGrad`, and the fold is safe.
//
//   The activation folds only on a forward-only graph. Every activation's VJP reads the
//   activation's own input — sin' needs cos(z), relu' needs z > 0 — so after `appendGrad`
//   that value has two readers and the sole-consumer guard refuses. Which is the right
//   outcome on the merits as well: a training step wants the GPU saturated, while a render
//   loop capped at the display's refresh rate is spending the difference on heat.
//
// Applied to `matmul` (rhs rank 2 — what `Linear.fwd` emits) and `conv2d`, and only for a
// bias that broadcasts along exactly the axis the kernel indexes: [N] for a matmul, whose
// epilogue reads `bias[n]`, and [1, C_out, 1, 1] for a conv, whose epilogue reads the
// channel. Anything else is an ordinary elementwise add and is left alone, so neither
// epilogue has to reimplement a general broadcast walk.
//
// Not covered, in rough order of what they would be worth: `matmul_batched` (attention's
// Q@Kᵀ), `mul_scalar` / `add_scalar` (free — no binding, the constant bakes into the WGSL),
// and the composed activations `gelu` / `silu`, which are several ops each and would need
// either their own op kinds or a general chain epilogue.

import type { Graph, OpNode, UnaryExprKind } from './ir.js'
import { getOpInputs } from './ir.js'

const FUSABLE_ACTS: ReadonlySet<string> = new Set<UnaryExprKind>([
  'sqrt', 'rsqrt', 'log', 'exp', 'relu', 'neg', 'abs', 'tanh', 'sigmoid', 'erf', 'sin', 'cos',
])

/**
 * Rewrite `matmul → add(bias)` and `matmul → unary` chains into a single fused `matmul`,
 * in place. Mutates `graph.ops`; returns how many folds happened.
 *
 * The rewrite replaces the *consuming* op (the add, or the unary) with a fused matmul
 * carrying the original operands. That keeps the consumer's own output tensor id — so
 * everything downstream, including `graph.outputs` and capture sites, is untouched — and
 * keeps the op at an index whose inputs are all produced earlier, so topological order
 * holds. The now-orphaned upstream matmul is left for `eliminateDeadCode`, which the
 * caller runs next.
 */
export function fuseMatmulEpilogue(graph: Graph): number {
  const ops = graph.ops as OpNode[]
  const tensors = graph.tensors
  const numel = (s: readonly number[]) => s.reduce((a, b) => a * b, 1)

  const consumers = new Map<number, number>()
  for (const op of ops) for (const id of getOpInputs(op)) consumers.set(id, (consumers.get(id) ?? 0) + 1)
  // A tensor the caller can observe is not an intermediate, however few ops read it.
  const pinned = new Set<number>([...graph.outputs, ...graph.captures.values()])
  const solelyFeeds = (id: number) => consumers.get(id) === 1 && !pinned.has(id)

  /** The fusable matmul that produced `id` and feeds nothing else, or null. */
  const gemmFeeding = (id: number): { op: OpNode & { kind: 'matmul' }; index: number } | null => {
    if (!solelyFeeds(id)) return null
    const src = tensors[id]!.source
    if (src === null) return null
    const op = ops[src]!
    return op.kind === 'matmul' ? { op, index: src } : null
  }

  /** The fusable conv2d that produced `id` and feeds nothing else, or null. */
  const convFeeding = (id: number): (OpNode & { kind: 'conv2d' }) | null => {
    if (!solelyFeeds(id)) return null
    const src = tensors[id]!.source
    if (src === null) return null
    const op = ops[src]!
    return op.kind === 'conv2d' ? op : null
  }

  /** The `[C_out]` worth of floats behind a conv bias operand, or null if this operand is
   *  not a per-channel bias. `Conv2d.fwd` broadcasts by reshaping its `[C_out]` param to
   *  `[1, C_out, 1, 1]`, so the shape test is on that exact form — a rank-1 `[C_out]` would
   *  broadcast along W instead, and a full-shape operand is ordinary arithmetic. Sees
   *  through the reshape to the param itself where it can, which lets DCE drop the reshape's
   *  copy kernel too; the reshape's own output would work as well, since both hold the same
   *  C_out floats in the same order. */
  const channelBias = (id: number, cOut: number): number | null => {
    const s = tensors[id]!.shape
    if (s.length !== 4 || s[0] !== 1 || s[1] !== cOut || s[2] !== 1 || s[3] !== 1) return null
    const src = tensors[id]!.source
    const producer = src === null ? null : ops[src]!
    if (producer?.kind === 'reshape' && numel(tensors[producer.a]!.shape) === cOut) return producer.a
    return id
  }

  let folds = 0

  // Pass 1 — bias. `Linear.fwd` is `add(matmul(x, W), b)`; fold b into the store.
  for (let i = 0; i < ops.length; i++) {
    const op = ops[i]!
    if (op.kind !== 'add') continue
    const g = gemmFeeding(op.a)
    if (g && g.op.bias === undefined) {
      const N = tensors[g.op.out]!.shape.at(-1)!
      const bias = tensors[op.b]!.shape
      if (bias.length !== 1 || bias[0] !== N) continue
      ops[i] = { kind: 'matmul', out: op.out, a: g.op.a, b: g.op.b, bias: op.b }
      folds++
      continue
    }
    // `Conv2d.fwd` is `add(conv2d(x, W), reshape(b, [1, C_out, 1, 1]))`.
    const c = convFeeding(op.a)
    if (!c || c.bias !== undefined) continue
    const cOut = tensors[c.weight]!.shape[0]!
    const bias = channelBias(op.b, cOut)
    if (bias === null) continue
    ops[i] = { ...c, out: op.out, bias }
    folds++
  }

  // Pass 2 — activation, after bias so `matmul → add → sin` collapses to one op.
  for (let i = 0; i < ops.length; i++) {
    const op = ops[i]!
    if (!FUSABLE_ACTS.has(op.kind)) continue
    const src = (op as OpNode & { a: number }).a
    const act = op.kind as UnaryExprKind
    const g = gemmFeeding(src)
    if (g && g.op.act === undefined) { ops[i] = { ...g.op, out: op.out, act }; folds++; continue }
    const c = convFeeding(src)
    if (!c || c.act !== undefined) continue
    ops[i] = { ...c, out: op.out, act }
    folds++
  }

  return folds
}
