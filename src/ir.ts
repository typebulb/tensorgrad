// Intermediate representation. Pure data: tracing, shape inference, autograd,
// and codegen all live in other modules and consume `Graph` / `OpNode`.

/** Element type of a tensor.
 *  - `f32` — 32-bit float. The only dtype for params, activations, and gradients.
 *  - `i32` — 32-bit signed int. Indices, token IDs, argmax results.
 *  - `bool` — 1-bit logical. Produced by comparisons (`less`/`greater`),
 *    consumed by `where`. Stored as `u32` on the GPU; not user-facing as
 *    an input dtype. */
export type Dtype = 'f32' | 'i32' | 'bool'

/** Tensor shape: a tuple of non-negative integer dim sizes. Read-only by
 *  convention — every shape in the IR is frozen at trace time. Rank 0 means
 *  a scalar (`[]`). */
export type Shape = readonly number[]

/**
 * The fundamental handle for everything in the IR. A `Tensor` is *metadata
 * only* — it carries shape, dtype, and a pointer back to the op that produced
 * it. The actual GPU storage doesn't exist until the graph is compiled and
 * run on a device.
 *
 * User code receives `Tensor`s from op calls (`add`, `matmul`, etc.) and
 * threads them through further ops. Tensors are immutable: each op call
 * returns a freshly-allocated handle pointing at a new node in the graph.
 *
 * - `id` — index into `Graph.tensors`. Unique within a single trace.
 * - `shape` / `dtype` — static at trace time; identical to the producing
 *   op's output shape/dtype.
 * - `source` — index into `Graph.ops` of the op that produced this tensor,
 *   or `null` for graph leaves (params, external inputs, persistent state).
 * - `site` — call-site captured when the op was invoked, used to attribute
 *   later shape errors to the user's frame rather than the library's.
 */
export interface Tensor {
  readonly id: number
  readonly shape: Shape
  readonly dtype: Dtype
  /** Index into `Graph.ops` of the producing op, or `null` for leaves
   *  (`param_input` / `tensor_input` / `state_input`). */
  readonly source: number | null
  /** Captured at op-call time so shape errors blame the user's frame, not
   *  the library's. Lazy: only formatted on demand. */
  readonly site: CallSite | null
}

/** Origin of an op invocation. Captured eagerly at the point each op is
 *  called so errors raised downstream (during shape inference, autograd,
 *  or codegen) can point at the user's source line rather than inside
 *  tensorgrad. Use `formatSite` to render for display. */
export interface CallSite {
  readonly opName: string
  /** Full `Error.stack` at the point of op invocation. Format on demand
   *  via `formatSite` — parsing/trimming is deferred to error reporting. */
  readonly stack: string
}

/**
 * Discriminated union over every op the IR knows about. Each variant carries
 * the inputs (tensor ids), the output tensor id (`out`), and any
 * op-specific scalar parameters baked at trace time.
 *
 * Adding a new op means:
 *   1. add a variant here, plus a `getOpInputs` case below,
 *   2. add a shape rule in `src/shape.ts`,
 *   3. add an adjoint rule in `src/grad.ts`,
 *   4. add a kernel template in `src/codegen.ts`,
 *   5. add a reference case in `test/_eval.ts` so the grad/smoke tests run it.
 *
 * The kinds intentionally match the surface API in `src/ops.ts` one-to-one
 * (with the exception of autograd-internal kinds like `*_grad`, `broadcast_to`,
 * `sum_to_shape` which are emitted by `appendGrad` but not user-facing).
 */
export type OpNode =
  // ---- Leaves ----------------------------------------------------------------
  | { kind: 'param_input'; out: number; name: string }
  | { kind: 'tensor_input'; out: number; name: string }
  // Persistent buffer (e.g. Adam's m/v). Allocated at compile time, filled
  // with `initValue`, survives across step() calls; updated via writebacks.
  | { kind: 'state_input'; out: number; name: string; initValue: number }

  // ---- Element-wise --------------------------------------------------------
  | { kind: 'add'; out: number; a: number; b: number }
  | { kind: 'sub'; out: number; a: number; b: number }
  | { kind: 'mul'; out: number; a: number; b: number }
  | { kind: 'div'; out: number; a: number; b: number }
  | { kind: 'min'; out: number; a: number; b: number }
  | { kind: 'max'; out: number; a: number; b: number }
  | { kind: 'mul_scalar'; out: number; a: number; scalar: number }
  | { kind: 'add_scalar'; out: number; a: number; scalar: number }

  // ---- Unary ---------------------------------------------------------------
  | { kind: 'sqrt'; out: number; a: number }
  | { kind: 'rsqrt'; out: number; a: number }
  | { kind: 'log'; out: number; a: number }
  | { kind: 'exp'; out: number; a: number }
  | { kind: 'relu'; out: number; a: number }
  | { kind: 'neg'; out: number; a: number }
  | { kind: 'abs'; out: number; a: number }
  | { kind: 'tanh'; out: number; a: number }
  | { kind: 'sigmoid'; out: number; a: number }
  | { kind: 'sin'; out: number; a: number }
  | { kind: 'cos'; out: number; a: number }
  // Inverted dropout. Same kernel runs forward (a = x) and backward (a = dy):
  // mask value is 0 or 1/(1-p), reproducible from (seed, salt, thread_id).
  // `salt` is a per-call counter; backward emits a matching `dropout` op with
  // the same (seed, salt) so masks line up. `seed` is the id of the shared
  // i32 scalar tensor_input (`__prngSeed`) the runtime updates per step.
  | { kind: 'dropout'; out: number; a: number; seed: number; p: number; salt: number }
  // Standard-normal sampler. Shares the per-step seed; `salt` is unique per
  // call (counted across all stochastic ops). Each thread does two PCG draws
  // + Box-Muller to emit one N(0, 1) value. Output shape is baked into the op
  // (no input tensor — randn synthesizes its values).
  | { kind: 'randn'; out: number; seed: number; salt: number; shape: Shape }
  // Identity-copy forward, no-op backward. Used to detach a tensor from the
  // autograd graph so gradient stops flowing through it (PyTorch's `.detach()`,
  // JAX's `lax.stop_gradient`). Works on any dtype.
  | { kind: 'stop_gradient'; out: number; a: number }

  // ---- Reductions (over last axis only; permute first for other axes) -----
  | { kind: 'mean_last'; out: number; a: number }   // keepdims=true
  | { kind: 'sum_last'; out: number; a: number }    // keepdims=false
  | { kind: 'argmax_last'; out: number; a: number } // i32, non-differentiable

  // ---- Shape ---------------------------------------------------------------
  | { kind: 'reshape'; out: number; a: number; newShape: Shape }
  | { kind: 'permute'; out: number; a: number; perm: readonly number[] }

  // ---- Linear algebra -----------------------------------------------------
  // Two kinds for two kernel shapes; public `matmul` dispatches on rhs rank.
  // Kept separate so autograd adjoint rules stay simple.
  | { kind: 'matmul'; out: number; a: number; b: number }          // [..., M, K] · [K, N]
  | { kind: 'matmul_batched'; out: number; a: number; b: number }  // [..., M, K] · [..., K, N]

  // ---- Indexing / casting --------------------------------------------------
  | { kind: 'one_hot'; out: number; indices: number; depth: number; dtype: Dtype }
  | { kind: 'arange'; out: number; n: number; dtype: Dtype }

  // ---- ML primitives (fused for cleaner autograd) -------------------------
  | { kind: 'softmax_causal_last'; out: number; a: number }
  | { kind: 'log_softmax_last'; out: number; a: number }
  // Pre-softmax causal mask. Upper-triangle (i < j) entries become `fillValue`
  // (typically a large negative); lower triangle passes through.
  | { kind: 'where_causal'; out: number; a: number; fillValue: number }
  // Sample one categorical index per leading position from logits along the
  // last axis. Output is i32, one rank less than input. Non-differentiable
  // (sampling is discrete). Implemented via Gumbel-max + PCG; salt + shared
  // PRNG seed plumbing match dropout / randn. Named `categorical` (not
  // `multinomial`) because we sample one index per row, not N with
  // replacement; matches `jax.random.categorical`.
  | { kind: 'categorical_last'; out: number; a: number; seed: number; salt: number }

  // ---- Comparisons + selection -------------------------------------------
  // Bool result (lowered to u32 in storage). Trailing-axis broadcast.
  | { kind: 'less'; out: number; a: number; b: number }
  | { kind: 'greater'; out: number; a: number; b: number }
  // out[i] = cond[i] ? a[i] : b[i]. cond is bool; a/b/cond broadcast to out.
  | { kind: 'where'; out: number; cond: number; a: number; b: number }

  // ---- Optimizer-fused ops (Adam) ----------------------------------------
  // Single kernel per param-element. Used by appendAdam to avoid decomposing
  // the update into ~12 element-wise dispatches per param.
  | { kind: 'adam_update_m'; out: number; m: number; g: number; beta1: number }
  | { kind: 'adam_update_v'; out: number; v: number; g: number; beta2: number }
  // p_new = decayShrink * p - lrt[0] * m_new / (sqrt(v_new) + eps).
  // `lrt` is a 0-d scalar tensor_input updated per step, already including
  // Adam's bias correction: lrt = lr * sqrt(1-beta2^t) / (1-beta1^t).
  // `decayShrink` is AdamW's `1 - lr * weightDecay` (or 1 for non-decayed
  // params). Static-lr training bakes it as a literal; scheduled lr routes
  // it through `decayShrinkTensor` (a per-step scalar input), which takes
  // precedence when non-null.
  | {
      kind: 'adam_update_p'
      out: number
      p: number
      mNew: number
      vNew: number
      lrt: number
      eps: number
      decayShrink: number
      decayShrinkTensor: number | null
    }

  // ---- Slicing / broadcasting / autograd infrastructure -------------------
  // General-axis slice. `axis` is non-negative (ops.ts normalizes negatives).
  | { kind: 'slice_range'; out: number; a: number; axis: number; start: number; end: number }
  // Adjoint of slice_range: scatter `a` into `[start, end)` along `axis` of
  // an otherwise-zero tensor of shape `outShape`. `axis` is non-negative.
  | { kind: 'scatter_axis'; out: number; a: number; outShape: Shape; axis: number; start: number; end: number }
  // Variable-arity concat along `axis`. Capped at 7 inputs by codegen
  // (WebGPU bind-group limit: 8 storage buffers per stage minus the output).
  | { kind: 'concat'; out: number; inputs: readonly number[]; axis: number }
  // Right-aligned NumPy broadcast. Emitted by autograd to expand cotangents.
  | { kind: 'broadcast_to'; out: number; a: number; targetShape: Shape }
  // Inverse of broadcast_to: sum-reduce to `targetShape`. Emitted by autograd
  // to un-broadcast a cotangent back to the smaller operand's shape.
  | { kind: 'sum_to_shape'; out: number; a: number; targetShape: Shape }
  // 0-d const. Used to seed the loss cotangent (1.0).
  | { kind: 'const_scalar'; out: number; value: number; dtype: Dtype }
  // n-d const fill of a single value. Backs `zeros` / `ones` (user-facing) and
  // any future const-tensor builder restricted to a uniform value. Shape lives
  // on the output tensor as usual; `value` and `dtype` ride on the op.
  | { kind: 'const_fill'; out: number; value: number; dtype: Dtype }
  // ReLU's backward: dy where x > 0, else 0. Fused so autograd doesn't have
  // to emit a where+const-zero+broadcast chain.
  | { kind: 'relu_grad'; out: number; x: number; dy: number }

  // ---- 2D convolution (NCHW). Bias is added separately via `add` + reshape
  // so these ops stay pure. Stride/padding are non-negative ints.
  | {
      kind: 'conv2d'                                // [B, C_in, H, W] · [C_out, C_in, K_h, K_w]
      out: number
      input: number
      weight: number
      strideH: number; strideW: number
      padH: number; padW: number
    }
  // dInput as transposed-conv of dy with weight. Implemented as a gather:
  // each input position sums contributions from every output whose receptive
  // field covered it.
  | {
      kind: 'conv2d_input_grad'
      out: number
      weight: number
      dy: number
      inH: number; inW: number
      strideH: number; strideW: number
      padH: number; padW: number
    }
  // dWeight as correlation between input and dy. Gather over (B, H_out, W_out)
  // per (C_out, C_in, K_h, K_w).
  | {
      kind: 'conv2d_weight_grad'
      out: number
      input: number
      dy: number
      kH: number; kW: number
      strideH: number; strideW: number
      padH: number; padW: number
    }

  // ---- 2D max pooling (NCHW). Argmax indices are not stored; backward
  // recomputes them on the fly. Padded regions are treated as -inf.
  | {
      kind: 'max_pool_2d'
      out: number
      input: number
      kH: number; kW: number
      strideH: number; strideW: number
      padH: number; padW: number
    }
  // Scatter dy to whichever input position won the argmax. Implemented as a
  // gather (one thread per input element) to avoid atomics; needs the
  // original forward `input` to recompute the argmax.
  | {
      kind: 'max_pool_2d_grad'
      out: number
      input: number
      dy: number
      kH: number; kW: number
      strideH: number; strideW: number
      padH: number; padW: number
    }

/**
 * A traced computation graph: a flat array of ops in topological (=
 * construction) order, plus the tensors they produce. Built by `traceFn(...)`;
 * consumed by `appendGrad` (autograd), `planBuffers` (memory layout), and
 * `emitKernels` (codegen).
 *
 * Once tracing is done a `Graph` should be treated as immutable — though
 * `traceInto` may re-enter it to append more ops (used by autograd and the
 * optimizer passes). `compiled.graph` exposes the final graph for
 * inspection (op list, tensor metadata, capture registry).
 */
export interface Graph {
  /** Ops in topological / construction order. */
  readonly ops: OpNode[]
  /** Every tensor produced in this trace, indexed by `Tensor.id`. */
  readonly tensors: Tensor[]
  /** Tensor ids exposed as outputs of the compiled function. Set by the
   *  trace driver — for a loss function, this is `[lossTensor.id]`. */
  readonly outputs: number[]
  /** Tensors registered for activation readback via `capture(name, t)`.
   *  Keyed by user-supplied name, insertion order preserved. Empty when
   *  no captures are registered (the common training case — zero overhead). */
  readonly captures: Map<string, number>
}

export function makeGraph(): Graph {
  return { ops: [], tensors: [], outputs: [], captures: new Map() }
}

export function addTensor(g: Graph, shape: Shape, dtype: Dtype, source: number | null, site: CallSite | null): Tensor {
  const id = g.tensors.length
  const t: Tensor = { id, shape, dtype, source, site }
  g.tensors.push(t)
  return t
}

// Generic over op kind so callers don't need `as any` casts: `Extract<OpNode,
// { kind: K }>` narrows the union; `Omit` strips the fields addOp supplies.
export function addOp<K extends OpNode['kind']>(
  g: Graph,
  kind: K,
  shape: Shape,
  dtype: Dtype,
  site: CallSite | null,
  fields: Omit<Extract<OpNode, { kind: K }>, 'kind' | 'out'>,
): Tensor {
  const opIndex = g.ops.length
  const out = addTensor(g, shape, dtype, opIndex, site)
  const node = { kind, out: out.id, ...fields } as Extract<OpNode, { kind: K }>
  g.ops.push(node)
  return out
}

/**
 * The tensor ids an op reads as inputs, in a canonical order. Centralizes
 * the op-kind → input-field mapping so external graph walkers (IR viewers,
 * custom optimizers, correctness harnesses) don't have to maintain their
 * own switch over every op kind — that switch silently breaks every time
 * the IR grows a new op.
 *
 * Returns tensor ids only (the `out` field is the *output*, not an input,
 * so it's excluded). Order matches the op's natural reading order: lhs
 * before rhs for binops, condition before branches for `where`, etc. Scalar
 * parameters baked into the op (e.g. `dropout.p`, `mul_scalar.scalar`) are
 * not tensors and don't appear.
 */
export function getOpInputs(op: OpNode): readonly number[] {
  switch (op.kind) {
    case 'param_input': case 'tensor_input': case 'state_input':
    case 'arange': case 'const_scalar': case 'const_fill': case 'randn':
      return []
    case 'add': case 'sub': case 'mul': case 'div': case 'min': case 'max':
    case 'less': case 'greater':
    case 'matmul': case 'matmul_batched':
      return [op.a, op.b]
    case 'mul_scalar': case 'add_scalar':
    case 'sqrt': case 'rsqrt': case 'log': case 'exp': case 'relu':
    case 'neg': case 'abs': case 'tanh': case 'sigmoid': case 'sin': case 'cos':
    case 'mean_last': case 'sum_last': case 'argmax_last':
    case 'reshape': case 'permute':
    case 'softmax_causal_last': case 'log_softmax_last':
    case 'where_causal': case 'stop_gradient':
    case 'slice_range': case 'scatter_axis':
    case 'broadcast_to': case 'sum_to_shape':
      return [op.a]
    case 'dropout': return [op.a, op.seed]
    case 'categorical_last': return [op.a, op.seed]
    case 'one_hot': return [op.indices]
    case 'where': return [op.cond, op.a, op.b]
    case 'concat': return op.inputs
    case 'relu_grad': return [op.x, op.dy]
    case 'adam_update_m': return [op.m, op.g]
    case 'adam_update_v': return [op.v, op.g]
    case 'adam_update_p':
      return op.decayShrinkTensor !== null
        ? [op.p, op.mNew, op.vNew, op.lrt, op.decayShrinkTensor]
        : [op.p, op.mNew, op.vNew, op.lrt]
    case 'conv2d': return [op.input, op.weight]
    case 'conv2d_input_grad': return [op.weight, op.dy]
    case 'conv2d_weight_grad': return [op.input, op.dy]
    case 'max_pool_2d': return [op.input]
    case 'max_pool_2d_grad': return [op.input, op.dy]
  }
}

// Cheap: materializes the stack string but defers parsing to format time.
export function captureSite(opName: string): CallSite {
  const stack = (new Error()).stack ?? ''
  return { opName, stack }
}

export function formatSite(site: CallSite): string {
  const lines = site.stack.split('\n')
  const userFrames: string[] = []
  // Skip the "Error" header and every frame inside the library; first
  // surviving frame is user code.
  for (const line of lines.slice(1)) {
    if (line.includes('/tensorgrad/src/') || line.includes('\\tensorgrad\\src\\')) continue
    userFrames.push(line.trim())
    if (userFrames.length >= 3) break
  }
  if (userFrames.length === 0) return `[${site.opName}] (no user frame found)`
  return `[${site.opName}]\n  ${userFrames.join('\n  ')}`
}
