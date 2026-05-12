// Intermediate representation for tensor computations.
//
// A `Graph` is a flat array of `OpNode`s in topological (= construction) order.
// A `Tensor` is an opaque handle: shape + dtype + a pointer back to the OpNode
// that produced it (or `null` for graph leaves — params and external inputs).
//
// This is the data structure everything else operates on:
//   - tracing builds it (src/trace.ts)
//   - autograd walks it in reverse to add backward nodes (src/grad.ts, later)
//   - codegen reads it to emit WGSL kernels and a dispatch plan (src/codegen.ts, later)
//
// Design intent: keep this file boring. No tracing logic, no shape inference,
// no codegen — those live in their own modules and consume `Graph` / `OpNode`.

export type Dtype = 'f32' | 'i32' | 'bool'
export type Shape = readonly number[]

// A Tensor is just metadata + a unique id. The actual storage doesn't exist
// until the graph is compiled and run on a device.
export interface Tensor {
  readonly id: number
  readonly shape: Shape
  readonly dtype: Dtype
  // null for leaves (params, external inputs); otherwise the index into Graph.ops.
  readonly source: number | null
  // Captured at op-call time so shape errors blame the user's frame, not the
  // library's. Lazy: only formatted on demand.
  readonly site: CallSite | null
}

export interface CallSite {
  readonly opName: string
  // Full Error stack at the point of op invocation. Format on demand.
  readonly stack: string
}

// Discriminated union over every op the IR knows about. Adding an op means:
//   1. add a variant here,
//   2. add a shape rule in src/shape.ts,
//   3. add an adjoint (autograd "transpose") rule in src/grad.ts,
//   4. add a kernel template in src/codegen.ts.
// The kinds intentionally match the surface API in src/ops.ts one-to-one.
export type OpNode =
  // ---- Leaves ----------------------------------------------------------------
  // A trainable parameter, supplied by the caller as a Float32Array at runtime.
  | { kind: 'param_input'; out: number; name: string }
  // A non-trainable input (tokens, targets, constants). Bound at runtime.
  | { kind: 'tensor_input'; out: number; name: string }
  // Persistent state buffer (e.g. Adam's m/v). Allocated and zero-initialized
  // at compile time; survives across step() calls. Updated via writebacks
  // declared in the compile result.
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
  // Inverted dropout. Same kernel runs forward (a = x) and backward (a = dy):
  // applies a per-element mask of value 0 or 1/(1-p), reproducibly from
  // (seed, salt, thread_id). `salt` is a per-dropout-call counter unique
  // within this graph; backward emits another `dropout` op with the same
  // salt + seed so the masks match. `seed` is the id of a shared i32 scalar
  // tensor_input (`__dropoutSeed`) the runtime updates per step.
  | { kind: 'dropout'; out: number; a: number; seed: number; p: number; salt: number }

  // ---- Reductions (over last axis only; reshape if you need other axes) ----
  | { kind: 'mean_last'; out: number; a: number }   // keepdims=true
  | { kind: 'sum_last'; out: number; a: number }    // keepdims=false
  // argmax over the last axis (keepdims=false). Returns i32; non-differentiable.
  | { kind: 'argmax_last'; out: number; a: number }

  // ---- Shape ---------------------------------------------------------------
  | { kind: 'reshape'; out: number; a: number; newShape: Shape }
  | { kind: 'permute'; out: number; a: number; perm: readonly number[] }

  // ---- Linear algebra -----------------------------------------------------
  // Public `matmul(a, b)` (src/ops.ts) dispatches between the two kinds below
  // based on rhs rank — rank-2 rhs uses 'matmul', rank-matched batched rhs
  // uses 'matmul_batched'. Two kinds for two distinct kernel shapes; one
  // public name. Kept separate so autograd adjoint rules stay simple.
  //
  // matmul: a [..., M, K] · b [K, N] -> [..., M, N]. b is unbatched.
  | { kind: 'matmul'; out: number; a: number; b: number }
  // matmul_batched: a [..., M, K] · b [..., K, N] -> [..., M, N]. Used by attention.
  | { kind: 'matmul_batched'; out: number; a: number; b: number }

  // ---- Indexing / casting --------------------------------------------------
  | { kind: 'one_hot'; out: number; indices: number; depth: number; dtype: Dtype }
  | { kind: 'arange'; out: number; n: number; dtype: Dtype }

  // ---- ML primitives (fused for cleaner autograd) -------------------------
  | { kind: 'softmax_causal_last'; out: number; a: number }
  | { kind: 'log_softmax_last'; out: number; a: number }
  // Sets cells where (i >= j) on the last two axes; for masking attention scores
  // *before* softmax. Lower-triangle entries pass through; upper-triangle entries
  // become `fillValue` (typically -inf or a large negative number).
  | { kind: 'where_causal'; out: number; a: number; fillValue: number }

  // ---- Comparisons + selection -------------------------------------------
  // Element-wise comparison; result is bool (lowered to u32 in storage).
  // Supports the same trailing-axis broadcast as element-wise binops.
  | { kind: 'less'; out: number; a: number; b: number }
  | { kind: 'greater'; out: number; a: number; b: number }
  // Element-wise select: out[i] = cond[i] ? a[i] : b[i]. cond must be bool.
  // a, b, cond all broadcast-compatible to out's shape.
  | { kind: 'where'; out: number; cond: number; a: number; b: number }

  // ---- Optimizer-fused ops (Adam) ----------------------------------------
  // Each is a single kernel doing the full per-element math, baking in the
  // hyperparameter constant. Used by appendAdam() to avoid decomposing the
  // update into ~12 element-wise dispatches per param.
  | { kind: 'adam_update_m'; out: number; m: number; g: number; b1: number }
  | { kind: 'adam_update_v'; out: number; v: number; g: number; b2: number }
  // adam_update_p: p_new = decayShrink * p - lrt[0] * m_new / (sqrt(v_new) + eps).
  // `lrt` is a scalar tensor (provided as a tensor_input updated per step) that
  // already includes Adam's bias-correction factor: lrt = lr * sqrt(1-b2^t) / (1-b1^t).
  // `decayShrink` is the decoupled-weight-decay factor (Loshchilov & Hutter,
  // "AdamW"): 1 - lr * weightDecay when the param is being decayed, 1 otherwise.
  // It can be either a compile-time literal (number) for fixed-lr training, or a
  // tensor id pointing at a scalar input that the runtime updates per step (used
  // when the user supplies an lr schedule via `adam: { lr: (step) => ... }`).
  | {
      kind: 'adam_update_p'
      out: number
      p: number
      mNew: number
      vNew: number
      lrt: number
      eps: number
      decayShrink: number               // literal (used when decayShrinkTensor is null)
      decayShrinkTensor: number | null  // tensor id of a scalar input; takes precedence when set
    }

  // ---- Slicing / broadcasting / autograd infrastructure -------------------
  // Slice [start, end) along the last axis. Output shape: input shape with
  // last axis replaced by (end - start). Used for splitting Q/K/V from a
  // single fused QKV matmul.
  | { kind: 'slice_last_range'; out: number; a: number; start: number; end: number }
  // General-axis slice: take elements [start, end) along `axis`. Axis is
  // non-negative; ops.ts normalizes negative axes before constructing.
  | { kind: 'slice_range'; out: number; a: number; axis: number; start: number; end: number }
  // Concatenation along `axis` of two or more inputs. All inputs must have
  // identical shape except along `axis`; output's size on `axis` is the sum.
  // Variable-arity input — the only such op in the IR. Capped at 7 inputs
  // by codegen (WebGPU bind group limit: 8 storage buffers per stage,
  // minus 1 for the output).
  | { kind: 'concat'; out: number; inputs: readonly number[]; axis: number }
  // Broadcast `a` to `targetShape`. Standard right-aligned NumPy broadcast.
  // Used by autograd to expand cotangents back over reduced/broadcast axes.
  | { kind: 'broadcast_to'; out: number; a: number; targetShape: Shape }
  // Inverse of broadcast_to: sum-reduce `a` to `targetShape`. Used by autograd
  // to "un-broadcast" a cotangent back to the smaller operand's shape.
  | { kind: 'sum_to_shape'; out: number; a: number; targetShape: Shape }
  // 0-d tensor with a constant value. Used to seed loss cotangent (1.0).
  | { kind: 'const_scalar'; out: number; value: number; dtype: Dtype }
  // ReLU's backward: passes `dy` through where `x > 0`, else 0. Output shape = x's.
  | { kind: 'relu_grad'; out: number; x: number; dy: number }

  // ---- 2D convolution (NCHW) ----------------------------------------------
  // conv2d: forward. Input [B, C_in, H, W] · weight [C_out, C_in, K_h, K_w]
  // -> [B, C_out, H_out, W_out]. Bias is added separately (via `add` + broadcast)
  // so this op stays pure. Stride and per-side padding are non-negative ints.
  | {
      kind: 'conv2d'
      out: number
      input: number          // [B, C_in, H, W]
      weight: number         // [C_out, C_in, K_h, K_w]
      strideH: number; strideW: number
      padH: number; padW: number
    }
  // conv2d input gradient: dInput = "transposed conv" of dy with weight.
  // Computed as a gather: each input position sums contributions from every
  // output position whose receptive field contained it. Shape matches `input`.
  | {
      kind: 'conv2d_input_grad'
      out: number
      weight: number         // [C_out, C_in, K_h, K_w]
      dy: number             // [B, C_out, H_out, W_out]
      inH: number; inW: number  // input spatial dims (target shape carries C_in implicitly via weight)
      strideH: number; strideW: number
      padH: number; padW: number
    }
  // conv2d weight gradient: dWeight = correlation between input and dy.
  // Computed as a gather over (B, H_out, W_out) per (C_out, C_in, K_h, K_w).
  // Shape matches `weight`.
  | {
      kind: 'conv2d_weight_grad'
      out: number
      input: number          // [B, C_in, H, W]
      dy: number             // [B, C_out, H_out, W_out]
      kH: number; kW: number
      strideH: number; strideW: number
      padH: number; padW: number
    }

  // ---- 2D max pooling (NCHW) ----------------------------------------------
  // max_pool_2d: forward. Input [B, C, H, W] -> [B, C, H_out, W_out]. Argmax
  // indices are not stored; the backward kernel recomputes them on the fly.
  // Padded regions are treated as -inf (don't contribute to argmax).
  | {
      kind: 'max_pool_2d'
      out: number
      input: number          // [B, C, H, W]
      kH: number; kW: number
      strideH: number; strideW: number
      padH: number; padW: number
    }
  // max_pool_2d backward: scatters dy to whichever input position was the
  // argmax. Implemented as a gather (one thread per input element) to avoid
  // atomics: each input position checks every output whose receptive field
  // covers it, and if it's the argmax for that output, accumulates dy.
  // `input` is the original forward input (needed to recompute argmax).
  | {
      kind: 'max_pool_2d_grad'
      out: number
      input: number          // [B, C, H, W] — the original forward input
      dy: number             // [B, C, H_out, W_out]
      kH: number; kW: number
      strideH: number; strideW: number
      padH: number; padW: number
    }

// A Graph collects ops and tensors during tracing, then becomes the input to
// autograd and codegen. Once tracing is done it should be treated as immutable.
export interface Graph {
  readonly ops: OpNode[]
  readonly tensors: Tensor[]
  // Names of tensors that should be exposed as outputs of the compiled function.
  // Set by the trace driver; for a loss function, this is `[lossTensor]`.
  readonly outputs: number[]
  // Tensors registered for activation readback via `capture(name, t)`.
  // Keyed by user-supplied name; insertion order preserved. Empty when no
  // captures registered (the common training case — zero overhead).
  readonly captures: Map<string, number>
}

export function makeGraph(): Graph {
  return { ops: [], tensors: [], outputs: [], captures: new Map() }
}

// Internal: register a fresh tensor in the graph and return its id.
export function addTensor(g: Graph, shape: Shape, dtype: Dtype, source: number | null, site: CallSite | null): Tensor {
  const id = g.tensors.length
  const t: Tensor = { id, shape, dtype, source, site }
  g.tensors.push(t)
  return t
}

// Internal: append an op and the tensor it produces. Returns the produced tensor.
// Generic over the specific op kind so callers don't need `as any` casts.
// `Extract<OpNode, { kind: K }>` narrows the union to the chosen variant, then
// `Omit` strips the parts addOp itself supplies (the kind tag and out tensor id).
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

// Capture a call site without paying full Error formatting cost up-front.
// The stack is materialised but parsing/trimming is deferred to error reporting.
export function captureSite(opName: string): CallSite {
  // Skip our own frame plus the op wrapper's frame; user's frame is what's left.
  const stack = (new Error()).stack ?? ''
  return { opName, stack }
}

// Format a CallSite for inclusion in a thrown error. Strips Tensorgrad frames
// and library internals so the user sees their code first.
export function formatSite(site: CallSite): string {
  const lines = site.stack.split('\n')
  // Stack starts with "Error" line; drop it. Then drop frames from this file
  // and from src/ops.ts so the first surviving frame is user code.
  const userFrames: string[] = []
  for (const line of lines.slice(1)) {
    if (line.includes('/tensorgrad/src/') || line.includes('\\tensorgrad\\src\\')) continue
    userFrames.push(line.trim())
    if (userFrames.length >= 3) break
  }
  if (userFrames.length === 0) return `[${site.opName}] (no user frame found)`
  return `[${site.opName}]\n  ${userFrames.join('\n  ')}`
}
