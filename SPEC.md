# Tensorgrad — Architecture

This document covers the design decisions, IR, and internals of tensorgrad.
For installation and user-facing API, see `README.md`. For pre-1.0
implementation status and what to pick up next, see `HANDOFF.md`.

## Scope and non-goals (load-bearing)

The library only does what it does because of what it doesn't do. Each
"out of scope" decision is the *precondition* that lets the rest stay small.

**In scope:**
- Static-shape models. Every shape is fixed at compile time.
- WebGPU only.
- f32 only.
- `grad` (reverse-mode autograd) as the only transformation.
- A closed set of ~25 ops covering transformers + MLPs.
- Adam optimizer in-IR.

**Out of scope (deliberately):**
- Wasm or WebGL fallback.
- Dynamic shapes, shape polymorphism.
- `vmap`, `pmap`, `jvp`, `custom_vjp`, higher-order gradients.
- Dtype promotion, mixed precision.
- General PyTree machinery (we use `Module` + property paths instead).
- Inference of pre-trained models (use ONNX Runtime Web or transformers.js).
- ONNX import / safetensors / model loaders.
- Distributed training, gradient accumulation across devices.

The non-goals are load-bearing. Trying to add any of them without rethinking
the IR forces complexity throughout.

## Architecture overview

```
┌────────────────────────────────────────────────────────────┐
│ User code                                                   │
│   class Model extends Module { /* params */ }               │
│   function forward(m: Model, x: Tensor): Tensor { /* ... */ }│
└────────────────────────────────────────────────────────────┘
                          │
                          ▼  trace()
┌────────────────────────────────────────────────────────────┐
│ Forward IR build (src/trace.ts, src/ops.ts, src/shape.ts)   │
│   Each op call appends a node to the Graph and returns a    │
│   fresh Tensor handle. Shapes inferred + validated per op.  │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼  appendGrad()
┌────────────────────────────────────────────────────────────┐
│ Reverse-mode autograd (src/grad.ts)                         │
│   Topological walk; each forward op contributes its         │
│   transpose rule, building the backward graph in-place.     │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼  appendAdam() (optional)
┌────────────────────────────────────────────────────────────┐
│ Optimizer (src/adam.ts)                                     │
│   Per-param: m_state, v_state state_inputs + fused          │
│   adam_update_{m,v,p} ops. Writebacks declared.             │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼  planBuffers()
┌────────────────────────────────────────────────────────────┐
│ Buffer plan (src/buffers.ts)                                │
│   One GPU buffer per IR tensor, categorized:                │
│   param / param_grad / state / tensor_input / intermediate. │
│   Writebacks resolved to (source_buf → dest_buf) pairs.     │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼  emitKernels()
┌────────────────────────────────────────────────────────────┐
│ WGSL codegen (src/codegen.ts)                               │
│   Per op kind: a kernel template with shapes baked in.      │
│   Returns dispatch-ready KernelSpec[].                      │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼  createRuntime()
┌────────────────────────────────────────────────────────────┐
│ Runtime (src/runtime.ts)                                    │
│   GPUDevice setup, pipeline cache, bind groups,             │
│   step(): upload inputs → dispatch all kernels → writebacks │
│           → loss readback. Compile errors surface via       │
│           pushErrorScope+getCompilationInfo.                │
└────────────────────────────────────────────────────────────┘
```

## Key design decisions

**D1. Runtime tracing, not build-time.** Forward function is traced once on
first compile; the IR is built from those op calls. Build-time tracing via
a TypeScript transformer plugin would be cleaner but adds a build-step
requirement. v2 candidate.

**D2. Tensors are opaque handles, not Proxies.** Each op returns a fresh
`Tensor` (just `{ id, shape, dtype, source, site }`). Proxy-based tracing
gives nicer error UX but couples the IR to runtime introspection.

**D3. No reference counting.** Every IR tensor gets its own GPU buffer,
allocated once and never freed. Our scope (one model, fixed shapes,
training in a loop) means there's nothing to gain from refcount discipline.
Memory cost is bounded; buffer pooling is a v2 optimization, not v1
correctness.

**D4. Closed op set.** The IR knows about exactly the ops it supports.
Adding a new op means adding (a) shape rule, (b) WGSL kernel template,
(c) autograd transpose rule. This is intentional — a closed op set makes
each piece tractable to write and verify by hand.

**D5. Shapes checked at trace time, not at type level.** Type-level shape
encoding in TypeScript is real but hits recursion limits and adds
significant generic complexity to user code. Runtime shape errors at trace
time, with call-site capture, are good enough.

**D6. Adam state in the IR.** Optimizer state (m, v, plus a per-step `lrt`
scalar for bias correction) lives in dedicated `state_input` buffers that
persist across `step()` calls. Writebacks at the end of each step copy new
values into their persistent homes. No CPU↔GPU round-trip per step.

**D7. Module separates from forward.** Mutable parameter storage lives in
`Module` subclasses. Forward functions are pure, take the materialized
model as the first argument, and call ordinary op functions. State and
computation never mix — JAX's lesson, applied to TypeScript with
class-based ergonomics.

**D8. JS-number scalar overloads.** `add(x, 1e-5)` and `add(x, y)` both
work. The scalar variants dispatch to fused IR ops internally.

## IR

```ts
interface Tensor {
  readonly id: number
  readonly shape: Shape
  readonly dtype: Dtype
  readonly source: number | null   // op index, or null for leaves
  readonly site: CallSite | null   // user's stack at op-call time
}

type OpNode =
  | { kind: 'param_input'; ... } | { kind: 'tensor_input'; ... }
  | { kind: 'state_input'; ... } | { kind: 'arange'; ... }
  | { kind: 'const_scalar'; ... }
  | { kind: 'add' | 'sub' | 'mul' | 'div'; ... }
  | { kind: 'add_scalar' | 'mul_scalar'; ... }
  | { kind: 'sqrt' | 'rsqrt' | 'log' | 'exp' | 'relu'; ... }
  | { kind: 'less' | 'greater'; ... } | { kind: 'where'; ... }
  | { kind: 'mean_last' | 'sum_last'; ... }
  | { kind: 'reshape' | 'transpose' | 'slice_last_range'; ... }
  | { kind: 'broadcast_to' | 'sum_to_shape'; ... }
  | { kind: 'matmul' | 'matmul_batched'; ... }
  | { kind: 'one_hot'; ... }
  | { kind: 'softmax_causal_last' | 'log_softmax_last' | 'where_causal'; ... }
  | { kind: 'relu_grad'; ... }
  | { kind: 'adam_update_m' | 'adam_update_v' | 'adam_update_p'; ... }

interface Graph {
  readonly ops: OpNode[]
  readonly tensors: Tensor[]
  readonly outputs: number[]   // tensor ids — typically just the loss
}
```

The op kinds are intentionally split fine-grained (`mean_last` not
`mean(axis)`) because each kind maps to a hand-written WGSL kernel. Adding
generality later is fine; pretending to be more general than we are isn't.

## Op set (current)

**Leaves:** `param_input`, `tensor_input`, `state_input`, `arange`, `const_scalar`

**Element-wise binops** (NumPy broadcasting): `add`, `sub`, `mul`, `div`,
plus fused `add_scalar`, `mul_scalar`

**Element-wise unary:** `sqrt`, `rsqrt`, `log`, `exp`, `relu`

**Comparisons + select:** `less`, `greater`, `where`

**Reductions over last axis:** `mean_last`, `sum_last`

**Shape:** `reshape`, `transpose`, `slice_last_range`, `broadcast_to`, `sum_to_shape`

**Linear algebra:** `matmul` (2D rhs), `matmul_batched` (both batched)

**Indexing / casting:** `one_hot`

**ML primitives** (fused for clean autograd): `softmax_causal_last`,
`log_softmax_last`, `where_causal`

**Autograd-internal:** `relu_grad`

**Adam-internal:** `adam_update_m`, `adam_update_v`, `adam_update_p`

## Module abstraction

The `Module` base class enables Domeleon-style auto-discovery of nested
modules and parameters via property reflection:

```ts
class Linear extends Module {
  W: Tensor; b: Tensor
  constructor(public inDim: number, public outDim: number) {
    super()
    this.W = this.param([inDim, outDim])  // returns ParamSentinel cast to Tensor
    this.b = this.param([outDim])
  }
}
```

`this.param(shape)` returns a `ParamSentinel` typed as `Tensor`. At compile
time, `materializeParams(root)` walks enumerable properties of the model
tree (recursing into nested `Module` instances and arrays of modules),
replaces every sentinel with a real `paramInput` tensor whose name is the
property path (`layers.0.attn.q.W`), and returns a flat `Record<path,
Tensor>` for autograd to use.

This is the JAX/Equinox separation: parameter storage is mutable
(state-bearing components), forward computation is pure (functions over
materialized parameters and inputs).

## WGSL codegen

Each op kind has a kernel template in `codegen.ts`. Shapes are **baked into
the WGSL as compile-time constants** rather than passed as uniforms — this
gives the WGSL compiler full freedom to specialize and means each shape
combination produces a distinct shader. Fine for our static-shape model.

**Dispatch:** WebGPU caps each dimension at 65535 workgroups. The runtime
dispatches as `(min(N, 65535), ceil(N/65535), 1)`; kernels compute their
global index as `gid.x + gid.y * (65535 * workgroup_size)`. Workgroup size
is 256 — large enough that our biggest kernel (~8M threads in
`matmul_bwd_dW`) fits in 1D.

**Error reporting:** `runtime.ts` wraps each pipeline creation in
`pushErrorScope('validation')` and pulls `getCompilationInfo()` on
failure, so shader bugs surface with file/line/message rather than the
useless "previous error" you get when a broken pipeline is dispatched.

## Buffer plan

`planBuffers(graph, paramGrads, writebacks)` walks every tensor and
categorizes it:

| Kind | Purpose | Lifetime |
|---|---|---|
| `param` | trainable parameter | persistent |
| `param_grad` | gradient w.r.t. a param | one step |
| `state` | optimizer state (Adam m, v) | persistent |
| `tensor_input` | data input (tokens, targets) | one step |
| `intermediate` | any other op output | one step |
| `output` | exposed graph output (loss) | one step |

State buffers are zero-initialized at runtime creation. Writebacks (declared
by `appendAdam`) describe end-of-step `copyBufferToBuffer` operations from
freshly-computed values into their persistent homes.

## Autograd

`appendGrad(graph)` walks the forward ops in reverse and emits backward ops
into the same graph. Each op's transpose rule is hand-written in
`grad.ts`. The cotangents map (`tensorId → Tensor`) accumulates
contributions from multiple consumers via `add`.

Two notable workarounds:

- **Embedding lookup is implemented as `oneHot @ table`** rather than
  `gather`. Gather has no transpose rule (it'd need scatter-with-atomic-add
  or similar), but `oneHot @ table` decomposes into ops that *do* have
  rules, so autograd works through it for free.
- **`slice_last_range` has no backward yet.** Forward works (used in any
  axis-2 slicing pattern); backward is unimplemented because it'd need a
  scatter-style "place into zeros" op. Workaround: use multiple separate
  matmuls (e.g. `W_q`/`W_k`/`W_v`) instead of a fused `W_qkv`.

## Verification approach

Two layers:

1. **Smoke test** (`pnpm test` → `test/smoke.ts`) — runs in Node without
   GPU. Builds the IR, attaches grad, plans buffers, emits all WGSL, and
   verifies structure (kernel count, binding count, shape errors). Catches
   codegen regressions without needing a browser.
2. **Live samples** (`pnpm --filter samples dev`) — Vite dev server with
   a `/__log` endpoint that streams browser logs to stdout, used during
   development to bypass copy-paste-from-console debugging.

## What this spec is not

A contract. The API will change before 1.0. The load-bearing decisions
are in **Scope and non-goals** and **Key design decisions** above —
those are the points where the design deliberately diverges from JAX or
PyTorch, and where reverting any of them effectively re-creates the
failure mode that motivated this library.
