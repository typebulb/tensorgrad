# tensorgrad

A tiny TypeScript-native tensor library with autograd that compiles to WebGPU.
For training small models in the browser without hand-writing WGSL kernels and
without dragging in a multi-megabyte ML framework. Zero dependencies. Static
shapes, `f32` parameters with `i32` indices, Adam / AdamW optimizer, forward +
reverse-mode autograd. Browser-only. All GPU work runs in a library-internal
Web Worker — every method on a compiled module returns a `Promise`.

```sh
npm i tensorgrad
```

## Minimal example

A 2-layer MLP fitting `y = sin(x)`:

```ts
import {
  Module, compile, Linear,
  sub, mean, square, relu,
  type Tensor,
} from 'tensorgrad'

const B = 256

class MLP extends Module {
  l1 = new Linear(1, 64)
  l2 = new Linear(64, 64)
  l3 = new Linear(64, 1)
}

function modelFwd(m: MLP, x: Tensor): Tensor {
  return m.l3.fwd(relu(m.l2.fwd(relu(m.l1.fwd(x)))))
}

function lossFn(m: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  return mean(square(sub(modelFwd(m, x), y)))
}

const train = await compile({
  model: new MLP(),
  loss: lossFn,
  optimizer: { kind: 'adam', lr: 0.005 },
  inputs: { x: [B, 1], y: [B, 1] },   // shape tuples; dtype defaults to f32
})

for (let step = 0; step < 1000; step++) {
  const { x, y } = generateBatch()
  const r = await train.step({ x, y })
  if (r.kind === 'completed' && step % 100 === 0) {
    console.log('step', step, 'loss', r.loss)
  }
}
```

## Mental model

- A `Module` subclass declares parameters via `this.param([shape], opts)` and
  composes child modules as plain fields. The class is a tree of params.
- A *forward function* takes the materialized module + a record of named
  input tensors and returns a tensor — the loss for a training spec, or
  any output for a forward spec. Forwards are free functions, not methods.
- `compile({ model, loss, inputs, optimizer })` traces the forward, derives
  gradients, wires the optimizer, generates WGSL, spawns a worker, and
  returns a `CompiledTraining`. Every method on it is async. For inference
  compiles that share the training graph's params, use
  `train.attach({ forward, inputs })`.

## Porting from PyTorch

If you're translating a PyTorch model or training loop. Assumes the
**Mental model** above.

### Direct mappings

| PyTorch | tensorgrad |
|---|---|
| `class Net(nn.Module): def forward(self, x): ...` | `class Net extends Module { ... }` + a free `forward(m, x)` function |
| `model(x)` | `forward(m, x)` |
| `linear(x)` on `nn.Linear` / `nn.LayerNorm` | `linear.fwd(x)` (`.fwd` is the convention for built-in leaf modules) |
| `model.parameters()` / `model.state_dict()` | `train.paramNames`, `train.downloadParams()` (flat `Record<string, Float32Array>`) |
| `optimizer.zero_grad(); out = model(x); loss = ...; loss.backward(); optimizer.step()` | `await train.step(inputs)` — forward + backward + Adam update are fused |
| `optim.Adam(params, lr=...)` | `optimizer: { kind: 'adam', lr }` in `compile({ ... })` |
| `optim.AdamW(params, lr=..., weight_decay=w)` | `optimizer: { kind: 'adamw', lr, weightDecay: w }` |
| `optim.SGD(params, lr=..., momentum=..., nesterov=...)` | `optimizer: { kind: 'sgd', lr, momentum?, nesterov? }` in `compile({ ... })` |
| `StepLR(opt, step_size=N, gamma=g)` | `lr.step({ peak, every: N, gamma: g })` |
| `MultiStepLR(opt, milestones=[..], gamma=g)` | `lr.multiStep({ peak, milestones: [..], gamma: g })` |
| `CosineAnnealingLR(opt, T_max=N, eta_min=m)` | `lr.cosineAnnealing({ peak, final: m, steps: N })` |
| `LinearLR(opt, …, total_iters=N)` | `lr.linear({ peak, final, steps: N })` |
| `torch.narrow(t, axis, start, length)` | `narrow(t, axis, start, length)` |
| `nn.Dropout(p)` as a child module | `dropout(x, p)` as a free-function call inside the training forward |
| `x.mean(dim=k)` / `x.sum(dim=k)` | `mean(x, k)` / `sum(x, k)` — negative `k` counts from the end |
| `x.mean()` / `x.sum()` | `mean(x)` / `sum(x)` — 0-d scalar |
| `x.mean(dim=k, keepdim=True)` | `mean(x, k, { keepDims: true })` |
| `F.softmax(x, dim=k)` / `F.log_softmax(x, dim=k)` | `softmax(x, k)` / `logSoftmax(x, k)` — both default to last axis |
| Causal-masked softmax (`tril` + `masked_fill` + `softmax`) | `softmaxCausal(scores)` (fused; preferred over composing the mask yourself) |
| `x.argmax(dim=k)` / `x.argmin(dim=k)` | `argmax(x, k)` / `argmin(x, k)` (default to last axis; flat over the whole tensor with no axis) |
| `x.transpose(a, b)` | `swapAxes(x, a, b)` (NumPy/JAX call this `swapaxes`; tensorgrad matches them — PyTorch's `transpose` is the cross-library outlier) |
| `x.permute(*dims)` | `permute(x, [...])` (NumPy/JAX semantics: full-axis reorder) |
| `x.view(B, T, H, -1)` / `x.reshape(B, -1)` | `reshape(x, [B, T, H, -1])` — exactly one `-1` allowed, inferred from total size |
| `torch.matmul(a, b)` / `a @ b` | `matmul(a, b)` — dispatches between unbatched and batched on rhs rank |
| `torch.split(x, sizes, dim)` | `split(x, sizes, dim)` |
| `nn.Embedding(V, D)` | `new Embedding(V, D)` — `.fwd(idx)` returns `[..., D]` |
| `torch.flatten(x, start_dim=1)` | `flatten(x, 1)` (or `reshape(x, [B, -1])`) |
| `nn.Conv2d(in, out, k, stride=s, padding=p)` | `new Conv2d(in, out, k, { stride: s, padding: p })` |
| `F.max_pool2d(x, k, stride=s, padding=p)` | `maxPool2d(x, k, { stride: s, padding: p })` |
| `F.interpolate(x, scale_factor=k, mode='nearest')` | `nearestUpsample2d(x, k)` |
| `torch.randn(shape)` | `randn(shape)` — uses the per-step PRNG; zero gradient |
| `x.detach()` / `torch.no_grad()` (for a single tensor) | `stopGradient(x)` |
| `x ** 2` / `x.square()` | `square(x)` |
| `torch.sin(x)` / `torch.cos(x)` | `sin(x)` / `cos(x)` |
| `torch.gather(input, dim, index)` / `jnp.take_along_axis(arr, idx, axis)` | `takeAlongAxis(input, indices, axis)` — same-rank, NumPy/JAX naming |

### Things that aren't 1-to-1

**Pass raw logits to the loss, not log-probs.** PyTorch tutorials often
write `F.log_softmax(logits, dim=-1)` in `forward` and `F.nll_loss(...)`
in the loss. Tensorgrad's `crossEntropy(logits, targets)` fuses
log-softmax + NLL into one call. Pass raw logits — don't apply
log-softmax yourself. Applying it twice silently
double-log-softmaxes; the model trains but converges to garbage. This
is the worst class of bug: it runs.

If you specifically want the log-probability intermediate visible (e.g.
to `capture` it for inspection), use `nllLoss(logSoftmax(logits),
targets)` instead — same numerics, just unfused.

**No `.train()` / `.eval()` mode flag.** Write two forwards: a training
one (`lossFn`, includes `dropout` etc.) and an inference one
(`predictFn`, deterministic). Compile each as its own spec; attach the
inference graph to the training compile via `train.attach({ ... })`
so it reuses the training compile's param buffers. Stochastic ops are
physically absent from the inference graph.

```ts
const model = new Model()
const train = await compile({ model, loss: lossFn, inputs, optimizer })
const infer = await train.attach({ forward: predictFn, inputs: inferInputs })
```

Keep the two bodies separate rather than factoring shared helpers
between them: side-effecting ops (`dropout`, `capture`) in a shared
helper leak into both graphs.

**No eager mode.** The forward is traced once and compiled. To read an
intermediate, mark it with `capture(name, t)` inside the forward; the
activation surfaces on the result's `captures` field every call. Graphs
with no `capture()` sites pay nothing.

**Tensorgrad runs in a worker.** Every method on a compiled module is
async. `step` and `infer.run` return a discriminated result:

```ts
const r = await train.step({ x, y })
switch (r.kind) {
  case 'completed': useLoss(r.loss); break  // r.captures also available
  case 'aborted':   return                  // graph was replaced mid-flight
}
```

No try/catch on `AbortError` needed — the cancellation surfaces as the
`'aborted'` discriminator across `step`, `run`, and the `singleFlight`
helper (below) for displaced live-preview callers. One vocabulary for
the whole library.

## Public API

### Compile entry points

```ts
compile(trainingSpec): Promise<CompiledTraining>
train.attach(forwardSpec): Promise<CompiledForward>      // shares worker + param buffers
train.replaceModel(newModel, { seed?, optimizer? }): Promise<void>
isWebGPUAvailable(): boolean                             // friendly pre-flight check
```

`compile()` is the worker-spawning executor; `train.attach()` adds a
sibling forward graph that shares the training compile's worker and
param buffers. Both take plain options objects — types are inferred
from the model + forward function, so you rarely need to import them:

```ts
const model = new Model()

const train = await compile({
  model,
  loss: lossFn,
  inputs: { tokens: [B, T], targets: [B, T], mask: [T] },
  optimizer: { kind: 'adam', lr: 0.001 },
})

const infer = await train.attach({
  forward: predictFn,
  inputs: { tokens: { shape: [null, T], dtype: 'i32' } },  // null = parametric batch
})
```

The training `model` is cloned internally before tracing; the user's
instance is never mutated. The forward function reads the parent's
params — every training step is immediately visible through `infer.run()`.

Training-step updates are immediately visible through `infer.run()` —
the forward compile binds the training compile's actual param buffers,
no readback round-trip.

**Parametric batch dim.** When you need the same forward function at
multiple batch sizes (B=1 for live prediction, B=256 for held-out eval),
mark the dim as `null` and the proxy compiles + caches a sibling graph
per actual size on demand:

```ts
const infer = await train.attach({
  forward: predictFn, inputs: { x: [null, 784] },
})
await infer.run({ x: arr1 })       // first call at B=1 → compile + cache
await infer.run({ x: arr256 })     // first call at B=256 → compile + cache
await infer.run({ x: arr1Again })  // cache hit
```

Wildcards follow the TF/ONNX/MLIR convention: `null` for an inferred dim.
One `null` per shape (multi-wildcard isn't exposed yet). The first `run()`
at each new shape pays the trace + codegen cost; the cache is LRU-bounded
(default 8 shapes, override via `maxCachedShapes`). For latency-sensitive
paths warm the cache at startup with a dummy `run()` per expected shape.

**Replacing the model.** If your UI lets the user change the model
topology (layer count, hidden width, etc.), `replaceModel(newModel)`
swaps it in place — same handle, same worker. Forward compiles attached
via `train.attach(forwardSpec)` stay registered; their per-shape kernel
caches are cleared and recompile lazily on the next `run()`:

```ts
await train.replaceModel(new MLP(newLayerSpec))
// train and any attached forward compiles are still valid.

// Update optimizer config atomically with the swap (e.g. user also
// changed LR via a UI control):
await train.replaceModel(
  new MLP(newLayerSpec),
  { optimizer: { kind: 'adam', lr: 0.005 } },
)
```

For mid-training optimizer changes *without* a topology swap (LR
schedule update on the existing weights), use `setLR`.

### `singleFlight` (live-preview helper)

A generic concurrency primitive, in the public surface because the async
worker boundary turns naive UI event handlers into bad behavior (queued
strokes, stale predictions racing fresh ones). The same `'completed'` /
`'aborted'` vocabulary used by `step` and `run` composes through it.

For live-preview patterns where stale calls (earlier mouse positions,
partial drawings) should be dropped in favor of the newest, wrap a
promise-returning function with `singleFlight`. Matches RxJS `switchMap` /
p-debounce semantics: at most one in-flight, at most one queued; only
the most recent call resolves with `'completed'`, displaced callers
resolve with `'aborted'`.

```ts
import { singleFlight } from 'tensorgrad'

const predict = singleFlight((tokens: Int32Array) => infer.run({ tokens }))

canvas.addEventListener('pointermove', async () => {
  const r = await predict(latestTokens())
  if (r.kind === 'completed') updateUI(r.value)
})
```

Generic — works around `run`, `step`, or any single-argument promise function.

### CompiledTraining methods (all `Promise`-returning)

```ts
train.step(inputs)                           // → { kind: 'completed', loss, captures } | { kind: 'aborted' }
train.uploadParams(record)                   // partial by default; missing keys keep current values
train.downloadParams()                       // → Record<'layers.0.W' | …, Float32Array> (round-trips through uploadParams)
train.attach(forwardSpec)                    // → CompiledForward; sibling that shares params + worker
train.reset({ params?, optimizer? })         // defaults to both; pass false to skip either
train.setLR(lr)                              // mutate LR without recompile
train.replaceModel(newModel)                 // swap topology, same worker
train.destroy()                              // tear down worker + GPU (cascades to attached forwards)
```

`CompiledForward` (from `train.attach(forwardSpec)`) exposes a narrower
surface: `run`, `uploadParams`, `downloadParams`, `destroy`, and
`paramNames`. Params are shared with the parent training compile, so
reads/writes are visible there too.

```ts
infer.run(inputs)                            // → { kind: 'completed', output, captures } | { kind: 'aborted' }
```

**Concurrent `step` / `run` auto-serialize.** A `run()` issued while a
`step()` is in flight is queued automatically — same worker, same single
output staging buffer; the runtime chains the second call so the two
`mapAsync`s don't collide. Useful for the "training in the background,
refresh preview on every input change" pattern: just fire both — no
manual lock needed.

`train.graph`, `train.kernels`, `train.outputShape`, `train.paramNames`,
and `train.seed` are sync properties for inspection. Forward compiles
expose only `paramNames` (the same names as the parent training graph)
— output shape isn't stable on a proxy that caches multiple shape
variants. Use `await infer.graphFor(inputs)` to fetch the IR at a
specific resolved shape (compiles + caches lazily, like `run`).

**Inspecting the compiled IR.** `train.graph` exposes ops, tensors,
connectivity, captures, and outputs. `Graph`, `OpNode`, `Tensor`,
`Shape`, `Dtype`, and `CallSite` are exported for walking it. Each
`Tensor.site` carries the user-frame stack from op-call time, useful for
"where in user code did this op come from" displays. Use
`getOpInputs(op): readonly number[]` to read the input tensor ids of any
op without re-implementing a switch over every kind — that switch belongs
inside the library, where new op kinds get added.

```ts
import type { Graph } from 'tensorgrad'

// List parameters with shapes.
const params = train.graph.ops
  .filter(op => op.kind === 'param_input')
  .map(op => ({ name: op.name, shape: train.graph.tensors[op.out].shape }))
// [{ name: 'l1.W', shape: [1, 64] }, { name: 'l1.b', shape: [64] }, ...]
```

**Flat param record.** `downloadParams()` returns a flat
`Record<'l1.W' | 'l1.b' | ..., Float32Array>` — dotted keys mirror the
Module class path. Round-trips directly back through `uploadParams`
(e.g. save weights to IndexedDB, load on next visit). The string-literal
union autocompletes in TS, so `params['l1.W']` is typed access without a
separate tree variant.

**Typed inputs.** `step` / `run` are typed against the declared `inputs`
shape, so each named input expects the right TypedArray: a dtype-`'f32'`
input (or a tuple shape, which defaults to f32) expects a `Float32Array`;
a dtype-`'i32'` input expects an `Int32Array`. Passing the wrong array
type is a compile-time error.

**Shape declaration forms.** Two canonical shapes:

```ts
inputs: {
  x:       [B, 784],                              // tuple → f32 (the common case)
  tokens:  { shape: [B, T], dtype: 'i32' },       // object → required for non-f32
}
```

The tuple shorthand is `f32`-only — i32 / bool indices use the object
form. Mixing `null` wildcards for parametric dims works in either form.

**Wildcard consistency.** Every `null` wildcard across all inputs in a
single `run()` must resolve to the same value (matches Keras `None` /
ONNX dynamic-axis convention). Mismatched inferred dims throw at the
call boundary, not deep in kernel dispatch.

**Cancellation as value.** If your UI tears down or rebuilds the model
while a `step` / `run` is in flight (e.g. the user picks a new layer
size mid-training, triggering `replaceModel`), the in-flight call is
aborted. The result discriminator surfaces this without try/catch:

```ts
const r = await train.step(batch)
switch (r.kind) {
  case 'completed': useLoss(r.loss); break  // r.captures also available
  case 'aborted':   return                  // graph was replaced
}
```

`r.captures` lives only on the `'completed'` branch; type narrowing
makes it inaccessible from `'aborted'` paths.

**Model is a value, not a factory.** Pass a `model: new Model()`
instance to `compile({ model })`. The compile pipeline clones the module
tree before tracing, so the same instance can feed both a training compile
and a subsequent `replaceModel` without surprising mutation.

**Reproducible init.** A deterministic Mulberry32 PRNG seeds compile-time
init. Pass `seed` to control it; whatever seed was used is exposed as
`train.seed` so you can replay later:

```ts
const a = await compile({ ..., seed: 42 })   // pin
const b = await compile({ ... })             // fresh; b.seed exposes it
b.reset()                                                  // re-inits with the current seed
await b.replaceModel(newModel)                             // fresh seed by default
await b.replaceModel(newModel, { seed: b.seed })           // keep current
```

### Operators

Imported from `'tensorgrad'`:

- Arithmetic (binary): `add`, `sub`, `mul`, `div`, `min`, `max`
- Unary math: `sqrt`, `rsqrt`, `log`, `exp`, `neg`, `abs`, `square`, `sin`, `cos`
- Activations: `relu`, `tanh`, `sigmoid`, `gelu`, `silu`
- Clamping: `clamp(x, lo, hi)` (scalar bounds)
- Stochastic: `dropout(x, p)` (inverted dropout, p ∈ [0, 1)), `randn(shape)` (N(0, 1) sampler, zero gradient)
- Autograd control: `stopGradient(x)` (identity forward, no-op backward — PyTorch's `.detach()`)
- Comparisons / select: `less`, `greater`, `where`
- Reductions: `mean(x, axis?, { keepDims? })`, `sum(x, axis?, { keepDims? })`, `argmax(x, axis?)`, `argmin(x, axis?)`
- Shape: `reshape`, `permute`, `swapAxes` (`permute` is full-axis reorder, like PyTorch's `permute` / JAX's `jnp.transpose`)
- Attention layout: `splitHeads(x, nHeads)`, `mergeHeads(x)`
- Linear algebra: `matmul` (dispatches unbatched [..., M, K] · [K, N] vs both-batched [..., M, K] · [..., K, N] on rhs rank)
- Indexing / casting: `oneHot`, `arange`, `embedding(table, indices)`, `takeAlongAxis(input, indices, axis)` (general per-axis gather; both array/data first to match PyTorch functional, JAX, NumPy)
- Slicing / structural: `narrow(t, axis, start, length)` (PyTorch `torch.narrow`), `concat(tensors, axis)`, `stack(tensors, axis)`, `split(t, sizes, axis)`
- Fused ML primitives: `softmax(x, axis?)`, `logSoftmax(x, axis?)`, `softmaxCausal(x, axis?)`, `whereCausal(x, fillValue)` (mask below the diagonal; pairs with `softmaxCausal` when you need a non-softmax causal mask)
- 2D conv / pool / upsample (NCHW): `conv2d(input, weight, { stride?, padding? })`, `maxPool2d(x, k, { stride?, padding? })`, `nearestUpsample2d(x, factor)`, `flatten(x, startAxis?)`

`add`, `sub`, `mul`, `div`, `min`, `max`, `less`, `greater` all accept
`(Tensor, Tensor)`, `(Tensor, number)`, or `(number, Tensor)` — scalar
broadcasts on either side. Non-commutative ops (`sub`, `div`, `less`,
`greater`) honor the operand order: `sub(2, x) === 2 - x`. `argmax`
and `argmin` return `i32` and are non-differentiable. The standard loss
tail is `crossEntropy(logits, targets)` (reduces to scalar mean by
default).

**Structural ops.** `concat([a, b], axis)` joins along an existing axis;
`stack([a, b], axis)` joins along a new axis (sugar for
`reshape` + `concat`). Negative axes index from the end (Python
convention). Concat over the WebGPU 7-binding cap is auto-chained
internally — call signature is the same whether you pass 2 or 200
tensors. `split(t, sizes, axis)` is the inverse, built from `narrow`.

### Layer modules and loss helpers

```ts
import {
  Linear, LayerNorm, RMSNorm, Embedding, Conv2d,
  crossEntropy, nllLoss,
} from 'tensorgrad'

new Linear(inDim, outDim, { bias?, init?, decay? })   // .fwd(x); W: [inDim, outDim], b: [outDim]
new LayerNorm(dim, { eps?, bias?, decay? })  // .fwd(x); g (gain) [dim], b (bias) [dim] or null when bias:false
new RMSNorm(dim, { eps?, decay? })           // .fwd(x); g (gain) only — Llama-style
new Embedding(vocab, dim, { init?, decay? })          // .fwd(idx); W: [vocab, dim]; idx is i32 [...]
new Conv2d(inC, outC, k, { stride?, padding?, bias?, init?, decay? }) // .fwd(x); NCHW
                                      // x: [B, inC, H, W] -> [B, outC, H', W']
crossEntropy(logits, targets, { reduction? })  // fused log-softmax + NLL; default mean
nllLoss(logProbs, targets, { reduction? })     // NLL only; pair with logSoftmax for the log-prob intermediate
```

Convention: leaf modules (`Linear`, `LayerNorm`) expose `.fwd(x)` for ergonomic
chaining. Composite modules you write yourself are typically free functions
taking `(p: ModuleType, x: Tensor)`.

`crossEntropy` and `nllLoss` reduce to a scalar mean by default (matches
PyTorch's `F.cross_entropy(..., reduction='mean')`). Pass `{ reduction:
'none' }` for a per-position tensor when you need to mask or weight
positions yourself before reducing; `'sum'` for an unscaled sum.

Multi-head shape helpers (`splitHeads(x, nHeads)`, `mergeHeads(x)`) are
pure tensor ops, not modules. The captures-side counterpart is
`captures.perHead(name)` — a method on the `Captures` instance returned
by `step()` / `run()`, splits a flat capture into one `Float32Array` per
head.

### Optimizers

`compile()` takes an `optimizer` discriminated by
`kind: 'adam' | 'adamw' | 'sgd'`. Splits mirror PyTorch:
`torch.optim.Adam` (no decay) vs `torch.optim.AdamW` (decoupled decay).
All kinds accept the same LR schedule shapes from the `lr` namespace.

```ts
import { lr } from 'tensorgrad'

// Plain Adam
optimizer: { kind: 'adam', lr: 0.005 }
optimizer: { kind: 'adam', lr: 0.005, clipGradNorm: 1.0 }

// AdamW — decoupled weight decay (Loshchilov & Hutter)
optimizer: { kind: 'adamw', lr: 0.005, weightDecay: 0.01 }
optimizer: { kind: 'adamw', lr: 0.005, weightDecay: 0.01, decayFilter: n => n.endsWith('.W') }

// SGD / SGD-with-momentum / Nesterov. Plain SGD when momentum is 0 (default).
optimizer: { kind: 'sgd', lr: 0.05 }
optimizer: { kind: 'sgd', lr: 0.05, momentum: 0.9 }
optimizer: { kind: 'sgd', lr: 0.05, momentum: 0.9, nesterov: true }
optimizer: { kind: 'sgd', lr: 0.05, weightDecay: 5e-4 }   // PyTorch-style L2 (injected into gradient)
```

### LR schedules (`lr` namespace)

```ts
optimizer: { kind: 'adamw', lr: lr.linear({ peak: 0.005, final: 0.0005, steps: 1500 }) }
optimizer: { kind: 'adam',  lr: lr.cosineAnnealing({ peak: 0.005, final: 0.0001, steps: 5000 }) }
optimizer: { kind: 'sgd',   lr: lr.cosineAnnealing({ peak: 0.1, final: 0.001, steps: 10000 }), momentum: 0.9 }
optimizer: { kind: 'adam',  lr: lr.warmup({ peak: 0.001, steps: 200, after: 0.001 }) }
optimizer: { kind: 'adam',  lr: lr.step({ peak: 1.0, every: 1, gamma: 0.7 }) }               // PyTorch StepLR's step_size
optimizer: { kind: 'adam', lr: lr.multiStep({ peak: 0.1, milestones: [30000, 60000], gamma: 0.1 }) }  // MultiStepLR
```

LR schedules are serializable shapes, not closures (they cross the worker
boundary). Use a `number` for constant LR, or one of the constructors above.

### `setLR` (mid-training)

Update the learning rate live, without recompiling. Works for both Adam
and SGD graphs. The step counter is preserved.

```ts
await train.setLR(0.001)
await train.setLR(
  lr.cosineAnnealing({ peak: 0.001, final: 1e-5, steps: 5000 }),
)  // non-constant schedules auto-rebase so step 1 = next training step
```

Which params receive weight decay is baked at compile time (per-param
`{ decay: true | false }` metadata). To change `weightDecay`, `beta1`, `beta2`,
or any other non-LR hyperparameter, recompile via `replaceModel`.

### Gradient clipping

Global L2-norm clipping matches PyTorch's `clip_grad_norm_` and optax's
`clip_by_global_norm`. Set `clipGradNorm` on either the Adam or SGD
optimizer config:

```ts
const compiled = await compile({
  ...,
  optimizer: { kind: 'adam', lr: 0.001, clipGradNorm: 1.0 },   // bake clipping into the graph
})
```

The clip is **global** across all params (one shared scale factor),
applied between backward and the optimizer update. Constant at compile time
— there's no runtime knob to change `clipGradNorm` after compile.

For custom optimizers, `appendGradClip(graph, paramGrads, maxNorm)` is
the composable extension hook — call it before whatever optimizer pass
you append, the same way `appendAdam` does internally.

### Param init (`init` namespace)

```ts
import { init } from 'tensorgrad'

this.param([D, D], { init: init.kaiming() })           // gain=sqrt(2), fan_in=D
this.param([D, D], { init: init.kaiming({ gain: 1 }) })
this.param([D],    { init: init.zeros() })
this.param([D],    { init: init.ones() })
this.param([D, D], { init: init.randn({ scale: 0.02 }) })
this.param([D],    { init: init.literal(myFloat32Array) })
```

Default init is `init.randn()` (std 0.02). AdamW weight decay defaults to
`true` for randn/kaiming/literal init, `false` for zeros/ones — override
per-param with `{ decay: true | false }`.

### Dropout (no mode flag)

`dropout(x, p)` is inverted dropout: elements survive with probability
`1 - p` and are scaled by `1 / (1 - p)`; the rest are zeroed. The mask
is reproducible from the (per-step seed, per-call salt, thread id) via
a PCG hash inside the kernel — backward recomputes the same mask, no
memory cost. The runtime auto-threads the per-step seed; users never
plumb it.

There is no `.train()/.eval()` mode flag. Instead, follow the
free-function-forward pattern tensorgrad already encourages: call
`dropout` inside your *training* forward (`lossFn`), and omit it from
your *inference* forward (`predictFn`). The two functions compile into
separate graphs — dropout is literally absent from the inference path.

```ts
function lossFn(m: Model, { x, y }: { x: Tensor; y: Tensor }) {
  const h = relu(dropout(m.l1.fwd(x), 0.1))    // dropout in training
  return crossEntropy(m.l2.fwd(h), y)
}

function predictFn(m: Model, { x }: { x: Tensor }) {
  const h = relu(m.l1.fwd(x))                  // no dropout
  return m.l2.fwd(h)
}
```

`dropout(x, 0)` short-circuits to identity (no IR node emitted), so a
config-driven `dropout(x, cfg.pDrop)` with `cfg.pDrop === 0` is free.

### Captures (debugging / mech-interp)

Wrap any tensor inside a forward to expose its activation post-run:

```ts
import { capture } from 'tensorgrad'

const attn = capture(`attn.${i}`, softmaxCausal(scores))
```

```ts
const r = await infer.run(inputs)
if (r.kind === 'completed') {
  const attn0 = r.captures.get('attn.0')        // Float32Array
  r.captures.shape('attn.0')                    // readonly number[]
}
```

Captures are zero-overhead when the graph has no `capture()` sites.
When it does, they're read back via a single batched `mapAsync`
alongside the loss/output — no opt-in flag, the activation is just
there on the result. A capture site in a *training* forward therefore
costs a readback on every `train.step()` (megabytes for transformer
activations); keep them in inference-only forwards.

## Constraints

The library is small because of what it doesn't do. Plan accordingly:

- **WebGPU only.** No Wasm, WebGL, or native fallback.
- **Static shapes.** Every shape is fixed at compile time. Changing a batch
  size means recompiling.
- **`f32` only.** No mixed precision. Inputs may be `i32` for indices.
- **One transformation: `grad`.** No `vmap`, `pmap`, `jvp`, `custom_vjp`.
  Batch your data explicitly.
- **Only `lr` is hot-mutable.** `weightDecay`, `beta1`, `beta2`, `clipGradNorm`,
  and which params receive decay are baked at compile time. Use `setLR` for
  live LR changes; everything else needs `replaceModel({ optimizer })`.
- **Loss must be a scalar.** A training spec's `loss` returns a rank-0 tensor.
- **Closures don't cross the worker boundary.** LR schedules and inits are
  serializable shapes, not functions. Anything per-step you write into a
  user-defined optimizer (see *Extending* below) follows the same rule.
- **One model per training compile.** Forward specs attach via
  `train.attach(forwardSpec)` to share params; otherwise each `compile()`
  of a training spec spawns its own worker.

## Extending

The IR is open. Adam is built in only because it's the most common starting
point — other optimizers, custom losses, or extra ops are user code following
the same pattern as `appendAdam`:

```ts
import { appendAdam, appendGrad } from 'tensorgrad/internal'
```

Anything under `tensorgrad/internal` is an extension hook that tracks the
IR — expect breakage on 0.0.x bumps. The `tensorgrad` (public) surface is
stable within 0.0.x; `internal` is not.

A custom optimizer is a function that takes the autograd output (graph +
`paramGrads`) and the materialized param tensors, appends its update ops
to the graph, and returns writeback declarations the buffer planner uses
to wire each new value back into its persistent home. SGD, Lion, RMSProp
all fit this shape; see `src/adam.ts` for the canonical example.

The same applies to ops: anything missing from the built-in set can be
expressed as a composition of existing ops (GELU, RMSNorm, etc. are a few
lines), or — if you need a new primitive — added to the IR with a
forward + backward + WGSL emit.

## Potential future additions

**Activation patching.** The dual of `capture` — a `patch(name, t)`
marker that exposes any intermediate for *write* at runtime, the way
`capture` exposes it for read. Sites are declared in the forward;
whether they're active and what values they carry are per-`run()` /
`step()` inputs. Covers mech-interp's core ablation toolkit (zero
ablation, mean ablation, cross-input transplant) without per-experiment
recompilation. Free when sites are inactive.

## When not to use this

- **Inference of pretrained models** → use ONNX Runtime Web or
  transformers.js.
- **Server-side training** → use PyTorch or JAX.

## License

MIT
