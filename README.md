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
  Module, compileModule, nn,
  mul, sub, mean, relu,
  type Tensor,
} from 'tensorgrad'

const B = 256

class MLP extends Module {
  l1 = new nn.Linear(1, 64)
  l2 = new nn.Linear(64, 64)
  l3 = new nn.Linear(64, 1)
}

function modelFwd(m: MLP, x: Tensor): Tensor {
  return m.l3.fwd(relu(m.l2.fwd(relu(m.l1.fwd(x)))))
}

function lossFn(m: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  const diff = sub(modelFwd(m, x), y)
  return mean(mul(diff, diff))
}

const compiled = await compileModule({
  factory: () => new MLP(),
  loss: lossFn,
  optimizer: { kind: 'adam', lr: 0.005 },
  inputs: { x: [B, 1], y: [B, 1] },   // shape tuples; dtype defaults to f32
})

for (let step = 0; step < 1000; step++) {
  const { x, y } = generateBatch()
  const lossVal = await compiled.step({ x, y })
  if (step % 100 === 0) console.log('step', step, 'loss', lossVal)
}
```

## Mental model

- A `Module` subclass declares parameters via `this.param([shape], opts)` and
  composes child modules as plain fields. The class is a tree of params.
- A *forward function* takes the materialized module + a record of named
  input tensors and returns a tensor — the loss for `compileModule`, or any
  output for `compileForward`. Forwards are free functions, not methods.
- `compileModule({ factory, loss, inputs, adam })` traces the forward,
  derives gradients, wires Adam, generates WGSL, spawns a worker, and
  returns a `CompiledModule`. Every method on it is async.

## Porting from PyTorch

If you're translating a PyTorch model or training loop. Assumes the
**Mental model** above.

### Direct mappings

| PyTorch | tensorgrad |
|---|---|
| `class Net(nn.Module): def forward(self, x): ...` | `class Net extends Module { ... }` + a free `forward(m, x)` function |
| `model(x)` | `forward(m, x)` |
| `linear(x)` on `nn.Linear` / `nn.LayerNorm` | `linear.fwd(x)` (`.fwd` is the convention for built-in leaf modules) |
| `model.parameters()` | `compiled.paramNames`, `compiled.downloadParams()` |
| `optimizer.zero_grad(); out = model(x); loss = ...; loss.backward(); optimizer.step()` | `await compiled.step(inputs)` — forward + backward + Adam update are fused |
| `optim.Adam(params, lr=...)` | `optimizer: { kind: 'adam', lr }` in `compileModule({ ... })` |
| `optim.SGD(params, lr=..., momentum=..., nesterov=...)` | `optimizer: { kind: 'sgd', lr, momentum?, nesterov? }` in `compileModule({ ... })` |
| `StepLR(opt, step_size=N, gamma=g)` | `lr.step({ peak, stepSize: N, gamma: g })` |
| `MultiStepLR(opt, milestones=[..], gamma=g)` | `lr.multiStep({ peak, milestones: [..], gamma: g })` |
| `CosineAnnealingLR(opt, T_max=N, eta_min=m)` | `lr.cosineDecay({ peak, final: m, steps: N })` |
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
| `nn.Embedding(V, D)` | `new nn.Embedding(V, D)` — `.fwd(idx)` returns `[..., D]` |
| `torch.flatten(x, start_dim=1)` | `flatten(x, 1)` (or `reshape(x, [B, -1])`) |
| `nn.Conv2d(in, out, k, stride=s, padding=p)` | `new nn.Conv2d(in, out, k, { stride: s, padding: p })` |
| `F.max_pool2d(x, k, stride=s, padding=p)` | `maxPool2d(x, k, { stride: s, padding: p })` |
| `F.interpolate(x, scale_factor=k, mode='nearest')` | `nearestUpsample2d(x, k)` |
| `torch.randn(shape)` | `randn(shape)` — uses the per-step PRNG; zero gradient |
| `x ** 2` / `x.square()` | `square(x)` |
| `torch.sin(x)` / `torch.cos(x)` | `sin(x)` / `cos(x)` |
| `torch.take(t, idx)` / 1-D `torch.gather` | `take(table, indices)` — for 1-D tables; use `embedding` for 2-D |

### Things that aren't 1-to-1

**Pass raw logits to the loss, not log-probs.** PyTorch tutorials often
write `F.log_softmax(logits, dim=-1)` in `forward` and `F.nll_loss(...)`
in the loss. Tensorgrad's `nn.crossEntropy(logits, targets)` fuses
log-softmax + NLL into one call. Pass raw logits — don't apply
log-softmax yourself. Applying it twice silently
double-log-softmaxes; the model trains but converges to garbage. This
is the worst class of bug: it runs.

If you specifically want the log-probability intermediate visible (e.g.
to `capture` it for inspection), use `nn.nllLoss(logSoftmax(logits),
targets)` instead — same numerics, just unfused.

**No `.train()` / `.eval()` mode flag.** Write two forwards: a training
one (`lossFn`, includes `dropout` etc.) and an inference one
(`predictFn`, deterministic). Compile training with `compileModule`,
then call `compiled.compileForward({ forward: predictFn, ... })` for the
shared-params inference graph. Stochastic ops are physically absent from
the inference graph.

**No eager mode.** The forward is traced once and compiled. To read an
intermediate, mark it with `capture(name, t)` inside the forward and
call `step` / `run` with `{ withCaptures: true }`.

**Tensorgrad runs in a worker.** Every method on a compiled module is
async. Cancellation (e.g. `replaceModel` while a `step` is in flight)
shows up as a rejected promise with `name === 'AbortError'`; pass
`{ abortAsValue: true }` to get a discriminated result instead of having
to try/catch.

## Public API

### Compile entry points

```ts
compileModule({ factory, loss, inputs, optimizer? }): Promise<CompiledModule>
compiled.compileForward({ forward, inputs }): Promise<CompiledForwardModule>
compiled.replaceModel(newFactory, { seed?, optimizer? }): Promise<void>
isWebGPUAvailable(): boolean              // friendly pre-flight check
```

There's one entry point: `compileModule`. Inference graphs are created
via the `compileForward` method on the training compile — they share its
worker (one GPUDevice) and its param GPUBuffers, so training-step updates
are immediately visible to inference:

```ts
const train = await compileModule({
  factory: () => new Model(),
  loss: lossFn,
  inputs: { tokens: [B, T], targets: [B, T], mask: [T] },
  optimizer: { kind: 'adam', lr: 0.001 },
})

const infer = await train.compileForward({
  forward: predictFn,
  inputs: { tokens: { shape: [null, T], dtype: 'i32' } },  // null = parametric batch
})
```

**Parametric batch dim.** When you need the same forward function at
multiple batch sizes (B=1 for live prediction, B=256 for held-out eval),
mark the dim as `null` and the proxy compiles + caches a sibling graph
per actual size on demand:

```ts
const infer = await train.compileForward({
  forward: predictFn,
  inputs: { x: [null, 784] },   // batch is parametric
})
await infer.run({ x: arr1 })       // first call at B=1 → compile + cache
await infer.run({ x: arr256 })     // first call at B=256 → compile + cache
await infer.run({ x: arr1Again })  // cache hit
```

Wildcards follow the TF/ONNX/MLIR convention: `null` for an inferred dim.
One `null` per shape (multi-wildcard isn't exposed yet). The first `run()`
at each new shape pays the trace + codegen cost; the cache grows
unbounded, so for latency-sensitive paths warm the cache at startup with
a dummy `run()` per expected shape.

**Replacing the model.** If your UI lets the user change the model
topology (layer count, hidden width, etc.), `replaceModel(newFactory)`
swaps it in place — same handle, same worker. Sibling forward proxies
created via `compileForward` stay registered; their per-shape kernel
caches are cleared and recompile lazily on the next `run()`:

```ts
await compiled.replaceModel(() => new MLP(newLayerSpec))
// compiled and any forward proxies are still valid.

// Update optimizer config atomically with the swap (e.g. user also
// changed LR via a UI control):
await compiled.replaceModel(
  () => new MLP(newLayerSpec),
  { optimizer: { kind: 'adam', lr: 0.005 } },
)
```

For mid-training optimizer changes *without* a topology swap (LR
schedule update on the existing weights), use `setOptimizerConfig`.

### `singleFlight` (live-preview helper)

For live-preview patterns where stale calls (earlier mouse positions,
partial drawings) should be dropped in favor of the newest, wrap a
promise-returning function with `singleFlight`. Matches RxJS `switchMap` /
p-debounce semantics: displaced callers reject with `AbortError`; only
the most recent call resolves.

```ts
import { singleFlight } from 'tensorgrad'

const predict = singleFlight((tokens: Int32Array) => infer.run({ tokens }))

canvas.addEventListener('pointermove', async () => {
  try { updateUI(await predict(latestTokens())) }
  catch (e: any) { if (e?.name !== 'AbortError') throw e }
})
```

Generic — works around `run`, `step`, or any single-argument promise function.

### CompiledModule methods (all `Promise`-returning)

```ts
compiled.step(inputs)                           // → loss: number
compiled.step(inputs, { withCaptures: true })   // → { loss, captures }
compiled.step(inputs, { abortAsValue: true })     // → { kind: 'ok', loss } | { kind: 'aborted' }
compiled.run(inputs)                            // → Float32Array
compiled.run(inputs, { withCaptures: true })    // → { output, captures }
compiled.run(inputs, { abortAsValue: true })      // → { kind: 'ok', output } | { kind: 'aborted' }
compiled.uploadParams(record, { partial? })
compiled.downloadParams()                       // → ParamTree<M> (typed tree, mirrors class)
compiled.downloadParamsFlat()                   // → Record<'layers.0.W' | …, Float32Array>
compiled.downloadParamGrads()                   // → ParamTree<M> (same tree shape)
compiled.reset()                                // re-init params + zero Adam state
compiled.resetOptimizerState()
compiled.setOptimizerConfig({ lr })             // mutate LR without recompile
compiled.compileForward({ forward, inputs })    // sibling forward graph
compiled.replaceModel(newFactory)               // swap topology, same worker
compiled.destroy()                              // tear down worker + GPU
```

`compiled.graph`, `compiled.kernels`, `compiled.outputShape`,
`compiled.paramNames`, and `compiled.seed` are sync properties for
inspection. Forward proxies expose only `paramNames` (the same names as
the parent training graph) — output shape isn't stable on a proxy that
caches multiple shape variants.

**Inspecting the compiled IR.** `compiled.graph` exposes ops, tensors,
connectivity, captures, and outputs. `Graph`, `OpNode`, `Tensor`,
`Shape`, `Dtype`, and `CallSite` are exported for walking it. Each
`Tensor.site` carries the user-frame stack from op-call time, useful for
"where in user code did this op come from" displays.

```ts
import type { Graph } from 'tensorgrad'

// List parameters with shapes.
const params = compiled.graph.ops
  .filter(op => op.kind === 'param_input')
  .map(op => ({ name: op.name, shape: compiled.graph.tensors[op.out].shape }))
// [{ name: 'l1.W', shape: [1, 64] }, { name: 'l1.b', shape: [64] }, ...]
```

**Standalone IR (no worker, no GPU).** `compileToIR` is the same pipeline
without the worker spawn — trace + autograd + buffer plan + codegen,
returning a `CompiledIR` synchronously. Useful for unit tests that walk
the graph, op-count regressions in CI, or environments without WebGPU.

```ts
import { compileToIR, paramInput, tensorInput, matmul, mean, mul } from 'tensorgrad'

const ir = compileToIR(() => {
  const W = paramInput('W', [4, 8])
  const x = tensorInput('x', [16, 4])
  const y = matmul(x, W)
  return mean(mul(y, y))            // scalar loss; appendGrad runs internally
})

ir.graph.ops.length             // count of fwd + bwd ops
ir.kernels.length               // emitted WGSL kernels
Object.keys(ir.paramGrads)      // [ 'W' ] — gradient tensors keyed by param name
ir.paramGrads.W.shape           // [4, 8] — matches the param
```

**Typed param tree.** `downloadParams()` returns a tree that mirrors the
model class. If `MLP` has `l1: Linear; l2: Linear; l3: Linear` and `Linear`
has `W: Tensor; b: Tensor`, then the return type is
`{ l1: { W, b }; l2: { W, b }; l3: { W, b } }` with `Float32Array` leaves —
typed access (`params.l1.W`), no string indexing. Arrays of Modules
(e.g. `layers: Linear[]`) become arrays of subtrees
(`params.layers[0].W`). Non-param fields on the model class (numbers,
config, etc.) are pruned. `downloadParamsFlat()` is the escape hatch
returning the legacy `Record<string, Float32Array>` — useful for
serialization, full-tree iteration, or partial re-upload via
`uploadParams`.

**Typed inputs.** `step` / `run` are typed against the declared `inputs`
shape, so each named input expects the right TypedArray: a dtype-`'f32'`
input (or a tuple shape, which defaults to f32) expects a `Float32Array`;
a dtype-`'i32'` input expects an `Int32Array`. Passing the wrong array
type is a compile-time error.

**Wildcard consistency.** Every `null` wildcard across all inputs in a
single `run()` must resolve to the same value (matches Keras `None` /
ONNX dynamic-axis convention). Mismatched inferred dims throw at the
call boundary, not deep in kernel dispatch.

**Cancellation as value.** If your UI tears down or rebuilds the model
while a `step` / `run` is in flight (e.g. the user picks a new layer
size mid-training, triggering `replaceModel`), the in-flight call is
aborted. By default it rejects with `AbortError`; pass `{ abortAsValue:
true }` to get a discriminated result instead:

```ts
const r = await compiled.step(batch, { abortAsValue: true })
if (r.kind === 'aborted') return    // graph was replaced; nothing to do
useLoss(r.loss)
```

Composes with `{ withCaptures: true }`: the `'ok'` branch carries
`captures`, the `'aborted'` branch carries no payload.

**Factory hygiene.** `factory` must return a *fresh* `Module` each call —
the pipeline mutates `ParamSentinel` fields into `Tensor`s on first
compile. Returning the same instance twice throws a clear error.

**Reproducible init.** A deterministic Mulberry32 PRNG seeds compile-time
init. Pass `seed` to control it; whatever seed was used is exposed as
`compiled.seed` so you can replay later:

```ts
const a = await compileModule({ ..., seed: 42 })   // pin
const b = await compileModule({ ... })             // fresh; b.seed exposes it
compiled.reset()                                   // re-inits with the current seed
await compiled.replaceModel(newFactory)            // fresh seed by default
await compiled.replaceModel(newFactory, { seed: compiled.seed })  // keep current
```

### Operators

Imported from `'tensorgrad'`:

- Arithmetic (binary): `add`, `sub`, `mul`, `div`, `min`, `max`
- Unary math: `sqrt`, `rsqrt`, `log`, `exp`, `neg`, `abs`, `square`, `sin`, `cos`
- Activations: `relu`, `tanh`, `sigmoid`, `gelu`, `silu`
- Clamping: `clamp(x, lo, hi)` (scalar bounds)
- Stochastic: `dropout(x, p)` (inverted dropout, p ∈ [0, 1)), `randn(shape)` (N(0, 1) sampler, zero gradient)
- Comparisons / select: `less`, `greater`, `where`
- Reductions: `mean(x, axis?, { keepDims? })`, `sum(x, axis?, { keepDims? })`, `argmax(x, axis?)`, `argmin(x, axis?)`
- Shape: `reshape`, `permute`, `swapAxes` (`permute` is full-axis reorder, like PyTorch's `permute` / JAX's `jnp.transpose`)
- Attention layout: `splitHeads(x, nHeads)`, `mergeHeads(x)`
- Linear algebra: `matmul` (dispatches unbatched [..., M, K] · [K, N] vs both-batched [..., M, K] · [..., K, N] on rhs rank)
- Indexing / casting: `oneHot`, `arange`, `embedding`, `take(table, indices)` (1-D-table gather)
- Slicing / structural: `sliceRange(t, axis, start, end)`, `concat(tensors, axis)`, `stack(tensors, axis)`, `split(t, sizes, axis)`
- Fused ML primitives: `softmax(x, axis?)`, `logSoftmax(x, axis?)`, `softmaxCausal(x, axis?)`, `whereCausal`
- 2D conv / pool / upsample (NCHW): `conv2d(input, weight, { stride?, padding? })`, `maxPool2d(x, k, { stride?, padding? })`, `nearestUpsample2d(x, factor)`, `flatten(x, startAxis?)`

`add`, `sub`, `mul`, `div`, `min`, `max`, `less`, `greater` all accept
`(Tensor, Tensor)` or `(Tensor, number)` — scalar broadcasts. `argmax`
and `argmin` return `i32` and are non-differentiable. The standard loss
tail is `nn.crossEntropy(logits, targets)` (reduces to scalar mean by
default).

**Structural ops.** `concat([a, b], axis)` joins along an existing axis;
`stack([a, b], axis)` joins along a new axis (sugar for
`reshape` + `concat`). Negative axes index from the end (Python
convention). Concat is capped at 7 inputs (WebGPU bind-group limit:
8 storage buffers per shader stage minus the output) — chain a second
concat if you need more. `split(t, sizes, axis)` is the inverse, built
from `sliceRange`.

### `nn` namespace

```ts
import { nn } from 'tensorgrad'

nn.Linear(inDim, outDim, { bias? })  // .fwd(x); W: [inDim, outDim], b: [outDim]
nn.LayerNorm(dim)                    // .fwd(x); g (gain) and b (bias) both [dim]
nn.RMSNorm(dim, eps?)                // .fwd(x); g (gain) only — Llama-style
nn.Embedding(vocab, dim)             // .fwd(idx); W: [vocab, dim]; idx is i32 [...]
nn.Conv2d(inC, outC, k, { stride?, padding?, bias? }) // .fwd(x); NCHW
                                     // x: [B, inC, H, W] -> [B, outC, H', W']
nn.crossEntropy(logits, targets, { reduction? })  // fused log-softmax + NLL; default mean
nn.nllLoss(logProbs, targets, { reduction? })     // NLL only; pair with logSoftmax for the log-prob intermediate
```

Convention: leaf modules (`Linear`, `LayerNorm`) expose `.fwd(x)` for ergonomic
chaining. Composite modules you write yourself are typically free functions
taking `(p: ModuleType, x: Tensor)`.

`crossEntropy` and `nllLoss` reduce to a scalar mean by default (matches
PyTorch's `F.cross_entropy(..., reduction='mean')`). Pass `{ reduction:
'none' }` for a per-position tensor when you need to mask or weight
positions yourself before reducing; `'sum'` for an unscaled sum.

Multi-head shape helpers (`splitHeads(x, nHeads)`, `mergeHeads(x)`) and
the captures-side `unsplitHeads(captures, name)` live at the top level
(not under `nn`) — they're pure tensor / readback ops, not modules.

### Optimizers

`compileModule` takes an `optimizer` discriminated by `kind: 'adam' | 'sgd'`.
Both kinds accept the same LR schedule shapes from the `lr` namespace.

```ts
import { lr } from 'tensorgrad'

// Adam / AdamW
optimizer: { kind: 'adam', lr: 0.005 }
optimizer: { kind: 'adam', lr: 0.005, weightDecay: 0.01 }
optimizer: { kind: 'adam', lr: 0.005, weightDecay: 0.01, decayFilter: n => n.endsWith('.W') }
optimizer: { kind: 'adam', lr: 0.005, clipGradNorm: 1.0 }

// SGD / SGD-with-momentum / Nesterov. Plain SGD when momentum is 0 (default).
optimizer: { kind: 'sgd', lr: 0.05 }
optimizer: { kind: 'sgd', lr: 0.05, momentum: 0.9 }
optimizer: { kind: 'sgd', lr: 0.05, momentum: 0.9, nesterov: true }
optimizer: { kind: 'sgd', lr: 0.05, weightDecay: 5e-4 }   // PyTorch-style L2 (injected into gradient)
```

### LR schedules (`lr` namespace)

```ts
optimizer: { kind: 'adam', lr: lr.linearDecay({ peak: 0.005, final: 0.0005, steps: 1500 }) }
optimizer: { kind: 'adam', lr: lr.cosineDecay({ peak: 0.005, final: 0.0001, steps: 5000 }) }
optimizer: { kind: 'sgd',  lr: lr.cosineDecay({ peak: 0.1, final: 0.001, steps: 10000 }), momentum: 0.9 }
optimizer: { kind: 'adam', lr: lr.warmup({ peak: 0.001, warmupSteps: 200, after: 0.001 }) }
optimizer: { kind: 'adam', lr: lr.step({ peak: 1.0, stepSize: 1, gamma: 0.7 }) }            // PyTorch StepLR
optimizer: { kind: 'adam', lr: lr.multiStep({ peak: 0.1, milestones: [30000, 60000], gamma: 0.1 }) }  // MultiStepLR
```

LR schedules are serializable shapes, not closures (they cross the worker
boundary). Use a `number` for constant LR, or one of the constructors above.

### `setOptimizerConfig` (mid-training)

Update the learning rate live, without recompiling. Works for both Adam
and SGD graphs. The step counter is preserved.

```ts
await compiled.setOptimizerConfig({ lr: 0.001 })
await compiled.setOptimizerConfig({
  lr: lr.cosineDecay({ peak: 0.001, final: 1e-5, steps: 5000 }),
})  // non-constant schedules auto-rebase so step 1 = next training step
```

Which params receive weight decay is baked at compile time (per-param
`{ decay: true | false }` metadata). To change `weightDecay`, `b1`, `b2`,
or any other non-LR hyperparameter, recompile via `replaceModel`.

### Gradient clipping

Global L2-norm clipping matches PyTorch's `clip_grad_norm_` and optax's
`clip_by_global_norm`. Set `AdamConfig.clipGradNorm` for the common case:

```ts
const compiled = await compileModule({
  ...,
  optimizer: { kind: 'adam', lr: 0.001, clipGradNorm: 1.0 },   // bake clipping into the graph
})
```

The clip is **global** across all params (one shared scale factor),
applied between backward and the Adam update. Constant at compile time
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
  return nn.crossEntropy(m.l2.fwd(h), y)
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
const { output, captures } = await compiled.run(inputs, { withCaptures: true })
const attn0 = captures.get('attn.0')        // Float32Array
captures.shapeOf('attn.0')                  // readonly number[]
```

Captures are zero-overhead unless `{ withCaptures: true }` is passed; they
add a single batched mapAsync on the readback.

## Constraints

The library is small because of what it doesn't do. Plan accordingly:

- **WebGPU only.** No Wasm, WebGL, or native fallback.
- **Static shapes.** Every shape is fixed at compile time. Changing a batch
  size means recompiling.
- **`f32` only.** No mixed precision. Inputs may be `i32` for indices.
- **One transformation: `grad`.** No `vmap`, `pmap`, `jvp`, `custom_vjp`.
  Batch your data explicitly.
- **Loss must be a scalar.** `compileModule`'s forward returns a rank-0 tensor.
- **Closures don't cross the worker boundary.** LR schedules and inits are
  serializable shapes, not functions. Anything per-step you write into a
  user-defined optimizer (see *Extending* below) follows the same rule.
- **One model per `compileModule` call.** Sibling forward graphs share params
  via the method form; otherwise each compile spawns its own worker.

## Extending

The IR is open. Adam is built in only because it's the most common starting
point — other optimizers, custom losses, or extra ops are user code following
the same pattern as `appendAdam`:

```ts
import { appendAdam, appendGrad, compileToIR } from 'tensorgrad'
```

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
