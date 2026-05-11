# tensorgrad

A tiny TypeScript-native tensor library with autograd that compiles to WebGPU.
For training small models in the browser without hand-writing WGSL kernels and
without dragging in a multi-megabyte ML framework.

```sh
npm i tensorgrad
```

Roughly 3000 lines of zero-dependency TypeScript. Static shapes, `f32`, Adam
optimizer, ~25 ops, forward + reverse-mode autograd. Browser-only (uses
WebGPU). All training/inference work runs in a library-internal Web Worker —
every method on a compiled module returns a `Promise`.

## Minimal example

A 2-layer MLP fitting `y = sin(x)`:

```ts
import {
  Module, compileModule, init,
  add, mul, sub, sumLast, reshape, matmul, relu,
  type Tensor,
} from 'tensorgrad'

const B = 256

class Linear extends Module {
  W: Tensor; b: Tensor
  constructor(public inDim: number, public outDim: number) {
    super()
    this.W = this.param([inDim, outDim], { init: init.kaiming() })
    this.b = this.param([outDim], { init: 'zeros' })
  }
}

class MLP extends Module {
  l1 = new Linear(1, 64)
  l2 = new Linear(64, 64)
  l3 = new Linear(64, 1)
}

const linearFwd = (p: Linear, x: Tensor) => add(matmul(x, p.W), p.b)

function modelFwd(m: MLP, x: Tensor): Tensor {
  return linearFwd(m.l3, relu(linearFwd(m.l2, relu(linearFwd(m.l1, x)))))
}

function lossFn(m: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  const diff = sub(modelFwd(m, x), y)
  return mul(sumLast(reshape(mul(diff, diff), [B])), 1 / B)
}

const compiled = await compileModule({
  factory: () => new MLP(),
  loss: lossFn,
  adam: { lr: 0.005 },
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
  input tensors and returns a tensor (the loss for `compileModule`, or any
  output for `compileForward`).
- `compileModule({ factory, loss, inputs, adam })` traces the forward,
  derives gradients, wires Adam, plans buffers, generates WGSL, spawns a
  worker, and returns a `CompiledModule`. The factory `() => new Model()`
  is invoked once per compile (and again per shape variant of any
  polymorphic sibling forward); the model instance is consumed (its
  param sentinels are mutated into `Tensor`s).
- Every method on the compiled module is async. `await compiled.step(...)`
  resolves with the loss after the worker's GPU work finishes.

## Public API

### Compile entry points

```ts
compileModule({ factory, loss, inputs, adam? }): Promise<CompiledModule>
compiled.compileForward({ forward, inputs }): Promise<CompiledForwardModule>
compiled.replaceModel(newFactory): Promise<void>
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
  adam: { lr: 0.001 },
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
a dummy `run()` per expected shape. Before the first `run()`,
`infer.kernelCount` reports `0`, `infer.outputShape` is `[]`, and
`infer.ir` is `undefined`.

**Replacing the model.** If your UI lets the user change the model
topology (layer count, hidden width, etc.), `replaceModel(newFactory)`
swaps it in place — same handle, same worker. Sibling forward proxies
created via `compileForward` stay registered; their per-shape kernel
caches are cleared and recompile lazily on the next `run()`:

```ts
await compiled.replaceModel(() => new MLP(newLayerSpec))
// compiled, infer, predictDebounced — all still valid.
```

For live-preview patterns where stale intermediate inputs (earlier mouse
positions, partial drawings) should be dropped in favor of the newest
user state, wrap your `run` call with the `singleFlight` utility:

```ts
import { singleFlight } from 'tensorgrad'

const predict = singleFlight((tokens: Int32Array) => infer.run({ tokens }))

canvas.addEventListener('pointermove', async () => {
  try {
    const out = await predict(latestTokens())
    updateUI(out)
  } catch (e: any) {
    if (e?.name === 'AbortError') return  // superseded — newer call will update
    throw e
  }
})
```

`singleFlight` matches the RxJS `switchMap` / p-debounce convention:
displaced callers reject with `AbortError`; only the most recent call
actually runs when the in-flight one finishes. It's a generic async
helper — works around `run`, `step`, or any other promise-returning
function with a single argument.

### CompiledModule methods (all `Promise`-returning)

```ts
compiled.step(inputs)                           // → loss: number
compiled.step(inputs, { withCaptures: true })   // → { loss, captures }
compiled.run(inputs)                            // → Float32Array
compiled.run(inputs, { withCaptures: true })    // → { output, captures }
compiled.uploadParams(record, { partial? })
compiled.downloadParams()                       // → Record<name, Float32Array>
compiled.downloadParamGrads()                   // → Record<name, Float32Array>
compiled.reset()                                // re-init params + zero Adam state
compiled.resetOptimizerState()
compiled.setOptimizerConfig({ lr?, weightDecay?, b1?, b2? })  // mutate without recompile
compiled.compileForward({ forward, inputs })    // sibling forward graph
compiled.replaceModel(newFactory)               // swap topology, same worker
compiled.destroy()                              // tear down worker + GPU
```

`compiled.kernelCount`, `compiled.outputShape`, `compiled.paramNames`, and
`compiled.ir` are sync properties for inspection. Forward proxies expose
only `paramNames` (the same names as the parent training graph) — kernel
count and output shape aren't stable on a proxy that caches multiple
shape variants.

**Typed inputs.** `step` / `run` are typed against the declared `inputs`
shape, so each named input expects the right TypedArray: a dtype-`'f32'`
input (or a tuple shape, which defaults to f32) expects a `Float32Array`;
a dtype-`'i32'` input expects an `Int32Array`. Passing the wrong array
type is a compile-time error.

**Wildcard consistency.** When more than one input declares a `null`
wildcard, every wildcard in a single `run()` call must resolve to the
same value (matches Keras `None` / ONNX dynamic-axis convention). If
two inputs imply different parametric dims, the proxy throws at the
call boundary rather than letting kernels run with mismatched shapes.

**Factory hygiene.** `compileModule({ factory: ... })` calls the factory
once per compile (and once per shape variant of any polymorphic forward).
Each call must return a *fresh* `Module` instance — the compile pipeline
consumes the instance by mutating its `ParamSentinel` fields into
`Tensor`s. If your factory returns the same instance twice, the second
compile sees Tensors where sentinels are expected; the library detects
this and throws a clear error.

### Operators

Imported from `'tensorgrad'`:

- Element-wise: `add`, `sub`, `mul`, `div`, `sqrt`, `rsqrt`, `log`, `exp`, `relu`
- Comparisons / select: `less`, `greater`, `where`
- Reductions (last axis): `meanLast`, `sumLast`, `sumAll`
- Shape: `reshape`, `transpose`, `swapAxes`
- Linear algebra: `matmul`, `matmulBatched`
- Indexing / casting: `oneHot`, `arange`, `embedding`
- Slicing: `sliceLastRange`
- Fused ML primitives: `softmaxCausalLast`, `logSoftmaxLast`, `whereCausal`

`add`, `sub`, `mul`, `div` accept `(Tensor, Tensor)` or `(Tensor, number)`.

### `nn` namespace

```ts
import { nn } from 'tensorgrad'

nn.Linear(inDim, outDim, { bias? }) // .fwd(x)
nn.LayerNorm(dim)                    // .fwd(x)
nn.splitHeads(x, nHeads)             // [B, T, D] → [B, H, T, D/H]
nn.mergeHeads(x)                     // inverse of splitHeads
nn.unsplitHeads(captures, name)      // pull per-head slices off a capture
nn.crossEntropyLast(logits, targets) // standard CE
```

Convention: leaf modules (`Linear`, `LayerNorm`) expose `.fwd(x)` for ergonomic
chaining. Composite modules you write yourself are typically free functions
taking `(p: ModuleType, x: Tensor)`.

### LR schedules (`lr` namespace)

```ts
import { lr } from 'tensorgrad'

adam: { lr: 0.005 }                                                 // constant
adam: { lr: lr.linearDecay({ peak: 0.005, final: 0.0005, steps: 1500 }) }
adam: { lr: lr.cosineDecay({ peak: 0.005, final: 0.0001, steps: 5000 }) }
adam: { lr: lr.warmup({ peakLr: 0.001, warmupSteps: 200, after: lr.constant(0.001) }) }
```

LR schedules are serializable shapes, not closures (they cross the worker
boundary). Use a `number` for constant LR, or one of the constructors above.

Mutate Adam hyperparameters mid-training without recompiling via
`compiled.setOptimizerConfig({ ... })`. Pass any subset; absent fields
stay put. The step counter is preserved.

```ts
await compiled.setOptimizerConfig({ lr: 0.001 })
await compiled.setOptimizerConfig({ weightDecay: 0.01 })
await compiled.setOptimizerConfig({ lr: 0.0005, b2: 0.99 })  // any subset
```

When you set a non-constant schedule mid-training (cosine, linear decay,
warmup), the runtime auto-rebases it so its step 1 lines up with the next
training step ("decay from now"). Numbers and `constant` schedules don't
need rebasing. If you pass a schedule with an explicit `startStep`, that
takes precedence — the auto-rebase only fills in a missing one.

```ts
await compiled.setOptimizerConfig({
  lr: lr.cosineDecay({ peak: 0.001, final: 1e-5, steps: 5000 }),
})
```

Note: which params receive weight decay is baked at compile time (via
per-param `{ decay: true | false }` metadata). `setOptimizerConfig`
changes the shrink magnitude on already-decayed params; it doesn't
add decay to params that didn't have it.

### Param init (`init` namespace)

```ts
import { init } from 'tensorgrad'

this.param([D, D], { init: init.kaiming() })           // gain=sqrt(2), fan_in=D
this.param([D, D], { init: init.kaiming({ gain: 1 }) })
this.param([D],    { init: 'zeros' })
this.param([D],    { init: 'ones' })
this.param([D, D], { init: init.randn({ scale: 0.02 }) })
this.param([D],    { init: init.literal(myFloat32Array) })
```

Defaults: `'randn'` (std 0.02). AdamW weight decay defaults to `true` for
randn/kaiming/literal init, `false` for zeros/ones — override per-param with
`{ decay: true | false }`.

### Captures (debugging / mech-interp)

Wrap any tensor inside a forward to expose its activation post-run:

```ts
import { capture } from 'tensorgrad'

const attn = capture(`attn.${i}`, softmaxCausalLast(scores))
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

## When not to use this

- **Inference of pretrained models** → use ONNX Runtime Web or
  transformers.js.
- **Full JAX surface** (vmap, dynamic shapes, multi-backend) → use jax-js.
- **Server-side training** → use PyTorch or JAX.

## License

MIT
