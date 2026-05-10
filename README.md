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

const compiled = await compileModule(() => new MLP(), lossFn, {
  adam: { lr: 0.005 },
  inputs: {
    x: { shape: [B, 1], dtype: 'f32' },
    y: { shape: [B, 1], dtype: 'f32' },
  },
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
- `compileModule(factory, forward, opts)` traces the forward, derives
  gradients, wires Adam, plans buffers, generates WGSL, spawns a worker, and
  returns a `CompiledModule`. The factory `() => new Model()` is invoked
  once during compile; the model instance is consumed (its param sentinels
  are mutated into `Tensor`s).
- Every method on the compiled module is async. `await compiled.step(...)`
  resolves with the loss after the worker's GPU work finishes.

## Public API

### Compile entry points

```ts
compileModule(factory, forward, { adam?, inputs? }): Promise<CompiledModule>
compileForward(factory, forward, { inputs? }): Promise<CompiledForwardModule>
```

`compileForward` produces a forward-only graph in its own worker. To share
params with an existing training graph, use the sibling method:

```ts
const train  = await compileModule(() => new Model(), lossFn, { ... })
const infer  = await train.compileForward(predictFn, {
  inputs: { tokens: { shape: [1, T], dtype: 'i32' } },
})
// infer runs in train's worker — every step's param updates are visible.
```

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
compiled.compileForward(forward, { inputs? })   // sibling forward graph
compiled.destroy()                              // tear down worker + GPU
```

`compiled.kernelCount`, `compiled.outputShape`, `compiled.paramNames`, and
`compiled.ir` are sync properties for inspection.

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
