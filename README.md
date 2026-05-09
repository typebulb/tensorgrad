# tensorgrad

A tiny TypeScript-native tensor library with autograd that compiles directly
to WebGPU. Designed for training small models in the browser — without
hand-writing WGSL kernels and without dragging in a 5 MB ML framework.

```sh
npm i tensorgrad
```

Roughly **3000 lines of zero-dependency TypeScript**, ~10 KB gzipped after
build. Targets WebGPU only. Static shapes only. Forward + reverse-mode
autograd; Adam optimizer; the whole training pipeline runs as compiled WGSL.

## Quick example

A 2-layer MLP fitting `y = sin(x)`:

```ts
import {
  Module, compileModule,
  add, mul, sub, sumLast, reshape, matmul, relu,
  type Tensor,
} from 'tensorgrad'

class Linear extends Module {
  W: Tensor; b: Tensor
  constructor(public inDim: number, public outDim: number) {
    super()
    this.W = this.param([inDim, outDim])              // randn, scale 0.02
    this.b = this.param([outDim], { init: 'zeros' })
  }
}

class MLP extends Module {
  l1 = new Linear(1, 64)
  l2 = new Linear(64, 64)
  l3 = new Linear(64, 1)
}

const linear = (p: Linear, x: Tensor) => add(matmul(x, p.W), p.b)

function forward(m: MLP, x: Tensor): Tensor {
  return linear(m.l3, relu(linear(m.l2, relu(linear(m.l1, x)))))
}

function loss(m: MLP, x: Tensor, y: Tensor): Tensor {
  const diff = sub(forward(m, x), y)
  return mul(sumLast(reshape(mul(diff, diff), [B])), 1 / B)
}

const B = 256
const compiled = await compileModule(() => new MLP(), loss, {
  adam: { lr: 0.005 },
  inputs: [
    { name: 'x', shape: [B, 1], dtype: 'f32' },
    { name: 'y', shape: [B, 1], dtype: 'f32' },
  ],
})

compiled.uploadInitialParams()  // applies the per-param init declared above

for (let step = 0; step < 1000; step++) {
  const { x, y } = generateBatch()
  const lossVal = await compiled.step({ x, y })
  if (step % 100 === 0) console.log('step', step, 'loss', lossVal)
}
```

That's the whole user-facing surface for this model: `Module` for parameter
storage, plain functions for the forward pass, `compileModule` to JIT-compile
to WGSL with autograd + Adam wired in. No decorators, no `tf.GradientTape`,
no `register_pytree_node`.

For a more involved example — a 3-layer transformer trained from scratch on
2-digit addition — see the [`samples/`](./samples) workspace
(`pnpm --filter samples dev`).

## What this library is for

Small browser-side ML where you want to *train* the model, not just run
inference of a pretrained model. Educational artifacts, interactive
demos, on-device personalization, "transformer from scratch in your browser"
blog posts. Roughly the niche where the model is small enough to fit
comfortably in a browser tab but where you still want autograd and a real
optimizer.

If you want to ship inference of a pretrained model, use
[ONNX Runtime Web](https://github.com/microsoft/onnxruntime) or
[transformers.js](https://github.com/xenova/transformers.js).
If you need full JAX (vmap / pmap / dynamic shapes / multi-backend), use
[jax-js](https://github.com/jax-js/jax).

## Scope (deliberately small)

The library only does what it does because of what it doesn't do.
[`SPEC.md`](./SPEC.md) has the full design notes; the load-bearing
"out of scope" decisions are:

- **WebGPU only** — no Wasm or WebGL fallback.
- **Static shapes only** — every shape is fixed at compile time. This is
  what lets us bake constants into the WGSL instead of carrying shape
  uniforms.
- **`grad` is the only transformation** — no `vmap`, `pmap`, `jvp`,
  `custom_vjp`. Batch your data explicitly.
- **`f32` only** — no dtype promotion, no mixed precision.
- **Closed op set** — about 25 ops, listed in `SPEC.md`. Compositions of
  those handle most needs (GELU, RMS norm, etc. are a few lines on top).
- **Adam lives in the IR** — bias correction included; no CPU↔GPU
  round-trip per step.

## Status

Alpha. Two real working models (a transformer training to <0.1 loss on
addition, an MLP fitting `sin`). API may change before 1.0. Filing issues
welcome.

## License

MIT
