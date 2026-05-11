// Finite-difference vs autograd — exemplars covering classes of backward.
//
// Eight tests, one per class of derivative the autograd has to handle.
// If a new op fits an existing class, trust the FD machinery + samples
// to catch its regression; adding more per-op tests is padding. Only
// add an exemplar here when a genuinely new pattern appears (e.g.,
// a new structural op that breaks the variadic-input class).
//
// The machinery lives in test/_eval.ts (CPU evaluator) + test/_fdgrad.ts
// (FD harness). Both are reusable for any future op.

import {
  tensorInput,
  mul,
  dropout,
  mean,
  matmul,
  embedding,
  concat,
  softmaxCausal,
  gelu,
  conv2d, maxPool2d,
} from '../src/index.js'
import { section, done } from './_assert.js'
import { assertGradMatchesFD } from './_fdgrad.js'

section('FD vs autograd — class exemplars')

// 1. Elementwise binary: gradient flows to both operands; the unbroadcast
//    path handles shape contraction back to original operand shapes.
assertGradMatchesFD('mul (elementwise binary)', [4, 8], p => mean(mul(p, p)))

// 2. Reduction: gradient is broadcast back over the reduced axis.
assertGradMatchesFD('mean (axis + broadcast-back grad)', [4, 8], p => mean(mean(p, -1, { keepDims: true })))

// 3. Linear algebra: dC/dA = dC @ B^T, dC/dB = A^T @ dC. Two-sided
//    matmul backward is the most common source of transpose-rule bugs.
assertGradMatchesFD('matmul (linear algebra)', [4, 8], p => {
  const x = tensorInput('x', [8, 16])
  return mean(matmul(p, x))
}, { extraInputs: { x: makeRange([8, 16]) } })

// 4. Indexing-via-onehot: gradient flows into the embedding table only
//    where the indices point. No flow back through i32 indices.
assertGradMatchesFD('embedding (gradient routes to selected rows)', [10, 4], p => {
  const idx = tensorInput('idx', [3], 'i32')
  return mean(embedding(idx, p))
}, { extraInputs: { idx: new Int32Array([2, 5, 7]) } })

// 5. Fused ML primitive: softmax + mask combined in one IR op. The
//    backward formula is the most complex closed-form in the library;
//    most likely to have a sign or scaling error.
assertGradMatchesFD('softmaxCausal (fused ML)', [4, 4], p => mean(softmaxCausal(p)))

// 6. Stochastic: forward and backward must use the same mask. Tests that
//    the PCG hash + salt + seed plumbing produces a determinism-stable
//    gradient given a fixed seed input.
assertGradMatchesFD('dropout (forward/backward mask match)', [4, 8],
  p => mean(dropout(p, 0.1)),
  { extraInputs: { __dropoutSeed: new Int32Array([42]) } },
)

// 7. Structural / variadic: concat's gradient slices the cotangent back
//    into each input's shape along the concat axis.
assertGradMatchesFD('concat (variadic, gradient via sliceRange)', [3, 4], p => {
  const q = tensorInput('q', [3, 5])
  return mean(concat([p, q], 1))
}, { extraInputs: { q: makeRange([3, 5]) } })

// 8. Composed activation: chain rule through multiple primitives
//    (mul → add → mul → tanh → add → mul). Catches any layer-of-the-stack
//    bug in chain-rule composition.
assertGradMatchesFD('gelu (composed: chain rule through tanh approx)', [4], p => mean(gelu(p)))

// 9. Conv2d (input gradient, two-input op with stride+padding). The param
//    is the input image; the weight is a fixed tensor_input. Tests the
//    conv2d_input_grad kernel which is the input-side backward.
//    Shape [B=1, C_in=2, H=4, W=4]; weight [C_out=3, C_in=2, K_h=2, K_w=2];
//    stride 1, padding 0 → output [1, 3, 3, 3].
assertGradMatchesFD('conv2d (input gradient, stride 1, no padding)', [1, 2, 4, 4], p => {
  const k = tensorInput('k', [3, 2, 2, 2])
  return mean(conv2d(p, k))
}, { extraInputs: { k: makeRange([3, 2, 2, 2]) } })

// 9b. Conv2d weight gradient. Param is the weight; input is a fixed tensor.
assertGradMatchesFD('conv2d (weight gradient, stride 2, padding 1)', [3, 2, 2, 2], p => {
  const x = tensorInput('x', [1, 2, 4, 4])
  return mean(conv2d(x, p, { stride: 2, padding: 1 }))
}, { extraInputs: { x: makeRange([1, 2, 4, 4]) } })

// 10. MaxPool2D: gradient routes only to the argmax position in each window.
//     Use deterministic, well-separated values so ties don't muddy the FD
//     comparison.
assertGradMatchesFD('maxPool2d (argmax-routing gradient)', [1, 2, 4, 4], p => {
  return mean(maxPool2d(p, 2))   // 2x2 pool, stride 2 (default)
}, { paramInit: makeRange([1, 2, 4, 4]) })

done('test/grad.ts')

function makeRange(shape: readonly number[]): Float32Array {
  const n = shape.reduce((p, d) => p * d, 1)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = Math.sin(i * 0.1) * 0.5
  return out
}
