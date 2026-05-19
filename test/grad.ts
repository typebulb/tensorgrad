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
  mul, add,
  dropout,
  mean,
  matmul,
  embedding,
  concat,
  narrow, split,
  softmaxCausal, whereCausal,
  gelu,
  conv2d, maxPool2d,
  stopGradient,
} from '../src/index.js'
import { traceFn, paramInput, tensorInput } from '../src/trace.js'
import { appendGrad } from '../src/grad.js'
import { evalGraph } from './_eval.js'
import { section, ok, fail, done } from './_assert.js'
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
  return mean(embedding(p, idx))
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
  { extraInputs: { __prngSeed: new Int32Array([42]) } },
)

// 7. Structural / variadic: concat's gradient slices the cotangent back
//    into each input's shape along the concat axis.
assertGradMatchesFD('concat (variadic, gradient via narrow)', [3, 4], p => {
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

// 9c. Slice + scatter-into-zero backward. `narrow` on a non-last axis is the
//     general path; its adjoint emits the scatter_axis op (narrow's reverse). A
//     stable test takes a slice and reduces — gradient is 1/N inside the slice,
//     0 outside.
assertGradMatchesFD('narrow (non-last axis, scatter backward)', [4, 6], p => {
  return mean(narrow(p, 1, 1, 3))   // [4, 6] -> [4, 3]; backward scatters
})

// 9d. Split: composes from narrow. Each piece flows back through its own
//     scatter; sums recombine cleanly in the input cotangent. Catches a
//     missing accumulate() in slice's adjoint.
assertGradMatchesFD('split (two pieces, both contribute to loss)', [4, 6], p => {
  const [a, b] = split(p, [2, 4], 1)
  return add(mean(mul(a!, a!)), mean(mul(b!, b!)))
})

// 9e. whereCausal: lower triangle passes through, upper is replaced by
//     fillValue. Backward zeroes the upper triangle. Pair with mean so the
//     scalar loss has gradient 1/N on the lower triangle, 0 elsewhere.
assertGradMatchesFD('whereCausal (lower-triangle pass, upper-triangle zero)', [4, 4], p => {
  return mean(whereCausal(p, 0))
})

// 10. MaxPool2D: gradient routes only to the argmax position in each window.
//     Use deterministic, well-separated values so ties don't muddy the FD
//     comparison.
assertGradMatchesFD('maxPool2d (argmax-routing gradient)', [1, 2, 4, 4], p => {
  return mean(maxPool2d(p, 2))   // 2x2 pool, stride 2 (default)
}, { paramInit: makeRange([1, 2, 4, 4]) })

// 11. stopGradient: structural new pattern — autograd is supposed to *diverge*
//     from FD here. Forward is identity, but the backward rule deliberately
//     drops the cotangent. Test: build `mean(stopGradient(p) + p)` where the
//     numerical derivative w.r.t. each p[i] is 2/N (both paths contribute),
//     but autograd reports 1/N (only the un-detached path contributes).
{
  const N = 6
  const init = new Float32Array(N).map((_, i) => i * 0.1 - 0.3)
  const graph = traceFn(() => {
    const p = paramInput('w', [N])
    return mean(add(stopGradient(p), p))
  })
  const { paramGrads } = appendGrad(graph)
  const vals = evalGraph(graph, { w: init })
  const g = vals.get(paramGrads['w']!.id) as Float32Array
  const expected = 1 / N
  for (let i = 0; i < N; i++) {
    if (Math.abs(g[i]! - expected) > 1e-5) {
      fail(`stopGradient: autograd g[${i}]=${g[i]} expected ${expected} (gradient should flow only through the un-detached path)`)
    }
  }
  ok(`stopGradient blocks backward — autograd grad is 1/N on the un-detached path only`)
}

done('test/grad.ts')

function makeRange(shape: readonly number[]): Float32Array {
  const n = shape.reduce((p, d) => p * d, 1)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = Math.sin(i * 0.1) * 0.5
  return out
}
