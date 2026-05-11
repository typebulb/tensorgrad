// Backward emission per differentiable op. For each op, build a small graph
// terminating in a scalar loss, call appendGrad, and verify:
//   * every declared param has an entry in paramGrads
//   * each gradient's shape matches its param's shape
//
// Non-differentiable ops (less, greater, argmaxLast, oneHot, arange) are
// verified to not produce gradients into discrete inputs — the test ensures
// grad doesn't crash on a graph containing them.

import {
  trace, tensorInput, paramInput,
  add, sub, mul, div, min, max,
  sqrt, rsqrt, log, exp, relu, neg, abs, tanh, sigmoid, gelu, silu,
  dropout,
  greater, where,
  meanLast, sumLast, sumAll, meanAll, argmaxLast,
  reshape, transpose, swapAxes,
  matmul, matmulBatched,
  oneHot, embedding,
  concat, stack,
  softmaxLast, logSoftmaxLast, softmaxCausalLast,
  appendGrad,
  type Tensor,
} from '../src/index.js'
import { section, assertShape, ok, fail, done } from './_assert.js'

interface GradCheck {
  paramShape: readonly number[]
  build: (p: Tensor) => Tensor   // returns a scalar loss
}

function checkGrad(name: string, c: GradCheck): void {
  const g = trace(() => {
    const p = paramInput('w', c.paramShape as number[])
    return c.build(p)
  })
  const lossT = g.tensors[g.outputs[0]!]!
  if (lossT.shape.length !== 0) {
    fail(`${name}: loss must be rank-0 scalar, got ${JSON.stringify(lossT.shape)}`)
  }
  const { paramGrads } = appendGrad(g)
  const grad = paramGrads['w']
  if (!grad) fail(`${name}: no gradient for param 'w'`)
  assertShape(grad!.shape, c.paramShape, `${name} grad shape`)
}

section('arithmetic backward')
checkGrad('add', { paramShape: [4, 8], build: p => meanAll(add(p, p)) })
checkGrad('sub', { paramShape: [4, 8], build: p => meanAll(sub(p, p)) })
checkGrad('mul', { paramShape: [4, 8], build: p => meanAll(mul(p, p)) })
checkGrad('div', {
  paramShape: [4, 8],
  build: p => {
    const c = tensorInput('c', [4, 8])
    return meanAll(div(p, c))
  },
})
checkGrad('mul_scalar (via mul(t, num))', { paramShape: [4], build: p => meanAll(mul(p, 0.5)) })
checkGrad('add_scalar (via add(t, num))', { paramShape: [4], build: p => meanAll(add(p, 0.5)) })
checkGrad('min', { paramShape: [4], build: p => meanAll(min(p, 0)) })
checkGrad('max', { paramShape: [4], build: p => meanAll(max(p, 0)) })

section('unary backward')
checkGrad('sqrt', { paramShape: [4], build: p => meanAll(sqrt(p)) })
checkGrad('rsqrt', { paramShape: [4], build: p => meanAll(rsqrt(p)) })
checkGrad('log', { paramShape: [4], build: p => meanAll(log(p)) })
checkGrad('exp', { paramShape: [4], build: p => meanAll(exp(p)) })
checkGrad('relu', { paramShape: [4], build: p => meanAll(relu(p)) })
checkGrad('neg', { paramShape: [4], build: p => meanAll(neg(p)) })
checkGrad('abs', { paramShape: [4], build: p => meanAll(abs(p)) })
checkGrad('tanh', { paramShape: [4], build: p => meanAll(tanh(p)) })
checkGrad('sigmoid', { paramShape: [4], build: p => meanAll(sigmoid(p)) })
checkGrad('gelu', { paramShape: [4], build: p => meanAll(gelu(p)) })
checkGrad('silu', { paramShape: [4], build: p => meanAll(silu(p)) })

section('reduction backward')
checkGrad('meanLast', {
  paramShape: [4, 8],
  build: p => meanAll(meanLast(p)),
})
checkGrad('sumLast', {
  paramShape: [4, 8],
  build: p => meanAll(sumLast(p)),
})
checkGrad('sumAll', { paramShape: [4, 8], build: p => sumAll(p) })
checkGrad('meanAll', { paramShape: [4, 8], build: p => meanAll(p) })

section('shape backward')
checkGrad('reshape', {
  paramShape: [4, 8],
  build: p => meanAll(reshape(p, [32])),
})
checkGrad('transpose', {
  paramShape: [4, 8, 16],
  build: p => meanAll(transpose(p, [2, 0, 1])),
})
checkGrad('swapAxes', {
  paramShape: [4, 8],
  build: p => meanAll(swapAxes(p, 0, 1)),
})

section('linear algebra backward')
checkGrad('matmul', {
  paramShape: [4, 8],
  build: p => {
    const x = tensorInput('x', [4, 8])
    return meanAll(matmul(x, transpose(p, [1, 0])))
  },
})
checkGrad('matmulBatched', {
  paramShape: [2, 4, 8],
  build: p => {
    const x = tensorInput('x', [2, 4, 4])
    return meanAll(matmulBatched(x, p))
  },
})

section('indexing backward')
checkGrad('embedding', {
  paramShape: [10, 4],
  build: p => {
    const idx = tensorInput('idx', [3], 'i32')
    return meanAll(embedding(p, idx))
  },
})

section('selection backward')
checkGrad('where (flows to a/b, not cond)', {
  paramShape: [4],
  build: p => {
    const cond = greater(p, 0)  // cond is non-differentiable in cond input
    return meanAll(where(cond, p, p))
  },
})

section('fused ML backward')
checkGrad('logSoftmaxLast', {
  paramShape: [4, 10],
  build: p => meanAll(logSoftmaxLast(p)),
})
checkGrad('softmaxLast (composed)', {
  paramShape: [4, 10],
  build: p => meanAll(softmaxLast(p)),
})
checkGrad('softmaxCausalLast', {
  paramShape: [4, 4],
  build: p => meanAll(softmaxCausalLast(p)),
})

section('structural backward')
checkGrad('concat (gradient flows to each input via sliceRange)', {
  paramShape: [3, 4],
  build: p => {
    const q = tensorInput('q', [3, 5])
    return meanAll(concat([p, q], 1))
  },
})
checkGrad('stack', {
  paramShape: [4],
  build: p => {
    const q = tensorInput('q', [4])
    return meanAll(stack([p, q], 0))
  },
})
// sliceLastRange / sliceRange backward is intentionally unimplemented —
// gradient flows through them aren't currently expected. Concat's backward
// uses sliceRange but doesn't differentiate *through* it.

section('stochastic backward')
checkGrad('dropout (forward + backward share salt/seed)', {
  paramShape: [4, 8],
  build: p => meanAll(dropout(p, 0.1)),
})

section('non-differentiable ops alongside diff path')
// less, greater, argmaxLast, oneHot, arange: gradients should not flow into
// these, but their presence in the graph must not crash appendGrad. We
// verify by tracing a graph that mixes diff and non-diff ops.
{
  const g = trace(() => {
    const p = paramInput('w', [4, 10])
    const labels = tensorInput('labels', [4], 'i32')
    // Take p, soft-max it, dot with one-hot of labels: a basic NLL-ish loss.
    const lp = logSoftmaxLast(p)
    const oh = oneHot(labels, 10)
    return meanAll(mul(lp, oh))
  })
  const { paramGrads } = appendGrad(g)
  if (!paramGrads['w']) fail('non-diff: gradient missing for w')
  assertShape(paramGrads['w']!.shape, [4, 10], 'logSoftmaxLast + oneHot path → w gradient')
}
{
  // argmaxLast is non-differentiable; consuming it in a non-differentiated
  // computation should be fine (argmax → oneHot → mul → meanAll) so long as
  // gradients don't try to flow through the discrete index.
  const g = trace(() => {
    const p = paramInput('w', [4, 10])
    const preds = argmaxLast(p)         // [4] i32 — non-diff branch
    const ohPred = oneHot(preds, 10)    // [4, 10]
    // Mix the non-diff branch with a diff branch (p itself).
    return meanAll(mul(p, ohPred))
  })
  const { paramGrads } = appendGrad(g)
  if (!paramGrads['w']) fail('argmaxLast: gradient missing for w')
  ok('argmaxLast in graph alongside differentiable path: w gradient still computed')
}

done('test/grad.ts')
