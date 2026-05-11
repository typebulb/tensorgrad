// Per-op shape inference, dtype, and error cases. Each op gets a small
// trace that exercises its shape rule, then a few error cases for the
// shape/dtype guards. No GPU — pure main-thread IR construction.

import {
  trace, tensorInput, paramInput,
  add, sub, mul, div, min, max, clamp,
  sqrt, rsqrt, log, exp, relu, neg, abs, tanh, sigmoid, gelu, silu,
  dropout,
  less, greater, where,
  meanLast, sumLast, sumAll, meanAll, argmaxLast,
  reshape, transpose, swapAxes,
  matmul, matmulBatched,
  oneHot, arange, embedding,
  sliceLastRange, sliceRange, concat, stack, split,
  softmaxLast, logSoftmaxLast, softmaxCausalLast, whereCausal,
  type Tensor,
} from '../src/index.js'
import { section, assertShape, assertEq, assertThrows, done, ok, fail } from './_assert.js'

// Helper: trace a single-expression graph and pull the last produced tensor.
function tr(fn: () => Tensor): Tensor {
  const g = trace(fn)
  return g.tensors[g.outputs[0]!]!
}

section('arithmetic binops (Tensor, Tensor)')
{
  const r = tr(() => {
    const a = tensorInput('a', [4, 8])
    const b = tensorInput('b', [4, 8])
    return add(a, b)
  })
  assertShape(r.shape, [4, 8], 'add same-shape')
  assertEq(r.dtype, 'f32', 'add dtype')
}
{
  const r = tr(() => add(tensorInput('a', [4, 8]), tensorInput('b', [8])))
  assertShape(r.shape, [4, 8], 'add trailing-broadcast')
}
{
  const r = tr(() => mul(tensorInput('a', [3, 4]), tensorInput('b', [3, 4])))
  assertShape(r.shape, [3, 4], 'mul')
}
{
  const r = tr(() => div(tensorInput('a', [2, 5]), tensorInput('b', [2, 5])))
  assertShape(r.shape, [2, 5], 'div')
}
{
  const r = tr(() => min(tensorInput('a', [3, 4]), tensorInput('b', [3, 4])))
  assertShape(r.shape, [3, 4], 'min')
  assertEq(r.dtype, 'f32', 'min dtype')
}
{
  const r = tr(() => max(tensorInput('a', [3, 4]), tensorInput('b', [3, 4])))
  assertShape(r.shape, [3, 4], 'max')
}

section('scalar overloads (Tensor, number)')
{
  const r = tr(() => add(tensorInput('a', [3, 4]), 1.5))
  assertShape(r.shape, [3, 4], 'add(t, number)')
}
{
  const r = tr(() => mul(tensorInput('a', [3, 4]), 2))
  assertShape(r.shape, [3, 4], 'mul(t, number)')
}
{
  const r = tr(() => div(tensorInput('a', [3, 4]), 4))
  assertShape(r.shape, [3, 4], 'div(t, number)')
}
{
  const r = tr(() => min(tensorInput('a', [3, 4]), 0))
  assertShape(r.shape, [3, 4], 'min(t, number)')
}
{
  const r = tr(() => greater(tensorInput('a', [4]), 0))
  assertShape(r.shape, [4], 'greater(t, number)')
  assertEq(r.dtype, 'bool', 'greater(t, number) dtype')
}
{
  const r = tr(() => clamp(tensorInput('a', [4]), -1, 1))
  assertShape(r.shape, [4], 'clamp')
}
assertThrows(
  () => trace(() => div(tensorInput('a', [3]), 0)),
  'div: scalar divisor cannot be zero', 'div(t, 0) rejected',
)

section('unary math')
{
  const ops: [string, (t: Tensor) => Tensor][] = [
    ['sqrt', sqrt], ['rsqrt', rsqrt], ['log', log], ['exp', exp],
    ['relu', relu], ['neg', neg], ['abs', abs],
    ['tanh', tanh], ['sigmoid', sigmoid],
  ]
  for (const [name, fn] of ops) {
    const r = tr(() => fn(tensorInput('a', [2, 5])))
    assertShape(r.shape, [2, 5], `${name} preserves shape`)
    assertEq(r.dtype, 'f32', `${name} dtype`)
  }
}

section('composed activations (gelu, silu)')
{
  // gelu and silu compose from tanh/sigmoid; they should emit multiple ops
  // but produce a same-shape output. We don't care about the exact op count,
  // just that they trace cleanly with the right shape.
  const r1 = tr(() => gelu(tensorInput('a', [4])))
  assertShape(r1.shape, [4], 'gelu')
  const r2 = tr(() => silu(tensorInput('a', [4])))
  assertShape(r2.shape, [4], 'silu')
}

section('comparisons + select')
{
  const r = tr(() => less(tensorInput('a', [3, 4]), tensorInput('b', [3, 4])))
  assertShape(r.shape, [3, 4], 'less')
  assertEq(r.dtype, 'bool', 'less dtype = bool')
}
{
  const r = tr(() => greater(tensorInput('a', [3, 4]), tensorInput('b', [3, 4])))
  assertEq(r.dtype, 'bool', 'greater dtype = bool')
}
{
  const r = tr(() => {
    const a = tensorInput('a', [4])
    const b = tensorInput('b', [4])
    const cond = greater(a, b)
    return where(cond, a, b)
  })
  assertShape(r.shape, [4], 'where')
  assertEq(r.dtype, 'f32', 'where dtype follows a/b')
}
assertThrows(
  () => trace(() => where(tensorInput('cond', [4]), tensorInput('a', [4]), tensorInput('b', [4]))),
  'where: cond must be bool', 'where rejects non-bool cond',
)

section('reductions')
{
  const r = tr(() => meanLast(tensorInput('a', [4, 8, 16])))
  assertShape(r.shape, [4, 8, 1], 'meanLast keepdims=true')
}
{
  const r = tr(() => sumLast(tensorInput('a', [4, 8, 16])))
  assertShape(r.shape, [4, 8], 'sumLast keepdims=false')
}
{
  const r = tr(() => sumAll(tensorInput('a', [3, 5, 7])))
  assertShape(r.shape, [], 'sumAll → scalar')
}
{
  const r = tr(() => meanAll(tensorInput('a', [3, 5, 7])))
  assertShape(r.shape, [], 'meanAll → scalar')
}
{
  const r = tr(() => argmaxLast(tensorInput('a', [4, 10])))
  assertShape(r.shape, [4], 'argmaxLast drops last axis')
  assertEq(r.dtype, 'i32', 'argmaxLast dtype = i32')
}
assertThrows(
  () => trace(() => meanAll(tensorInput('a', [0, 4]))),
  'meanAll: cannot mean over zero elements', 'meanAll rejects 0-element',
)

section('shape ops')
{
  const r = tr(() => reshape(tensorInput('a', [4, 8]), [32]))
  assertShape(r.shape, [32], 'reshape explicit')
}
{
  const r = tr(() => reshape(tensorInput('a', [4, 8]), [-1, 2]))
  assertShape(r.shape, [16, 2], 'reshape with -1')
}
{
  const r = tr(() => transpose(tensorInput('a', [4, 8, 16]), [2, 0, 1]))
  assertShape(r.shape, [16, 4, 8], 'transpose')
}
{
  const r = tr(() => swapAxes(tensorInput('a', [4, 8, 16]), 0, 2))
  assertShape(r.shape, [16, 8, 4], 'swapAxes(0, 2)')
}

section('linear algebra')
{
  const r = tr(() => matmul(tensorInput('a', [4, 8]), tensorInput('b', [8, 16])))
  assertShape(r.shape, [4, 16], 'matmul [4,8] x [8,16]')
}
{
  const r = tr(() => matmul(tensorInput('a', [3, 4, 8]), tensorInput('b', [8, 16])))
  assertShape(r.shape, [3, 4, 16], 'matmul batched-on-left')
}
{
  const r = tr(() => matmulBatched(tensorInput('a', [2, 4, 8]), tensorInput('b', [2, 8, 16])))
  assertShape(r.shape, [2, 4, 16], 'matmulBatched')
}

section('indexing / casting')
{
  const r = tr(() => oneHot(tensorInput('idx', [5], 'i32'), 10))
  assertShape(r.shape, [5, 10], 'oneHot [5] depth=10')
  assertEq(r.dtype, 'f32', 'oneHot default dtype')
}
{
  const r = tr(() => arange(8, 'i32'))
  assertShape(r.shape, [8], 'arange(8)')
  assertEq(r.dtype, 'i32', 'arange i32 dtype')
}
{
  const r = tr(() => embedding(paramInput('table', [100, 64]), tensorInput('idx', [4, 8], 'i32')))
  assertShape(r.shape, [4, 8, 64], 'embedding [V,D] x [B,T] → [B,T,D]')
}

section('slicing')
{
  const r = tr(() => sliceLastRange(tensorInput('a', [4, 16]), 4, 12))
  assertShape(r.shape, [4, 8], 'sliceLastRange [4,16] [4..12) → [4,8]')
}
{
  const r = tr(() => sliceRange(tensorInput('a', [4, 16, 8]), 1, 4, 12))
  assertShape(r.shape, [4, 8, 8], 'sliceRange axis=1')
}
{
  const r = tr(() => sliceRange(tensorInput('a', [4, 16, 8]), -1, 0, 4))
  assertShape(r.shape, [4, 16, 4], 'sliceRange negative axis')
}
assertThrows(
  () => trace(() => sliceRange(tensorInput('a', [4, 16]), 5, 0, 1)),
  'axis 5 out of range', 'sliceRange axis out of range',
)
assertThrows(
  () => trace(() => sliceRange(tensorInput('a', [4, 16]), 0, 2, 10)),
  'invalid range [2, 10) for axis 0 of size 4', 'sliceRange end > axis size',
)

section('concat / stack / split')
{
  const r = tr(() => {
    const a = tensorInput('a', [3, 4])
    const b = tensorInput('b', [3, 5])
    return concat([a, b], 1)
  })
  assertShape(r.shape, [3, 9], 'concat [3,4]+[3,5] axis=1')
}
{
  const r = tr(() => {
    const a = tensorInput('a', [2, 8])
    const b = tensorInput('b', [3, 8])
    const c = tensorInput('c', [5, 8])
    return concat([a, b, c], 0)
  })
  assertShape(r.shape, [10, 8], 'concat 3-way axis=0')
}
{
  const r = tr(() => {
    const a = tensorInput('a', [4])
    const b = tensorInput('b', [4])
    return stack([a, b], 0)
  })
  assertShape(r.shape, [2, 4], 'stack [4]+[4] axis=0')
}
{
  const r = tr(() => {
    const a = tensorInput('a', [4])
    const b = tensorInput('b', [4])
    return stack([a, b], 1)
  })
  assertShape(r.shape, [4, 2], 'stack axis=1')
}
{
  const g = trace(() => {
    const t = tensorInput('t', [3, 12])
    const [a, b, c] = split(t, 1, [3, 4, 5])
    return concat([a!, b!, c!], 1)  // identity round-trip
  })
  const out = g.tensors[g.outputs[0]!]!
  assertShape(out.shape, [3, 12], 'split + concat round-trip')
}
assertThrows(
  () => trace(() => {
    const a = tensorInput('a', [3, 4])
    const b = tensorInput('b', [5, 4])
    return concat([a, b], 1)
  }),
  'must match', 'concat: shape mismatch on non-axis dim',
)
assertThrows(
  () => trace(() => {
    const inputs = Array.from({ length: 9 }, (_, i) => tensorInput(`a${i}`, [3, 4]))
    return concat(inputs, 1)
  }),
  'exceeds the bind-group cap', 'concat > 7 inputs rejected',
)
assertThrows(
  () => trace(() => concat([], 0)),
  'needs at least one input', 'concat empty list rejected',
)
assertThrows(
  () => trace(() => split(tensorInput('t', [3, 10]), 1, [2, 3, 4])),
  'sizes sum to 9, but axis 1 has size 10', 'split sizes mismatch',
)

section('fused ML primitives')
{
  const r = tr(() => softmaxLast(tensorInput('a', [4, 10])))
  assertShape(r.shape, [4, 10], 'softmaxLast preserves shape')
}
{
  const r = tr(() => logSoftmaxLast(tensorInput('a', [4, 10])))
  assertShape(r.shape, [4, 10], 'logSoftmaxLast preserves shape')
}
{
  const r = tr(() => softmaxCausalLast(tensorInput('a', [2, 8, 8])))
  assertShape(r.shape, [2, 8, 8], 'softmaxCausalLast square last-2')
}
{
  const r = tr(() => whereCausal(tensorInput('a', [2, 8, 8]), -1e30))
  assertShape(r.shape, [2, 8, 8], 'whereCausal preserves shape')
}

section('dropout')
{
  // p === 0 short-circuits to the input (no op emitted).
  const a = tensorInput
  const g = trace(() => {
    const x = a('x', [4, 8])
    const out = dropout(x, 0)
    return out
  })
  const opKinds = g.ops.map(o => o.kind)
  if (opKinds.includes('dropout')) fail('dropout(x, 0) should not emit an op')
  ok('dropout(x, 0) short-circuits to identity (no op)')

  const r = tr(() => dropout(a('x', [4, 8]), 0.1))
  assertShape(r.shape, [4, 8], 'dropout(x, 0.1) preserves shape')

  // Multiple dropouts in one graph share the seed input.
  const g2 = trace(() => {
    const x = a('x', [4])
    const d1 = dropout(x, 0.1)
    const d2 = dropout(d1, 0.2)
    return d2
  })
  const seedInputs = g2.ops.filter(o => o.kind === 'tensor_input' && o.name === '__dropoutSeed').length
  if (seedInputs !== 1) fail(`dropout seed input should be shared; got ${seedInputs} occurrences`)
  ok('multiple dropouts share __dropoutSeed tensor_input')

  // Each dropout op gets a unique salt.
  const dropoutOps = g2.ops.filter(o => o.kind === 'dropout') as Array<{ kind: 'dropout'; salt: number }>
  const salts = dropoutOps.map(o => o.salt)
  if (new Set(salts).size !== salts.length) fail(`dropout salts must be unique within a graph; got ${salts}`)
  ok(`dropout salts unique within graph: [${salts.join(', ')}]`)
}
assertThrows(
  () => trace(() => dropout(tensorInput('x', [4]), 1.5)),
  'p must be in [0, 1)', 'dropout rejects p >= 1',
)
assertThrows(
  () => trace(() => dropout(tensorInput('x', [4]), -0.1)),
  'p must be in [0, 1)', 'dropout rejects p < 0',
)

done('test/ops.ts')
