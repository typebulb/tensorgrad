// Dead-code elimination. Two layers of coverage:
//
//   1. The pass itself (src/dce.ts) on hand-traced graphs: dead gradients
//      pruned, live values untouched (numerically identical pre/post),
//      captures and named leaves preserved.
//   2. The compile-pipeline integration via the public `trace()` entry
//      point: the motivating workload — a conv whose weights arrive as a
//      frozen tensor input — must compile to a graph with no
//      conv2d_weight_grad kernel for the frozen weight.
//
// The motivating numbers (growing-nca bulb, 2026-07-10): 48 of 144
// conv2d_weight_grad kernels computed the gradient of a frozen filter bank
// nobody read — ~16% of the bulb's conv compute per step.

import {
  mul, mean, square, sub, conv2d, capture, relu,
  Module, Linear, trace,
  type Tensor,
} from '../src/index.js'
import { traceFn, paramInput, tensorInput } from '../src/trace.js'
import { appendGrad } from '../src/grad.js'
import { eliminateDeadCode } from '../src/dce.js'
import { evalGraph } from './_eval.js'
import { section, ok, fail, done } from './_assert.js'

function makeRange(shape: readonly number[]): Float32Array {
  const n = shape.reduce((p, d) => p * d, 1)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = Math.sin(i * 0.1) * 0.5
  return out
}

section('dce — frozen conv weights: weight-grad pruned, values unchanged')
{
  // conv2d(param image, frozen filter input) → autograd emits BOTH
  // conv2d_input_grad (live: feeds the param's gradient) and
  // conv2d_weight_grad (dead: nothing consumes d(loss)/d(frozen filters)).
  const build = () => traceFn(() => {
    const p = paramInput('img', [1, 2, 4, 4])
    const k = tensorInput('k', [3, 2, 2, 2])
    return mean(conv2d(p, k))
  })
  const inputs = { img: makeRange([1, 2, 4, 4]), k: makeRange([3, 2, 2, 2]) }

  // Reference: no DCE.
  const gRef = build()
  const { paramGrads: pgRef } = appendGrad(gRef)
  const refVals = evalGraph(gRef, inputs)
  const refLoss = (refVals.get(gRef.outputs[0]!) as Float32Array)[0]!
  const refGrad = refVals.get(pgRef['img']!.id) as Float32Array

  // Same graph, DCE'd.
  const g = build()
  const { paramGrads } = appendGrad(g)
  if (!g.ops.some(o => o.kind === 'conv2d_weight_grad')) {
    fail('precondition: autograd should emit conv2d_weight_grad before DCE')
  }
  const opCountBefore = g.ops.length
  const idMap = eliminateDeadCode(g, Object.values(paramGrads).map(t => t.id))
  if (g.ops.some(o => o.kind === 'conv2d_weight_grad')) {
    fail('conv2d_weight_grad for a frozen tensor-input weight survived DCE')
  }
  if (!g.ops.some(o => o.kind === 'conv2d_input_grad')) {
    fail('conv2d_input_grad (live — feeds the param gradient) was wrongly eliminated')
  }
  ok(`frozen-weight grad pruned (${opCountBefore} → ${g.ops.length} ops)`)

  // Ids were renumbered; the graph must still be dense and self-consistent.
  for (let i = 0; i < g.ops.length; i++) {
    if (g.ops[i]!.out !== i && g.tensors[g.ops[i]!.out]!.source !== i) {
      fail(`op #${i} out/source linkage broken after renumbering`)
    }
  }
  if (g.ops.length !== g.tensors.length) {
    fail(`ops (${g.ops.length}) and tensors (${g.tensors.length}) diverged — dead tensors left behind`)
  }
  ok('surviving graph is dense (one tensor per op, sources consistent)')

  // Numerics: loss and the param gradient are bit-identical to the
  // un-DCE'd graph.
  const vals = evalGraph(g, inputs)
  const loss = (vals.get(g.outputs[0]!) as Float32Array)[0]!
  const gradT = idMap.get(paramGrads['img']!.id)!
  const grad = vals.get(gradT.id) as Float32Array
  if (loss !== refLoss) fail(`loss changed after DCE: ${loss} vs ${refLoss}`)
  for (let i = 0; i < grad.length; i++) {
    if (grad[i] !== refGrad[i]) fail(`param grad changed at ${i}: ${grad[i]} vs ${refGrad[i]}`)
  }
  ok('loss + param gradient identical pre/post DCE')
}

section('dce — captures are roots; unused forward work is pruned')
{
  const g = traceFn(() => {
    const p = paramInput('w', [4])
    const x = tensorInput('x', [4])
    capture('kept', mul(p, p))          // dead w.r.t. the loss, but captured
    void relu(sub(x, 1))                // genuinely dead: nothing consumes it
    return mean(mul(p, x))
  })
  const { paramGrads } = appendGrad(g)
  eliminateDeadCode(g, Object.values(paramGrads).map(t => t.id))
  if (!g.captures.has('kept')) fail('capture entry lost')
  const capId = g.captures.get('kept')!
  if (!g.tensors[capId]) fail('capture points at an eliminated tensor')
  if (g.ops.some(o => o.kind === 'relu')) fail('unconsumed relu survived DCE')
  // Named leaves stay even when dead — x IS consumed here, so check with a
  // fresh graph where an input is entirely unused.
  const g2 = traceFn(() => {
    const p = paramInput('w', [4])
    void tensorInput('unused', [4])
    return mean(p)
  })
  appendGrad(g2)
  eliminateDeadCode(g2)
  if (!g2.ops.some(o => o.kind === 'tensor_input' && o.name === 'unused')) {
    fail('named tensor_input leaf must survive DCE (declared call surface)')
  }
  ok('captures kept, dead compute pruned, named leaves preserved')
}

section('dce — compile pipeline (trace): frozen-filter conv training graph')
{
  class NCAish extends Module {
    head = new Linear(8, 8)
    // Trainable transform UPSTREAM of the frozen conv — mirrors growing-nca,
    // where the conv input depends on the update net via earlier CA steps.
    // Makes conv2d_input_grad live while conv2d_weight_grad stays dead.
    gain = this.param([1, 4, 1, 1])
  }
  // Perception-style stage: frozen filters come in as an input; only the
  // head's params should get weight-grad kernels.
  const lossFn = (m: NCAish, { state, filters }: { state: Tensor; filters: Tensor }) => {
    const perceived = conv2d(mul(state, m.gain), filters, { padding: 1, groups: 2 })
    const flat = mean(mean(perceived, -1), -1)              // [B, 8]
    return mean(square(sub(m.head.fwd(flat), 1)))
  }
  const ir = await trace({
    model: new NCAish(),
    loss: lossFn,
    inputs: { state: [2, 4, 8, 8], filters: [8, 2, 3, 3] },
    optimizer: { kind: 'adam', lr: 1e-3 },
  })
  const weightGrads = ir.graph.ops.filter(o => o.kind === 'conv2d_weight_grad')
  if (weightGrads.length !== 0) {
    fail(`trace() emitted ${weightGrads.length} conv2d_weight_grad kernel(s) for a frozen-input conv`)
  }
  if (!ir.graph.ops.some(o => o.kind === 'conv2d_input_grad')) {
    fail("conv2d_input_grad missing — it feeds the upstream 'gain' param's gradient")
  }
  // The param grads + writeback machinery survived: Adam update ops exist
  // for the head weight, and the plan carries its gradient buffer.
  if (!ir.graph.ops.some(o => o.kind === 'adam_update_p')) fail('adam update ops missing after DCE')
  if (!ir.plan.paramGradsByName.has('head.W')) fail("plan lost the 'head.W' gradient buffer")
  ok('trace(): 0 weight-grad kernels for the frozen conv; optimizer wiring intact')
}

done('test/dce.ts')
