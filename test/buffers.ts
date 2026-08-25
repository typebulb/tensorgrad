// Buffer planning. Two layers of coverage:
//
//   1. A *simulator* for the whole allocator: walk the ops in dispatch order,
//      tracking which tensor's value each buffer currently holds, and check at
//      every op that each input's buffer still holds that input. This catches
//      the failure mode pooling can produce and nothing else in the suite can —
//      a buffer recycled while a later op still needs its old contents. That
//      bug is invisible to `test/_eval.ts` (which evaluates the graph, never
//      the plan) and on a GPU shows up only as a wrong pixel.
//
//   2. The reshape/stop_gradient buffer alias specifically: those kernels are
//      element-for-element copies, so the plan gives both ends one buffer and
//      codegen drops the kernel. Asserted where it must happen (interior
//      intermediates) and where it must NOT (a value that outlives the dispatch
//      sequence: graph output, capture, param).

import {
  add, mul, mean, relu, sin, concat, permute, reshape, narrow, matmul, conv2d,
  capture, stopGradient, Module, Linear, type Tensor,
} from '../src/index.js'
import { traceFn, paramInput, tensorInput } from '../src/trace.js'
import { appendGrad } from '../src/grad.js'
import { planBuffers, type BufferPlan } from '../src/buffers.js'
import { emitKernels } from '../src/codegen.js'
import { getOpInputs, type Graph } from '../src/ir.js'
import { section, ok, assert, assertEq, fail, done } from './_assert.js'

/**
 * Replay the dispatch sequence over the plan, tracking buffer contents.
 *
 * A kernel reads its input buffers and writes its output buffer. So after op
 * `i`, `holds[buf(out)]` is `out` — whatever the buffer held before is gone.
 * If some later op reads a tensor whose buffer has since been overwritten by
 * another tensor, pooling reclaimed a buffer too early. A skipped kernel
 * (`wgsl === ''`) writes nothing, which is exactly how an aliased reshape can
 * share its input's buffer without destroying it.
 */
function simulate(graph: Graph, plan: BufferPlan, label: string): void {
  const kernels = emitKernels(graph, plan)
  const holds = new Map<number, number>()   // buffer id -> tensor id it currently holds
  const buf = (tid: number) => plan.tensorToBuffer.get(tid)!
  for (let i = 0; i < graph.ops.length; i++) {
    const op = graph.ops[i]!
    const k = kernels[i]!
    const writes = !!k.wgsl && k.threads > 0
    for (const inId of getOpInputs(op)) {
      const b = buf(inId)
      const held = holds.get(b)
      if (held !== inId) {
        fail(
          `${label}: op ${i} (${op.kind}) reads tensor #${inId} from buffer ${b}, ` +
          `but that buffer holds ${held === undefined ? 'nothing' : `#${held}`}`,
        )
      }
      // A kernel binds its inputs read-only and its output read_write. WebGPU
      // rejects the same buffer in both roles, so an op that actually writes
      // must not land its output on one of its own live inputs.
      if (writes && buf(op.out) === b) {
        fail(`${label}: op ${i} (${op.kind}) writes buffer ${b} while reading tensor #${inId} from it`)
      }
    }
    // A skipped copy (reshape/stop_gradient with no kernel) is only sound if
    // the bytes are already where the output claims to be — otherwise the
    // relabel below would bless reads of a tensor nothing ever wrote.
    if (!writes && (op.kind === 'reshape' || op.kind === 'stop_gradient')) {
      const a = (op as { a: number }).a
      if (buf(a) !== buf(op.out)) {
        fail(`${label}: op ${i} (${op.kind}) emitted no kernel but input buffer ${buf(a)} != output buffer ${buf(op.out)}`)
      }
    }
    holds.set(buf(op.out), op.out)
  }
  // Everything the runtime reads out after the dispatch sequence must still be
  // intact: graph outputs and every capture.
  for (const outId of graph.outputs) {
    if (holds.get(buf(outId)) !== outId) fail(`${label}: graph output #${outId} was overwritten before readback`)
  }
  for (const [name, tid] of graph.captures) {
    if (holds.get(buf(tid)) !== tid) fail(`${label}: capture '${name}' (#${tid}) was overwritten before readback`)
  }
  ok(`${label}: every read sees its own value (${graph.ops.length} ops, ${plan.buffers.length} buffers)`)
}

const C = 8, G = 6, S = 4, GH = G * S, GGH = GH * GH

/** The chameleon bulb's decode: the reshape-heaviest graph we ship. */
function decodeGraph(): Graph {
  return traceFn(() => {
    const s = tensorInput('s', [1, C, G, G], 'f32')
    const bilin = paramInput('bilin', [C * S * S, 1, 3, 3], 'f32')
    const coords = paramInput('coords', [GGH, 4], 'f32')
    const w1 = paramInput('w1', [4 + C, 4], 'f32')
    const shifted = conv2d(s, bilin, { padding: 1, groups: C })
    const up = reshape(permute(reshape(shifted, [1, C, S, S, G, G]), [0, 4, 2, 5, 3, 1]), [GGH, C])
    const img = sin(matmul(concat([coords, up], 1), w1))
    const a = reshape(narrow(up, 1, 3, 1), [1, 1, GH, GH])
    return mul(img, reshape(relu(a), [GGH, 1])) as Tensor
  })
}

section('the allocator never hands a live buffer to someone else')
{
  simulate(decodeGraph(), planBuffers(decodeGraph(), {}), 'decode')

  // A long-lived reshape: `flat` is read at the very end, well after `x`'s own
  // last direct use — the case where sharing a buffer must extend its lifetime.
  const straddle = () => traceFn(() => {
    const x = tensorInput('x', [4, 16], 'f32')
    const flat = reshape(x, [64])
    let acc = mul(x, 2)
    for (let i = 0; i < 6; i++) acc = relu(add(acc, mul(acc, 0.5)))   // churn the free list
    return add(reshape(acc, [64]), flat) as Tensor
  })
  simulate(straddle(), planBuffers(straddle(), {}), 'reshape straddling other work')

  // Forward + backward: the pass that pooling exists for.
  const trainGraph = () => {
    const g = traceFn(() => {
      const x = tensorInput('x', [8, 16], 'f32')
      const w = paramInput('w', [16, 16], 'f32')
      const h = relu(matmul(x, w))
      return mean(mul(reshape(h, [128]), 0.5)) as Tensor
    })
    appendGrad(g)
    return g
  }
  const gTrain = trainGraph()
  simulate(gTrain, planBuffers(gTrain, {}), 'forward + backward')
}

section('an interior reshape costs nothing to dispatch')
{
  const g = decodeGraph()
  const plan = planBuffers(g, {})
  const kernels = emitKernels(g, plan)
  const reshapes = g.ops.map((op, i) => ({ op, i })).filter(x => x.op.kind === 'reshape')
  assert(reshapes.length >= 4, `decode has ${reshapes.length} reshapes to account for`)
  const dispatched = reshapes.filter(x => kernels[x.i]!.wgsl !== '')
  assertEq(dispatched.length, 0, 'reshapes that still emit a copy kernel')
  for (const { op } of reshapes) {
    if (plan.tensorToBuffer.get(op.out) !== plan.tensorToBuffer.get((op as { a: number }).a))
      fail(`reshape #${op.out} did not share its input's buffer`)
  }
  ok('every interior reshape shares its input buffer')

  const gsg = traceFn(() => {
    const x = tensorInput('x', [4, 4], 'f32')
    return mul(stopGradient(relu(x)), 2) as Tensor
  })
  const ksg = emitKernels(gsg, planBuffers(gsg, {}))
  const sgIdx = gsg.ops.findIndex(o => o.kind === 'stop_gradient')
  assertEq(ksg[sgIdx]!.wgsl, '', 'an interior stop_gradient emits no kernel either')
}

section('a reshape whose result outlives the dispatch keeps its copy')
{
  // The graph output is read back after every kernel has run; a capture is too.
  // Aliasing either onto a pooled intermediate would hand the runtime a buffer
  // some later kernel is free to overwrite, so the plan must keep them apart.
  const asOutput = traceFn(() => reshape(relu(tensorInput('x', [4, 4], 'f32')), [16]) as Tensor)
  const kOut = emitKernels(asOutput, planBuffers(asOutput, {}))
  const iOut = asOutput.ops.findIndex(o => o.kind === 'reshape')
  assert(kOut[iOut]!.wgsl !== '', 'a reshape that IS the graph output still copies')

  const withCapture = traceFn(() => {
    const x = tensorInput('x', [4, 4], 'f32')
    const flat = capture('flat', reshape(relu(x), [16]))
    return mul(flat, 2) as Tensor
  })
  const kCap = emitKernels(withCapture, planBuffers(withCapture, {}))
  const iCap = withCapture.ops.findIndex(o => o.kind === 'reshape')
  assert(kCap[iCap]!.wgsl !== '', 'a captured reshape still copies')

  // Reshaping a param would otherwise alias the param's own upload buffer.
  const fromParam = traceFn(() => {
    const w = paramInput('w', [4, 4], 'f32')
    return mul(reshape(w, [16]), 2) as Tensor
  })
  const kP = emitKernels(fromParam, planBuffers(fromParam, {}))
  const iP = fromParam.ops.findIndex(o => o.kind === 'reshape')
  assert(kP[iP]!.wgsl !== '', 'a reshape reading a param still copies')
}

section('a compiled module still plans cleanly')
{
  class MLP extends Module {
    l1 = new Linear(16, 32)
    l2 = new Linear(32, 8)
  }
  const m = new MLP()
  void m
  const g = traceFn(() => {
    const x = tensorInput('x', [4, 16], 'f32')
    const w1 = paramInput('l1.W', [16, 32], 'f32')
    const b1 = paramInput('l1.b', [32], 'f32')
    const w2 = paramInput('l2.W', [32, 8], 'f32')
    const h = relu(add(matmul(x, w1), b1))
    return matmul(reshape(h, [4, 32]), w2) as Tensor
  })
  simulate(g, planBuffers(g, {}), 'MLP with a redundant reshape')
}

done('test/buffers.ts')
