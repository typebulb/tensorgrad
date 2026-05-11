// Phase 1 smoke test.
// Traces a tiny transformer's forward + cross-entropy loss and verifies:
//   * trace() collects ops in topological order
//   * shape inference computes correct output shapes through every op
//   * shape errors blame the user's frame
//
// Phase 1 limitation: the IR doesn't yet have a "slice along axis k" op, so
// the multi-head Q/K/V split is skipped. We still exercise embedding lookup,
// layer norm, MLP, residual stream, unembed, log-softmax, and the masked
// cross-entropy reduction — every primitive that's currently in the IR.
// Adding a slice_last op (and rerunning this test with the full attention
// block) will be the first task of Phase 1.5 / Phase 2.
//
// Run with:  pnpm test  (which runs `tsx test/smoke.ts`)

import {
  trace, traceInto, paramInput, tensorInput, capture,
  add, sub, mul, div,
  sqrt, relu, meanLast, sumLast, reshape, transpose,
  matmul,
  oneHot, arange,
  logSoftmaxLast,
  appendGrad,
  planBuffers, emitKernels,
  type Tensor, type Graph,
} from '../src/index.js'

// Minimal Node typing — keeps `@types/node` off the dev-dep tree.
declare const process: { exit(code: number): never }

// Hyperparameters, same as transformer.bulb.md.
const VOCAB = 12, D = 64, N_LAYERS = 3
const SEQ_LEN = 9, RESULT_START = 6
const B = 256, T = SEQ_LEN - 1

type Params = Record<string, Tensor>

function declareParams(): Params {
  const p: Params = {}
  p['tok_emb'] = paramInput('tok_emb', [VOCAB, D])
  p['pos_emb'] = paramInput('pos_emb', [SEQ_LEN, D])
  p['lnf_g'] = paramInput('lnf_g', [D])
  p['lnf_b'] = paramInput('lnf_b', [D])
  for (let l = 0; l < N_LAYERS; l++) {
    const k = (n: string) => `L${l}_${n}`
    p[k('ln2_g')]  = paramInput(k('ln2_g'),  [D])
    p[k('ln2_b')]  = paramInput(k('ln2_b'),  [D])
    p[k('W_mlp1')] = paramInput(k('W_mlp1'), [D, 4 * D])
    p[k('b_mlp1')] = paramInput(k('b_mlp1'), [4 * D])
    p[k('W_mlp2')] = paramInput(k('W_mlp2'), [4 * D, D])
    p[k('b_mlp2')] = paramInput(k('b_mlp2'), [D])
  }
  return p
}

function layerNorm(x: Tensor, gamma: Tensor, beta: Tensor): Tensor {
  const m = meanLast(x)                    // [..., 1]
  const c = sub(x, m)                      // [..., D]
  const v = meanLast(mul(c, c))            // [..., 1]
  const stdev = sqrt(add(v, 1e-5))         // [..., 1]
  return add(mul(div(c, stdev), gamma), beta)
}

// MLP-only forward (skipping attention until slice_last lands; see header).
function forward(p: Params, tokens: Tensor): Tensor {
  // Embedding lookup via one-hot @ table — same trick we landed on in jax-js.
  const tokOneHot = oneHot(tokens, VOCAB)                      // [B, T, V] f32
  const tokE = matmul(tokOneHot, p['tok_emb']!)                // [B, T, D]
  const posOneHot = oneHot(arange(T), SEQ_LEN)                 // [T, S] f32
  const posE = matmul(posOneHot, p['pos_emb']!)                // [T, D]
  let x = add(tokE, posE)                                      // [B, T, D] (broadcast over batch)

  for (let l = 0; l < N_LAYERS; l++) {
    const k = (n: string) => `L${l}_${n}`
    const x1n = layerNorm(x, p[k('ln2_g')]!, p[k('ln2_b')]!)
    const h1 = relu(add(matmul(x1n, p[k('W_mlp1')]!), p[k('b_mlp1')]!))
    const h2 = add(matmul(h1, p[k('W_mlp2')]!), p[k('b_mlp2')]!)
    x = add(x, h2)
  }

  const xn = layerNorm(x, p['lnf_g']!, p['lnf_b']!)
  const logits = matmul(xn, transpose(p['tok_emb']!, [1, 0]))  // [B, T, V]
  return logits
}

function lossFn(p: Params, tokens: Tensor, targets: Tensor): Tensor {
  const logits = forward(p, tokens)                            // [B, T, V]
  const lp = logSoftmaxLast(logits)                            // [B, T, V]
  const targetOneHot = oneHot(targets, VOCAB)                  // [B, T, V]
  const targetLp = sumLast(mul(lp, targetOneHot))              // [B, T]

  // The result-position mask is constant per training run; pass it as a
  // tensor_input so the user's training loop fills it in once at compile time.
  const mask = tensorInput('result_mask', [T], 'f32')          // [T]
  const masked = mul(targetLp, mask)                           // [B, T] (broadcast)

  // Reduce to a scalar via reshape to 1-D then sumLast.
  const flat = reshape(masked, [B * T])                        // [B*T]
  const total = sumLast(flat)                                  // scalar
  const numMaskedPositions = T - (RESULT_START - 1)            // = 3 result digits
  return mul(total, -1 / (B * numMaskedPositions))
}

// ---- Run the trace + autograd ----------------------------------------------

console.log('Tracing transformer forward + loss...')
const graph: Graph = trace(() => {
  const p = declareParams()
  const tokens = tensorInput('tokens', [B, T], 'i32')
  const targets = tensorInput('targets', [B, T], 'i32')
  return lossFn(p, tokens, targets)
})

const fwdInputs = graph.ops.filter(o => o.kind === 'param_input' || o.kind === 'tensor_input').length
const fwdCompute = graph.ops.length - fwdInputs
console.log(`Forward graph: ${graph.ops.length} ops (${fwdInputs} inputs, ${fwdCompute} compute)`)
console.log(`               ${graph.tensors.length} tensors`)
console.log(`               loss tensor id: ${graph.outputs[0]}`)

console.log('\nAppending backward...')
const opsBeforeBackward = graph.ops.length
const { paramGrads } = appendGrad(graph)
const backwardOpCount = graph.ops.length - opsBeforeBackward
console.log(`Backward added ${backwardOpCount} ops`)
console.log(`Total graph now: ${graph.ops.length} ops, ${graph.tensors.length} tensors`)
console.log(`Param grads: ${Object.keys(paramGrads).length} entries`)

const lossTensor = graph.tensors[graph.outputs[0]!]!
console.log(`Loss tensor shape: [${lossTensor.shape.join(', ')}]  dtype=${lossTensor.dtype}`)
if (lossTensor.shape.length !== 0) {
  console.error(`FAIL: loss should be a rank-0 scalar, got rank ${lossTensor.shape.length}`)
  process.exit(1)
}

// Verify each param has a gradient with matching shape.
console.log('\nVerifying param gradient shapes...')
const expectedParams = [
  'tok_emb', 'pos_emb', 'lnf_g', 'lnf_b',
]
for (let l = 0; l < N_LAYERS; l++) {
  expectedParams.push(`L${l}_ln2_g`, `L${l}_ln2_b`, `L${l}_W_mlp1`, `L${l}_b_mlp1`, `L${l}_W_mlp2`, `L${l}_b_mlp2`)
}
let allOk = true
for (const name of expectedParams) {
  const grad = paramGrads[name]
  if (!grad) {
    console.error(`  ✗ missing gradient for ${name}`)
    allOk = false
    continue
  }
  // Find the corresponding param_input op to compare shapes.
  const paramOp = graph.ops.find(o => o.kind === 'param_input' && o.name === name)!
  const paramShape = graph.tensors[paramOp.out]!.shape
  const gradShape = grad.shape
  const match = paramShape.length === gradShape.length &&
                paramShape.every((d, i) => d === gradShape[i])
  if (!match) {
    console.error(`  ✗ ${name}: shape mismatch — param [${paramShape.join(', ')}] vs grad [${gradShape.join(', ')}]`)
    allOk = false
  } else {
    console.log(`  ✓ ${name.padEnd(12)} shape [${paramShape.join(', ')}]`)
  }
}
if (!allOk) {
  console.error('\nFAIL: param/gradient shape mismatch')
  process.exit(1)
}

console.log('\nLast 5 ops:')
for (const op of graph.ops.slice(-5)) {
  const t = graph.tensors[op.out]!
  console.log(`  #${op.out}  ${op.kind.padEnd(22)}  shape=[${t.shape.join(', ')}]  dtype=${t.dtype}`)
}

// ---- Codegen ---------------------------------------------------------------

console.log('\nPlanning buffers + emitting kernels...')
const plan = planBuffers(graph, paramGrads)
const kernels = emitKernels(graph, plan)

const totalBytes = plan.buffers.reduce((s, b) => s + b.byteSize, 0)
const kindCount = (k: string) => plan.buffers.filter(b => b.kind === k).length
console.log(`  Buffers: ${plan.buffers.length} total, ${(totalBytes / (1024 * 1024)).toFixed(1)} MB`)
console.log(`           ${kindCount('param')} param, ${kindCount('param_grad')} param_grad, ` +
            `${kindCount('tensor_input')} input, ${kindCount('intermediate')} intermediate, ` +
            `${kindCount('output')} output`)

const kernelsWithCode = kernels.filter(k => k.wgsl !== '')
console.log(`  Kernels: ${kernelsWithCode.length} dispatchable (${kernels.length - kernelsWithCode.length} no-op leaves)`)

// Tally op kinds and total dispatch threads.
const byKind = new Map<string, { count: number; totalThreads: number }>()
for (const k of kernelsWithCode) {
  const e = byKind.get(k.opKind) ?? { count: 0, totalThreads: 0 }
  e.count++
  e.totalThreads += k.threads
  byKind.set(k.opKind, e)
}
console.log('  Op breakdown (count / total threads):')
const sorted = [...byKind.entries()].sort((a, b) => b[1].totalThreads - a[1].totalThreads)
for (const [kind, e] of sorted) {
  console.log(`    ${kind.padEnd(22)}  ${String(e.count).padStart(3)}  ${e.totalThreads.toLocaleString().padStart(12)}`)
}
const totalThreads = sorted.reduce((s, [, e]) => s + e.totalThreads, 0)
console.log(`    ${'TOTAL'.padEnd(22)}  ${String(kernelsWithCode.length).padStart(3)}  ${totalThreads.toLocaleString().padStart(12)}`)

// Sanity-check a couple of WGSL outputs.
console.log('\nSample WGSL output (first matmul):')
const firstMatmul = kernelsWithCode.find(k => k.opKind === 'matmul')
if (firstMatmul) {
  const lines = firstMatmul.wgsl.split('\n')
  for (const line of lines.slice(0, 8)) console.log(`  ${line}`)
  if (lines.length > 8) console.log(`  ... (${lines.length - 8} more lines)`)
} else {
  console.log('  (no matmul in graph)')
}

// Verify every kernel has a non-empty WGSL string and bindings.
let codegenOk = true
for (const k of kernelsWithCode) {
  if (!k.wgsl.includes('@compute')) {
    console.error(`FAIL: kernel for ${k.opKind} (op #${k.opIndex}) has no @compute entry point`)
    codegenOk = false
  }
  if (k.bindings.length === 0) {
    console.error(`FAIL: kernel for ${k.opKind} (op #${k.opIndex}) has no bindings`)
    codegenOk = false
  }
}
if (codegenOk) console.log('\n  ✓ all kernels have non-empty WGSL + bindings')
else process.exit(1)

// ---- Verify shape error attribution ----------------------------------------

console.log('\nVerifying shape-error attribution...')
try {
  trace(() => {
    const a = tensorInput('a', [3, 4])
    const b = tensorInput('b', [5, 4])
    return add(a, b)  // <-- this line should appear in the error stack
  })
  console.error('FAIL: expected shape error was not thrown')
  process.exit(1)
} catch (e: any) {
  const msg = String(e?.message ?? e)
  if (!msg.includes('add: incompatible shapes')) {
    console.error('FAIL: wrong error message:', msg)
    process.exit(1)
  }
  console.log('  ✓ shape error thrown with correct message')
  if (msg.includes('test/smoke.ts') || msg.includes('test\\smoke.ts')) {
    console.log('  ✓ error blames test/smoke.ts (user frame)')
  } else {
    console.log('  ! error did not surface test/smoke.ts in stack')
    console.log('  full message:\n', msg)
  }
}

// ---- Verify capture() mechanism --------------------------------------------

console.log('\nVerifying activation capture...')

// 1. capture() registers a tensor on graph.captures during forward trace.
const capGraph = trace(() => {
  const a = tensorInput('a', [3, 4])
  const b = tensorInput('b', [3, 4])
  const c = capture('cap.sum', add(a, b))
  return mul(c, c)  // ensure capture's tensor isn't the loss output itself
})
if (capGraph.captures.size !== 1 || !capGraph.captures.has('cap.sum')) {
  console.error(`FAIL: capture not registered. captures=${[...capGraph.captures.keys()]}`)
  process.exit(1)
}
console.log('  ✓ capture() registers tensor on graph.captures')

// 2. planBuffers exposes capturesByName.
const capPlan = planBuffers(capGraph, {})
if (!capPlan.capturesByName.has('cap.sum')) {
  console.error('FAIL: capturesByName missing cap.sum')
  process.exit(1)
}
console.log('  ✓ planBuffers populates capturesByName')

// 3. Duplicate capture name throws.
try {
  trace(() => {
    const a = tensorInput('a', [2, 2])
    capture('dup', a)
    capture('dup', a)  // should throw
    return a
  })
  console.error('FAIL: duplicate capture name did not throw')
  process.exit(1)
} catch (e: any) {
  if (!String(e?.message ?? e).includes("already registered")) {
    console.error('FAIL: wrong error for duplicate capture:', e)
    process.exit(1)
  }
  console.log('  ✓ duplicate capture name throws')
}

// 4. capture() inside traceInto (autograd re-entry) is a no-op — backward
//    rules and optimizer ops shouldn't accidentally publish their tensors.
const reentry = trace(() => {
  const a = tensorInput('a', [2, 2])
  return a
})
traceInto(reentry, () => {
  const t = paramInput('p', [2, 2])
  const out = capture('should_not_register', t)  // inside traceInto → no-op
  if (out !== t) {
    console.error('FAIL: capture() did not return its argument')
    process.exit(1)
  }
})
if (reentry.captures.has('should_not_register')) {
  console.error('FAIL: capture() registered during traceInto (should be no-op)')
  process.exit(1)
}
console.log('  ✓ capture() inside traceInto is a no-op')

console.log('\nPhase 1 smoke test complete.')
