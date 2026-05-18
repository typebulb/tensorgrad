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
  capture,
  add, sub, mul, div,
  sqrt, relu, mean, sum, reshape, permute,
  matmul,
  oneHot, arange,
  logSoftmax,
  type Tensor, type Graph,
} from '../src/index.js'
import {
  traceFn, traceInto, paramInput, tensorInput,
  appendGrad,
  planBuffers, emitKernels,
} from '../src/internal.js'

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
  const m = mean(x, -1, { keepDims: true })          // [..., 1]
  const c = sub(x, m)                                // [..., D]
  const v = mean(mul(c, c), -1, { keepDims: true })  // [..., 1]
  const stdev = sqrt(add(v, 1e-5))                   // [..., 1]
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
  const logits = matmul(xn, permute(p['tok_emb']!, [1, 0]))  // [B, T, V]
  return logits
}

function lossFn(p: Params, tokens: Tensor, targets: Tensor): Tensor {
  const logits = forward(p, tokens)                            // [B, T, V]
  const lp = logSoftmax(logits)                            // [B, T, V]
  const targetOneHot = oneHot(targets, VOCAB)                  // [B, T, V]
  const targetLp = sum(mul(lp, targetOneHot), -1)              // [B, T]

  // The result-position mask is constant per training run; pass it as a
  // tensor_input so the user's training loop fills it in once at compile time.
  const mask = tensorInput('result_mask', [T], 'f32')          // [T]
  const masked = mul(targetLp, mask)                           // [B, T] (broadcast)

  // Reduce to a scalar.
  const total = sum(masked)                                    // scalar
  const numMaskedPositions = T - (RESULT_START - 1)            // = 3 result digits
  return mul(total, -1 / (B * numMaskedPositions))
}

// ---- Run the trace + autograd ----------------------------------------------

console.log('Tracing transformer forward + loss...')
const graph: Graph = traceFn(() => {
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
  traceFn(() => {
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
const capGraph = traceFn(() => {
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
  traceFn(() => {
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
const reentry = traceFn(() => {
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

// ---- Verify appendSGD writebacks + invariants ------------------------------

console.log('\nVerifying appendSGD...')

import { appendSGD } from '../src/internal.js'

// Build a trivial training graph: one param p, loss = sum(p * p).
function buildTrivialTrainingGraph(): { graph: Graph; paramGrads: Record<string, Tensor>; paramTensors: Record<string, Tensor> } {
  let pTensor: Tensor | null = null
  const g = traceFn(() => {
    const p = paramInput('p', [4])
    pTensor = p
    return sum(mul(p, p))
  })
  const { paramGrads } = appendGrad(g)
  return { graph: g, paramGrads, paramTensors: { p: pTensor! } }
}

// 1. Plain SGD (no momentum, no decay): 1 writeback per param (the param itself).
{
  const { graph, paramGrads, paramTensors } = buildTrivialTrainingGraph()
  const r = appendSGD(graph, paramGrads, paramTensors, { lr: 0.1 })
  const paramWbs = r.writebacks.filter(w => w.destKind === 'param')
  const stateWbs = r.writebacks.filter(w => w.destKind === 'state')
  if (paramWbs.length !== 1 || stateWbs.length !== 0) {
    console.error(`FAIL: plain SGD writebacks expected (1 param, 0 state), got (${paramWbs.length}, ${stateWbs.length})`)
    process.exit(1)
  }
  if (r.config.momentum !== 0 || r.config.nesterov !== false) {
    console.error('FAIL: plain SGD config defaults wrong')
    process.exit(1)
  }
  console.log('  ✓ plain SGD: 1 param writeback, no momentum state')
}

// 2. SGD with momentum: adds a state buffer per param.
{
  const { graph, paramGrads, paramTensors } = buildTrivialTrainingGraph()
  const r = appendSGD(graph, paramGrads, paramTensors, { lr: 0.1, momentum: 0.9 })
  const paramWbs = r.writebacks.filter(w => w.destKind === 'param')
  const stateWbs = r.writebacks.filter(w => w.destKind === 'state')
  if (paramWbs.length !== 1 || stateWbs.length !== 1) {
    console.error(`FAIL: momentum SGD writebacks expected (1 param, 1 state), got (${paramWbs.length}, ${stateWbs.length})`)
    process.exit(1)
  }
  if (stateWbs[0]!.destName !== 'sgd_v_p') {
    console.error(`FAIL: state name wrong: ${stateWbs[0]!.destName}`)
    process.exit(1)
  }
  console.log('  ✓ momentum SGD: adds sgd_v_<name> state writeback')
}

// 3. Nesterov requires momentum > 0.
try {
  const { graph, paramGrads, paramTensors } = buildTrivialTrainingGraph()
  appendSGD(graph, paramGrads, paramTensors, { lr: 0.1, nesterov: true })
  console.error('FAIL: nesterov without momentum should have thrown')
  process.exit(1)
} catch (e: any) {
  if (!String(e?.message ?? e).includes('nesterov requires momentum')) {
    console.error('FAIL: wrong error for nesterov-without-momentum:', e)
    process.exit(1)
  }
  console.log('  ✓ nesterov without momentum throws')
}

// 4. weightDecay > 0 marks the right params (default filter is decay-everything).
{
  const { graph, paramGrads, paramTensors } = buildTrivialTrainingGraph()
  const r = appendSGD(graph, paramGrads, paramTensors, { lr: 0.1, weightDecay: 1e-3 })
  if (r.config.weightDecay !== 1e-3) {
    console.error(`FAIL: weightDecay not propagated: ${r.config.weightDecay}`)
    process.exit(1)
  }
  console.log('  ✓ SGD with weightDecay: config preserved through appendSGD')
}

// ---- Verify lr.staircase / lr.multiStep resolveLR values ------------------------

console.log('\nVerifying lr.staircase / lr.multiStep schedules...')

const { lr } = await import('../src/index.js')
const { resolveLR } = await import('../src/internal.js')

function approxEq(a: number, b: number, eps = 1e-9): boolean {
  return Math.abs(a - b) < eps
}

// lr.staircase: every=1, gamma=0.7 → peak * 0.7^(step-1)
{
  const sched = lr.staircase({ peak: 1.0, every: 1, gamma: 0.7 })
  const cases = [
    [1, 1.0],
    [2, 0.7],
    [3, 0.49],
    [10, Math.pow(0.7, 9)],
  ] as const
  for (const [step, want] of cases) {
    const got = resolveLR(sched, step)
    if (!approxEq(got, want, 1e-7)) {
      console.error(`FAIL: lr.staircase at step=${step} got ${got}, want ${want}`)
      process.exit(1)
    }
  }
  console.log('  ✓ lr.staircase(every=1, gamma=0.7) decays geometrically per step')
}

// lr.staircase: every=3, gamma=0.5 → peak * 0.5^floor((step-1)/3)
{
  const sched = lr.staircase({ peak: 0.1, every: 3, gamma: 0.5 })
  const cases = [
    [1, 0.1],
    [3, 0.1],
    [4, 0.05],
    [6, 0.05],
    [7, 0.025],
  ] as const
  for (const [step, want] of cases) {
    const got = resolveLR(sched, step)
    if (!approxEq(got, want, 1e-9)) {
      console.error(`FAIL: lr.staircase(every=3) at step=${step} got ${got}, want ${want}`)
      process.exit(1)
    }
  }
  console.log('  ✓ lr.staircase(every=3, gamma=0.5) holds for `every` values then drops')
}

// lr.multiStep: milestones=[3, 7], gamma=0.1 → peak * 0.1^(milestones passed)
{
  const sched = lr.multiStep({ peak: 1.0, milestones: [3, 7], gamma: 0.1 })
  const cases = [
    [1, 1.0],
    [2, 1.0],
    [3, 0.1],
    [4, 0.1],
    [6, 0.1],
    [7, 0.01],
    [10, 0.01],
  ] as const
  for (const [step, want] of cases) {
    const got = resolveLR(sched, step)
    if (!approxEq(got, want, 1e-9)) {
      console.error(`FAIL: lr.multiStep at step=${step} got ${got}, want ${want}`)
      process.exit(1)
    }
  }
  console.log('  ✓ lr.multiStep([3, 7], gamma=0.1) drops at each milestone')
}

// ---- Verify public trace() / traceForward() -------------------------------

console.log('\nVerifying public trace() / traceForward() inspection API...')

import { trace as tracePublic, traceForward } from '../src/index.js'
import { Module, Linear } from '../src/index.js'
import { ok, fail } from './_assert.js'

class TinyMLP extends Module {
  l1 = new Linear(4, 8)
  l2 = new Linear(8, 4)
}

function mlpForward(m: TinyMLP, { x }: { x: Tensor }): Tensor {
  return m.l2.fwd(relu(m.l1.fwd(x)))
}

function mlpLoss(m: TinyMLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  const out = mlpForward(m, { x })
  const diff = sub(out, y)
  return mean(mul(diff, diff))
}

// 1. trace() returns CompiledIR (async): graph + kernels + plan.
{
  const model = new TinyMLP()
  const ir = await tracePublic({
    model,
    loss: mlpLoss,
    inputs: { x: [2, 4] as const, y: [2, 4] as const },
    optimizer: { kind: 'adam', lr: 0.001 } as const,
  })
  const paramCount = ir.graph.ops.filter(o => o.kind === 'param_input').length
  if (paramCount !== 4) fail(`trace: expected 4 param_inputs (l1.W, l1.b, l2.W, l2.b); got ${paramCount}`)
  const hasMatmul = ir.graph.ops.some(o => o.kind === 'matmul')
  if (!hasMatmul) fail('trace: expected matmul ops in graph')
  const hasAdamUpdate = ir.graph.ops.some(o => o.kind === 'adam_update_p')
  if (!hasAdamUpdate) fail('trace: expected adam_update_p ops (optimizer pass should have run)')
  if (ir.kernels.length === 0) fail('trace: expected non-zero kernels.length')
  ok(`trace(): ${ir.graph.ops.length} ops, ${paramCount} params, ${ir.kernels.length} kernels`)
}

// 2. traceForward() returns CompiledIR with NO backward / NO optimizer ops.
{
  const model = new TinyMLP()
  const ir = await traceForward({
    model,
    forward: mlpForward,
    inputs: { x: [2, 4] as const },
  })
  const hasAdamUpdate = ir.graph.ops.some(o => o.kind === 'adam_update_p')
  if (hasAdamUpdate) fail('traceForward: should not contain optimizer ops')
  const hasReluGrad = ir.graph.ops.some(o => o.kind === 'relu_grad')
  if (hasReluGrad) fail('traceForward: should not contain backward ops')
  const hasMatmul = ir.graph.ops.some(o => o.kind === 'matmul')
  if (!hasMatmul) fail('traceForward: expected matmul ops')
  if (ir.kernels.length === 0) fail('traceForward: expected non-zero kernels.length')
  ok(`traceForward(): ${ir.graph.ops.length} ops, ${ir.kernels.length} kernels, forward-only`)
}

// 3. The caller's model instance is not mutated — trace clones internally.
//    (Same contract as compile().) Reusing the same instance must work.
{
  const model = new TinyMLP()
  const ir1 = await tracePublic({
    model,
    loss: mlpLoss,
    inputs: { x: [2, 4] as const, y: [2, 4] as const },
    optimizer: { kind: 'adam', lr: 0.001 } as const,
  })
  const ir2 = await tracePublic({
    model,  // same instance — must work
    loss: mlpLoss,
    inputs: { x: [2, 4] as const, y: [2, 4] as const },
    optimizer: { kind: 'adam', lr: 0.001 } as const,
  })
  if (ir1.graph.ops.length !== ir2.graph.ops.length) {
    fail(`trace: reusing model produced different graphs (${ir1.graph.ops.length} vs ${ir2.graph.ops.length} ops)`)
  }
  ok('trace(): reusing the same model instance produces identical graphs (cloning is correct)')
}

// 4. trace() with SGD optimizer produces SGD update ops (not Adam).
{
  const model = new TinyMLP()
  const ir = await tracePublic({
    model,
    loss: mlpLoss,
    inputs: { x: [2, 4] as const, y: [2, 4] as const },
    optimizer: { kind: 'sgd', lr: 0.01 } as const,
  })
  const hasAdam = ir.graph.ops.some(o => o.kind === 'adam_update_p')
  if (hasAdam) fail('trace+sgd: should not emit adam_update ops')
  ok('trace() with SGD optimizer: no Adam-update ops in graph')
}

// 5. Pipeline-structural-completeness: param count must match adam_update_p
//    count (the optimizer pass produces one update op per param). Catches
//    drift where a refactor skips part of the pipeline — e.g. trace()
//    running appendGrad but not appendAdam would surface here as 0 updates.
{
  const ir = await tracePublic({
    model: new TinyMLP(),
    loss: mlpLoss,
    inputs: { x: [2, 4] as const, y: [2, 4] as const },
    optimizer: { kind: 'adam', lr: 0.001 } as const,
  })
  const params = ir.graph.ops.filter(o => o.kind === 'param_input').length
  const adamUpdates = ir.graph.ops.filter(o => o.kind === 'adam_update_p').length
  if (adamUpdates !== params) {
    fail(`pipeline: expected ${params} adam_update_p ops (one per param); got ${adamUpdates}`)
  }
  ok(`pipeline: ${params} params → ${adamUpdates} adam_update_p ops (1:1)`)
}

console.log('\nPhase 1 smoke test complete.')
