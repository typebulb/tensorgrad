// The matmul epilogue fusion (src/fuse.ts). Two things need pinning, and neither is
// visible to any other test: that a fold changes no numbers, and that it declines every
// case where folding would be wrong.
//
// The declines matter more than the folds. This pass deletes ops, so a fold that should not
// have happened does not fail loudly — it silently drops a value someone downstream still
// needs. The autograd case is the sharpest: the pass never inspects gradients, and relies
// entirely on an activation's VJP counting as a second consumer of its own input. If that
// ever stopped being true, training graphs would start fusing and their backward would read
// a tensor nothing writes. The `trace()` case at the end is what would catch it.
//
// Correctness of the fused arithmetic is checked against the same graph left unfused, which
// is the only reference that cannot drift. Tolerance, not equality: the unfused path stores
// its intermediates into Float32Arrays and so rounds twice more than the fused one — the
// same rounding the fused GPU kernel skips by keeping the accumulator in a register.

import { traceFn, tensorInput } from '../src/trace.js'
import { matmul, add, sin, relu, mul, mean, conv2d, reshape } from '../src/ops.js'
import { capture } from '../src/capture.js'
import { fuseMatmulEpilogue } from '../src/fuse.js'
import { eliminateDeadCode } from '../src/dce.js'
import { traceForward, trace } from '../src/compile.js'
import { Module, Linear, Conv2d } from '../src/index.js'
import type { Graph, OpNode, Tensor } from '../src/ir.js'
import { evalOutput } from './_eval.js'
import { section, assert, done } from './_assert.js'

const M = 12, K = 32, N = 8

const rand = (n: number, seed: number) => {
  const a = new Float32Array(n)
  let h = (seed * 0x9e3779b9) >>> 0
  for (let i = 0; i < n; i++) { h = (h * 1664525 + 1013904223) >>> 0; a[i] = (h / 4294967296) * 2 - 1 }
  return a
}
const INPUTS: Record<string, Float32Array> = { x: rand(M * K, 1), W: rand(K * N, 2), b: rand(N, 3), wide: rand(M * N, 4) }

const matmuls = (g: Graph) => g.ops.filter(o => o.kind === 'matmul') as (OpNode & { kind: 'matmul' })[]
const kinds = (g: Graph) => g.ops.map(o => o.kind)
const has = (g: Graph, k: string) => kinds(g).includes(k as OpNode['kind'])

const maxRel = (a: Float32Array, b: Float32Array) => {
  let worst = 0
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i]! - b[i]!) / Math.max(1e-6, Math.abs(a[i]!))
    if (d > worst) worst = d
  }
  return worst
}

/** The pass leaves the GEMMs it absorbed orphaned for DCE to collect, exactly as the compile
 *  pipeline does — so every case here runs the pair, not the fold alone. */
const fuseAndSweep = (g: Graph) => { fuseMatmulEpilogue(g); eliminateDeadCode(g) }

/** Trace `build` twice — once left alone as the reference, once fused — and evaluate both. */
function bothWays(build: () => Tensor) {
  const plain = traceFn(build)
  const fused = traceFn(build)
  fuseAndSweep(fused)
  return {
    fused,
    err: maxRel(evalOutput(plain, INPUTS) as Float32Array, evalOutput(fused, INPUTS) as Float32Array),
  }
}

const xWb = () => [tensorInput('x', [M, K]), tensorInput('W', [K, N]), tensorInput('b', [N])] as const

// A conv the same way Conv2d.fwd builds one: NCHW input, [cOut, cIn, 3, 3] weight, and the
// bias broadcast by reshaping [cOut] to [1, cOut, 1, 1].
const CB = 2, CIN = 3, COUT = 4, CHW = 6
const convParts = () => [
  tensorInput('cx', [CB, CIN, CHW, CHW]),
  tensorInput('cw', [COUT, CIN, 3, 3]),
  tensorInput('cb', [COUT]),
] as const
Object.assign(INPUTS, {
  cx: rand(CB * CIN * CHW * CHW, 5), cw: rand(COUT * CIN * 9, 6), cb: rand(COUT, 7), wbias: rand(CHW, 8),
})

section('matmul -> add(bias) -> sin folds into one op')
{
  const { fused, err } = bothWays(() => { const [x, W, b] = xWb(); return sin(add(matmul(x, W), b)) })
  const mm = matmuls(fused)
  assert(mm.length === 1, `expected exactly one matmul, got ${mm.length}`)
  assert(mm[0]!.bias !== undefined, 'the bias should be folded onto the matmul')
  assert(mm[0]!.act === 'sin', `the activation should be folded onto it, got ${mm[0]!.act}`)
  assert(!has(fused, 'add'), 'no separate add op should remain')
  assert(!has(fused, 'sin'), 'no separate sin op should remain')
  assert(err < 1e-5, `fused values should match the unfused graph, max rel err ${err.toExponential(1)}`)
  section(`  ✓ one fused matmul, add + sin gone, max rel err ${err.toExponential(1)}`)
}

section('bias alone folds when there is no activation')
{
  const { fused, err } = bothWays(() => { const [x, W, b] = xWb(); return add(matmul(x, W), b) })
  const mm = matmuls(fused)
  assert(mm.length === 1 && mm[0]!.bias !== undefined, 'bias should be folded')
  assert(mm[0]!.act === undefined, 'no activation should be claimed')
  assert(err < 1e-5, `values should match, max rel err ${err.toExponential(1)}`)
  section('  ✓ bias folded, no activation claimed, values match')
}

section('an activation alone folds when there is no bias')
{
  const { fused, err } = bothWays(() => { const [x, W] = xWb(); return relu(matmul(x, W)) })
  const mm = matmuls(fused)
  assert(mm.length === 1 && mm[0]!.act === 'relu', 'activation should be folded')
  assert(mm[0]!.bias === undefined, 'no bias should be claimed')
  assert(err < 1e-5, `values should match, max rel err ${err.toExponential(1)}`)
  section('  ✓ relu folded, no bias claimed, values match')
}

// ---- the declines --------------------------------------------------------

section('conv2d -> add(channel bias) -> relu folds into one op')
{
  const { fused, err } = bothWays(() => {
    const [x, w, b] = convParts()
    return relu(add(conv2d(x, w, { padding: 1 }), reshape(b, [1, COUT, 1, 1])))
  })
  const cv = fused.ops.filter(o => o.kind === 'conv2d') as (OpNode & { kind: 'conv2d' })[]
  assert(cv.length === 1, `expected one conv2d, got ${cv.length}`)
  assert(cv[0]!.bias !== undefined, 'the channel bias should be folded onto the conv')
  assert(cv[0]!.act === 'relu', `the activation should be folded, got ${cv[0]!.act}`)
  assert(!has(fused, 'add') && !has(fused, 'relu'), 'both consumed ops should be gone')
  assert(!has(fused, 'reshape'), "the bias reshape should be dead too — the fold binds the [cOut] param")
  assert(err < 1e-5, `values should match, max rel err ${err.toExponential(1)}`)
  section(`  ✓ one fused conv2d, add + relu + reshape gone, max rel err ${err.toExponential(1)}`)
}

section('a conv bias that broadcasts along the WRONG axis is left alone')
{
  // [cOut] rank-1 broadcasts along W, not channels. Folding it as a channel bias would be
  // a silent wrong answer, which is the whole reason the shape test is on [1, cOut, 1, 1].
  const g = traceFn(() => {
    const [x, w] = convParts()
    return add(conv2d(x, w, { padding: 1 }), tensorInput('wbias', [CHW]))
  })
  fuseAndSweep(g)
  const cv = g.ops.filter(o => o.kind === 'conv2d') as (OpNode & { kind: 'conv2d' })[]
  assert(cv[0]!.bias === undefined, 'a W-axis broadcast is not a channel bias')
  assert(has(g, 'add'), 'the add op must survive')
  section('  ✓ declined; only [1, cOut, 1, 1] counts as a channel bias')
}

section('a second consumer of the intermediate blocks the fold')
{
  // h is read twice, so folding the sin would delete a value the mul still needs.
  const g = traceFn(() => { const [x, W, b] = xWb(); const h = add(matmul(x, W), b); return mul(sin(h), h) })
  fuseAndSweep(g)
  assert(matmuls(g)[0]!.act === undefined, 'sin must not be folded onto the matmul')
  assert(has(g, 'sin'), 'the sin op must survive')
  section('  ✓ declined; sin survives')
}

section('a captured intermediate blocks the fold')
{
  const g = traceFn(() => { const [x, W, b] = xWb(); return sin(capture('pre', add(matmul(x, W), b))) })
  fuseAndSweep(g)
  assert(matmuls(g)[0]!.act === undefined, 'a captured value must stay materialized')
  assert(has(g, 'sin'), 'the sin op must survive')
  section('  ✓ declined; the capture still has something to read')
}

section('a bias that is not shaped [N] is left alone')
{
  const g = traceFn(() => { const [x, W] = xWb(); return add(matmul(x, W), tensorInput('wide', [M, N])) })
  fuseAndSweep(g)
  assert(matmuls(g)[0]!.bias === undefined, 'a full-shape add is not a bias')
  assert(has(g, 'add'), 'the add op must survive')
  section('  ✓ declined; only a [N] operand is treated as a bias')
}

// ---- through the real pipeline -------------------------------------------

class Net extends Module {
  l1 = new Linear(K, N)
}

section('a forward-only compile fuses through the real pipeline')
{
  const ir = await traceForward({
    model: new Net(),
    forward: (m: Net, { x }: { x: Tensor }) => sin(m.l1.fwd(x)),
    inputs: { x: [M, K] },
  })
  const mm = matmuls(ir.graph)
  assert(mm.length === 1, `expected one matmul, got ${mm.length}`)
  assert(mm[0]!.bias !== undefined && mm[0]!.act === 'sin', "Linear's bias and the sin should both fold")
  assert(!has(ir.graph, 'add') && !has(ir.graph, 'sin'), 'both consumed ops should be gone')
  const k = ir.kernels.find(k => k.opKind === 'matmul' && k.wgsl !== '')!
  assert(k.bindings.length === 4, `the fused kernel should bind a, b, bias, out — got ${k.bindings.length}`)
  assert(k.wgsl.includes('bias[n]'), 'the emitted WGSL should read the bias at the store')
  assert(/out\[[^\]]*\] = sin\(/.test(k.wgsl) || /c\[[^\]]*\] = sin\(/.test(k.wgsl), 'the store should apply sin')
  section('  ✓ fused end to end; the kernel binds 4 buffers and stores sin((acc + bias[n]))')
}

section('a training graph folds the bias but NOT the activation')
{
  const ir = await trace({
    model: new Net(),
    loss: (m: Net, { x }: { x: Tensor }) => mean(mul(sin(m.l1.fwd(x)), 2)),
    inputs: { x: [M, K] },
    optimizer: { kind: 'sgd', lr: 0.1 },
  })
  const mm = matmuls(ir.graph)
  // No VJP reads a matmul's output — dA = dY·Bᵀ, dB = Aᵀ·dY, and add just routes the
  // cotangent — so the bias intermediate still has one consumer and folds even here.
  assert(mm.some(o => o.bias !== undefined), "the forward Linear's bias should still fold")
  assert(!has(ir.graph, 'add'), 'the bias add should be gone')
  // sin' needs cos(z), so z has two readers and the activation must stay put.
  assert(mm.every(o => o.act === undefined), 'no activation should fold into any matmul')
  assert(has(ir.graph, 'sin'), 'the forward sin must survive for its VJP to read')
  section('  ✓ bias folded; sin survives so cos(z) has an input')
}

done('test/fuse.ts')
