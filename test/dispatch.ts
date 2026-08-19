// Batched kernels must carry the batch on the z dispatch axis, never packed into the linear
// thread index and divided back out.
//
// This is a correctness test wearing the clothes of a style test. Qualcomm's Adreno
// miscompiles `bi = i / (M*N)` — with M*N at 512 or 1024 and a reduction of 72 or more it
// returns a wrong answer, silently, with every operand exact. Measured over 68 shapes; the
// naive kernel was wrong at 22 of them and right at all 68 once the division went away. The
// full account is in specs/Adreno-matmul.md.
//
// Nothing here needs a GPU: it reads the emitted WGSL and the dispatch shape, which is where
// the bug lives. The device that actually miscompiles is in nobody's CI, so a shape-level
// guard is the only kind of regression test available.

import { matmul, reshape, conv2d, maxPool2d, type Tensor } from '../src/index.js'
import { traceFn, tensorInput } from '../src/trace.js'
import { planBuffers } from '../src/buffers.js'
import { emitKernels, type KernelSpec } from '../src/codegen.js'
import { section, ok, assert, assertEq, fail, done } from './_assert.js'

/** Emit the kernels for one matmul at a shape, without a device. */
function kernelsFor(aShape: number[], bShape: number[]): KernelSpec[] {
  const graph = traceFn(() => {
    const a = tensorInput('a', aShape, 'f32')
    const b = tensorInput('b', bShape, 'f32')
    return matmul(a, b)
  })
  return emitKernels(graph, planBuffers(graph, {}))
}

const matmulKernels = (ks: KernelSpec[]) => ks.filter(k => k.wgsl && k.opKind.startsWith('matmul'))

/** The construct under interdict: a batch index recovered by dividing a packed thread id. */
const PACKED_BATCH = /let\s+bi\s*=\s*\w+\s*\/\s*\d+u/

section('no kernel recovers a batch index by division')
{
  // Each of these took the packed form before 0.4.7, and the first is the exact shape the
  // watermarking bulb's attention emits — 16 heads of [256,256] @ [256,4].
  const cases: { label: string; a: number[]; b: number[] }[] = [
    { label: 'attention attn@v  [1,16,256,256] @ [1,16,256,4]', a: [1, 16, 256, 256], b: [1, 16, 256, 4] },
    { label: 'attention qk^T    [1,16,256,4] @ [1,16,4,256]', a: [1, 16, 256, 4], b: [1, 16, 4, 256] },
    { label: 'tiled GEMM        [1,4,256,64] @ [1,4,64,64]', a: [1, 4, 256, 64], b: [1, 4, 64, 64] },
    { label: 'reduced path      [1,2,64,512] @ [512,4]', a: [1, 2, 64, 512], b: [512, 4] },
    { label: 'unbatched         [1,256,64] @ [64,64]', a: [1, 256, 64], b: [64, 64] },
  ]
  for (const c of cases) {
    const ks = matmulKernels(kernelsFor(c.a, c.b))
    if (ks.length === 0) fail(`${c.label}: no matmul kernel emitted`)
    for (const k of ks) {
      if (PACKED_BATCH.test(k.wgsl)) fail(`${c.label}: ${k.opKind} still divides a packed index for bi`)
    }
    ok(c.label)
  }
}

section('the z axis carries the batch, and threads count one slice')
{
  const ks = matmulKernels(kernelsFor([1, 16, 256, 256], [1, 16, 256, 4]))
  const k = ks[0]!
  assertEq(k.dispatchZ, 16, 'dispatchZ is the batch')
  assertEq(k.threads, 256 * 4, 'threads counts one z-slice')
  assert(/let\s+bi\s*=\s*gid\.z/.test(k.wgsl), 'bi reads the z builtin')
}

section('an unbatched matmul dispatches a flat grid')
{
  const ks = matmulKernels(kernelsFor([1, 256, 64], [64, 64]))
  const k = ks[0]!
  assertEq(k.dispatchZ, 1, 'dispatchZ is 1 when there is one slice')
}

section('past the 65535 cap the batch goes back into the index')
{
  // dispatchWorkgroups caps every dimension at 65535, so beyond it the z axis cannot carry
  // the batch and the packed form is the only option. It is safe there: the miscompile needs
  // M*N at 512 or 1024, and this many slices at that width is a 33M-element output.
  const ks = matmulKernels(kernelsFor([70000, 8, 8], [8, 8]))
  const k = ks[0]!
  assertEq(k.dispatchZ, undefined, 'no z axis above the cap')
  assert(PACKED_BATCH.test(k.wgsl), 'falls back to the packed index')
}

section('every dispatch dimension stays inside its cap')
{
  const cases: [number[], number[]][] = [
    [[1, 16, 256, 256], [1, 16, 256, 4]],
    [[1, 1, 1, 64], [64, 50257]],
    [[1, 3, 197, 197], [1, 3, 197, 64]],
    [[70000, 8, 8], [8, 8]],
  ]
  for (const [a, b] of cases) {
    for (const k of matmulKernels(kernelsFor(a, b))) {
      const wgCount = Math.max(1, Math.ceil(k.threads / k.workgroupSize))
      const x = Math.min(wgCount, 65535)
      const y = Math.ceil(wgCount / 65535)
      const z = k.dispatchZ ?? 1
      if (x > 65535 || y > 65535 || z > 65535) {
        fail(`${JSON.stringify(a)}: dispatch (${x}, ${y}, ${z}) exceeds the per-dimension cap`)
      }
    }
  }
  ok('all four shapes dispatch inside (65535, 65535, 65535)')
}

section('reshape-only graphs still emit nothing to dispatch')
{
  const graph = traceFn(() => {
    const a = tensorInput('a', [1, 16, 256, 256], 'f32')
    return reshape(a, [16, 256, 256]) as Tensor
  })
  const ks = emitKernels(graph, planBuffers(graph, {}))
  assert(ks.every(k => !k.wgsl || k.threads > 0), 'no kernel asks for a zero-thread dispatch')
}

section('the emitted guard and the dispatch extent agree')
{
  // A kernel's bounds check and the grid it is dispatched on are two halves of one decision.
  // If they diverge the kernel silently drops work or writes past its output — no validation
  // error, no device loss, just a wrong answer, which is the failure mode this whole file
  // exists to prevent. `emitTiledMatmul` returns its WGSL and its spec together so the pair
  // cannot drift; this checks every path, including the two that build theirs separately.
  const cases: [number[], number[]][] = [
    [[1, 4, 256, 64], [1, 4, 64, 64]],     // tiled GEMM, batched
    [[1, 256, 64], [64, 64]],              // tiled GEMM, unbatched
    [[1, 16, 256, 256], [1, 16, 256, 4]],  // naive, one thread per output
    [[1, 2, 64, 512], [512, 4]],           // workgroup-per-output reduced path
  ]
  for (const [a, b] of cases) {
    for (const k of matmulKernels(kernelsFor(a, b))) {
      const m = />=\s*(\d+)u/.exec(k.wgsl)
      if (!m) fail(`${JSON.stringify(a)}: ${k.opKind} emits no bounds check`)
      // A kernel indexed by workgroup counts workgroups' worth of threads; one indexed by
      // thread counts threads. Either way the guard is the per-slice extent.
      const perSlice = k.wgsl.includes('wid.x') ? k.threads / k.workgroupSize : k.threads
      assertEq(Number(m![1]), perSlice, `${JSON.stringify(a)} ${k.opKind}: guard matches its grid`)
    }
  }
}

section('conv and pool put two axes on the dispatch grid')
{
  // Four output axes, three dispatch axes, so one division is unavoidable — but it is by the
  // last extent rather than by a product, and the index it divides is a bare `gid.x` with no
  // 2-D fold. That combination is the only one measured correct across the whole Adreno
  // sweep; the fully packed form is wrong at 20 of 93 shapes. specs/Adreno-matmul.md.
  const conv = traceFn(() => {
    const x = tensorInput('x', [8, 32, 32, 32], 'f32')
    const w = tensorInput('w', [128, 1, 3, 3], 'f32')
    return conv2d(x, w, { padding: 1, groups: 32 }) as Tensor
  })
  const ck = emitKernels(conv, planBuffers(conv, {})).find(k => k.opKind === 'conv2d')!
  assertEq(ck.dispatchZ, 8, 'conv2d: batch on z')
  assertEq(ck.dispatchY, 128, 'conv2d: output channel on y')
  assertEq(ck.threads, 32 * 32, 'conv2d: threads count one spatial slice')
  assert(!/gid\.x\s*\+\s*gid\.y/.test(ck.wgsl), 'conv2d: no 2-D fold in the index expression')

  const pool = traceFn(() => {
    const x = tensorInput('x', [8, 32, 32, 32], 'f32')
    return maxPool2d(x, 2) as Tensor
  })
  const pk = emitKernels(pool, planBuffers(pool, {})).find(k => k.opKind === 'max_pool_2d')!
  assertEq(pk.dispatchZ, 8, 'max_pool_2d: batch on z')
  assertEq(pk.dispatchY, 32, 'max_pool_2d: channel on y')
  assertEq(pk.threads, 16 * 16, 'max_pool_2d: threads count one spatial slice')
  assert(!/gid\.x\s*\+\s*gid\.y/.test(pk.wgsl), 'max_pool_2d: no 2-D fold in the index expression')

  // The guard and the grid have to agree here too, and these kernels build the pair in one
  // place so they cannot drift.
  for (const k of [ck, pk]) {
    const m = />=\s*(\d+)u/.exec(k.wgsl)
    if (!m) fail(`${k.opKind} emits no bounds check`)
    assertEq(Number(m![1]), k.threads, `${k.opKind}: guard matches its grid`)
  }
}

done('test/dispatch.ts')
