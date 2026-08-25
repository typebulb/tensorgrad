// Op-level invariants that aren't covered by samples, smoke, or the FD
// harness. Specifically: dropout's two structural invariants (a single
// shared seed input across all dropouts in a graph, and a unique salt
// per dropout call), plus the p=0 short-circuit.
//
// Per-op shape rules, forward correctness, and backward correctness are
// all validated elsewhere — by samples (live use), by smoke (full
// transformer trace), and by the FD harness in test/grad.ts. Tests for
// stable shape rules and obvious literal guards were pruned as padding.

import { dropout, leakyRelu, packRGBA8 } from '../src/index.js'
import { traceFn, tensorInput } from '../src/trace.js'
import { evalOutput } from './_eval.js'
import { section, ok, fail, done } from './_assert.js'

section('dropout — auto-managed per-op salt + shared seed input')

// 1. p === 0 short-circuits to identity (no `dropout` IR op emitted).
//    The fast-path is what makes `dropout(x, cfg.pDrop)` with cfg.pDrop=0
//    a true no-op rather than a useless mask-of-ones kernel.
{
  const g = traceFn(() => dropout(tensorInput('x', [4]), 0))
  if (g.ops.some(o => o.kind === 'dropout')) fail('dropout(x, 0) should not emit an op')
  ok('dropout(x, 0) short-circuits (no IR emitted)')
}

// 2. Multiple dropouts in one graph share a single __prngSeed input
//    AND each gets a unique salt. The shared seed is what lets the
//    runtime auto-inject one i32 per step; the unique salt is what
//    makes different dropout calls produce different masks while
//    forward + backward of the same call produce identical masks.
{
  const g = traceFn(() => {
    const x = tensorInput('x', [4])
    return dropout(dropout(x, 0.1), 0.2)
  })
  const seedCount = g.ops.filter(o => o.kind === 'tensor_input' && o.name === '__prngSeed').length
  if (seedCount !== 1) fail(`__prngSeed should be shared; got ${seedCount} occurrences`)

  const dropoutOps = g.ops.filter(o => o.kind === 'dropout') as Array<{ kind: 'dropout'; salt: number }>
  const salts = dropoutOps.map(o => o.salt)
  if (new Set(salts).size !== salts.length) fail(`dropout salts must be unique: ${salts}`)
  ok(`${dropoutOps.length} dropouts: 1 shared seed input, unique salts [${salts.join(', ')}]`)
}

// leakyRelu forward correctness for alpha >= 1 — not covered by the FD harness
// (self-consistent against its own forward) nor samples (which use alpha < 1).
// The old `max(x, alpha·x)` form was silently wrong here.
section('leakyRelu — correct for alpha >= 1 (regression guard)')
{
  const g = traceFn(() => leakyRelu(tensorInput('x', [4]), 2))
  const out = evalOutput(g, { x: new Float32Array([-1, -0.5, 0.5, 1]) }) as Float32Array
  const want = [-2, -1, 0.5, 1]  // PyTorch: x<0 → 2x, x>=0 → x
  if (!want.every((w, i) => Math.abs(out[i]! - w) < 1e-6)) {
    fail(`leakyRelu(α=2) = [${[...out]}], want [${want.join(', ')}]`)
  }
  ok(`leakyRelu(α=2): x<0 → 2x, x>=0 → x — [${[...out].join(', ')}]`)
}

// packRGBA8 pins WGSL pack4x8unorm's exact byte semantics on the CPU
// reference: saturate, ×255, round-half-up, R in the low byte — so an
// i32 readback views as ImageData bytes with no host-side pass.
section('packRGBA8 — pack4x8unorm semantics: saturate, round, RGBA in memory order')
{
  const g = traceFn(() => packRGBA8(tensorInput('x', [2, 4])))
  const outT = g.tensors[g.outputs[0]!]!
  if (outT.dtype !== 'i32' || outT.shape.length !== 1 || outT.shape[0] !== 2) {
    fail(`packRGBA8([2, 4]) should be i32 [2], got ${outT.dtype} [${outT.shape}]`)
  }
  const out = evalOutput(g, { x: new Float32Array([0, 0.5, 1, 2, -1, 0.2, 0.998, 0.4]) }) as Int32Array
  const bytes = new Uint8ClampedArray(out.buffer, out.byteOffset, out.byteLength)
  // 0.5 → floor(0.5 + 127.5) = 128; 2 and -1 saturate; 0.998 → floor(0.5 + 254.49) = 254
  const want = [0, 128, 255, 255, 0, 51, 254, 102]
  for (let i = 0; i < want.length; i++) {
    if (bytes[i] !== want[i]) fail(`packRGBA8 byte ${i}: got ${bytes[i]}, want ${want[i]}`)
  }
  ok(`bytes [${[...bytes].join(', ')}] — i32 readback views straight as Uint8ClampedArray`)

  let threw = false
  try { traceFn(() => packRGBA8(tensorInput('x', [3, 3]))) } catch { threw = true }
  if (!threw) fail('packRGBA8 should reject a last axis that is not 4')
  ok('a last axis other than 4 is a trace-time shape error')
}

done('test/ops.ts')
