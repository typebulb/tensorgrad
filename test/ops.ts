// Op-level invariants that aren't covered by samples, smoke, or the FD
// harness. Specifically: dropout's two structural invariants (a single
// shared seed input across all dropouts in a graph, and a unique salt
// per dropout call), plus the p=0 short-circuit.
//
// Per-op shape rules, forward correctness, and backward correctness are
// all validated elsewhere — by samples (live use), by smoke (full
// transformer trace), and by the FD harness in test/grad.ts. Tests for
// stable shape rules and obvious literal guards were pruned as padding.

import { dropout } from '../src/index.js'
import { traceFn, tensorInput } from '../src/trace.js'
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

done('test/ops.ts')
