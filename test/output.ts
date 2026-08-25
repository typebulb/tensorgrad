// The forward spec's `output` declaration is validated against the traced
// graph at compile; that check is what keeps `r.output`'s TypedArray class
// honest at every call site. `'rgba8'` is the one declaration that names an
// OP rather than a dtype — the graph must end in packRGBA8 — so its rule is
// pinned here alongside the plain dtype matches.

import { Module, traceForward, packRGBA8, relu, argmax, type Tensor } from '../src/index.js'
import { assertOutputDtype } from '../src/compile.js'
import { section, ok, fail, done } from './_assert.js'

class Plain extends Module {}
const forwardIR = (f: (x: Tensor) => Tensor) =>
  traceForward({ model: new Plain(), forward: (_m: Plain, { x }: { x: Tensor }) => f(x), inputs: { x: [3, 4] } })

const throws = (f: () => void): boolean => { try { f() } catch { return true } return false }

;(async () => {
  section('output declaration vs traced graph')
  const packed = await forwardIR(x => packRGBA8(relu(x)))
  const floats = await forwardIR(x => relu(x))
  const ints = await forwardIR(x => argmax(x))

  if (throws(() => assertOutputDtype(packed, 'rgba8', 'compileForward'))) fail("'rgba8' should accept a graph ending in packRGBA8")
  if (throws(() => assertOutputDtype(packed, 'i32', 'compileForward'))) fail("'i32' is the packed tensor's real dtype and should still be accepted")
  if (!throws(() => assertOutputDtype(packed, 'f32', 'compileForward'))) fail("'f32' on a packed output must throw")
  ok("packRGBA8 tail: 'rgba8' and 'i32' accepted, 'f32' rejected")

  if (!throws(() => assertOutputDtype(floats, 'rgba8', 'attach'))) fail("'rgba8' on an f32 graph must throw")
  if (!throws(() => assertOutputDtype(ints, 'rgba8', 'attach'))) fail("'rgba8' on an argmax graph must throw — i32, but indices, not bytes")
  if (throws(() => assertOutputDtype(ints, 'i32', 'attach'))) fail("'i32' on an argmax graph should pass")
  if (throws(() => assertOutputDtype(floats, 'f32', 'attach'))) fail("'f32' on an f32 graph should pass")
  ok("'rgba8' names the op, not the dtype: rejected on f32 and on argmax's i32")

  done('test/output.ts')
})()
