// Activation capture — opt-in readback of intermediate tensors at training step.
//
// Usage (inside the user's forward pass):
//
//   import { capture } from 'tensorgrad'
//
//   function attentionFwd(p, x) {
//     const scores = mul(matmulBatched(q, kT), SCALE_QK)
//     const attn = capture(`attn.${layerIdx}`, softmaxCausalLast(scores))
//     return matmulBatched(attn, v)
//   }
//
// Pass-through return type: `capture(name, t)` returns `t` unchanged so it
// inlines at the point of computation. Behind the scenes it registers `t.id`
// against `name` on the current graph; runtime exposes the registered tensors
// via `step(inputs, { withCaptures: true })`.
//
// Outside the user's forward trace (during `appendGrad` / `appendAdam`'s
// `traceInto` re-entry), `capture()` is a no-op — gradient and optimizer
// internals shouldn't accidentally publish themselves to the UI.

import type { Tensor } from './ir.js'
import { currentGraph, isCaptureEnabled } from './trace.js'

export function capture<T extends Tensor>(name: string, t: T): T {
  if (!isCaptureEnabled()) return t
  const g = currentGraph()
  if (g.captures.has(name)) {
    throw new Error(
      `capture: name '${name}' already registered. Use unique names ` +
      `(e.g. \`attn.\${layerIdx}\`) when capturing across a loop.`,
    )
  }
  g.captures.set(name, t.id)
  return t
}
