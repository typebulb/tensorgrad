import type { Tensor } from './ir.js'
import { currentGraph, isCaptureEnabled } from './trace.js'

/**
 * Mark a tensor for readback at runtime. Returns `t` unchanged so the call
 * inlines at the point of computation. The marked tensor is read back as
 * part of every `step()`/`run()` and surfaced via the result's `captures`
 * field; graphs with no `capture(...)` sites pay nothing.
 *
 * Names must be unique within a single trace. Outside the user's forward
 * trace (during `appendGrad` / `appendAdam`'s `traceInto` re-entry), this
 * is a no-op so gradient and optimizer internals don't accidentally publish
 * themselves.
 */
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
