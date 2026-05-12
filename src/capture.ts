import type { Tensor } from './ir.js'
import { currentGraph, isCaptureEnabled } from './trace.js'
import type { Captures } from './runtime.js'

/**
 * Mark a tensor for readback at runtime. Returns `t` unchanged so the call
 * inlines at the point of computation. Pair with `step`/`run`'s
 * `{ withCaptures: true }` option to retrieve the activation values; without
 * that option, captures cost nothing (the tag stays metadata-only).
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

/** Slice a captured tensor named `name` into one Float32Array per head, using
 *  the static shape registered at compile time. The leading axis is treated as
 *  heads (matching `splitHeads` layout at B=1); a leading singleton batch is
 *  stripped if present so callers can pass capture names directly. Throws if
 *  the capture isn't registered or wasn't read back this call. */
export function unsplitHeads(captures: Captures, name: string): Float32Array[] {
  const flat = captures.get(name)
  const shape = captures.shapeOf(name)
  if (shape.length < 2) {
    throw new Error(`unsplitHeads: '${name}' shape needs >= 2 dims, got [${shape.join(', ')}]`)
  }
  const s = shape[0] === 1 ? shape.slice(1) : shape
  const H = s[0]!
  let stride = 1
  for (let i = 1; i < s.length; i++) stride *= s[i]!
  const expected = H * stride
  if (flat.length !== expected) {
    throw new Error(`unsplitHeads: '${name}' length ${flat.length} doesn't match shape product ${expected}`)
  }
  return Array.from({ length: H }, (_, h) => flat.slice(h * stride, (h + 1) * stride))
}
