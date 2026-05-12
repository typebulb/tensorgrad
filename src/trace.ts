// Holds the "current graph" in module-local state so user code can call ops
// without threading a graph parameter through every function.

import type { Graph, Tensor, Shape, Dtype } from './ir.js'
import { makeGraph, addOp, captureSite } from './ir.js'

let _current: Graph | null = null
// True only during the user's forward trace; false during `traceInto` so
// autograd / optimizer rules don't accidentally publish via `capture()`.
let _captureEnabled = false

export function currentGraph(): Graph {
  if (!_current) {
    throw new Error(
      'tensorgrad: ops can only be called inside trace(). ' +
      'Did you forget to wrap your forward pass?',
    )
  }
  return _current
}

export function isCaptureEnabled(): boolean {
  return _captureEnabled
}

/**
 * Run `fn` with a fresh `Graph` as the active trace context; return the graph.
 * Every op call inside `fn` appends to this graph. `fn` returns the tensor
 * (or array of tensors) to mark as the graph's outputs.
 *
 * Single-threaded and non-reentrant — calling `trace` while another trace is
 * active throws. Op calls outside any `trace(...)` also throw.
 */
export function trace(fn: () => Tensor | Tensor[]): Graph {
  if (_current) {
    throw new Error('tensorgrad: nested trace() is not supported')
  }
  const g = makeGraph()
  _current = g
  _captureEnabled = true
  try {
    const result = fn()
    const outputs = Array.isArray(result) ? result : [result]
    for (const t of outputs) {
      ;(g.outputs as number[]).push(t.id)
    }
  } finally {
    _current = null
    _captureEnabled = false
  }
  return g
}

/**
 * Re-enter an existing `Graph` to append more ops. Used by autograd
 * (`appendGrad`) and optimizer passes (`appendAdam`, `appendSGD`) to extend
 * a graph that's already been traced. Capture is intentionally disabled
 * here — backward / optimizer rules shouldn't publish their internal
 * tensors via `capture()`. Returns whatever `fn` returns.
 */
export function traceInto<T>(g: Graph, fn: () => T): T {
  if (_current) {
    throw new Error('tensorgrad: traceInto() called while another trace is active')
  }
  _current = g
  try {
    return fn()
  } finally {
    _current = null
  }
}

type NamedInputKind = 'param_input' | 'tensor_input' | 'state_input'
function assertNameUnused(g: Graph, name: string, kinds: NamedInputKind[], label: string): void {
  if (g.ops.some(op => kinds.includes(op.kind as NamedInputKind) && (op as { name?: string }).name === name)) {
    throw new Error(`tensorgrad: ${label} name '${name}' already used in this trace`)
  }
}

/** Declare a trainable parameter leaf. The compiled runtime expects a
 *  `Float32Array` of matching size at upload time, keyed by `name`. Shares
 *  its namespace with `tensorInput` — names must be unique across both. */
export function paramInput(name: string, shape: Shape, dtype: Dtype = 'f32'): Tensor {
  const g = currentGraph()
  assertNameUnused(g, name, ['param_input', 'tensor_input'], 'input')
  const site = captureSite('paramInput')
  return addOp(g, 'param_input', shape, dtype, site, { name } as any)
}

/** Declare an external tensor input (tokens, targets, masks). Bound at
 *  each `step`/`run` call from the inputs record keyed by `name`. Shares
 *  its namespace with `paramInput`. */
export function tensorInput(name: string, shape: Shape, dtype: Dtype = 'f32'): Tensor {
  const g = currentGraph()
  assertNameUnused(g, name, ['param_input', 'tensor_input'], 'input')
  const site = captureSite('tensorInput')
  return addOp(g, 'tensor_input', shape, dtype, site, { name } as any)
}

/** Declare a persistent state buffer (e.g. Adam's m/v). Allocated at compile
 *  time and filled with `initValue` (default 0). Survives across `step()`
 *  calls; updated via writebacks declared by the optimizer pass. Has its
 *  own namespace, distinct from param/tensor inputs. */
export function stateInput(name: string, shape: Shape, dtype: Dtype = 'f32', initValue = 0): Tensor {
  const g = currentGraph()
  assertNameUnused(g, name, ['state_input'], 'state')
  const site = captureSite('stateInput')
  return addOp(g, 'state_input', shape, dtype, site, { name, initValue } as any)
}
