// Trace driver. Holds the "current graph" in module-local state so user code
// can call ops without threading a graph parameter through every function.
//
// Usage:
//
//   const graph = trace(() => {
//     const x = tensorInput('x', [B, T], 'i32')
//     const w = paramInput('w', [V, D], 'f32')
//     // ... user computation building tensors ...
//     return finalLossTensor
//   })
//
// `trace` is single-threaded and re-entrant only via nested calls (which share
// the outer graph — but we don't currently have a use for nesting). Calling an
// op outside a `trace(...)` block is an error.

import type { Graph, Tensor, Shape, Dtype } from './ir.js'
import { makeGraph, addOp, captureSite } from './ir.js'

// Module-local: the graph being built right now, or null if no trace is active.
let _current: Graph | null = null

export function currentGraph(): Graph {
  if (!_current) {
    throw new Error(
      'tensorgrad: ops can only be called inside trace(). ' +
      'Did you forget to wrap your forward pass?',
    )
  }
  return _current
}

// Run `fn` with a fresh graph as the current one; capture and return the graph.
// `fn` must return the tensor (or array of tensors) to mark as graph outputs.
export function trace(fn: () => Tensor | Tensor[]): Graph {
  if (_current) {
    throw new Error('tensorgrad: nested trace() is not supported')
  }
  const g = makeGraph()
  _current = g
  try {
    const result = fn()
    const outputs = Array.isArray(result) ? result : [result]
    for (const t of outputs) {
      ;(g.outputs as number[]).push(t.id)
    }
  } finally {
    _current = null
  }
  return g
}

// Re-enter an existing graph to append more ops. Used by autograd to add
// backward ops to a graph that's already been traced. `fn` runs with the
// supplied graph as the current one; any ops it calls append to that graph.
// Returns whatever `fn` returns.
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

// ---- Leaf tensor builders --------------------------------------------------
// Inputs are added to the graph as `param_input` or `tensor_input` op nodes.
// Their .source on the Tensor points at that node so codegen knows where to
// bind external data.

export function paramInput(name: string, shape: Shape, dtype: Dtype = 'f32'): Tensor {
  const g = currentGraph()
  if (g.ops.some(op => (op.kind === 'param_input' || op.kind === 'tensor_input') && op.name === name)) {
    throw new Error(`tensorgrad: input name '${name}' already used in this trace`)
  }
  const site = captureSite('paramInput')
  return addOp(g, 'param_input', shape, dtype, site, { name } as any)
}

export function tensorInput(name: string, shape: Shape, dtype: Dtype = 'f32'): Tensor {
  const g = currentGraph()
  if (g.ops.some(op => (op.kind === 'param_input' || op.kind === 'tensor_input') && op.name === name)) {
    throw new Error(`tensorgrad: input name '${name}' already used in this trace`)
  }
  const site = captureSite('tensorInput')
  return addOp(g, 'tensor_input', shape, dtype, site, { name } as any)
}

// Persistent state buffer. Allocated at compile time, zero-(or initValue-)initialized,
// and updated across step() calls via writebacks declared by the optimizer helper.
export function stateInput(name: string, shape: Shape, dtype: Dtype = 'f32', initValue = 0): Tensor {
  const g = currentGraph()
  if (g.ops.some(op => op.kind === 'state_input' && op.name === name)) {
    throw new Error(`tensorgrad: state name '${name}' already used in this trace`)
  }
  const site = captureSite('stateInput')
  return addOp(g, 'state_input', shape, dtype, site, { name, initValue } as any)
}
