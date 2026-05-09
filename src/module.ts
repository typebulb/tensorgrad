// Module abstraction — a Domeleon-style component layer for parameter trees.
//
// User code defines a model as nested classes:
//
//   class Linear extends Module {
//     W: Tensor; b: Tensor
//     constructor(inDim: number, outDim: number) {
//       super()
//       this.W = this.param([inDim, outDim])
//       this.b = this.param([outDim])
//     }
//   }
//   class Block extends Module {
//     attn = new Attention(D)
//     mlp  = new MLP(D, 4 * D)
//   }
//   class Model extends Module {
//     embed = new Linear(VOCAB, D)
//     layers = range(N).map(() => new Block())
//   }
//
// The param tree is discovered automatically at compile time by walking
// enumerable instance properties. Each parameter gets a name auto-derived
// from its path (`layers.0.attn.W_q`); names are used for upload/download
// and writeback wiring. Forward functions are pure and stateless — they
// take the materialized model and inputs, return a Tensor.

import type { Tensor, Shape, Dtype } from './ir.js'
import { paramInput } from './trace.js'

// ============================================================================
// Internals: param sentinel
// ============================================================================
//
// `this.param(shape)` returns a placeholder that's replaced by a real Tensor
// during `materializeParams`. We type-cheat by declaring the return type as
// `Tensor` so user code can write `this.W` and have TS happy; the cheat is
// only valid post-materialization (which is always before forward runs).

class ParamSentinel {
  constructor(public readonly shape: Shape, public readonly dtype: Dtype) {}
}

// ============================================================================
// Module base class
// ============================================================================

export abstract class Module {
  /**
   * Declare a learnable parameter at this module. Must be called from inside
   * the constructor (typically as a field assignment). Returns a placeholder
   * that gets replaced with a real Tensor at compile time.
   *
   * The parameter's name is auto-derived from its property path in the model
   * tree (e.g. `layers.0.attn.W_q`).
   */
  protected param(shape: Shape, dtype: Dtype = 'f32'): Tensor {
    // Lie to TypeScript: the sentinel becomes a Tensor at materialize time.
    return new ParamSentinel(shape, dtype) as unknown as Tensor
  }
}

// ============================================================================
// Tree walking
// ============================================================================

/**
 * Walk the module tree and replace every ParamSentinel with a real Tensor
 * created via `paramInput(autoName, ...)`. Must be called inside an active
 * trace context (paramInput appends to the current graph).
 *
 * Returns a flat record of `{ path: tensor }` for every materialized param.
 */
export function materializeParams(root: Module): Record<string, Tensor> {
  const out: Record<string, Tensor> = {}
  visit(root, '', (path, val, owner, key) => {
    if (val instanceof ParamSentinel) {
      const t = paramInput(path, val.shape, val.dtype)
      ;(owner as any)[key] = t
      out[path] = t
    }
  })
  return out
}

// ----------------------------------------------------------------------------
// Visitor
// ----------------------------------------------------------------------------
//
// Walks enumerable own properties recursively, building a path string. Recurses
// into nested Modules and arrays of Modules (or arrays of arrays, etc.).
// Calls `visitor` on every leaf — including ParamSentinels (pre-materialize)
// and real Tensor leaves (post-materialize).

type Visitor = (path: string, val: unknown, owner: object, key: string | number) => void

function visit(node: unknown, path: string, visitor: Visitor): void {
  if (node === null || node === undefined) return
  if (typeof node !== 'object') return

  if (node instanceof Module) {
    for (const key of Object.keys(node as object)) {
      const child = (node as any)[key]
      const childPath = path ? `${path}.${key}` : key
      visitChild(child, childPath, node, key, visitor)
    }
    return
  }
  if (Array.isArray(node)) {
    node.forEach((item, i) => {
      const childPath = path ? `${path}.${i}` : String(i)
      visitChild(item, childPath, node as unknown as object, i, visitor)
    })
    return
  }
  // Plain leaf object (sentinel / tensor / something else): visitor decides.
  // No deeper recursion.
}

function visitChild(child: unknown, path: string, owner: object, key: string | number, visitor: Visitor): void {
  if (child instanceof Module || Array.isArray(child)) {
    visit(child, path, visitor)
  } else {
    visitor(path, child, owner, key)
  }
}
