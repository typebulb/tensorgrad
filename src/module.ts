// Module abstraction — a Domeleon-style component layer for parameter trees.
//
// User code defines a model as nested classes:
//
//   class Linear extends Module {
//     W: Tensor; b: Tensor
//     constructor(inDim: number, outDim: number) {
//       super()
//       this.W = this.param([inDim, outDim])               // randn, scale 0.02
//       this.b = this.param([outDim], { init: 'zeros' })
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
// Init metadata
// ============================================================================

/** How a parameter's initial values are produced.
 *  - `'randn'` — Gaussian, with `scale` (default 0.02). The common case for
 *    weight matrices and embeddings.
 *  - `'zeros'` — fill with 0. Common for biases and LayerNorm beta.
 *  - `'ones'`  — fill with 1. Common for LayerNorm gain.
 *  - Custom function — receives total element count and shape, returns the
 *    Float32Array. Use for fan-in scaling or any non-standard scheme.
 */
export type InitSpec =
  | 'randn'
  | 'zeros'
  | 'ones'
  | ((size: number, shape: readonly number[]) => Float32Array)

export interface ParamOptions {
  dtype?: Dtype
  /** Init kind. Default: `'randn'`. */
  init?: InitSpec
  /** Std dev for `'randn'`. Default 0.02. Ignored for non-randn init. */
  scale?: number
  /** Whether AdamW (when `weightDecay > 0`) should apply decoupled weight
   *  decay to this param. Default: `true` for `'randn'` init (weight matrices,
   *  embeddings), `false` for `'zeros'` / `'ones'` (biases, LN gains). Override
   *  to force or skip. Replaces `adam.decayFilter` for the common case. */
  decay?: boolean
}

type InitFn = (size: number, shape: readonly number[]) => Float32Array

function boxMuller(): number {
  return Math.sqrt(-2 * Math.log(Math.max(1e-10, Math.random()))) * Math.cos(2 * Math.PI * Math.random())
}

function resolveInit(opts: ParamOptions | undefined): InitFn {
  const init = opts?.init ?? 'randn'
  if (init === 'randn') {
    const scale = opts?.scale ?? 0.02
    return (size) => {
      const arr = new Float32Array(size)
      for (let i = 0; i < size; i++) arr[i] = boxMuller() * scale
      return arr
    }
  }
  if (init === 'zeros') return (size) => new Float32Array(size)
  if (init === 'ones') return (size) => { const a = new Float32Array(size); a.fill(1); return a }
  if (typeof init === 'function') return init
  throw new Error(`Unknown init: ${String(init)}`)
}

/** Resolve the decay default for a param. Decay weight matrices and
 *  embedding tables (randn-initialized); skip biases (zeros) and LN gains
 *  (ones). Custom init functions default to "decay" — most user-supplied
 *  inits are weight-shaped (Kaiming etc.). Explicit `decay: false` overrides. */
function resolveDecay(opts: ParamOptions | undefined): boolean {
  if (opts?.decay !== undefined) return opts.decay
  const init = opts?.init ?? 'randn'
  if (init === 'zeros' || init === 'ones') return false
  return true   // 'randn' or function
}

// ============================================================================
// Internals: param sentinel
// ============================================================================
//
// `this.param(shape)` returns a placeholder that's replaced by a real Tensor
// during `materializeParams`. We type-cheat by declaring the return type as
// `Tensor` so user code can write `this.W` and have TS happy; the cheat is
// only valid post-materialization (which is always before forward runs).

class ParamSentinel {
  constructor(
    public readonly shape: Shape,
    public readonly dtype: Dtype,
    public readonly initFn: InitFn,
    public readonly decay: boolean,
  ) {}
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
   * tree (e.g. `layers.0.attn.W_q`). Init metadata travels with the param;
   * call `compiled.uploadInitialParams()` to apply it after compile.
   */
  protected param(shape: Shape, opts?: ParamOptions): Tensor {
    const dtype = opts?.dtype ?? 'f32'
    // Lie to TypeScript: the sentinel becomes a Tensor at materialize time.
    return new ParamSentinel(shape, dtype, resolveInit(opts), resolveDecay(opts)) as unknown as Tensor
  }
}

// ============================================================================
// Tree walking
// ============================================================================

export interface MaterializedParams {
  /** Map from auto-derived path (e.g. `layers.0.attn.W_q`) to its Tensor. */
  tensors: Record<string, Tensor>
  /** Init function per param path. Used by `uploadInitialParams`. */
  initFns: Record<string, InitFn>
  /** Whether this param should receive AdamW weight decay. Resolved at
   *  `param()` time from `ParamOptions.decay` (with init-based default). */
  decayFlags: Record<string, boolean>
}

/**
 * Walk the module tree and replace every ParamSentinel with a real Tensor
 * created via `paramInput(autoName, ...)`. Must be called inside an active
 * trace context (paramInput appends to the current graph).
 *
 * Returns the param tensors keyed by path, plus init functions for use by
 * `uploadInitialParams`.
 */
export function materializeParams(root: Module): MaterializedParams {
  const tensors: Record<string, Tensor> = {}
  const initFns: Record<string, InitFn> = {}
  const decayFlags: Record<string, boolean> = {}
  visit(root, '', (path, val, owner, key) => {
    if (val instanceof ParamSentinel) {
      const t = paramInput(path, val.shape, val.dtype)
      ;(owner as any)[key] = t
      tensors[path] = t
      initFns[path] = val.initFn
      decayFlags[path] = val.decay
    }
  })
  return { tensors, initFns, decayFlags }
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
