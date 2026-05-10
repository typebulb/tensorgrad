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

/** How a parameter's initial values are produced. Serializable shape — no
 *  closures, since the initial values cross the worker boundary at compile
 *  time. Use the `init` helpers for ergonomic construction.
 *
 *  String shorthands:
 *  - `'randn'` — Gaussian with std 0.02 (the common weight-matrix init).
 *  - `'zeros'` — fill with 0 (biases, LayerNorm beta).
 *  - `'ones'`  — fill with 1 (LayerNorm gain).
 *
 *  Object shapes:
 *  - `{ kind: 'randn', scale }` — randn with explicit std.
 *  - `{ kind: 'kaiming', gain? }` — `std = gain / sqrt(fan_in)`. Default
 *    gain `sqrt(2)` (good for ReLU). `fan_in = shape[0]`.
 *  - `{ kind: 'literal', data }` — explicit Float32Array; length must
 *    match the parameter's element count.
 */
export type InitSpec =
  | 'randn'
  | 'zeros'
  | 'ones'
  | { readonly kind: 'randn'; readonly scale: number }
  | { readonly kind: 'kaiming'; readonly gain?: number }
  | { readonly kind: 'literal'; readonly data: Float32Array }

/** Ergonomic constructors for InitSpec object shapes. */
export const init = {
  randn: (opts: { scale?: number } = {}): InitSpec => ({ kind: 'randn', scale: opts.scale ?? 0.02 }),
  kaiming: (opts: { gain?: number } = {}): InitSpec =>
    opts.gain !== undefined ? { kind: 'kaiming', gain: opts.gain } : { kind: 'kaiming' },
  literal: (data: Float32Array): InitSpec => ({ kind: 'literal', data }),
}

export interface ParamOptions {
  dtype?: Dtype
  /** Init shape. Default: `'randn'` (std 0.02). */
  init?: InitSpec
  /** Whether AdamW (when `weightDecay > 0`) should apply decoupled weight
   *  decay to this param. Default: `true` for randn/kaiming/literal init
   *  (weight matrices, embeddings); `false` for zeros/ones (biases, LN
   *  gains). Override to force or skip. Replaces `adam.decayFilter` for
   *  the common case. */
  decay?: boolean
}

type InitFn = (size: number, shape: readonly number[]) => Float32Array

function boxMuller(): number {
  return Math.sqrt(-2 * Math.log(Math.max(1e-10, Math.random()))) * Math.cos(2 * Math.PI * Math.random())
}

function randnFn(scale: number): InitFn {
  return (size) => {
    const arr = new Float32Array(size)
    for (let i = 0; i < size; i++) arr[i] = boxMuller() * scale
    return arr
  }
}

/** Compile-time-only: resolve an InitSpec shape into the closure that
 *  generates the initial Float32Array for a given parameter shape. Runs
 *  on the main thread before initial values are transferred to the worker. */
function resolveInit(spec: InitSpec | undefined): InitFn {
  if (!spec || spec === 'randn') return randnFn(0.02)
  if (spec === 'zeros') return (size) => new Float32Array(size)
  if (spec === 'ones') return (size) => { const a = new Float32Array(size); a.fill(1); return a }
  switch (spec.kind) {
    case 'randn': return randnFn(spec.scale)
    case 'kaiming': {
      const gain = spec.gain ?? Math.sqrt(2)
      return (size, shape) => {
        const fanIn = shape[0] ?? size
        const std = gain / Math.sqrt(fanIn)
        const arr = new Float32Array(size)
        for (let i = 0; i < size; i++) arr[i] = boxMuller() * std
        return arr
      }
    }
    case 'literal': {
      const data = spec.data
      return (size) => {
        if (data.length !== size) {
          throw new Error(`init.literal: data length ${data.length} doesn't match param size ${size}`)
        }
        return new Float32Array(data)
      }
    }
  }
}

/** Resolve the decay default for a param. Weight-shaped inits (randn,
 *  kaiming, literal) default to decay=true; ones/zeros default to false
 *  (biases, LN gains). Explicit `decay` opt overrides. */
function resolveDecay(opts: ParamOptions | undefined): boolean {
  if (opts?.decay !== undefined) return opts.decay
  const spec = opts?.init ?? 'randn'
  return spec !== 'zeros' && spec !== 'ones'
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
    return new ParamSentinel(shape, dtype, resolveInit(opts?.init), resolveDecay(opts)) as unknown as Tensor
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
