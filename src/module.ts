// Module abstraction — Domeleon-style component layer for parameter trees.
// Param tree is auto-discovered at compile time by walking enumerable instance
// properties. Forward functions are pure and stateless (take the materialized
// model + inputs, return a Tensor) — see `Module` JSDoc for the pattern.

import type { Tensor, Shape, Dtype } from './ir.js'
import { paramInput } from './trace.js'

/** How a parameter's initial values are produced. Serializable shape — no
 *  closures, since the initial values cross the worker boundary at compile
 *  time. Construct via the `init` helpers.
 *
 *  Kinds:
 *  - `init.randn({ scale? })` — Gaussian with given std (default 0.02).
 *  - `init.zeros()` — fill with 0 (biases, LayerNorm beta).
 *  - `init.ones()` — fill with 1 (LayerNorm gain).
 *  - `init.kaiming({ gain? })` — `std = gain / sqrt(fan_in)`. Default gain
 *    `sqrt(2)` (good for ReLU); `fan_in = shape[0]`.
 *  - `init.literal(data)` — explicit Float32Array; length must match the
 *    parameter's element count.
 */
export type InitSpec =
  | { readonly kind: 'randn'; readonly scale: number }
  | { readonly kind: 'zeros' }
  | { readonly kind: 'ones' }
  | { readonly kind: 'kaiming'; readonly gain?: number }
  | { readonly kind: 'literal'; readonly data: Float32Array }

/** Ergonomic constructors for InitSpec. */
export const init = {
  /** Gaussian draw with std `scale` (default 0.02). The library's default
   *  param init; PyTorch ports that don't specify init match this. */
  randn: (opts: { scale?: number } = {}): InitSpec => ({ kind: 'randn', scale: opts.scale ?? 0.02 }),
  /** Fill with 0. Standard for biases and LayerNorm β. AdamW weight decay
   *  defaults to off for zeros-init params. */
  zeros: (): InitSpec => ({ kind: 'zeros' }),
  /** Fill with 1. Standard for LayerNorm / RMSNorm gain. AdamW weight decay
   *  defaults to off for ones-init params. */
  ones: (): InitSpec => ({ kind: 'ones' }),
  /** Kaiming-normal: `std = gain / sqrt(fan_in)` where `fan_in = shape[0]`.
   *  Default `gain = sqrt(2)` (the He init for ReLU). PyTorch:
   *  `nn.init.kaiming_normal_(..., mode='fan_in')`. */
  kaiming: (opts: { gain?: number } = {}): InitSpec =>
    opts.gain !== undefined ? { kind: 'kaiming', gain: opts.gain } : { kind: 'kaiming' },
  /** Explicit data. `data.length` must match the parameter's total element
   *  count. Use for loading pretrained weights or seeding test fixtures. */
  literal: (data: Float32Array): InitSpec => ({ kind: 'literal', data }),
}

/** Per-parameter options accepted by `Module.param(shape, opts)`. All
 *  fields are optional; sensible defaults apply (`f32` dtype, `init.randn()`,
 *  decay-true for weight-shaped inits). */
export interface ParamOptions {
  /** Element type. Default `'f32'`. */
  dtype?: Dtype
  /** Init spec. Default: `init.randn()` (std 0.02). */
  init?: InitSpec
  /** Whether AdamW (when `weightDecay > 0`) should apply decoupled weight
   *  decay to this param. Default: `true` for randn/kaiming/literal init
   *  (weight matrices, embeddings); `false` for zeros/ones (biases, LN
   *  gains). Override to force or skip. Replaces `adam.decayFilter` for
   *  the common case. */
  decay?: boolean
}

/** RNG closure: returns a uniform value in [0, 1). Threaded through init
 *  functions so the same `seed` produces identical params across runs. */
export type Rng = () => number

/** Mulberry32 — small, fast, sufficient for param init. Returns an Rng
 *  seeded deterministically from the given 32-bit integer. */
export function mulberry32(seed: number): Rng {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6D2B79F5) | 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** A resolved initializer: given a flat element count, the original shape,
 *  and an `Rng`, produce the param's initial `Float32Array`. Built by
 *  `resolveInit` from an `InitSpec`. */
export type InitFn = (size: number, shape: readonly number[], rng: Rng) => Float32Array

function boxMuller(rng: Rng): number {
  return Math.sqrt(-2 * Math.log(Math.max(1e-10, rng()))) * Math.cos(2 * Math.PI * rng())
}

function randnFn(scale: number): InitFn {
  return (size, _shape, rng) => {
    const arr = new Float32Array(size)
    for (let i = 0; i < size; i++) arr[i] = boxMuller(rng) * scale
    return arr
  }
}

/** Compile-time-only: resolve an InitSpec shape into the closure that
 *  generates the initial Float32Array for a given parameter shape. Runs
 *  on the main thread before initial values are transferred to the worker. */
function resolveInit(spec: InitSpec | undefined): InitFn {
  if (!spec) return randnFn(0.02)
  switch (spec.kind) {
    case 'randn': return randnFn(spec.scale)
    case 'zeros': return (size) => new Float32Array(size)
    case 'ones': return (size) => { const a = new Float32Array(size); a.fill(1); return a }
    case 'kaiming': {
      const gain = spec.gain ?? Math.sqrt(2)
      return (size, shape, rng) => {
        const fanIn = shape[0] ?? size
        const std = gain / Math.sqrt(fanIn)
        const arr = new Float32Array(size)
        for (let i = 0; i < size; i++) arr[i] = boxMuller(rng) * std
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
  const kind = opts?.init?.kind
  return kind !== 'zeros' && kind !== 'ones'
}

// Immutable placeholder produced by `this.param(...)`. Replaced by a real
// Tensor in `materializeParams`, which always runs on a *clone* of the
// user's Module tree (never the original instance). The `param` return
// type is cast to `Tensor` so user code can read `this.W` directly; the
// cast is sound post-materialization, which always happens before any
// forward call.
class Param {
  constructor(
    public readonly shape: Shape,
    public readonly dtype: Dtype,
    public readonly initFn: InitFn,
    public readonly decay: boolean,
  ) {
    Object.freeze(this)
  }
}

/**
 * Base class for model components. Subclass to declare a tree of parameters
 * and child modules:
 *
 * ```ts
 * class Linear extends Module {
 *   W: Tensor; b: Tensor
 *   constructor(inDim: number, outDim: number) {
 *     super()
 *     this.W = this.param([inDim, outDim])
 *     this.b = this.param([outDim], { init: init.zeros() })
 *   }
 * }
 * ```
 *
 * Parameters are declared via `this.param(shape, opts?)` from inside the
 * constructor. Nested modules are plain instance fields (`this.l1 = new
 * Linear(...)`); arrays of modules (`this.layers = [...]`) are walked too.
 * Parameter names are auto-derived from the property path
 * (`layers.0.attn.W_q`).
 *
 * Forward functions are *free functions*, not methods — they take the
 * materialized module plus inputs and return a `Tensor`. The built-in
 * leaf modules (`Linear`, `LayerNorm`, etc.) expose `.fwd(x)` as a
 * convenience for chaining, but composite modules you write should follow
 * the free-function pattern.
 */
export abstract class Module {
  /**
   * Declare a learnable parameter at this module. Must be called from inside
   * the constructor (typically as a field assignment). Returns a placeholder
   * that gets replaced with a real Tensor at compile time.
   *
   * The parameter's name is auto-derived from its property path in the model
   * tree (e.g. `layers.0.attn.W_q`). Init metadata travels with the param;
   * `compile()` applies it during compile, and `compiled.reset()`
   * re-applies it later.
   */
  protected param(shape: Shape, opts?: ParamOptions): Tensor {
    const dtype = opts?.dtype ?? 'f32'
    // Lie to TypeScript: the Param placeholder becomes a Tensor at
    // materialize time (on a fresh clone of this module).
    return new Param(shape, dtype, resolveInit(opts?.init), resolveDecay(opts)) as unknown as Tensor
  }
}

export interface MaterializedParams {
  /** Map from auto-derived path (e.g. `layers.0.attn.W_q`) to its Tensor. */
  tensors: Record<string, Tensor>
  /** Init function per param path. Used by the compile pipeline to generate
   *  the initial Float32Array transferred to the worker. */
  initFns: Record<string, InitFn>
  /** Whether this param should receive AdamW weight decay. Resolved at
   *  `param()` time from `ParamOptions.decay` (with init-based default). */
  decayFlags: Record<string, boolean>
}

/**
 * Walk the module tree and replace every Param placeholder with a real
 * Tensor created via `paramInput(autoName, ...)`. Must be called inside
 * an active trace context (paramInput appends to the current graph) AND
 * on a fresh clone of the user's module (see `cloneModule`) — the
 * mutation never touches the caller's instance.
 *
 * Returns the param tensors keyed by path, plus init functions the compile
 * pipeline uses to generate initial values.
 */
export function materializeParams(root: Module): MaterializedParams {
  const tensors: Record<string, Tensor> = {}
  const initFns: Record<string, InitFn> = {}
  const decayFlags: Record<string, boolean> = {}
  visit(root, '', (path, val, owner, key) => {
    if (val instanceof Param) {
      const t = paramInput(path, val.shape, val.dtype)
      ;(owner as any)[key] = t
      tensors[path] = t
      initFns[path] = val.initFn
      decayFlags[path] = val.decay
    }
  })
  return { tensors, initFns, decayFlags }
}

/**
 * Deep clone a Module tree, preserving each instance's prototype (so
 * `cloneModule(new MLP())` is still an MLP) and copying through `Param`
 * placeholders unchanged. Arrays of Modules are mapped to fresh arrays.
 * Non-Module fields (numbers, config records, etc.) are shared by
 * reference — they're not mutated by the trace pipeline.
 *
 * The compile pipeline calls this before `materializeParams` so the
 * user's original Module instance is never mutated. Each polymorphic
 * sibling cache-miss clones fresh too.
 */
export function cloneModule<M extends Module>(m: M): M {
  return cloneNode(m) as M
}

function cloneNode(node: unknown): unknown {
  if (node instanceof Module) {
    const out = Object.create(Object.getPrototypeOf(node))
    for (const key of Object.keys(node as object)) {
      ;(out as any)[key] = cloneNode((node as any)[key])
    }
    return out
  }
  if (Array.isArray(node)) {
    return node.map(cloneNode)
  }
  // Params, primitives, plain objects: share by reference. Params are
  // frozen; non-Module objects shouldn't carry mutable state the trace
  // would touch.
  return node
}

// Walks enumerable own properties, recursing into Modules and arrays.
// `visitor` is called on each leaf (Param pre-materialize, Tensor
// post-materialize, or anything else the user stored).
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
}

function visitChild(child: unknown, path: string, owner: object, key: string | number, visitor: Visitor): void {
  if (child instanceof Module || Array.isArray(child)) {
    visit(child, path, visitor)
  } else {
    visitor(path, child, owner, key)
  }
}
