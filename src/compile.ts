// Top-level compile pipeline.
//
//   compileModule({ factory, loss, inputs, adam })
//     → trace, autograd, Adam, buffer plan, codegen, worker spawn → CompiledModule
//   compiled.compileForward({ forward, inputs })
//     → sibling forward graph sharing parent's param GPUBuffers, polymorphic
//       on `null` wildcards
//   compiled.replaceModel(newFactory)
//     → swap topology in place; same handle, same worker. Sibling forward
//       proxies stay registered — their per-shape kernel caches are cleared
//       and recompiled lazily on the next run().
//
// Compile-time work runs on the main thread; everything past createRuntime
// runs in a worker (see specs/WorkerArchitecture.md, runtime.ts, worker.ts).
//
// Compiled-forward proxies are always polymorphic-capable: shapes you declare
// can include `null` to mark a dim parametric (inferred at run time from the
// actual TypedArray length). Concrete shapes are just the cache-size-1 case;
// no separate eager/lazy code paths. See specs/SimplifyV01.md.

import type { Tensor, Shape, Dtype } from './ir.js'
import { trace, tensorInput } from './trace.js'
import { appendGrad, type GradResult } from './grad.js'
import {
  appendAdam, resolveLR,
  type AdamConfig, type AdamResult, type AdamResolvedConfig, type LRSchedule,
} from './adam.js'
import { planBuffers, type BufferPlan } from './buffers.js'
import { emitKernels, type KernelSpec } from './codegen.js'
import {
  Captures, type RunResult, type StepResult, type RunOptions, type UploadParamsOptions,
} from './runtime.js'
import { Module, materializeParams, type MaterializedParams } from './module.js'
import { WorkerProxy } from './worker-proxy.js'
import {
  transferablesOfRecord,
  type Req, type WireIR, type WireAdamConfig,
  type CreateRuntimeResult, type CompileForwardResult,
  type StepResultWire, type RunResultWire, type DownloadParamsResult,
} from './worker-protocol.js'

declare const __WORKER_SOURCE__: string

// ============================================================================
// Public types
// ============================================================================

/** Shape of a declared input. Each dim is a fixed number or `null` to mark
 *  the dim parametric (resolved from actual TypedArray length at run time).
 *  At most one `null` per shape. Matches the TF/ONNX/MLIR wildcard convention. */
export type InputShape = readonly (number | null)[]

/** Object form of an input declaration. Only needed when dtype isn't `f32`
 *  (the default), or for stylistic explicitness. */
export interface InputDeclObject {
  readonly shape: InputShape
  readonly dtype?: Dtype
}

/** An input declaration value: shape tuple (dtype defaults to `f32`) or
 *  `{ shape, dtype }` for non-f32 (`i32` / `bool`). */
export type InputDecl = InputShape | InputDeclObject

/** Inputs declaration: name -> shape (tuple or object). */
export type InputDecls = Record<string, InputDecl>

/** Maps an `InputDecls` Record to its forward-time tensor counterpart —
 *  same keys, each value is a Tensor. */
export type InputsTensors<I extends InputDecls> = { [K in keyof I]: Tensor }

/** Extract the declared dtype from an `InputDecl`. Tuple shapes and object
 *  forms without an explicit `dtype` default to `'f32'`; the object form's
 *  `dtype` field is preserved as a literal when present. */
export type DtypeOf<D extends InputDecl> =
  D extends readonly (number | null)[] ? 'f32'
  : D extends { dtype: infer T extends Dtype } ? T
  : 'f32'

/** The TypedArray type the runtime expects for a given dtype. `bool` isn't
 *  user-facing as an input (it's an internal IR dtype produced by
 *  comparisons), so it has no input ArrayType. */
export type TypedArrayFor<D extends Dtype> =
  D extends 'f32' ? Float32Array
  : D extends 'i32' ? Int32Array
  : never

/** Inputs record typed against an `InputDecls` shape: each declared input's
 *  TypedArray type matches its declared dtype. Used by `step`/`run` on the
 *  compiled-module proxies so dtype mismatches fail at compile time, not
 *  at marshal time. */
export type TypedInputs<I extends InputDecls> = { [K in keyof I]: TypedArrayFor<DtypeOf<I[K]>> }

/** Forward function shape. */
export type ForwardFn<M extends Module, I extends InputDecls = InputDecls> =
  (m: M, inputs: InputsTensors<I>) => Tensor

export interface CompiledIR {
  graph: GradResult['graph']
  paramGrads: GradResult['paramGrads']
  loss: Tensor
  plan: BufferPlan
  kernels: KernelSpec[]
}

/** Trace + autograd + buffer-plan + codegen, without touching WebGPU. */
export function compileToIR(traceFn: () => Tensor): CompiledIR {
  const graph = trace(traceFn)
  const { paramGrads, loss } = appendGrad(graph)
  const plan = planBuffers(graph, paramGrads)
  const kernels = emitKernels(graph, plan)
  return { graph, paramGrads, loss, plan, kernels }
}

// ============================================================================
// compileModule entry — options + result interfaces
// ============================================================================

export interface CompileModuleOptions<M extends Module, I extends InputDecls = InputDecls> {
  /** Model factory `() => new Model()`. Invoked once per compile (and once
   *  per cache-miss in any polymorphic sibling forward). The model instance
   *  is consumed: its `ParamSentinel` fields are mutated into Tensors. */
  factory: () => M
  /** Forward function that returns the scalar loss tensor. */
  loss: ForwardFn<M, I>
  /** Input shape declarations (one per named tensor input). */
  inputs: I
  /** Optional Adam config. When absent, the module compiles but has no
   *  optimizer — `step()` will fail. (Used internally for `compileToIR`-like
   *  flows where the user only wants the forward pass.) */
  adam?: AdamConfig
}

export interface CompileForwardMethodOptions<M extends Module, I extends InputDecls = InputDecls> {
  /** Forward function returning the output tensor (one per shape value). */
  forward: ForwardFn<M, I>
  /** Input shape declarations. `null` dims become parametric; the proxy
   *  compiles + caches a sibling per distinct resolved shape on first
   *  `run()` at that shape. */
  inputs: I
}

/** Optional fields to mutate on a CompiledModule's Adam state via
 *  `setOptimizerConfig`. Any subset is allowed; absent fields keep their
 *  current values. */
export interface OptimizerConfigUpdate {
  lr?: LRSchedule
  weightDecay?: number
  b1?: number
  b2?: number
}

/** Returned by `compileModule`. Proxies all GPU work to an internal worker;
 *  every method returns a Promise. Generic over the declared inputs shape
 *  `I` so `step` / `run` accept inputs with the right TypedArray per dtype. */
export interface CompiledModule<M extends Module, I extends InputDecls = InputDecls> {
  readonly ir: CompiledIR
  readonly kernelCount: number
  readonly outputShape: readonly number[]
  /** Names of the model's parameters, in materialization order. */
  readonly paramNames: readonly string[]

  step(inputs: TypedInputs<I>): Promise<number>
  step(inputs: TypedInputs<I>, opts: { withCaptures: true }): Promise<StepResult>

  run(inputs: TypedInputs<I>): Promise<Float32Array>
  run(inputs: TypedInputs<I>, opts: { withCaptures: true }): Promise<RunResult>

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void>
  downloadParams(): Promise<Record<string, Float32Array>>
  downloadParamGrads(): Promise<Record<string, Float32Array>>

  /** Re-initialize all params + zero optimizer state. */
  reset(): Promise<void>
  resetOptimizerState(): Promise<void>

  /** Update one or more Adam hyperparameters at runtime, without recompiling.
   *
   *  When `update.lr` is a non-constant schedule and the schedule has no
   *  explicit `startStep`, the schedule is rebased so its step 1 aligns with
   *  the next training step ("decay from now"). Numbers and `constant`
   *  schedules pass through unchanged. */
  setOptimizerConfig(update: OptimizerConfigUpdate): Promise<void>

  /** Compile a sibling forward-only graph that shares this runtime's worker
   *  (and therefore its param GPUBuffers). Polymorphic by default — `null`
   *  dims in the input shapes resolve per-call. */
  compileForward<I2 extends InputDecls>(
    opts: CompileForwardMethodOptions<M, I2>,
  ): Promise<CompiledForwardModule<I2>>

  /** Swap the model topology in place: destroy the current training graph in
   *  the worker, compile a fresh one with the same loss/inputs/adam config
   *  but the new factory. This handle remains valid (same object, same `I`
   *  generic, same `paramNames`/`kernelCount`/`ir` after the call). Sibling
   *  forward proxies created via `compileForward` stay registered: their
   *  per-shape kernel caches are cleared and recompile lazily on next `run()`.
   *
   *  Use when the user changes layer count, hidden width, or any other
   *  shape-affecting model parameter — you don't need to re-create the
   *  worker or re-wire siblings. */
  replaceModel(newFactory: () => M): Promise<void>

  /** Tear down the worker + GPU resources. */
  destroy(): void
}

/** Returned by `compiled.compileForward({...})`. Polymorphic by default:
 *  `run()` at a new resolved shape lazily compiles + caches a sibling. Param
 *  reads/writes route to the parent training graph (shared buffers).
 *
 *  No sync inspection surface for kernel count / output shape / IR — those
 *  would lie on a polymorphic proxy that caches multiple shape variants. */
export interface CompiledForwardModule<I extends InputDecls = InputDecls> {
  /** Same as the parent training graph's param names. */
  readonly paramNames: readonly string[]

  run(inputs: TypedInputs<I>): Promise<Float32Array>
  run(inputs: TypedInputs<I>, opts: { withCaptures: true }): Promise<RunResult>

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void>
  downloadParams(): Promise<Record<string, Float32Array>>

  destroy(): void
}

// ============================================================================
// compileModule
// ============================================================================

/**
 * Compile a Module-based model into a runtime that lives in an internal
 * Web Worker. `factory: () => new Model()` is invoked once per compile (and
 * once per cache-miss in any polymorphic sibling forward); the model
 * instance is consumed (its `ParamSentinel` fields are mutated into Tensors).
 *
 * ```ts
 * const compiled = await compileModule({
 *   factory: () => new MLP(),
 *   loss: (m, { x, y }) => mse(m(x), y),
 *   inputs: { x: [128, 784], y: [128] },
 *   adam: { lr: 0.001 },
 * })
 * ```
 */
export async function compileModule<M extends Module, I extends InputDecls>(
  opts: CompileModuleOptions<M, I>,
): Promise<CompiledModule<M, I>> {
  const proxy = new WorkerProxy(__WORKER_SOURCE__)
  try {
    const built = await buildTrainingGraph(proxy, opts, /* graphId */ 0)
    return new CompiledModuleProxy<M>(
      proxy, /* graphId */ 0, built.ir, built.meta,
      opts as unknown as CompileModuleOptions<M, InputDecls>,
      built.initFns,
      /* nextGraphId */ { v: 1 },
    ) as unknown as CompiledModule<M, I>
  } catch (e) {
    proxy.terminate()
    throw e
  }
}

interface BuiltTrainingGraph {
  ir: CompiledIR
  meta: CreateRuntimeResult
  initFns: Record<string, InitFn>
}

/** Trace + autograd + Adam + buffer-plan + codegen + worker createRuntime,
 *  using an existing worker proxy. Used both by `compileModule` (fresh worker)
 *  and by `replaceModel` (existing worker, same graphId). */
async function buildTrainingGraph<M extends Module, I extends InputDecls>(
  proxy: WorkerProxy,
  opts: CompileModuleOptions<M, I>,
  graphId: number,
): Promise<BuiltTrainingGraph> {
  const loss = opts.loss as unknown as ForwardFn<M, InputDecls>
  const inputs = opts.inputs as InputDecls
  const { graph, materialized } = traceModule(opts.factory(), loss, inputs)
  const { paramGrads, loss: lossTensor } = appendGrad(graph)
  const adamResult = opts.adam
    ? appendAdam(graph, paramGrads, materialized.tensors, opts.adam, materialized.decayFlags)
    : undefined

  const plan = planBuffers(graph, paramGrads, adamResult?.writebacks ?? [])
  const kernels = emitKernels(graph, plan)
  const ir: CompiledIR = { graph, paramGrads, loss: lossTensor, plan, kernels }

  const initialParams = buildInitialParams(plan, materialized.initFns)
  const wireIR: WireIR = { graph, plan, kernels }
  const wireAdam = adamResult ? wireAdamConfig(adamResult) : null
  const transfers = transferablesOfRecord(initialParams)

  const meta = await proxy.request<CreateRuntimeResult>(
    { kind: 'createRuntime', payload: { graphId, ir: wireIR, initialParams, adam: wireAdam } },
    transfers,
  )
  return { ir, meta, initFns: materialized.initFns }
}

// ============================================================================
// Proxy implementations
// ============================================================================

/** Internal-only loose input record. The public proxy interfaces type these
 *  by declared dtype via `TypedInputs<I>`; the implementation keeps the
 *  wider type since it doesn't know `I` at the class level. */
type LooseInputs = Record<string, Int32Array | Float32Array>

class CompiledModuleProxy<M extends Module> implements CompiledModule<M, InputDecls> {
  /** Forward proxies created via `compileForward` — tracked so `destroy()`
   *  can clean them up cascade-style, and `replaceModel()` can invalidate
   *  their per-shape kernel caches without unregistering them. */
  private readonly children = new Set<ForwardProxy<M>>()

  // Mutable: `replaceModel` swaps these in place so callers' references
  // (and any sibling ForwardProxy holding `this`) stay valid.
  ir: CompiledIR
  private meta: CreateRuntimeResult
  private opts: CompileModuleOptions<M, InputDecls>
  private initFns: Record<string, InitFn>

  constructor(
    private readonly proxy: WorkerProxy,
    readonly graphId: number,
    ir: CompiledIR,
    meta: CreateRuntimeResult,
    opts: CompileModuleOptions<M, InputDecls>,
    initFns: Record<string, InitFn>,
    private readonly nextGraphId: { v: number },
  ) {
    this.ir = ir
    this.meta = meta
    this.opts = opts
    this.initFns = initFns
  }

  get kernelCount(): number { return this.meta.kernelCount }
  get outputShape(): readonly number[] { return this.meta.outputShape }
  get paramNames(): readonly string[] { return this.meta.paramNames }

  /** Current model factory. Sibling ForwardProxies read this through the
   *  proxy ref so they always re-trace against the latest topology. */
  currentFactory(): M { return this.opts.factory() }

  step(inputs: LooseInputs): Promise<number>
  step(inputs: LooseInputs, opts: { withCaptures: true }): Promise<StepResult>
  async step(
    inputs: LooseInputs,
    opts?: { withCaptures?: boolean },
  ): Promise<number | StepResult> {
    const r = await this.proxy.request<StepResultWire>(
      { kind: 'step', payload: { graphId: this.graphId, inputs, withCaptures: opts?.withCaptures === true } },
    )
    if (opts?.withCaptures) return { loss: r.loss, captures: makeCaptures(r.captures, this.meta.captureShapes) }
    return r.loss
  }

  run(inputs: LooseInputs): Promise<Float32Array>
  run(inputs: LooseInputs, opts: { withCaptures: true }): Promise<RunResult>
  async run(
    inputs: LooseInputs,
    opts?: { withCaptures?: boolean },
  ): Promise<Float32Array | RunResult> {
    const r = await this.proxy.request<RunResultWire>(
      { kind: 'run', payload: { graphId: this.graphId, inputs, withCaptures: opts?.withCaptures === true } },
    )
    if (opts?.withCaptures) return { output: r.output, captures: makeCaptures(r.captures, this.meta.captureShapes) }
    return r.output
  }

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'uploadParams', payload: { graphId: this.graphId, params, partial: !!opts?.partial } },
    ).then(() => undefined)
  }

  async downloadParams(): Promise<Record<string, Float32Array>> {
    const r = await this.proxy.request<DownloadParamsResult>(
      { kind: 'downloadParams', payload: { graphId: this.graphId } },
    )
    return r.params
  }

  async downloadParamGrads(): Promise<Record<string, Float32Array>> {
    const r = await this.proxy.request<DownloadParamsResult>(
      { kind: 'downloadParamGrads', payload: { graphId: this.graphId } },
    )
    return r.params
  }

  async reset(): Promise<void> {
    const initialParams = buildInitialParams(this.ir.plan, this.initFns)
    await this.uploadParams(initialParams)
    await this.resetOptimizerState()
  }

  resetOptimizerState(): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'resetOptimizer', payload: { graphId: this.graphId } },
    ).then(() => undefined)
  }

  setOptimizerConfig(update: OptimizerConfigUpdate): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'setOptimizerConfig', payload: { graphId: this.graphId, update } },
    ).then(() => undefined)
  }

  async compileForward<I2 extends InputDecls>(
    opts: CompileForwardMethodOptions<M, I2>,
  ): Promise<CompiledForwardModule<I2>> {
    const child = new ForwardProxy<M>(
      this.proxy,
      this,
      opts.forward as ForwardFn<M, InputDecls>,
      normalizeDecls(opts.inputs as InputDecls),
      this.nextGraphId,
      () => this.children.delete(child),
    )
    this.children.add(child)
    return child as unknown as CompiledForwardModule<I2>
  }

  async replaceModel(newFactory: () => M): Promise<void> {
    // 1. Invalidate sibling caches: their per-shape kernels are model-specific
    //    so the worker-side graphs must go, but the proxy objects stay alive
    //    (callers hold references to them) and recompile lazily on next run().
    for (const child of this.children) child._invalidateForReplace()
    // 2. Destroy current training graph in worker.
    this.proxy.send({ kind: 'destroy', payload: { graphId: this.graphId } })
    // 3. Rebuild the training graph on the same worker, same graphId.
    const newOpts = { ...this.opts, factory: newFactory } as CompileModuleOptions<M, InputDecls>
    const built = await buildTrainingGraph<M, InputDecls>(this.proxy, newOpts, this.graphId)
    // 4. Mutate this proxy in place — same object, new internals.
    this.ir = built.ir
    this.meta = built.meta
    this.initFns = built.initFns
    this.opts = newOpts
  }

  destroy(): void {
    for (const child of this.children) child._destroyInternal()
    this.children.clear()
    this.proxy.send({ kind: 'destroy', payload: { graphId: this.graphId } })
    this.proxy.terminate()
  }
}

/** Single forward proxy class — handles both fully-concrete and polymorphic
 *  shapes (concrete is the cache-size-1 case). Sibling of a training graph;
 *  shares its param GPUBuffers. Holds a reference to the parent proxy so it
 *  picks up the current model factory after `replaceModel`. */
class ForwardProxy<M extends Module> implements CompiledForwardModule<InputDecls> {
  private readonly cache = new Map<string, ForwardSiblingMeta>()

  constructor(
    private readonly proxy: WorkerProxy,
    private readonly parent: CompiledModuleProxy<M>,
    private readonly forward: ForwardFn<M, InputDecls>,
    private readonly decls: NormalizedDecls,
    private readonly nextGraphId: { v: number },
    private readonly onDestroy: () => void,
  ) {}

  get paramNames(): readonly string[] { return this.parent.paramNames }

  private async siblingFor(inputs: Record<string, Int32Array | Float32Array>): Promise<ForwardSiblingMeta> {
    const resolved = resolveDecls(this.decls, inputs)
    const key = shapeKey(resolved)
    const hit = this.cache.get(key)
    if (hit) return hit
    const sib = await compileSibling<M>(
      this.proxy, this.parent.graphId, this.parent.currentFactory(), this.forward,
      resolved, this.nextGraphId,
    )
    this.cache.set(key, sib)
    return sib
  }

  run(inputs: LooseInputs): Promise<Float32Array>
  run(inputs: LooseInputs, opts: { withCaptures: true }): Promise<RunResult>
  async run(
    inputs: LooseInputs,
    opts?: { withCaptures?: boolean },
  ): Promise<Float32Array | RunResult> {
    const sib = await this.siblingFor(inputs)
    const r = await this.proxy.request<RunResultWire>(
      { kind: 'run', payload: { graphId: sib.graphId, inputs, withCaptures: opts?.withCaptures === true } },
    )
    if (opts?.withCaptures) return { output: r.output, captures: makeCaptures(r.captures, sib.meta.captureShapes) }
    return r.output
  }

  async uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void> {
    // Params live on the parent training graph (shared with all siblings).
    await this.proxy.request<null>(
      { kind: 'uploadParams', payload: { graphId: this.parent.graphId, params, partial: !!opts?.partial } },
    )
  }

  async downloadParams(): Promise<Record<string, Float32Array>> {
    const r = await this.proxy.request<DownloadParamsResult>(
      { kind: 'downloadParams', payload: { graphId: this.parent.graphId } },
    )
    return r.params
  }

  destroy(): void {
    this._destroyInternal()
    this.onDestroy()
  }

  /** Internal: destroy without unregistering from parent. Called by the
   *  parent during its own destroy — the parent clears its children set
   *  after iteration. */
  _destroyInternal(): void {
    this._invalidateForReplace()
  }

  /** Invalidate per-shape kernel caches because the parent's model topology
   *  changed. The proxy object stays alive; next run() recompiles against
   *  the new model. */
  _invalidateForReplace(): void {
    for (const sib of this.cache.values()) {
      this.proxy.send({ kind: 'destroy', payload: { graphId: sib.graphId } })
    }
    this.cache.clear()
  }
}

/** Bundles the per-shape compile result the proxy caches. */
interface ForwardSiblingMeta {
  graphId: number
  ir: CompiledIR
  meta: CompileForwardResult
}

/** Compile a per-shape sibling forward graph against the parent training
 *  graph. Used inside the forward proxy on each new resolved shape. */
async function compileSibling<M extends Module>(
  proxy: WorkerProxy,
  parentGraphId: number,
  model: M,
  forward: ForwardFn<M, InputDecls>,
  decls: ResolvedDecls,
  nextGraphId: { v: number },
): Promise<ForwardSiblingMeta> {
  const { graph } = traceModule(model, forward, decls)
  const outputTensor = graph.tensors[graph.outputs[0]!]!
  const plan = planBuffers(graph, {})
  const kernels = emitKernels(graph, plan)
  const ir: CompiledIR = { graph, paramGrads: {}, loss: outputTensor, plan, kernels }

  const childGraphId = nextGraphId.v++
  const wireIR: WireIR = { graph, plan, kernels }
  const meta = await proxy.request<CompileForwardResult>(
    { kind: 'compileForward', payload: { graphId: childGraphId, parentGraphId, ir: wireIR } },
  )
  return { graphId: childGraphId, ir, meta }
}

// ============================================================================
// Internals
// ============================================================================

type Graph = ReturnType<typeof trace>
type InitFn = (size: number, shape: readonly number[]) => Float32Array

/** Internal-only fully-normalized inputs decl (concrete dtype, shape with
 *  possibly-null wildcards). */
type NormalizedDecls = Record<string, { shape: InputShape; dtype: Dtype }>

/** Internal-only fully-resolved inputs decl (concrete dtype + shape, no nulls). */
type ResolvedDecls = Record<string, { shape: Shape; dtype: Dtype }>

function normalizeDecl(d: InputDecl): { shape: InputShape; dtype: Dtype } {
  if (Array.isArray(d)) return { shape: d as InputShape, dtype: 'f32' }
  const obj = d as InputDeclObject
  return { shape: obj.shape, dtype: obj.dtype ?? 'f32' }
}

function normalizeDecls(decls: InputDecls): NormalizedDecls {
  const out: NormalizedDecls = {}
  for (const [k, v] of Object.entries(decls)) out[k] = normalizeDecl(v)
  return out
}

/** Tracks Module instances we've already materialized, so we can detect when
 *  a factory accidentally returns the same object on a second call (the param
 *  sentinels get mutated to Tensors on the first compile; reusing the
 *  instance silently corrupts later compiles). WeakSet — does not retain. */
const seenModels = new WeakSet<Module>()

/** Trace the forward function with a fresh model + tensor inputs and capture
 *  the materialized params. Caller provides the model instance directly. */
function traceModule<M extends Module>(
  model: M,
  forward: ForwardFn<M, InputDecls>,
  inputDecls: InputDecls,
): { graph: Graph; materialized: MaterializedParams } {
  if (seenModels.has(model)) {
    throw new Error(
      `compile: factory returned a Module instance that has already been compiled. ` +
      `Factories must return a fresh \`new Model()\` per call — the compile pipeline ` +
      `consumes the instance by mutating its ParamSentinel fields into Tensors, so a ` +
      `reused instance has Tensor fields where sentinels are expected. Replace ` +
      `\`compileModule({ factory: () => existingInstance })\` with ` +
      `\`compileModule({ factory: () => new Model() })\`.`,
    )
  }
  seenModels.add(model)
  const normalized = normalizeDecls(inputDecls)
  let materialized: MaterializedParams = { tensors: {}, initFns: {}, decayFlags: {} }
  const graph = trace(() => {
    materialized = materializeParams(model)
    const inputTensors: Record<string, Tensor> = {}
    for (const [name, n] of Object.entries(normalized)) {
      const concrete = asConcreteShape(n.shape, name)
      inputTensors[name] = tensorInput(name, concrete, n.dtype)
    }
    return forward(model, inputTensors as InputsTensors<InputDecls>)
  })
  return { graph, materialized }
}

function asConcreteShape(shape: InputShape, inputName: string): Shape {
  const out: number[] = []
  for (let i = 0; i < shape.length; i++) {
    const d = shape[i]
    if (d === null || d === undefined) {
      throw new Error(
        `compile: input '${inputName}' has an unresolved parametric dim at index ${i}. ` +
        `Polymorphic shapes are only supported via the sibling method form ` +
        `(\`compiled.compileForward({ forward, inputs: { x: [null, 784] } })\`); ` +
        `compileModule requires fully concrete shapes.`,
      )
    }
    out.push(d)
  }
  return out
}

/** Resolve null-wildcard dims by inferring from each input array's length.
 *
 *  Invariant: all `null` wildcards across all inputs in a single call must
 *  resolve to the same value (matches the named-axis convention of Keras
 *  `None` / ONNX dynamic axes). If you genuinely want two different
 *  parametric dims, name them with explicit concrete shapes and recompile,
 *  or use separate `compileForward` graphs. */
function resolveDecls(
  decls: NormalizedDecls,
  inputs: Record<string, Int32Array | Float32Array>,
): ResolvedDecls {
  // First pass: validate + collect per-input wildcard info.
  type WildcardInfo = { nullIdx: number; concreteProduct: number; resolvedDim: number }
  const wildcards: Record<string, WildcardInfo | null> = {}
  for (const [name, decl] of Object.entries(decls)) {
    let nullCount = 0
    let nullIdx = -1
    let concreteProduct = 1
    for (let i = 0; i < decl.shape.length; i++) {
      const d = decl.shape[i]
      if (d === null) { nullCount++; nullIdx = i } else concreteProduct *= d!
    }
    if (nullCount === 0) { wildcards[name] = null; continue }
    if (nullCount > 1) {
      throw new Error(
        `run: input '${name}' has ${nullCount} parametric dims in shape [${decl.shape.join(', ')}]. ` +
        `Only one \`null\` wildcard per shape is supported.`,
      )
    }
    const arr = inputs[name]
    if (!arr) throw new Error(`run: missing input '${name}'`)
    if (arr.length % concreteProduct !== 0) {
      throw new Error(
        `run: input '${name}' length ${arr.length} is not divisible by ` +
        `the product of concrete dims (${concreteProduct}) in shape [${decl.shape.join(', ')}].`,
      )
    }
    wildcards[name] = { nullIdx, concreteProduct, resolvedDim: arr.length / concreteProduct }
  }

  // Cross-input consistency: every wildcard must resolve to the same value.
  let agreedDim: number | undefined
  let agreedSource: string | undefined
  for (const [name, w] of Object.entries(wildcards)) {
    if (!w) continue
    if (agreedDim === undefined) { agreedDim = w.resolvedDim; agreedSource = name; continue }
    if (w.resolvedDim !== agreedDim) {
      throw new Error(
        `run: parametric dim resolved inconsistently across inputs — '${agreedSource}' implies ${agreedDim}, ` +
        `but '${name}' implies ${w.resolvedDim}. All \`null\` wildcards in a single call must resolve to the ` +
        `same value (matches Keras None / ONNX dynamic-axis convention). Either feed matching batch lengths ` +
        `or use explicit concrete shapes for the dims that should vary independently.`,
      )
    }
  }

  // Second pass: build the resolved decls record.
  const out: ResolvedDecls = {}
  for (const [name, decl] of Object.entries(decls)) {
    const w = wildcards[name]
    if (!w) { out[name] = { shape: decl.shape as number[], dtype: decl.dtype }; continue }
    const concrete = decl.shape.slice() as number[]
    concrete[w.nullIdx] = w.resolvedDim
    out[name] = { shape: concrete, dtype: decl.dtype }
  }
  return out
}

function shapeKey(decls: ResolvedDecls): string {
  const parts: string[] = []
  for (const name of Object.keys(decls).sort()) {
    parts.push(`${name}:${decls[name]!.shape.join('x')}:${decls[name]!.dtype}`)
  }
  return parts.join('|')
}

function buildInitialParams(plan: BufferPlan, initFns: Record<string, InitFn>): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {}
  for (const [name, bufId] of plan.paramsByName) {
    const shape = plan.buffers[bufId]!.shape
    const size = shape.reduce((a, b) => a * b, 1)
    const initFn = initFns[name]
    if (!initFn) throw new Error(`compile: no init for param '${name}'`)
    out[name] = initFn(size, shape)
  }
  return out
}

function wireAdamConfig(r: AdamResult): WireAdamConfig {
  const c: AdamResolvedConfig = r.config
  return {
    lr: c.lr,
    b1: c.b1,
    b2: c.b2,
    eps: c.eps,
    weightDecay: c.weightDecay,
    lrIsScheduled: c.lrIsScheduled,
    lrtInputName: r.lrtInputName,
    decayShrinkInputName: r.decayShrinkInputName,
  }
}

function makeCaptures(
  captures: Record<string, Float32Array> | null,
  captureShapes: Record<string, number[]>,
): Captures {
  const data = new Map<string, Float32Array>()
  if (captures) for (const [name, arr] of Object.entries(captures)) data.set(name, arr)
  return new Captures(captureShapes, data)
}
