// Top-level compile pipeline.
//
//   compileModule(model, { loss, inputs, adam })
//     → trace, autograd, Adam, buffer plan, codegen, worker spawn → CompiledModule
//   compiled.compileForward({ forward, inputs })
//     → sibling forward graph sharing parent's param GPUBuffers, polymorphic
//       on `null` wildcards
//   compiled.replaceModel(newModel)
//     → swap topology, preserve worker, return fresh CompiledModule
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
 *  every method returns a Promise. */
export interface CompiledModule<M extends Module> {
  readonly ir: CompiledIR
  readonly kernelCount: number
  readonly outputShape: readonly number[]
  /** Names of the model's parameters, in materialization order. */
  readonly paramNames: readonly string[]

  step(inputs: Record<string, Int32Array | Float32Array>): Promise<number>
  step(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<StepResult>

  run(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  run(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void>
  downloadParams(): Promise<Record<string, Float32Array>>
  downloadParamGrads(): Promise<Record<string, Float32Array>>

  /** Re-initialize all params + zero optimizer state. */
  reset(): Promise<void>
  resetOptimizerState(): Promise<void>

  /** Update one or more Adam hyperparameters at runtime, without recompiling.
   *  Pass `{ rebaseLrSchedule: true }` to make a non-constant schedule's
   *  step 1 align with the next training step ("decay from now"). */
  setOptimizerConfig(update: OptimizerConfigUpdate, opts?: { rebaseLrSchedule?: boolean }): Promise<void>

  /** Compile a sibling forward-only graph that shares this runtime's worker
   *  (and therefore its param GPUBuffers). Polymorphic by default — `null`
   *  dims in the input shapes resolve per-call. */
  compileForward<I extends InputDecls>(
    opts: CompileForwardMethodOptions<M, I>,
  ): Promise<CompiledForwardModule>

  /** Swap the model topology: destroy the current training graph (and any
   *  forward siblings of it), compile a fresh training graph with the same
   *  loss / inputs / adam config but the new model factory, and return a
   *  new `CompiledModule`. The worker is reused — only the graphs inside
   *  it change. Use this when the user changes layer count, hidden width,
   *  or any other shape-affecting model parameter; you do not need to
   *  re-create the worker each time. */
  replaceModel(newFactory: () => M): Promise<CompiledModule<M>>

  /** Tear down the worker + GPU resources. */
  destroy(): void
}

/** Returned by `compiled.compileForward({...})`. Polymorphic by default:
 *  `run()` at a new resolved shape lazily compiles + caches a sibling.
 *  `kernelCount` / `outputShape` / `ir` report the most-recently-compiled
 *  sibling's values, or sensible defaults (0 / [] / undefined) before the
 *  first `run()`. */
export interface CompiledForwardModule {
  readonly ir: CompiledIR | undefined
  readonly kernelCount: number
  readonly outputShape: readonly number[]

  run(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  run(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void>
  downloadParams(): Promise<Record<string, Float32Array>>

  destroy(): void
}

// ============================================================================
// compileModule
// ============================================================================

/**
 * Compile a Module-based model into a runtime that lives in an internal
 * Web Worker. Takes a *factory* `() => new Model()` (not the instance):
 * compilation consumes a model by mutating its `ParamSentinel` fields into
 * Tensors, and sibling forward compiles need to re-trace against a fresh
 * model per resolved shape — which means the factory is called once per
 * compile, including each cache miss in a polymorphic forward proxy.
 *
 * ```ts
 * const compiled = await compileModule(() => new MLP(), {
 *   loss: (m, { x, y }) => mse(m(x), y),
 *   inputs: { x: [128, 784], y: [128] },
 *   adam: { lr: 0.001 },
 * })
 * ```
 */
export async function compileModule<M extends Module, I extends InputDecls>(
  modelFactory: () => M,
  opts: CompileModuleOptions<M, I>,
): Promise<CompiledModule<M>> {
  const proxy = new WorkerProxy(__WORKER_SOURCE__)
  try {
    return await compileModuleOnWorker(proxy, modelFactory, opts)
  } catch (e) {
    proxy.terminate()
    throw e
  }
}

/** Internal: same as `compileModule` but uses an existing worker. Used by
 *  `replaceModel` to swap the model topology without respawning the worker. */
async function compileModuleOnWorker<M extends Module, I extends InputDecls>(
  proxy: WorkerProxy,
  modelFactory: () => M,
  opts: CompileModuleOptions<M, I>,
  graphId = 0,
): Promise<CompiledModule<M>> {
  // ---- Compile-time work (main thread) ------------------------------------
  const loss = opts.loss as unknown as ForwardFn<M, InputDecls>
  const inputs = opts.inputs as InputDecls
  const { graph, materialized } = traceModule(modelFactory(), loss, inputs)
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

  return new CompiledModuleProxy<M>(
    proxy, graphId, ir, meta,
    /* modelFactory */ modelFactory,
    /* initFns */ materialized.initFns,
    /* originalOpts */ opts as unknown as CompileModuleOptions<M, InputDecls>,
    /* nextGraphId */ { v: graphId + 1 },
  )
}

// ============================================================================
// Proxy implementations
// ============================================================================

class CompiledModuleProxy<M extends Module> implements CompiledModule<M> {
  /** Forward proxies created via `compileForward` — tracked so `destroy()`
   *  and `replaceModel()` can clean them up cascade-style. */
  private readonly children = new Set<ForwardProxy<M>>()

  constructor(
    private readonly proxy: WorkerProxy,
    private readonly graphId: number,
    public readonly ir: CompiledIR,
    private readonly meta: CreateRuntimeResult,
    private readonly modelFactory: () => M,
    private readonly initFns: Record<string, InitFn>,
    private readonly originalOpts: CompileModuleOptions<M, InputDecls>,
    private readonly nextGraphId: { v: number },
  ) {}

  get kernelCount(): number { return this.meta.kernelCount }
  get outputShape(): readonly number[] { return this.meta.outputShape }
  get paramNames(): readonly string[] { return this.meta.paramNames }

  step(inputs: Record<string, Int32Array | Float32Array>): Promise<number>
  step(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<StepResult>
  async step(
    inputs: Record<string, Int32Array | Float32Array>,
    opts?: { withCaptures?: boolean },
  ): Promise<number | StepResult> {
    const r = await this.proxy.request<StepResultWire>(
      { kind: 'step', payload: { graphId: this.graphId, inputs, withCaptures: opts?.withCaptures === true } },
    )
    if (opts?.withCaptures) return { loss: r.loss, captures: makeCaptures(r.captures, this.meta.captureShapes) }
    return r.loss
  }

  run(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  run(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>
  async run(
    inputs: Record<string, Int32Array | Float32Array>,
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

  setOptimizerConfig(update: OptimizerConfigUpdate, opts?: { rebaseLrSchedule?: boolean }): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'setOptimizerConfig', payload: { graphId: this.graphId, update, rebaseLrSchedule: opts?.rebaseLrSchedule } },
    ).then(() => undefined)
  }

  async compileForward<I extends InputDecls>(
    opts: CompileForwardMethodOptions<M, I>,
  ): Promise<CompiledForwardModule> {
    const child = new ForwardProxy<M>(
      this.proxy, this.graphId,
      opts.forward as ForwardFn<M, InputDecls>,
      normalizeDecls(opts.inputs as InputDecls),
      this.nextGraphId,
      this.modelFactory,
      () => this.children.delete(child),
    )
    this.children.add(child)
    return child
  }

  async replaceModel(newFactory: () => M): Promise<CompiledModule<M>> {
    // 1. Destroy children siblings (their shared params point at our buffers).
    for (const child of this.children) child._destroyInternal()
    this.children.clear()
    // 2. Destroy current training graph in worker.
    this.proxy.send({ kind: 'destroy', payload: { graphId: this.graphId } })
    // 3. Build the new training graph on the same worker.
    return await compileModuleOnWorker<M, InputDecls>(
      this.proxy, newFactory,
      this.originalOpts as CompileModuleOptions<M, InputDecls>,
      this.graphId,
    )
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
 *  shares its param GPUBuffers. */
class ForwardProxy<M extends Module> implements CompiledForwardModule {
  private readonly cache = new Map<string, ForwardSiblingMeta>()
  private last: ForwardSiblingMeta | null = null
  private warnedCacheSize = false

  constructor(
    private readonly proxy: WorkerProxy,
    private readonly parentGraphId: number,
    private readonly forward: ForwardFn<M, InputDecls>,
    private readonly decls: NormalizedDecls,
    private readonly nextGraphId: { v: number },
    private readonly modelFactory: () => M,
    private readonly onDestroy: () => void,
  ) {}

  get ir(): CompiledIR | undefined { return this.last?.ir }
  get kernelCount(): number { return this.last?.meta.kernelCount ?? 0 }
  get outputShape(): readonly number[] { return this.last?.meta.outputShape ?? [] }

  private async siblingFor(inputs: Record<string, Int32Array | Float32Array>): Promise<ForwardSiblingMeta> {
    const resolved = resolveDecls(this.decls, inputs)
    const key = shapeKey(resolved)
    const hit = this.cache.get(key)
    if (hit) { this.last = hit; return hit }
    const sib = await compileSibling<M>(
      this.proxy, this.parentGraphId, this.modelFactory, this.forward,
      resolved, this.nextGraphId,
    )
    this.cache.set(key, sib)
    this.last = sib
    if (!this.warnedCacheSize && this.cache.size > POLYMORPHIC_CACHE_WARN_AT) {
      this.warnedCacheSize = true
      console.warn(
        `tensorgrad: forward proxy has compiled ${this.cache.size} distinct input shapes. ` +
        `Each entry holds its own kernels and bind groups. If unexpected, your call sites ` +
        `may be feeding too many distinct shapes.`,
      )
    }
    return sib
  }

  run(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  run(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>
  async run(
    inputs: Record<string, Int32Array | Float32Array>,
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
      { kind: 'uploadParams', payload: { graphId: this.parentGraphId, params, partial: !!opts?.partial } },
    )
  }

  async downloadParams(): Promise<Record<string, Float32Array>> {
    const r = await this.proxy.request<DownloadParamsResult>(
      { kind: 'downloadParams', payload: { graphId: this.parentGraphId } },
    )
    return r.params
  }

  destroy(): void {
    this._destroyInternal()
    this.onDestroy()
  }

  /** Internal: destroy without unregistering from parent. Called by the
   *  parent during its own destroy / replaceModel — the parent will clear
   *  the whole children set after iteration. */
  _destroyInternal(): void {
    for (const sib of this.cache.values()) {
      this.proxy.send({ kind: 'destroy', payload: { graphId: sib.graphId } })
    }
    this.cache.clear()
    this.last = null
  }
}

/** Bundles the per-shape compile result the proxy caches. */
interface ForwardSiblingMeta {
  graphId: number
  ir: CompiledIR
  meta: CompileForwardResult
}

const POLYMORPHIC_CACHE_WARN_AT = 8

/** Compile a per-shape sibling forward graph against the parent training
 *  graph. Used inside the forward proxy on each new resolved shape. */
async function compileSibling<M extends Module>(
  proxy: WorkerProxy,
  parentGraphId: number,
  modelFactory: () => M,
  forward: ForwardFn<M, InputDecls>,
  decls: ResolvedDecls,
  nextGraphId: { v: number },
): Promise<ForwardSiblingMeta> {
  const { graph } = traceModule(modelFactory(), forward, decls)
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

/** Trace the forward function with a fresh model + tensor inputs and capture
 *  the materialized params. Caller provides the model instance directly. */
function traceModule<M extends Module>(
  model: M,
  forward: ForwardFn<M, InputDecls>,
  inputDecls: InputDecls,
): { graph: Graph; materialized: MaterializedParams } {
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

/** Resolve null-wildcard dims by inferring from each input array's length. */
function resolveDecls(
  decls: NormalizedDecls,
  inputs: Record<string, Int32Array | Float32Array>,
): ResolvedDecls {
  const out: ResolvedDecls = {}
  for (const [name, decl] of Object.entries(decls)) {
    let nullCount = 0
    let nullIdx = -1
    let concreteProduct = 1
    for (let i = 0; i < decl.shape.length; i++) {
      const d = decl.shape[i]
      if (d === null) { nullCount++; nullIdx = i } else concreteProduct *= d!
    }
    if (nullCount === 0) {
      out[name] = { shape: decl.shape as number[], dtype: decl.dtype }
      continue
    }
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
    const concrete = decl.shape.slice() as number[]
    concrete[nullIdx] = arr.length / concreteProduct
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
