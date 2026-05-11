// Top-level compile(): trace → autograd → buffer plan → codegen → runtime.
//
// Two entry points:
//   * `compile(traceFn)`        — low-level. User declares params via
//                                 paramInput() inside the trace.
//   * `compileModule(model, …)` — high-level. User defines the model as a
//                                 Module tree; the library auto-discovers
//                                 params, traces the forward, appends grad
//                                 and Adam, and returns a runtime.
//
// As of the worker-architecture refactor: compile-time work (trace, autograd,
// buffer planning, codegen) runs on the main thread. createRuntime and all
// dispatch/mapAsync work runs in a Web Worker spawned per top-level compile;
// the returned `CompiledModule` is a thin proxy over the worker channel.
// See specs/WorkerArchitecture.md.

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

// `__WORKER_SOURCE__` is replaced at build time by scripts/build.mjs with the
// stringified contents of the bundled src/worker.ts. Declared here so TS is
// happy; substituted as a string literal by esbuild's `define` during
// `npm run build:js`. See scripts/build.mjs.
declare const __WORKER_SOURCE__: string

// ============================================================================
// Public types
// ============================================================================

/** Shape of a declared input. Each dim is either a fixed number, or `null`
 *  to mark the dim as parametric (its concrete value is inferred from the
 *  actual TypedArray length at `run()` time). At most one `null` per shape
 *  in this iteration; multiple parametric dims require named symbols, which
 *  aren't exposed yet. Matches the TF/ONNX/MLIR convention of using a
 *  wildcard for dynamic dims. */
export type InputShape = readonly (number | null)[]

/** Declares one input tensor of the model's forward function. The name is the
 *  key in the `inputs:` Record at compile time and the key on the `step()`/
 *  `run()` data object at runtime. */
export interface InputDecl {
  shape: InputShape
  dtype?: Dtype
}

/** Inputs declaration: a Record from input name to its shape/dtype. */
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
// CompiledModule / CompiledForwardModule — main-thread proxy surface
// ============================================================================

export interface CompileModuleOptions<I extends InputDecls = InputDecls> {
  inputs?: I
  adam?: AdamConfig
}

export interface CompileForwardOptions<I extends InputDecls = InputDecls> {
  inputs?: I
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

export interface CompileForwardMethodOptions<I extends InputDecls = InputDecls> {
  inputs?: I
}

/** Returned by `compileModule`. Proxies all GPU work to a worker held
 *  internally; user code awaits Promises and never sees the worker. */
export interface CompiledModule<M extends Module> {
  readonly ir: CompiledIR
  readonly kernelCount: number
  readonly outputShape: readonly number[]
  /** Names of the model's parameters, in materialization order. The actual
   *  GPUBuffers live in the worker; use `downloadParams()` for values. */
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
   *  Only the fields you pass are changed; everything else stays put. The
   *  step counter is preserved (the new `lr` schedule, if any, resolves at
   *  the current step). Use a number for constant `lr`, or one of the
   *  `lr.*` shape constructors. Note: which params receive weight decay is
   *  baked at compile time, so adjusting `weightDecay` here changes the
   *  shrink magnitude on already-decayed params, not which params decay. */
  setOptimizerConfig(update: OptimizerConfigUpdate): Promise<void>

  /** Compile a sibling forward-only graph that shares this runtime's worker
   *  (and therefore its param GPUBuffers). */
  compileForward<I extends InputDecls>(
    forward: ForwardFn<M, I>,
    opts?: CompileForwardMethodOptions<I>,
  ): Promise<CompiledForwardModule>

  /** Free the runtime's GPU resources and terminate the worker. */
  destroy(): void
}

/** Returned by `compileForward` (and by the `compileForward` method). */
export interface CompiledForwardModule {
  readonly ir: CompiledIR
  readonly kernelCount: number
  readonly outputShape: readonly number[]
  readonly paramNames: readonly string[]

  run(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  run(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>

  /** Single-flight variant of `run`: if a previous `runLatest` call is still
   *  in flight, this one waits for it; if multiple `runLatest` calls queue up
   *  while one is in flight, only the *most recent* arguments actually run
   *  when the in-flight call finishes, and every queued caller resolves with
   *  that latest result. Useful for live-preview UI patterns where stale
   *  inputs (e.g. earlier mouse positions, partial drawings) should be
   *  dropped in favor of the newest user state. */
  runLatest(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  runLatest(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void>
  downloadParams(): Promise<Record<string, Float32Array>>

  /** Polymorphic-only: pre-compile a sibling at a specific resolved shape
   *  so the first `run()` at that shape doesn't pay the trace + codegen
   *  cost. Throws on non-polymorphic proxies (their shape is already
   *  fixed) or on shapes that still contain a `null` wildcard. */
  precompile?(decls: InputDecls): Promise<CompiledForwardModule>

  /** Polymorphic-only: number of distinct input shapes this proxy has
   *  cached. Stays at 1 for non-polymorphic proxies. */
  readonly cachedShapeCount?: number

  destroy(): void
}

// ============================================================================
// compileModule / compileForward
// ============================================================================

/**
 * Compile a Module-based model. Pass a *factory* `() => new Model()`, not the
 * model instance itself: compilation mutates the tree (every `ParamSentinel`
 * field becomes a real `Tensor`), so the instance is consumed and shouldn't be
 * referenced afterwards.
 *
 * The forward function takes the materialized model and a Record of named
 * input tensors, returns the loss tensor:
 *
 *   inputs: {
 *     tokens:  { shape: [B, T], dtype: 'i32' },
 *     targets: { shape: [B, T], dtype: 'i32' },
 *   }
 *   forward: (m, { tokens, targets }) => …
 *
 * Returns a `CompiledModule` proxy. All GPU work (createRuntime, step, run,
 * mapAsync) happens in an internal worker; calls return Promises that resolve
 * when the worker replies.
 */
export async function compileModule<M extends Module, I extends InputDecls = InputDecls>(
  modelFactory: () => M,
  forward: ForwardFn<M, I>,
  opts: CompileModuleOptions<I> = {},
): Promise<CompiledModule<M>> {
  // ---- Compile-time work (main thread) ------------------------------------
  const { graph, materialized } = traceModule(modelFactory, forward, opts.inputs ?? {})
  const { paramGrads, loss } = appendGrad(graph)
  const adamResult = opts.adam
    ? appendAdam(graph, paramGrads, materialized.tensors, opts.adam, materialized.decayFlags)
    : undefined

  const plan = planBuffers(graph, paramGrads, adamResult?.writebacks ?? [])
  const kernels = emitKernels(graph, plan)
  const ir: CompiledIR = { graph, paramGrads, loss, plan, kernels }

  // Initial params: resolve init shapes to Float32Arrays now (main thread).
  // These transfer (zero-copy) to the worker as part of createRuntime.
  const initialParams = buildInitialParams(plan, materialized.initFns)

  // ---- Spawn worker, send IR + initial params -----------------------------
  const proxy = new WorkerProxy(__WORKER_SOURCE__)
  const wireIR: WireIR = { graph, plan, kernels }
  const wireAdam = adamResult ? wireAdamConfig(adamResult) : null
  const transfers = transferablesOfRecord(initialParams)

  let meta: CreateRuntimeResult
  try {
    meta = await proxy.request<CreateRuntimeResult>(
      { kind: 'createRuntime', payload: { graphId: 0, ir: wireIR, initialParams, adam: wireAdam } },
      transfers,
    )
  } catch (e) {
    proxy.terminate()
    throw e
  }

  return new CompiledModuleProxy<M>(
    proxy, /* graphId */ 0, ir, meta, modelFactory,
    /* initFns */ materialized.initFns,
    /* nextGraphId */ { v: 1 },
  )
}

/**
 * Forward-only compile. Spawns its own worker. For sibling graphs that share
 * params with a training graph, prefer the `compileForward` method on the
 * CompiledModule returned by `compileModule()`.
 */
export async function compileForward<M extends Module, I extends InputDecls = InputDecls>(
  modelFactory: () => M,
  forward: ForwardFn<M, I>,
  opts: CompileForwardOptions<I> = {},
): Promise<CompiledForwardModule> {
  const { graph, materialized } = traceModule(modelFactory, forward, opts.inputs ?? {})
  const outputTensor = graph.tensors[graph.outputs[0]!]!
  const plan = planBuffers(graph, /* paramGrads */ {})
  const kernels = emitKernels(graph, plan)
  const ir: CompiledIR = { graph, paramGrads: {}, loss: outputTensor, plan, kernels }

  const initialParams = buildInitialParams(plan, materialized.initFns)
  const proxy = new WorkerProxy(__WORKER_SOURCE__)
  const wireIR: WireIR = { graph, plan, kernels }
  const transfers = transferablesOfRecord(initialParams)

  let meta: CreateRuntimeResult
  try {
    meta = await proxy.request<CreateRuntimeResult>(
      { kind: 'createRuntime', payload: { graphId: 0, ir: wireIR, initialParams, adam: null } },
      transfers,
    )
  } catch (e) {
    proxy.terminate()
    throw e
  }

  return new CompiledForwardModuleProxy(proxy, /* graphId */ 0, ir, meta, /* ownsWorker */ true)
}

// ============================================================================
// Proxy implementations
// ============================================================================

class CompiledModuleProxy<M extends Module> implements CompiledModule<M> {
  constructor(
    private readonly proxy: WorkerProxy,
    private readonly graphId: number,
    public readonly ir: CompiledIR,
    private readonly meta: CreateRuntimeResult,
    private readonly modelFactory: () => M,
    /** Init closures captured from materializeParams at compile time. Used
     *  by reset() to regenerate initial param values. */
    private readonly initFns: Record<string, InitFn>,
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
    // Note: inputs are copied (not transferred) into the worker. Callers
    // commonly reuse the same TypedArray as a scratch buffer across step()
    // calls; transferring would detach it. The copy cost is small relative
    // to a training step's GPU work.
    const r = await this.proxy.request<StepResultWire>(
      { kind: 'step', payload: { graphId: this.graphId, inputs, withCaptures: opts?.withCaptures === true } },
    )
    if (opts?.withCaptures) {
      return { loss: r.loss, captures: makeCaptures(r.captures, this.meta.captureShapes) }
    }
    return r.loss
  }

  run(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  run(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>
  async run(
    inputs: Record<string, Int32Array | Float32Array>,
    opts?: { withCaptures?: boolean },
  ): Promise<Float32Array | RunResult> {
    // Inputs copied (see note in step()).
    const r = await this.proxy.request<RunResultWire>(
      { kind: 'run', payload: { graphId: this.graphId, inputs, withCaptures: opts?.withCaptures === true } },
    )
    if (opts?.withCaptures) {
      return { output: r.output, captures: makeCaptures(r.captures, this.meta.captureShapes) }
    }
    return r.output
  }

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void> {
    // Params copied (see note in step()) — caller's Float32Arrays stay valid.
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
    // Re-init main-thread, upload, then reset Adam state on worker. Two
    // round-trips but reset() is rare. The init closures were captured at
    // compile time and stashed on the proxy.
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

  async compileForward<I extends InputDecls>(
    forward: ForwardFn<M, I>,
    opts: CompileForwardMethodOptions<I> = {},
  ): Promise<CompiledForwardModule> {
    const decls = opts.inputs ?? {}
    // Polymorphic path: any `null` wildcard in an input shape defers
    // compilation. The proxy compiles + caches lazily on each first call
    // at a new resolved shape.
    if (isPolymorphicDecls(decls)) {
      return new PolymorphicForwardProxy<M>(
        this.proxy, this.graphId, this.modelFactory, forward as ForwardFn<M, InputDecls>,
        decls, this.nextGraphId,
      )
    }
    return await compileForwardEager<M>(
      this.proxy, this.graphId, this.modelFactory, forward as ForwardFn<M, InputDecls>,
      decls, this.nextGraphId,
    )
  }

  destroy(): void {
    // Fire-and-forget destroy; postMessage ordering ensures the worker
    // processes any in-flight requests before we terminate it.
    this.proxy.send({ kind: 'destroy', payload: { graphId: this.graphId } })
    this.proxy.terminate()
  }
}

class CompiledForwardModuleProxy implements CompiledForwardModule {
  constructor(
    private readonly proxy: WorkerProxy,
    private readonly graphId: number,
    public readonly ir: CompiledIR,
    private readonly meta: CompileForwardResult | CreateRuntimeResult,
    private readonly ownsWorker: boolean,
  ) {}

  // Single-flight state for `runLatest`. One run is allowed in flight; at
  // most one queued waiter. If a new call arrives while one is queued, the
  // older queued waiter rejects with `AbortError` (matching RxJS switchMap
  // / p-debounce convention). Only the *most recent* caller's args run when
  // the in-flight call finishes.
  private latestActive: Promise<unknown> | null = null
  private latestPending:
    | { inputs: Record<string, Int32Array | Float32Array>; opts: { withCaptures?: boolean } | undefined;
        resolver: { resolve: (r: Float32Array | RunResult) => void; reject: (e: Error) => void } }
    | null = null

  get kernelCount(): number { return this.meta.kernelCount }
  get outputShape(): readonly number[] { return this.meta.outputShape }
  get paramNames(): readonly string[] { return this.meta.paramNames }

  run(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  run(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>
  async run(
    inputs: Record<string, Int32Array | Float32Array>,
    opts?: { withCaptures?: boolean },
  ): Promise<Float32Array | RunResult> {
    // Inputs copied; caller's TypedArrays stay valid.
    const r = await this.proxy.request<RunResultWire>(
      { kind: 'run', payload: { graphId: this.graphId, inputs, withCaptures: opts?.withCaptures === true } },
    )
    if (opts?.withCaptures) {
      return { output: r.output, captures: makeCaptures(r.captures, this.meta.captureShapes) }
    }
    return r.output
  }

  runLatest(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  runLatest(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>
  runLatest(
    inputs: Record<string, Int32Array | Float32Array>,
    opts?: { withCaptures?: boolean },
  ): Promise<Float32Array | RunResult> {
    if (!this.latestActive) {
      const p = this.run(inputs, opts as { withCaptures: true })
      this.latestActive = p.finally(() => {
        this.latestActive = null
        // Drain the queue, if any: only the latest waiter runs.
        if (this.latestPending) {
          const pend = this.latestPending
          this.latestPending = null
          this.runLatest(pend.inputs, pend.opts as { withCaptures: true }).then(
            pend.resolver.resolve, pend.resolver.reject,
          )
        }
      })
      return p
    }
    // Already in flight: this call replaces any older queued waiter. The
    // displaced waiter rejects with AbortError — caller should swallow it
    // (it just means a newer call superseded them).
    if (this.latestPending) {
      this.latestPending.resolver.reject(abortErr('runLatest: superseded by newer call'))
    }
    return new Promise<Float32Array | RunResult>((resolve, reject) => {
      this.latestPending = { inputs, opts, resolver: { resolve, reject } }
    })
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

  destroy(): void {
    this.proxy.send({ kind: 'destroy', payload: { graphId: this.graphId } })
    if (this.ownsWorker) this.proxy.terminate()
  }
}

function abortErr(msg: string): Error {
  const e = new Error(msg)
  e.name = 'AbortError'
  return e
}

/** Threshold at which the polymorphic forward proxy emits a one-time
 *  console warning about cache growth. Picked as a reasonable upper bound
 *  for legitimate use (B=1 predict + B=N eval + a few sizes for autoregressive
 *  generation) but low enough to catch accidental shape fuzzing early. */
const POLYMORPHIC_CACHE_WARN_AT = 8

/** Polymorphic sibling forward proxy. Returned by `train.compileForward(...)`
 *  when any input shape contains a `null` wildcard. Compiles a fresh sibling
 *  graph the first time `run()` is called at each distinct resolved shape;
 *  subsequent calls at the same shape hit the cache. All sibling graphs
 *  share the training graph's param GPUBuffers, so a single set of params
 *  is visible across every batch size. */
class PolymorphicForwardProxy<M extends Module> implements CompiledForwardModule {
  private readonly cache = new Map<string, CompiledForwardModuleProxy>()
  /** Most-recently-compiled sibling. Used as a fallback for `ir`,
   *  `kernelCount`, `outputShape`, `paramNames` after the first call. */
  private last: CompiledForwardModuleProxy | null = null
  /** Single warning per proxy when the cache crosses the threshold; further
   *  growth is silent (the user has been told). */
  private warnedCacheSize = false

  // Single-flight state for runLatest. Same reject-queued semantics as the
  // eager proxy: an older queued waiter rejects with AbortError when a
  // newer call arrives.
  private latestActive: Promise<unknown> | null = null
  private latestPending:
    | { inputs: Record<string, Int32Array | Float32Array>; opts: { withCaptures?: boolean } | undefined;
        resolver: { resolve: (r: Float32Array | RunResult) => void; reject: (e: Error) => void } }
    | null = null

  constructor(
    private readonly proxy: WorkerProxy,
    private readonly parentGraphId: number,
    private readonly modelFactory: () => M,
    private readonly forward: ForwardFn<M, InputDecls>,
    private readonly decls: InputDecls,
    private readonly nextGraphId: { v: number },
  ) {}

  get ir(): CompiledIR {
    if (!this.last) throw new Error('polymorphic forward graph: no compile yet — call run() at least once first')
    return this.last.ir
  }
  get kernelCount(): number {
    if (!this.last) throw new Error('polymorphic forward graph: no compile yet — call run() at least once first')
    return this.last.kernelCount
  }
  get outputShape(): readonly number[] {
    if (!this.last) throw new Error('polymorphic forward graph: no compile yet — call run() at least once first')
    return this.last.outputShape
  }
  get paramNames(): readonly string[] {
    if (!this.last) throw new Error('polymorphic forward graph: no compile yet — call run() at least once first')
    return this.last.paramNames
  }

  private async siblingFor(inputs: Record<string, Int32Array | Float32Array>): Promise<CompiledForwardModuleProxy> {
    const resolved = resolveDecls(this.decls, inputs)
    return await this.siblingForDecls(resolved)
  }

  private async siblingForDecls(resolved: InputDecls): Promise<CompiledForwardModuleProxy> {
    const key = shapeKey(resolved)
    const hit = this.cache.get(key)
    if (hit) { this.last = hit; return hit }
    const sib = await compileForwardEager<M>(
      this.proxy, this.parentGraphId, this.modelFactory, this.forward,
      resolved, this.nextGraphId,
    )
    this.cache.set(key, sib)
    this.last = sib
    if (!this.warnedCacheSize && this.cache.size > POLYMORPHIC_CACHE_WARN_AT) {
      this.warnedCacheSize = true
      console.warn(
        `tensorgrad: polymorphic forward proxy has compiled ${this.cache.size} distinct ` +
        `input shapes. Each entry holds its own kernels and bind groups — if this is ` +
        `unexpected, your call sites may be feeding too many distinct shapes. ` +
        `Use .precompile(...) to pre-warm a known set of shapes, or .destroy() to clear.`,
      )
    }
    return sib
  }

  /** Pre-compile a sibling at a specific resolved shape so the first
   *  matching `run()` is hot. Useful for latency-sensitive paths that
   *  would otherwise pay the trace + codegen cost on first call. Returns
   *  the resolved sibling proxy so its `ir`/`kernelCount`/`outputShape`
   *  metadata are also available eagerly. Shapes here must be fully
   *  concrete (no `null`). */
  async precompile(decls: InputDecls): Promise<CompiledForwardModule> {
    if (isPolymorphicDecls(decls)) {
      throw new Error('precompile: shapes must be fully concrete (no `null` wildcards)')
    }
    return await this.siblingForDecls(decls)
  }

  /** Number of distinct input shapes this proxy has compiled (and is
   *  caching). Useful for monitoring if you suspect cache growth. */
  get cachedShapeCount(): number { return this.cache.size }

  run(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  run(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>
  async run(
    inputs: Record<string, Int32Array | Float32Array>,
    opts?: { withCaptures?: boolean },
  ): Promise<Float32Array | RunResult> {
    const sib = await this.siblingFor(inputs)
    return await sib.run(inputs, opts as { withCaptures: true })
  }

  runLatest(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  runLatest(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunResult>
  runLatest(
    inputs: Record<string, Int32Array | Float32Array>,
    opts?: { withCaptures?: boolean },
  ): Promise<Float32Array | RunResult> {
    if (!this.latestActive) {
      const p = this.run(inputs, opts as { withCaptures: true })
      this.latestActive = p.finally(() => {
        this.latestActive = null
        if (this.latestPending) {
          const pend = this.latestPending
          this.latestPending = null
          this.runLatest(pend.inputs, pend.opts as { withCaptures: true }).then(
            pend.resolver.resolve, pend.resolver.reject,
          )
        }
      })
      return p
    }
    if (this.latestPending) {
      this.latestPending.resolver.reject(abortErr('runLatest: superseded by newer call'))
    }
    return new Promise<Float32Array | RunResult>((resolve, reject) => {
      this.latestPending = { inputs, opts, resolver: { resolve, reject } }
    })
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
    for (const sib of this.cache.values()) sib.destroy()
    this.cache.clear()
    this.last = null
  }
}

/** Eager (concrete-shape) compileForward sibling. Extracted so both the
 *  method form (when shapes are concrete) and the polymorphic proxy (one
 *  call per distinct resolved shape) share the same compilation pipeline. */
async function compileForwardEager<M extends Module>(
  proxy: WorkerProxy,
  parentGraphId: number,
  modelFactory: () => M,
  forward: ForwardFn<M, InputDecls>,
  decls: InputDecls,
  nextGraphId: { v: number },
): Promise<CompiledForwardModuleProxy> {
  const { graph, materialized: _m } = traceModule(modelFactory, forward, decls)
  const outputTensor = graph.tensors[graph.outputs[0]!]!
  const plan = planBuffers(graph, {})
  const kernels = emitKernels(graph, plan)
  const ir: CompiledIR = { graph, paramGrads: {}, loss: outputTensor, plan, kernels }

  const childGraphId = nextGraphId.v++
  const wireIR: WireIR = { graph, plan, kernels }
  const meta = await proxy.request<CompileForwardResult>(
    { kind: 'compileForward', payload: { graphId: childGraphId, parentGraphId, ir: wireIR } },
  )
  return new CompiledForwardModuleProxy(proxy, childGraphId, ir, meta, /* ownsWorker */ false)
}

// ============================================================================
// Internals
// ============================================================================

type Graph = ReturnType<typeof trace>
type InitFn = (size: number, shape: readonly number[]) => Float32Array

/** Trace the forward function with a fresh model + tensor inputs and capture
 *  the materialized params. Shared by both compile entry points; everything
 *  past this point (grad/adam/buffer plan/runtime) diverges. */
function traceModule<M extends Module, I extends InputDecls>(
  modelFactory: () => M,
  forward: ForwardFn<M, I>,
  inputDecls: InputDecls,
): { graph: Graph; materialized: MaterializedParams } {
  const model = modelFactory()
  let materialized: MaterializedParams = { tensors: {}, initFns: {}, decayFlags: {} }
  const graph = trace(() => {
    materialized = materializeParams(model)
    const inputTensors: Record<string, Tensor> = {}
    for (const [name, decl] of Object.entries(inputDecls)) {
      const concrete = asConcreteShape(decl.shape, name)
      inputTensors[name] = tensorInput(name, concrete, decl.dtype ?? 'f32')
    }
    return forward(model, inputTensors as InputsTensors<I>)
  })
  return { graph, materialized }
}

/** Assert every dim in an InputShape is a number (no `null` wildcards left).
 *  Called from paths that require fully-resolved shapes — `compileModule`,
 *  standalone `compileForward`, and per-shape compiles inside the
 *  polymorphic proxy after wildcard resolution. */
function asConcreteShape(shape: InputShape, inputName: string): Shape {
  const out: number[] = []
  for (let i = 0; i < shape.length; i++) {
    const d = shape[i]
    if (d === null || d === undefined) {
      throw new Error(
        `compile: input '${inputName}' has an unresolved parametric dim at index ${i}. ` +
        `Polymorphic shapes are only supported via the sibling method form ` +
        `(\`train.compileForward(predictFwd, { inputs: { x: { shape: [null, 784] } } })\`); ` +
        `standalone compileForward and compileModule require fully concrete shapes.`,
      )
    }
    out.push(d)
  }
  return out
}

/** Detect any `null` wildcard in an `InputDecls` shape, which signals the
 *  caller wants per-call shape inference (the polymorphic proxy path). */
function isPolymorphicDecls(decls: InputDecls): boolean {
  for (const decl of Object.values(decls)) {
    for (const d of decl.shape) if (d === null) return true
  }
  return false
}

/** Resolve null-wildcard dims in an InputDecls against actual input
 *  TypedArrays. For each input that has a wildcard, infer the missing dim
 *  from `arr.length / product(concrete dims)`. At most one `null` per
 *  shape allowed (multi-null requires named symbols, deferred). */
function resolveDecls(
  decls: InputDecls,
  inputs: Record<string, Int32Array | Float32Array>,
): InputDecls {
  const out: InputDecls = {}
  for (const [name, decl] of Object.entries(decls)) {
    let nullCount = 0
    let nullIdx = -1
    let concreteProduct = 1
    for (let i = 0; i < decl.shape.length; i++) {
      const d = decl.shape[i]
      if (d === null) { nullCount++; nullIdx = i } else concreteProduct *= d!
    }
    if (nullCount === 0) { out[name] = decl; continue }
    if (nullCount > 1) {
      throw new Error(
        `run: input '${name}' has ${nullCount} parametric dims in shape [${decl.shape.join(', ')}]. ` +
        `Only one \`null\` wildcard per shape is supported (multi-wildcard requires named symbols, not yet exposed).`,
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
    const resolved = arr.length / concreteProduct
    const concrete = decl.shape.slice() as (number | null)[]
    concrete[nullIdx] = resolved
    const dt: Dtype = decl.dtype ?? 'f32'
    out[name] = { shape: concrete as number[], dtype: dt }
  }
  return out
}

/** Canonical cache key for a fully concrete `InputDecls`. Used by the
 *  polymorphic forward proxy to look up an already-compiled graph at the
 *  resolved input shapes. */
function shapeKey(decls: InputDecls): string {
  const parts: string[] = []
  for (const name of Object.keys(decls).sort()) {
    parts.push(`${name}:${decls[name]!.shape.join('x')}:${decls[name]!.dtype ?? 'f32'}`)
  }
  return parts.join('|')
}

/** Run each param's init function against its declared shape to produce the
 *  initial Float32Arrays. Runs main-thread before transfer to the worker. */
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

/** Subset of AdamResolvedConfig that crosses the wire (drops decayFilter,
 *  which is only used at compile time). */
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

/** Wrap a worker-returned `Record<name, Float32Array>` in a Captures instance
 *  using the static capture shapes captured at compile time. */
function makeCaptures(
  captures: Record<string, Float32Array> | null,
  captureShapes: Record<string, number[]>,
): Captures {
  const data = new Map<string, Float32Array>()
  if (captures) {
    for (const [name, arr] of Object.entries(captures)) data.set(name, arr)
  }
  return new Captures(captureShapes, data)
}

