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
  type AdamConfig, type AdamResult, type AdamResolvedConfig,
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

/** Declares one input tensor of the model's forward function. The name is the
 *  key in the `inputs:` Record at compile time and the key on the `step()`/
 *  `run()` data object at runtime. */
export interface InputDecl {
  shape: Shape
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

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void>
  downloadParams(): Promise<Record<string, Float32Array>>

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

  async compileForward<I extends InputDecls>(
    forward: ForwardFn<M, I>,
    opts: CompileForwardMethodOptions<I> = {},
  ): Promise<CompiledForwardModule> {
    const { graph, materialized: _materialized } = traceModule(this.modelFactory, forward, opts.inputs ?? {})
    const outputTensor = graph.tensors[graph.outputs[0]!]!
    const plan = planBuffers(graph, /* paramGrads */ {})
    const kernels = emitKernels(graph, plan)
    const ir: CompiledIR = { graph, paramGrads: {}, loss: outputTensor, plan, kernels }

    const childGraphId = this.nextGraphId.v++
    const wireIR: WireIR = { graph, plan, kernels }

    const meta = await this.proxy.request<CompileForwardResult>(
      { kind: 'compileForward', payload: { graphId: childGraphId, parentGraphId: this.graphId, ir: wireIR } },
    )

    return new CompiledForwardModuleProxy(this.proxy, childGraphId, ir, meta, /* ownsWorker */ false)
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
      inputTensors[name] = tensorInput(name, decl.shape, decl.dtype ?? 'f32')
    }
    return forward(model, inputTensors as InputsTensors<I>)
  })
  return { graph, materialized }
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

