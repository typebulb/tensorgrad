// Top-level compile pipeline:
//   compileModule        — trace, autograd, optimizer, buffers, codegen, spawn worker
//   compileForward (sib) — forward-only graph sharing the parent's worker + params
//   replaceModel         — swap topology in place; sibling proxies invalidate their caches
//
// Compile-time work runs on the main thread; everything past createRuntime
// runs in a worker (see specs/WorkerArchitecture.md).
//
// Forward proxies are always polymorphic-capable: `null` dims in the declared
// inputs resolve from each call's TypedArray length, and a sibling is
// compiled + cached per distinct resolved shape.

import type { Tensor, Shape, Dtype } from './ir.js'
import { trace, tensorInput } from './trace.js'
import { appendGrad, type GradResult } from './grad.js'
import {
  appendAdam, resolveLR,
  type AdamConfig, type AdamResult, type AdamResolvedConfig, type LR,
} from './adam.js'
import { appendSGD, type SGDConfig, type SGDResult, type SGDResolvedConfig } from './sgd.js'
import { planBuffers, type BufferPlan } from './buffers.js'
import { emitKernels, type KernelSpec } from './codegen.js'
import {
  Captures, type RunResult, type StepResult, type RunOptions, type Outcome, type UploadParamsOptions,
} from './runtime.js'
import { Module, materializeParams, mulberry32, type MaterializedParams, type Rng, type InitFn } from './module.js'
import { WorkerProxy } from './worker-proxy.js'
import {
  transferablesOfRecord,
  type Req, type WireIR, type WireAdamConfig, type WireSGDConfig, type WireOptimizerConfig,
  type CreateRuntimeResult, type CompileForwardResult,
  type StepResultWire, type RunResultWire, type DownloadParamsResult,
} from './worker-protocol.js'

declare const __WORKER_SOURCE__: string

/** Shape of a declared input. Each dim is a fixed number or `null` to mark
 *  the dim parametric (resolved from actual TypedArray length at run time).
 *  At most one `null` per shape. Matches the TF/ONNX/MLIR wildcard convention. */
export type InputShape = readonly (number | null)[]

/** Object form of an input declaration. Required when dtype isn't `f32` (use
 *  the tuple form for f32). `dtype` is required so the input's TypedArray type
 *  always resolves to a single literal, not a `Float32Array | Int32Array` union. */
export interface InputDeclObject {
  readonly shape: InputShape
  readonly dtype: Dtype
}

/** An input declaration value: shape tuple (dtype = `f32`) or
 *  `{ shape, dtype }` for non-f32 (`i32` / `bool`). */
export type InputDecl = InputShape | InputDeclObject

/** Inputs declaration: name -> shape (tuple or object). */
export type InputDecls = Record<string, InputDecl>

/** Maps an `InputDecls` Record to its forward-time tensor counterpart —
 *  same keys, each value is a Tensor. */
export type InputsTensors<I extends InputDecls> = { [K in keyof I]: Tensor }

/** Extract the declared dtype from an `InputDecl`. Tuple shapes resolve to
 *  `'f32'`; the object form's required `dtype` field is preserved as a literal. */
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

/** A typed view of a Module's params, mirroring the class structure with
 *  every Tensor leaf replaced by its backing `Float32Array`. Returned by
 *  `compiled.downloadParams()` so user code can write
 *  `params.layers[0].W` instead of `params['layers.0.W']` — typed,
 *  autocompletable, no string indexing.
 *
 *  Filters keep only fields that contribute params: `Tensor`, nested
 *  `Module`, and arrays of `Module`. Other instance fields (`inDim`,
 *  config metadata, etc.) are pruned. */
export type ParamTree<M> = {
  [K in keyof M as
      M[K] extends Tensor ? K
    : M[K] extends Module ? K
    : M[K] extends readonly Module[] ? K
    : never
  ]:
      M[K] extends Tensor ? Float32Array
    : M[K] extends readonly (infer U extends Module)[] ? ParamTree<U>[]
    : M[K] extends Module ? ParamTree<M[K]>
    : never
}

/** Forward function shape. */
export type ForwardFn<M extends Module, I extends InputDecls = InputDecls> =
  (m: M, inputs: InputsTensors<I>) => Tensor

/** The compile pipeline's IR bundle: the augmented graph, per-param
 *  gradient tensors, the loss output, the buffer plan, and the emitted
 *  kernel specs. Exposed on `compiled.ir` for inspection (see the
 *  "Inspecting the compiled IR" section in the README). */
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

/** Options to `compileModule`. Generic over `M` (the model class) and `I`
 *  (the inputs decl) so the resulting `CompiledModule<M, I>` carries enough
 *  type info for `downloadParams` to mirror the model tree and for `step`
 *  / `run` to type their inputs against declared dtypes. */
export interface CompileModuleOptions<M extends Module, I extends InputDecls = InputDecls> {
  /** Model factory `() => new Model()`. Invoked once per compile (and once
   *  per cache-miss in any polymorphic sibling forward). The model instance
   *  is consumed: its `ParamSentinel` fields are mutated into Tensors. */
  factory: () => M
  /** Forward function that returns the scalar loss tensor. */
  loss: ForwardFn<M, I>
  /** Input shape declarations (one per named tensor input). */
  inputs: I
  /** Adam / AdamW optimizer. Mutually exclusive with `sgd`. When neither is
   *  present, the module compiles but has no optimizer — `step()` will fail.
   *  (Used internally for `compileToIR`-like flows where the user only wants
   *  the forward pass.) */
  adam?: AdamConfig
  /** SGD / SGD-with-momentum / Nesterov optimizer. Mutually exclusive with `adam`. */
  sgd?: SGDConfig
  /** 32-bit integer seed for the param-init RNG. Same seed + same model
   *  topology → identical initial params, every time. If omitted, a seed
   *  is generated and exposed as `compiled.seed` so you can reproduce a
   *  run by passing it back. */
  seed?: number
}

/** Options to `compiled.compileForward(...)` — a sibling forward-only graph
 *  that shares the parent training graph's worker and param GPUBuffers.
 *  Polymorphic by default: `null` dims in `inputs` resolve from each call's
 *  TypedArray length; a sibling is compiled and cached per distinct shape. */
export interface CompileForwardMethodOptions<M extends Module, I extends InputDecls = InputDecls> {
  /** Forward function returning the output tensor (one per shape value). */
  forward: ForwardFn<M, I>
  /** Input shape declarations. `null` dims become parametric; the proxy
   *  compiles + caches a sibling per distinct resolved shape on first
   *  `run()` at that shape. */
  inputs: I
}


/** Returned by `compileModule`. Proxies all GPU work to an internal worker;
 *  every method returns a Promise. Generic over the declared inputs shape
 *  `I` so `step` / `run` accept inputs with the right TypedArray per dtype. */
export interface CompiledModule<M extends Module, I extends InputDecls = InputDecls> {
  /** The compiled IR: forward graph, autograd, optimizer ops, buffer plan,
   *  kernels. Use `compiled.ir.graph` to inspect ops, tensors, and
   *  captures (see README). Swapped in place by `replaceModel`. */
  readonly ir: CompiledIR
  readonly kernelCount: number
  readonly outputShape: readonly number[]
  /** Names of the model's parameters, in materialization order. */
  readonly paramNames: readonly string[]
  /** The actual seed used for param init (either the one you passed, or a
   *  freshly-generated one if you didn't). Pass this back as
   *  `compileModule({ seed: ... })` to reproduce a run. */
  readonly seed: number

  step(inputs: TypedInputs<I>): Promise<number>
  step(inputs: TypedInputs<I>, opts: { withCaptures: true }): Promise<StepResult>
  step(inputs: TypedInputs<I>, opts: { abortAsValue: true }): Promise<Outcome<{ loss: number }>>
  step(
    inputs: TypedInputs<I>,
    opts: { withCaptures: true; abortAsValue: true },
  ): Promise<Outcome<{ loss: number; captures: Captures }>>

  run(inputs: TypedInputs<I>): Promise<Float32Array>
  run(inputs: TypedInputs<I>, opts: { withCaptures: true }): Promise<RunResult>
  run(inputs: TypedInputs<I>, opts: { abortAsValue: true }): Promise<Outcome<{ output: Float32Array }>>
  run(
    inputs: TypedInputs<I>,
    opts: { withCaptures: true; abortAsValue: true },
  ): Promise<Outcome<{ output: Float32Array; captures: Captures }>>

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void>
  /** Read params back as a typed tree mirroring the model class structure.
   *  `params.layers[0].W` etc. — typed, autocompletable. */
  downloadParams(): Promise<ParamTree<M>>
  /** Escape hatch: read params back as a flat `{ 'layers.0.W': Float32Array, ... }`
   *  record. Useful for serialization, iteration over all params, or partial
   *  re-uploads via `uploadParams`. */
  downloadParamsFlat(): Promise<Record<string, Float32Array>>
  /** Gradients in the same tree shape as `downloadParams`. */
  downloadParamGrads(): Promise<ParamTree<M>>

  /** Re-initialize all params + zero optimizer state. */
  reset(): Promise<void>
  resetOptimizerState(): Promise<void>

  /** Update the learning rate at runtime, without recompiling. Works for
   *  both Adam and SGD graphs.
   *
   *  When `update.lr` is a non-constant schedule with no explicit
   *  `startStep`, the schedule is rebased so its step 1 aligns with the
   *  next training step ("decay from now"). Numbers and schedules with an
   *  explicit `startStep` pass through unchanged. */
  setOptimizerConfig(update: { lr?: LR }): Promise<void>

  /** Compile a sibling forward-only graph that shares this runtime's worker
   *  (and therefore its param GPUBuffers). Polymorphic by default — `null`
   *  dims in the input shapes resolve per-call. */
  compileForward<I2 extends InputDecls>(
    opts: CompileForwardMethodOptions<M, I2>,
  ): Promise<CompiledForwardModule<M, I2>>

  /** Swap the model topology in place: destroy the current training graph in
   *  the worker, compile a fresh one with the same loss/inputs config and
   *  the new factory. This handle remains valid (same object, same `I`
   *  generic, same `paramNames`/`kernelCount`/`ir` after the call). Sibling
   *  forward proxies created via `compileForward` stay registered: their
   *  per-shape kernel caches are cleared and recompile lazily on next `run()`.
   *
   *  Defaults to a *fresh* seed — replaceModel is for "different model now,"
   *  so fresh init is the natural expectation. Pass `{ seed }` to pin (for
   *  reproducible topology comparisons). Use the existing seed explicitly via
   *  `{ seed: compiled.seed }` if you want strict determinism across the swap.
   *
   *  Pass `{ adam }` or `{ sgd }` to update optimizer config atomically with
   *  the topology swap (must match the optimizer kind used at compileModule
   *  time). Without it, the existing optimizer config carries over. For
   *  mid-training LR changes without a topology swap, use `setOptimizerConfig`.
   *
   *  Use when the user changes layer count, hidden width, or any other
   *  shape-affecting model parameter — you don't need to re-create the
   *  worker or re-wire siblings. */
  replaceModel(
    newFactory: () => M,
    opts?: { seed?: number; adam?: AdamConfig; sgd?: SGDConfig },
  ): Promise<void>

  /** Tear down the worker + GPU resources. */
  destroy(): void
}

/** Returned by `compiled.compileForward({...})`. Polymorphic by default:
 *  `run()` at a new resolved shape lazily compiles + caches a sibling. Param
 *  reads/writes route to the parent training graph (shared buffers).
 *
 *  No sync inspection surface for kernel count / output shape / IR — those
 *  would lie on a polymorphic proxy that caches multiple shape variants. */
export interface CompiledForwardModule<M extends Module = Module, I extends InputDecls = InputDecls> {
  /** Same as the parent training graph's param names. */
  readonly paramNames: readonly string[]

  run(inputs: TypedInputs<I>): Promise<Float32Array>
  run(inputs: TypedInputs<I>, opts: { withCaptures: true }): Promise<RunResult>
  run(inputs: TypedInputs<I>, opts: { abortAsValue: true }): Promise<Outcome<{ output: Float32Array }>>
  run(
    inputs: TypedInputs<I>,
    opts: { withCaptures: true; abortAsValue: true },
  ): Promise<Outcome<{ output: Float32Array; captures: Captures }>>

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void>
  /** Typed tree view of the shared param state — same shape as the parent
   *  training graph's `downloadParams()` (params are physically shared). */
  downloadParams(): Promise<ParamTree<M>>
  /** Escape hatch: flat `{ 'layers.0.W': Float32Array, ... }` record. */
  downloadParamsFlat(): Promise<Record<string, Float32Array>>

  destroy(): void
}

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
  const optsWithSeed: CompileModuleOptions<M, I> = { ...opts, seed: opts.seed ?? randomSeed() }
  try {
    const built = await buildTrainingGraph(proxy, optsWithSeed, 0)
    return new CompiledModuleProxy<M, I>(
      proxy, 0, built.ir, built.meta, optsWithSeed, built.initFns, { v: 1 },
    )
  } catch (e) {
    proxy.terminate()
    throw e
  }
}

/** Fresh 32-bit integer seed for compile runs that didn't pass one. Exposed
 *  on the returned `CompiledModule.seed` so users can reproduce. */
function randomSeed(): number {
  return (Math.random() * 0x100000000) >>> 0
}

/** True when this environment has WebGPU. Use as a friendly gate before
 *  `compileModule` so you can surface a "WebGPU required" message rather
 *  than crash deep inside the worker. */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator
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
  if (opts.adam && opts.sgd) {
    throw new Error('compileModule: pass either `adam` or `sgd`, not both')
  }
  const loss = opts.loss as unknown as ForwardFn<M, InputDecls>
  const inputs = opts.inputs as InputDecls
  const { graph, materialized } = traceModule(opts.factory(), loss, inputs)
  const { paramGrads, loss: lossTensor } = appendGrad(graph)
  const adamResult = opts.adam
    ? appendAdam(graph, paramGrads, materialized.tensors, opts.adam, materialized.decayFlags)
    : undefined
  const sgdResult = opts.sgd
    ? appendSGD(graph, paramGrads, materialized.tensors, opts.sgd, materialized.decayFlags)
    : undefined

  const optimizerWritebacks =
    adamResult?.writebacks ?? sgdResult?.writebacks ?? []
  const plan = planBuffers(graph, paramGrads, optimizerWritebacks)
  const kernels = emitKernels(graph, plan)
  const ir: CompiledIR = { graph, paramGrads, loss: lossTensor, plan, kernels }

  const initialParams = buildInitialParams(plan, materialized.initFns, mulberry32(opts.seed!))
  const wireIR: WireIR = { graph, plan, kernels }
  const wireOptimizer: WireOptimizerConfig | null =
    adamResult ? { kind: 'adam', config: wireAdamConfig(adamResult) }
    : sgdResult ? { kind: 'sgd', config: wireSGDConfig(sgdResult) }
    : null
  const transfers = transferablesOfRecord(initialParams)

  const meta = await proxy.request<CreateRuntimeResult>(
    { kind: 'createRuntime', payload: { graphId, ir: wireIR, initialParams, optimizer: wireOptimizer } },
    transfers,
  )
  return { ir, meta, initFns: materialized.initFns }
}

/** Implementation-side input type. Public surface narrows this to
 *  `TypedInputs<I>` per-dtype; method bodies don't know `I`. */
type LooseInputs = Record<string, Int32Array | Float32Array>

/** Non-generic supertype so CompiledModuleProxy can hold heterogeneous
 *  `ForwardProxy<M, I_k>` instances in a single Set. */
interface ChildProxy {
  _invalidateForReplace(): void
  _destroyInternal(): void
}

/** Child's view of its parent training graph. */
interface ParentRef<M extends Module> {
  readonly graphId: number
  readonly paramNames: readonly string[]
  currentFactory(): M
}

class CompiledModuleProxy<M extends Module, I extends InputDecls> implements CompiledModule<M, I> {
  /** Forward proxies created via `compileForward` — tracked so `destroy()`
   *  can clean them up cascade-style, and `replaceModel()` can invalidate
   *  their per-shape kernel caches without unregistering them. */
  private readonly children = new Set<ChildProxy>()

  // Swapped in place by `replaceModel` so callers' references (and any
  // sibling ForwardProxy holding `this`) stay valid across topology changes.
  // `opts.seed` is always populated (compileModule fills it before construction).
  ir: CompiledIR
  private meta: CreateRuntimeResult
  private opts: CompileModuleOptions<M, I>
  private initFns: Record<string, InitFn>

  constructor(
    private readonly proxy: WorkerProxy,
    readonly graphId: number,
    ir: CompiledIR,
    meta: CreateRuntimeResult,
    opts: CompileModuleOptions<M, I>,
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
  get seed(): number { return this.opts.seed! }

  /** Sibling ForwardProxies read this through the proxy ref so they always
   *  re-trace against the latest topology. */
  currentFactory(): M { return this.opts.factory() }

  step(inputs: LooseInputs): Promise<number>
  step(inputs: LooseInputs, opts: { withCaptures: true }): Promise<StepResult>
  step(inputs: LooseInputs, opts: { abortAsValue: true }): Promise<Outcome<{ loss: number }>>
  step(
    inputs: LooseInputs,
    opts: { withCaptures: true; abortAsValue: true },
  ): Promise<Outcome<{ loss: number; captures: Captures }>>
  async step(
    inputs: LooseInputs,
    opts?: { withCaptures?: boolean; abortAsValue?: boolean },
  ): Promise<number | StepResult | Outcome<{ loss: number }> | Outcome<{ loss: number; captures: Captures }>> {
    try {
      const r = await this.proxy.request<StepResultWire>(
        { kind: 'step', payload: { graphId: this.graphId, inputs, withCaptures: opts?.withCaptures === true } },
      )
      if (opts?.withCaptures) {
        const captures = makeCaptures(r.captures, this.meta.captureShapes)
        return opts.abortAsValue === true
          ? { kind: 'ok', loss: r.loss, captures }
          : { loss: r.loss, captures }
      }
      return opts?.abortAsValue === true ? { kind: 'ok', loss: r.loss } : r.loss
    } catch (e) {
      if (opts?.abortAsValue === true && (e as { name?: string })?.name === 'AbortError') {
        return { kind: 'aborted' }
      }
      throw e
    }
  }

  run(inputs: LooseInputs): Promise<Float32Array>
  run(inputs: LooseInputs, opts: { withCaptures: true }): Promise<RunResult>
  run(inputs: LooseInputs, opts: { abortAsValue: true }): Promise<Outcome<{ output: Float32Array }>>
  run(
    inputs: LooseInputs,
    opts: { withCaptures: true; abortAsValue: true },
  ): Promise<Outcome<{ output: Float32Array; captures: Captures }>>
  async run(
    inputs: LooseInputs,
    opts?: { withCaptures?: boolean; abortAsValue?: boolean },
  ): Promise<Float32Array | RunResult | Outcome<{ output: Float32Array }> | Outcome<{ output: Float32Array; captures: Captures }>> {
    try {
      const r = await this.proxy.request<RunResultWire>(
        { kind: 'run', payload: { graphId: this.graphId, inputs, withCaptures: opts?.withCaptures === true } },
      )
      if (opts?.withCaptures) {
        const captures = makeCaptures(r.captures, this.meta.captureShapes)
        return opts.abortAsValue === true
          ? { kind: 'ok', output: r.output, captures }
          : { output: r.output, captures }
      }
      return opts?.abortAsValue === true ? { kind: 'ok', output: r.output } : r.output
    } catch (e) {
      if (opts?.abortAsValue === true && (e as { name?: string })?.name === 'AbortError') {
        return { kind: 'aborted' }
      }
      throw e
    }
  }

  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'uploadParams', payload: { graphId: this.graphId, params, partial: !!opts?.partial } },
    ).then(() => undefined)
  }

  async downloadParams(): Promise<ParamTree<M>> {
    return buildParamTree(await this.fetchParams('downloadParams')) as ParamTree<M>
  }

  downloadParamsFlat(): Promise<Record<string, Float32Array>> {
    return this.fetchParams('downloadParams')
  }

  async downloadParamGrads(): Promise<ParamTree<M>> {
    return buildParamTree(await this.fetchParams('downloadParamGrads')) as ParamTree<M>
  }

  private async fetchParams(kind: 'downloadParams' | 'downloadParamGrads'): Promise<Record<string, Float32Array>> {
    const r = await this.proxy.request<DownloadParamsResult>(
      { kind, payload: { graphId: this.graphId } },
    )
    return r.params
  }

  async reset(): Promise<void> {
    const initialParams = buildInitialParams(this.ir.plan, this.initFns, mulberry32(this.opts.seed!))
    await Promise.all([this.uploadParams(initialParams), this.resetOptimizerState()])
  }

  resetOptimizerState(): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'resetOptimizer', payload: { graphId: this.graphId } },
    ).then(() => undefined)
  }

  setOptimizerConfig(update: { lr?: LR }): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'setOptimizerConfig', payload: { graphId: this.graphId, update } },
    ).then(() => undefined)
  }

  async compileForward<I2 extends InputDecls>(
    opts: CompileForwardMethodOptions<M, I2>,
  ): Promise<CompiledForwardModule<M, I2>> {
    const child: ForwardProxy<M, I2> = new ForwardProxy<M, I2>(
      this.proxy,
      this,
      opts.forward,
      normalizeDecls(opts.inputs),
      this.nextGraphId,
      () => this.children.delete(child),
    )
    this.children.add(child)
    return child
  }

  async replaceModel(
    newFactory: () => M,
    replaceOpts?: { seed?: number; adam?: AdamConfig; sgd?: SGDConfig },
  ): Promise<void> {
    if (replaceOpts?.adam && replaceOpts?.sgd) {
      throw new Error('replaceModel: pass either `adam` or `sgd`, not both')
    }
    // Invalidate (don't destroy) siblings — their kernels are model-specific
    // but the proxy objects must outlive the swap so callers' references stay
    // valid. Each sibling recompiles lazily on its next run().
    for (const child of this.children) child._invalidateForReplace()
    this.proxy.send({ kind: 'destroy', payload: { graphId: this.graphId } })
    const newOpts: CompileModuleOptions<M, I> = {
      ...this.opts,
      factory: newFactory,
      seed: replaceOpts?.seed ?? randomSeed(),
      ...(replaceOpts?.adam !== undefined ? { adam: replaceOpts.adam } : {}),
      ...(replaceOpts?.sgd !== undefined ? { sgd: replaceOpts.sgd } : {}),
    }
    const built = await buildTrainingGraph(this.proxy, newOpts, this.graphId)
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
class ForwardProxy<M extends Module, I extends InputDecls>
  implements CompiledForwardModule<M, I>, ChildProxy
{
  private readonly cache = new Map<string, ForwardSiblingMeta>()

  constructor(
    private readonly proxy: WorkerProxy,
    private readonly parent: ParentRef<M>,
    private readonly forward: ForwardFn<M, I>,
    private readonly decls: NormalizedDecls,
    private readonly nextGraphId: { v: number },
    private readonly onDestroy: () => void,
  ) {}

  get paramNames(): readonly string[] { return this.parent.paramNames }

  private async siblingFor(inputs: LooseInputs): Promise<ForwardSiblingMeta> {
    const resolved = resolveDecls(this.decls, inputs)
    const key = shapeKey(resolved)
    const hit = this.cache.get(key)
    if (hit) return hit
    const sib = await compileSibling<M, I>(
      this.proxy, this.parent.graphId, this.parent.currentFactory(), this.forward,
      resolved, this.nextGraphId,
    )
    this.cache.set(key, sib)
    return sib
  }

  run(inputs: LooseInputs): Promise<Float32Array>
  run(inputs: LooseInputs, opts: { withCaptures: true }): Promise<RunResult>
  run(inputs: LooseInputs, opts: { abortAsValue: true }): Promise<Outcome<{ output: Float32Array }>>
  run(
    inputs: LooseInputs,
    opts: { withCaptures: true; abortAsValue: true },
  ): Promise<Outcome<{ output: Float32Array; captures: Captures }>>
  async run(
    inputs: LooseInputs,
    opts?: { withCaptures?: boolean; abortAsValue?: boolean },
  ): Promise<Float32Array | RunResult | Outcome<{ output: Float32Array }> | Outcome<{ output: Float32Array; captures: Captures }>> {
    try {
      const sib = await this.siblingFor(inputs)
      const r = await this.proxy.request<RunResultWire>(
        { kind: 'run', payload: { graphId: sib.graphId, inputs, withCaptures: opts?.withCaptures === true } },
      )
      if (opts?.withCaptures) {
        const captures = makeCaptures(r.captures, sib.meta.captureShapes)
        return opts.abortAsValue === true
          ? { kind: 'ok', output: r.output, captures }
          : { output: r.output, captures }
      }
      return opts?.abortAsValue === true ? { kind: 'ok', output: r.output } : r.output
    } catch (e) {
      if (opts?.abortAsValue === true && (e as { name?: string })?.name === 'AbortError') {
        return { kind: 'aborted' }
      }
      throw e
    }
  }

  async uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): Promise<void> {
    // Params live on the parent (shared with all siblings).
    await this.proxy.request<null>(
      { kind: 'uploadParams', payload: { graphId: this.parent.graphId, params, partial: !!opts?.partial } },
    )
  }

  async downloadParams(): Promise<ParamTree<M>> {
    return buildParamTree(await this.fetchParams()) as ParamTree<M>
  }

  downloadParamsFlat(): Promise<Record<string, Float32Array>> {
    return this.fetchParams()
  }

  private async fetchParams(): Promise<Record<string, Float32Array>> {
    const r = await this.proxy.request<DownloadParamsResult>(
      { kind: 'downloadParams', payload: { graphId: this.parent.graphId } },
    )
    return r.params
  }

  destroy(): void {
    this._destroyInternal()
    this.onDestroy()
  }

  // Destroy without unregistering from parent — used when the parent is
  // tearing itself down and will clear its children set after iteration.
  _destroyInternal(): void {
    this._invalidateForReplace()
  }

  // Drop per-shape kernel caches after a parent topology swap. The proxy
  // object stays alive; next run() recompiles against the new model.
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
async function compileSibling<M extends Module, I extends InputDecls>(
  proxy: WorkerProxy,
  parentGraphId: number,
  model: M,
  forward: ForwardFn<M, I>,
  decls: ResolvedDecls,
  nextGraphId: { v: number },
): Promise<ForwardSiblingMeta> {
  const { graph } = traceModule(model, forward as ForwardFn<M, InputDecls>, decls)
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

type Graph = ReturnType<typeof trace>

/** Normalized inputs decl: concrete dtype, shape with possibly-null wildcards. */
type NormalizedDecls = Record<string, { shape: InputShape; dtype: Dtype }>

/** Fully-resolved inputs decl: concrete dtype + shape (no nulls). */
type ResolvedDecls = Record<string, { shape: Shape; dtype: Dtype }>

function normalizeDecl(d: InputDecl): { shape: InputShape; dtype: Dtype } {
  if (Array.isArray(d)) return { shape: d as InputShape, dtype: 'f32' }
  const obj = d as InputDeclObject
  return { shape: obj.shape, dtype: obj.dtype }
}

function normalizeDecls(decls: InputDecls): NormalizedDecls {
  const out: NormalizedDecls = {}
  for (const [k, v] of Object.entries(decls)) out[k] = normalizeDecl(v)
  return out
}

// Catches a factory that returns the same instance twice: ParamSentinel
// fields are mutated into Tensors on first compile, so a reused instance
// would silently corrupt later compiles. WeakSet — does not retain.
const seenModels = new WeakSet<Module>()

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

  // All wildcards in a single call must resolve to the same value.
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

function buildInitialParams(
  plan: BufferPlan,
  initFns: Record<string, InitFn>,
  rng: Rng,
): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {}
  for (const [name, bufId] of plan.paramsByName) {
    const shape = plan.buffers[bufId]!.shape
    const size = shape.reduce((a, b) => a * b, 1)
    const initFn = initFns[name]
    if (!initFn) throw new Error(`compile: no init for param '${name}'`)
    out[name] = initFn(size, shape, rng)
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

function wireSGDConfig(r: SGDResult): WireSGDConfig {
  const c: SGDResolvedConfig = r.config
  return {
    lr: c.lr,
    momentum: c.momentum,
    nesterov: c.nesterov,
    weightDecay: c.weightDecay,
    lrIsScheduled: c.lrIsScheduled,
    lrInputName: r.lrInputName,
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

/** Inflate a flat `{ 'layers.0.W': ..., ... }` record into a tree mirroring
 *  the Module class structure. Numeric path segments become array indices. */
function buildParamTree(flat: Record<string, Float32Array>): Record<string, unknown> {
  const tree: Record<string, unknown> = {}
  for (const [name, value] of Object.entries(flat)) {
    setDeep(tree, name.split('.'), value)
  }
  return tree
}

function setDeep(root: Record<string, unknown>, parts: string[], value: Float32Array): void {
  let cur: any = root
  for (let i = 0; i < parts.length; i++) {
    const seg = parts[i]!
    const segIsIdx = /^\d+$/.test(seg)
    const key: string | number = segIsIdx ? parseInt(seg, 10) : seg
    if (i === parts.length - 1) { cur[key] = value; return }
    const next = parts[i + 1]!
    const nextIsIdx = /^\d+$/.test(next)
    if (cur[key] === undefined) cur[key] = nextIsIdx ? [] : {}
    cur = cur[key]
  }
}
