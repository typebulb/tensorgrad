// Top-level compile pipeline:
//   trainingSpec() / forwardSpec()  — pure value builders, no side effects
//   compile()                       — takes a TrainingSpec, spawns a worker
//   train.attach(spec)       — attach a ForwardSpec as a sibling
//                                     (shares worker + param buffers)
//   replaceModel                    — in-place topology swap on a training
//                                     compile; attached forward children
//                                     invalidate their per-shape caches
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
  type AdamConfig, type AdamWConfig, type AdamResult, type AdamResolvedConfig, type LR,
} from './adam.js'
import { appendSGD, type SGDConfig, type SGDResult, type SGDResolvedConfig } from './sgd.js'
import { planBuffers, type BufferPlan } from './buffers.js'
import { emitKernels, type KernelSpec } from './codegen.js'
import { Captures } from './runtime.js'
import { Module, materializeParams, cloneModule, mulberry32, type MaterializedParams, type Rng, type InitFn } from './module.js'
import { WorkerProxy } from './worker-proxy.js'
import {
  transferablesOfRecord,
  type Req, type WireIR, type WireAdamConfig, type WireSGDConfig, type WireOptimizerConfig,
  type CompileResult,
  type StepResultWire, type RunResultWire, type DownloadParamsResult, type ReadLossResult,
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

/** Discriminated result of `compiled.step(...)`. `'completed'` carries the
 *  scalar loss plus a `Captures` instance (empty when the graph has no
 *  `capture(...)` sites). `'aborted'` is returned when the graph was
 *  destroyed mid-flight (e.g. by `replaceModel`); no try/catch needed. */
export type StepResult =
  | { kind: 'completed'; loss: number; captures: Captures }
  | { kind: 'aborted' }

/** Discriminated result of `compiled.run(...)`. Same shape as `StepResult`
 *  but `'completed'` carries the full output tensor (not just a scalar). */
export type RunResult =
  | { kind: 'completed'; output: Float32Array; captures: Captures }
  | { kind: 'aborted' }

/** The compile pipeline's IR bundle: the augmented graph, per-param
 *  gradient tensors, the loss output, the buffer plan, and the emitted
 *  kernel specs. Surfaced via `CompiledForward.graphFor(inputs)` for
 *  per-shape inspection. On a runtime handle, `compiled.graph` and
 *  `compiled.kernels` surface the equivalent fields. */
export interface CompiledIR {
  graph: GradResult['graph']
  paramGrads: GradResult['paramGrads']
  loss: Tensor
  plan: BufferPlan
  kernels: KernelSpec[]
}

/** Optimizer config — discriminated by `kind`. `'adam'` is plain Adam (no
 *  weight decay); `'adamw'` is decoupled-decay AdamW (requires `weightDecay`).
 *  `'sgd'` is SGD / momentum / Nesterov. Splits mirror PyTorch's
 *  `torch.optim.Adam` vs `torch.optim.AdamW`. */
export type OptimizerConfig =
  | ({ readonly kind: 'adam'  } & AdamConfig)
  | ({ readonly kind: 'adamw' } & AdamWConfig)
  | ({ readonly kind: 'sgd'   } & SGDConfig)

/** Options passed to `trainingSpec({ ... })`. */
export interface TrainingSpecOptions<M extends Module, I extends InputDecls = InputDecls> {
  /** A model instance — `new Model()`. The compile pipeline clones this
   *  before tracing, so the caller's instance is never mutated and can be
   *  reused across multiple compiles (polymorphic siblings, replaceModel,
   *  etc.). */
  model: M
  /** Forward function that returns the scalar loss tensor. */
  loss: ForwardFn<M, I>
  /** Input shape declarations (one per named tensor input). */
  inputs: I
  /** Optimizer. Discriminated by `kind: 'adam' | 'sgd'`. */
  optimizer: OptimizerConfig
  /** 32-bit integer seed for the param-init RNG. Same seed + same model
   *  topology → identical initial params, every time. If omitted, a seed
   *  is generated and exposed as `compiled.seed` so you can reproduce a
   *  run by passing it back. */
  seed?: number
}

/** Options passed to `forwardSpec({ ... })`. Forward specs compile against
 *  a parent training compile (via `train.attach(forwardSpec)`) and
 *  share its param buffers — every training-step update is immediately
 *  visible. */
export interface ForwardSpecOptions<M extends Module, I extends InputDecls = InputDecls> {
  /** A model instance with the same param tree as the parent training
   *  spec. The forward function reads those params but is otherwise
   *  independent. Cloned per trace, like the training spec's model. */
  model: M
  /** Forward function returning the output tensor (one per shape value). */
  forward: ForwardFn<M, I>
  /** Input shape declarations. `null` dims become parametric; the
   *  compile proxy caches a sibling per distinct resolved shape on first
   *  `run()` at that shape. */
  inputs: I
}

/** A training spec value: pure data, no side effects, no worker. Built by
 *  `trainingSpec({ model, loss, inputs, optimizer })`. Consumed by `compile()`. */
export interface TrainingSpec<M extends Module = Module, I extends InputDecls = InputDecls>
  extends TrainingSpecOptions<M, I> {
  readonly kind: 'training'
}

/** A forward spec value: pure data, no side effects, no worker. Built by
 *  `forwardSpec({ model, forward, inputs })`. Pass to
 *  `train.attach(spec)` to attach a sibling inference graph
 *  to an existing training compile. */
export interface ForwardSpec<M extends Module = Module, I extends InputDecls = InputDecls>
  extends ForwardSpecOptions<M, I> {
  readonly kind: 'forward'
}

/** Discriminated union of the two spec shapes. */
export type Spec<M extends Module = Module, I extends InputDecls = InputDecls> =
  | TrainingSpec<M, I>
  | ForwardSpec<M, I>

/** Build a training spec — plain data passed to `compile()`.
 *
 *  ```ts
 *  const train = await compile(trainingSpec({ model: new MLP(), loss, inputs, optimizer }))
 *  ``` */
export function trainingSpec<M extends Module, I extends InputDecls>(
  opts: TrainingSpecOptions<M, I>,
): TrainingSpec<M, I> {
  return { ...opts, kind: 'training' }
}

/** Build a forward spec — plain data passed to `train.attach()`.
 *
 *  ```ts
 *  const infer = await train.attach(forwardSpec({ model, forward: predictFn, inputs }))
 *  ``` */
export function forwardSpec<M extends Module, I extends InputDecls>(
  opts: ForwardSpecOptions<M, I>,
): ForwardSpec<M, I> {
  return { ...opts, kind: 'forward' }
}


/** Returned by `compile(trainingSpec)`. Proxies all GPU work to an
 *  internal worker; every method returns a Promise. Generic over the
 *  declared inputs shape `I` so `step` / `run` accept inputs with the
 *  right TypedArray per dtype. */
export interface CompiledTraining<M extends Module, I extends InputDecls = InputDecls> {
  /** The compiled IR graph: ops, tensors, connectivity, captures, outputs.
   *  Walk it to inspect what `compile` produced (see README). Swapped
   *  in place by `replaceModel`. */
  readonly graph: GradResult['graph']
  /** Emitted WGSL kernels (one per dispatch). `kernels.length` is the kernel
   *  count surfaced for status displays. */
  readonly kernels: readonly KernelSpec[]
  /** Shape of the loss output (always `[]` for a training graph — loss is
   *  required to be scalar). Exposed for symmetry with forward graphs. */
  readonly outputShape: readonly number[]
  /** Names of the model's parameters, in materialization order. */
  readonly paramNames: readonly string[]
  /** The actual seed used for param init (either the one you passed, or a
   *  freshly-generated one if you didn't). Pass this back as
   *  back via `trainingSpec({ seed })` to reproduce a run. */
  readonly seed: number

  /** One full forward + backward + optimizer step. Always reads back the
   *  scalar loss plus any registered captures; the result discriminator
   *  distinguishes successful completion from a mid-flight abort (graph
   *  destroyed, typically by `replaceModel`). */
  step(inputs: TypedInputs<I>): Promise<StepResult>

  /** Submit a training step without awaiting the loss readback. Each
   *  loss `mapAsync` costs ~1 ms on desktop but 10–30 ms on Android
   *  Chrome; on mobile, `queueStep` + occasional `readLoss()` keeps the
   *  main thread responsive while training runs at GPU speed.
   *
   *  Fire-and-forget by design: returns `Promise<void>`. If the graph is
   *  destroyed mid-flight (`replaceModel` / `destroy`), the call resolves
   *  silently — the loop that submitted it has already exited. */
  queueStep(inputs: TypedInputs<I>): Promise<void>

  /** Read the most recent step's loss. Pair with `queueStep`. */
  readLoss(): Promise<number>

  /** Same dispatch as `step` but returns the full output tensor. For
   *  training graphs the output is a scalar loss, so `step` is usually
   *  more convenient. */
  run(inputs: TypedInputs<I>): Promise<RunResult>

  /** Upload params from a flat record (the shape `downloadParamsFlat` returns).
   *  Partial by default — missing keys leave existing GPU values unchanged.
   *  Unknown keys throw (typo guard). */
  uploadParams(params: Record<string, Float32Array>): Promise<void>
  /** Read params back as a typed tree mirroring the model class structure.
   *  `params.layers[0].W` etc. — typed, autocompletable. Mirror of
   *  `downloadParamsFlat` and `downloadParamGrads` — same underlying data,
   *  three views; pick whichever matches your call site. */
  downloadParams(): Promise<ParamTree<M>>
  /** Escape hatch: read params back as a flat `{ 'layers.0.W': Float32Array, ... }`
   *  record. The natural feed for `uploadParams` (round-trip works without
   *  any reshaping); also handy for serialization or iterating all params. */
  downloadParamsFlat(): Promise<Record<string, Float32Array>>
  /** Gradients in the same tree shape as `downloadParams`. Mirror of the
   *  params-side download; same per-tensor sizes. */
  downloadParamGrads(): Promise<ParamTree<M>>

  /** Re-initialize all params (from `seed`) and zero optimizer state in one
   *  call. The "start over with the same compile" button. For just zeroing
   *  optimizer state while keeping params, use `resetOptimizerState`. */
  reset(): Promise<void>
  /** Zero Adam's m/v buffers (or SGD's momentum buffer). Params untouched.
   *  Reach for this when you want to discard accumulated optimizer state
   *  without re-initializing params (e.g. after a hyperparameter change). */
  resetOptimizerState(): Promise<void>

  /** Update the learning rate at runtime, without recompiling. Works for
   *  both Adam and SGD graphs.
   *
   *  When `lr` is a non-constant schedule with no explicit `startStep`, the
   *  schedule is rebased so its step 1 aligns with the next training step
   *  ("decay from now"). Numbers and schedules with an explicit `startStep`
   *  pass through unchanged. */
  setLR(lr: LR): Promise<void>

  /** Swap the model topology in place: destroy the current training graph in
   *  the worker, compile a fresh one with the same loss/inputs config and
   *  the new model instance. This handle remains valid (same object, same
   *  `I` generic, same `paramNames`/`graph` after the call). Sibling
   *  forward compiles attached via `this.attach(forwardSpec)` stay
   *  registered: their per-shape kernel caches are cleared and recompile
   *  lazily on next `run()`.
   *
   *  Defaults to a *fresh* seed — replaceModel is for "different model now,"
   *  so fresh init is the natural expectation. Pass `{ seed }` to pin (for
   *  reproducible topology comparisons). Use the existing seed explicitly via
   *  `{ seed: compiled.seed }` if you want strict determinism across the swap.
   *
   *  Pass `{ optimizer }` to update optimizer config atomically with the
   *  topology swap. Without it, the existing optimizer config carries over.
   *  For mid-training LR changes without a topology swap, use `setLR`.
   *
   *  Use when the user changes layer count, hidden width, or any other
   *  shape-affecting model parameter — you don't need to re-create the
   *  worker or re-wire siblings. */
  replaceModel(
    newModel: M,
    opts?: { seed?: number; optimizer?: OptimizerConfig },
  ): Promise<void>

  /** Attach a forward (inference) spec as a sibling — shares this training
   *  compile's worker and param GPUBuffers. Cascading destroy: when this
   *  training compile is destroyed (or `replaceModel`-d), attached forward
   *  proxies are cleaned up / invalidated automatically.
   *
   *  Returned `CompiledForward` proxy is polymorphic over `null`-wildcard
   *  inputs: per-shape kernels are compiled lazily and cached on first
   *  `run()` at each new resolved shape. */
  attach<I2 extends InputDecls>(s: ForwardSpec<M, I2>): Promise<CompiledForward<M, I2>>

  /** Tear down the worker + GPU resources, plus any attached forward
   *  compiles. */
  destroy(): void
}

/** Returned by `train.attach(forwardSpec)`. Polymorphic by
 *  default: `run()` at a new resolved shape lazily compiles + caches a
 *  sibling. Param reads/writes route to the parent training graph
 *  (shared buffers).
 *
 *  No top-level `graph` / `kernels`: forward proxies are polymorphic and
 *  hold one IR per shape. Use `graphFor(inputs)` to fetch (and lazily
 *  compile) the IR for a specific shape. */
export interface CompiledForward<M extends Module = Module, I extends InputDecls = InputDecls> {
  /** Same as the parent training graph's param names. */
  readonly paramNames: readonly string[]

  /** Run the forward dispatch. Returns the discriminated `RunResult`:
   *  `'completed'` with the output tensor + any captures, or `'aborted'`
   *  if the parent training graph was destroyed mid-flight. */
  run(inputs: TypedInputs<I>): Promise<RunResult>

  /** The compiled IR for the resolved shape of `inputs`. Compiles + caches
   *  a sibling on first call for that shape (same lazy-compile behavior as
   *  `run()`). Use for inspecting the kernel count / IR ops at the actual
   *  shape you're running. */
  graphFor(inputs: TypedInputs<I>): Promise<CompiledIR>

  /** Upload params. Partial by default — missing keys leave existing values
   *  unchanged. Reads/writes go to the parent training compile's buffers
   *  (shared, so updates are immediately visible there too). */
  uploadParams(params: Record<string, Float32Array>): Promise<void>
  /** Typed tree view of the shared param state — identical to the parent
   *  training graph's `downloadParams()` since params are physically
   *  shared. Mirror of `downloadParamsFlat`. */
  downloadParams(): Promise<ParamTree<M>>
  /** Flat `{ 'layers.0.W': Float32Array, ... }` record — the natural feed
   *  for `uploadParams`. */
  downloadParamsFlat(): Promise<Record<string, Float32Array>>

  destroy(): void
}

/**
 * Compile a `TrainingSpec` to a worker-backed runtime — spawns a worker,
 * owns the param GPUBuffers, and returns a `CompiledTraining` handle.
 *
 * ```ts
 * const model = new MLP()
 * const train = await compile(trainingSpec({ model, loss, inputs, optimizer }))
 * ```
 *
 * For forward (inference) compiles that share the training graph's params
 * and worker, use `train.attach(forwardSpec({ ... }))` — the
 * lifecycle relationship is named at the call site.
 */
export async function compile<M extends Module, I extends InputDecls>(
  s: TrainingSpec<M, I>,
): Promise<CompiledTraining<M, I>> {
  return compileTrainingSpec(s)
}

async function compileTrainingSpec<M extends Module, I extends InputDecls>(
  s: TrainingSpec<M, I>,
): Promise<CompiledTraining<M, I>> {
  const proxy = new WorkerProxy(__WORKER_SOURCE__)
  const specWithSeed: TrainingSpec<M, I> = { ...s, seed: s.seed ?? randomSeed() }
  try {
    const built = await buildTrainingGraph(proxy, specWithSeed, 0)
    return new CompiledTrainingProxy<M, I>(
      proxy, 0, built.ir, built.meta, specWithSeed, built.initFns, { v: 1 },
    )
  } catch (e) {
    proxy.terminate()
    throw e
  }
}

/** Fresh 32-bit integer seed for compile runs that didn't pass one. Exposed
 *  on the returned `CompiledTraining.seed` so users can reproduce. */
function randomSeed(): number {
  return (Math.random() * 0x100000000) >>> 0
}

/** True when this environment has WebGPU. Use as a friendly gate before
 *  `compile` so you can surface a "WebGPU required" message rather than
 *  crash deep inside the worker. */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator
}

interface BuiltTrainingGraph {
  ir: CompiledIR
  meta: CompileResult
  initFns: Record<string, InitFn>
}

// Trace + autograd + optimizer + buffer-plan + codegen + worker createRuntime,
// using an existing worker proxy. Shared by `compileTrainingSpec` (fresh worker)
// and `replaceModel` (existing worker, same graphId).
async function buildTrainingGraph<M extends Module, I extends InputDecls>(
  proxy: WorkerProxy,
  s: TrainingSpec<M, I>,
  graphId: number,
): Promise<BuiltTrainingGraph> {
  const loss = s.loss as unknown as ForwardFn<M, InputDecls>
  const inputs = s.inputs as InputDecls
  const { graph, materialized } = traceModule(s.model, loss, inputs)
  const { paramGrads, loss: lossTensor } = appendGrad(graph)
  const adamResult = s.optimizer.kind === 'adam' || s.optimizer.kind === 'adamw'
    ? appendAdam(graph, paramGrads, materialized.tensors, s.optimizer, materialized.decayFlags)
    : undefined
  const sgdResult = s.optimizer.kind === 'sgd'
    ? appendSGD(graph, paramGrads, materialized.tensors, s.optimizer, materialized.decayFlags)
    : undefined

  const optimizerWritebacks =
    adamResult?.writebacks ?? sgdResult?.writebacks ?? []
  const plan = planBuffers(graph, paramGrads, optimizerWritebacks)
  const kernels = emitKernels(graph, plan)
  const ir: CompiledIR = { graph, paramGrads, loss: lossTensor, plan, kernels }

  const initialParams = buildInitialParams(plan, materialized.initFns, mulberry32(s.seed!))
  const wireIR: WireIR = { graph, plan, kernels }
  const wireOptimizer: WireOptimizerConfig | null =
    adamResult ? { kind: 'adam', config: wireAdamConfig(adamResult) }
    : sgdResult ? { kind: 'sgd', config: wireSGDConfig(sgdResult) }
    : null
  const transfers = transferablesOfRecord(initialParams)

  const meta = await proxy.request<CompileResult>(
    { kind: 'createRuntime', payload: { graphId, ir: wireIR, initialParams, optimizer: wireOptimizer } },
    transfers,
  )
  return { ir, meta, initFns: materialized.initFns }
}

/** Implementation-side input type. Public surface narrows this to
 *  `TypedInputs<I>` per-dtype; method bodies don't know `I`. */
type LooseInputs = Record<string, Int32Array | Float32Array>

/** Non-generic supertype so CompiledTrainingProxy can hold heterogeneous
 *  `ForwardProxy<M, I_k>` instances in a single Set. */
interface ChildProxy {
  _invalidateForReplace(): void
  _destroyInternal(): void
}

/** Child's view of its parent training graph. */
interface ParentRef<M extends Module> {
  readonly graphId: number
  readonly paramNames: readonly string[]
  currentModel(): M
}

class CompiledTrainingProxy<M extends Module, I extends InputDecls> implements CompiledTraining<M, I> {
  /** Forward children attached via `this.attach(forwardSpec)`.
   *  Tracked so `destroy()` can clean them up cascade-style and
   *  `replaceModel()` can invalidate their per-shape kernel caches without
   *  unregistering them. */
  private readonly children = new Set<ChildProxy>()

  // Swapped in place by `replaceModel` so callers' references (and any
  // sibling ForwardProxy holding `this`) stay valid across topology changes.
  // `spec.seed` is always populated (compileTrainingSpec fills it).
  private _ir: CompiledIR
  private meta: CompileResult
  private spec: TrainingSpec<M, I>
  private initFns: Record<string, InitFn>

  constructor(
    private readonly proxy: WorkerProxy,
    readonly graphId: number,
    ir: CompiledIR,
    meta: CompileResult,
    spec: TrainingSpec<M, I>,
    initFns: Record<string, InitFn>,
    private readonly nextGraphId: { v: number },
  ) {
    this._ir = ir
    this.meta = meta
    this.spec = spec
    this.initFns = initFns
  }

  get graph(): GradResult['graph'] { return this._ir.graph }
  get kernels(): readonly KernelSpec[] { return this._ir.kernels }
  get outputShape(): readonly number[] { return this.meta.outputShape }
  get paramNames(): readonly string[] { return this.meta.paramNames }
  get seed(): number { return this.spec.seed! }

  /** Sibling ForwardProxies read this through the proxy ref so they always
   *  re-trace against the latest topology. */
  currentModel(): M { return this.spec.model }

  async step(inputs: LooseInputs): Promise<StepResult> {
    try {
      const r = await this.proxy.request<StepResultWire>(
        { kind: 'step', payload: { graphId: this.graphId, inputs } },
      )
      return { kind: 'completed', loss: r.loss, captures: makeCaptures(r.captures, this.meta.captureShapes) }
    } catch (e) {
      if ((e as { name?: string })?.name === 'AbortError') return { kind: 'aborted' }
      throw e
    }
  }

  async queueStep(inputs: LooseInputs): Promise<void> {
    try {
      await this.proxy.request<null>({ kind: 'queueStep', payload: { graphId: this.graphId, inputs } })
    } catch (e) {
      // Fire-and-forget: swallow mid-flight abort (graph destroyed). The
      // submitter's loop has already exited; surfacing the abort would
      // just force a try/catch at every call site for no payload.
      if ((e as { name?: string })?.name === 'AbortError') return
      throw e
    }
  }

  async readLoss(): Promise<number> {
    const r = await this.proxy.request<ReadLossResult>(
      { kind: 'readLoss', payload: { graphId: this.graphId } },
    )
    return r.loss
  }

  async run(inputs: LooseInputs): Promise<RunResult> {
    try {
      const r = await this.proxy.request<RunResultWire>(
        { kind: 'run', payload: { graphId: this.graphId, inputs } },
      )
      return { kind: 'completed', output: r.output, captures: makeCaptures(r.captures, this.meta.captureShapes) }
    } catch (e) {
      if ((e as { name?: string })?.name === 'AbortError') return { kind: 'aborted' }
      throw e
    }
  }

  uploadParams(params: Record<string, Float32Array>): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'uploadParams', payload: { graphId: this.graphId, params } },
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
    const initialParams = buildInitialParams(this._ir.plan, this.initFns, mulberry32(this.spec.seed!))
    await Promise.all([this.uploadParams(initialParams), this.resetOptimizerState()])
  }

  resetOptimizerState(): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'resetOptimizer', payload: { graphId: this.graphId } },
    ).then(() => undefined)
  }

  setLR(lr: LR): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'setLR', payload: { graphId: this.graphId, lr } },
    ).then(() => undefined)
  }

  async attach<I2 extends InputDecls>(s: ForwardSpec<M, I2>): Promise<CompiledForward<M, I2>> {
    const child: ForwardProxy<M, I2> = new ForwardProxy<M, I2>(
      this.proxy,
      this,
      s.forward,
      normalizeDecls(s.inputs),
      this.nextGraphId,
      () => this.children.delete(child),
    )
    this.children.add(child)
    return child
  }

  async replaceModel(
    newModel: M,
    replaceOpts?: { seed?: number; optimizer?: OptimizerConfig },
  ): Promise<void> {
    // Optimizer kind must match — swapping kinds (e.g. Adam → SGD) requires
    // re-creating state buffers (Adam has m/v slots; SGD has just v with
    // momentum). The runtime tracks those per-kind and a mid-graph kind
    // switch would silently produce garbage updates. If the user wants to
    // change kind, destroy + recompile fresh.
    if (replaceOpts?.optimizer && replaceOpts.optimizer.kind !== this.spec.optimizer.kind) {
      throw new Error(
        `replaceModel: optimizer kind cannot change ('${this.spec.optimizer.kind}' → '${replaceOpts.optimizer.kind}'). ` +
        `State buffers (Adam m/v vs SGD momentum) are kind-specific — destroy the compile and recompile to switch kinds.`,
      )
    }
    // Invalidate (don't destroy) siblings — their kernels are model-specific
    // but the proxy objects must outlive the swap so callers' references stay
    // valid. Each sibling recompiles lazily on its next run().
    for (const child of this.children) child._invalidateForReplace()
    this.proxy.send({ kind: 'destroy', payload: { graphId: this.graphId } })
    const newSpec: TrainingSpec<M, I> = {
      ...this.spec,
      model: newModel,
      seed: replaceOpts?.seed ?? randomSeed(),
      ...(replaceOpts?.optimizer !== undefined ? { optimizer: replaceOpts.optimizer } : {}),
    }
    const built = await buildTrainingGraph(this.proxy, newSpec, this.graphId)
    this._ir = built.ir
    this.meta = built.meta
    this.initFns = built.initFns
    this.spec = newSpec
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
  implements CompiledForward<M, I>, ChildProxy
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
      this.proxy, this.parent.graphId, this.parent.currentModel(), this.forward,
      resolved, this.nextGraphId,
    )
    this.cache.set(key, sib)
    return sib
  }

  async run(inputs: LooseInputs): Promise<RunResult> {
    try {
      const sib = await this.siblingFor(inputs)
      const r = await this.proxy.request<RunResultWire>(
        { kind: 'run', payload: { graphId: sib.graphId, inputs } },
      )
      return { kind: 'completed', output: r.output, captures: makeCaptures(r.captures, sib.meta.captureShapes) }
    } catch (e) {
      if ((e as { name?: string })?.name === 'AbortError') return { kind: 'aborted' }
      throw e
    }
  }

  async graphFor(inputs: LooseInputs): Promise<CompiledIR> {
    return (await this.siblingFor(inputs)).ir
  }

  async uploadParams(params: Record<string, Float32Array>): Promise<void> {
    // Params live on the parent (shared with all siblings).
    await this.proxy.request<null>(
      { kind: 'uploadParams', payload: { graphId: this.parent.graphId, params } },
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
  meta: CompileResult
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
  const meta = await proxy.request<CompileResult>(
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

function traceModule<M extends Module>(
  model: M,
  forward: ForwardFn<M, InputDecls>,
  inputDecls: InputDecls,
): { graph: Graph; materialized: MaterializedParams } {
  // Clone the user's model before materialization. `materializeParams`
  // mutates Param placeholders into Tensors; we run it on a fresh clone
  // so the caller's instance is never touched and can be reused across
  // multiple compiles (polymorphic siblings, replaceModel, etc.).
  const cloned = cloneModule(model)
  const normalized = normalizeDecls(inputDecls)
  let materialized: MaterializedParams = { tensors: {}, initFns: {}, decayFlags: {} }
  const graph = trace(() => {
    materialized = materializeParams(cloned)
    const inputTensors: Record<string, Tensor> = {}
    for (const [name, n] of Object.entries(normalized)) {
      const concrete = asConcreteShape(n.shape, name)
      inputTensors[name] = tensorInput(name, concrete, n.dtype)
    }
    return forward(cloned, inputTensors as InputsTensors<InputDecls>)
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
        `Polymorphic shapes are only supported via a forward spec attached as a ` +
        `sibling (\`train.attach(forwardSpec({ forward, inputs: { x: [null, 784] } }))\`); ` +
        `training specs require fully concrete shapes.`,
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
 *  or use separate forward specs. */
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
