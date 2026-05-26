// Top-level compile pipeline:
//   compile(opts)                  — spawns a worker; returns CompiledTraining
//   train.attach(opts)             — attach a sibling forward graph
//                                    (shares worker + param buffers)
//   replaceModel                   — in-place topology swap on a training
//                                    compile; attached forward children
//                                    invalidate their per-shape caches
//
// Compile-time work runs on the main thread; everything past createRuntime
// runs in a worker (see specs/WorkerArchitecture.md).
//
// Forward proxies are always polymorphic-capable: `null` dims in the declared
// inputs resolve from each call's TypedArray length, and a sibling is
// compiled + cached per distinct resolved shape.

import type { Tensor, Shape, Dtype } from './ir.js'
import { traceFn, tensorInput } from './trace.js'
import { appendGrad, type GradResult } from './grad.js'
import {
  appendAdam, wireAdamConfig,
  type AdamConfig, type AdamWConfig, type AdamResult,
} from './adam.js'
import { resolveLR, type LR } from './lr.js'
import { appendSGD, wireSGDConfig, type SGDConfig, type SGDResult } from './sgd.js'
import { planBuffers, type BufferPlan } from './buffers.js'
import { emitKernels, type KernelSpec } from './codegen.js'
import { Captures, type OutputArray, type DtypeArray } from './runtime.js'
import { Module, materializeParams, cloneModule, mulberry32, type MaterializedParams, type Rng, type InitFn } from './module.js'
import { WorkerProxy } from './worker-proxy.js'
import {
  transferablesOfRecord,
  type Req, type WireIR, type WireOptimizerConfig,
  type CompileResult,
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

/** Forward function shape. */
export type ForwardFn<M extends Module, I extends InputDecls = InputDecls> =
  (m: M, inputs: InputsTensors<I>) => Tensor

/** Discriminated result of `compiled.step(...)`. `'completed'` carries the
 *  scalar loss plus a `Captures` instance (empty when the graph has no
 *  `capture(...)` sites). `'aborted'` is returned when the graph was
 *  destroyed mid-flight (e.g. by `replaceModel`). `'failed'` is returned
 *  when the worker pipeline (kernel dispatch, mapAsync, internal IR, input
 *  validation) raises — the alternative would be a thrown rejection that
 *  silently kills unawaited training loops. Discriminator-only: no
 *  try/catch needed on `step` or `run` for any control-flow path. */
export type StepResult =
  | { kind: 'completed'; loss: number; captures: Captures }
  | { kind: 'aborted' }
  | { kind: 'failed'; error: Error }

/** Discriminated result of `compiled.run(...)`. Same shape as `StepResult`
 *  but `'completed'` carries the full output tensor (not just a scalar).
 *  Parameterized by the forward spec's `output` dtype: defaults to `'f32'`
 *  (output is `Float32Array`), use `output: 'i32'` in the spec for graphs
 *  that end in `categorical` / `argmax` / `argmin` (output is `Int32Array`). */
export type RunResult<O extends 'f32' | 'i32' = 'f32'> =
  | { kind: 'completed'; output: DtypeArray<O>; captures: Captures }
  | { kind: 'aborted' }
  | { kind: 'failed'; error: Error }

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

/** Training spec passed to `compile({ ... })`. Plain data — the compile
 *  pipeline consumes it once and clones `model` internally. */
export interface TrainingSpec<M extends Module, I extends InputDecls = InputDecls> {
  /** A model instance — `new Model()`. The compile pipeline clones this
   *  before tracing, so the caller's instance is never mutated and can be
   *  reused across multiple compiles (polymorphic siblings, replaceModel,
   *  etc.). */
  model: M
  /** Forward function that returns the scalar loss tensor. */
  loss: ForwardFn<M, I>
  /** Input shape declarations (one per named tensor input). */
  inputs: I
  /** Optimizer. Discriminated by `kind: 'adam' | 'adamw' | 'sgd'`. */
  optimizer: OptimizerConfig
  /** 32-bit integer seed for the param-init RNG. Same seed + same model
   *  topology → identical initial params, every time. If omitted, a seed
   *  is generated and exposed as `compiled.seed` so you can reproduce a
   *  run by passing it back. */
  seed?: number
}

/** Forward spec passed to `train.attach({ ... })`. The forward function
 *  reads the parent training compile's params — every training-step
 *  update is immediately visible. */
export interface ForwardSpec<M extends Module, I extends InputDecls = InputDecls, O extends 'f32' | 'i32' = 'f32'> {
  /** Forward function returning the output tensor. The first argument is
   *  the parent compile's model (cloned internally per trace). */
  forward: ForwardFn<M, I>
  /** Input shape declarations. `null` dims become parametric; the
   *  proxy caches a sibling per distinct resolved shape on first `run()`. */
  inputs: I
  /** Output tensor dtype. Defaults to `'f32'`. Set `'i32'` when the forward
   *  returns indices (`categorical`, `argmax`, `argmin`) so `r.output` types
   *  as `Int32Array` instead of `Float32Array`. Validated at compile against
   *  the actual graph output's dtype; mismatch throws. */
  output?: O
  /** Maximum number of distinct resolved shapes to cache simultaneously.
   *  Evicts the least-recently-used shape when full. Default: 8. */
  maxCachedShapes?: number
}

/** Spec passed to `compileForward({ ... })` — a standalone forward executor
 *  that owns its *own* param buffers (no parent training compile, no loss, no
 *  optimizer). Same as `ForwardSpec` plus the `model` instance and an optional
 *  init `seed`. The returned `CompiledForward` exposes
 *  `run`/`uploadParams`/`downloadParams`/`destroy`/`paramNames` — load weights
 *  in via `uploadParams` (e.g. from `loadSafetensors`) before running. */
export interface ForwardExecutorSpec<M extends Module, I extends InputDecls = InputDecls, O extends 'f32' | 'i32' = 'f32'>
  extends ForwardSpec<M, I, O>
{
  /** A model instance — `new Model()`. Cloned internally before tracing,
   *  exactly like `compile({ model })`. */
  model: M
  /** 32-bit integer seed for param init. Params are typically overwritten by
   *  `uploadParams` (imported weights), so this matters only for the values of
   *  any params you never upload. Defaults to a fresh random seed. */
  seed?: number
}


/** Returned by `compile(opts)`. Proxies all GPU work to an internal
 *  worker; every method returns a Promise. Generic over the declared
 *  inputs shape `I` so `step` / `run` accept inputs with the right
 *  TypedArray per dtype. */
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
   *  back via `compile({ ..., seed })` to reproduce a run. */
  readonly seed: number

  /** One full forward + backward + optimizer step. Always reads back the
   *  scalar loss plus any registered captures; the result discriminator
   *  distinguishes successful completion from a mid-flight abort (graph
   *  destroyed, typically by `replaceModel`). */
  step(inputs: TypedInputs<I>): Promise<StepResult>

  /** Upload params from a flat record (the shape `downloadParams` returns).
   *  Strict — the record must cover every param; missing and unknown keys both
   *  throw. Update a subset via `downloadParams()` + overlay. */
  uploadParams(params: Record<string, Float32Array>): Promise<void>
  /** Read params back as a flat `{ 'layers.0.W': Float32Array, ... }`
   *  record. Round-trips directly through `uploadParams`. */
  downloadParams(): Promise<Record<string, Float32Array>>

  /** Re-initialize params (from `seed`) and/or zero optimizer state. Defaults
   *  to both — the "start over with the same compile" button. Pass
   *  `{ params: false }` to wipe only optimizer state (e.g. after a
   *  hyperparameter change), or `{ optimizer: false }` to re-init params
   *  while keeping accumulated momentum. */
  reset(opts?: { params?: boolean; optimizer?: boolean }): Promise<void>

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
   *  forward compiles attached via `this.attach(...)` stay
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

  /** Attach a forward (inference) graph as a sibling — shares this training
   *  compile's worker and param GPUBuffers. Cascading destroy: when this
   *  training compile is destroyed (or `replaceModel`-d), attached forward
   *  proxies are cleaned up / invalidated automatically.
   *
   *  Returned `CompiledForward` proxy is polymorphic over `null`-wildcard
   *  inputs: per-shape kernels are compiled lazily and cached on first
   *  `run()` at each new resolved shape. The cache is LRU-bounded
   *  (default 8 shapes; tune via `maxCachedShapes`). */
  attach<I2 extends InputDecls, O2 extends 'f32' | 'i32' = 'f32'>(opts: ForwardSpec<M, I2, O2>): Promise<CompiledForward<M, I2, O2>>

  /** Tear down the worker + GPU resources, plus any attached forward
   *  compiles. */
  destroy(): void
}

/** Returned by `train.attach(opts)`. Polymorphic by default: `run()`
 *  at a new resolved shape lazily compiles + caches a sibling. Param
 *  reads/writes route to the parent training graph (shared buffers).
 *
 *  No top-level `graph` / `kernels`: forward proxies are polymorphic and
 *  hold one IR per shape. Use `graphFor(inputs)` to fetch (and lazily
 *  compile) the IR for a specific shape. */
export interface CompiledForward<M extends Module = Module, I extends InputDecls = InputDecls, O extends 'f32' | 'i32' = 'f32'> {
  /** Same as the parent training graph's param names. */
  readonly paramNames: readonly string[]

  /** Run the forward dispatch. Returns the discriminated `RunResult`:
   *  `'completed'` with the output tensor + any captures, or `'aborted'`
   *  if the parent training graph was destroyed mid-flight. */
  run(inputs: TypedInputs<I>): Promise<RunResult<O>>

  /** The compiled IR for the resolved shape of `inputs`. Compiles + caches
   *  a sibling on first call for that shape (same lazy-compile behavior as
   *  `run()`). Use for inspecting the kernel count / IR ops at the actual
   *  shape you're running. */
  graphFor(inputs: TypedInputs<I>): Promise<CompiledIR>

  /** Upload params. Strict — the record must cover every param; missing and
   *  unknown keys both throw. Reads/writes go to the parent training compile's
   *  buffers (shared, so updates are immediately visible there too). */
  uploadParams(params: Record<string, Float32Array>): Promise<void>
  /** Flat `{ 'layers.0.W': Float32Array, ... }` record of the shared param
   *  state — identical to the parent training graph's `downloadParams()`
   *  since params are physically shared. */
  downloadParams(): Promise<Record<string, Float32Array>>

  destroy(): void
}

/**
 * Compile a training graph to a worker-backed runtime — spawns a worker,
 * owns the param GPUBuffers, and returns a `CompiledTraining` handle.
 *
 * ```ts
 * const model = new MLP()
 * const train = await compile({ model, loss, inputs, optimizer })
 * ```
 *
 * For forward (inference) compiles that share the training graph's params
 * and worker, use `train.attach({ forward, inputs })` — the lifecycle
 * relationship is named at the call site.
 */
export async function compile<M extends Module, I extends InputDecls>(
  opts: TrainingSpec<M, I>,
): Promise<CompiledTraining<M, I>> {
  const proxy = new WorkerProxy(__WORKER_SOURCE__)
  const sealed: TrainingSpecSealed<M, I> = { ...opts, seed: opts.seed ?? randomSeed() }
  try {
    const built = await buildTrainingGraph(proxy, sealed, 0)
    return new CompiledTrainingProxy<M, I>(
      proxy, 0, built.ir, built.meta, sealed, built.initFns, { v: 1 },
    )
  } catch (e) {
    proxy.terminate()
    throw e
  }
}

/** Internal: TrainingSpec with `seed` filled in (either user-supplied or
 *  freshly generated). Kept distinct from the public type so callers can
 *  omit `seed`, while the runtime always sees a concrete number. */
type TrainingSpecSealed<M extends Module, I extends InputDecls> =
  TrainingSpec<M, I> & { seed: number }

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

/**
 * Build the full training IR (forward + backward + optimizer ops + buffer
 * plan + emitted kernels) without spawning a worker, compiling WGSL in the
 * GPU driver, or allocating GPU memory. Pair with `compile()` for execution.
 *
 * Async because the JS pipeline can run for several seconds at frontier
 * scales (~5s for a GPT-4-shaped model). Yields between phases keep the
 * host's main thread responsive — without them, a browser tab calling
 * this would freeze for the duration.
 *
 * ```ts
 * const { graph, kernels } = await trace({ model, loss, inputs, optimizer })
 * ```
 */
export async function trace<M extends Module, I extends InputDecls>(
  opts: TrainingSpec<M, I>,
): Promise<CompiledIR> {
  const { ir } = await buildTrainingIR(opts)
  return ir
}

/**
 * Build a forward-only IR — same as what `train.attach({ forward, inputs })`
 * produces internally, but as a standalone `CompiledIR` with no parent
 * `CompiledTraining` and no GPU work.
 *
 * `inputs` must be fully concrete — null-wildcard dims aren't supported
 * (there's no per-shape cache to lazily compile against). Returned
 * `loss` field is the forward output tensor; `paramGrads` is empty.
 *
 * ```ts
 * const { graph, kernels } = await traceForward({ model, forward, inputs })
 * ```
 */
export async function traceForward<M extends Module, I extends InputDecls>(
  opts: { model: M; forward: ForwardFn<M, I>; inputs: I },
): Promise<CompiledIR> {
  return (await buildForwardIR(opts.model, opts.forward as unknown as ForwardFn<M, InputDecls>, opts.inputs)).ir
}

/**
 * Compile a *standalone forward-only* graph to a worker-backed runtime — spawns
 * its own worker and owns its *own* param buffers. The third executor path
 * alongside `compile()` (needs a loss + optimizer) and `train.attach()` (needs
 * a parent training compile and *shares* its params): `compileForward` runs a
 * model that has neither a training counterpart nor a parent.
 *
 * The headline use is running an imported pretrained backbone: load weights via
 * `uploadParams` (e.g. from `loadSafetensors`), then `run` to extract features.
 *
 * ```ts
 * const backbone = await compileForward({ model: new Backbone(), forward, inputs })
 * await backbone.uploadParams(loadSafetensors(buf).tensors)
 * const { output } = await backbone.run({ x })
 * ```
 *
 * Polymorphic over `null`-wildcard input dims like `train.attach`: the owner
 * graph (which owns the params) is compiled at the first resolved shape, and
 * each additional shape is cached as a sibling that shares the owner's params.
 * With fully concrete input shapes the owner is created eagerly, so
 * `uploadParams` / `downloadParams` work before the first `run()`.
 */
export async function compileForward<M extends Module, I extends InputDecls, O extends 'f32' | 'i32' = 'f32'>(
  spec: ForwardExecutorSpec<M, I, O>,
): Promise<CompiledForward<M, I, O>> {
  const proxy = new WorkerProxy(__WORKER_SOURCE__)
  try {
    const decls = normalizeDecls(spec.inputs)
    const seed = spec.seed ?? randomSeed()
    const concrete = declsAreConcrete(decls)
    // Eager metadata trace: enumerate the model's params (names + init fns) and
    // surface any trace-time error up front. When the declared shape is fully
    // concrete this same IR is promoted to the live owner (no second trace);
    // for a parametric shape it's metadata-only (wildcards substituted with 1,
    // which is shape-independent for params) and the owner traces fresh at the
    // first run()'s actual shape.
    const metaShape = ownerMetaShape(decls)
    const { ir, materialized } = await buildForwardIR(
      spec.model, spec.forward as unknown as ForwardFn<M, InputDecls>, metaShape,
    )
    const executor = new ForwardExecutorProxy<M, I, O>(
      proxy, spec.model, spec.forward, decls, spec.output ?? 'f32',
      spec.maxCachedShapes ?? DEFAULT_MAX_CACHED_SHAPES, seed,
      materialized.initFns, [...ir.plan.paramsByName.keys()],
    )
    if (concrete) await executor._initOwnerEager(metaShape, ir)
    return executor
  } catch (e) {
    proxy.terminate()
    throw e
  }
}

/** Yields a macrotask via setTimeout(0) so the browser can paint between
 *  phases. Microtasks (queueMicrotask, Promise.resolve) don't release the
 *  thread to the renderer — only macrotasks do. */
function yieldToUI(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0))
}

interface BuiltTrainingIR {
  ir: CompiledIR
  materialized: MaterializedParams
  adamResult: AdamResult | undefined
  sgdResult: SGDResult | undefined
}

/** Pure-JS portion of the training pipeline: trace + autograd + optimizer
 *  pass + plan + emit. Shared by `trace()` (public inspection) and
 *  `buildTrainingGraph()` (which adds worker setup). Yields between phases
 *  for UI responsiveness; see `yieldToUI`. */
async function buildTrainingIR<M extends Module, I extends InputDecls>(
  opts: TrainingSpec<M, I>,
): Promise<BuiltTrainingIR> {
  const loss = opts.loss as unknown as ForwardFn<M, InputDecls>
  const inputs = opts.inputs as InputDecls
  const { graph, materialized } = traceModule(opts.model, loss, inputs)
  await yieldToUI()
  const { paramGrads, loss: lossTensor } = appendGrad(graph)
  await yieldToUI()
  const adamResult = opts.optimizer.kind === 'adam' || opts.optimizer.kind === 'adamw'
    ? appendAdam(graph, paramGrads, materialized.tensors, opts.optimizer, materialized.decayFlags)
    : undefined
  const sgdResult = opts.optimizer.kind === 'sgd'
    ? appendSGD(graph, paramGrads, materialized.tensors, opts.optimizer, materialized.decayFlags)
    : undefined
  const writebacks = adamResult?.writebacks ?? sgdResult?.writebacks ?? []
  await yieldToUI()
  const plan = planBuffers(graph, paramGrads, writebacks)
  const kernels = emitKernels(graph, plan)
  const ir: CompiledIR = { graph, paramGrads, loss: lossTensor, plan, kernels }
  return { ir, materialized, adamResult, sgdResult }
}

/** Pure-JS portion of a forward-only graph build. Shared by `traceForward()`
 *  (public inspection), `compileSibling()` (per-shape lazy compile against a
 *  parent), and `compileForward()` (standalone executor — needs `materialized`
 *  to build its own initial params, since it owns its buffers). */
async function buildForwardIR<M extends Module>(
  model: M,
  forward: ForwardFn<M, InputDecls>,
  decls: InputDecls,
): Promise<{ ir: CompiledIR; materialized: MaterializedParams }> {
  const { graph, materialized } = traceModule(model, forward, decls)
  await yieldToUI()
  const outputTensor = graph.tensors[graph.outputs[0]!]!
  const plan = planBuffers(graph, {})
  const kernels = emitKernels(graph, plan)
  return { ir: { graph, paramGrads: {}, loss: outputTensor, plan, kernels }, materialized }
}

interface BuiltTrainingGraph {
  ir: CompiledIR
  meta: CompileResult
  initFns: Record<string, InitFn>
}

// Builds the IR via `buildTrainingIR`, then submits it to the worker.
// Shared by `compile` (fresh worker) and `replaceModel` (existing worker).
async function buildTrainingGraph<M extends Module, I extends InputDecls>(
  proxy: WorkerProxy,
  opts: TrainingSpecSealed<M, I>,
  graphId: number,
): Promise<BuiltTrainingGraph> {
  const { ir, materialized, adamResult, sgdResult } = await buildTrainingIR(opts)
  const initialParams = buildInitialParams(ir.plan, materialized.initFns, mulberry32(opts.seed))
  const wireOptimizer: WireOptimizerConfig | null =
    adamResult ? { kind: 'adam', config: wireAdamConfig(adamResult) }
    : sgdResult ? { kind: 'sgd', config: wireSGDConfig(sgdResult) }
    : null
  const transfers = transferablesOfRecord(initialParams)
  const meta = await proxy.request<CompileResult>(
    { kind: 'createRuntime', payload: { graphId, ir: toWireIR(ir), initialParams, optimizer: wireOptimizer } },
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
  /** Forward children attached via `this.attach({...})`.
   *  Tracked so `destroy()` can clean them up cascade-style and
   *  `replaceModel()` can invalidate their per-shape kernel caches without
   *  unregistering them. */
  private readonly children = new Set<ChildProxy>()

  // Swapped in place by `replaceModel` so callers' references (and any
  // sibling ForwardProxy holding `this`) stay valid across topology changes.
  private _ir: CompiledIR
  private meta: CompileResult
  private opts: TrainingSpecSealed<M, I>
  private initFns: Record<string, InitFn>

  constructor(
    private readonly proxy: WorkerProxy,
    readonly graphId: number,
    ir: CompiledIR,
    meta: CompileResult,
    opts: TrainingSpecSealed<M, I>,
    initFns: Record<string, InitFn>,
    private readonly nextGraphId: { v: number },
  ) {
    this._ir = ir
    this.meta = meta
    this.opts = opts
    this.initFns = initFns
  }

  get graph(): GradResult['graph'] { return this._ir.graph }
  get kernels(): readonly KernelSpec[] { return this._ir.kernels }
  get outputShape(): readonly number[] { return this.meta.outputShape }
  get paramNames(): readonly string[] { return this.meta.paramNames }
  get seed(): number { return this.opts.seed }

  /** Sibling ForwardProxies read this through the proxy ref so they always
   *  re-trace against the latest topology. */
  currentModel(): M { return this.opts.model }

  step(inputs: LooseInputs): Promise<StepResult> {
    return guarded(async () => {
      const r = await this.proxy.request<StepResultWire>(
        { kind: 'step', payload: { graphId: this.graphId, inputs } },
      )
      return { kind: 'completed' as const, loss: r.loss, captures: makeCaptures(r.captures, this.meta.captureShapes) }
    })
  }

  uploadParams(params: Record<string, Float32Array>): Promise<void> {
    return uploadParamsTo(this.proxy, this.graphId, params)
  }

  downloadParams(): Promise<Record<string, Float32Array>> {
    return downloadParamsFrom(this.proxy, this.graphId)
  }

  async reset(opts: { params?: boolean; optimizer?: boolean } = {}): Promise<void> {
    const doParams = opts.params !== false
    const doOptimizer = opts.optimizer !== false
    const tasks: Promise<void>[] = []
    if (doParams) {
      const initialParams = buildInitialParams(this._ir.plan, this.initFns, mulberry32(this.opts.seed))
      tasks.push(this.uploadParams(initialParams))
    }
    if (doOptimizer) {
      tasks.push(
        this.proxy.request<null>(
          { kind: 'resetOptimizer', payload: { graphId: this.graphId } },
        ).then(() => undefined),
      )
    }
    await Promise.all(tasks)
  }

  setLR(lr: LR): Promise<void> {
    return this.proxy.request<null>(
      { kind: 'setLR', payload: { graphId: this.graphId, lr } },
    ).then(() => undefined)
  }

  async attach<I2 extends InputDecls, O2 extends 'f32' | 'i32' = 'f32'>(
    opts: ForwardSpec<M, I2, O2>,
  ): Promise<CompiledForward<M, I2, O2>> {
    const declaredOutput: 'f32' | 'i32' = opts.output ?? 'f32'
    const child: ForwardProxy<M, I2, O2> = new ForwardProxy<M, I2, O2>(
      this.proxy,
      this,
      opts.forward,
      normalizeDecls(opts.inputs),
      declaredOutput,
      this.nextGraphId,
      opts.maxCachedShapes ?? DEFAULT_MAX_CACHED_SHAPES,
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
    if (replaceOpts?.optimizer && replaceOpts.optimizer.kind !== this.opts.optimizer.kind) {
      throw new Error(
        `replaceModel: optimizer kind cannot change ('${this.opts.optimizer.kind}' → '${replaceOpts.optimizer.kind}'). ` +
        `State buffers (Adam m/v vs SGD momentum) are kind-specific — destroy the compile and recompile to switch kinds.`,
      )
    }
    // Invalidate (don't destroy) siblings — their kernels are model-specific
    // but the proxy objects must outlive the swap so callers' references stay
    // valid. Each sibling recompiles lazily on its next run().
    for (const child of this.children) child._invalidateForReplace()
    this.proxy.send({ kind: 'destroy', payload: { graphId: this.graphId } })
    const newOpts: TrainingSpecSealed<M, I> = {
      ...this.opts,
      model: newModel,
      seed: replaceOpts?.seed ?? randomSeed(),
      ...(replaceOpts?.optimizer !== undefined ? { optimizer: replaceOpts.optimizer } : {}),
    }
    const built = await buildTrainingGraph(this.proxy, newOpts, this.graphId)
    this._ir = built.ir
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

/** Default LRU cap on per-shape sibling cache. Real workloads see 1–3
 *  distinct shapes (B=1 / B=eval / B=train); 8 leaves headroom without
 *  permitting unbounded growth on UIs that hit many shapes. Tune via
 *  `attach({ maxCachedShapes })`. */
const DEFAULT_MAX_CACHED_SHAPES = 8

/** Single forward proxy class — handles both fully-concrete and polymorphic
 *  shapes (concrete is the cache-size-1 case). Sibling of a training graph;
 *  shares its param GPUBuffers. Holds a reference to the parent proxy so it
 *  picks up the current model factory after `replaceModel`. */
class ForwardProxy<M extends Module, I extends InputDecls, O extends 'f32' | 'i32' = 'f32'>
  implements CompiledForward<M, I, O>, ChildProxy
{
  /** Per-shape sibling cache (LRU; see `ShapeCache`). */
  private readonly cache: ShapeCache

  constructor(
    private readonly proxy: WorkerProxy,
    private readonly parent: ParentRef<M>,
    private readonly forward: ForwardFn<M, I>,
    private readonly decls: NormalizedDecls,
    private readonly declaredOutput: 'f32' | 'i32',
    private readonly nextGraphId: { v: number },
    maxCachedShapes: number,
    private readonly onDestroy: () => void,
  ) {
    this.cache = new ShapeCache(proxy, maxCachedShapes)
  }

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
    assertOutputDtype(sib.ir, this.declaredOutput, 'attach')
    this.cache.set(key, sib)
    return sib
  }

  run(inputs: LooseInputs): Promise<RunResult<O>> {
    return guarded(async () => {
      const sib = await this.siblingFor(inputs)
      const r = await this.proxy.request<RunResultWire>(
        { kind: 'run', payload: { graphId: sib.graphId, inputs } },
      )
      // r.output's concrete TypedArray class matches the declared output
      // dtype (validated above + runtime returns the right class via
      // wrapReadback). Cast to the typed array the generic O resolves to.
      return { kind: 'completed' as const, output: r.output as DtypeArray<O>, captures: makeCaptures(r.captures, sib.meta.captureShapes) }
    })
  }

  async graphFor(inputs: LooseInputs): Promise<CompiledIR> {
    return (await this.siblingFor(inputs)).ir
  }

  uploadParams(params: Record<string, Float32Array>): Promise<void> {
    // Params live on the parent (shared with all siblings).
    return uploadParamsTo(this.proxy, this.parent.graphId, params)
  }

  downloadParams(): Promise<Record<string, Float32Array>> {
    return downloadParamsFrom(this.proxy, this.parent.graphId)
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
  const { ir } = await buildForwardIR(model, forward as unknown as ForwardFn<M, InputDecls>, decls)
  const childGraphId = nextGraphId.v++
  const wireIR: WireIR = { graph: ir.graph, plan: ir.plan, kernels: ir.kernels }
  const meta = await proxy.request<CompileResult>(
    { kind: 'compileForward', payload: { graphId: childGraphId, parentGraphId, ir: toWireIR(ir) } },
  )
  return { graphId: childGraphId, ir, meta }
}

/** The fields of a `CompiledIR` that cross the wire to the worker. */
function toWireIR(ir: CompiledIR): WireIR {
  return { graph: ir.graph, plan: ir.plan, kernels: ir.kernels }
}

/** Validate a forward graph's actual output dtype against the spec's declared
 *  `output`. A mismatch (e.g. `output: 'i32'` on an f32-returning forward)
 *  would otherwise produce wrong-class TypedArray reads at every call site, so
 *  fail loudly at compile. `label` names the entry point in the message. */
function assertOutputDtype(ir: CompiledIR, declared: 'f32' | 'i32', label: 'attach' | 'compileForward'): void {
  const actual = ir.graph.tensors[ir.graph.outputs[0]!]!.dtype
  if (actual !== declared) {
    throw new Error(
      `${label}: forward declares output: '${declared}' but the traced graph's output tensor is '${actual}'. ` +
      `Use \`output: '${actual}'\` in the forward spec (or default by omitting the field for f32).`,
    )
  }
}

/** LRU cache of per-shape compiled siblings, keyed by resolved-shape string.
 *  Map insertion order = recency: a hit bumps to most-recent, and inserting
 *  past capacity evicts the oldest — sending the worker a `destroy` so its
 *  kernels free their GPU buffers. Shared by both forward proxies; the
 *  proxy-specific owner / parent logic stays in their `siblingFor`. */
class ShapeCache {
  private readonly map = new Map<string, ForwardSiblingMeta>()
  constructor(private readonly proxy: WorkerProxy, private readonly max: number) {}

  get(key: string): ForwardSiblingMeta | undefined {
    const hit = this.map.get(key)
    if (!hit) return undefined
    this.map.delete(key); this.map.set(key, hit)   // bump to most-recently-used
    return hit
  }

  set(key: string, sib: ForwardSiblingMeta): void {
    if (this.map.size >= this.max) {
      const oldest = this.map.keys().next().value
      if (oldest !== undefined) { this.destroyGraph(this.map.get(oldest)!); this.map.delete(oldest) }
    }
    this.map.set(key, sib)
  }

  /** Destroy every cached graph in the worker and empty the cache. */
  clear(): void {
    for (const sib of this.map.values()) this.destroyGraph(sib)
    this.map.clear()
  }

  private destroyGraph(sib: ForwardSiblingMeta): void {
    this.proxy.send({ kind: 'destroy', payload: { graphId: sib.graphId } })
  }
}

/** Standalone forward executor proxy (returned by `compileForward`). Owns its
 *  own worker + param buffers — no parent training compile. The owner graph
 *  (graphId 0) is created via `createRuntime` with `optimizer: null` and holds
 *  the params; additional resolved shapes reuse the existing sibling mechanism
 *  (a `compileForward` payload pointed at the owner), so polymorphism falls out
 *  of composing the two shipping worker paths — no worker change.
 *
 *  The owner is held separately from the LRU sibling cache so it's never
 *  evicted: evicting it would free the param buffers every sibling shares. */
class ForwardExecutorProxy<M extends Module, I extends InputDecls, O extends 'f32' | 'i32' = 'f32'>
  implements CompiledForward<M, I, O>
{
  /** Owner graph (graphId 0). Null until created — eagerly for concrete input
   *  shapes (see `compileForward`), else on the first `run()`. */
  private ownerMeta: ForwardSiblingMeta | null = null
  private ownerKey: string | null = null
  /** Sibling shapes only — the owner is held in `ownerMeta` so it's never
   *  evicted (freeing it would drop the shared param buffers). LRU; see `ShapeCache`. */
  private readonly cache: ShapeCache
  private readonly nextGraphId = { v: 1 }
  /** `uploadParams` calls issued before the owner exists (parametric spec, no
   *  run yet) accumulate here and flush right after the owner is created. Merged
   *  (later keys win); the merged union is strict-checked at flush. */
  private pendingUpload: Record<string, Float32Array> | null = null

  constructor(
    private readonly proxy: WorkerProxy,
    private readonly model: M,
    private readonly forward: ForwardFn<M, I>,
    private readonly decls: NormalizedDecls,
    private readonly declaredOutput: 'f32' | 'i32',
    maxCachedShapes: number,
    private readonly seed: number,
    private readonly initFns: Record<string, InitFn>,
    private _paramNames: readonly string[],
  ) {
    this.cache = new ShapeCache(proxy, maxCachedShapes)
  }

  get paramNames(): readonly string[] { return this._paramNames }

  /** Create the owner graph eagerly from a prebuilt IR (concrete-shape path,
   *  called once from `compileForward`). Same effect as the lazy owner-create
   *  inside `siblingFor`, but reuses the trace already done for metadata. */
  async _initOwnerEager(resolved: ResolvedDecls, prebuilt: CompiledIR): Promise<void> {
    await this.createOwner(resolved, prebuilt)
  }

  private async createOwner(resolved: ResolvedDecls, prebuilt?: CompiledIR): Promise<ForwardSiblingMeta> {
    const ir = prebuilt ?? (await buildForwardIR(
      this.model, this.forward as unknown as ForwardFn<M, InputDecls>, resolved,
    )).ir
    const initialParams = buildInitialParams(ir.plan, this.initFns, mulberry32(this.seed))
    const meta = await this.proxy.request<CompileResult>(
      { kind: 'createRuntime', payload: { graphId: 0, ir: toWireIR(ir), initialParams, optimizer: null } },
      transferablesOfRecord(initialParams),
    )
    assertOutputDtype(ir, this.declaredOutput, 'compileForward')
    this.ownerMeta = { graphId: 0, ir, meta }
    this.ownerKey = shapeKey(resolved)
    this._paramNames = meta.paramNames
    if (this.pendingUpload) {
      await uploadParamsTo(this.proxy, 0, this.pendingUpload)
      this.pendingUpload = null
    }
    return this.ownerMeta
  }

  private async siblingFor(inputs: LooseInputs): Promise<ForwardSiblingMeta> {
    const resolved = resolveDecls(this.decls, inputs)
    const key = shapeKey(resolved)
    if (!this.ownerMeta) return this.createOwner(resolved)
    if (key === this.ownerKey) return this.ownerMeta
    const hit = this.cache.get(key)
    if (hit) return hit
    const sib = await compileSibling<M, I>(
      this.proxy, this.ownerMeta.graphId, this.model, this.forward, resolved, this.nextGraphId,
    )
    assertOutputDtype(sib.ir, this.declaredOutput, 'compileForward')
    this.cache.set(key, sib)
    return sib
  }

  run(inputs: LooseInputs): Promise<RunResult<O>> {
    return guarded(async () => {
      const sib = await this.siblingFor(inputs)
      const r = await this.proxy.request<RunResultWire>(
        { kind: 'run', payload: { graphId: sib.graphId, inputs } },
      )
      return { kind: 'completed' as const, output: r.output as DtypeArray<O>, captures: makeCaptures(r.captures, sib.meta.captureShapes) }
    })
  }

  async graphFor(inputs: LooseInputs): Promise<CompiledIR> {
    return (await this.siblingFor(inputs)).ir
  }

  uploadParams(params: Record<string, Float32Array>): Promise<void> {
    // Before the owner exists (parametric spec, no run yet) there are no GPU
    // buffers to write — buffer the upload and flush it after owner creation.
    if (!this.ownerMeta) {
      this.pendingUpload = { ...(this.pendingUpload ?? {}), ...params }
      return Promise.resolve()
    }
    return uploadParamsTo(this.proxy, this.ownerMeta.graphId, params)
  }

  downloadParams(): Promise<Record<string, Float32Array>> {
    if (!this.ownerMeta) {
      return Promise.reject(new Error(
        'compileForward: downloadParams() before any params exist on the GPU. With a parametric ' +
        '(null-wildcard) input shape the param buffers are allocated on the first run(); call ' +
        'run() once (or declare fully concrete input shapes) before downloadParams().',
      ))
    }
    return downloadParamsFrom(this.proxy, this.ownerMeta.graphId)
  }

  destroy(): void {
    // Destroying the owner (graphId 0) cascades to every sibling sharing its
    // params (worker handleDestroy walks parentGraphId), then we tear down the
    // worker. Nothing to destroy if the owner was never created.
    if (this.ownerMeta) this.proxy.send({ kind: 'destroy', payload: { graphId: this.ownerMeta.graphId } })
    this.proxy.terminate()
  }
}

/** True when no declared input shape contains a `null` wildcard — the owner
 *  graph can then be created eagerly at `compileForward` time. */
function declsAreConcrete(decls: NormalizedDecls): boolean {
  return Object.values(decls).every(d => d.shape.every(x => x !== null))
}

/** Shape to trace for the eager metadata pass: concrete dims as-is, `null`
 *  wildcards substituted with 1. Param shapes are batch-independent, so this is
 *  exact for enumerating params; for a concrete spec it equals the declared
 *  shape and the resulting IR is promoted to the live owner. */
function ownerMetaShape(decls: NormalizedDecls): ResolvedDecls {
  const out: ResolvedDecls = {}
  for (const [name, d] of Object.entries(decls)) {
    out[name] = { shape: d.shape.map(x => (x === null ? 1 : x)) as number[], dtype: d.dtype }
  }
  return out
}

type Graph = ReturnType<typeof traceFn>

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
  const graph = traceFn(() => {
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
        `sibling (\`train.attach({ forward, inputs: { x: [null, 784] } })\`); ` +
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

function makeCaptures(
  captures: Record<string, OutputArray> | null,
  captureShapes: Record<string, number[]>,
): Captures {
  const data = new Map<string, OutputArray>()
  if (captures) for (const [name, arr] of Object.entries(captures)) data.set(name, arr)
  return new Captures(captureShapes, data)
}

/** Run a worker call and translate its exceptions into the `'aborted'` /
 *  `'failed'` result discriminator. The single source of that contract —
 *  `step` and `run` supply only the `'completed'` branch. */
async function guarded<C>(body: () => Promise<C>): Promise<C | { kind: 'aborted' } | { kind: 'failed'; error: Error }> {
  try {
    return await body()
  } catch (e) {
    if ((e as { name?: string })?.name === 'AbortError') return { kind: 'aborted' }
    return { kind: 'failed', error: e instanceof Error ? e : new Error(String(e)) }
  }
}

/** Param upload/download over the worker protocol. The only thing that varies
 *  between the two proxies is whose `graphId` owns the params (a training
 *  graph, or the parent a forward sibling shares with). */
function uploadParamsTo(proxy: WorkerProxy, graphId: number, params: Record<string, Float32Array>): Promise<void> {
  return proxy.request<null>({ kind: 'uploadParams', payload: { graphId, params } }).then(() => undefined)
}

function downloadParamsFrom(proxy: WorkerProxy, graphId: number): Promise<Record<string, Float32Array>> {
  return proxy.request<DownloadParamsResult>({ kind: 'downloadParams', payload: { graphId } }).then(r => r.params)
}

