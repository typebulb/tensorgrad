// Top-level compile(): trace → autograd → buffer plan → codegen → runtime.
//
// Two entry points:
//   * `compile(traceFn)`        — low-level. User declares params via
//                                 paramInput() inside the trace.
//   * `compileModule(model, …)` — high-level. User defines the model as a
//                                 Module tree; the library auto-discovers
//                                 params, traces the forward, appends grad
//                                 and Adam, and returns a runtime.

import type { Tensor, Shape, Dtype } from './ir.js'
import { trace, tensorInput } from './trace.js'
import { appendGrad, type GradResult } from './grad.js'
import { appendAdam, type AdamConfig, type AdamResult } from './adam.js'
import { planBuffers, type BufferPlan } from './buffers.js'
import { emitKernels, type KernelSpec } from './codegen.js'
import { createRuntime, createForwardRuntime, type CompiledRuntime, type CompiledForward, type RuntimeOpts } from './runtime.js'
import { Module, materializeParams, type MaterializedParams } from './module.js'

/** Declares one input tensor of the model's forward function. The name is the
 *  key in the `inputs:` Record at compile time and the key on the `step()`/
 *  `run()` data object at runtime. */
export interface InputDecl {
  shape: Shape
  dtype?: Dtype
}

/** Inputs declaration: a Record from input name to its shape/dtype. The name
 *  doubles as the key the forward fn destructures and the key the runtime
 *  expects in `step({...})` / `run({...})`. */
export type InputDecls = Record<string, InputDecl>

/** Maps an `InputDecls` Record to its forward-time tensor counterpart —
 *  same keys, each value is a Tensor. Used to type the forward function's
 *  `inputs` argument from the declared shape Record. */
export type InputsTensors<I extends InputDecls> = { [K in keyof I]: Tensor }

/** Forward function shape: takes the materialized model and a Record of
 *  named input tensors (matching the declared `inputs:` keys), returns the
 *  output tensor (loss for compileModule; logits/etc. for compileForward).
 *  The second generic flows from the inputs declaration so destructuring
 *  the input record stays typed. */
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

/** Full compile pipeline. Browser-only because it creates a GPUDevice. */
export async function compile(traceFn: () => Tensor, opts: RuntimeOpts = {}): Promise<CompiledRuntime & { ir: CompiledIR }> {
  const ir = compileToIR(traceFn)
  const lossBufferId = ir.plan.tensorToBuffer.get(ir.loss.id)!
  const runtime = await createRuntime(ir.plan, ir.kernels, lossBufferId, opts)
  return Object.assign(runtime, { ir })
}

// ============================================================================
// Module-aware compile
// ============================================================================

export interface CompileModuleOptions<I extends InputDecls = InputDecls> extends RuntimeOpts {
  /** Per-step data inputs to the forward function, keyed by name. The forward
   *  fn destructures these out of its second argument; runtime calls to
   *  `step()` / `run()` pass typed arrays under the same keys. */
  inputs?: I
  /** Adam hyperparameters. If omitted, no optimizer is appended (forward-only). */
  adam?: AdamConfig
}

export interface CompileForwardOptions<I extends InputDecls = InputDecls> extends RuntimeOpts {
  /** Per-step data inputs to the forward function, keyed by name. */
  inputs?: I
}

/** Forward-only compile options as taken by the `compileForward` *method* on
 *  a training runtime — no `device` (inherited) and no `sharedParams`
 *  (auto-supplied from the train graph's params). */
export interface CompileForwardMethodOptions<I extends InputDecls = InputDecls> {
  inputs?: I
}

/** Returned by `compileModule`. Adds training-graph extras (auto-init, reset,
 *  sibling-graph compile) on top of the base runtime. */
export interface CompiledModule<M extends Module> extends CompiledRuntime {
  ir: CompiledIR
  /** Number of dispatchable kernels (excludes leaf no-ops). */
  kernelCount: number
  /** Re-initialize all params from their declared init specs and zero the
   *  optimizer state. Use to start training over without recompiling. */
  reset(): void
  /** Compile a sibling forward-only graph (e.g., a B=1 inference graph or a
   *  B=N held-out eval graph) that shares this runtime's device and param
   *  buffers. Pass the forward fn (typically distinct from your loss fn —
   *  it returns logits, not a scalar) and any shape changes via `inputs`.
   *  Auto-initialization is a no-op since params are shared. */
  compileForward<I extends InputDecls>(
    forward: ForwardFn<M, I>,
    opts?: CompileForwardMethodOptions<I>,
  ): Promise<CompiledForwardModule>
}

/** Returned by `compileForward` (and by the `compileForward` method). */
export interface CompiledForwardModule extends CompiledForward {
  ir: CompiledIR
  /** Number of dispatchable kernels (excludes leaf no-ops). */
  kernelCount: number
}

/**
 * Compile a Module-based model. Pass a *factory* `() => new Model()`, not the
 * model instance itself: compilation mutates the tree (every `ParamSentinel`
 * field becomes a real `Tensor`), so the instance is consumed and shouldn't be
 * referenced afterwards. Re-call the factory if you need a fresh tree.
 *
 * The forward function takes the materialized model and a Record of named
 * input tensors, returns the loss tensor. Inputs are matched by name with the
 * `inputs:` declaration:
 *
 *   inputs: {
 *     tokens:  { shape: [B, T], dtype: 'i32' },
 *     targets: { shape: [B, T], dtype: 'i32' },
 *   }
 *   forward: (m, { tokens, targets }) => …
 *
 * Walks the module tree to materialize params with auto-derived names, then
 * runs trace → grad → adam → buffer plan → codegen → runtime. Initial
 * parameter values are uploaded automatically before this function returns;
 * call `reset()` later to re-randomize.
 *
 * If `opts.adam` is set, the runtime's `step()` automatically tracks an
 * internal step count and injects the bias-corrected `lrt` scalar each call;
 * users don't need to provide it themselves.
 */
export async function compileModule<M extends Module, I extends InputDecls = InputDecls>(
  modelFactory: () => M,
  forward: ForwardFn<M, I>,
  opts: CompileModuleOptions<I> = {},
): Promise<CompiledModule<M>> {
  const { graph, materialized } = traceModule(modelFactory, forward, opts.inputs ?? {})
  const { paramGrads, loss } = appendGrad(graph)
  const adamResult = opts.adam
    ? appendAdam(graph, paramGrads, materialized.tensors, opts.adam, materialized.decayFlags)
    : undefined

  const plan = planBuffers(graph, paramGrads, adamResult?.writebacks ?? [])
  const kernels = emitKernels(graph, plan)
  const lossBufferId = plan.tensorToBuffer.get(loss.id)!
  const runtime = await createRuntime(plan, kernels, lossBufferId, opts)

  if (adamResult) wrapStepForAdam(runtime, adamResult)
  uploadInitialParams(plan, materialized.initFns, runtime, /* sharedParams */ undefined)

  const ir: CompiledIR = { graph, paramGrads, loss, plan, kernels }
  const kernelCount = countKernels(kernels)

  const reset = () => {
    uploadInitialParams(plan, materialized.initFns, runtime, undefined)
    runtime.resetOptimizerState()
  }

  const compileForwardMethod = <J extends InputDecls>(
    forwardFn: ForwardFn<M, J>,
    fOpts: CompileForwardMethodOptions<J> = {},
  ): Promise<CompiledForwardModule> =>
    compileForward<M, J>(modelFactory, forwardFn, {
      ...fOpts,
      device: runtime.device,
      sharedParams: runtime.params,
    })

  return Object.assign(runtime, { ir, kernelCount, reset, compileForward: compileForwardMethod })
}

// ============================================================================
// Forward-only compile
// ============================================================================

/**
 * Compile a Module-based model in forward-only mode (no autograd, no Adam).
 * The forward function returns the output tensor (e.g., logits) instead of a
 * scalar loss; runtime exposes `run(inputs)` returning the full output as a
 * `Float32Array`.
 *
 * **Prefer the `compileForward` method on a training runtime** when both
 * graphs use the same Module class — it auto-supplies `device` and
 * `sharedParams`. This standalone form is for forward-only models with no
 * training graph at all, or for sharing params across a different model.
 *
 * **Sharing params with a training compile.** Pass `opts.sharedParams =
 * trainCompiled.params` to bind this graph's param buffers to an existing
 * training runtime's GPU buffers — every train step is then immediately
 * visible to `run()` calls here, no copies.
 *
 * Initial param values are uploaded automatically for params *not* covered
 * by `sharedParams` (those are owned by the sibling compile).
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
  const outputBufferId = plan.tensorToBuffer.get(outputTensor.id)!
  const runtime = await createForwardRuntime(plan, kernels, outputBufferId, opts)

  uploadInitialParams(plan, materialized.initFns, runtime, opts.sharedParams)

  const ir: CompiledIR = { graph, paramGrads: {}, loss: outputTensor, plan, kernels }
  return Object.assign(runtime, { ir, kernelCount: countKernels(kernels) })
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

const countKernels = (kernels: KernelSpec[]): number => kernels.filter(k => k.wgsl).length

/** Wrap the runtime's step() to inject Adam's per-step `lrt` (bias-corrected
 *  effective LR) and, when the user supplied a per-step lr schedule, the
 *  decayShrink scalar. Also wraps resetOptimizerState() so a reset zeros
 *  Adam's m/v *and* the bias-correction step counter — otherwise the next
 *  step would skip Adam's warmup phase. */
function wrapStepForAdam(runtime: CompiledRuntime, adamResult: AdamResult): void {
  const { lrtInputName, decayShrinkInputName, config } = adamResult
  let t = 0
  const lrtBuf = new Float32Array(1)
  const decayShrinkBuf = decayShrinkInputName ? new Float32Array(1) : null
  const innerStep = runtime.step.bind(runtime) as CompiledRuntime['step']
  const innerReset = runtime.resetOptimizerState.bind(runtime)
  const wrappedStep = ((
    inputs: Record<string, Int32Array | Float32Array>,
    opts?: { withCaptures?: boolean },
  ) => {
    t++
    const lrNow = config.lr(t)
    lrtBuf[0] = lrNow * Math.sqrt(1 - Math.pow(config.b2, t)) / (1 - Math.pow(config.b1, t))
    const merged: Record<string, Int32Array | Float32Array> = { ...inputs, [lrtInputName]: lrtBuf }
    if (decayShrinkBuf && decayShrinkInputName) {
      decayShrinkBuf[0] = 1 - lrNow * config.weightDecay
      merged[decayShrinkInputName] = decayShrinkBuf
    }
    return opts?.withCaptures ? innerStep(merged, { withCaptures: true }) : innerStep(merged)
  }) as CompiledRuntime['step']
  runtime.step = wrappedStep
  runtime.resetOptimizerState = () => {
    t = 0
    innerReset()
  }
}

/** Build a Record<paramName, Float32Array> by running each param's init
 *  function against its shape and uploading them to the runtime. Skips any
 *  param covered by `sharedParams` (those are owned by a sibling compile). */
function uploadInitialParams(
  plan: BufferPlan,
  initFns: Record<string, InitFn>,
  runtime: CompiledRuntime | CompiledForward,
  sharedParams: Map<string, GPUBuffer> | undefined,
): void {
  const out: Record<string, Float32Array> = {}
  for (const [name, bufId] of plan.paramsByName) {
    if (sharedParams?.has(name)) continue
    const shape = plan.buffers[bufId]!.shape
    const size = shape.reduce((a, b) => a * b, 1)
    const initFn = initFns[name]
    if (!initFn) throw new Error(`compile: no init for param '${name}'`)
    out[name] = initFn(size, shape)
  }
  if (Object.keys(out).length > 0) runtime.uploadParams(out, { partial: !!sharedParams })
}
