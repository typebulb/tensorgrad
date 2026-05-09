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
import { appendAdam, type AdamConfig } from './adam.js'
import { planBuffers, type BufferPlan } from './buffers.js'
import { emitKernels, type KernelSpec } from './codegen.js'
import { createRuntime, createForwardRuntime, type CompiledRuntime, type CompiledForward, type RuntimeOpts } from './runtime.js'
import { Module, materializeParams } from './module.js'

/** Declares one input tensor of the model's forward function. Order matches
 *  the function's parameter list (after `model`). The `name` is used at
 *  runtime to upload data via `step({ [name]: data })`. */
export interface InputDecl {
  name: string
  shape: Shape
  dtype?: Dtype
}

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

export interface CompileModuleOptions extends RuntimeOpts {
  /** Per-step data inputs to the forward function. Order matches the forward
   *  function's parameters (after the model). e.g. for
   *  `(model, tokens, targets, mask) => loss`, inputs is
   *  `[{name:'tokens',...}, {name:'targets',...}, {name:'mask',...}]`. */
  inputs?: InputDecl[]
  /** Adam hyperparameters. If omitted, no optimizer is appended (forward-only). */
  adam?: AdamConfig
}

export interface CompileForwardOptions extends RuntimeOpts {
  /** Per-step data inputs to the forward function. */
  inputs?: InputDecl[]
}

/**
 * Compile a Module-based model. Pass a *factory* `() => new Model()`, not the
 * model instance itself: compilation mutates the tree (every `ParamSentinel`
 * field becomes a real `Tensor`), so the instance is consumed and shouldn't be
 * referenced afterwards. Re-call the factory if you need a fresh tree.
 *
 * The forward function takes the materialized model and returns the loss
 * tensor.
 *
 * Walks the module tree to materialize params with auto-derived names, then
 * runs trace → grad → adam → buffer plan → codegen → runtime.
 *
 * If `opts.adam` is set, the runtime's `step()` automatically tracks an
 * internal step count and injects the bias-corrected `lrt` scalar each call;
 * users don't need to provide it themselves.
 */
export async function compileModule<M extends Module>(
  modelFactory: () => M,
  forward: (m: M, ...inputs: Tensor[]) => Tensor,
  opts: CompileModuleOptions = {},
): Promise<CompiledRuntime & { ir: CompiledIR; uploadInitialParams: () => void }> {
  const inputDecls = opts.inputs ?? []
  const model = modelFactory()
  let materialized: ReturnType<typeof materializeParams> = { tensors: {}, initFns: {} }
  const graph = trace(() => {
    materialized = materializeParams(model)
    const inputTensors = inputDecls.map(d => tensorInput(d.name, d.shape, d.dtype ?? 'f32'))
    return forward(model, ...inputTensors)
  })

  const { paramGrads, loss } = appendGrad(graph)

  let adamResult: ReturnType<typeof appendAdam> | undefined
  if (opts.adam) {
    adamResult = appendAdam(graph, paramGrads, materialized.tensors, opts.adam)
  }

  const plan = planBuffers(graph, paramGrads, adamResult?.writebacks ?? [])
  const kernels = emitKernels(graph, plan)
  const lossBufferId = plan.tensorToBuffer.get(loss.id)!
  const runtime = await createRuntime(plan, kernels, lossBufferId, opts)

  // If Adam is enabled, wrap step() to track the step count and supply lrt.
  // Wrap resetOptimizerState() too, so a reset zeros m/v *and* the bias-correction
  // counter — otherwise the next step would skip Adam's warmup phase.
  if (adamResult) {
    const { lrtInputName, config } = adamResult
    let t = 0
    const lrtBuf = new Float32Array(1)
    const innerStep = runtime.step.bind(runtime) as CompiledRuntime['step']
    const innerReset = runtime.resetOptimizerState.bind(runtime)
    const wrappedStep = (
      inputs: Record<string, Int32Array | Float32Array>,
      opts?: { withCaptures?: boolean },
    ): Promise<number | { loss: number; captures: Record<string, Float32Array> }> => {
      t++
      lrtBuf[0] = config.lr * Math.sqrt(1 - Math.pow(config.b2, t)) / (1 - Math.pow(config.b1, t))
      const merged = { ...inputs, [lrtInputName]: lrtBuf }
      return opts?.withCaptures ? innerStep(merged, { withCaptures: true }) : innerStep(merged)
    }
    runtime.step = wrappedStep as CompiledRuntime['step']
    runtime.resetOptimizerState = () => {
      t = 0
      innerReset()
    }
  }

  const { initFns } = materialized
  const uploadInitialParams = () => {
    const out: Record<string, Float32Array> = {}
    for (const [name, bufId] of plan.paramsByName) {
      const shape = plan.buffers[bufId]!.shape
      const size = shape.reduce((a, b) => a * b, 1)
      const initFn = initFns[name]
      if (!initFn) throw new Error(`uploadInitialParams: no init for param '${name}'`)
      out[name] = initFn(size, shape)
    }
    runtime.uploadParams(out)
  }

  const ir: CompiledIR = { graph, paramGrads, loss, plan, kernels }
  return Object.assign(runtime, { ir, uploadInitialParams })
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
 * **Sharing params with a training compile.** Pass `opts.sharedParams =
 * trainCompiled.params` to bind this graph's param buffers to an existing
 * training runtime's GPU buffers — every train step is then immediately
 * visible to `run()` calls here, no copies. The forward graph's
 * `uploadInitialParams()` skips any param covered by `sharedParams`.
 *
 * Typical use: a B=1 inference graph alongside a B=512 training graph,
 * built from the same `Module` factory.
 */
export async function compileForward<M extends Module>(
  modelFactory: () => M,
  forward: (m: M, ...inputs: Tensor[]) => Tensor,
  opts: CompileForwardOptions = {},
): Promise<CompiledForward & { ir: CompiledIR; uploadInitialParams: () => void }> {
  const inputDecls = opts.inputs ?? []
  const model = modelFactory()
  let materialized: ReturnType<typeof materializeParams> = { tensors: {}, initFns: {} }
  const graph = trace(() => {
    materialized = materializeParams(model)
    const inputTensors = inputDecls.map(d => tensorInput(d.name, d.shape, d.dtype ?? 'f32'))
    return forward(model, ...inputTensors)
  })

  const plan = planBuffers(graph, /* paramGrads */ {})
  const kernels = emitKernels(graph, plan)
  const outputTensor = graph.tensors[graph.outputs[0]!]!
  const outputBufferId = plan.tensorToBuffer.get(outputTensor.id)!
  const runtime = await createForwardRuntime(plan, kernels, outputBufferId, opts)

  const sharedParams = opts.sharedParams
  const { initFns } = materialized
  const uploadInitialParams = () => {
    const out: Record<string, Float32Array> = {}
    let needsUpload = false
    for (const [name, bufId] of plan.paramsByName) {
      // Skip params covered by sharedParams — those are owned by the providing
      // compile and already initialized there.
      if (sharedParams?.has(name)) continue
      const shape = plan.buffers[bufId]!.shape
      const size = shape.reduce((a, b) => a * b, 1)
      const initFn = initFns[name]
      if (!initFn) throw new Error(`uploadInitialParams: no init for param '${name}'`)
      out[name] = initFn(size, shape)
      needsUpload = true
    }
    if (needsUpload) runtime.uploadParams(out, { partial: !!sharedParams })
  }

  // CompiledIR.loss is the field name; for forward-only, it carries the user's
  // returned tensor (e.g., logits). Same shape conceptually; just no autograd.
  const ir: CompiledIR = { graph, paramGrads: {}, loss: outputTensor, plan, kernels }
  return Object.assign(runtime, { ir, uploadInitialParams })
}
