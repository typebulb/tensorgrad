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
import { createRuntime, type CompiledRuntime, type RuntimeOpts } from './runtime.js'
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

/**
 * Compile a Module-based model. The forward function takes the materialized
 * model and returns the loss tensor (typically by also calling tensorInput
 * for tokens/targets/masks inside).
 *
 * Walks the module tree to materialize params with auto-derived names, then
 * runs trace → grad → adam → buffer plan → codegen → runtime.
 *
 * If `opts.adam` is set, the runtime's `step()` automatically tracks an
 * internal step count and injects the bias-corrected `lrt` scalar each call;
 * users don't need to provide it themselves.
 */
export async function compileModule<M extends Module>(
  model: M,
  forward: (m: M, ...inputs: Tensor[]) => Tensor,
  opts: CompileModuleOptions = {},
): Promise<CompiledRuntime & { ir: CompiledIR }> {
  const inputDecls = opts.inputs ?? []
  let paramTensors: Record<string, Tensor> = {}
  const graph = trace(() => {
    paramTensors = materializeParams(model)
    const inputTensors = inputDecls.map(d => tensorInput(d.name, d.shape, d.dtype ?? 'f32'))
    return forward(model, ...inputTensors)
  })

  const { paramGrads, loss } = appendGrad(graph)

  let adamResult: ReturnType<typeof appendAdam> | undefined
  if (opts.adam) {
    adamResult = appendAdam(graph, paramGrads, paramTensors, opts.adam)
  }

  const plan = planBuffers(graph, paramGrads, adamResult?.writebacks ?? [])
  const kernels = emitKernels(graph, plan)
  const lossBufferId = plan.tensorToBuffer.get(loss.id)!
  const runtime = await createRuntime(plan, kernels, lossBufferId, opts)

  // If Adam is enabled, wrap step() to track the step count and supply lrt.
  if (adamResult) {
    const { lrtInputName, config } = adamResult
    let t = 0
    const lrtBuf = new Float32Array(1)
    const innerStep = runtime.step.bind(runtime)
    runtime.step = async (inputs) => {
      t++
      lrtBuf[0] = config.lr * Math.sqrt(1 - Math.pow(config.b2, t)) / (1 - Math.pow(config.b1, t))
      return innerStep({ ...inputs, [lrtInputName]: lrtBuf })
    }
  }

  const ir: CompiledIR = { graph, paramGrads, loss, plan, kernels }
  return Object.assign(runtime, { ir })
}
