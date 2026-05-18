// Extension barrel — `tensorgrad/internal`. Contracts for custom optimizers,
// codegen / IR-only consumers, and trace-time wiring you'd reach for when
// building something on top of tensorgrad rather than just calling
// `spec` + `compile`. Public end-user code should import from `tensorgrad`.

export { traceFn, traceInto, paramInput, tensorInput, stateInput } from './trace.js'
export { appendGrad, type GradResult } from './grad.js'
export {
  appendAdam, appendGradClip, resolveLR,
  type AdamResult,
} from './adam.js'
export { appendSGD, type SGDResult } from './sgd.js'
export {
  planBuffers,
  type BufferPlan, type BufferSpec, type Writeback, type WritebackDecl,
} from './buffers.js'
export { emitKernels, type KernelSpec } from './codegen.js'
export {
  materializeParams, mulberry32,
  type MaterializedParams, type InitFn, type Rng,
} from './module.js'
