// Public surface. Bulb code imports from here.
//
// Phase 1 exports: IR types, op surface, trace driver. Autograd (Phase 2) and
// codegen / compile() (Phase 3+) come later.

export type { Tensor, Shape, Dtype, OpNode, Graph, CallSite } from './ir.js'
export { ShapeError } from './shape.js'
export { trace, traceInto, paramInput, tensorInput, stateInput } from './trace.js'
export { capture } from './capture.js'
export {
  // Element-wise arithmetic. The binops accept Tensor or JS-number for the second arg.
  add, sub, mul, div,
  // Element-wise unary
  sqrt, rsqrt, log, exp, relu,
  // Comparisons + select
  less, greater, where,
  // Reductions over the last axis (other axes via reshape/transpose first)
  meanLast, sumLast,
  // Shape ops
  reshape, transpose,
  // Linear algebra
  matmul, matmulBatched,
  // Indexing / casting
  oneHot, arange,
  // ML primitives — fused for the transformer
  softmaxCausalLast, logSoftmaxLast, whereCausal,
  // Slicing
  sliceLastRange,
} from './ops.js'

// Note: addScalar/mulScalar/broadcastTo/sumToShape/constScalar/reluGrad/adam_update_*
// are autograd/optimizer building blocks. They live in ops.ts (so grad.ts and
// adam.ts can import them) but aren't part of the public API — `add`/`mul`
// overload on JS numbers, `where` subsumes the rest.
export { appendGrad, type GradResult } from './grad.js'
export { appendAdam, type AdamConfig, type AdamResult } from './adam.js'
export { planBuffers, type BufferPlan, type BufferSpec, type Writeback, type WritebackDecl } from './buffers.js'
export { emitKernels, type KernelSpec } from './codegen.js'
export { createRuntime, createForwardRuntime, type CompiledRuntime, type CompiledForward, type RuntimeOpts, type StepOptions, type StepWithCaptures, type RunOptions, type RunWithCaptures } from './runtime.js'
export { compile, compileToIR, compileModule, compileForward, type CompiledIR, type CompileModuleOptions, type CompileForwardOptions, type InputDecl } from './compile.js'
export { Module, materializeParams, type InitSpec, type ParamOptions, type MaterializedParams } from './module.js'
export * as nn from './nn.js'
