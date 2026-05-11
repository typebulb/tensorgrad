// Public surface. Bulb code imports from here.
//
// Phase 1 exports: IR types, op surface, trace driver. Autograd (Phase 2) and
// codegen / compile() (Phase 3+) come later.

export type { Tensor, Shape, Dtype, OpNode, Graph, CallSite } from './ir.js'
export { ShapeError } from './shape.js'
export { trace, traceInto, paramInput, tensorInput, stateInput } from './trace.js'
export { capture } from './capture.js'
export {
  // Element-wise arithmetic. Binops accept Tensor or JS-number for the second arg.
  add, sub, mul, div, min, max, clamp,
  // Element-wise unary
  sqrt, rsqrt, log, exp, relu,
  neg, abs, tanh, sigmoid, gelu, silu,
  // Stochastic regularization (training-forward only; no mode flag)
  dropout,
  // Comparisons + select
  less, greater, where,
  // Reductions: `mean(x, axis?, { keepDims? })` / `sum(x, axis?, { keepDims? })`.
  // Negative axis counts from the end; omit `axis` to reduce all → 0-d scalar.
  mean, sum, argmaxLast,
  type ReduceOpts,
  // Shape ops
  reshape, transpose, swapAxes,
  // PyTorch-style flatten — collapse trailing axes into one. Sugar for reshape(-1).
  flatten,
  // Linear algebra
  matmul, matmulBatched,
  // Indexing / casting
  oneHot, arange, embedding,
  // ML primitives — fused for the transformer
  softmaxCausalLast, logSoftmaxLast, softmaxLast, whereCausal,
  // 2D convolution + pooling (NCHW, PyTorch-shape). conv2d/maxPool2D are the
  // user surface; the *Grad helpers are emitted by autograd.
  conv2d, maxPool2D,
  type Conv2dOptions, type MaxPool2dOptions,
  // Slicing / structural
  sliceLastRange, sliceRange, concat, stack, split,
} from './ops.js'

// Note: addScalar/mulScalar/broadcastTo/sumToShape/constScalar/reluGrad/adam_update_*
// are autograd/optimizer building blocks. They live in ops.ts (so grad.ts and
// adam.ts can import them) but aren't part of the public API — `add`/`mul`
// overload on JS numbers, `where` subsumes the rest.
export { appendGrad, type GradResult } from './grad.js'
export { appendAdam, appendGradClip, lr, resolveLR, type AdamConfig, type AdamResult, type LR } from './adam.js'
export { appendSGD, type SGDConfig, type SGDResult } from './sgd.js'
export { planBuffers, type BufferPlan, type BufferSpec, type Writeback, type WritebackDecl } from './buffers.js'
export { emitKernels, type KernelSpec } from './codegen.js'
// Runtime types: only the user-facing pieces. CompiledRuntime/CompiledForward
// (worker-internal) and createRuntime/createForwardRuntime aren't part of the
// public API — users get CompiledModule/CompiledForwardModule (proxies) from
// compileModule/compileForward instead.
export { Captures, type RunOptions, type StepResult, type RunResult, type Outcome, type UploadParamsOptions } from './runtime.js'
export {
  compileToIR, compileModule, isWebGPUAvailable,
  type CompiledIR, type CompileModuleOptions, type CompileForwardMethodOptions, type OptimizerConfigUpdate,
  type CompiledModule, type CompiledForwardModule,
  type InputDecl, type InputDeclObject, type InputDecls, type InputShape, type InputsTensors, type ForwardFn,
  type DtypeOf, type TypedArrayFor, type TypedInputs, type ParamTree,
} from './compile.js'
export {
  Module, materializeParams, init, mulberry32,
  type InitSpec, type ParamOptions, type MaterializedParams, type InitFn, type Rng,
} from './module.js'
export { singleFlight } from './single-flight.js'
export * as nn from './nn.js'
