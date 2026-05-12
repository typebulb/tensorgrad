// Public surface — everything tensorgrad exposes lives behind this barrel.
// IR types, op surface, trace driver, autograd, codegen, runtime, compile
// entry points, Module abstraction, and the `nn` namespace.

export type { Tensor, Shape, Dtype, OpNode, Graph, CallSite } from './ir.js'
export { ShapeError } from './shape.js'
export { trace, traceInto, paramInput, tensorInput, stateInput } from './trace.js'
export { capture, unsplitHeads } from './capture.js'
export {
  // Element-wise arithmetic (binops accept Tensor or JS number)
  add, sub, mul, div, min, max, clamp,
  // Unary
  sqrt, rsqrt, log, exp, relu, neg, abs, tanh, sigmoid, sin, cos, gelu, silu,
  square,
  // Stochastic regularization
  dropout, randn,
  // Comparisons + select
  less, greater, where,
  // Reductions
  mean, sum, argmax, argmin,
  type ReduceOpts,
  // Shape
  reshape, permute, swapAxes, flatten,
  splitHeads, mergeHeads,
  // Linear algebra
  matmul,
  // Indexing / casting
  oneHot, arange, embedding, take,
  // ML primitives
  softmaxCausal, logSoftmax, softmax, whereCausal,
  // 2D conv + pool (NCHW)
  conv2d, maxPool2d, nearestUpsample2d,
  type Conv2dOptions, type MaxPool2dOptions,
  // Slicing / structural
  sliceRange, concat, stack, split,
} from './ops.js'

export { appendGrad, type GradResult } from './grad.js'
export { appendAdam, appendGradClip, lr, resolveLR, type AdamConfig, type AdamResult, type LR } from './adam.js'
export { appendSGD, type SGDConfig, type SGDResult } from './sgd.js'
export { planBuffers, type BufferPlan, type BufferSpec, type Writeback, type WritebackDecl } from './buffers.js'
export { emitKernels, type KernelSpec } from './codegen.js'
export { Captures, type RunOptions, type StepResult, type RunResult, type Outcome, type UploadParamsOptions } from './runtime.js'
export {
  compileToIR, compileModule, isWebGPUAvailable,
  type CompiledIR, type CompileModuleOptions, type CompileForwardMethodOptions,
  type CompiledModule, type CompiledForwardModule, type OptimizerConfig,
  type InputDecl, type InputDeclObject, type InputDecls, type InputShape, type InputsTensors, type ForwardFn,
  type DtypeOf, type TypedArrayFor, type TypedInputs, type ParamTree,
} from './compile.js'
export {
  Module, materializeParams, init, mulberry32,
  type InitSpec, type ParamOptions, type MaterializedParams, type InitFn, type Rng,
} from './module.js'
export { singleFlight } from './single-flight.js'
export * as nn from './nn.js'
