// Public surface — `tensorgrad`. Documented end-user API: ops, modules,
// the `nn` namespace, `spec`/`compile`/`compileIR`, runtime helpers.
// Extension hooks (custom optimizers, IR walks, codegen visualization)
// live in the sibling `tensorgrad/internal` barrel.

export type { Tensor, Shape, Dtype, OpNode, Graph, CallSite } from './ir.js'
export { ShapeError } from './shape.js'
export { capture } from './capture.js'
export {
  // Element-wise arithmetic (binops accept Tensor or JS number)
  add, sub, mul, div, min, max, clamp,
  // Unary
  sqrt, rsqrt, log, exp, relu, neg, abs, tanh, sigmoid, sin, cos, gelu, silu,
  square,
  // Stochastic regularization
  dropout, randn,
  // Autograd control
  stopGradient,
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
  oneHot, arange, embedding, takeAlongAxis,
  // ML primitives
  softmaxCausal, logSoftmax, softmax, whereCausal,
  // 2D conv + pool (NCHW)
  conv2d, maxPool2d, nearestUpsample2d,
  type Conv2dOptions, type MaxPool2dOptions,
  // Slicing / structural
  narrow, concat, stack, split,
} from './ops.js'

export { lr, type AdamConfig, type AdamWConfig, type LR } from './adam.js'
export { type SGDConfig } from './sgd.js'
export { Captures, type UploadParamsOptions } from './runtime.js'
export {
  spec, compile, compileIR, isWebGPUAvailable,
  type CompiledIR, type TrainingSpecOptions, type ForwardSpecOptions,
  type TrainingSpec, type ForwardSpec, type Spec,
  type CompiledTraining, type CompiledForward, type OptimizerConfig,
  type StepResult, type RunResult, type QueueResult,
  type InputDecl, type InputDecls, type TypedInputs, type ParamTree,
} from './compile.js'
export { Module, init, type InitSpec, type ParamOptions } from './module.js'
export { singleFlight } from './single-flight.js'
export * as nn from './nn.js'
