// Public surface — `tensorgrad`. Documented end-user API: ops, layer
// modules (Linear, LayerNorm, ...), `compile`, runtime helpers.

export type { Tensor, Shape, Dtype, OpNode, Graph, CallSite } from './ir.js'
export { getOpInputs } from './ir.js'
export { ShapeError } from './shape.js'
export { capture } from './capture.js'
export {
  // Element-wise arithmetic (binops accept Tensor or JS number)
  add, sub, mul, div, min, max, clamp,
  // Unary
  sqrt, rsqrt, log, exp, relu, neg, abs, tanh, sigmoid, sin, cos, gelu, silu, leakyRelu, softplus,
  square,
  // Stochastic regularization
  dropout, randn,
  // Autograd control
  stopGradient,
  // Comparisons + select
  less, greater, where,
  // Reductions
  mean, sum, argmax, argmin,
  type ReduceOptions,
  // Shape
  reshape, permute, swapAxes,
  splitHeads, mergeHeads, rope,
  type RopeOptions,
  // Linear algebra
  matmul,
  // Indexing / casting
  oneHot, arange, embedding, takeAlongAxis,
  // Const-tensor builders
  zeros, ones,
  // ML primitives
  softmaxCausal, logSoftmax, softmax, whereCausal, categorical,
  // 2D conv + pool (NCHW)
  conv2d, maxPool2d, nearestUpsample2d,
  type Conv2dOptions, type MaxPool2dOptions,
  // Slicing / structural
  narrow, concat, stack, split,
} from './ops.js'

export { lr, type AdamConfig, type AdamWConfig, type LR } from './adam.js'
export { type SGDConfig } from './sgd.js'
export { Captures } from './runtime.js'
export {
  compile, trace, traceForward, isWebGPUAvailable,
  type TrainingSpec, type ForwardSpec, type CompiledIR, type ForwardFn,
  type CompiledTraining, type CompiledForward, type OptimizerConfig,
  type StepResult, type RunResult,
  type InputDecl, type InputDecls, type TypedInputs,
} from './compile.js'
export { Module, init, type InitSpec, type ParamOptions } from './module.js'
export { singleFlight, type SingleFlightResult } from './single-flight.js'
export {
  Linear, LayerNorm, RMSNorm, Embedding, Conv2d,
  crossEntropy, nllLoss,
  type LinearOptions, type EmbeddingOptions, type Conv2dLayerOptions,
  type LayerNormOptions, type RMSNormOptions,
  type LossOptions,
} from './nn.js'
