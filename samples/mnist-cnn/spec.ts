// Import-safe spec for the MNIST CNN: model + loss + compile shape, no boot
// side effects. Consumed by main.ts; also exports `irSpec` for paste into
// the NN Blueprint bulb.

import {
  Module, compile, Linear, Conv2d, crossEntropy,
  relu, flatten, maxPool2d,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'

export const BATCH_SIZE = 64
export const EVAL_BATCH = 256
export const N_CLASSES = 10
export const CONV1_OUT = 16
export const CONV2_OUT = 32
export const HIDDEN = 64

export class CNN extends Module {
  conv1 = new Conv2d(1, CONV1_OUT, 3, { padding: 1 })
  conv2 = new Conv2d(CONV1_OUT, CONV2_OUT, 3, { padding: 1 })
  // After two 2x2 pools: 28 → 14 → 7. Conv2 output is [B, 32, 7, 7] → 1568.
  fc1 = new Linear(CONV2_OUT * 7 * 7, HIDDEN)
  fc2 = new Linear(HIDDEN, N_CLASSES)
}

export function forwardLogits(m: CNN, x: Tensor): Tensor {
  let h = relu(m.conv1.fwd(x))
  h = maxPool2d(h, 2)
  h = relu(m.conv2.fwd(h))
  h = maxPool2d(h, 2)
  h = flatten(h, 1)
  h = relu(m.fc1.fwd(h))
  return m.fc2.fwd(h)
}

export function lossFn(m: CNN, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  return crossEntropy(forwardLogits(m, x), y)
}

export function predictFn(m: CNN, { x }: { x: Tensor }): Tensor {
  return forwardLogits(m, x)
}

export const inputs = {
  x: [BATCH_SIZE, 1, 28, 28],
  y: { shape: [BATCH_SIZE], dtype: 'i32' },
} as const

export const optimizer = {
  kind: 'adamw', lr: 1e-3, weightDecay: 0.01, clipGradNorm: 1.0,
} as const

export function compileTraining(): Promise<CompiledTraining<CNN>> {
  return compile({ model: new CNN(), loss: lossFn, inputs, optimizer })
}

// Used by the NN Blueprint bulb to visualize this network as a computation graph.
// Paste the whole file at typebulb.com/u/samples/nn-blueprint/full to render it.
export const irSpec = {
  label: 'MNIST CNN',
  compile: compileTraining,
  dims: [
    { size: BATCH_SIZE, name: 'B',   desc: 'batch' },
    { size: 1,          name: 'Cin', desc: 'input channels' },
    { size: CONV1_OUT,  name: 'C1',  desc: 'conv1 channels' },
    { size: CONV2_OUT,  name: 'C2',  desc: 'conv2 channels' },
    { size: 28,         name: '28',  desc: 'image H/W' },
    { size: 14,         name: '14',  desc: 'after pool1' },
    { size: 7,          name: '7',   desc: 'after pool2' },
    { size: CONV2_OUT * 7 * 7, name: '1568', desc: 'flatten size' },
    { size: HIDDEN,     name: 'H',   desc: 'fc hidden' },
    { size: N_CLASSES,  name: 'K',   desc: 'classes' },
  ],
}

