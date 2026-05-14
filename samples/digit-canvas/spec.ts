// Import-safe spec for the digit-canvas sample. The model is parameterized
// (hidden layer size + LR are runtime-controlled via UI dropdowns), so
// `compileTraining` takes those as arguments. Also exports `irSpec` for
// paste into the NN Blueprint bulb (uses defaults).

import {
  Module, compile, Linear, crossEntropy,
  relu, dropout, softmax,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'

export const INPUT_DIM = 784
export const N_CLASSES = 10
export const BATCH_SIZE = 64
export const EVAL_BATCH = 256
export const DROP_P = 0.1

export const DEFAULT_HIDDEN = 64
export const DEFAULT_LR = 1e-3

export class MLP extends Module {
  layers: Linear[]
  constructor(sizes: readonly number[]) {
    super()
    this.layers = []
    for (let i = 0; i < sizes.length - 1; i++) {
      this.layers.push(new Linear(sizes[i]!, sizes[i + 1]!))
    }
  }
}

function netFwd(m: MLP, x: Tensor, applyDropout: boolean): Tensor {
  let h = x
  for (let i = 0; i < m.layers.length; i++) {
    h = m.layers[i]!.fwd(h)
    if (i < m.layers.length - 1) {
      h = relu(h)
      if (applyDropout) h = dropout(h, DROP_P)
    }
  }
  return h
}

export function lossFn(m: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  const logits = netFwd(m, x, true)
  return crossEntropy(logits, y)
}

function predictFn(m: MLP, { x }: { x: Tensor }): Tensor {
  return softmax(netFwd(m, x, false))
}

const baseInputs = {
  x: [BATCH_SIZE, INPUT_DIM],
  y: { shape: [BATCH_SIZE], dtype: 'i32' },
} as const

export const predictInputs = { x: [BATCH_SIZE, INPUT_DIM] } as const

export type DigitInputs = typeof baseInputs

export function compileTraining(
  hidden: number = DEFAULT_HIDDEN,
  lr: number = DEFAULT_LR,
): Promise<CompiledTraining<MLP, DigitInputs>> {
  return compile({
    model: new MLP([INPUT_DIM, hidden, N_CLASSES]),
    loss: lossFn,
    optimizer: { kind: 'adamw', lr, weightDecay: 0.01, clipGradNorm: 1.0 },
    inputs: baseInputs,
  })
}

// Used by the NN Blueprint bulb to visualize this network as a computation graph.
// Paste the whole file at typebulb.com/u/samples/nn-blueprint/full to render it.
export const irSpec = {
  label: 'Digit Canvas (MNIST)',
  compile: () => compileTraining(),
  predict: predictFn,
  predictInputs,
  dims: [
    { size: BATCH_SIZE,     name: 'B',   desc: 'batch' },
    { size: INPUT_DIM,      name: '784', desc: 'pixels (28²)' },
    { size: DEFAULT_HIDDEN, name: 'H',   desc: `hidden (${DEFAULT_HIDDEN})` },
    { size: N_CLASSES,      name: 'K',   desc: 'classes' },
  ],
}

