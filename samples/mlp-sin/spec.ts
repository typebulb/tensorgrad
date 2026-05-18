// Import-safe spec: model + loss + compile shape, with no boot side effects.
// Consumed by main.ts; also exports `irSpec` for paste into the NN
// Blueprint bulb.

import {
  Module, compile, Linear, mul, sub, mean, relu,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'

export const HIDDEN = 64
export const B = 256
export const LR = 0.005

export class MLP extends Module {
  l1 = new Linear(1, HIDDEN)
  l2 = new Linear(HIDDEN, HIDDEN)
  l3 = new Linear(HIDDEN, 1)
}

export function modelFwd(p: MLP, x: Tensor): Tensor {
  return p.l3.fwd(relu(p.l2.fwd(relu(p.l1.fwd(x)))))
}

export function lossFn(p: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  const diff = sub(modelFwd(p, x), y)
  return mean(mul(diff, diff))
}

function predictFn(p: MLP, { x }: { x: Tensor }): Tensor {
  return modelFwd(p, x)
}

export const inputs = { x: [B, 1], y: [B, 1] } as const
export const predictInputs = { x: [B, 1] } as const
export const optimizer = { kind: 'adam', lr: LR } as const

const trainingSpec = { model: new MLP(), loss: lossFn, inputs, optimizer }

export function compileTraining(): Promise<CompiledTraining<MLP>> {
  return compile(trainingSpec)
}

// Used by the NN Blueprint bulb to visualize this network as a computation graph.
// Paste the whole file at typebulb.com/u/samples/nn-blueprint/full to render it.
export const irSpec = {
  label: 'MLP fits sin(x)',
  ...trainingSpec,
  predict: predictFn,
  predictInputs,
  dims: [
    { size: B,      name: 'B', desc: 'batch' },
    { size: HIDDEN, name: 'H', desc: 'hidden width' },
    { size: 1,      name: '1', desc: 'scalar in/out' },
  ],
}

