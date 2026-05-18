// Import-safe spec for NeRF-tiny. Consumed by main.ts; also exports
// `irSpec` for paste into the NN Blueprint bulb.

import {
  Module, compile, Linear,
  mul, sub, mean, reshape, relu, sigmoid, concat,
  sin, cos, square,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'

export const IMG_W = 64
export const IMG_H = 64
export const N_PIXELS = IMG_W * IMG_H
export const BATCH_SIZE = 1024
export const L_FREQS = 8
export const HIDDEN = 64

export class NeRFTiny extends Module {
  l1 = new Linear(4 * L_FREQS, HIDDEN)
  l2 = new Linear(HIDDEN, HIDDEN)
  l3 = new Linear(HIDDEN, HIDDEN)
  l4 = new Linear(HIDDEN, 3)
}

function posEnc(coords: Tensor, freqs: Tensor): Tensor {
  const B = coords.shape[0]!
  const scaled = mul(reshape(coords, [B, 2, 1]), reshape(freqs, [1, 1, L_FREQS]))
  const sinF = reshape(sin(scaled), [B, 2 * L_FREQS])
  const cosF = reshape(cos(scaled), [B, 2 * L_FREQS])
  return concat([sinF, cosF], 1)
}

export function modelFwd(m: NeRFTiny, coords: Tensor, freqs: Tensor): Tensor {
  let h = posEnc(coords, freqs)
  h = relu(m.l1.fwd(h))
  h = relu(m.l2.fwd(h))
  h = relu(m.l3.fwd(h))
  return sigmoid(m.l4.fwd(h))
}

export function lossFn(
  m: NeRFTiny,
  { coords, rgb, freqs }: { coords: Tensor; rgb: Tensor; freqs: Tensor },
): Tensor {
  return mean(square(sub(modelFwd(m, coords, freqs), rgb)))
}

function predictFn(m: NeRFTiny, { coords, freqs }: { coords: Tensor; freqs: Tensor }): Tensor {
  return modelFwd(m, coords, freqs)
}

export const inputs = {
  coords: [BATCH_SIZE, 2],
  rgb:    [BATCH_SIZE, 3],
  freqs:  [L_FREQS],
} as const

export const predictInputs = {
  coords: [BATCH_SIZE, 2],
  freqs:  [L_FREQS],
} as const

export const optimizer = { kind: 'adam', lr: 1e-3 } as const

const trainingSpec = { model: new NeRFTiny(), loss: lossFn, inputs, optimizer }

export function compileTraining(): Promise<CompiledTraining<NeRFTiny>> {
  return compile(trainingSpec)
}

// Used by the NN Blueprint bulb to visualize this network as a computation graph.
// Paste the whole file at typebulb.com/u/samples/nn-blueprint/full to render it.
export const irSpec = {
  label: 'NeRF-tiny (image INR)',
  ...trainingSpec,
  predict: predictFn,
  predictInputs,
  dims: [
    { size: BATCH_SIZE,    name: 'B',  desc: 'batch (random pixels)' },
    { size: 2,             name: '2',  desc: 'xy coords' },
    { size: 3,             name: '3',  desc: 'RGB' },
    { size: L_FREQS,       name: 'L',  desc: 'frequency bands' },
    { size: 4 * L_FREQS,   name: '4L', desc: 'pos-enc features' },
    { size: 2 * L_FREQS,   name: '2L', desc: 'sin/cos features' },
    { size: HIDDEN,        name: 'H',  desc: 'hidden' },
  ],
}

