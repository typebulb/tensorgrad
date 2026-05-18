// Import-safe spec for the tic-tac-toe self-play sample. Consumed by
// main.ts; also exports `irSpec` for paste into the NN Blueprint bulb.

import {
  Module, compile, Linear,
  mul, sum,
  tanh, oneHot, logSoftmax, softmax,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'

export const K = 16
export const MAX_MOVES = 9
export const N_SLOTS = K * MAX_MOVES
export const STATE_DIM = 27
export const N_ACTIONS = 9
export const HIDDEN = 64
export const LR = 5e-3

export class Policy extends Module {
  l1 = new Linear(STATE_DIM, HIDDEN)
  l2 = new Linear(HIDDEN,    HIDDEN)
  out = new Linear(HIDDEN,    N_ACTIONS)
}

export function policyLogits(m: Policy, state: Tensor): Tensor {
  return m.out.fwd(tanh(m.l2.fwd(tanh(m.l1.fwd(state)))))
}

export function lossFn(
  m: Policy,
  { states, actions, outcomes, mask }:
    { states: Tensor; actions: Tensor; outcomes: Tensor; mask: Tensor },
): Tensor {
  const logProbs = logSoftmax(policyLogits(m, states), -1)
  const taken = sum(mul(logProbs, oneHot(actions, N_ACTIONS, 'f32')), -1)
  return mul(sum(mul(mul(taken, outcomes), mask)), -1 / N_SLOTS)
}

function predictFn(m: Policy, { state }: { state: Tensor }): Tensor {
  return softmax(policyLogits(m, state), -1)
}

export const inputs = {
  states:   [N_SLOTS, STATE_DIM],
  actions:  { shape: [N_SLOTS], dtype: 'i32' },
  outcomes: [N_SLOTS],
  mask:     [N_SLOTS],
} as const

export const predictInputs = { state: [K, STATE_DIM] } as const

export const optimizer = { kind: 'adam', lr: LR } as const

const trainingSpec = { model: new Policy(), loss: lossFn, inputs, optimizer }

export function compileTraining(): Promise<CompiledTraining<Policy>> {
  return compile(trainingSpec)
}

// Used by the NN Blueprint bulb to visualize this network as a computation graph.
// Paste the whole file at typebulb.com/u/samples/nn-blueprint/full to render it.
export const irSpec = {
  label: 'Tic-Tac-Toe self-play',
  ...trainingSpec,
  predict: predictFn,
  predictInputs,
  dims: [
    { size: N_SLOTS,   name: 'N',  desc: 'rollout slots (K·9)' },
    { size: K,         name: 'K',  desc: 'parallel games' },
    { size: STATE_DIM, name: '27', desc: 'state (3ch·9cells)' },
    { size: N_ACTIONS, name: '9',  desc: 'actions / cells' },
    { size: HIDDEN,    name: 'H',  desc: 'hidden' },
  ],
}

