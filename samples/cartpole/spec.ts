// Import-safe spec for the CartPole REINFORCE sample. Consumed by main.ts
// to drive the live sample.

import {
  Module, compile, Linear,
  mul, sum,
  tanh, oneHot, logSoftmax, softmax,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'

export const K = 16
export const MAX_T = 200
export const STATE_DIM = 4
export const N_ACTIONS = 2
export const HIDDEN = 16
export const LR = 5e-3

export class Policy extends Module {
  l1 = new Linear(STATE_DIM, HIDDEN)
  l2 = new Linear(HIDDEN, N_ACTIONS)
}

export function policyLogits(m: Policy, state: Tensor): Tensor {
  return m.l2.fwd(tanh(m.l1.fwd(state)))
}

export function lossFn(
  m: Policy,
  { states, actions, returns, mask }:
    { states: Tensor; actions: Tensor; returns: Tensor; mask: Tensor },
): Tensor {
  const logits = policyLogits(m, states)
  const logProbs = logSoftmax(logits, -1)
  const taken = sum(mul(logProbs, oneHot(actions, N_ACTIONS, 'f32')), -1)
  return mul(sum(mul(mul(taken, returns), mask)), -1 / (MAX_T * K))
}

function predictFn(m: Policy, { state }: { state: Tensor }): Tensor {
  return softmax(policyLogits(m, state), -1)
}

export const inputs = {
  states:  [MAX_T * K, STATE_DIM],
  actions: { shape: [MAX_T * K], dtype: 'i32' },
  returns: [MAX_T * K],
  mask:    [MAX_T * K],
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
  label: 'CartPole REINFORCE',
  ...trainingSpec,
  predict: predictFn,
  predictInputs,
  dims: [
    { size: MAX_T * K,  name: 'N',   desc: 'rollout slots (MAX_T·K)' },
    { size: K,          name: 'K',   desc: 'parallel envs' },
    { size: STATE_DIM,  name: 'S',   desc: 'state dim' },
    { size: N_ACTIONS,  name: 'A',   desc: 'actions' },
    { size: HIDDEN,     name: 'H',   desc: 'hidden' },
  ],
}

