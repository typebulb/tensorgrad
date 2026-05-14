// Import-safe spec for the flow-matching sample. Consumed by main.ts and
// by the IR viewer picker.

import {
  Module, compile, Linear, Embedding, Conv2d,
  add, mul, sub, mean, reshape, relu,
  randn, takeAlongAxis, square,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'
import type { IRSpec } from 'tensorgrad-viewer'

export const IMG_H = 28
export const IMG_W = 28
export const IMG_LEN = IMG_H * IMG_W
export const T_STEPS = 100
export const BATCH_SIZE = 64
export const CONV_CH = 32
export const EMB_DIM = 32

export class TinyFlow extends Module {
  tEmb  = new Embedding(T_STEPS + 1, EMB_DIM)
  tProj = new Linear(EMB_DIM, CONV_CH)
  conv1 = new Conv2d(1,       CONV_CH, 3, { padding: 1 })
  conv2 = new Conv2d(CONV_CH, CONV_CH, 3, { padding: 1 })
  conv3 = new Conv2d(CONV_CH, CONV_CH, 3, { padding: 1 })
  conv4 = new Conv2d(CONV_CH, 1,       3, { padding: 1 })
}

export function modelFwd(m: TinyFlow, x_t: Tensor, t: Tensor): Tensor {
  const B = x_t.shape[0]!
  const tFeat = reshape(m.tProj.fwd(m.tEmb.fwd(t)), [B, CONV_CH, 1, 1])
  let h = relu(m.conv1.fwd(x_t))
  h = add(h, tFeat)
  h = relu(m.conv2.fwd(h))
  h = relu(m.conv3.fwd(h))
  return m.conv4.fwd(h)
}

export function lossFn(
  m: TinyFlow,
  { x_0, t, tNorm_table }: { x_0: Tensor; t: Tensor; tNorm_table: Tensor },
): Tensor {
  const B = x_0.shape[0]!
  const x_1 = randn([B, 1, IMG_H, IMG_W])
  const tNorm = reshape(takeAlongAxis(tNorm_table, t, 0), [B, 1, 1, 1])
  const v_target = sub(x_1, x_0)
  const x_t = add(x_0, mul(tNorm, v_target))
  return mean(square(sub(modelFwd(m, x_t, t), v_target)))
}

export function predictFn(m: TinyFlow, { x_t, t }: { x_t: Tensor; t: Tensor }): Tensor {
  return modelFwd(m, x_t, t)
}

export const inputs = {
  x_0: [BATCH_SIZE, 1, IMG_H, IMG_W],
  t:            { shape: [BATCH_SIZE], dtype: 'i32' },
  tNorm_table:  [T_STEPS + 1],
} as const

export const optimizer = { kind: 'adam', lr: 2e-4, clipGradNorm: 1.0 } as const

export function compileTraining(): Promise<CompiledTraining<TinyFlow>> {
  return compile({ model: new TinyFlow(), loss: lossFn, inputs, optimizer })
}

export const irSpec: IRSpec = {
  label: 'Flow-matching-tiny (MNIST)',
  compile: compileTraining,
  dims: [
    { size: BATCH_SIZE, name: 'B',    desc: 'batch' },
    { size: 1,          name: 'C₁',   desc: 'image channel' },
    { size: CONV_CH,    name: 'C',    desc: 'conv channels' },
    { size: IMG_H,      name: '28',   desc: 'image H/W' },
    { size: EMB_DIM,    name: 'E',    desc: 'time embed dim' },
    { size: T_STEPS + 1, name: 'T+1', desc: 'time buckets' },
  ],
}
