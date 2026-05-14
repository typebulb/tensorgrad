// Import-safe spec for KAN-tiny. Consumed by main.ts to drive the live
// sample.

import {
  Module, compile, init,
  add, sub, mul, mean, abs, neg, max, where, less, square,
  silu, reshape, swapAxes, matmul, arange,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'

export const G = 5
export const DEG = 3
export const KNOTS = G + DEG       // 8
export const HIDDEN = 4
export const BATCH = 256
export const LR = 0.005

export class KANLayer extends Module {
  C: Tensor       // spline coefficients [O, I, KNOTS]
  Wres: Tensor    // residual linear weight [I, O]
  constructor(public readonly I: number, public readonly O: number) {
    super()
    this.C = this.param([O, I, KNOTS], { init: init.randn({ scale: 0.1 }) })
    this.Wres = this.param([I, O])
  }
}

function bsplineBasis(u: Tensor): Tensor {
  const absU = abs(u)
  const u2 = mul(u, u)
  const absU3 = mul(absU, u2)
  const case1 = add(sub(mul(absU3, 0.5), u2), 2 / 3)
  const outerBase = max(neg(sub(absU, 2)), 0)
  const case2 = mul(mul(outerBase, mul(outerBase, outerBase)), 1 / 6)
  return where(less(absU, 1), case1, case2)
}

function kanFwd(layer: KANLayer, x: Tensor): Tensor {
  const B = x.shape[0]!
  const I = layer.I
  const O = layer.O
  const xScaled = mul(add(x, 1), (KNOTS - 1) / 2)
  const kVec = arange(KNOTS, 'f32')
  const u = sub(reshape(xScaled, [B, I, 1]), reshape(kVec, [1, 1, KNOTS]))
  const basis = bsplineBasis(u)
  const basisFlat = reshape(basis, [B, I * KNOTS])
  const CFlat = reshape(layer.C, [O, I * KNOTS])
  const Yspline = matmul(basisFlat, swapAxes(CFlat, 0, 1))
  const Yres = matmul(silu(x), layer.Wres)
  return add(Yspline, Yres)
}

export class KAN extends Module {
  l1 = new KANLayer(1, HIDDEN)
  l2 = new KANLayer(HIDDEN, 1)
}

export function modelFwd(m: KAN, x: Tensor): Tensor {
  return kanFwd(m.l2, kanFwd(m.l1, x))
}

export function lossFn(m: KAN, { x, y }: { x: Tensor; y: Tensor }): Tensor {
  return mean(square(sub(modelFwd(m, x), y)))
}

function predictFn(m: KAN, { x }: { x: Tensor }): Tensor {
  return modelFwd(m, x)
}

export const inputs = { x: [BATCH, 1], y: [BATCH, 1] } as const
export const predictInputs = { x: [BATCH, 1] } as const
export const optimizer = { kind: 'adam', lr: LR } as const

export function compileTraining(): Promise<CompiledTraining<KAN>> {
  return compile({ model: new KAN(), loss: lossFn, inputs, optimizer })
}

// Used by the NN Blueprint bulb to visualize this network as a computation graph.
// Paste the whole file at typebulb.com/u/samples/nn-blueprint/full to render it.
export const irSpec = {
  label: 'KAN-tiny (per-edge splines)',
  compile: compileTraining,
  predict: predictFn,
  predictInputs,
  dims: [
    { size: BATCH,  name: 'B',   desc: 'batch' },
    { size: HIDDEN, name: 'H',   desc: 'hidden width' },
    { size: KNOTS,  name: 'K',   desc: 'spline knots' },
    { size: 1,      name: '1',   desc: 'scalar I/O' },
  ],
}

