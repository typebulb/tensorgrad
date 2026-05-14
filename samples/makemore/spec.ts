// Import-safe spec for makemore. Consumed by main.ts; also exports
// `irSpec` for paste into the NN Blueprint bulb.

import {
  Module, compile, lr,
  Linear, LayerNorm, crossEntropy,
  add, mul, sum, swapAxes,
  relu, matmul, embedding, arange,
  softmaxCausal, splitHeads, mergeHeads,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'

export const VOCAB = 27
export const D = 64
export const N_LAYERS = 3
export const N_HEADS = 4
export const D_HEAD = D / N_HEADS
export const SEQ_LEN = 17
export const T = SEQ_LEN - 1
export const B = 32
export const LR = lr.linear({ peak: 0.005, final: 0.0005, steps: 1500 })
const SCALE_QK = 1 / Math.sqrt(D_HEAD)

export class Attention extends Module {
  q = new Linear(D, D, { bias: false })
  k = new Linear(D, D, { bias: false })
  v = new Linear(D, D, { bias: false })
  o = new Linear(D, D, { bias: false })
}

export class MLP extends Module {
  up   = new Linear(D, 4 * D)
  down = new Linear(4 * D, D)
}

export class Block extends Module {
  ln1  = new LayerNorm(D)
  attn = new Attention()
  ln2  = new LayerNorm(D)
  mlp  = new MLP()
}

export class Transformer extends Module {
  tok_emb: Tensor; pos_emb: Tensor
  layers: Block[]
  lnf: LayerNorm
  constructor() {
    super()
    this.tok_emb = this.param([VOCAB, D])
    this.pos_emb = this.param([SEQ_LEN, D])
    this.layers = []
    for (let i = 0; i < N_LAYERS; i++) this.layers.push(new Block())
    this.lnf = new LayerNorm(D)
  }
}

function attentionFwd(p: Attention, x: Tensor): Tensor {
  const q = splitHeads(p.q.fwd(x), N_HEADS)
  const k = splitHeads(p.k.fwd(x), N_HEADS)
  const v = splitHeads(p.v.fwd(x), N_HEADS)
  const scores = mul(matmul(q, swapAxes(k, -1, -2)), SCALE_QK)
  return p.o.fwd(mergeHeads(matmul(softmaxCausal(scores), v)))
}

function mlpFwd(p: MLP, x: Tensor): Tensor {
  return p.down.fwd(relu(p.up.fwd(x)))
}

function blockFwd(p: Block, x: Tensor): Tensor {
  const a = attentionFwd(p.attn, p.ln1.fwd(x))
  const x1 = add(x, a)
  return add(x1, mlpFwd(p.mlp, p.ln2.fwd(x1)))
}

export function modelFwd(p: Transformer, tokens: Tensor): Tensor {
  const tokE = embedding(p.tok_emb, tokens)
  const posE = embedding(p.pos_emb, arange(T))
  let x = add(tokE, posE)
  for (let i = 0; i < p.layers.length; i++) x = blockFwd(p.layers[i]!, x)
  return matmul(p.lnf.fwd(x), swapAxes(p.tok_emb, -1, -2))
}

export function lossFn(p: Transformer, { tokens, targets, mask }: { tokens: Tensor; targets: Tensor; mask: Tensor }): Tensor {
  return sum(mul(crossEntropy(modelFwd(p, tokens), targets, { reduction: 'none' }), mask))
}

function predictFwd(p: Transformer, { tokens }: { tokens: Tensor }): Tensor {
  return modelFwd(p, tokens)
}

export const inputs = {
  tokens:  { shape: [B, T], dtype: 'i32' },
  targets: { shape: [B, T], dtype: 'i32' },
  mask:    [B, T],
} as const

export const predictInputs = {
  tokens: { shape: [B, T], dtype: 'i32' },
} as const

export const optimizer = { kind: 'adamw', lr: LR, weightDecay: 0.01 } as const

export function compileTraining(): Promise<CompiledTraining<Transformer>> {
  return compile({ model: new Transformer(), loss: lossFn, inputs, optimizer })
}

// Used by the NN Blueprint bulb to visualize this network as a computation graph.
// Paste the whole file at typebulb.com/u/samples/nn-blueprint/full to render it.
export const irSpec = {
  label: 'Makemore (name generation)',
  compile: compileTraining,
  predict: predictFwd,
  predictInputs,
  dims: [
    { size: B,       name: 'B',  desc: 'batch' },
    { size: T,       name: 'T',  desc: 'seq len (16)' },
    { size: SEQ_LEN, name: 'T+1', desc: 'full seq incl. last' },
    { size: D,       name: 'D',  desc: 'model dim' },
    { size: N_HEADS, name: 'H',  desc: 'heads' },
    { size: D_HEAD,  name: 'D/H', desc: 'per-head dim' },
    { size: VOCAB,   name: 'V',  desc: 'vocab (27)' },
    { size: 4 * D,   name: '4D', desc: 'MLP hidden' },
  ],
}

