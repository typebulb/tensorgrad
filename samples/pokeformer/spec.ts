// Import-safe spec for the Pokéformer sample. Consumed by main.ts; also
// exports `irSpec` for paste into the NN Blueprint bulb.
//
// Tiny Llama-style decoder transformer: RMSNorm pre-norm, RoPE on Q/K,
// tied input/output embeddings. The inference forward bakes `categorical`
// sampling into the graph with a runtime temperature input, so the model
// returns sampled token ids directly (i32 readback) — exercising the
// in-graph stochastic + i32-output paths end-to-end.

import {
  Module, compile, lr,
  Linear, RMSNorm, crossEntropy,
  add, mul, sum, div, matmul, swapAxes, reshape, embedding,
  splitHeads, mergeHeads, rope, softmaxCausal, categorical,
  relu,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'

export const VOCAB = 27        // a-z + EOS
export const EOS = 26
export const T = 16
export const D = 64
export const N_LAYERS = 2
export const N_HEADS = 4
export const D_HEAD = D / N_HEADS
export const FFN = 4 * D
export const B = 32
export const B_INFER = 1
export const LR = lr.linear({ peak: 0.005, final: 0.0005, steps: 1500 })
const SCALE_QK = 1 / Math.sqrt(D_HEAD)

export class Attention extends Module {
  q = new Linear(D, D, { bias: false })
  k = new Linear(D, D, { bias: false })
  v = new Linear(D, D, { bias: false })
  o = new Linear(D, D, { bias: false })
}

export class MLP extends Module {
  up   = new Linear(D, FFN)
  down = new Linear(FFN, D)
}

export class Block extends Module {
  n1   = new RMSNorm(D)
  attn = new Attention()
  n2   = new RMSNorm(D)
  mlp  = new MLP()
}

export class NameLM extends Module {
  // Tied input/output embedding — used as lookup table on the way in and
  // (transposed) as the LM head on the way out.
  tok_emb: Tensor
  layers: Block[]
  nf: RMSNorm
  constructor() {
    super()
    this.tok_emb = this.param([VOCAB, D])
    this.layers = []
    for (let i = 0; i < N_LAYERS; i++) this.layers.push(new Block())
    this.nf = new RMSNorm(D)
  }
}

function attnFwd(p: Attention, x: Tensor): Tensor {
  // splitHeads: [B, T, D] → [B, H, T, D/H]
  const q = splitHeads(p.q.fwd(x), N_HEADS)
  const k = splitHeads(p.k.fwd(x), N_HEADS)
  const v = splitHeads(p.v.fwd(x), N_HEADS)
  // RoPE rotates Q and K per-head before the scores matmul — V is left alone.
  const [qr, kr] = rope(q, k)
  const scores = mul(matmul(qr, swapAxes(kr, -1, -2)), SCALE_QK)
  return p.o.fwd(mergeHeads(matmul(softmaxCausal(scores), v)))
}

function mlpFwd(p: MLP, x: Tensor): Tensor {
  return p.down.fwd(relu(p.up.fwd(x)))
}

function blockFwd(p: Block, x: Tensor): Tensor {
  const a = attnFwd(p.attn, p.n1.fwd(x))
  const x1 = add(x, a)
  return add(x1, mlpFwd(p.mlp, p.n2.fwd(x1)))
}

function logitsFwd(m: NameLM, tokens: Tensor): Tensor {
  let x = embedding(m.tok_emb, tokens)
  for (const block of m.layers) x = blockFwd(block, x)
  const xn = m.nf.fwd(x)
  return matmul(xn, swapAxes(m.tok_emb, -1, -2))   // tied LM head
}

// Mask weights non-padding positions only. Without it, ~half of every target
// sequence is post-EOS padding predicting post-EOS padding — trivially zero
// loss within the first couple of steps, which drags reported loss far below
// the actual prefix-prediction difficulty and starves the hard positions
// of gradient. Mask sums to 1 per batch so loss is per-valid-token mean.
export function lossFn(m: NameLM, { tokens, targets, mask }: { tokens: Tensor; targets: Tensor; mask: Tensor }): Tensor {
  const ce = crossEntropy(logitsFwd(m, tokens), targets, { reduction: 'none' })
  return sum(mul(ce, mask))
}

// Inference: returns sampled tokens (not logits). `temperature` is a [1]
// f32 input; broadcasts over [B, T, V] before categorical. Output dtype is
// i32 — `r.output` is `Int32Array`.
export function predictFn(m: NameLM, { tokens, temperature }: { tokens: Tensor; temperature: Tensor }): Tensor {
  const logits = logitsFwd(m, tokens)
  return categorical(div(logits, temperature))
}

export const inputs = {
  tokens:  { shape: [B, T], dtype: 'i32' },
  targets: { shape: [B, T], dtype: 'i32' },
  mask:    [B, T],
} as const

export const predictInputs = {
  tokens:      { shape: [B_INFER, T], dtype: 'i32' },
  temperature: { shape: [1],          dtype: 'f32' },
} as const

// Forward output dtype — `categorical` returns i32 indices.
export const predictOutput = 'i32' as const

export const optimizer = { kind: 'adamw', lr: LR, weightDecay: 0.01, clipGradNorm: 1.0 } as const

const trainingSpec = { model: new NameLM(), loss: lossFn, inputs, optimizer }

export function compileTraining(): Promise<CompiledTraining<NameLM>> {
  return compile(trainingSpec)
}

// Used by the NN Blueprint bulb to visualize this network as a computation graph.
export const irSpec = {
  label: 'Pokéformer (in-graph categorical sampling)',
  ...trainingSpec,
  predict: predictFn,
  predictInputs,
  predictOutput,
  dims: [
    { size: B,       name: 'B',  desc: 'batch' },
    { size: T,       name: 'T',  desc: 'seq len (16)' },
    { size: D,       name: 'D',  desc: 'model dim' },
    { size: N_HEADS, name: 'H',  desc: 'heads' },
    { size: D_HEAD,  name: 'D/H', desc: 'per-head dim' },
    { size: VOCAB,   name: 'V',  desc: 'vocab (27)' },
    { size: FFN,     name: '4D', desc: 'MLP hidden' },
  ],
}
