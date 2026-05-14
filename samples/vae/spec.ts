// Import-safe spec for the VAE. Consumed by main.ts; also exports
// `irSpec` for paste into the NN Blueprint bulb.

import {
  Module, compile, Linear,
  add, sub, mul, sum, exp, sigmoid, relu,
  randn, square,
  type Tensor, type CompiledTraining,
} from 'tensorgrad'

export const INPUT_DIM = 28 * 28
export const LATENT_DIM = 8
export const HIDDEN = 256
export const BATCH_SIZE = 128
export const BETA = 1.0

export class VAE extends Module {
  enc1     = new Linear(INPUT_DIM, HIDDEN)
  enc2     = new Linear(HIDDEN,    HIDDEN)
  encMu    = new Linear(HIDDEN,    LATENT_DIM)
  encLogV  = new Linear(HIDDEN,    LATENT_DIM)
  dec1     = new Linear(LATENT_DIM, HIDDEN)
  dec2     = new Linear(HIDDEN,    HIDDEN)
  decOut   = new Linear(HIDDEN,    INPUT_DIM)
}

export function encoder(m: VAE, x: Tensor): { mu: Tensor; logVar: Tensor } {
  let h = relu(m.enc1.fwd(x))
  h = relu(m.enc2.fwd(h))
  return { mu: m.encMu.fwd(h), logVar: m.encLogV.fwd(h) }
}

export function decoder(m: VAE, z: Tensor): Tensor {
  let h = relu(m.dec1.fwd(z))
  h = relu(m.dec2.fwd(h))
  return sigmoid(m.decOut.fwd(h))
}

export function lossFn(m: VAE, { x }: { x: Tensor }): Tensor {
  const B = x.shape[0]!
  const { mu, logVar } = encoder(m, x)
  const eps = randn([B, LATENT_DIM])
  const sigma = exp(mul(logVar, 0.5))
  const z = add(mu, mul(sigma, eps))
  const xHat = decoder(m, z)
  const recon = mul(sum(square(sub(xHat, x))), 1 / B)
  const klElem = mul(sub(sub(add(logVar, 1), square(mu)), exp(logVar)), -0.5)
  const kl = mul(sum(klElem), 1 / B)
  return add(recon, mul(kl, BETA))
}

export function encodeFn(m: VAE, { x }: { x: Tensor }): Tensor {
  return encoder(m, x).mu
}

export function decodeFn(m: VAE, { z }: { z: Tensor }): Tensor {
  return decoder(m, z)
}

// Full deterministic reconstruction: x → encoder.mu → decoder → xHat. Used
// by the NN Blueprint bulb to render the inference graph; main.ts attaches
// encodeFn/decodeFn separately for its UI's encode-only and decode-only
// paths, so this function is not exported.
function predictFn(m: VAE, { x }: { x: Tensor }): Tensor {
  return decoder(m, encoder(m, x).mu)
}

export const inputs = { x: [BATCH_SIZE, INPUT_DIM] } as const
export const predictInputs = { x: [BATCH_SIZE, INPUT_DIM] } as const
export const optimizer = { kind: 'adam', lr: 1e-3, clipGradNorm: 1.0 } as const

export function compileTraining(): Promise<CompiledTraining<VAE>> {
  return compile({ model: new VAE(), loss: lossFn, inputs, optimizer })
}

// Used by the NN Blueprint bulb to visualize this network as a computation graph.
// Paste the whole file at typebulb.com/u/samples/nn-blueprint/full to render it.
export const irSpec = {
  label: 'VAE (MNIST)',
  compile: compileTraining,
  predict: predictFn,
  predictInputs,
  dims: [
    { size: BATCH_SIZE, name: 'B',   desc: 'batch' },
    { size: INPUT_DIM,  name: '784', desc: 'pixels (28²)' },
    { size: HIDDEN,     name: 'H',   desc: 'hidden' },
    { size: LATENT_DIM, name: 'Z',   desc: 'latent dim' },
  ],
}

