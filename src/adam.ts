// Adam / AdamW optimizer, in-graph.
//
// `appendAdam` extends a graph that already has a forward pass + autograd-emitted
// backward (i.e., has paramGrads from `appendGrad`) with the Adam update math.
//
// Per parameter P with gradient g:
//   m_new = b1 * m + (1 - b1) * g
//   v_new = b2 * v + (1 - b2) * g²
//   p_new = decayShrink * p - lrt * m_new / (sqrt(v_new) + eps)
//
// `decayShrink = 1 - lr * weightDecay` when the param is being decayed
// (Loshchilov & Hutter, "AdamW") and 1 otherwise — at which point the
// multiply folds out and you're left with plain Adam. `lrt` is supplied
// per-step from CPU and includes the bias-correction factor
// `sqrt(1-b2^t)/(1-b1^t)`; that's why convergence isn't affected by the
// first-step warmup that bias-correction-free Adam suffers.
//
// Returns writeback declarations the buffer planner uses to wire up the
// "after step, copy the new value into the persistent home" path. m and v
// are state_inputs (zero-initialized, persistent across steps); the param
// updates are aliased back to the param buffers.

import type { Tensor } from './ir.js'
import type { Graph } from './ir.js'
import type { WritebackDecl } from './buffers.js'
import { traceInto, stateInput, tensorInput } from './trace.js'
import { adamUpdateM, adamUpdateV, adamUpdateP } from './ops.js'

export interface AdamConfig {
  lr: number
  b1?: number   // default 0.9
  b2?: number   // default 0.999
  eps?: number  // default 1e-8
  /** AdamW: decoupled weight decay coefficient. Default 0 (plain Adam).
   *  When non-zero, every step shrinks each decayed param by a factor of
   *  `1 - lr * weightDecay` before the gradient update. */
  weightDecay?: number
  /** Filter deciding which params get weight decay. Only consulted when
   *  weightDecay > 0. Default: decay every param. Override for the standard
   *  transformer convention (decay weights/embeddings, skip biases + LN gains).
   *  Example: `(name) => name.includes('.W') || name.endsWith('_emb')`. */
  decayFilter?: (paramName: string) => boolean
}

export interface AdamResult {
  /** Writebacks the buffer planner should wire into the runtime. */
  writebacks: WritebackDecl[]
  /** Name of the per-step scalar tensor_input. The runtime fills this each call
   * with `lr * sqrt(1-b2^t)/(1-b1^t)` (Adam's bias-corrected effective LR). */
  lrtInputName: string
  /** Hyperparameters as captured (so the runtime can compute lrt). */
  config: Required<Omit<AdamConfig, 'decayFilter'>> & { decayFilter: (name: string) => boolean }
}

/**
 * Append Adam update ops to `graph`. Must be called inside an active trace
 * context (or after a trace, since traceInto re-enters the graph).
 *
 * @param graph the graph (already containing forward + backward)
 * @param paramGrads param name -> gradient tensor (output of `appendGrad`)
 * @param paramTensors param name -> the param's leaf Tensor (the param_input).
 *                     Needed because the param_input lives in the graph but we
 *                     don't have a direct map by name in `Graph` — caller passes it.
 * @param config Adam hyperparameters. Set `weightDecay > 0` for AdamW; an
 *               optional `decayFilter` selects which params receive decay.
 */
export function appendAdam(
  graph: Graph,
  paramGrads: Record<string, Tensor>,
  paramTensors: Record<string, Tensor>,
  config: AdamConfig,
): AdamResult {
  const fullConfig = {
    lr: config.lr,
    b1: config.b1 ?? 0.9,
    b2: config.b2 ?? 0.999,
    eps: config.eps ?? 1e-8,
    weightDecay: config.weightDecay ?? 0,
    decayFilter: config.decayFilter ?? (() => true),
  }
  const writebacks: WritebackDecl[] = []
  const lrtInputName = '_adam_lrt'

  return traceInto(graph, () => {
    // One scalar lrt input shared by every adam_update_p call. Runtime supplies
    // it per step as `lr * sqrt(1-b2^t) / (1-b1^t)`.
    const lrt = tensorInput(lrtInputName, [], 'f32')

    for (const name of Object.keys(paramGrads)) {
      const p = paramTensors[name]
      const g = paramGrads[name]
      if (!p) throw new Error(`appendAdam: missing param tensor for '${name}'`)
      if (!g) throw new Error(`appendAdam: missing gradient for '${name}'`)

      const mState = stateInput(`adam_m_${name}`, p.shape, 'f32', 0)
      const vState = stateInput(`adam_v_${name}`, p.shape, 'f32', 0)

      // decayShrink baked at compile time. 1.0 for plain Adam (no extra cost
      // — the WGSL compiler folds the constant multiply); 1 - lr * weightDecay
      // for the params the filter selects.
      const decayShrink = (fullConfig.weightDecay > 0 && fullConfig.decayFilter(name))
        ? 1 - fullConfig.lr * fullConfig.weightDecay
        : 1

      // Three fused kernels per parameter — one for each of m_new / v_new / p_new.
      const newM = adamUpdateM(mState, g, fullConfig.b1)
      const newV = adamUpdateV(vState, g, fullConfig.b2)
      const newP = adamUpdateP(p, newM, newV, lrt, fullConfig.eps, decayShrink)

      writebacks.push({ source: newM, destName: `adam_m_${name}`, destKind: 'state' })
      writebacks.push({ source: newV, destName: `adam_v_${name}`, destKind: 'state' })
      writebacks.push({ source: newP, destName: name,             destKind: 'param' })
    }
    return { writebacks, lrtInputName, config: fullConfig }
  })
}
