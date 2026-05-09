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
// **Static vs scheduled lr.** When `config.lr` is a number, decayShrink is
// baked into the kernel as a literal. When it's a function `(step) => lr`,
// decayShrink for decayed params becomes a per-step scalar input that the
// runtime updates each call (computed from the current step's lr). lrt is
// always per-step; the bias-correction factor changes every step regardless.
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
  /** Constant scalar (e.g., `0.005`) or a per-step schedule function
   *  `(step) => lr`. Schedule fn lets the user implement linear/cosine decay
   *  or warmup; first call passes `step=1`. Decay-shrink (AdamW) updates
   *  per-step automatically when this is a function. */
  lr: number | ((step: number) => number)
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

/** Resolved hyperparameters: lr is the schedule fn (constants are wrapped). */
export interface AdamResolvedConfig {
  lr: (step: number) => number
  b1: number
  b2: number
  eps: number
  weightDecay: number
  decayFilter: (name: string) => boolean
  /** True iff the user supplied an lr function (vs a constant). When false,
   *  decayShrink is baked at compile time and never updated. */
  lrIsScheduled: boolean
}

export interface AdamResult {
  /** Writebacks the buffer planner should wire into the runtime. */
  writebacks: WritebackDecl[]
  /** Name of the per-step scalar tensor_input. The runtime fills this each call
   * with `lr * sqrt(1-b2^t)/(1-b1^t)` (Adam's bias-corrected effective LR). */
  lrtInputName: string
  /** Name of the per-step decayShrink scalar tensor_input, or null when lr is
   *  static (decayShrink baked into the kernel) or no params are decayed. */
  decayShrinkInputName: string | null
  /** Hyperparameters as captured (so the runtime can compute lrt and decayShrink). */
  config: AdamResolvedConfig
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
  /** Per-param decay flags from `materializeParams`. When supplied, overrides
   *  `config.decayFilter` for any name in the map; falls back to `decayFilter`
   *  for names not present (e.g., for low-level callers using `compile()`
   *  directly without a Module). */
  decayFlags?: Record<string, boolean>,
): AdamResult {
  const lrIsScheduled = typeof config.lr === 'function'
  const lrFn = lrIsScheduled
    ? config.lr as (step: number) => number
    : (() => config.lr as number)
  const initialLr = lrFn(1)
  const fullConfig: AdamResolvedConfig = {
    lr: lrFn,
    b1: config.b1 ?? 0.9,
    b2: config.b2 ?? 0.999,
    eps: config.eps ?? 1e-8,
    weightDecay: config.weightDecay ?? 0,
    decayFilter: config.decayFilter ?? (() => true),
    lrIsScheduled,
  }
  const writebacks: WritebackDecl[] = []
  const lrtInputName = '_adam_lrt'
  // Tensor input for runtime-updated decayShrink (only created when lr is a
  // schedule fn AND at least one param will receive weight decay).
  let decayShrinkInputName: string | null = null

  return traceInto(graph, () => {
    const lrt = tensorInput(lrtInputName, [], 'f32')

    // Resolve per-param decay decision: decayFlags (per-param metadata from
    // Module.param's options) wins; fall back to decayFilter for names not in
    // the map. Captured here so the dynamic-shrink check below and the loop
    // below agree on what's decayed.
    const isParamDecayed = (name: string): boolean => {
      if (decayFlags && name in decayFlags) return decayFlags[name]!
      return fullConfig.decayFilter(name)
    }

    // Decide up-front whether we need a runtime decayShrink scalar. Only does
    // something when both (a) lr varies per step and (b) some param is decayed.
    const needsDynamicShrink = lrIsScheduled
      && fullConfig.weightDecay > 0
      && Object.keys(paramGrads).some(isParamDecayed)
    let decayShrinkScalar: Tensor | null = null
    if (needsDynamicShrink) {
      decayShrinkInputName = '_adam_decay_shrink'
      decayShrinkScalar = tensorInput(decayShrinkInputName, [], 'f32')
    }

    for (const name of Object.keys(paramGrads)) {
      const p = paramTensors[name]
      const g = paramGrads[name]
      if (!p) throw new Error(`appendAdam: missing param tensor for '${name}'`)
      if (!g) throw new Error(`appendAdam: missing gradient for '${name}'`)

      const mState = stateInput(`adam_m_${name}`, p.shape, 'f32', 0)
      const vState = stateInput(`adam_v_${name}`, p.shape, 'f32', 0)

      // Choose the decayShrink form per param:
      //   - non-decayed params: literal 1 (kernel multiply folds out).
      //   - decayed + static lr: literal `1 - lr * wd` baked at compile.
      //   - decayed + scheduled lr: tensor input updated per step.
      const isDecayed = fullConfig.weightDecay > 0 && isParamDecayed(name)
      let decayShrink: number | Tensor
      if (!isDecayed) {
        decayShrink = 1
      } else if (decayShrinkScalar !== null) {
        decayShrink = decayShrinkScalar
      } else {
        decayShrink = 1 - initialLr * fullConfig.weightDecay
      }

      // Three fused kernels per parameter — one for each of m_new / v_new / p_new.
      const newM = adamUpdateM(mState, g, fullConfig.b1)
      const newV = adamUpdateV(vState, g, fullConfig.b2)
      const newP = adamUpdateP(p, newM, newV, lrt, fullConfig.eps, decayShrink)

      writebacks.push({ source: newM, destName: `adam_m_${name}`, destKind: 'state' })
      writebacks.push({ source: newV, destName: `adam_v_${name}`, destKind: 'state' })
      writebacks.push({ source: newP, destName: name,             destKind: 'param' })
    }
    return { writebacks, lrtInputName, decayShrinkInputName, config: fullConfig }
  })
}
