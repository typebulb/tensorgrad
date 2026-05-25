// Adam / AdamW optimizer, in-graph. Per parameter P with gradient g:
//   m_new = beta1 * m + (1 - beta1) * g
//   v_new = beta2 * v + (1 - beta2) * g²
//   p_new = decayShrink * p - lrt * m_new / (sqrt(v_new) + eps)
//
// `decayShrink = 1 - lr * weightDecay` for decayed params (Loshchilov & Hutter
// "AdamW"), else 1 (so the multiply folds out). `lrt` is supplied per step
// from CPU and includes Adam's bias correction `sqrt(1-beta2^t)/(1-beta1^t)`.
//
// Static vs scheduled lr: a number bakes decayShrink as a kernel literal; a
// schedule routes decayShrink through a per-step scalar input. lrt is always
// per-step (bias correction changes every step regardless).

import type { Tensor } from './ir.js'
import type { Graph } from './ir.js'
import type { WritebackDecl } from './buffers.js'
import { traceInto, stateInput, tensorInput } from './trace.js'
import { adamUpdateM, adamUpdateV, adamUpdateP } from './ops.js'
import { appendGradClip } from './grad.js'
import { type LR, resolveLR, isLRDynamic } from './lr.js'
import type { WireAdamConfig } from './worker-protocol.js'

/** Plain Adam hyperparameters. Pass via `compile({ ..., optimizer:
 *  { kind: 'adam', ... } })`. Only `lr` is required; the rest match PyTorch
 *  `torch.optim.Adam`'s defaults. For decoupled weight decay, use the
 *  separate `kind: 'adamw'` variant + `AdamWConfig`. */
export interface AdamConfig {
  /** Learning rate schedule. Pass a number for fixed lr, or a shape from
   *  the `lr` helpers (e.g., `lr.linear({ peak: 0.005, final: 0.0005, steps: 1500 })`). */
  lr: LR
  beta1?: number   // default 0.9
  beta2?: number   // default 0.999
  eps?: number  // default 1e-8
  /** Global L2-norm gradient clipping. When set, every gradient is scaled
   *  by `min(1, maxNorm / (totalNorm + 1e-6))` before the Adam update,
   *  where `totalNorm = sqrt(sum_p sum(grad_p ** 2))`. Standard
   *  training-stability hygiene; matches PyTorch's `clip_grad_norm_` and
   *  optax's `clip_by_global_norm`. Use `appendGradClip` directly if you
   *  need to compose clipping with a custom optimizer. */
  clipGradNorm?: number
}

/** AdamW hyperparameters (Loshchilov & Hutter — decoupled weight decay).
 *  Pass via `compile({ ..., optimizer: { kind: 'adamw', ... } })`. `weightDecay`
 *  is required (use plain `{ kind: 'adam' }` if you don't want decay). Every
 *  step shrinks each decayed param by `1 - lr * weightDecay` before the
 *  Adam gradient update. PyTorch parity: `torch.optim.AdamW`. */
export interface AdamWConfig extends AdamConfig {
  /** Decoupled weight decay coefficient. Must be > 0 — if you don't want
   *  decay, use `{ kind: 'adam' }`. */
  weightDecay: number
  /** Filter deciding which params get weight decay. Default: decay every
   *  param. Override for the standard transformer convention (decay
   *  weights/embeddings, skip biases + LN gains). Per-param `{ decay }` set
   *  via `this.param(shape, { decay })` overrides this filter when present.
   *  Example: `(name) => name.includes('.W') || name.endsWith('_emb')`. */
  decayFilter?: (paramName: string) => boolean
}

/** Union accepted by `appendAdam`. The compile pipeline narrows to one of
 *  these based on `OptimizerConfig.kind`. */
export type AdamOrAdamW = AdamConfig | AdamWConfig

function adamWeightDecay(c: AdamOrAdamW): number {
  return 'weightDecay' in c && c.weightDecay > 0 ? c.weightDecay : 0
}

function adamDecayFilter(c: AdamOrAdamW): (name: string) => boolean {
  return 'decayFilter' in c && c.decayFilter ? c.decayFilter : () => true
}

/** Resolved hyperparameters with all fields populated. `lr` stays as the
 *  shape (not pre-resolved) so the runtime can compute per-step values. */
export interface AdamResolvedConfig {
  lr: LR
  beta1: number
  beta2: number
  eps: number
  weightDecay: number
  decayFilter: (name: string) => boolean
  /** True iff the lr shape varies with step (linear, cosineAnnealing,
   *  warmup, step, multiStep). When false, decayShrink is baked at compile time. */
  lrIsScheduled: boolean
}

/** Output of `appendAdam`. Carries the writeback declarations the buffer
 *  planner needs to wire up persistent state + param updates, plus the
 *  scalar-input names the runtime fills per step (`lrt`, and `decayShrink`
 *  when both lr is scheduled and at least one param is being decayed). */
export interface AdamResult {
  /** Writebacks the buffer planner should wire into the runtime. */
  writebacks: WritebackDecl[]
  /** Name of the per-step scalar tensor_input. The runtime fills this each call
   * with `lr * sqrt(1-beta2^t)/(1-beta1^t)` (Adam's bias-corrected effective LR). */
  lrtInputName: string
  /** Name of the per-step decayShrink scalar tensor_input, or null when lr is
   *  static (decayShrink baked into the kernel) or no params are decayed. */
  decayShrinkInputName: string | null
  /** Hyperparameters as captured (so the runtime can compute lrt and decayShrink). */
  config: AdamResolvedConfig
}

/** Project an `appendAdam` result into the serializable config the worker
 *  reconstructs Adam state from. */
export function wireAdamConfig(r: AdamResult): WireAdamConfig {
  const c: AdamResolvedConfig = r.config
  return {
    lr: c.lr,
    beta1: c.beta1,
    beta2: c.beta2,
    eps: c.eps,
    weightDecay: c.weightDecay,
    lrIsScheduled: c.lrIsScheduled,
    lrtInputName: r.lrtInputName,
    decayShrinkInputName: r.decayShrinkInputName,
  }
}

/** The two per-step scalars the runtime feeds Adam's update kernel: the
 *  bias-corrected effective LR `lrt = lr·√(1−β2ᵗ)/(1−β1ᵗ)`, and AdamW's
 *  `decayShrink = 1 − lr·weightDecay` (folds out to 1 for non-decayed params,
 *  unused when weightDecay is 0). */
export function adamStepScalars(
  c: { lr: LR; beta1: number; beta2: number; weightDecay: number },
  t: number,
): { lrt: number; decayShrink: number } {
  const lrNow = resolveLR(c.lr, t)
  return {
    lrt: lrNow * Math.sqrt(1 - Math.pow(c.beta2, t)) / (1 - Math.pow(c.beta1, t)),
    decayShrink: 1 - lrNow * c.weightDecay,
  }
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
  config: AdamOrAdamW,
  /** Per-param decay flags from `materializeParams`. When supplied, overrides
   *  the config's `decayFilter` for any name in the map; falls back to
   *  `decayFilter` for names not present (low-level callers without a Module). */
  decayFlags?: Record<string, boolean>,
): AdamResult {
  // Clipping happens in-graph before Adam consumes the gradients.
  if (config.clipGradNorm !== undefined && config.clipGradNorm > 0) {
    paramGrads = appendGradClip(graph, paramGrads, config.clipGradNorm)
  }
  const lrIsScheduled = isLRDynamic(config.lr)
  const initialLr = resolveLR(config.lr, 1)
  const fullConfig: AdamResolvedConfig = {
    lr: config.lr,
    beta1: config.beta1 ?? 0.9,
    beta2: config.beta2 ?? 0.999,
    eps: config.eps ?? 1e-8,
    weightDecay: adamWeightDecay(config),
    decayFilter: adamDecayFilter(config),
    lrIsScheduled,
  }
  const writebacks: WritebackDecl[] = []
  const lrtInputName = '_adam_lrt'
  // Only allocated when lr is scheduled AND some param receives weight decay.
  let decayShrinkInputName: string | null = null

  return traceInto(graph, () => {
    const lrt = tensorInput(lrtInputName, [], 'f32')

    // Per-param decayFlags (from Module.param's options) win; decayFilter is
    // the fallback. Empty set when weightDecay = 0.
    const decayedNames = new Set<string>(
      fullConfig.weightDecay > 0
        ? Object.keys(paramGrads).filter(name =>
            (decayFlags && name in decayFlags) ? decayFlags[name]! : fullConfig.decayFilter(name))
        : [],
    )

    let decayShrinkScalar: Tensor | null = null
    if (lrIsScheduled && decayedNames.size > 0) {
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

      // decayShrink form per param:
      //   non-decayed:           literal 1 (multiply folds out)
      //   decayed + scheduled:   per-step tensor input
      //   decayed + static lr:   literal `1 - lr * wd` baked at compile
      const decayShrink: number | Tensor =
        !decayedNames.has(name) ? 1
        : decayShrinkScalar !== null ? decayShrinkScalar
        : 1 - initialLr * fullConfig.weightDecay

      const newM = adamUpdateM(mState, g, fullConfig.beta1)
      const newV = adamUpdateV(vState, g, fullConfig.beta2)
      const newP = adamUpdateP(p, newM, newV, lrt, fullConfig.eps, decayShrink)

      writebacks.push({ source: newM, destName: `adam_m_${name}`, destKind: 'state' })
      writebacks.push({ source: newV, destName: `adam_v_${name}`, destKind: 'state' })
      writebacks.push({ source: newP, destName: name,             destKind: 'param' })
    }
    return { writebacks, lrtInputName, decayShrinkInputName, config: fullConfig }
  })
}
