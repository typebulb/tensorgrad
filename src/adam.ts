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
import { adamUpdateM, adamUpdateV, adamUpdateP, add, mul, sqrt, sum, div, min, broadcastTo, constScalar } from './ops.js'

/**
 * Per-step learning-rate schedule. Either a fixed number or one of the
 * serializable shape forms below. Closures aren't supported — the schedule
 * crosses the worker boundary and every realistic LR pattern (constant,
 * linear/cosine decay, warmup-then-decay, step, multi-step) maps to one of
 * these shapes. Use the `lr` helper namespace for ergonomic construction.
 *
 * Each non-constant variant carries an optional `startStep` (default 0).
 * The intrinsic step is `max(1, currentStep - startStep)`, so users can
 * say "decay starting from where I am now" without us baking closures into
 * the shape. `setLR(...)` auto-fills `startStep = current_t` when not set,
 * so the new schedule's step 1 = the next training step.
 */
export type LR =
  | number
  | { readonly kind: 'linear'; readonly peak: number; readonly final: number; readonly steps: number; readonly startStep?: number }
  | { readonly kind: 'cosineAnnealing'; readonly peak: number; readonly final: number; readonly steps: number; readonly startStep?: number }
  | { readonly kind: 'warmup'; readonly peak: number; readonly steps: number; readonly after: LR; readonly startStep?: number }
  | { readonly kind: 'staircase'; readonly peak: number; readonly every: number; readonly gamma: number; readonly startStep?: number }
  | { readonly kind: 'multiStep'; readonly peak: number; readonly milestones: readonly number[]; readonly gamma: number; readonly startStep?: number }

/** Ergonomic constructors for LR schedule shapes. For constant lr, pass a
 *  raw number — every LR field on `compile({ optimizer })` / `setLR`
 *  accepts `number | LR`. Names mirror PyTorch's `torch.optim.lr_scheduler`. */
export const lr = {
  /** Linearly interpolate from `peak` at intrinsic step 1 to `final` at
   *  intrinsic step `steps`, then hold at `final`. Optional `startStep`
   *  shifts the timeline (intrinsic = current - startStep + 1).
   *  PyTorch: `LinearLR`. */
  linear: (opts: { peak: number; final: number; steps: number; startStep?: number }): LR =>
    ({ kind: 'linear', ...opts }),
  /** Half-cosine from `peak` at intrinsic step 1 down to `final` at intrinsic
   *  step `steps`, then hold at `final`. Optional `startStep` shifts the
   *  timeline. PyTorch: `CosineAnnealingLR`. */
  cosineAnnealing: (opts: { peak: number; final: number; steps: number; startStep?: number }): LR =>
    ({ kind: 'cosineAnnealing', ...opts }),
  /** Linear ramp from 0 to `peak` over `steps`, then hand off to
   *  `after` (offset so step 1 of `after` = first post-warmup step).
   *  Optional `startStep` shifts the timeline. */
  warmup: (opts: { peak: number; steps: number; after: LR; startStep?: number }): LR =>
    ({ kind: 'warmup', ...opts }),
  /** Geometric decay: `peak * gamma^floor(step / every)`. PyTorch's
   *  `torch.optim.lr_scheduler.StepLR` (their `step_size`). With `every: 1`,
   *  every step multiplies lr by `gamma` (exponential decay). Named for the
   *  staircase shape of the resulting LR-vs-step curve (and to keep the
   *  schedule discriminator out of the `step` action-verb namespace — see
   *  `specs/architecture.md`). */
  staircase: (opts: { peak: number; every: number; gamma: number; startStep?: number }): LR =>
    ({ kind: 'staircase', ...opts }),
  /** Piecewise-constant decay at specific step milestones: `peak * gamma^(count
   *  of milestones <= step)`. PyTorch's
   *  `torch.optim.lr_scheduler.MultiStepLR`. Useful for "drop lr by 10x at
   *  steps 30k and 60k" patterns. Milestones must be sorted ascending. */
  multiStep: (opts: { peak: number; milestones: readonly number[]; gamma: number; startStep?: number }): LR =>
    ({ kind: 'multiStep', ...opts }),
}

/** Apply a schedule's `startStep` offset to a current step and clamp to ≥ 1. */
function intrinsicStep(startStep: number | undefined, currentStep: number): number {
  return Math.max(1, currentStep - (startStep ?? 0))
}

/** Resolve a schedule to its scalar value at a given 1-based step. */
export function resolveLR(schedule: LR, step: number): number {
  if (typeof schedule === 'number') return schedule
  switch (schedule.kind) {
    case 'linear': {
      const s = intrinsicStep(schedule.startStep, step)
      const f = Math.min(s / schedule.steps, 1)
      return schedule.peak + (schedule.final - schedule.peak) * f
    }
    case 'cosineAnnealing': {
      const s = intrinsicStep(schedule.startStep, step)
      const f = Math.min(s / schedule.steps, 1)
      return schedule.final + 0.5 * (schedule.peak - schedule.final) * (1 + Math.cos(Math.PI * f))
    }
    case 'warmup': {
      const s = intrinsicStep(schedule.startStep, step)
      if (s <= schedule.steps) return schedule.peak * (s / schedule.steps)
      return resolveLR(schedule.after, s - schedule.steps)
    }
    case 'staircase': {
      // 1-based step indexing; `(s - 1) / every` matches PyTorch StepLR's
      // boundary semantics (first `every` values unscaled).
      const s = intrinsicStep(schedule.startStep, step)
      const k = Math.floor((s - 1) / schedule.every)
      return schedule.peak * Math.pow(schedule.gamma, k)
    }
    case 'multiStep': {
      const s = intrinsicStep(schedule.startStep, step)
      let k = 0
      for (const m of schedule.milestones) if (s >= m) k++
      return schedule.peak * Math.pow(schedule.gamma, k)
    }
  }
}

/** Rewrite a schedule to start its timeline at `baseStep` (sets startStep on
 *  the outer shape to `baseStep - 1`, so the schedule's intrinsic step 1
 *  aligns with the user's `baseStep`). Raw numbers have no notion of time
 *  and pass through unchanged. Schedules that already have an explicit
 *  `startStep` also pass through — caller intent wins. */
export function rebaseLR(schedule: LR, baseStep: number): LR {
  if (typeof schedule === 'number') return schedule
  if (schedule.startStep !== undefined) return schedule
  return { ...schedule, startStep: baseStep - 1 }
}

/** True for shapes that produce different values at different steps (so the
 *  AdamW decayShrink scalar must be a per-step input rather than baked).
 *  Raw numbers are static; every schedule shape varies. */
export function isLRDynamic(schedule: LR): boolean {
  return typeof schedule !== 'number'
}

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

/**
 * Append global L2-norm gradient clipping to `graph`. Reads every gradient
 * in `paramGrads`, computes the total norm across all of them, and emits a
 * fresh `paramGrads` record where each tensor has been scaled by
 * `min(1, maxNorm / (totalNorm + 1e-6))`.
 *
 * Matches PyTorch's `clip_grad_norm_` semantics and optax's
 * `clip_by_global_norm`. The scale is global — every gradient shares one
 * scaling factor — not per-parameter.
 *
 * Standalone extension hook: use this directly when composing clipping with
 * a custom optimizer. For Adam, set `AdamConfig.clipGradNorm` and
 * `appendAdam` calls this internally.
 *
 * Cross-parameter reduction is a chained add — fine at browser-scale param
 * counts (tens to low hundreds). Each add is a rank-0 dispatch.
 */
export function appendGradClip(
  graph: Graph,
  paramGrads: Record<string, Tensor>,
  maxNorm: number,
): Record<string, Tensor> {
  return traceInto(graph, () => {
    const entries = Object.entries(paramGrads)
    if (entries.length === 0) return paramGrads
    // sum_p sum(grad_p²) — chained adds across params; each summand is rank-0.
    let sumSq: Tensor = sum(mul(entries[0]![1], entries[0]![1]))
    for (let i = 1; i < entries.length; i++) {
      sumSq = add(sumSq, sum(mul(entries[i]![1], entries[i]![1])))
    }
    const scale = min(div(constScalar(maxNorm, 'f32'), add(sqrt(sumSq), 1e-6)), 1)
    const clipped: Record<string, Tensor> = {}
    for (const [name, g] of entries) {
      clipped[name] = mul(g, broadcastTo(scale, g.shape))
    }
    return clipped
  })
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
