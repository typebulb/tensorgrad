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

/** Per-step learning-rate schedule. Either a fixed number or one of the
 *  serializable shape forms below. Functions/closures are not supported —
 *  the schedule needs to cross thread boundaries and survive serialization
 *  for the worker-internal runtime, and every realistic LR pattern (constant,
 *  linear decay, cosine, warmup-then-decay) maps to a finite set of shapes.
 *  Use the `lr` helper namespace to construct shapes ergonomically. */
/**
 * Each non-constant schedule variant carries an optional `startStep`
 * (default 0). The schedule's intrinsic step is `current_step - startStep`,
 * clamped to be ≥ 1. This lets a user say "decay starting from where I am
 * now" instead of "decay measured from step 1 of training," without us
 * having to bake closures into the schedule shape.
 *
 * `setOptimizerConfig({ lr: <schedule> })` auto-fills `startStep = current_t`
 * for any non-constant schedule that doesn't already specify one, so the
 * schedule's "step 1" lines up with the Adam step it took effect. If the
 * caller sets `startStep` explicitly, it's respected as-is.
 */
export type LRSchedule =
  | number
  | { readonly kind: 'constant'; readonly value: number }
  | { readonly kind: 'linearDecay'; readonly peak: number; readonly final: number; readonly steps: number; readonly startStep?: number }
  | { readonly kind: 'cosineDecay'; readonly peak: number; readonly final: number; readonly steps: number; readonly startStep?: number }
  | { readonly kind: 'warmup'; readonly peakLr: number; readonly warmupSteps: number; readonly after: LRSchedule; readonly startStep?: number }

/** Ergonomic constructors for LRSchedule shapes. */
export const lr = {
  constant: (value: number): LRSchedule => ({ kind: 'constant', value }),
  /** Linearly interpolate from `peak` at intrinsic step 1 to `final` at
   *  intrinsic step `steps`, then hold at `final`. Optional `startStep` shifts
   *  the timeline (intrinsic = current - startStep + 1). */
  linearDecay: (opts: { peak: number; final: number; steps: number; startStep?: number }): LRSchedule =>
    ({ kind: 'linearDecay', ...opts }),
  /** Half-cosine from `peak` at intrinsic step 1 down to `final` at intrinsic
   *  step `steps`, then hold at `final`. Optional `startStep` shifts the
   *  timeline. */
  cosineDecay: (opts: { peak: number; final: number; steps: number; startStep?: number }): LRSchedule =>
    ({ kind: 'cosineDecay', ...opts }),
  /** Linear ramp from 0 to `peakLr` over `warmupSteps`, then hand off to
   *  `after` (offset so step 1 of `after` = first post-warmup step). Optional
   *  `startStep` shifts the timeline. */
  warmup: (opts: { peakLr: number; warmupSteps: number; after: LRSchedule; startStep?: number }): LRSchedule =>
    ({ kind: 'warmup', ...opts }),
}

/** Apply a schedule's `startStep` offset to a current step and clamp to ≥ 1. */
function intrinsicStep(startStep: number | undefined, currentStep: number): number {
  return Math.max(1, currentStep - (startStep ?? 0))
}

/** Resolve a schedule to its scalar value at a given 1-based step. */
export function resolveLR(schedule: LRSchedule, step: number): number {
  if (typeof schedule === 'number') return schedule
  switch (schedule.kind) {
    case 'constant': return schedule.value
    case 'linearDecay': {
      const s = intrinsicStep(schedule.startStep, step)
      const f = Math.min(s / schedule.steps, 1)
      return schedule.peak + (schedule.final - schedule.peak) * f
    }
    case 'cosineDecay': {
      const s = intrinsicStep(schedule.startStep, step)
      const f = Math.min(s / schedule.steps, 1)
      return schedule.final + 0.5 * (schedule.peak - schedule.final) * (1 + Math.cos(Math.PI * f))
    }
    case 'warmup': {
      const s = intrinsicStep(schedule.startStep, step)
      if (s <= schedule.warmupSteps) return schedule.peakLr * (s / schedule.warmupSteps)
      return resolveLR(schedule.after, s - schedule.warmupSteps)
    }
  }
}

/** Rewrite a schedule to start its timeline at `baseStep` (sets startStep on
 *  the outer shape to `baseStep - 1`, so the schedule's intrinsic step 1
 *  aligns with the user's `baseStep`). Numbers and `{kind:'constant'}` have
 *  no notion of time and pass through unchanged. Used by the worker to
 *  auto-rebase non-constant schedules handed to `setOptimizerConfig`. */
export function rebaseLR(schedule: LRSchedule, baseStep: number): LRSchedule {
  if (typeof schedule === 'number' || schedule.kind === 'constant') return schedule
  return { ...schedule, startStep: baseStep - 1 }
}

/** True for shapes that produce different values at different steps (so the
 *  AdamW decayShrink scalar must be a per-step input rather than baked).
 *  Numbers and `{kind:'constant'}` are static; everything else varies. */
export function isLRDynamic(schedule: LRSchedule): boolean {
  if (typeof schedule === 'number') return false
  return schedule.kind !== 'constant'
}

export interface AdamConfig {
  /** Learning rate schedule. Pass a number for fixed lr, or a shape from
   *  the `lr` helpers (e.g., `lr.linearDecay({ peak: 0.005, final: 0.0005, steps: 1500 })`). */
  lr: LRSchedule
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

/** Resolved hyperparameters with all fields populated. `lr` stays as the
 *  shape (not pre-resolved) so the runtime can compute per-step values. */
export interface AdamResolvedConfig {
  lr: LRSchedule
  b1: number
  b2: number
  eps: number
  weightDecay: number
  decayFilter: (name: string) => boolean
  /** True iff the lr shape varies with step (linearDecay, cosineDecay,
   *  warmup). When false, decayShrink is baked at compile time. */
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
  const lrIsScheduled = isLRDynamic(config.lr)
  const initialLr = resolveLR(config.lr, 1)
  const fullConfig: AdamResolvedConfig = {
    lr: config.lr,
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

    // Up-front: which params receive weight decay? Per-param decayFlags (set
    // by Module.param's options) wins; falls back to decayFilter for names
    // not in the map. Empty when weightDecay = 0 so the rest of the function
    // can just ask "is this name in the set?".
    const decayedNames = new Set<string>(
      fullConfig.weightDecay > 0
        ? Object.keys(paramGrads).filter(name =>
            (decayFlags && name in decayFlags) ? decayFlags[name]! : fullConfig.decayFilter(name))
        : [],
    )

    // We only need a runtime decayShrink scalar when lr varies per step AND
    // at least one param is being decayed. Otherwise the value is constant
    // and bakes into the kernel as a literal.
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

      // Choose the decayShrink form per param:
      //   - non-decayed params: literal 1 (kernel multiply folds out).
      //   - decayed + scheduled lr: tensor input updated per step.
      //   - decayed + static lr: literal `1 - lr * wd` baked at compile.
      const decayShrink: number | Tensor =
        !decayedNames.has(name) ? 1
        : decayShrinkScalar !== null ? decayShrinkScalar
        : 1 - initialLr * fullConfig.weightDecay

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
