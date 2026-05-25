// Learning-rate schedules. Optimizer-agnostic — Adam, AdamW, and SGD all
// route their per-step lr through here. Pure functions + serializable shapes
// (no closures): the schedule crosses the worker boundary, so it's data, not
// behavior. `lr` and `LR` are the public surface (re-exported from index).

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
