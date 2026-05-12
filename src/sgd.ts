// SGD / SGD-with-momentum / Nesterov, in-graph. No fused IR ops needed —
// the per-param update composes from existing primitives:
//   g_eff = g + wd * P                  (when wd > 0 and P is decayed)
//   v_new = momentum * v + g_eff        (when momentum > 0)
//   update = nesterov ? g_eff + momentum * v_new : v_new
//   P_new = P - lr * update
//
// Weight decay is PyTorch-style (injected into the gradient), not decoupled
// like AdamW.

import type { Tensor, Graph } from './ir.js'
import type { WritebackDecl } from './buffers.js'
import { traceInto, stateInput, tensorInput } from './trace.js'
import { add, sub, mul, mulScalar, broadcastTo } from './ops.js'
import { appendGradClip } from './adam.js'
import type { LR } from './adam.js'
import { isLRDynamic, resolveLR } from './adam.js'

/** SGD hyperparameters. Pass via `spec({ ..., optimizer:
 *  { kind: 'sgd', ... } })`. Only `lr` is required. With `momentum: 0`
 *  (default) you get plain SGD; non-zero adds per-param velocity state;
 *  `nesterov: true` (requires `momentum > 0`) switches to Nesterov momentum. */
export interface SGDConfig {
  /** Learning rate schedule. Pass a number for fixed lr, or a shape from
   *  the `lr` helpers (e.g. `lr.cosineDecay({ peak: 0.05, final: 0.001, steps: 10000 })`). */
  lr: LR
  /** Momentum coefficient. Default 0 (plain SGD). When non-zero, a per-param
   *  velocity buffer accumulates `momentum * v + g_eff` each step. */
  momentum?: number
  /** Nesterov momentum. Requires `momentum > 0`. Default false. */
  nesterov?: boolean
  /** L2-style weight decay coefficient (injected into the gradient, PyTorch
   *  convention). Default 0. When non-zero, each decayed param's effective
   *  gradient becomes `g + weightDecay * p` before the momentum + update steps. */
  weightDecay?: number
  /** Filter deciding which params get weight decay. Only consulted when
   *  `weightDecay > 0`. Default: decay every param. Override for the standard
   *  convention (decay weights, skip biases). */
  decayFilter?: (paramName: string) => boolean
  /** Global L2-norm gradient clipping. When set, every gradient is scaled
   *  by `min(1, maxNorm / (totalNorm + 1e-6))` before the SGD update.
   *  Matches PyTorch's `clip_grad_norm_`. */
  clipGradNorm?: number
}

/** Resolved hyperparameters with all fields populated. `lr` stays as the
 *  shape (not pre-resolved) so the runtime can compute per-step values. */
export interface SGDResolvedConfig {
  lr: LR
  momentum: number
  nesterov: boolean
  weightDecay: number
  decayFilter: (name: string) => boolean
  /** True iff the lr shape varies with step. (For SGD this only affects
   *  what the runtime needs to update each step; there's no constant-bake
   *  optimization like AdamW's decayShrink.) */
  lrIsScheduled: boolean
}

/** Output of `appendSGD`. Carries the writeback declarations the buffer
 *  planner needs (param + optional velocity state) and the name of the
 *  per-step `lr` scalar input the runtime fills each step. */
export interface SGDResult {
  /** Writebacks the buffer planner should wire into the runtime. */
  writebacks: WritebackDecl[]
  /** Name of the per-step scalar tensor_input. The runtime fills this each
   *  step with the current lr. */
  lrInputName: string
  /** Resolved hyperparameters (defaults applied; `lr` left as the schedule
   *  shape so the runtime can compute per-step values). */
  config: SGDResolvedConfig
}

/**
 * Append SGD update ops to `graph`. Must be called inside an active trace
 * context (or after a trace, since traceInto re-enters the graph).
 *
 * @param graph the graph (already containing forward + backward)
 * @param paramGrads param name -> gradient tensor (output of `appendGrad`)
 * @param paramTensors param name -> the param's leaf Tensor (the param_input)
 * @param config SGD hyperparameters
 * @param decayFlags optional per-param decay flags from `materializeParams`
 */
export function appendSGD(
  graph: Graph,
  paramGrads: Record<string, Tensor>,
  paramTensors: Record<string, Tensor>,
  config: SGDConfig,
  decayFlags?: Record<string, boolean>,
): SGDResult {
  // Clipping happens in-graph before the rest of the update.
  if (config.clipGradNorm !== undefined && config.clipGradNorm > 0) {
    paramGrads = appendGradClip(graph, paramGrads, config.clipGradNorm)
  }

  const momentum = config.momentum ?? 0
  const nesterov = config.nesterov ?? false
  if (nesterov && momentum <= 0) {
    throw new Error(`appendSGD: nesterov requires momentum > 0 (got momentum=${momentum})`)
  }

  const fullConfig: SGDResolvedConfig = {
    lr: config.lr,
    momentum,
    nesterov,
    weightDecay: config.weightDecay ?? 0,
    decayFilter: config.decayFilter ?? (() => true),
    lrIsScheduled: isLRDynamic(config.lr),
  }

  const writebacks: WritebackDecl[] = []
  const lrInputName = '_sgd_lr'

  return traceInto(graph, () => {
    const lr = tensorInput(lrInputName, [], 'f32')

    const decayedNames = new Set<string>(
      fullConfig.weightDecay > 0
        ? Object.keys(paramGrads).filter(name =>
            (decayFlags && name in decayFlags) ? decayFlags[name]! : fullConfig.decayFilter(name))
        : [],
    )

    for (const name of Object.keys(paramGrads)) {
      const p = paramTensors[name]
      const g = paramGrads[name]
      if (!p) throw new Error(`appendSGD: missing param tensor for '${name}'`)
      if (!g) throw new Error(`appendSGD: missing gradient for '${name}'`)

      const gEff = decayedNames.has(name)
        ? add(g, mulScalar(p, fullConfig.weightDecay))
        : g

      let update: Tensor
      if (momentum > 0) {
        const vState = stateInput(`sgd_v_${name}`, p.shape, 'f32', 0)
        const vNew = add(mulScalar(vState, momentum), gEff)
        update = nesterov ? add(gEff, mulScalar(vNew, momentum)) : vNew
        writebacks.push({ source: vNew, destName: `sgd_v_${name}`, destKind: 'state' })
      } else {
        update = gEff
      }

      const pNew = sub(p, mul(broadcastTo(lr, p.shape), update))
      writebacks.push({ source: pNew, destName: name, destKind: 'param' })
    }

    return { writebacks, lrInputName, config: fullConfig }
  })
}

/** Resolve the per-step lr (scalar) for SGD. Trivial wrapper around
 *  `resolveLR` for symmetry with how the worker invokes Adam's update path. */
export function resolveSGDLr(config: SGDResolvedConfig, step: number): number {
  return resolveLR(config.lr, step)
}
