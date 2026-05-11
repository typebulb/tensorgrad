// SGD / SGD-with-momentum / Nesterov, in-graph.
//
// `appendSGD` extends a graph that already has a forward pass + autograd-
// emitted backward (i.e., has paramGrads from `appendGrad`) with the SGD
// update math. The math is simple enough that no fused IR ops are needed:
// the per-param update composes from existing primitives (add, sub,
// mulScalar, broadcastTo).
//
// Per parameter P with gradient g:
//   g_eff = g + wd * P                      (when wd > 0 and P is decayed)
//   v_new = momentum * v + g_eff            (when momentum > 0)
//   update = nesterov ? g_eff + momentum * v_new : v_new
//   P_new = P - lr * update
//
// lr is supplied per step from CPU (so LR schedules work the same way they
// do for Adam — see `LR` in adam.ts).
//
// Edge cases the implementation respects:
//   - momentum = 0: no v state buffer, no v writeback. Plain SGD.
//   - wd = 0 or param not in decayedNames: no `g + wd*p` injection.
//   - nesterov requires momentum > 0; validated at appendSGD time.
//
// Note: this is PyTorch-style L2 weight decay (injected into the gradient),
// not decoupled like AdamW. The decoupled form is a less common SGD variant
// and would surface as a config flag if anyone asks.

import type { Tensor, Graph } from './ir.js'
import type { WritebackDecl } from './buffers.js'
import { traceInto, stateInput, tensorInput } from './trace.js'
import { add, sub, mul, mulScalar, broadcastTo } from './ops.js'
import { appendGradClip } from './adam.js'
import type { LR } from './adam.js'
import { isLRDynamic, resolveLR } from './adam.js'

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

export interface SGDResult {
  writebacks: WritebackDecl[]
  /** Name of the per-step scalar tensor_input. The runtime fills this each
   *  step with the current lr. */
  lrInputName: string
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
  // Global L2-norm clipping (if requested) runs before the rest of the
  // update; gradients are scaled in-graph and we consume the clipped values
  // as if they were the originals.
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

      // g_eff = g + wd * p   (when wd > 0 and p is decayed)
      const gEff = decayedNames.has(name)
        ? add(g, mulScalar(p, fullConfig.weightDecay))
        : g

      // velocity (only when momentum > 0)
      let update: Tensor
      if (momentum > 0) {
        const vState = stateInput(`sgd_v_${name}`, p.shape, 'f32', 0)
        const vNew = add(mulScalar(vState, momentum), gEff)
        update = nesterov ? add(gEff, mulScalar(vNew, momentum)) : vNew
        writebacks.push({ source: vNew, destName: `sgd_v_${name}`, destKind: 'state' })
      } else {
        update = gEff
      }

      // p_new = p - lr * update; broadcast 0-d lr to param shape.
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
