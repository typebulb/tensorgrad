// Finite-difference vs autograd comparison harness.
//
// For each gradient test:
//   1. Build a small graph terminating in a scalar loss.
//   2. Compute the analytic gradient with appendGrad + CPU eval.
//   3. Compute the numerical gradient by perturbing the param value
//      ±eps and evaluating the forward loss.
//   4. Assert the two agree to a tolerance.
//
// This is the test that catches wrong-derivative bugs — the kind
// shape-only tests miss. A typo in tanh's backward (`(1 + c²)` instead
// of `(1 - c²)`) fails the FD comparison; the shape-only check passes.

import type { Tensor, Graph } from '../src/index.js'
import { traceFn, paramInput, appendGrad } from '../src/internal.js'
import { evalGraph, evalOutput } from './_eval.js'
import { fail, ok } from './_assert.js'

type LossBuilder = (p: Tensor) => Tensor

interface FDOpts {
  /** Perturbation step. f32 precision means too-small eps drowns in noise;
   *  too-large means we step into nonlinearity. 1e-3 is a reasonable default
   *  for activations / smooth ops; for more curvature use smaller. */
  eps?: number
  /** Per-element tolerance for absolute difference between FD and autograd. */
  atol?: number
  /** Per-element tolerance for relative difference. */
  rtol?: number
  /** Extra non-param inputs to bind during eval. Names must match
   *  tensorInput calls inside `build`. Defaults to `{}`. */
  extraInputs?: Record<string, Float32Array | Int32Array>
  /** Initial value for the param. Defaults to random uniform in [-0.5, 0.5]. */
  paramInit?: Float32Array
  /** Override the perturbation index step. Useful for large param tensors
   *  where you don't want to FD every entry. Default: every entry. */
  fdStride?: number
}

/** Assert that the autograd gradient of `build(p)` w.r.t. `p` matches the
 *  finite-difference gradient at a random initial point. */
export function assertGradMatchesFD(
  name: string,
  paramShape: readonly number[],
  build: LossBuilder,
  opts: FDOpts = {},
): void {
  const eps = opts.eps ?? 1e-3
  const atol = opts.atol ?? 5e-3
  const rtol = opts.rtol ?? 5e-3
  const extraInputs = opts.extraInputs ?? {}
  const fdStride = opts.fdStride ?? 1

  // Build the graph once for autograd evaluation.
  const graph: Graph = traceFn(() => build(paramInput('w', paramShape as number[])))
  appendGrad(graph)

  const paramSize = paramShape.reduce((p, d) => p * d, 1)
  const pInit = opts.paramInit ?? randomBuf(paramSize, /* seed */ 12345)

  // Autograd: bind p and evaluate. Find the gradient tensor for 'w'.
  // appendGrad attaches grad tensors but doesn't expose them by name post-hoc
  // on the graph itself — the param_input op's `out` is the param tensor; we
  // located the grad through paramGrads which we lost when we threw away the
  // appendGrad return. Re-call appendGrad on a fresh graph to get the handle.
  const freshGraph: Graph = traceFn(() => build(paramInput('w', paramShape as number[])))
  const { paramGrads } = appendGrad(freshGraph)
  const gradTensor = paramGrads['w']
  if (!gradTensor) fail(`${name}: appendGrad produced no gradient for 'w'`)
  const vals = evalGraph(freshGraph, { w: pInit, ...extraInputs })
  const autogradGrad = vals.get(gradTensor!.id) as Float32Array
  if (!autogradGrad) fail(`${name}: autograd gradient tensor not found in eval result`)

  // FD: perturb each entry of p by ±eps, evaluate the loss, compute the
  // central-difference numerical gradient.
  const fdGrad = new Float32Array(paramSize)
  for (let i = 0; i < paramSize; i += fdStride) {
    const orig = pInit[i]!
    pInit[i] = orig + eps
    const lossPlus = scalarLoss(graph, { w: pInit, ...extraInputs }, name)
    pInit[i] = orig - eps
    const lossMinus = scalarLoss(graph, { w: pInit, ...extraInputs }, name)
    pInit[i] = orig
    fdGrad[i] = (lossPlus - lossMinus) / (2 * eps)
  }

  // Compare.
  let worstAbs = 0, worstRel = 0, worstAt = 0
  for (let i = 0; i < paramSize; i += fdStride) {
    const a = autogradGrad[i]!, f = fdGrad[i]!
    const absErr = Math.abs(a - f)
    const relErr = absErr / Math.max(Math.abs(f), 1e-6)
    if (absErr > worstAbs) { worstAbs = absErr; worstAt = i }
    if (relErr > worstRel) worstRel = relErr
  }

  if (worstAbs > atol && worstRel > rtol) {
    fail(
      `${name}: gradient mismatch at index ${worstAt}. ` +
      `autograd=${autogradGrad[worstAt]!.toFixed(6)}, fd=${fdGrad[worstAt]!.toFixed(6)}, ` +
      `|Δ|=${worstAbs.toExponential(2)} (atol=${atol}), |Δ|/|fd|=${worstRel.toExponential(2)} (rtol=${rtol})`,
    )
  }
  ok(`${name} grad matches FD: max |Δ|=${worstAbs.toExponential(2)}, max rel=${worstRel.toExponential(2)}`)
}

function scalarLoss(graph: Graph, inputs: Record<string, Float32Array | Int32Array>, name: string): number {
  const out = evalOutput(graph, inputs) as Float32Array
  if (out.length !== 1) fail(`${name}: loss must be a scalar, got length ${out.length}`)
  return out[0]!
}

/** Deterministic small random buffer. Mulberry32-equivalent to the lib's
 *  init seed PRNG so behavior is reproducible across test runs. */
function randomBuf(size: number, seed: number): Float32Array {
  let s = seed >>> 0
  const out = new Float32Array(size)
  for (let i = 0; i < size; i++) {
    s = (s + 0x6D2B79F5) | 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    out[i] = ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5  // [-0.5, 0.5)
  }
  return out
}
