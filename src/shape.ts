// Shape inference and validation for each op kind.
//
// Every op in src/ops.ts validates its inputs and computes its output shape
// through helpers here. Errors throw with the captured call-site so the
// stack trace points at the user's line, not into the library.
//
// Broadcasting rules (deliberately limited):
//   * For element-wise binops (add/sub/mul/div), we support trailing-axis
//     broadcasting: the smaller operand's shape must be a suffix of the
//     larger's, with axes of size 1 broadcasting to any size. Examples
//     ALLOWED:  [B, T, D] op [D]  →  [B, T, D]
//               [B, T, D] op [1, D]  → [B, T, D]
//               [B, T, D] op [B, T, D]  → [B, T, D]
//     Examples REJECTED:  [B, T, D] op [B]   (suffix mismatch)
//                         [B, T, D] op [T, D] when T != B (legal numpy, banned here)
//   The restriction makes codegen and autograd much simpler and covers every
//   broadcast pattern in our transformer (biases, layernorm gain/bias, masks).

import type { Shape, CallSite } from './ir.js'
import { formatSite } from './ir.js'

// ============================================================================
// Errors
// ============================================================================

export class ShapeError extends Error {
  constructor(message: string, site: CallSite | null) {
    const formatted = site ? `${message}\n  at ${formatSite(site)}` : message
    super(formatted)
    this.name = 'ShapeError'
  }
}

function fail(message: string, site: CallSite | null): never {
  throw new ShapeError(message, site)
}

// ============================================================================
// Shape utilities
// ============================================================================

export function shapesEqual(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

export function shapeSize(shape: Shape): number {
  let n = 1
  for (const d of shape) n *= d
  return n
}

export function showShape(shape: Shape): string {
  return `[${shape.join(', ')}]`
}

// Standard right-aligned NumPy-style broadcasting. Pad the shorter shape with
// leading 1s, then per-axis: equal dims unify, size-1 dims broadcast on either
// side, otherwise incompatible. Returns the resulting shape or null.
export function broadcastTrailing(a: Shape, b: Shape): Shape | null {
  const rank = Math.max(a.length, b.length)
  const out: number[] = new Array(rank)
  for (let i = 0; i < rank; i++) {
    const ai = i - (rank - a.length)
    const bi = i - (rank - b.length)
    const av = ai < 0 ? 1 : a[ai]!
    const bv = bi < 0 ? 1 : b[bi]!
    if (av === bv) out[i] = av
    else if (av === 1) out[i] = bv
    else if (bv === 1) out[i] = av
    else return null
  }
  return out
}

// ============================================================================
// Per-op shape rules
// ============================================================================
//
// Each rule takes the input shapes and returns the output shape, or throws.
// All rules accept a `site` for error attribution.

export function inferElementwiseBinop(
  opName: string, aShape: Shape, bShape: Shape, site: CallSite | null,
): Shape {
  const result = broadcastTrailing(aShape, bShape)
  if (!result) {
    fail(
      `${opName}: incompatible shapes ${showShape(aShape)} and ${showShape(bShape)}. ` +
      `Trailing-suffix broadcasting only — the smaller shape must be a suffix of the larger, ` +
      `with size-1 axes broadcasting to any size.`,
      site,
    )
  }
  return result
}

export function inferUnary(_opName: string, aShape: Shape, _site: CallSite | null): Shape {
  return aShape
}

export function inferMeanLast(opName: string, aShape: Shape, site: CallSite | null): Shape {
  if (aShape.length === 0) fail(`${opName}: cannot reduce a 0-d tensor`, site)
  // keepdims=true: replace last axis with 1.
  return [...aShape.slice(0, -1), 1]
}

export function inferSumLast(opName: string, aShape: Shape, site: CallSite | null): Shape {
  if (aShape.length === 0) fail(`${opName}: cannot reduce a 0-d tensor`, site)
  // keepdims=false: drop the last axis.
  return aShape.slice(0, -1)
}

/** argmax_last shares sum_last's shape rule (drops the last axis). The
 *  output dtype is i32, set by the caller. */
export function inferArgmaxLast(opName: string, aShape: Shape, site: CallSite | null): Shape {
  return inferSumLast(opName, aShape, site)
}

export function inferReshape(opName: string, aShape: Shape, newShape: Shape, site: CallSite | null): Shape {
  // Validate -1 placeholder (at most one allowed) and total size match.
  let inferIdx = -1
  let knownSize = 1
  for (let i = 0; i < newShape.length; i++) {
    const d = newShape[i]!
    if (d === -1) {
      if (inferIdx !== -1) fail(`${opName}: at most one -1 dim allowed in newShape ${showShape(newShape)}`, site)
      inferIdx = i
    } else if (d <= 0) {
      fail(`${opName}: invalid dim ${d} in newShape ${showShape(newShape)}`, site)
    } else {
      knownSize *= d
    }
  }
  const totalIn = shapeSize(aShape)
  const out = [...newShape]
  if (inferIdx !== -1) {
    if (totalIn % knownSize !== 0) {
      fail(`${opName}: cannot reshape ${showShape(aShape)} (size ${totalIn}) to ${showShape(newShape)} — known dims multiply to ${knownSize}`, site)
    }
    out[inferIdx] = totalIn / knownSize
  } else if (knownSize !== totalIn) {
    fail(`${opName}: size mismatch — input ${showShape(aShape)} has ${totalIn} elements but newShape ${showShape(newShape)} has ${knownSize}`, site)
  }
  return out
}

export function inferTranspose(opName: string, aShape: Shape, perm: readonly number[], site: CallSite | null): Shape {
  if (perm.length !== aShape.length) {
    fail(`${opName}: perm length ${perm.length} must equal input rank ${aShape.length}`, site)
  }
  const seen = new Set<number>()
  for (const p of perm) {
    if (p < 0 || p >= aShape.length) fail(`${opName}: perm index ${p} out of range for rank ${aShape.length}`, site)
    if (seen.has(p)) fail(`${opName}: perm has duplicate index ${p}`, site)
    seen.add(p)
  }
  return perm.map(p => aShape[p]!)
}

// matmul: a [..., M, K] · b [K, N]  →  [..., M, N].  b is unbatched.
export function inferMatmul(opName: string, aShape: Shape, bShape: Shape, site: CallSite | null): Shape {
  if (aShape.length < 2) fail(`${opName}: lhs must have rank >= 2, got ${showShape(aShape)}`, site)
  if (bShape.length !== 2) fail(`${opName}: rhs must have rank 2, got ${showShape(bShape)} — use matmulBatched for batched rhs`, site)
  const M = aShape[aShape.length - 2]!
  const Ka = aShape[aShape.length - 1]!
  const Kb = bShape[0]!
  const N = bShape[1]!
  if (Ka !== Kb) fail(`${opName}: inner dims don't match — ${showShape(aShape)} · ${showShape(bShape)} (last axis of lhs = ${Ka}, first axis of rhs = ${Kb})`, site)
  return [...aShape.slice(0, -2), M, N]
}

// matmul_batched: a [..., M, K] · b [..., K, N]  →  [..., M, N].  Both have leading batch dims.
export function inferMatmulBatched(opName: string, aShape: Shape, bShape: Shape, site: CallSite | null): Shape {
  if (aShape.length < 2 || bShape.length < 2) {
    fail(`${opName}: both inputs must have rank >= 2, got ${showShape(aShape)} and ${showShape(bShape)}`, site)
  }
  if (aShape.length !== bShape.length) {
    fail(`${opName}: ranks must match (got ${aShape.length} vs ${bShape.length}). Reshape if you need different batch dims.`, site)
  }
  const aBatch = aShape.slice(0, -2)
  const bBatch = bShape.slice(0, -2)
  for (let i = 0; i < aBatch.length; i++) {
    if (aBatch[i] !== bBatch[i]) {
      fail(`${opName}: batch dims must match — ${showShape(aShape)} vs ${showShape(bShape)}`, site)
    }
  }
  const M = aShape[aShape.length - 2]!
  const Ka = aShape[aShape.length - 1]!
  const Kb = bShape[bShape.length - 2]!
  const N = bShape[bShape.length - 1]!
  if (Ka !== Kb) fail(`${opName}: inner dims don't match — last axis of lhs = ${Ka}, second-to-last of rhs = ${Kb}`, site)
  return [...aBatch, M, N]
}

export function inferOneHot(opName: string, indicesShape: Shape, depth: number, site: CallSite | null): Shape {
  if (depth <= 0) fail(`${opName}: depth must be positive, got ${depth}`, site)
  return [...indicesShape, depth]
}

// where_causal preserves shape but requires the last two axes to be square.
export function inferWhereCausal(opName: string, aShape: Shape, site: CallSite | null): Shape {
  if (aShape.length < 2) fail(`${opName}: requires rank >= 2, got ${showShape(aShape)}`, site)
  const m = aShape[aShape.length - 2]!
  const n = aShape[aShape.length - 1]!
  if (m !== n) fail(`${opName}: last two axes must be equal (square mask), got ${showShape(aShape)}`, site)
  return aShape
}

export function inferSliceLastRange(opName: string, aShape: Shape, start: number, end: number, site: CallSite | null): Shape {
  if (aShape.length === 0) fail(`${opName}: cannot slice 0-d tensor`, site)
  const last = aShape[aShape.length - 1]!
  if (start < 0 || end > last || start >= end) {
    fail(`${opName}: invalid range [${start}, ${end}) for last axis of size ${last}`, site)
  }
  return [...aShape.slice(0, -1), end - start]
}

// broadcast_to: validate that `aShape` can broadcast to `targetShape` under
// right-aligned NumPy rules. Returns targetShape on success.
export function inferBroadcastTo(opName: string, aShape: Shape, targetShape: Shape, site: CallSite | null): Shape {
  if (aShape.length > targetShape.length) {
    fail(`${opName}: source rank ${aShape.length} > target rank ${targetShape.length}`, site)
  }
  const offset = targetShape.length - aShape.length
  for (let i = 0; i < aShape.length; i++) {
    const av = aShape[i]!
    const tv = targetShape[offset + i]!
    if (av !== tv && av !== 1) {
      fail(`${opName}: cannot broadcast ${showShape(aShape)} to ${showShape(targetShape)} — axis ${i} (size ${av}) doesn't match target axis ${offset + i} (size ${tv}) and isn't 1`, site)
    }
  }
  return targetShape
}

// sum_to_shape: validate that `targetShape` is a valid right-aligned reduction
// of `aShape` (i.e., aShape can have been produced by broadcasting targetShape).
export function inferSumToShape(opName: string, aShape: Shape, targetShape: Shape, site: CallSite | null): Shape {
  if (targetShape.length > aShape.length) {
    fail(`${opName}: target rank ${targetShape.length} > source rank ${aShape.length}`, site)
  }
  const offset = aShape.length - targetShape.length
  for (let i = 0; i < targetShape.length; i++) {
    const av = aShape[offset + i]!
    const tv = targetShape[i]!
    if (av !== tv && tv !== 1) {
      fail(`${opName}: cannot sum-reduce ${showShape(aShape)} to ${showShape(targetShape)} — target axis ${i} (size ${tv}) must be 1 or match source`, site)
    }
  }
  return targetShape
}

// Three-way broadcast for `where(cond, a, b)`. All three shapes must broadcast
// to a common shape under standard NumPy rules.
export function inferWhere(opName: string, condShape: Shape, aShape: Shape, bShape: Shape, site: CallSite | null): Shape {
  const ab = broadcastTrailing(aShape, bShape)
  if (!ab) fail(`${opName}: a/b incompatible: ${showShape(aShape)} vs ${showShape(bShape)}`, site)
  const result = broadcastTrailing(condShape, ab)
  if (!result) fail(`${opName}: cond ${showShape(condShape)} incompatible with broadcast(a, b) ${showShape(ab)}`, site)
  return result
}

export function inferReluGrad(opName: string, xShape: Shape, dyShape: Shape, site: CallSite | null): Shape {
  if (!shapesEqual(xShape, dyShape)) {
    fail(`${opName}: x and dy must have matching shapes, got ${showShape(xShape)} and ${showShape(dyShape)}`, site)
  }
  return xShape
}
