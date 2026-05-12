// Shape inference and validation. Broadcasting is deliberately limited to
// trailing-axis (suffix) broadcasts with size-1 axes — `[B,T,D] op [D]` is
// allowed, `[B,T,D] op [B]` and middle-axis broadcasts are not. This covers
// every pattern in transformer/CNN code we care about and keeps codegen and
// autograd straightforward.

import type { Shape, CallSite } from './ir.js'
import { formatSite } from './ir.js'

/** Thrown when an op's shape constraints are violated (rank mismatch,
 *  non-broadcastable dims, axis out of range, etc.). The message includes
 *  the captured `CallSite` so the user's frame appears first in the stack. */
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

// Right-aligned NumPy broadcasting. Equal dims unify, size-1 dims broadcast,
// otherwise incompatible. Returns null on incompatibility.
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

// Per-op shape rules. Each returns the output shape or throws via `fail`.

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

// mean_last keeps dims (last axis → 1); sum_last drops the last axis.
export function inferMeanLast(opName: string, aShape: Shape, site: CallSite | null): Shape {
  if (aShape.length === 0) fail(`${opName}: cannot reduce a 0-d tensor`, site)
  return [...aShape.slice(0, -1), 1]
}

export function inferSumLast(opName: string, aShape: Shape, site: CallSite | null): Shape {
  if (aShape.length === 0) fail(`${opName}: cannot reduce a 0-d tensor`, site)
  return aShape.slice(0, -1)
}

/** argmax_last shares sum_last's shape rule (drops the last axis). The
 *  output dtype is i32, set by the caller. */
export function inferArgmaxLast(opName: string, aShape: Shape, site: CallSite | null): Shape {
  return inferSumLast(opName, aShape, site)
}

export function inferReshape(opName: string, aShape: Shape, newShape: Shape, site: CallSite | null): Shape {
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

export function inferPermute(opName: string, aShape: Shape, perm: readonly number[], site: CallSite | null): Shape {
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

export function inferMatmul(opName: string, aShape: Shape, bShape: Shape, site: CallSite | null): Shape {
  if (aShape.length < 2) fail(`${opName}: lhs must have rank >= 2, got ${showShape(aShape)}`, site)
  if (bShape.length !== 2) fail(`${opName}: internal: inferMatmul expects rank-2 rhs, got ${showShape(bShape)}`, site)
  const M = aShape[aShape.length - 2]!
  const Ka = aShape[aShape.length - 1]!
  const Kb = bShape[0]!
  const N = bShape[1]!
  if (Ka !== Kb) fail(`${opName}: inner dims don't match — ${showShape(aShape)} · ${showShape(bShape)} (last axis of lhs = ${Ka}, first axis of rhs = ${Kb})`, site)
  return [...aShape.slice(0, -2), M, N]
}

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

// Requires the last two axes to be square; shape preserved.
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

/** General-axis slice. `axis` is non-negative; callers must normalize any
 *  negative-axis input before calling. */
export function inferSliceRange(opName: string, aShape: Shape, axis: number, start: number, end: number, site: CallSite | null): Shape {
  if (aShape.length === 0) fail(`${opName}: cannot slice 0-d tensor`, site)
  if (axis < 0 || axis >= aShape.length) {
    fail(`${opName}: axis ${axis} out of range for shape ${showShape(aShape)}`, site)
  }
  const dim = aShape[axis]!
  if (start < 0 || end > dim || start >= end) {
    fail(`${opName}: invalid range [${start}, ${end}) for axis ${axis} of size ${dim}`, site)
  }
  const out = aShape.slice()
  out[axis] = end - start
  return out
}

/** Scatter-into-zero: place `a` into `[start, end)` along `axis` of an
 *  otherwise-zero tensor of `outShape`. Validates that `a` matches `outShape`
 *  everywhere except along `axis`, where its size equals `end - start`. */
export function inferScatterAxis(
  opName: string, aShape: Shape, outShape: Shape,
  axis: number, start: number, end: number, site: CallSite | null,
): Shape {
  if (outShape.length === 0) fail(`${opName}: cannot scatter into 0-d tensor`, site)
  if (axis < 0 || axis >= outShape.length) {
    fail(`${opName}: axis ${axis} out of range for output shape ${showShape(outShape)}`, site)
  }
  const dim = outShape[axis]!
  if (start < 0 || end > dim || start >= end) {
    fail(`${opName}: invalid range [${start}, ${end}) for axis ${axis} of size ${dim}`, site)
  }
  if (aShape.length !== outShape.length) {
    fail(`${opName}: input rank ${aShape.length} must equal output rank ${outShape.length}`, site)
  }
  for (let i = 0; i < outShape.length; i++) {
    const expected = i === axis ? end - start : outShape[i]!
    if (aShape[i]! !== expected) {
      fail(`${opName}: input ${showShape(aShape)} doesn't match output ${showShape(outShape)} at axis ${i} (expected ${expected})`, site)
    }
  }
  return outShape
}

/** Concat along `axis`. All inputs must have identical shape except along
 *  `axis`; output's size on `axis` is the sum. `axis` is non-negative. */
export function inferConcat(opName: string, shapes: readonly Shape[], axis: number, site: CallSite | null): Shape {
  if (shapes.length === 0) fail(`${opName}: needs at least one input`, site)
  const first = shapes[0]!
  if (axis < 0 || axis >= first.length) {
    fail(`${opName}: axis ${axis} out of range for shape ${showShape(first)}`, site)
  }
  let axisTotal = first[axis]!
  for (let i = 1; i < shapes.length; i++) {
    const s = shapes[i]!
    if (s.length !== first.length) {
      fail(`${opName}: input ${i} has rank ${s.length}, expected ${first.length}`, site)
    }
    for (let d = 0; d < first.length; d++) {
      if (d === axis) continue
      if (s[d]! !== first[d]!) {
        fail(`${opName}: input ${i} has shape ${showShape(s)}, must match ${showShape(first)} except along axis ${axis}`, site)
      }
    }
    axisTotal += s[axis]!
  }
  const out = first.slice()
  out[axis] = axisTotal
  return out
}

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

// Inverse of broadcast_to: targetShape must be a valid right-aligned reduction
// of aShape (i.e. aShape could have been produced by broadcasting targetShape).
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

// Three-way broadcast for where(cond, a, b).
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

/** Output spatial size for a 2D conv / pool. PyTorch convention:
 *  `out = floor((in + 2*pad - kernel) / stride) + 1`.
 *  Throws if the result would be < 1. */
export function inferConv2dOutputSpatial(
  opName: string,
  inSize: number, kernel: number, stride: number, pad: number,
  axisName: string, site: CallSite | null,
): number {
  if (kernel < 1) fail(`${opName}: ${axisName} kernel must be >= 1, got ${kernel}`, site)
  if (stride < 1) fail(`${opName}: ${axisName} stride must be >= 1, got ${stride}`, site)
  if (pad < 0) fail(`${opName}: ${axisName} padding must be >= 0, got ${pad}`, site)
  const out = Math.floor((inSize + 2 * pad - kernel) / stride) + 1
  if (out < 1) fail(`${opName}: ${axisName} output size would be ${out} (in=${inSize}, kernel=${kernel}, stride=${stride}, pad=${pad})`, site)
  return out
}

/** conv2d: input [B, C_in, H, W] · weight [C_out, C_in, K_h, K_w]
 *  -> [B, C_out, H_out, W_out]. */
export function inferConv2d(
  opName: string,
  inputShape: Shape, weightShape: Shape,
  strideH: number, strideW: number, padH: number, padW: number,
  site: CallSite | null,
): Shape {
  if (inputShape.length !== 4) {
    fail(`${opName}: input must be rank-4 [B, C_in, H, W], got ${showShape(inputShape)}`, site)
  }
  if (weightShape.length !== 4) {
    fail(`${opName}: weight must be rank-4 [C_out, C_in, K_h, K_w], got ${showShape(weightShape)}`, site)
  }
  const [B, cIn, H, W] = [inputShape[0]!, inputShape[1]!, inputShape[2]!, inputShape[3]!]
  const [cOut, wInC, kH, kW] = [weightShape[0]!, weightShape[1]!, weightShape[2]!, weightShape[3]!]
  if (cIn !== wInC) {
    fail(`${opName}: input C_in=${cIn} doesn't match weight C_in=${wInC}`, site)
  }
  const hOut = inferConv2dOutputSpatial(opName, H, kH, strideH, padH, 'H', site)
  const wOut = inferConv2dOutputSpatial(opName, W, kW, strideW, padW, 'W', site)
  return [B, cOut, hOut, wOut]
}

/** max_pool_2d: input [B, C, H, W] -> [B, C, H_out, W_out]. */
export function inferMaxPool2d(
  opName: string,
  inputShape: Shape,
  kH: number, kW: number, strideH: number, strideW: number, padH: number, padW: number,
  site: CallSite | null,
): Shape {
  if (inputShape.length !== 4) {
    fail(`${opName}: input must be rank-4 [B, C, H, W], got ${showShape(inputShape)}`, site)
  }
  const [B, C, H, W] = [inputShape[0]!, inputShape[1]!, inputShape[2]!, inputShape[3]!]
  const hOut = inferConv2dOutputSpatial(opName, H, kH, strideH, padH, 'H', site)
  const wOut = inferConv2dOutputSpatial(opName, W, kW, strideW, padW, 'W', site)
  return [B, C, hOut, wOut]
}
