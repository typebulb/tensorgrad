// CPU evaluator for the tensorgrad IR. Walks the graph in topological order
// and computes each op's output value in JS — same arithmetic the WGSL
// kernels are supposed to implement, just running on the CPU. Used by the
// test suite to:
//
//   * verify forward results numerically (does relu(-3) really return 0?)
//   * verify gradients by finite-difference comparison against autograd
//
// Mirrors src/codegen.ts closely on purpose: if the two diverge, the GPU
// is doing something different from the spec, and one or the other is
// wrong. Both should compute the same answer up to f32 precision.

import type { Graph, OpNode, Shape } from '../src/ir.js'

type Val = Float32Array | Int32Array

/** Evaluate every op in `graph` in order, returning a Map from tensor id to
 *  its computed value. Inputs (tensor_input / param_input) come from
 *  `inputs`; state_input slots are zero-initialized unless overridden in
 *  inputs. */
export function evalGraph(graph: Graph, inputs: Record<string, Val>): Map<number, Val> {
  const vals = new Map<number, Val>()
  for (const op of graph.ops) {
    vals.set(op.out, evalOp(op, vals, inputs, graph))
  }
  return vals
}

/** Convenience: evaluate and return the named output tensor's value. */
export function evalOutput(graph: Graph, inputs: Record<string, Val>): Val {
  const vals = evalGraph(graph, inputs)
  return vals.get(graph.outputs[0]!)!
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function shapeSize(s: Shape): number { let n = 1; for (const d of s) n *= d; return n }

function makeBuf(size: number, dtype: 'f32' | 'i32' | 'bool'): Val {
  if (dtype === 'i32' || dtype === 'bool') return new Int32Array(size)
  return new Float32Array(size)
}

/** Right-aligned NumPy broadcasting: decompose `outIdx` into multi-dim
 *  according to `outShape`, then map back to a flat index into a value
 *  with `srcShape` (size-1 axes broadcast). */
function broadcastIdx(outIdx: number, outShape: Shape, srcShape: Shape): number {
  const rankDiff = outShape.length - srcShape.length
  // Compute strides from the right (NumPy convention).
  let aIdx = 0
  let aStride = 1
  let rem = outIdx
  // Walk axes right-to-left; for each output axis, get its coordinate.
  // For src axes that align to it (axis index ≥ rankDiff), contribute
  // (coord % srcDim) * stride; src size-1 axes contribute 0.
  for (let d = outShape.length - 1; d >= 0; d--) {
    const outDim = outShape[d]!
    const coord = rem % outDim
    rem = Math.floor(rem / outDim)
    const srcAxis = d - rankDiff
    if (srcAxis >= 0) {
      const srcDim = srcShape[srcAxis]!
      if (srcDim !== 1) aIdx += coord * aStride
      aStride *= srcDim
    }
  }
  return aIdx
}

function strides(shape: Shape): number[] {
  const s = new Array(shape.length).fill(1)
  for (let i = shape.length - 2; i >= 0; i--) s[i] = s[i + 1]! * shape[i + 1]!
  return s
}

/** Resolve reshape's -1 placeholder against an explicit total size. */
function resolveReshape(newShape: Shape, totalSize: number): number[] {
  const out = newShape.slice() as number[]
  let inferIdx = -1, known = 1
  for (let i = 0; i < out.length; i++) {
    if (out[i] === -1) inferIdx = i
    else known *= out[i]!
  }
  if (inferIdx >= 0) out[inferIdx] = totalSize / known
  return out
}

// ---------------------------------------------------------------------------
// Per-op evaluation
// ---------------------------------------------------------------------------

function evalOp(op: OpNode, vals: Map<number, Val>, inputs: Record<string, Val>, graph: Graph): Val {
  const t = graph.tensors[op.out]!
  const shape = t.shape
  const dtype = t.dtype

  const v = (id: number): Val => vals.get(id)!

  switch (op.kind) {
    // ---- Leaves ----------------------------------------------------------
    case 'param_input':
    case 'tensor_input': {
      const src = inputs[op.name]
      if (!src) throw new Error(`eval: missing input '${op.name}'`)
      return src
    }
    case 'state_input': {
      const size = shapeSize(shape)
      const buf = makeBuf(size, dtype)
      if (op.initValue !== 0 && dtype === 'f32') (buf as Float32Array).fill(op.initValue)
      return inputs[op.name] ?? buf
    }
    case 'const_scalar': {
      const buf = makeBuf(1, dtype)
      buf[0] = op.value
      return buf
    }
    case 'arange': {
      const buf = makeBuf(op.n, dtype)
      for (let i = 0; i < op.n; i++) buf[i] = i
      return buf
    }

    // ---- Element-wise binary (broadcasting) -----------------------------
    case 'add':
    case 'sub':
    case 'mul':
    case 'div':
    case 'min':
    case 'max': {
      const a = v(op.a), b = v(op.b)
      const aShape = graph.tensors[op.a]!.shape, bShape = graph.tensors[op.b]!.shape
      const total = shapeSize(shape)
      const out = makeBuf(total, dtype) as Float32Array
      const apply =
        op.kind === 'add' ? (x: number, y: number) => x + y :
        op.kind === 'sub' ? (x: number, y: number) => x - y :
        op.kind === 'mul' ? (x: number, y: number) => x * y :
        op.kind === 'div' ? (x: number, y: number) => x / y :
        op.kind === 'min' ? (x: number, y: number) => Math.min(x, y) :
        /* max */            (x: number, y: number) => Math.max(x, y)
      for (let i = 0; i < total; i++) {
        out[i] = apply(a[broadcastIdx(i, shape, aShape)]!, b[broadcastIdx(i, shape, bShape)]!)
      }
      return out
    }
    case 'mul_scalar':
    case 'add_scalar': {
      const a = v(op.a) as Float32Array
      const total = shapeSize(shape)
      const out = new Float32Array(total)
      const s = op.scalar
      if (op.kind === 'mul_scalar') for (let i = 0; i < total; i++) out[i] = a[i]! * s
      else                          for (let i = 0; i < total; i++) out[i] = a[i]! + s
      return out
    }

    // ---- Unary -----------------------------------------------------------
    case 'sqrt':
    case 'rsqrt':
    case 'log':
    case 'exp':
    case 'relu':
    case 'neg':
    case 'abs':
    case 'tanh':
    case 'sigmoid': {
      const a = v(op.a) as Float32Array
      const out = new Float32Array(a.length)
      const fn =
        op.kind === 'sqrt'    ? Math.sqrt :
        op.kind === 'rsqrt'   ? (x: number) => 1 / Math.sqrt(x) :
        op.kind === 'log'     ? Math.log :
        op.kind === 'exp'     ? Math.exp :
        op.kind === 'relu'    ? (x: number) => Math.max(x, 0) :
        op.kind === 'neg'     ? (x: number) => -x :
        op.kind === 'abs'     ? Math.abs :
        op.kind === 'tanh'    ? Math.tanh :
        /* sigmoid via tanh identity — matches the WGSL kernel exactly */
                                (x: number) => 0.5 + 0.5 * Math.tanh(0.5 * x)
      for (let i = 0; i < a.length; i++) out[i] = fn(a[i]!)
      return out
    }

    // ---- Reductions ------------------------------------------------------
    case 'mean_last':
    case 'sum_last': {
      const a = v(op.a) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const D = aShape[aShape.length - 1]!
      const outer = shapeSize(aShape) / D
      const out = new Float32Array(outer)
      const div = op.kind === 'mean_last' ? D : 1
      for (let i = 0; i < outer; i++) {
        let s = 0
        for (let j = 0; j < D; j++) s += a[i * D + j]!
        out[i] = s / div
      }
      return out
    }
    case 'argmax_last': {
      const a = v(op.a) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const D = aShape[aShape.length - 1]!
      const outer = shapeSize(aShape) / D
      const out = new Int32Array(outer)
      for (let i = 0; i < outer; i++) {
        let best = 0, bestVal = a[i * D]!
        for (let j = 1; j < D; j++) {
          const x = a[i * D + j]!
          if (x > bestVal) { bestVal = x; best = j }
        }
        out[i] = best
      }
      return out
    }

    // ---- Shape -----------------------------------------------------------
    case 'reshape': {
      // Same backing data, just a new shape; copy to a fresh buffer to keep
      // value provenance independent.
      const a = v(op.a)
      return a.slice() as Val
    }
    case 'permute': {
      const a = v(op.a)
      const aShape = graph.tensors[op.a]!.shape
      const aStr = strides(aShape)
      // Output stride layout follows op.perm: out[d] index → a[perm[d]] index.
      const outStr = strides(shape)
      const out = makeBuf(shapeSize(shape), dtype)
      const total = shapeSize(shape)
      const rank = shape.length
      for (let i = 0; i < total; i++) {
        // Decompose i into out's coords
        let rem = i
        let aIdx = 0
        for (let d = 0; d < rank; d++) {
          const coord = Math.floor(rem / outStr[d]!)
          rem -= coord * outStr[d]!
          aIdx += coord * aStr[op.perm[d]!]!
        }
        out[i] = a[aIdx]!
      }
      return out
    }

    // ---- Linear algebra --------------------------------------------------
    case 'matmul': {
      // a [..., M, K] · b [K, N] → [..., M, N]
      const a = v(op.a) as Float32Array, b = v(op.b) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const bShape = graph.tensors[op.b]!.shape
      const K = bShape[0]!, N = bShape[1]!
      const M = aShape[aShape.length - 2]!
      const batch = shapeSize(aShape) / (M * K)
      const out = new Float32Array(batch * M * N)
      for (let bi = 0; bi < batch; bi++) {
        for (let m = 0; m < M; m++) {
          for (let n = 0; n < N; n++) {
            let s = 0
            for (let k = 0; k < K; k++) {
              s += a[bi * M * K + m * K + k]! * b[k * N + n]!
            }
            out[bi * M * N + m * N + n] = s
          }
        }
      }
      return out
    }
    case 'matmul_batched': {
      // a [..., M, K] · b [..., K, N] → [..., M, N]
      const a = v(op.a) as Float32Array, b = v(op.b) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const bShape = graph.tensors[op.b]!.shape
      const M = aShape[aShape.length - 2]!, K = aShape[aShape.length - 1]!
      const N = bShape[bShape.length - 1]!
      const batch = shapeSize(aShape) / (M * K)
      const out = new Float32Array(batch * M * N)
      for (let bi = 0; bi < batch; bi++) {
        for (let m = 0; m < M; m++) {
          for (let n = 0; n < N; n++) {
            let s = 0
            for (let k = 0; k < K; k++) {
              s += a[bi * M * K + m * K + k]! * b[bi * K * N + k * N + n]!
            }
            out[bi * M * N + m * N + n] = s
          }
        }
      }
      return out
    }

    // ---- Indexing / casting ---------------------------------------------
    case 'one_hot': {
      const idx = v(op.indices) as Int32Array
      const out = makeBuf(shapeSize(shape), dtype) as Float32Array
      const D = op.depth
      for (let i = 0; i < idx.length; i++) {
        const k = idx[i]!
        if (k >= 0 && k < D) out[i * D + k] = 1
      }
      return out
    }

    // ---- Comparisons + select -------------------------------------------
    case 'less':
    case 'greater': {
      const a = v(op.a), b = v(op.b)
      const aShape = graph.tensors[op.a]!.shape, bShape = graph.tensors[op.b]!.shape
      const total = shapeSize(shape)
      const out = new Int32Array(total)
      const cmp = op.kind === 'less' ? (x: number, y: number) => x < y : (x: number, y: number) => x > y
      for (let i = 0; i < total; i++) {
        const av = (a[broadcastIdx(i, shape, aShape)] as number)
        const bv = (b[broadcastIdx(i, shape, bShape)] as number)
        out[i] = cmp(av, bv) ? 1 : 0
      }
      return out
    }
    case 'where': {
      const cond = v(op.cond) as Int32Array
      const a = v(op.a), b = v(op.b)
      const cShape = graph.tensors[op.cond]!.shape
      const aShape = graph.tensors[op.a]!.shape
      const bShape = graph.tensors[op.b]!.shape
      const total = shapeSize(shape)
      const out = makeBuf(total, dtype) as Float32Array
      for (let i = 0; i < total; i++) {
        const c = cond[broadcastIdx(i, shape, cShape)]
        const av = (a[broadcastIdx(i, shape, aShape)] as number)
        const bv = (b[broadcastIdx(i, shape, bShape)] as number)
        out[i] = c !== 0 ? av : bv
      }
      return out
    }

    // ---- Fused ML --------------------------------------------------------
    case 'log_softmax_last': {
      const a = v(op.a) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const D = aShape[aShape.length - 1]!
      const outer = shapeSize(aShape) / D
      const out = new Float32Array(a.length)
      for (let i = 0; i < outer; i++) {
        // Numerically stable: subtract max, exponentiate, log of sum.
        let max = -Infinity
        for (let j = 0; j < D; j++) if (a[i * D + j]! > max) max = a[i * D + j]!
        let sumExp = 0
        for (let j = 0; j < D; j++) sumExp += Math.exp(a[i * D + j]! - max)
        const logSum = Math.log(sumExp) + max
        for (let j = 0; j < D; j++) out[i * D + j] = a[i * D + j]! - logSum
      }
      return out
    }
    case 'softmax_causal_last': {
      // softmax(causal mask + a) along last axis. Lower-triangle including
      // diagonal pass through; upper-triangle treated as -inf (mask to 0
      // post-softmax). Square last-2 axes.
      const a = v(op.a) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const D = aShape[aShape.length - 1]!
      // square last-2: aShape[-2] == D
      const outer = shapeSize(aShape) / (D * D)
      const out = new Float32Array(a.length)
      for (let bi = 0; bi < outer; bi++) {
        for (let i = 0; i < D; i++) {
          // For row i: only columns 0..i (inclusive) contribute. Others = 0.
          let max = -Infinity
          for (let j = 0; j <= i; j++) {
            const x = a[bi * D * D + i * D + j]!
            if (x > max) max = x
          }
          let sumExp = 0
          for (let j = 0; j <= i; j++) sumExp += Math.exp(a[bi * D * D + i * D + j]! - max)
          for (let j = 0; j < D; j++) {
            if (j <= i) out[bi * D * D + i * D + j] = Math.exp(a[bi * D * D + i * D + j]! - max) / sumExp
            else        out[bi * D * D + i * D + j] = 0
          }
        }
      }
      return out
    }
    case 'where_causal': {
      const a = v(op.a) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const D = aShape[aShape.length - 1]!
      const outer = shapeSize(aShape) / (D * D)
      const out = new Float32Array(a.length)
      for (let bi = 0; bi < outer; bi++) {
        for (let i = 0; i < D; i++) {
          for (let j = 0; j < D; j++) {
            out[bi * D * D + i * D + j] = j <= i ? a[bi * D * D + i * D + j]! : op.fillValue
          }
        }
      }
      return out
    }

    // ---- Slicing / structural -------------------------------------------
    case 'slice_last_range': {
      const a = v(op.a) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const D_in = aShape[aShape.length - 1]!
      const D_out = op.end - op.start
      const outer = shapeSize(aShape) / D_in
      const out = new Float32Array(outer * D_out)
      for (let i = 0; i < outer; i++) {
        for (let j = 0; j < D_out; j++) {
          out[i * D_out + j] = a[i * D_in + op.start + j]!
        }
      }
      return out
    }
    case 'slice_range': {
      const a = v(op.a) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const ax = op.axis
      const inner = aShape.slice(ax + 1).reduce((p, d) => p * d, 1)
      const D_in = aShape[ax]!
      const D_out = op.end - op.start
      const outer = shapeSize(aShape) / (D_in * inner)
      const out = new Float32Array(outer * D_out * inner)
      for (let i = 0; i < outer; i++) {
        for (let j = 0; j < D_out; j++) {
          for (let k = 0; k < inner; k++) {
            out[i * D_out * inner + j * inner + k] = a[i * D_in * inner + (j + op.start) * inner + k]!
          }
        }
      }
      return out
    }
    case 'scatter_axis': {
      const a = v(op.a) as Float32Array
      const ax = op.axis
      const inner = shape.slice(ax + 1).reduce((p, d) => p * d, 1)
      const D_out = shape[ax]!
      const D_in = op.end - op.start
      const outer = shapeSize(shape) / (D_out * inner)
      const out = new Float32Array(shapeSize(shape))
      for (let i = 0; i < outer; i++) {
        for (let j = 0; j < D_out; j++) {
          if (j < op.start || j >= op.end) continue
          const ja = j - op.start
          for (let k = 0; k < inner; k++) {
            out[i * D_out * inner + j * inner + k] = a[i * D_in * inner + ja * inner + k]!
          }
        }
      }
      return out
    }
    case 'concat': {
      const axis = op.axis
      const inner = shape.slice(axis + 1).reduce((p, d) => p * d, 1)
      const D_out = shape[axis]!
      const outer = shapeSize(shape) / (D_out * inner)
      const out = new Float32Array(shapeSize(shape))
      const inputAxisSizes = op.inputs.map(id => graph.tensors[id]!.shape[axis]!)
      const inputBufs = op.inputs.map(id => v(id) as Float32Array)
      for (let i = 0; i < outer; i++) {
        for (let j = 0; j < D_out; j++) {
          // Locate which input this axis-position belongs to.
          let inputIdx = 0, localJ = j
          while (localJ >= inputAxisSizes[inputIdx]!) { localJ -= inputAxisSizes[inputIdx]!; inputIdx++ }
          const D_in = inputAxisSizes[inputIdx]!
          for (let k = 0; k < inner; k++) {
            out[i * D_out * inner + j * inner + k] =
              inputBufs[inputIdx]![i * D_in * inner + localJ * inner + k]!
          }
        }
      }
      return out
    }

    // ---- Broadcast / sum-to-shape (autograd internals) -------------------
    case 'broadcast_to': {
      const a = v(op.a) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const total = shapeSize(shape)
      const out = new Float32Array(total)
      for (let i = 0; i < total; i++) out[i] = a[broadcastIdx(i, shape, aShape)]!
      return out
    }
    case 'sum_to_shape': {
      // Inverse of broadcast_to: each output cell sums over the source axes
      // that were broadcast (right-aligned).
      const a = v(op.a) as Float32Array
      const aShape = graph.tensors[op.a]!.shape
      const total = shapeSize(shape)
      const out = new Float32Array(total)
      const aTotal = shapeSize(aShape)
      for (let aIdx = 0; aIdx < aTotal; aIdx++) {
        // Find which output cell this source cell contributes to.
        // Decompose aIdx into multi-dim, drop leading axes that target doesn't have,
        // collapse size-1 target axes to coord 0.
        const rankDiff = aShape.length - shape.length
        let rem = aIdx
        let outIdx = 0
        let outStride = 1
        // Walk axes right-to-left
        const coords: number[] = new Array(aShape.length)
        for (let d = aShape.length - 1; d >= 0; d--) {
          coords[d] = rem % aShape[d]!
          rem = Math.floor(rem / aShape[d]!)
        }
        for (let d = shape.length - 1; d >= 0; d--) {
          const tgtDim = shape[d]!
          const srcCoord = coords[d + rankDiff]!
          const coord = tgtDim === 1 ? 0 : srcCoord
          outIdx += coord * outStride
          outStride *= tgtDim
        }
        out[outIdx] = out[outIdx]! + a[aIdx]!
      }
      return out
    }

    // ---- ReLU's autograd-internal grad op -------------------------------
    case 'relu_grad': {
      // out = dy where x > 0, else 0
      const x = v(op.x) as Float32Array, dy = v(op.dy) as Float32Array
      const out = new Float32Array(x.length)
      for (let i = 0; i < x.length; i++) out[i] = x[i]! > 0 ? dy[i]! : 0
      return out
    }

    // ---- Stochastic -----------------------------------------------------
    case 'dropout': {
      // Mirrors codegen.ts's PCG hash exactly so CPU + GPU produce the same
      // mask given the same (seed, salt, thread). Seed comes from the input
      // bound at op.seed (a 1-element i32 tensor).
      const a = v(op.a) as Float32Array
      const seedBuf = v(op.seed) as Int32Array
      const seed = seedBuf[0]! >>> 0
      const saltConst = (op.salt * 0x9E3779B1) >>> 0
      const scale = 1 / (1 - op.p)
      const out = new Float32Array(a.length)
      for (let i = 0; i < a.length; i++) {
        let h = ((seed ^ saltConst ^ i) >>> 0)
        h = ((h * 747796405 + 2891336453) >>> 0)
        h = ((((h ^ (h >>> ((h >>> 28) + 4))) >>> 0) * 277803737) >>> 0)
        h = (h ^ (h >>> 22)) >>> 0
        const u = h / 4294967296
        const mask = u >= op.p ? scale : 0
        out[i] = a[i]! * mask
      }
      return out
    }

    // ---- Adam-fused ops (optimizer internals; eval is provided for       )
    //      completeness but the test suite doesn't usually run them).      )
    case 'adam_update_m':
    case 'adam_update_v':
    case 'adam_update_p':
      throw new Error(`eval: ${op.kind} is an optimizer-fused op; tests should not need to evaluate it`)

    // ---- 2D conv + pool. Mirror WGSL in src/codegen.ts byte-for-byte
    //      in terms of index math + scan order (especially for the pool
    //      argmax-on-ties behavior). ----------------------------------------
    case 'conv2d': {
      const input = vals.get(op.input)! as Float32Array
      const weight = vals.get(op.weight)! as Float32Array
      const inT = graph.tensors[op.input]!
      const wT = graph.tensors[op.weight]!
      const outT = graph.tensors[op.out]!
      const [, cIn, H, W] = inT.shape
      const [cOut, , kH, kW] = wT.shape
      const [, , hOut, wOut] = outT.shape
      const out = new Float32Array(shapeSize(outT.shape))
      const B = outT.shape[0]!
      for (let b = 0; b < B; b++) for (let co = 0; co < cOut!; co++) for (let ho = 0; ho < hOut!; ho++) for (let wo = 0; wo < wOut!; wo++) {
        let s = 0
        for (let ci = 0; ci < cIn!; ci++) for (let kh = 0; kh < kH!; kh++) {
          const hi = ho * op.strideH + kh - op.padH
          if (hi < 0 || hi >= H!) continue
          for (let kw = 0; kw < kW!; kw++) {
            const wi = wo * op.strideW + kw - op.padW
            if (wi < 0 || wi >= W!) continue
            s += input[b * cIn! * H! * W! + ci * H! * W! + hi * W! + wi]!
               * weight[co * cIn! * kH! * kW! + ci * kH! * kW! + kh * kW! + kw]!
          }
        }
        out[b * cOut! * hOut! * wOut! + co * hOut! * wOut! + ho * wOut! + wo] = s
      }
      return out
    }
    case 'conv2d_input_grad': {
      const weight = vals.get(op.weight)! as Float32Array
      const dy = vals.get(op.dy)! as Float32Array
      const wT = graph.tensors[op.weight]!
      const dyT = graph.tensors[op.dy]!
      const outT = graph.tensors[op.out]!
      const [cOut, cIn, kH, kW] = wT.shape
      const [, , hOut, wOut] = dyT.shape
      const [B, , inH, inW] = outT.shape
      const out = new Float32Array(shapeSize(outT.shape))
      for (let b = 0; b < B!; b++) for (let ci = 0; ci < cIn!; ci++) for (let hi = 0; hi < inH!; hi++) for (let wi = 0; wi < inW!; wi++) {
        let s = 0
        for (let co = 0; co < cOut!; co++) {
          for (let kh = 0; kh < kH!; kh++) {
            const numH = hi + op.padH - kh
            if (numH < 0 || numH % op.strideH !== 0) continue
            const ho = numH / op.strideH
            if (ho >= hOut!) continue
            for (let kw = 0; kw < kW!; kw++) {
              const numW = wi + op.padW - kw
              if (numW < 0 || numW % op.strideW !== 0) continue
              const wo = numW / op.strideW
              if (wo >= wOut!) continue
              s += weight[co * cIn! * kH! * kW! + ci * kH! * kW! + kh * kW! + kw]!
                 * dy[b * cOut! * hOut! * wOut! + co * hOut! * wOut! + ho * wOut! + wo]!
            }
          }
        }
        out[b * cIn! * inH! * inW! + ci * inH! * inW! + hi * inW! + wi] = s
      }
      return out
    }
    case 'conv2d_weight_grad': {
      const input = vals.get(op.input)! as Float32Array
      const dy = vals.get(op.dy)! as Float32Array
      const inT = graph.tensors[op.input]!
      const dyT = graph.tensors[op.dy]!
      const outT = graph.tensors[op.out]!
      const [B, cIn, H, W] = inT.shape
      const [, cOut, hOut, wOut] = dyT.shape
      const [, , kH, kW] = outT.shape
      const out = new Float32Array(shapeSize(outT.shape))
      for (let co = 0; co < cOut!; co++) for (let ci = 0; ci < cIn!; ci++) for (let kh = 0; kh < kH!; kh++) for (let kw = 0; kw < kW!; kw++) {
        let s = 0
        for (let b = 0; b < B!; b++) {
          for (let ho = 0; ho < hOut!; ho++) {
            const hi = ho * op.strideH + kh - op.padH
            if (hi < 0 || hi >= H!) continue
            for (let wo = 0; wo < wOut!; wo++) {
              const wi = wo * op.strideW + kw - op.padW
              if (wi < 0 || wi >= W!) continue
              s += input[b * cIn! * H! * W! + ci * H! * W! + hi * W! + wi]!
                 * dy[b * cOut! * hOut! * wOut! + co * hOut! * wOut! + ho * wOut! + wo]!
            }
          }
        }
        out[co * cIn! * kH! * kW! + ci * kH! * kW! + kh * kW! + kw] = s
      }
      return out
    }
    case 'max_pool_2d': {
      const input = vals.get(op.input)! as Float32Array
      const inT = graph.tensors[op.input]!
      const outT = graph.tensors[op.out]!
      const [B, C, H, W] = inT.shape
      const [, , hOut, wOut] = outT.shape
      const out = new Float32Array(shapeSize(outT.shape))
      for (let b = 0; b < B!; b++) for (let c = 0; c < C!; c++) for (let ho = 0; ho < hOut!; ho++) for (let wo = 0; wo < wOut!; wo++) {
        let m = -3.4e38
        for (let kh = 0; kh < op.kH; kh++) {
          const hi = ho * op.strideH + kh - op.padH
          if (hi < 0 || hi >= H!) continue
          for (let kw = 0; kw < op.kW; kw++) {
            const wi = wo * op.strideW + kw - op.padW
            if (wi < 0 || wi >= W!) continue
            const v = input[b * C! * H! * W! + c * H! * W! + hi * W! + wi]!
            if (v > m) m = v
          }
        }
        out[b * C! * hOut! * wOut! + c * hOut! * wOut! + ho * wOut! + wo] = m
      }
      return out
    }
    case 'max_pool_2d_grad': {
      const input = vals.get(op.input)! as Float32Array
      const dy = vals.get(op.dy)! as Float32Array
      const inT = graph.tensors[op.input]!
      const dyT = graph.tensors[op.dy]!
      const [B, C, H, W] = inT.shape
      const [, , hOut, wOut] = dyT.shape
      const out = new Float32Array(shapeSize(inT.shape))
      for (let b = 0; b < B!; b++) for (let c = 0; c < C!; c++) for (let hi = 0; hi < H!; hi++) for (let wi = 0; wi < W!; wi++) {
        let s = 0
        for (let kh = 0; kh < op.kH; kh++) {
          const numH = hi + op.padH - kh
          if (numH < 0 || numH % op.strideH !== 0) continue
          const ho = numH / op.strideH
          if (ho >= hOut!) continue
          for (let kw = 0; kw < op.kW; kw++) {
            const numW = wi + op.padW - kw
            if (numW < 0 || numW % op.strideW !== 0) continue
            const wo = numW / op.strideW
            if (wo >= wOut!) continue
            // Recompute argmax for (b, c, ho, wo).
            let m = -3.4e38, argH = -1, argW = -1
            for (let kkh = 0; kkh < op.kH; kkh++) {
              const hh = ho * op.strideH + kkh - op.padH
              if (hh < 0 || hh >= H!) continue
              for (let kkw = 0; kkw < op.kW; kkw++) {
                const ww = wo * op.strideW + kkw - op.padW
                if (ww < 0 || ww >= W!) continue
                const v = input[b * C! * H! * W! + c * H! * W! + hh * W! + ww]!
                if (v > m) { m = v; argH = hh; argW = ww }
              }
            }
            if (argH === hi && argW === wi) {
              s += dy[b * C! * hOut! * wOut! + c * hOut! * wOut! + ho * wOut! + wo]!
            }
          }
        }
        out[b * C! * H! * W! + c * H! * W! + hi * W! + wi] = s
      }
      return out
    }

    default: {
      const _exhaustive: never = op
      void _exhaustive
      throw new Error(`eval: unhandled op kind ${(op as OpNode).kind}`)
    }
  }
}
