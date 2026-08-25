// WGSL codegen: one kernel per IR op.
//
// All shapes are baked into the WGSL as compile-time constants — no shape
// uniforms. This means each shape combination produces a distinct shader
// (so `add([B, T, D], [D])` and `add([B, T, D], [B, T, D])` get different
// kernels), which is fine for our static-shape model and gives the WGSL
// compiler full freedom to specialize.

import type { Graph, OpNode, Tensor, Shape } from './ir.js'
import type { BufferPlan } from './buffers.js'
import { shapeSize } from './shape.js'

// 256 lets our biggest kernel (~8M threads in matmul_bwd_dW) fit in ~32K
// workgroups, well under WebGPU's 65535-per-dim cap. Smaller sizes forced
// 2D dispatch with significant over-dispatch.
const WG_SIZE = 256

/** WebGPU's per-dimension cap on `dispatchWorkgroups`, and the guaranteed minimum for
 *  `maxComputeWorkgroupsPerDimension`. Exported so runtime.ts folds on the same number the
 *  kernels were emitted against. */
export const MAX_DISPATCH = 65535

/** Stride between rows of a 1-D grid folded across x and y, in whatever the builtin counts:
 *  threads for `gid`, workgroups for `wid`. Interpolated as a literal so the WGSL compiler
 *  sees a constant. */
const FOLD_STRIDE = { gid: MAX_DISPATCH * WG_SIZE, wid: MAX_DISPATCH } as const

/** Global thread index packed across the 2-D dispatch grid (see runtime.ts). */
const GID_LINE = `let i = gid.x + gid.y * ${FOLD_STRIDE.gid}u;`

/** One emitted compute kernel. The runtime turns each `KernelSpec` with
 *  non-empty `wgsl` into a `GPUComputePipeline` + bind group; logical ops
 *  (param/tensor/state inputs, reshape no-ops) carry empty `wgsl` and
 *  produce no dispatch. Order matches `graph.ops` — `emitKernels` returns
 *  them in dispatch order. */
export interface KernelSpec {
  /** Index into graph.ops. */
  opIndex: number
  /** Op kind (for debugging / pipeline cache key). */
  opKind: OpNode['kind']
  /** Generated WGSL source. Empty string for "logical" ops with no kernel. */
  wgsl: string
  /**
   * Buffer ids in binding-index order. The runtime creates a bind group with
   * these in @binding(0..N) on @group(0). Inputs come first (read), output last
   * (read_write).
   */
  bindings: number[]
  /** Threads to dispatch — one grid slice's worth when `dispatchY` or `dispatchZ` is set,
   *  the whole grid otherwise. 0 means "skip" (e.g. reshape no-op). */
  threads: number
  /** Extent of the z dispatch axis, when an axis rides z rather than being packed into the
   *  thread index. Absent means a plain 1-D grid folded across x and y. */
  dispatchZ?: number
  /** Extent of the y dispatch axis, when a second axis rides y. That leaves no room for the
   *  fold, so x alone carries `threads`. */
  dispatchY?: number
  /** Workgroup size; usually WG_SIZE. */
  workgroupSize: number
}

// ---- Index prologues -------------------------------------------------------
//
// Qualcomm's Adreno miscompiles the arithmetic that recovers per-axis indices from a packed
// thread id: wrong answers, every operand exact, no error anywhere. The two helpers below
// emit the forms that survive it, and each returns its WGSL, the expression for the output
// index, and the dispatch shape together — those are three halves of one decision, and a
// caller recomputing any of them is how they drift apart.
//
// Both forms were chosen by measurement, not by reasoning about the source, because
// reasoning about the source does not work here: deleting a `+ gid.y * 0u` term that
// computes nothing flips correctness in BOTH directions depending on the shape. Sweeps in
// typebulbs-tentative/kernel-fixes/packed-index-lab.bulb.md; account in
// specs/Adreno-matmul.md.
//
// The rule they leave behind: never recover an axis index by dividing a packed thread id
// when a dispatch axis can carry it instead.

/** Prologue for a one-thread-per-output kernel over a 4-D output whose body loops — conv2d,
 *  its input gradient, and both pooling kernels.
 *
 *  Four output axes and three dispatch axes, so one division is unavoidable. The outer two
 *  axes ride real dispatch axes and only the innermost pair is packed, which leaves that
 *  division by the last extent (an image width) rather than by a product. Correct at 135 of
 *  135 swept shapes, where the fully packed form was wrong at 20 of 93 and the batch-on-z
 *  form below at 67 of 95.
 *
 *  The trade is occupancy: one workgroup covers at most 256 of `d2*d3`, so an output with a
 *  small spatial extent leaves lanes idle. kata-go's global max pool, 1x1 out of
 *  [8,96,1,1], dispatches 768 workgroups where the packed form needed 3 — a 256x
 *  over-dispatch, free on desktop and 3-8x on Adreno, unmeasured. If it ever bites the fix
 *  is a 2-D workgroup shaped to `d2*d3`, not a fallback to a form known to be wrong. */
function gridIndex4d(
  shape: readonly [number, number, number, number],
  names: readonly [string, string, string, string],
) {
  const [d0, d1, d2, d3] = shape
  const [n0, n1, n2, n3] = names
  const perSlice = d2 * d3
  if (d0 <= MAX_DISPATCH && d1 <= MAX_DISPATCH && Math.ceil(perSlice / WG_SIZE) <= MAX_DISPATCH) {
    return {
      wgsl: `  let _p = gid.x;\n` +
        `  if (_p >= ${perSlice}u) { return; }\n` +
        `  let ${n0} = gid.z;\n` +
        `  let ${n1} = gid.y;\n` +
        `  let ${n2} = _p / ${d3}u;\n` +
        `  let ${n3} = _p % ${d3}u;`,
      outIdx: `(${n0} * ${d1}u + ${n1}) * ${perSlice}u + _p`,
      spec: { threads: perSlice, dispatchY: d1, dispatchZ: d0 },
    }
  }
  // Past a dispatch cap the axes cannot ride the grid, so the packed form is the only option
  // left — a fallback rather than a choice. Reaching it needs more than 65535 batch items or
  // channels.
  const total = d0 * d1 * perSlice
  return {
    wgsl: `  ${GID_LINE}\n  if (i >= ${total}u) { return; }\n${decompose4d(shape, names)}`,
    outIdx: 'i',
    spec: { threads: total },
  }
}

/** Prologue for a matmul kernel whose grid carries a batch: the batch rides z and the rest
 *  of the index is folded across x and y as usual. Correct at all 68 swept shapes, where the
 *  packed form was wrong at 22.
 *
 *  `builtin` picks which id the kernel reads — `gid` for one-thread-per-output kernels, `wid`
 *  for one-workgroup-per-tile ones — which also sets the fold stride and whether `threads`
 *  counts threads or workgroups' worth of them. */
function batchIndex(name: string, perSlice: number, batch: number, builtin: 'gid' | 'wid') {
  const stride = FOLD_STRIDE[builtin]
  const scale = builtin === 'wid' ? WG_SIZE : 1
  if (batch <= MAX_DISPATCH) {
    return {
      wgsl: `  let ${name} = ${builtin}.x + ${builtin}.y * ${stride}u;\n` +
        `  if (${name} >= ${perSlice}u) { return; }\n` +
        `  let bi = ${builtin}.z;`,
      outIdx: `bi * ${perSlice}u + ${name}`,
      spec: { threads: perSlice * scale, dispatchZ: batch },
    }
  }
  // Past the cap the batch cannot ride z and goes back into the packed index — the form
  // Adreno miscompiles. It needs `perSlice` at 512 or 1024 to bite, and 65535 batch items at
  // that width is a 33M-element output, so nothing tensorgrad runs reaches both at once.
  const total = batch * perSlice
  return {
    wgsl: `  let _i = ${builtin}.x + ${builtin}.y * ${stride}u;\n` +
      `  if (_i >= ${total}u) { return; }\n` +
      `  let bi = _i / ${perSlice}u;\n` +
      `  let ${name} = _i % ${perSlice}u;`,
    outIdx: '_i',
    spec: { threads: total * scale },
  }
}

/** Generate a KernelSpec per compute op in graph.ops (in dispatch order). */
export function emitKernels(graph: Graph, plan: BufferPlan): KernelSpec[] {
  const out: KernelSpec[] = []
  for (let i = 0; i < graph.ops.length; i++) {
    const op = graph.ops[i]!
    const spec = emitKernel(op, graph, plan, i)
    out.push(spec)
  }
  return out
}

function emitKernel(op: OpNode, graph: Graph, plan: BufferPlan, opIndex: number): KernelSpec {
  const tof = (id: number) => graph.tensors[id]!
  const buf = (tensorId: number) => plan.tensorToBuffer.get(tensorId)!
  const empty = (): KernelSpec => ({ opIndex, opKind: op.kind, wgsl: '', bindings: [], threads: 0, workgroupSize: WG_SIZE })

  switch (op.kind) {
    // ---- Leaves: data is supplied externally; no kernel ---------------------
    case 'param_input':
    case 'tensor_input':
    case 'state_input':
      return empty()

    // ---- arange / const_scalar: kernel that fills the buffer once -----------
    case 'arange': {
      const out = tof(op.out)
      const wgsl = `
@group(0) @binding(0) var<storage, read_write> buf : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${op.n}u) { return; }
  buf[i] = ${castFromI32('i32(i)', out.dtype)};
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.out)], threads: op.n, workgroupSize: WG_SIZE }
    }
    case 'const_scalar': {
      const wgsl = `
@group(0) @binding(0) var<storage, read_write> buf : array<${wgslDtype(op.dtype)}>;
@compute @workgroup_size(1)
fn main() {
  buf[0] = ${wgslLiteral(op.value, op.dtype)};
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.out)], threads: 1, workgroupSize: 1 }
    }
    case 'const_fill': {
      const out = tof(op.out)
      const total = shapeSize(out.shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read_write> buf : array<${wgslDtype(op.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  buf[i] = ${wgslLiteral(op.value, op.dtype)};
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- Element-wise binops with broadcast --------------------------------
    case 'add':
    case 'sub':
    case 'mul':
    case 'div':
    case 'min':
    case 'max': {
      const out = tof(op.out)
      const a = tof(op.a)
      const b = tof(op.b)
      const total = shapeSize(out.shape)
      // Infix for arithmetic; WGSL builtin for min/max.
      const expr =
        op.kind === 'min' ? 'min(a[aIdx], b[bIdx])' :
        op.kind === 'max' ? 'max(a[aIdx], b[bIdx])' :
        `a[aIdx] ${({ add: '+', sub: '-', mul: '*', div: '/' } as const)[op.kind]} b[bIdx]`
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(1) var<storage, read> b : array<${wgslDtype(b.dtype)}>;
@group(0) @binding(2) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${broadcastIndexBlock('i', out.shape, a.shape, 'aIdx')}
${broadcastIndexBlock('i', out.shape, b.shape, 'bIdx')}
  out[i] = ${expr};
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.b), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- Element-wise scalar binops (scalar baked into WGSL) ---------------
    case 'mul_scalar':
    case 'add_scalar': {
      const out = tof(op.out)
      const a = tof(op.a)
      const opStr = op.kind === 'mul_scalar' ? '*' : '+'
      const total = shapeSize(out.shape)
      const lit = wgslLiteral(op.scalar, out.dtype)
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(1) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  out[i] = a[i] ${opStr} ${lit};
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- Unary -------------------------------------------------------------
    case 'sqrt':
    case 'rsqrt':
    case 'log':
    case 'exp':
    case 'relu':
    case 'neg':
    case 'abs':
    case 'tanh':
    case 'sigmoid':
    case 'erf':
    case 'sin':
    case 'cos': {
      const out = tof(op.out)
      const a = tof(op.a)
      const total = shapeSize(out.shape)
      const expr =
        op.kind === 'sqrt'    ? 'sqrt(x)' :
        op.kind === 'rsqrt'   ? '1.0 / sqrt(x)' :
        op.kind === 'log'     ? 'log(x)' :
        op.kind === 'exp'     ? 'exp(x)' :
        op.kind === 'relu'    ? 'max(x, 0.0)' :
        op.kind === 'neg'     ? '-x' :
        op.kind === 'abs'     ? 'abs(x)' :
        op.kind === 'tanh'    ? 'tanh(x)' :
        op.kind === 'sin'     ? 'sin(x)' :
        op.kind === 'cos'     ? 'cos(x)' :
        op.kind === 'erf'     ? 'erf_approx(x)' :
        // tanh identity for numerical stability: sigmoid(x) = 0.5 + 0.5 * tanh(0.5x)
        /* sigmoid */           '0.5 + 0.5 * tanh(0.5 * x)'
      // WGSL has no erf intrinsic — emit the A&S 7.1.26 rational-poly approx
      // (max abs error ~1.5e-7) as a helper. Empty for every other unary.
      const preamble = op.kind === 'erf' ? `
fn erf_approx(x : f32) -> f32 {
  let s = sign(x);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return s * (1.0 - p * exp(-ax * ax));
}
` : ''
      const wgsl = `${preamble}
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(1) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let x = a[i];
  out[i] = ${expr};
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- Stochastic ------------------------------------------------------
    case 'randn': {
      const total = shapeSize(op.shape)
      // Per-call salt mixes into the PCG hash so independent randn / dropout
      // sites get independent streams. Two PCG draws per thread → Box-Muller.
      const saltConst = ((op.salt * 0x9E3779B1) >>> 0).toString(10) + 'u'
      const wgsl = `
@group(0) @binding(0) var<storage, read> seed : array<i32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  // First PCG draw seeded from (seed, salt, thread).
  var h1 : u32 = u32(seed[0]) ^ ${saltConst} ^ i;
  h1 = h1 * 747796405u + 2891336453u;
  h1 = ((h1 >> ((h1 >> 28u) + 4u)) ^ h1) * 277803737u;
  h1 = (h1 >> 22u) ^ h1;
  // Second PCG draw chained off the first.
  var h2 : u32 = h1 * 747796405u + 2891336453u;
  h2 = ((h2 >> ((h2 >> 28u) + 4u)) ^ h2) * 277803737u;
  h2 = (h2 >> 22u) ^ h2;
  let u1 : f32 = max(1.0e-10, f32(h1) / 4294967296.0);
  let u2 : f32 = f32(h2) / 4294967296.0;
  // Box-Muller; the sin pair is discarded — we want one N(0,1) per thread.
  out[i] = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.seed), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    case 'dropout': {
      const out = tof(op.out)
      const a = tof(op.a)
      const total = shapeSize(out.shape)
      const p = op.p
      const scale = 1 / (1 - p)
      // Per-call salt mixes into the PCG hash so independent dropout sites
      // get independent masks. Forward and backward share salt → same mask.
      const saltConst = ((op.salt * 0x9E3779B1) >>> 0).toString(10) + 'u'
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> seed : array<i32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  // PCG-style hash of (seed, salt, thread) — cheap, decorrelated enough.
  var h : u32 = u32(seed[0]) ^ ${saltConst} ^ i;
  h = h * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  h = (h >> 22u) ^ h;
  let u : f32 = f32(h) / 4294967296.0;
  let mask : f32 = select(0.0, ${scale.toFixed(8)}, u >= ${p.toFixed(8)});
  out[i] = a[i] * mask;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.seed), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- Comparisons + select --------------------------------------------
    case 'less':
    case 'greater': {
      const out = tof(op.out)
      const a = tof(op.a)
      const b = tof(op.b)
      const opStr = op.kind === 'less' ? '<' : '>'
      const total = shapeSize(out.shape)
      // bool tensors lower to u32 in storage (1 if true, 0 if false).
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(1) var<storage, read> b : array<${wgslDtype(b.dtype)}>;
@group(0) @binding(2) var<storage, read_write> out : array<u32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${broadcastIndexBlock('i', out.shape, a.shape, 'aIdx')}
${broadcastIndexBlock('i', out.shape, b.shape, 'bIdx')}
  out[i] = select(0u, 1u, a[aIdx] ${opStr} b[bIdx]);
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.b), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }
    case 'where': {
      const out = tof(op.out)
      const cond = tof(op.cond)
      const a = tof(op.a)
      const b = tof(op.b)
      const total = shapeSize(out.shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read> cond : array<u32>;
@group(0) @binding(1) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(2) var<storage, read> b : array<${wgslDtype(b.dtype)}>;
@group(0) @binding(3) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${broadcastIndexBlock('i', out.shape, cond.shape, 'cIdx')}
${broadcastIndexBlock('i', out.shape, a.shape, 'aIdx')}
${broadcastIndexBlock('i', out.shape, b.shape, 'bIdx')}
  out[i] = select(b[bIdx], a[aIdx], cond[cIdx] != 0u);
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.cond), buf(op.a), buf(op.b), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    case 'relu_grad': {
      const out = tof(op.out)
      const total = shapeSize(out.shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read> dy : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  out[i] = select(0.0, dy[i], x[i] > 0.0);
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.x), buf(op.dy), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- Reductions over last axis -----------------------------------------
    case 'mean_last':
    case 'sum_last': {
      const a = tof(op.a)
      const D = a.shape[a.shape.length - 1]!
      const outerSize = shapeSize(a.shape) / D
      const divisor = op.kind === 'mean_last' ? `f32(${D}u)` : '1.0'
      // One-thread-per-row starves the GPU when there are few rows and each
      // row is long — the worst case is the loss's full-tensor mean, which is
      // ONE thread serially summing the whole tensor. Same pathology measured
      // for conv2d_weight_grad (2026-07-10); below the thread threshold,
      // switch to one WORKGROUP per row: WG_SIZE threads stride the row, then
      // tree-reduce in shared memory. Naive form kept as reference and
      // many-rows path. (outerSize == 1 still leaves just one workgroup live;
      // if profiling ever shows that mattering, the fix is a two-stage split,
      // not a bigger workgroup.)
      if (outerSize < 32768 && D >= 256) {
        const wgsl = `
var<workgroup> partial : array<f32, ${WG_SIZE}>;
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lid : u32) {
  // One workgroup per output row. wid is workgroup-uniform, so the
  // early-out keeps every barrier below in uniform control flow.
  let i = wid.x + wid.y * 65535u;
  if (i >= ${outerSize}u) { return; }
  let base = i * ${D}u;
  var s : f32 = 0.0;
  for (var j : u32 = lid; j < ${D}u; j = j + ${WG_SIZE}u) {
    s = s + a[base + j];
  }
${emitWorkgroupReduce('partial', 's')}
  if (lid == 0u) { out[i] = partial[0] / ${divisor}; }
}`.trim()
        return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: outerSize * WG_SIZE, workgroupSize: WG_SIZE }
      }
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${outerSize}u) { return; }
  let base = i * ${D}u;
  var s : f32 = 0.0;
  for (var j : u32 = 0u; j < ${D}u; j = j + 1u) {
    s = s + a[base + j];
  }
  out[i] = s / ${divisor};
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: outerSize, workgroupSize: WG_SIZE }
    }

    case 'argmax_last': {
      const a = tof(op.a)
      const D = a.shape[a.shape.length - 1]!
      const outerSize = shapeSize(a.shape) / D
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<i32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${outerSize}u) { return; }
  let base = i * ${D}u;
  var bestVal : f32 = a[base];
  var bestIdx : i32 = 0;
  for (var j : u32 = 1u; j < ${D}u; j = j + 1u) {
    let v = a[base + j];
    if (v > bestVal) { bestVal = v; bestIdx = i32(j); }
  }
  out[i] = bestIdx;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: outerSize, workgroupSize: WG_SIZE }
    }

    case 'pack_rgba8': {
      // pack4x8unorm saturates, scales to a byte and rounds each component,
      // component 0 in the low byte. Stored through a bitcast because the
      // library has no u32 dtype; the host views the readback as bytes.
      const outerSize = shapeSize(tof(op.out).shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<i32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${outerSize}u) { return; }
  let base = i * 4u;
  out[i] = bitcast<i32>(pack4x8unorm(vec4<f32>(a[base], a[base + 1u], a[base + 2u], a[base + 3u])));
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: outerSize, workgroupSize: WG_SIZE }
    }

    case 'categorical_last': {
      // Gumbel-max sampling: sample ~ argmax_j (logit_j + g_j) where
      // g_j = -log(-log(u_j)) and u_j ~ Uniform(0,1). Mathematically
      // equivalent to sampling from softmax(logits) but skips the
      // normalization pass — one fused kernel, no separate softmax.
      const a = tof(op.a)
      const D = a.shape[a.shape.length - 1]!
      const outerSize = shapeSize(a.shape) / D
      const saltConst = ((op.salt * 0x9E3779B1) >>> 0).toString(10) + 'u'
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> seed : array<i32>;
@group(0) @binding(2) var<storage, read_write> out : array<i32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${outerSize}u) { return; }
  let base = i * ${D}u;
  var bestVal : f32 = -1.0e30;
  var bestIdx : i32 = 0;
  for (var j : u32 = 0u; j < ${D}u; j = j + 1u) {
    // PCG hash of (seed, salt, row*D + j) → uniform u in (0,1]. The
    // max() clamp keeps log(u) finite; small u yields large positive
    // Gumbel noise, which is the desired tail behavior.
    var h : u32 = u32(seed[0]) ^ ${saltConst} ^ (i * ${D}u + j);
    h = h * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    let u : f32 = max(1.0e-10, f32(h) / 4294967296.0);
    let g : f32 = -log(-log(u));
    let v : f32 = a[base + j] + g;
    if (v > bestVal) { bestVal = v; bestIdx = i32(j); }
  }
  out[i] = bestIdx;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.seed), buf(op.out)], threads: outerSize, workgroupSize: WG_SIZE }
    }

    // ---- Shape / detach ----------------------------------------------------
    // Both ops are byte-identical memcpy; reshape relabels the shape, while
    // stop_gradient detaches from the autograd graph. When the buffer plan has
    // given input and output ONE buffer (see `aliasedTo` in buffers.ts) the
    // copy would be from a buffer to itself, so there is nothing to dispatch —
    // the bytes are already where the consumer will look for them. The plan
    // only does that when both ends are interior intermediates, so the
    // remaining copies here are the ones that cross a param/output boundary.
    case 'reshape':
    case 'stop_gradient': {
      if (buf(op.a) === buf(op.out)) return empty()
      const out = tof(op.out)
      const a = tof(op.a)
      const total = shapeSize(out.shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(1) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  out[i] = a[i];
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    case 'permute': {
      const out = tof(op.out)
      const a = tof(op.a)
      const total = shapeSize(out.shape)
      // For each output flat index, decompose into per-axis indices then
      // recombine via input strides: srcIdx = Σ outIdx[perm⁻¹(k)] * aStride[k].
      const aStrides = computeStrides(a.shape)
      const outDimDecls = decomposeFlatIndexBlock('i', out.shape, 'oIdx')
      const srcExpr: string[] = []
      for (let k = 0; k < a.shape.length; k++) {
        const srcAxis = op.perm.indexOf(k)  // which output axis came from input axis k
        srcExpr.push(`oIdx_${srcAxis} * ${aStrides[k]}u`)
      }
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(1) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${outDimDecls}
  let srcIdx = ${srcExpr.join(' + ')};
  out[i] = a[srcIdx];
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- Linear algebra ----------------------------------------------------
    case 'matmul': {
      const out = tof(op.out)
      const a = tof(op.a)
      const b = tof(op.b)
      const M = a.shape[a.shape.length - 2]!
      const K = a.shape[a.shape.length - 1]!
      const N = b.shape[1]!
      const batch = shapeSize(a.shape) / (M * K)
      const total = batch * M * N
      // Small output × long K is the dB-of-Linear case from grad.ts: the
      // batch-flattened [K_w, rows] @ [rows, N] contraction has a few
      // thousand outputs each reducing over every row the layer saw — at
      // render-res SIREN shapes, 4k threads × 262k serial iterations. Same
      // few-threads/long-loop pathology as conv2d_weight_grad; same fix:
      // one WORKGROUP per output element, WG_SIZE threads striding K, tree
      // reduce. Only for shapes the tiled GEMM below can't take (non-tile-
      // aligned): where both apply, tiled measured faster even at 16
      // workgroups (SIREN dW 193 → 117 ms/step; xf-small dW likewise) —
      // 16× less traffic beats occupancy. Naive form kept as reference.
      const tileable = K % TILE === 0 && M >= TILE && N >= TILE
      if (total < 32768 && K >= 256 && !tileable) {
        const mnIdx = batchIndex('mn', M * N, batch, 'wid')
        const wgsl = `
var<workgroup> partial : array<f32, ${WG_SIZE}>;
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lid : u32) {
  // One workgroup per output element. wid is workgroup-uniform, so the
  // early-out keeps every barrier below in uniform control flow.
${mnIdx.wgsl}
  let m = mn / ${N}u;
  let n = mn % ${N}u;
  let aBase = bi * ${M * K}u + m * ${K}u;
  var s : f32 = 0.0;
  for (var k : u32 = lid; k < ${K}u; k = k + ${WG_SIZE}u) {
    s = s + a[aBase + k] * b[k * ${N}u + n];
  }
${emitWorkgroupReduce('partial', 's')}
  if (lid == 0u) { c[${mnIdx.outIdx}] = partial[0]; }
}`.trim()
        return {
          opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.b), buf(op.out)],
          ...mnIdx.spec, workgroupSize: WG_SIZE,
        }
      }
      if (tileable) {
        const tiled = emitTiledMatmul(batch, M, K, N, false)
        return {
          opIndex, opKind: op.kind, wgsl: tiled.wgsl, bindings: [buf(op.a), buf(op.b), buf(op.out)],
          ...tiled.spec, workgroupSize: WG_SIZE,
        }
      }
      const mnIdx = batchIndex('mn', M * N, batch, 'gid')
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${mnIdx.wgsl}
  let m = mn / ${N}u;
  let n = mn % ${N}u;
  let aBase = bi * ${M * K}u + m * ${K}u;
  var s : f32 = 0.0;
  for (var k : u32 = 0u; k < ${K}u; k = k + 1u) {
    s = s + a[aBase + k] * b[k * ${N}u + n];
  }
  c[${mnIdx.outIdx}] = s;
}`.trim()
      return {
        opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.b), buf(op.out)],
        ...mnIdx.spec, workgroupSize: WG_SIZE,
      }
    }

    case 'matmul_batched': {
      const out = tof(op.out)
      const a = tof(op.a)
      const b = tof(op.b)
      const M = a.shape[a.shape.length - 2]!
      const K = a.shape[a.shape.length - 1]!
      const N = b.shape[b.shape.length - 1]!
      const batch = shapeSize(a.shape) / (M * K)
      if (K % TILE === 0 && M >= TILE && N >= TILE) {
        const tiled = emitTiledMatmul(batch, M, K, N, true)
        return {
          opIndex, opKind: op.kind, wgsl: tiled.wgsl, bindings: [buf(op.a), buf(op.b), buf(op.out)],
          ...tiled.spec, workgroupSize: WG_SIZE,
        }
      }
      const mnIdx = batchIndex('mn', M * N, batch, 'gid')
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${mnIdx.wgsl}
  let m = mn / ${N}u;
  let n = mn % ${N}u;
  let aBase = bi * ${M * K}u + m * ${K}u;
  let bBase = bi * ${K * N}u;
  var s : f32 = 0.0;
  for (var k : u32 = 0u; k < ${K}u; k = k + 1u) {
    s = s + a[aBase + k] * b[bBase + k * ${N}u + n];
  }
  c[${mnIdx.outIdx}] = s;
}`.trim()
      return {
        opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.b), buf(op.out)],
        ...mnIdx.spec, workgroupSize: WG_SIZE,
      }
    }

    // ---- One-hot ------------------------------------------------------------
    case 'one_hot': {
      const out = tof(op.out)
      const indices = tof(op.indices)
      const total = shapeSize(out.shape)
      const depth = op.depth
      const zeroLit = wgslLiteral(0, out.dtype)
      const oneLit = wgslLiteral(1, out.dtype)
      const wgsl = `
@group(0) @binding(0) var<storage, read> indices : array<i32>;
@group(0) @binding(1) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let outerIdx = i / ${depth}u;
  let depthIdx = i % ${depth}u;
  let tgt = u32(indices[outerIdx]);
  out[i] = select(${zeroLit}, ${oneLit}, tgt == depthIdx);
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.indices), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- ML primitives -----------------------------------------------------
    case 'log_softmax_last': {
      const a = tof(op.a)
      const D = a.shape[a.shape.length - 1]!
      const outerSize = shapeSize(a.shape) / D
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${outerSize}u) { return; }
  let base = i * ${D}u;
  var m : f32 = -1.0e30;
  for (var j : u32 = 0u; j < ${D}u; j = j + 1u) {
    let v = a[base + j];
    if (v > m) { m = v; }
  }
  var s : f32 = 0.0;
  for (var j : u32 = 0u; j < ${D}u; j = j + 1u) {
    s = s + exp(a[base + j] - m);
  }
  let logZ = m + log(s);
  for (var j : u32 = 0u; j < ${D}u; j = j + 1u) {
    out[base + j] = a[base + j] - logZ;
  }
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: outerSize, workgroupSize: WG_SIZE }
    }

    case 'softmax_causal_last': {
      const a = tof(op.a)
      const T = a.shape[a.shape.length - 1]!  // last 2 axes are square TxT
      const outerSize = shapeSize(a.shape) / T
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  // Each thread handles one (..., qpos)-row, softmaxing over kpos∈[0..qpos].
  ${GID_LINE}
  if (i >= ${outerSize}u) { return; }
  let qpos = i % ${T}u;
  let base = i * ${T}u;
  var m : f32 = -1.0e30;
  for (var k : u32 = 0u; k <= qpos; k = k + 1u) {
    let v = a[base + k];
    if (v > m) { m = v; }
  }
  var s : f32 = 0.0;
  for (var k : u32 = 0u; k <= qpos; k = k + 1u) {
    let e = exp(a[base + k] - m);
    out[base + k] = e;
    s = s + e;
  }
  for (var k : u32 = 0u; k <= qpos; k = k + 1u) {
    out[base + k] = out[base + k] / s;
  }
  for (var k : u32 = qpos + 1u; k < ${T}u; k = k + 1u) {
    out[base + k] = 0.0;
  }
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: outerSize, workgroupSize: WG_SIZE }
    }

    case 'where_causal': {
      const a = tof(op.a)
      const T = a.shape[a.shape.length - 1]!
      const total = shapeSize(a.shape)
      const fillLit = wgslLiteral(op.fillValue, 'f32')
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let kpos = i % ${T}u;
  let qpos = (i / ${T}u) % ${T}u;
  if (kpos > qpos) {
    out[i] = ${fillLit};
  } else {
    out[i] = a[i];
  }
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- Slicing -----------------------------------------------------------
    case 'slice_range': {
      // Decompose i into (outer, axisIdx, inner); shift axisIdx by `start`
      // and use the input's axis stride.
      const out = tof(op.out)
      const a = tof(op.a)
      const axis = op.axis
      const inner = a.shape.slice(axis + 1).reduce((p, d) => p * d, 1)
      const D_in = a.shape[axis]!
      const D_out = op.end - op.start
      const total = shapeSize(out.shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(1) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let outer = i / ${D_out * inner}u;
  let rest = i % ${D_out * inner}u;
  let axisIdx = rest / ${inner}u;
  let innerIdx = rest % ${inner}u;
  out[i] = a[outer * ${D_in * inner}u + (axisIdx + ${op.start}u) * ${inner}u + innerIdx];
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    case 'scatter_axis': {
      // Inverse of slice_range: copy `a` into [start, end) along `axis` of an
      // otherwise-zero output; one thread per output cell branches on whether
      // the cell sits inside the slice region.
      const out = tof(op.out)
      const a = tof(op.a)
      const axis = op.axis
      const inner = out.shape.slice(axis + 1).reduce((p, d) => p * d, 1)
      const D_out = out.shape[axis]!
      const D_in = op.end - op.start
      const total = shapeSize(out.shape)
      const zeroLit = wgslLiteral(0, out.dtype)
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(1) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let outer = i / ${D_out * inner}u;
  let rest = i % ${D_out * inner}u;
  let axisIdx = rest / ${inner}u;
  let innerIdx = rest % ${inner}u;
  if (axisIdx < ${op.start}u || axisIdx >= ${op.end}u) {
    out[i] = ${zeroLit};
  } else {
    out[i] = a[outer * ${D_in * inner}u + (axisIdx - ${op.start}u) * ${inner}u + innerIdx];
  }
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    case 'concat': {
      // Variadic. For each output element, walk the axis-offset chain to
      // find the source input. Inputs are bound 0..N-1; output is binding N.
      const out = tof(op.out)
      const axis = op.axis
      const inner = out.shape.slice(axis + 1).reduce((p, d) => p * d, 1)
      const D_out = out.shape[axis]!
      const total = shapeSize(out.shape)
      const inputDtypes = op.inputs.map(id => tof(id).dtype)
      const inputAxisSizes = op.inputs.map(id => tof(id).shape[axis]!)
      let cursor = 0
      const branches: string[] = []
      for (let k = 0; k < op.inputs.length; k++) {
        const sz = inputAxisSizes[k]!
        const lo = cursor
        const hi = cursor + sz
        const D_in = sz
        branches.push(
          `  ${k === 0 ? '' : 'else '}if (axisIdx < ${hi}u) {\n` +
          `    out[i] = src${k}[outer * ${D_in * inner}u + (axisIdx - ${lo}u) * ${inner}u + innerIdx];\n` +
          `    return;\n` +
          `  }`,
        )
        cursor += sz
      }
      const bindingDecls = op.inputs.map((_, k) =>
        `@group(0) @binding(${k}) var<storage, read> src${k} : array<${wgslDtype(inputDtypes[k]!)}>;`,
      ).join('\n')
      const wgsl = `
${bindingDecls}
@group(0) @binding(${op.inputs.length}) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let outer = i / ${D_out * inner}u;
  let rest = i % ${D_out * inner}u;
  let axisIdx = rest / ${inner}u;
  let innerIdx = rest % ${inner}u;
${branches.join('\n')}
}`.trim()
      return {
        opIndex, opKind: op.kind, wgsl,
        bindings: [...op.inputs.map(id => buf(id)), buf(op.out)],
        threads: total, workgroupSize: WG_SIZE,
      }
    }

    // ---- Broadcast / un-broadcast (autograd infrastructure) ----------------
    case 'broadcast_to': {
      const out = tof(op.out)
      const a = tof(op.a)
      const total = shapeSize(out.shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(1) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${broadcastIndexBlock('i', out.shape, a.shape, 'srcIdx')}
  out[i] = a[srcIdx];
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- Adam (fused per-element) -----------------------------------------
    // m_new = beta1 * m + (1 - beta1) * g
    case 'adam_update_m': {
      const out = tof(op.out)
      const total = shapeSize(out.shape)
      const beta1 = op.beta1
      const oneMinusBeta1 = 1 - beta1
      const wgsl = `
@group(0) @binding(0) var<storage, read> m : array<f32>;
@group(0) @binding(1) var<storage, read> g : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  out[i] = ${wgslLiteral(beta1, 'f32')} * m[i] + ${wgslLiteral(oneMinusBeta1, 'f32')} * g[i];
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.m), buf(op.g), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }
    // v_new = beta2 * v + (1 - beta2) * g²
    case 'adam_update_v': {
      const out = tof(op.out)
      const total = shapeSize(out.shape)
      const beta2 = op.beta2
      const oneMinusBeta2 = 1 - beta2
      const wgsl = `
@group(0) @binding(0) var<storage, read> v : array<f32>;
@group(0) @binding(1) var<storage, read> g : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let gv = g[i];
  out[i] = ${wgslLiteral(beta2, 'f32')} * v[i] + ${wgslLiteral(oneMinusBeta2, 'f32')} * gv * gv;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.v), buf(op.g), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }
    // p_new = decayShrink * p - lrt[0] * m_new / (sqrt(v_new) + eps).
    // decayShrink is baked as a literal under fixed lr, bound as a per-step
    // scalar input under a schedule. When literal=1 the multiply folds away.
    case 'adam_update_p': {
      const out = tof(op.out)
      const total = shapeSize(out.shape)
      const dynamicShrink = op.decayShrinkTensor !== null
      const shrinkExpr = dynamicShrink ? 'decayShrink[0]' : wgslLiteral(op.decayShrink, 'f32')
      const shrinkBinding = dynamicShrink
        ? `@group(0) @binding(4) var<storage, read> decayShrink : array<f32>;\n` +
          `@group(0) @binding(5) var<storage, read_write> out : array<f32>;`
        : `@group(0) @binding(4) var<storage, read_write> out : array<f32>;`
      const wgsl = `
@group(0) @binding(0) var<storage, read> p : array<f32>;
@group(0) @binding(1) var<storage, read> mNew : array<f32>;
@group(0) @binding(2) var<storage, read> vNew : array<f32>;
@group(0) @binding(3) var<storage, read> lrt : array<f32>;
${shrinkBinding}
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  out[i] = ${shrinkExpr} * p[i] - lrt[0] * mNew[i] / (sqrt(vNew[i]) + ${wgslLiteral(op.eps, 'f32')});
}`.trim()
      const bindings = dynamicShrink
        ? [buf(op.p), buf(op.mNew), buf(op.vNew), buf(op.lrt), buf(op.decayShrinkTensor!), buf(op.out)]
        : [buf(op.p), buf(op.mNew), buf(op.vNew), buf(op.lrt), buf(op.out)]
      return { opIndex, opKind: op.kind, wgsl, bindings, threads: total, workgroupSize: WG_SIZE }
    }

    case 'sum_to_shape': {
      // Sum over each axis where target is 1 or missing (prefix axes).
      const out = tof(op.out)
      const a = tof(op.a)
      const { wgsl, threads } = emitSumToShape(a.shape, out.shape, a.dtype)
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads, workgroupSize: WG_SIZE }
    }

    // ---- 2D conv + pool ----------------------------------------------------
    case 'conv2d': {
      // Grouped form: output channel cOut_'s group g reads only input channels
      // [g·cInPerG, (g+1)·cInPerG). groups=1 makes g 0 and cInPerG = C_in.
      const input = tof(op.input)
      const weight = tof(op.weight)
      const out = tof(op.out)
      const [, cIn, H, W] = input.shape
      const [cOut, cInPerG, kH, kW] = weight.shape
      const cOutPerG = cOut! / op.groups
      const tiledFwd = emitTiledConv('fwd', convDims(input.shape, weight.shape, out.shape, op))
      if (tiledFwd) {
        return {
          opIndex, opKind: op.kind, wgsl: tiledFwd.wgsl, bindings: [buf(op.input), buf(op.weight), buf(op.out)],
          ...tiledFwd.spec, workgroupSize: WG_SIZE,
        }
      }
      const idx = gridIndex4d(out.shape as [number, number, number, number], ['b', 'cOut_', 'h_out', 'w_out'])
      const wgsl = `
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${idx.wgsl}
  let g         = cOut_ / ${cOutPerG}u;
  let inBase    = b * ${cIn! * H! * W!}u + g * ${cInPerG! * H! * W!}u;
  let wBase     = cOut_ * ${cInPerG! * kH! * kW!}u;
  var s : f32 = 0.0;
  for (var c : u32 = 0u; c < ${cInPerG!}u; c = c + 1u) {
    let inChan = inBase + c * ${H! * W!}u;
    let wChan  = wBase  + c * ${kH! * kW!}u;
    for (var kh : u32 = 0u; kh < ${kH!}u; kh = kh + 1u) {
      let h_in = i32(h_out * ${op.strideH}u + kh) - ${op.padH};
      if (h_in < 0 || h_in >= ${H!}) { continue; }
      for (var kw : u32 = 0u; kw < ${kW!}u; kw = kw + 1u) {
        let w_in = i32(w_out * ${op.strideW}u + kw) - ${op.padW};
        if (w_in < 0 || w_in >= ${W!}) { continue; }
        s = s + input[inChan + u32(h_in) * ${W!}u + u32(w_in)]
              * weight[wChan + kh * ${kW!}u + kw];
      }
    }
  }
  out[${idx.outIdx}] = s;
}`.trim()
      return {
        opIndex, opKind: op.kind, wgsl, bindings: [buf(op.input), buf(op.weight), buf(op.out)],
        ...idx.spec, workgroupSize: WG_SIZE,
      }
    }

    case 'conv2d_input_grad': {
      // Invert the forward index relation:
      //   h_out = (h_in + padH - kh) / strideH (must divide evenly).
      // Grouped: input channel c_in_'s group g receives contributions only
      // from output channels [g·cOutPerG, (g+1)·cOutPerG).
      const weight = tof(op.weight)
      const dy = tof(op.dy)
      const out = tof(op.out)
      const [, , inH, inW] = out.shape
      const [cOut, cInPerG, kH, kW] = weight.shape
      const cOutPerG = cOut! / op.groups
      const [, , hOut, wOut] = dy.shape
      const tiledDx = emitTiledConv('dx', convDims(out.shape, weight.shape, dy.shape, op))
      if (tiledDx) {
        return {
          opIndex, opKind: op.kind, wgsl: tiledDx.wgsl, bindings: [buf(op.weight), buf(op.dy), buf(op.out)],
          ...tiledDx.spec, workgroupSize: WG_SIZE,
        }
      }
      const idx = gridIndex4d(out.shape as [number, number, number, number], ['b', 'c_in_', 'h_in', 'w_in'])
      const wgsl = `
@group(0) @binding(0) var<storage, read> weight : array<f32>;
@group(0) @binding(1) var<storage, read> dy : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${idx.wgsl}
  let g      = c_in_ / ${cInPerG!}u;
  let cLocal = c_in_ - g * ${cInPerG!}u;
  var s : f32 = 0.0;
  for (var co : u32 = 0u; co < ${cOutPerG}u; co = co + 1u) {
    let c_out  = g * ${cOutPerG}u + co;
    let wBase  = c_out * ${cInPerG! * kH! * kW!}u + cLocal * ${kH! * kW!}u;
    let dyBase = b * ${cOut! * hOut! * wOut!}u + c_out * ${hOut! * wOut!}u;
    for (var kh : u32 = 0u; kh < ${kH!}u; kh = kh + 1u) {
      let numH = i32(h_in) + ${op.padH} - i32(kh);
      if (numH < 0) { continue; }
      if ((numH % ${op.strideH}) != 0) { continue; }
      let h_out = numH / ${op.strideH};
      if (h_out >= ${hOut!}) { continue; }
      for (var kw : u32 = 0u; kw < ${kW!}u; kw = kw + 1u) {
        let numW = i32(w_in) + ${op.padW} - i32(kw);
        if (numW < 0) { continue; }
        if ((numW % ${op.strideW}) != 0) { continue; }
        let w_out = numW / ${op.strideW};
        if (w_out >= ${wOut!}) { continue; }
        s = s + weight[wBase + kh * ${kW!}u + kw]
              * dy[dyBase + u32(h_out) * ${wOut!}u + u32(w_out)];
      }
    }
  }
  out[${idx.outIdx}] = s;
}`.trim()
      return {
        opIndex, opKind: op.kind, wgsl, bindings: [buf(op.weight), buf(op.dy), buf(op.out)],
        ...idx.spec, workgroupSize: WG_SIZE,
      }
    }

    case 'conv2d_weight_grad': {
      // dWeight[c_out,c_in,kh,kw] = Σ_{b,h_out,w_out} input[b,c_in,h_in,w_in]
      // * dy[b,c_out,h_out,w_out], with h_in = h_out * strideH + kh - padH.
      // Grouped: c_in_ is group-local; the actual input channel is
      // g·cInPerG + c_in_ where g is c_out_'s group.
      const input = tof(op.input)
      const dy = tof(op.dy)
      const out = tof(op.out)
      const [cOut, cInPerG, kH, kW] = out.shape
      const cOutPerG = cOut! / op.groups
      const [B, cIn, H, W] = input.shape
      const [, , hOut, wOut] = dy.shape
      const total = shapeSize(out.shape)
      const redLen = B! * hOut! * wOut!
      const tiledDw = emitTiledConv('dw', convDims(input.shape, out.shape, dy.shape, op))
      if (tiledDw) {
        return {
          opIndex, opKind: op.kind, wgsl: tiledDw.wgsl, bindings: [buf(op.input), buf(op.dy), buf(op.out)],
          ...tiledDw.spec, workgroupSize: WG_SIZE,
        }
      }
      // One-thread-per-weight starves the GPU when the weight is small and
      // the reduction long: growing-nca's 1×1 convs give 8k/2k threads each
      // serially summing 16k terms — measured (2026-07-10, wgrad-probe bulb)
      // at ~63% of the whole training step, ~6× slower than the forward conv
      // doing identical MACs with ample threads. Below the thread threshold,
      // switch to one WORKGROUP per weight element: WG_SIZE threads split the
      // Σ over b·h_out·w_out, then tree-reduce in shared memory. The naive
      // form (below) remains the reference and the large-weight path.
      if (total < 32768 && redLen >= 256) {
        const wgsl = `
var<workgroup> partial : array<f32, ${WG_SIZE}>;
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> dy : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lid : u32) {
  // One workgroup per weight element. wid is workgroup-uniform, so the
  // early-out keeps every barrier below in uniform control flow.
  let i = wid.x + wid.y * 65535u;
  if (i >= ${total}u) { return; }
${decompose4d(out.shape as [number, number, number, number], ['c_out_', 'c_in_', 'kh', 'kw'])}
  let g = c_out_ / ${cOutPerG}u;
  let inChan = (g * ${cInPerG!}u + c_in_) * ${H! * W!}u;
  let dyChan = c_out_ * ${hOut! * wOut!}u;
  var s : f32 = 0.0;
  for (var r : u32 = lid; r < ${redLen}u; r = r + ${WG_SIZE}u) {
    let b = r / ${hOut! * wOut!}u;
    let hw = r % ${hOut! * wOut!}u;
    let h_out = hw / ${wOut!}u;
    let w_out = hw % ${wOut!}u;
    let h_in = i32(h_out * ${op.strideH}u + kh) - ${op.padH};
    let w_in = i32(w_out * ${op.strideW}u + kw) - ${op.padW};
    if (h_in >= 0 && h_in < ${H!} && w_in >= 0 && w_in < ${W!}) {
      s = s + input[b * ${cIn! * H! * W!}u + inChan + u32(h_in) * ${W!}u + u32(w_in)]
            * dy[b * ${cOut! * hOut! * wOut!}u + dyChan + h_out * ${wOut!}u + w_out];
    }
  }
${emitWorkgroupReduce('partial', 's')}
  if (lid == 0u) { out[i] = partial[0]; }
}`.trim()
        return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.input), buf(op.dy), buf(op.out)], threads: total * WG_SIZE, workgroupSize: WG_SIZE }
      }
      const wgsl = `
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> dy : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${decompose4d(out.shape as [number, number, number, number], ['c_out_', 'c_in_', 'kh', 'kw'])}
  let g = c_out_ / ${cOutPerG}u;
  var s : f32 = 0.0;
  for (var b : u32 = 0u; b < ${B!}u; b = b + 1u) {
    let inBase = b * ${cIn! * H! * W!}u + (g * ${cInPerG!}u + c_in_) * ${H! * W!}u;
    let dyBase = b * ${cOut! * hOut! * wOut!}u + c_out_ * ${hOut! * wOut!}u;
    for (var h_out : u32 = 0u; h_out < ${hOut!}u; h_out = h_out + 1u) {
      let h_in = i32(h_out * ${op.strideH}u + kh) - ${op.padH};
      if (h_in < 0 || h_in >= ${H!}) { continue; }
      for (var w_out : u32 = 0u; w_out < ${wOut!}u; w_out = w_out + 1u) {
        let w_in = i32(w_out * ${op.strideW}u + kw) - ${op.padW};
        if (w_in < 0 || w_in >= ${W!}) { continue; }
        s = s + input[inBase + u32(h_in) * ${W!}u + u32(w_in)]
              * dy[dyBase + h_out * ${wOut!}u + w_out];
      }
    }
  }
  out[i] = s;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.input), buf(op.dy), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    case 'max_pool_2d': {
      const input = tof(op.input)
      const out = tof(op.out)
      const [, C, H, W] = input.shape
      const poolIdx = gridIndex4d(out.shape as [number, number, number, number], ['b', 'c', 'h_out', 'w_out'])
      // Padding never wins; ties favor earliest in scan order (strictly-greater
      // comparison). Backward must replicate this exact scan to match.
      const NEG = '-3.4e38'
      const wgsl = `
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${poolIdx.wgsl}
  let inChan = b * ${C! * H! * W!}u + c * ${H! * W!}u;
  var m : f32 = ${NEG};
  for (var kh : u32 = 0u; kh < ${op.kH}u; kh = kh + 1u) {
    let h_in = i32(h_out * ${op.strideH}u + kh) - ${op.padH};
    if (h_in < 0 || h_in >= ${H!}) { continue; }
    for (var kw : u32 = 0u; kw < ${op.kW}u; kw = kw + 1u) {
      let w_in = i32(w_out * ${op.strideW}u + kw) - ${op.padW};
      if (w_in < 0 || w_in >= ${W!}) { continue; }
      let v = input[inChan + u32(h_in) * ${W!}u + u32(w_in)];
      if (v > m) { m = v; }
    }
  }
  out[${poolIdx.outIdx}] = m;
}`.trim()
      return {
        opIndex, opKind: op.kind, wgsl, bindings: [buf(op.input), buf(op.out)],
        ...poolIdx.spec, workgroupSize: WG_SIZE,
      }
    }

    case 'max_pool_2d_grad': {
      // Gather: for each input position, walk every output whose receptive
      // field covers it; recompute its argmax and accumulate dy when we won.
      const input = tof(op.input)
      const dy = tof(op.dy)
      const out = tof(op.out)
      const [, C, H, W] = input.shape
      const [, , hOut, wOut] = dy.shape
      const gradIdx = gridIndex4d(out.shape as [number, number, number, number], ['b', 'c', 'h_in', 'w_in'])
      const NEG = '-3.4e38'
      const wgsl = `
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> dy : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${gradIdx.wgsl}
  let inChan = b * ${C! * H! * W!}u + c * ${H! * W!}u;
  let dyChan = b * ${C! * hOut! * wOut!}u + c * ${hOut! * wOut!}u;
  var s : f32 = 0.0;
  for (var kh : u32 = 0u; kh < ${op.kH}u; kh = kh + 1u) {
    let numH = i32(h_in) + ${op.padH} - i32(kh);
    if (numH < 0) { continue; }
    if ((numH % ${op.strideH}) != 0) { continue; }
    let h_out = numH / ${op.strideH};
    if (h_out >= ${hOut!}) { continue; }
    for (var kw : u32 = 0u; kw < ${op.kW}u; kw = kw + 1u) {
      let numW = i32(w_in) + ${op.padW} - i32(kw);
      if (numW < 0) { continue; }
      if ((numW % ${op.strideW}) != 0) { continue; }
      let w_out = numW / ${op.strideW};
      if (w_out >= ${wOut!}) { continue; }
      var m : f32 = ${NEG};
      var argH : i32 = -1;
      var argW : i32 = -1;
      for (var kkh : u32 = 0u; kkh < ${op.kH}u; kkh = kkh + 1u) {
        let hh = i32(u32(h_out) * ${op.strideH}u + kkh) - ${op.padH};
        if (hh < 0 || hh >= ${H!}) { continue; }
        for (var kkw : u32 = 0u; kkw < ${op.kW}u; kkw = kkw + 1u) {
          let ww = i32(u32(w_out) * ${op.strideW}u + kkw) - ${op.padW};
          if (ww < 0 || ww >= ${W!}) { continue; }
          let v = input[inChan + u32(hh) * ${W!}u + u32(ww)];
          if (v > m) { m = v; argH = hh; argW = ww; }
        }
      }
      if (argH == i32(h_in) && argW == i32(w_in)) {
        s = s + dy[dyChan + u32(h_out) * ${wOut!}u + u32(w_out)];
      }
    }
  }
  out[${gradIdx.outIdx}] = s;
}`.trim()
      return {
        opIndex, opKind: op.kind, wgsl, bindings: [buf(op.input), buf(op.dy), buf(op.out)],
        ...gradIdx.spec, workgroupSize: WG_SIZE,
      }
    }
  }
}

// ---- WGSL helpers --------------------------------------------------------

/** Tile edge for the shared-memory GEMM; TILE² must equal WG_SIZE so one
 *  tile is exactly one workgroup. */
const TILE = 16

/** What emitTiledGemm needs from a caller beyond the template: the GEMM's extents, how its
 *  operands are addressed, and where its tile grid stops being worth dispatching. */
type TiledGemm = {
  M: number; N: number; K: number
  /** Dispatch z extent — the batch (× groups) the tile grid repeats over; `bi` in the kernel. */
  z: number
  /** Binding names of the A and B buffers, in binding order; the output is always `out`. */
  names: [string, string]
  /** WGSL run once `bi` is known (decoding it into batch and group); '' when the snippets
   *  use `bi` as it is. */
  prologue: string
  /** `a` assigns v = A[m,k] and `b` assigns v = B[k,n], each guarding its own row (m < M) or
   *  column (n < N); they are spliced where m, k, n and a zeroed v are in scope. `c` is the
   *  flat index of C[m,n]. */
  a: string; b: string; c: string
  /** Stage B with consecutive threads walking k instead of n — for a B contiguous along k. */
  bAlongK?: boolean
  /** A tile grid smaller than this returns null, so the caller's other paths take over. */
  minGrid?: number
}

/** The one tiled GEMM template behind matmul (both forms) and the conv family —
 *  Performance.md §Phase 1 (16×16 shared-memory tiles, re-implemented 2026-07-10 after the
 *  original perf-tiled-matmul branch was lost) and §Phase 7 (register blocking, M/N bounds,
 *  the conv operands gathered from NCHW; matmul folded onto the same template 2026-08-25 —
 *  its kernels gained the micro-tiles, transformer-heavy GPU 16.05 → 15.3 ms/step and the
 *  other demos within noise, conv kernels byte-identical before and after).
 *
 *  Each workgroup computes one TILE_M×TILE_N block of C, looping K in TILE chunks and
 *  cooperatively staging the A and B chunks in workgroup memory, so every global element is
 *  read once per tile instead of once per thread — the TILE× traffic cut behind Phase 1's
 *  MLP ~2.7× / small-transformer ~1.65×. Each thread owns a TM×TN micro-tile of C in
 *  registers, so one staged element feeds TM or TN FMAs instead of one: 4×4 when both
 *  extents allow, halved while the grid would leave most of the GPU idle (a handful of big
 *  tiles each striding a long K is the wrong trade). M and N are bounds-checked at the
 *  staging loads and the store, so the only gate is K % TILE == 0; every caller keeps its
 *  naive kernel as reference and odd-K fallback. Staging is arranged so consecutive threads
 *  touch consecutive addresses: A along k; B along n, or along k when the caller says B is
 *  contiguous that way (conv's weight-grad, where k is the pixel index), in which case the
 *  B tile is staged transposed.
 *
 *  The batch rides wid.z, never a packed-index division: Adreno miscompiles that construct
 *  (specs/Adreno-matmul.md), and this kernel measured correct there with the z form. wid is
 *  workgroup-uniform, so the tile-count early-out keeps every barrier below in uniform
 *  control flow. The WGSL and its dispatch spec are returned together because the guard and
 *  the grid are two halves of one decision — if they ever disagreed the kernel would drop
 *  work or write out of range, silently; test/dispatch.ts pins the pair. Known cost since
 *  Phase 1: small K/N GEMMs pay the barrier overhead (the MLP bench ~5.1 → ~6.5 ms/step at
 *  the per-step sync floor). The reduced long-K matmul variant yields to this one where both
 *  apply — measured 2026-07-10, even at 16 workgroups the traffic cut beats occupancy. */
function emitTiledGemm(g: TiledGemm): { wgsl: string; spec: { threads: number; dispatchZ?: number } } | null {
  const { M, N, K, z } = g
  if (K % TILE !== 0) return null
  let TM = M >= 64 ? 4 : M >= 32 ? 2 : 1
  let TN = N >= 64 ? 4 : N >= 32 ? 2 : 1
  const grid = () => Math.ceil(M / (TILE * TM)) * Math.ceil(N / (TILE * TN)) * z
  while (grid() < 64 && (TM > 1 || TN > 1)) { if (TM >= TN) TM >>= 1; else TN >>= 1 }
  if (g.minGrid && grid() < g.minGrid) return null
  const TILE_M = TILE * TM, TILE_N = TILE * TN
  const tilesM = Math.ceil(M / TILE_M), tilesN = Math.ceil(N / TILE_N)
  const idx = batchIndex('t', tilesM * tilesN, z, 'wid')
  const range = (n: number) => Array.from({ length: n }, (_, i) => i)
  // Which B[k,n] of the tile a thread stages on pass i (the header says which way and why).
  const bSlot = g.bAlongK
    ? (i: number): [string, string] => [`col`, `row + ${i * TILE}u`]
    : (i: number): [string, string] => [`lid / ${TILE_N}u + ${i * (WG_SIZE / TILE_N)}u`, `lid % ${TILE_N}u`]
  const stageA = range(TM).map(i =>
    `    { let m = m0 + row + ${i * TILE}u; let k = k0 + col; var v : f32 = 0.0; ${g.a} Atile[(row + ${i * TILE}u) * ${TILE}u + col] = v; }`)
  const stageB = range(TN).map(i => {
    const [bk, bn] = bSlot(i)
    return `    { let bk = ${bk}; let bn = ${bn}; let k = k0 + bk; let n = n0 + bn; var v : f32 = 0.0; ${g.b} Btile[bk * ${TILE_N}u + bn] = v; }`
  })
  const accs = range(TM).flatMap(i => range(TN).map(j => `acc${i}${j}`))
  const wgsl = `
var<workgroup> Atile : array<f32, ${TILE_M * TILE}>;
var<workgroup> Btile : array<f32, ${TILE * TILE_N}>;
@group(0) @binding(0) var<storage, read> ${g.names[0]} : array<f32>;
@group(0) @binding(1) var<storage, read> ${g.names[1]} : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lid : u32) {
  // One workgroup per ${TILE_M}×${TILE_N} output tile, ${TM}×${TN} outputs per thread. wid is
  // workgroup-uniform, so the early-out keeps the barriers below in uniform control flow.
${[idx.wgsl, g.prologue].filter(Boolean).join('\n')}
  let row = lid / ${TILE}u;
  let col = lid % ${TILE}u;
  let m0 = (t / ${tilesN}u) * ${TILE_M}u;
  let n0 = (t % ${tilesN}u) * ${TILE_N}u;
${accs.map(a => `  var ${a} : f32 = 0.0;`).join('\n')}
  for (var k0 : u32 = 0u; k0 < ${K}u; k0 = k0 + ${TILE}u) {
${stageA.join('\n')}
${stageB.join('\n')}
    workgroupBarrier();
    for (var kk : u32 = 0u; kk < ${TILE}u; kk = kk + 1u) {
${range(TM).map(i => `      let a${i} = Atile[(row + ${i * TILE}u) * ${TILE}u + kk];`).join('\n')}
${range(TN).map(j => `      let b${j} = Btile[kk * ${TILE_N}u + col + ${j * TILE}u];`).join('\n')}
${range(TM).flatMap(i => range(TN).map(j => `      acc${i}${j} = fma(a${i}, b${j}, acc${i}${j});`)).join('\n')}
    }
    workgroupBarrier();
  }
${range(TM).flatMap(i => range(TN).map(j =>
    `  { let m = m0 + row + ${i * TILE}u; let n = n0 + col + ${j * TILE}u; if (m < ${M}u && n < ${N}u) { out[${g.c}] = acc${i}${j}; } }`)).join('\n')}
}`.trim()
  return { wgsl, spec: idx.spec }
}

/** matmul on the tiled template: A and B read flat, `bi` addressing the batch slice of A
 *  and — when B is batched too — of B. Non-null by construction: the callers' `tileable`
 *  gate is exactly the K % TILE test, and matmul sets no grid floor. */
function emitTiledMatmul(batch: number, M: number, K: number, N: number, batched: boolean) {
  return emitTiledGemm({
    M, N, K, z: batch, names: ['a', 'b'], prologue: '',
    a: `if (m < ${M}u) { v = a[bi * ${M * K}u + m * ${K}u + k]; }`,
    b: `if (n < ${N}u) { v = b[${batched ? `bi * ${K * N}u + ` : ''}k * ${N}u + n]; }`,
    c: `bi * ${M * N}u + m * ${N}u + n`,
  })!
}

/** Shared-memory tree reduction across one workgroup. The caller declares
 *  `var<workgroup> <arrName> : array<f32, WG_SIZE>` at module scope, binds a
 *  per-thread accumulator in `accVar`, and names its local index `lid`; this
 *  emits the store + log₂(WG_SIZE) barrier/halve rounds leaving the sum in
 *  `<arrName>[0]`. Also the intended kernel for a future `sum_last` fix —
 *  specs/Performance.md §Phase 3 measured the same few-threads/long-loop
 *  pathology there. */
function emitWorkgroupReduce(arrName: string, accVar: string): string {
  const lines = [`  ${arrName}[lid] = ${accVar};`, `  workgroupBarrier();`]
  for (let off = WG_SIZE >> 1; off > 0; off >>= 1) {
    lines.push(`  if (lid < ${off}u) { ${arrName}[lid] = ${arrName}[lid] + ${arrName}[lid + ${off}u]; }`)
    lines.push(`  workgroupBarrier();`)
  }
  return lines.join('\n')
}

type ConvDims = {
  B: number; cIn: number; H: number; W: number
  cOut: number; cInPerG: number; kH: number; kW: number
  Hout: number; Wout: number
  sH: number; sW: number; pH: number; pW: number; groups: number
}

/** The one shape record the three conv kernels share: the activation `input` is
 *  `[B, cIn, H, W]`, `weight` is `[cOut, cInPerG, kH, kW]`, and `spatial` is whichever
 *  tensor carries `[.., .., Hout, Wout]` (the output for the forward, dy for the grads). */
function convDims(
  input: Shape, weight: Shape, spatial: Shape,
  op: { strideH: number; strideW: number; padH: number; padW: number; groups: number },
): ConvDims {
  return {
    B: input[0]!, cIn: input[1]!, H: input[2]!, W: input[3]!,
    cOut: weight[0]!, cInPerG: weight[1]!, kH: weight[2]!, kW: weight[3]!,
    Hout: spatial[2]!, Wout: spatial[3]!,
    sH: op.strideH, sW: op.strideW, pH: op.padH, pW: op.padW, groups: op.groups,
  }
}

/** The conv family on the tiled GEMM template (2026-08-25, Performance.md §Phase 7) —
 *  conv2d, its input gradient and its weight gradient are each a GEMM whose operands are
 *  gathered from the NCHW tensors instead of read as matrices:
 *
 *    fwd  out[b,co,p] = Σ_k W[co,k] · X[b, k→(ci,kh,kw), p→(ho,wo)]   M=cOutPerG  N=Hout·Wout      K=cInPerG·kH·kW
 *    dx   dx[b,ci,p]  = Σ_k W[k→(co,kh,kw),ci] · dY[b,co,p→(hi,wi)]  M=cInPerG   N=H·W            K=cOutPerG·kH·kW
 *    dw   dW[co,n]    = Σ_k dY[k→(b,ho,wo),co] · X[b, n→(ci,kh,kw)]  M=cOutPerG  N=cInPerG·kH·kW  K=B·Hout·Wout
 *
 *  The gathers carry the padding checks and the template bounds M and N, so the gate is
 *  K % TILE == 0 alone; the naive kernels remain the reference and the odd-K fallback
 *  (depthwise 3×3 has K = 9). fwd and dx repeat the tile grid over batch × group on z; dw
 *  over the group alone, its K already spanning the batch. dw's B is contiguous along k
 *  (there k is the pixel index), so it asks for the transposed staging.
 *
 *  Why it exists: regrow's step measured 80% conv — the naive one-thread-per-output
 *  kernels ran the VGG16 layers at ~1.1 TFLOP/s on a ~20 TFLOP/s part, and a 32768-element
 *  1×1 weight-grad fell one element outside the Phase 5 gate into the naive path (89 ms of
 *  a 404 ms step on its own). Measured (regrow-perf bulb, RTX 3080 Ti laptop, GPU-validated
 *  by conv-tiled-validate): this alone took the step 404 → 221 ms; conv family 254 → 66 ms
 *  of GPU time, the VGG 3×3 layers at 3.3-4.4 TFLOP/s, that weight-grad 89 → 16 ms. */
function emitTiledConv(
  kind: 'fwd' | 'dx' | 'dw', d: ConvDims,
): { wgsl: string; spec: { threads: number; dispatchZ?: number } } | null {
  const { B, cIn, H, W, cOut, cInPerG, kH, kW, Hout, Wout, sH, sW, pH, pW, groups } = d
  const cOutPerG = cOut / groups
  const kHW = kH * kW, HWo = Hout * Wout, HW = H * W
  const M = kind === 'dx' ? cInPerG : cOutPerG
  const N = kind === 'fwd' ? HWo : kind === 'dx' ? HW : cInPerG * kHW
  const K = kind === 'fwd' ? cInPerG * kHW : kind === 'dx' ? cOutPerG * kHW : B * HWo
  const z = kind === 'dw' ? groups : B * groups

  // (outer, kh, kw) from a flattened (outer·kH·kW) index; the 1×1 case folds to identity.
  const tap = (k: string, outer: string) => kHW === 1
    ? `let ${outer} = ${k}; let kh = 0u; let kw = 0u;`
    : `let ${outer} = ${k} / ${kHW}u; let _r = ${k} % ${kHW}u; let kh = _r / ${kW}u; let kw = _r % ${kW}u;`
  // v = X[b, ci, ho*sH + kh - pH, wo*sW + kw - pW], zero outside the image. `b` names the
  // batch variable in scope; ci, kh, kw, ho, wo are the names the decoders above declare.
  const gatherX = (b: string) =>
    `let hi = i32(ho * ${sH}u + kh) - ${pH}; let wi = i32(wo * ${sW}u + kw) - ${pW};\n` +
    `      if (hi >= 0 && hi < ${H} && wi >= 0 && wi < ${W}) { v = input[((${b} * ${cIn}u + g * ${cInPerG}u + ci) * ${H}u + u32(hi)) * ${W}u + u32(wi)]; }`
  const bg = groups === 1 ? `  let b = bi;\n  let g = 0u;` : `  let b = bi / ${groups}u;\n  let g = bi % ${groups}u;`

  // Where each kind's GEMM operands sit in the NCHW buffers, in the TiledGemm contract; the
  // prologue decodes bi into the batch variable and g, which the snippets use as well.
  const gemms: Record<typeof kind, Pick<TiledGemm, 'names' | 'prologue' | 'a' | 'b' | 'c'>> = {
    fwd: {
      names: ['input', 'weight'], prologue: bg,
      a: `if (m < ${M}u) { v = weight[(g * ${cOutPerG}u + m) * ${K}u + k]; }`,
      b: `if (n < ${N}u) {\n      ${tap('k', 'ci')}\n      let ho = n / ${Wout}u; let wo = n % ${Wout}u;\n      ${gatherX('b')}\n    }`,
      c: `(b * ${cOut}u + g * ${cOutPerG}u + m) * ${HWo}u + n`,
    },
    dx: {
      names: ['weight', 'dy'], prologue: bg,
      a: `if (m < ${M}u) { ${kHW === 1 ? `let co = k; let _r = 0u;` : `let co = k / ${kHW}u; let _r = k % ${kHW}u;`} v = weight[((g * ${cOutPerG}u + co) * ${cInPerG}u + m) * ${kHW}u + _r]; }`,
      b: `if (n < ${N}u) {\n      ${tap('k', 'co')}\n      let hi = n / ${W}u; let wi = n % ${W}u;\n` +
        `      let numH = i32(hi) + ${pH} - i32(kh); let numW = i32(wi) + ${pW} - i32(kw);\n` +
        `      if (numH >= 0 && numW >= 0${sH > 1 ? ` && (numH % ${sH}) == 0` : ''}${sW > 1 ? ` && (numW % ${sW}) == 0` : ''}) {\n` +
        `        let ho = numH / ${sH}; let wo = numW / ${sW};\n` +
        `        if (ho < ${Hout} && wo < ${Wout}) { v = dy[((b * ${cOut}u + g * ${cOutPerG}u + co) * ${Hout}u + u32(ho)) * ${Wout}u + u32(wo)]; }\n` +
        `      }\n    }`,
      c: `(b * ${cIn}u + g * ${cInPerG}u + m) * ${HW}u + n`,
    },
    dw: {
      names: ['input', 'dy'], prologue: `  let g = bi;`,
      a: `if (m < ${M}u) { let ab = k / ${HWo}u; let ahw = k % ${HWo}u; v = dy[(ab * ${cOut}u + g * ${cOutPerG}u + m) * ${HWo}u + ahw]; }`,
      b: `if (n < ${N}u) {\n      let bb = k / ${HWo}u; let bhw = k % ${HWo}u;\n      let ho = bhw / ${Wout}u; let wo = bhw % ${Wout}u;\n` +
        `      ${tap('n', 'ci')}\n      ${gatherX('bb')}\n    }`,
      c: `(g * ${cOutPerG}u + m) * ${N}u + n`,
    },
  }
  // dw's K is B·Hout·Wout, so a small weight leaves a handful of workgroups each striding
  // tens of thousands of terms while Phase 5's reduce kernel has one workgroup PER weight:
  // the sample bench's MNIST-sized CNN went 0.67 → 2.60 ms of GPU on the tiled path (its
  // [16,8,3,3] grad was 5 workgroups over K = 12544). Regrow's [32,256,1,1] still won at
  // 32, so that is the floor; below it the reduce/naive paths take over. Split-K would
  // remove the cliff — Performance.md §Phase 7 names it as the next lever.
  return emitTiledGemm({ M, N, K, z, ...gemms[kind], ...(kind === 'dw' ? { bAlongK: true, minGrid: 32 } : {}) })
}

/** Decompose a flat thread index `i` into 4 row-major named axes — emits
 *  six `let` lines ready to interpolate inside a kernel body. */
function decompose4d(shape: readonly [number, number, number, number], names: readonly [string, string, string, string]): string {
  const [, d1, d2, d3] = shape
  const [n0, n1, n2, n3] = names
  const stride0 = d1 * d2 * d3
  const stride1 = d2 * d3
  return [
    `  let ${n0} = i / ${stride0}u;`,
    `  let _r0 = i % ${stride0}u;`,
    `  let ${n1} = _r0 / ${stride1}u;`,
    `  let _r1 = _r0 % ${stride1}u;`,
    `  let ${n2} = _r1 / ${d3}u;`,
    `  let ${n3} = _r1 % ${d3}u;`,
  ].join('\n')
}

function wgslDtype(d: 'f32' | 'i32' | 'bool'): string {
  // bool can't be in storage buffers in WGSL; we lower bool-typed tensors to
  // u32 (0/1). In practice bool tensors only appear via explicit `less` /
  // `greater` / `where` — the causal mask is built inline in softmax kernels.
  if (d === 'bool') return 'u32'
  return d
}

function wgslLiteral(value: number, dtype: 'f32' | 'i32' | 'bool'): string {
  if (dtype === 'f32') {
    if (Number.isFinite(value)) {
      // WGSL float literals need a `.` or exponent — force one in.
      return value.toString().includes('.') || value.toString().includes('e')
        ? `${value}f`
        : `${value}.0f`
    }
    return value > 0 ? '1.0e30f' : '-1.0e30f'
  }
  if (dtype === 'i32') return `${Math.trunc(value)}i`
  return value ? '1u' : '0u'
}

function castFromI32(expr: string, dtype: 'f32' | 'i32' | 'bool'): string {
  if (dtype === 'f32') return `f32(${expr})`
  if (dtype === 'i32') return `i32(${expr})`
  return `u32(${expr})`
}

function computeStrides(shape: Shape): number[] {
  const strides: number[] = new Array(shape.length).fill(1)
  for (let i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1]! * shape[i + 1]!
  }
  return strides
}

/**
 * Generate WGSL that decomposes a flat index `flatVar` into per-axis indices
 * `outVar_0, outVar_1, ...` according to `shape`.
 */
function decomposeFlatIndexBlock(flatVar: string, shape: Shape, outVar: string): string {
  if (shape.length === 0) return `  let ${outVar}_0 : u32 = 0u;`
  const strides = computeStrides(shape)
  const lines: string[] = []
  let remaining = flatVar
  for (let i = 0; i < shape.length; i++) {
    if (i === shape.length - 1) {
      lines.push(`  let ${outVar}_${i} = ${remaining};`)
    } else {
      lines.push(`  let ${outVar}_${i} = ${remaining} / ${strides[i]}u;`)
      const newRem = `${outVar}_rem${i}`
      lines.push(`  let ${newRem} = ${remaining} % ${strides[i]}u;`)
      remaining = newRem
    }
  }
  return lines.join('\n')
}

/**
 * Compute the source flat index for an output flat index under right-aligned
 * NumPy broadcasting (size-1 source axes broadcast; output-only leading axes
 * drop). Decomposes the output index per-axis, picks 0 or the matching axis
 * index per source axis (broadcast vs pass-through), recombines via source
 * strides.
 */
function broadcastIndexBlock(flatVar: string, outShape: Shape, srcShape: Shape, srcVar: string): string {
  // Per-axis var names are prefixed with srcVar so multiple calls in the same
  // kernel don't collide.
  const prefix = `${srcVar}_ax`
  const decompose = decomposeFlatIndexBlock(flatVar, outShape, prefix)
  const offset = outShape.length - srcShape.length
  if (srcShape.length === 0) {
    return `${decompose}\n  let ${srcVar} : u32 = 0u;`
  }
  const srcStrides = computeStrides(srcShape)
  const terms: string[] = []
  for (let i = 0; i < srcShape.length; i++) {
    const outAxis = i + offset
    const srcDim = srcShape[i]!
    const term = srcDim === 1 ? '0u' : `${prefix}_${outAxis} * ${srcStrides[i]}u`
    terms.push(term)
  }
  return `${decompose}\n  let ${srcVar} = ${terms.join(' + ')};`
}

/**
 * One thread per output cell. Reduced source axes — leading-prefix axes
 * (in src, missing from tgt) and any tgt=1/src>1 axis — get explicit nested
 * for-loops; pass-through axes are indexed directly via tgt_k.
 *
 * Except: few output cells × long reduction (bias grads, broadcast grads —
 * e.g. a Linear bias reducing over every row the layer saw) starves the GPU
 * the same way conv2d_weight_grad did; below the thread threshold the
 * emitted kernel is one WORKGROUP per output cell, WG_SIZE threads striding
 * a flattened reduction index, shared-memory tree reduce. Returns `threads`
 * alongside the WGSL because the two forms dispatch differently.
 */
function emitSumToShape(srcShape: Shape, tgtShape: Shape, dtype: 'f32' | 'i32' | 'bool'): { wgsl: string, threads: number } {
  const srcStrides = computeStrides(srcShape)
  const tgtStrides = computeStrides(tgtShape)
  const offset = srcShape.length - tgtShape.length

  const decompose = decomposeFlatIndexBlock('i', tgtShape, 'tgt')

  const reducedAxes: number[] = []
  for (let k = 0; k < srcShape.length; k++) {
    if (k < offset) { reducedAxes.push(k); continue }
    const tDim = tgtShape[k - offset]!
    const sDim = srcShape[k]!
    if (tDim === 1 && sDim > 1) reducedAxes.push(k)
  }

  const baseTerms: string[] = []
  for (let k = 0; k < srcShape.length; k++) {
    if (reducedAxes.includes(k)) continue
    const tAxis = k - offset
    baseTerms.push(`tgt_${tAxis} * ${srcStrides[k]}u`)
  }
  const baseExpr = baseTerms.length > 0 ? baseTerms.join(' + ') : '0u'

  const total = tgtShape.length === 0 ? 1 : (tgtStrides[0]! * tgtShape[0]!)
  const redLen = reducedAxes.reduce((p, k) => p * srcShape[k]!, 1)
  const reducedTerms = reducedAxes.map(k => `r${k} * ${srcStrides[k]}u`)

  if (dtype === 'f32' && total < 32768 && redLen >= 256) {
    // Decompose the flat reduction index r (row-major over reducedAxes'
    // dims) back into per-axis coordinates, right to left.
    const rDecomp: string[] = ['    var rem = r;']
    for (let d = reducedAxes.length - 1; d >= 0; d--) {
      const k = reducedAxes[d]!
      if (d === 0) {
        rDecomp.push(`    let r${k} = rem;`)
      } else {
        rDecomp.push(`    let r${k} = rem % ${srcShape[k]!}u;`)
        rDecomp.push(`    rem = rem / ${srcShape[k]!}u;`)
      }
    }
    const wgsl = `
var<workgroup> partial : array<f32, ${WG_SIZE}>;
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lid : u32) {
  // One workgroup per output cell. wid is workgroup-uniform, so the
  // early-out keeps every barrier below in uniform control flow.
  let i = wid.x + wid.y * 65535u;
  if (i >= ${total}u) { return; }
${decompose}
  var s : f32 = 0.0;
  for (var r : u32 = lid; r < ${redLen}u; r = r + ${WG_SIZE}u) {
${rDecomp.join('\n')}
    s = s + a[${baseExpr} + ${reducedTerms.join(' + ')}];
  }
${emitWorkgroupReduce('partial', 's')}
  if (lid == 0u) { out[i] = partial[0]; }
}`.trim()
    return { wgsl, threads: total * WG_SIZE }
  }

  const indent = (depth: number) => '  '.repeat(depth + 1)
  const loops: string[] = []
  for (let depth = 0; depth < reducedAxes.length; depth++) {
    const k = reducedAxes[depth]!
    const dim = srcShape[k]!
    loops.push(`${indent(depth)}for (var r${k} : u32 = 0u; r${k} < ${dim}u; r${k} = r${k} + 1u) {`)
  }
  const fullExpr = reducedTerms.length > 0
    ? `${baseExpr} + ${reducedTerms.join(' + ')}`
    : baseExpr
  loops.push(`${indent(reducedAxes.length)}s = s + a[${fullExpr}];`)
  for (let depth = reducedAxes.length - 1; depth >= 0; depth--) {
    loops.push(`${indent(depth)}}`)
  }

  const loopBody = reducedAxes.length === 0
    ? `  s = s + a[${baseExpr}];`
    : loops.join('\n')

  const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(dtype)}>;
@group(0) @binding(1) var<storage, read_write> out : array<${wgslDtype(dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${decompose}
  var s : ${wgslDtype(dtype)} = ${dtype === 'f32' ? '0.0f' : (dtype === 'i32' ? '0i' : '0u')};
${loopBody}
  out[i] = s;
}`.trim()
  return { wgsl, threads: total }
}
