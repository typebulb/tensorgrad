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

// Global thread index packed across the 2D dispatch grid (see runtime.ts).
// `MAX_X * WG_SIZE = 65535 * 256 = 16776960`. Inlined as a string so the WGSL
// compiler sees the value as a literal constant.
const GID_LINE = 'let i = gid.x + gid.y * 16776960u;'

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
  /** Number of threads to dispatch (1-D). 0 means "skip" (e.g. reshape no-op). */
  threads: number
  /** Workgroup size; usually WG_SIZE. */
  workgroupSize: number
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
        // tanh identity for numerical stability: sigmoid(x) = 0.5 + 0.5 * tanh(0.5x)
        /* sigmoid */           '0.5 + 0.5 * tanh(0.5 * x)'
      const wgsl = `
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

    // ---- Shape / detach ----------------------------------------------------
    // Both ops are byte-identical memcpy; reshape relabels the shape, while
    // stop_gradient detaches from the autograd graph. Aliasing the buffers
    // would save a copy but complicates the buffer plan; we have headroom.
    case 'reshape':
    case 'stop_gradient': {
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
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let bi = i / ${M * N}u;
  let mn = i % ${M * N}u;
  let m = mn / ${N}u;
  let n = mn % ${N}u;
  let aBase = bi * ${M * K}u + m * ${K}u;
  var s : f32 = 0.0;
  for (var k : u32 = 0u; k < ${K}u; k = k + 1u) {
    s = s + a[aBase + k] * b[k * ${N}u + n];
  }
  c[i] = s;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.b), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    case 'matmul_batched': {
      const out = tof(op.out)
      const a = tof(op.a)
      const b = tof(op.b)
      const M = a.shape[a.shape.length - 2]!
      const K = a.shape[a.shape.length - 1]!
      const N = b.shape[b.shape.length - 1]!
      const batch = shapeSize(a.shape) / (M * K)
      const total = batch * M * N
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let bi = i / ${M * N}u;
  let mn = i % ${M * N}u;
  let m = mn / ${N}u;
  let n = mn % ${N}u;
  let aBase = bi * ${M * K}u + m * ${K}u;
  let bBase = bi * ${K * N}u;
  var s : f32 = 0.0;
  for (var k : u32 = 0u; k < ${K}u; k = k + 1u) {
    s = s + a[aBase + k] * b[bBase + k * ${N}u + n];
  }
  c[i] = s;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.b), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
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
    case 'slice_last_range': {
      const out = tof(op.out)
      const a = tof(op.a)
      const D_in = a.shape[a.shape.length - 1]!
      const D_out = op.end - op.start
      const total = shapeSize(out.shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read> a : array<${wgslDtype(a.dtype)}>;
@group(0) @binding(1) var<storage, read_write> out : array<${wgslDtype(out.dtype)}>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let outer = i / ${D_out}u;
  let inner = i % ${D_out}u;
  out[i] = a[outer * ${D_in}u + ${op.start}u + inner];
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

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
      const wgsl = emitSumToShape(a.shape, out.shape, a.dtype)
      const total = shapeSize(out.shape)
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    // ---- 2D conv + pool ----------------------------------------------------
    case 'conv2d': {
      const input = tof(op.input)
      const weight = tof(op.weight)
      const out = tof(op.out)
      const [, cIn, H, W] = input.shape
      const [cOut, , kH, kW] = weight.shape
      const [, , hOut, wOut] = out.shape
      const total = shapeSize(out.shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${decompose4d(out.shape as [number, number, number, number], ['b', 'cOut_', 'h_out', 'w_out'])}
  let inBase    = b * ${cIn! * H! * W!}u;
  let wBase     = cOut_ * ${cIn! * kH! * kW!}u;
  var s : f32 = 0.0;
  for (var c : u32 = 0u; c < ${cIn!}u; c = c + 1u) {
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
  out[i] = s;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.input), buf(op.weight), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    case 'conv2d_input_grad': {
      // Invert the forward index relation:
      //   h_out = (h_in + padH - kh) / strideH (must divide evenly).
      const weight = tof(op.weight)
      const dy = tof(op.dy)
      const out = tof(op.out)
      const [, cIn, inH, inW] = out.shape
      const [cOut, , kH, kW] = weight.shape
      const [, , hOut, wOut] = dy.shape
      const total = shapeSize(out.shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read> weight : array<f32>;
@group(0) @binding(1) var<storage, read> dy : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${decompose4d(out.shape as [number, number, number, number], ['b', 'c_in_', 'h_in', 'w_in'])}
  var s : f32 = 0.0;
  for (var c_out : u32 = 0u; c_out < ${cOut!}u; c_out = c_out + 1u) {
    let wBase  = c_out * ${cIn! * kH! * kW!}u + c_in_ * ${kH! * kW!}u;
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
  out[i] = s;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.weight), buf(op.dy), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    case 'conv2d_weight_grad': {
      // dWeight[c_out,c_in,kh,kw] = Σ_{b,h_out,w_out} input[b,c_in,h_in,w_in]
      // * dy[b,c_out,h_out,w_out], with h_in = h_out * strideH + kh - padH.
      const input = tof(op.input)
      const dy = tof(op.dy)
      const out = tof(op.out)
      const [cOut, cIn, kH, kW] = out.shape
      const [B, , H, W] = input.shape
      const [, , hOut, wOut] = dy.shape
      const total = shapeSize(out.shape)
      const wgsl = `
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> dy : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${decompose4d(out.shape as [number, number, number, number], ['c_out_', 'c_in_', 'kh', 'kw'])}
  var s : f32 = 0.0;
  for (var b : u32 = 0u; b < ${B!}u; b = b + 1u) {
    let inBase = b * ${cIn! * H! * W!}u + c_in_ * ${H! * W!}u;
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
      const [B, , hOut, wOut] = out.shape
      const total = shapeSize(out.shape)
      // Padding never wins; ties favor earliest in scan order (strictly-greater
      // comparison). Backward must replicate this exact scan to match.
      const NEG = '-3.4e38'
      const wgsl = `
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${decompose4d(out.shape as [number, number, number, number], ['b', 'c', 'h_out', 'w_out'])}
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
  out[i] = m;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.input), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }

    case 'max_pool_2d_grad': {
      // Gather: for each input position, walk every output whose receptive
      // field covers it; recompute its argmax and accumulate dy when we won.
      const input = tof(op.input)
      const dy = tof(op.dy)
      const out = tof(op.out)
      const [B, C, H, W] = input.shape
      const [, , hOut, wOut] = dy.shape
      const total = shapeSize(out.shape)
      const NEG = '-3.4e38'
      void B
      const wgsl = `
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> dy : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
${decompose4d(out.shape as [number, number, number, number], ['b', 'c', 'h_in', 'w_in'])}
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
  out[i] = s;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.input), buf(op.dy), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }
  }
}

// ---- WGSL helpers --------------------------------------------------------

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
 */
function emitSumToShape(srcShape: Shape, tgtShape: Shape, dtype: 'f32' | 'i32' | 'bool'): string {
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

  const indent = (depth: number) => '  '.repeat(depth + 1)
  const loops: string[] = []
  for (let depth = 0; depth < reducedAxes.length; depth++) {
    const k = reducedAxes[depth]!
    const dim = srcShape[k]!
    loops.push(`${indent(depth)}for (var r${k} : u32 = 0u; r${k} < ${dim}u; r${k} = r${k} + 1u) {`)
  }
  const reducedTerms = reducedAxes.map(k => `r${k} * ${srcStrides[k]}u`)
  const fullExpr = reducedTerms.length > 0
    ? `${baseExpr} + ${reducedTerms.join(' + ')}`
    : baseExpr
  loops.push(`${indent(reducedAxes.length)}s = s + a[${fullExpr}];`)
  for (let depth = reducedAxes.length - 1; depth >= 0; depth--) {
    loops.push(`${indent(depth)}}`)
  }

  const total = tgtShape.length === 0 ? 1 : (tgtStrides[0]! * tgtShape[0]!)
  const loopBody = reducedAxes.length === 0
    ? `  s = s + a[${baseExpr}];`
    : loops.join('\n')

  return `
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
}
