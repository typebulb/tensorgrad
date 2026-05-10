// WGSL codegen: one kernel per IR op.
//
// All shapes are baked into the WGSL as compile-time constants — no shape
// uniforms. This means each shape combination produces a distinct shader
// (so `add([B, T, D], [D])` and `add([B, T, D], [B, T, D])` get different
// kernels), which is fine for our static-shape model and gives the WGSL
// compiler full freedom to specialize.
//
// Most kernels are direct ports of `transformer-gpu.bulb.md`'s WGSL — those
// are already debugged and tuned. The autograd ops (broadcast_to, sum_to_shape,
// relu_grad, etc.) are new.

import type { Graph, OpNode, Tensor, Shape } from './ir.js'
import type { BufferPlan } from './buffers.js'
import { shapeSize } from './shape.js'

// Workgroup size of 256 means even our biggest kernel (~8M threads in
// matmul_bwd_dW) needs only ~32K workgroups, well under WebGPU's 65535-per-dim
// dispatch cap. Smaller WG_SIZE forced 2D dispatch with significant over-dispatch.
const WG_SIZE = 256

// Global thread index, packed across the 2D dispatch grid that lets us route
// past WebGPU's 65535-per-dim cap. Every kernel uses this exact line — keep
// the formula consistent with the dispatch-stride math in runtime.ts (MAX_X
// = 65535, so per-row stride = 65535 * WG_SIZE = 16776960). Inlined into
// each WGSL string via interpolation rather than a function so the WGSL
// compiler still sees a literal constant.
const GID_LINE = 'let i = gid.x + gid.y * 16776960u;'

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

// ============================================================================
// Public entry point
// ============================================================================

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

    // ---- Element-wise binops with broadcast --------------------------------
    case 'add':
    case 'sub':
    case 'mul':
    case 'div': {
      const out = tof(op.out)
      const a = tof(op.a)
      const b = tof(op.b)
      const opStr = { add: '+', sub: '-', mul: '*', div: '/' }[op.kind]
      const total = shapeSize(out.shape)
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
  out[i] = a[aIdx] ${opStr} b[bIdx];
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
    case 'relu': {
      const out = tof(op.out)
      const a = tof(op.a)
      const total = shapeSize(out.shape)
      const expr =
        op.kind === 'sqrt'  ? 'sqrt(x)' :
        op.kind === 'rsqrt' ? '1.0 / sqrt(x)' :
        op.kind === 'log'   ? 'log(x)' :
        op.kind === 'exp'   ? 'exp(x)' :
        /* relu */            'max(x, 0.0)'
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

    // ---- Shape ---------------------------------------------------------------
    // reshape: no kernel needed if buffers can alias (shape change only). For
    // v1 simplicity we emit a memcpy-style kernel rather than aliasing buffers,
    // because aliasing complicates the buffer plan and we have memory headroom.
    case 'reshape': {
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

    case 'transpose': {
      const out = tof(op.out)
      const a = tof(op.a)
      const total = shapeSize(out.shape)
      // Emit per-axis index computation. For each output flat index i, decompose
      // into per-axis output indices, then use op.perm to find the source axis order.
      // Source flat index = sum(outIdx[perm.invert()[k]] * a_stride[k] for k).
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
    // matmul: a [..., M, K] · b [K, N] -> [..., M, N]. b is unbatched.
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
  let bi = i / ${M * N}u;          // batch index
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
      const T = a.shape[a.shape.length - 1]!  // == second-to-last (square)
      // Outer size = (everything except last 2 axes) * (second-to-last axis)
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
    case 'adam_update_m': {
      // m_new = b1 * m + (1 - b1) * g
      const out = tof(op.out)
      const total = shapeSize(out.shape)
      const b1 = op.b1
      const oneMinusB1 = 1 - b1
      const wgsl = `
@group(0) @binding(0) var<storage, read> m : array<f32>;
@group(0) @binding(1) var<storage, read> g : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  out[i] = ${wgslLiteral(b1, 'f32')} * m[i] + ${wgslLiteral(oneMinusB1, 'f32')} * g[i];
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.m), buf(op.g), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }
    case 'adam_update_v': {
      // v_new = b2 * v + (1 - b2) * g²
      const out = tof(op.out)
      const total = shapeSize(out.shape)
      const b2 = op.b2
      const oneMinusB2 = 1 - b2
      const wgsl = `
@group(0) @binding(0) var<storage, read> v : array<f32>;
@group(0) @binding(1) var<storage, read> g : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  ${GID_LINE}
  if (i >= ${total}u) { return; }
  let gv = g[i];
  out[i] = ${wgslLiteral(b2, 'f32')} * v[i] + ${wgslLiteral(oneMinusB2, 'f32')} * gv * gv;
}`.trim()
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.v), buf(op.g), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }
    case 'adam_update_p': {
      // p_new = decayShrink * p - lrt[0] * m_new / (sqrt(v_new) + eps).
      // lrt is supplied per-step from CPU (already includes bias correction).
      // decayShrink is either baked as a literal (no schedule, fixed lr) or
      // bound as a per-step scalar input (when the user supplies an lr
      // schedule via `adam: { lr: (step) => ... }`). When literal=1 the WGSL
      // compiler folds the multiply away.
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
      // Sum-reduce src down to target by summing over each axis where target=1
      // or where target is missing (offset-prefix axes that get fully summed).
      const out = tof(op.out)
      const a = tof(op.a)
      const wgsl = emitSumToShape(a.shape, out.shape, a.dtype)
      const total = shapeSize(out.shape)
      return { opIndex, opKind: op.kind, wgsl, bindings: [buf(op.a), buf(op.out)], threads: total, workgroupSize: WG_SIZE }
    }
  }
}

// ============================================================================
// WGSL helpers
// ============================================================================

function wgslDtype(d: 'f32' | 'i32' | 'bool'): string {
  // bool can't be in storage buffers in WGSL; we lower bool-typed tensors to
  // u32 (0/1). For Phase 3a there are no bool-typed storage buffers in the
  // forward+backward graph (causal mask is built inline in softmax kernels),
  // so this only matters if the user explicitly creates a bool tensor.
  if (d === 'bool') return 'u32'
  return d
}

function wgslLiteral(value: number, dtype: 'f32' | 'i32' | 'bool'): string {
  if (dtype === 'f32') {
    if (Number.isFinite(value)) {
      // WGSL requires `.` in float literals; force decimal form.
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
  if (shape.length === 0) return `  let ${outVar}_0 : u32 = 0u;`  // not used but parser-safe
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
 * Generate WGSL that computes the source flat index in `srcVar` for an output
 * flat index `flatVar`, given output shape `outShape` and source shape `srcShape`
 * under right-aligned NumPy-style broadcasting (size-1 axes broadcast).
 *
 * Strategy:
 *   1. Decompose flat output index into per-axis output indices.
 *   2. For each output axis that maps onto a source axis (right-aligned), use
 *      the output index there if src.dim != 1, else 0 (broadcast).
 *   3. Drop output-only axes (those with no corresponding source axis).
 *   4. Combine source indices with source strides.
 */
function broadcastIndexBlock(flatVar: string, outShape: Shape, srcShape: Shape, srcVar: string): string {
  // Name the per-axis decomposition vars after `srcVar` so multiple
  // broadcastIndexBlock calls in the same WGSL function don't collide.
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
 * sum_to_shape: each output cell sums over the source axes that are reduced.
 * For source shape S and target shape T (right-aligned):
 *   - Axes in S not in T (leading prefix): fully reduced (sum over whole axis).
 *   - Axes where T=1 but S>1: reduced (sum over that axis).
 *   - Axes where T=S: passed through.
 *
 * Implementation: each thread = one output cell. It iterates over the reduced
 * axes via nested-loop unrolling (we generate explicit nested for-loops).
 */
function emitSumToShape(srcShape: Shape, tgtShape: Shape, dtype: 'f32' | 'i32' | 'bool'): string {
  const srcStrides = computeStrides(srcShape)
  const tgtStrides = computeStrides(tgtShape)
  const offset = srcShape.length - tgtShape.length

  // Decompose flat output index into per-axis target indices.
  const decompose = decomposeFlatIndexBlock('i', tgtShape, 'tgt')

  // Identify reduced axes of the SOURCE: axis k in src is reduced if either
  // it's in the leading prefix (k < offset) or its corresponding target axis
  // has size 1. For non-reduced axes (k >= offset and tgt=src), the source
  // index is the target index along that axis.
  const reducedAxes: number[] = []
  for (let k = 0; k < srcShape.length; k++) {
    if (k < offset) { reducedAxes.push(k); continue }
    const tDim = tgtShape[k - offset]!
    const sDim = srcShape[k]!
    if (tDim === 1 && sDim > 1) reducedAxes.push(k)
  }

  // Build the source flat index expression. Initialize from the non-reduced axes.
  const baseTerms: string[] = []
  for (let k = 0; k < srcShape.length; k++) {
    if (reducedAxes.includes(k)) continue  // contributed by loop var instead
    const tAxis = k - offset
    baseTerms.push(`tgt_${tAxis} * ${srcStrides[k]}u`)
  }
  const baseExpr = baseTerms.length > 0 ? baseTerms.join(' + ') : '0u'

  // Emit nested for loops over the reduced axes.
  const indent = (depth: number) => '  '.repeat(depth + 1)
  const loops: string[] = []
  for (let depth = 0; depth < reducedAxes.length; depth++) {
    const k = reducedAxes[depth]!
    const dim = srcShape[k]!
    loops.push(`${indent(depth)}for (var r${k} : u32 = 0u; r${k} < ${dim}u; r${k} = r${k} + 1u) {`)
  }
  // Inside innermost loop, compute source index.
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
