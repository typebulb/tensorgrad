// Buffer planning with liveness-based pooling. Static shapes make every
// tensor's size and lifetime known at compile time, so intermediates whose
// lifetimes don't overlap share one GPU buffer. Named buffers (params,
// param-grads, inputs, state), graph outputs, captures, and writeback sources
// get dedicated buffers that outlive their last op; every other tensor is an
// ephemeral intermediate, reclaimed to a free list the moment its last reader
// has run and handed to the next intermediate that fits.
//
// This is what makes deep unrolls (RNNs / NCAs / diffusion) fit: with one
// buffer per tensor, a K-step backward pass keeps O(K) gradient activations
// resident at once, even though each is dead one step after it's produced.
// Pooling collapses that to O(1) — roughly halving peak memory on any graph
// with a substantial backward pass.
//
// `BufferSpec.kind` is what the runtime branches on for allocation, upload,
// readback, and lifetime.

import type { Graph, Tensor, Dtype, Shape, OpNode } from './ir.js'
import { getOpInputs } from './ir.js'
import { shapeSize } from './shape.js'

/** One entry per GPU buffer — fewer than tensors, since intermediates with
 *  disjoint lifetimes are pooled (see `planBuffers` header). `kind` discriminates
 *  how the runtime should treat it (upload vs read-back vs persistent vs ephemeral).
 *  NOTE: for pooled intermediates, `dtype`/`shape` describe the FIRST occupant
 *  only — later tenants of the same buffer may differ (all dtypes are 4-byte and
 *  storage buffers are untyped, so nothing at runtime consumes them; don't let a
 *  future debug/readback path trust them for intermediates). */
export interface BufferSpec {
  /** Buffer id == index into `BufferPlan.buffers`. NOT a tensor id — pooling
   *  makes tensor→buffer many-to-one; always resolve through `tensorToBuffer`. */
  id: number
  /** Allocation size in bytes (padded to ≥ 4 even for 0-d scalars). */
  byteSize: number
  dtype: Dtype
  shape: Shape
  /** What this buffer is for. Drives runtime allocation, upload, readback,
   *  and lifetime decisions. */
  kind: 'param' | 'param_grad' | 'tensor_input' | 'state' | 'intermediate' | 'output'
  /** External name for `param`/`param_grad`/`tensor_input`/`state` bindings.
   *  `null` for `intermediate` / `output`. */
  name: string | null
  /** For state buffers: the value to fill on initial allocation. 0 by default. */
  initValue?: number
}

/**
 * After step(), copy `source`'s buffer into `dest`'s buffer.
 * Used to write back updated optimizer state and updated parameters into
 * their persistent home buffers.
 */
export interface Writeback {
  source: number  // buffer id of the tensor holding the new value
  dest: number    // buffer id of the persistent state/param to overwrite
  bytes: number
}

/** Compile-time GPU memory layout. Produced by `planBuffers`, consumed by
 *  the runtime to allocate buffers and by codegen to wire bind groups.
 *  Lookup maps avoid linear scans of `buffers` at runtime. */
export interface BufferPlan {
  /** Every allocation the runtime needs, indexed by `BufferSpec.id` (its array index). */
  buffers: BufferSpec[]
  /** Tensor id -> buffer id. Many-to-one: intermediates with disjoint lifetimes share a buffer. */
  tensorToBuffer: Map<number, number>
  /** Param name -> buffer id. Used for uploads/downloads. */
  paramsByName: Map<string, number>
  /** Tensor-input name -> buffer id. Filled per step from the inputs record. */
  inputsByName: Map<string, number>
  /** Param name -> buffer id of that param's gradient tensor. */
  paramGradsByName: Map<string, number>
  /** State name -> buffer id of its persistent home. */
  statesByName: Map<string, number>
  /** Capture name -> buffer id of the registered activation. */
  capturesByName: Map<string, number>
  /** Graph outputs mapped through `tensorToBuffer`. */
  outputBufferIds: number[]
  /** End-of-step writebacks (Adam updates for params, m, v, etc.) */
  writebacks: Writeback[]
}

const dtypeBytes: Record<Dtype, number> = { f32: 4, i32: 4, bool: 4 }

/**
 * Caller-supplied writeback declarations: "after each step, copy this Tensor's
 * buffer into the persistent home of this param/state."
 */
export interface WritebackDecl {
  /** The Tensor (output of some op) holding the new value to write back. */
  source: Tensor
  /** Either a param name (writes to that param's home buffer) or a state name. */
  destName: string
  destKind: 'param' | 'state'
}

/**
 * Build a BufferPlan from a graph + the param-grad map produced by appendGrad.
 * @param graph the full graph (forward + backward + any optimizer ops)
 * @param paramGrads map from param name -> the Tensor that holds its gradient
 * @param writebackDecls list of end-of-step writebacks (e.g. from appendAdam).
 *                       Empty when there's no optimizer in the graph.
 */
export function planBuffers(
  graph: Graph,
  paramGrads: Record<string, Tensor>,
  writebackDecls: WritebackDecl[] = [],
): BufferPlan {
  const buffers: BufferSpec[] = []
  const tensorToBuffer = new Map<number, number>()
  const paramsByName = new Map<string, number>()
  const inputsByName = new Map<string, number>()
  const paramGradsByName = new Map<string, number>()
  const statesByName = new Map<string, number>()

  const gradTensorIdToName = new Map<number, string>()
  for (const [name, tensor] of Object.entries(paramGrads)) {
    gradTensorIdToName.set(tensor.id, name)
  }
  const opByOutId = new Map<number, OpNode>()
  for (const op of graph.ops) opByOutId.set(op.out, op)
  const tensorsById = new Map<number, Tensor>()
  for (const t of graph.tensors) tensorsById.set(t.id, t)

  const outputSet = new Set(graph.outputs)
  const captureTensorIds = new Set<number>()
  for (const [, tid] of graph.captures) captureTensorIds.add(tid)
  const writebackSourceIds = new Set<number>(writebackDecls.map(d => d.source.id))

  // Classify each tensor into its buffer kind (drives the runtime) + external
  // name. The producing op decides param/input/state; the grad map and output
  // set pin the rest. Everything else is a plain intermediate.
  interface Cls { kind: BufferSpec['kind']; name: string | null; initValue?: number }
  const cls = new Map<number, Cls>()
  for (const t of graph.tensors) {
    const op = opByOutId.get(t.id)
    if (op?.kind === 'param_input') cls.set(t.id, { kind: 'param', name: op.name })
    else if (op?.kind === 'tensor_input') cls.set(t.id, { kind: 'tensor_input', name: op.name })
    else if (op?.kind === 'state_input') cls.set(t.id, { kind: 'state', name: op.name, initValue: op.initValue })
    else if (gradTensorIdToName.has(t.id)) cls.set(t.id, { kind: 'param_grad', name: gradTensorIdToName.get(t.id)! })
    else if (outputSet.has(t.id)) cls.set(t.id, { kind: 'output', name: null })
    else cls.set(t.id, { kind: 'intermediate', name: null })
  }

  // A tensor may share a buffer only if it's a plain intermediate whose value
  // isn't needed after its last op. Captures, outputs, param-grads, params,
  // inputs, state, and writeback sources all outlive their last op, so each
  // gets a dedicated buffer that's never reclaimed.
  const isPoolable = (id: number): boolean =>
    cls.get(id)!.kind === 'intermediate' && !captureTensorIds.has(id) && !writebackSourceIds.has(id)

  // Liveness over execution (== graph.ops) order: lastUse[t] = index of the
  // last op that reads t. graph.ops is topologically ordered, so this is the
  // order the runtime dispatches kernels, and a buffer freed at op i is
  // available from op i+1 on.
  const lastUse = new Map<number, number>()
  graph.ops.forEach((op, i) => {
    for (const inId of getOpInputs(op)) {
      const prev = lastUse.get(inId)
      if (prev === undefined || i > prev) lastUse.set(inId, i)
    }
  })

  // free list of reclaimed intermediate buffers, available for reuse
  const free: { bufId: number; byteSize: number }[] = []
  const alloc = (byteSize: number, t: Tensor, kind: BufferSpec['kind'], name: string | null, initValue?: number): number => {
    const id = buffers.length
    buffers.push({ id, byteSize, dtype: t.dtype, shape: t.shape, kind, name, ...(initValue !== undefined ? { initValue } : {}) })
    return id
  }

  // Walk ops in order. Per op: (1) give its output a buffer — reuse a free one
  // for poolable intermediates, else allocate a dedicated one; (2) THEN reclaim
  // inputs whose last use is this op. Freeing AFTER allocating is what keeps an
  // output from landing on one of its own live inputs, which would bind the
  // same buffer as read + read_write in one bind group (a WebGPU error).
  for (let i = 0; i < graph.ops.length; i++) {
    const op = graph.ops[i]!
    const t = tensorsById.get(op.out)!
    const need = Math.max(4, shapeSize(t.shape) * dtypeBytes[t.dtype])
    let bufId: number
    if (isPoolable(op.out)) {
      let best = -1  // best-fit: smallest free buffer that's big enough
      for (let k = 0; k < free.length; k++) {
        if (free[k]!.byteSize >= need && (best < 0 || free[k]!.byteSize < free[best]!.byteSize)) best = k
      }
      if (best >= 0) { bufId = free[best]!.bufId; free.splice(best, 1) }
      else bufId = alloc(need, t, 'intermediate', null)
    } else {
      const c = cls.get(op.out)!
      bufId = alloc(need, t, c.kind, c.name, c.initValue)
      if (c.kind === 'param') paramsByName.set(c.name!, bufId)
      else if (c.kind === 'tensor_input') inputsByName.set(c.name!, bufId)
      else if (c.kind === 'param_grad') paramGradsByName.set(c.name!, bufId)
      else if (c.kind === 'state') statesByName.set(c.name!, bufId)
    }
    tensorToBuffer.set(op.out, bufId)

    const freed = new Set<number>()
    for (const inId of getOpInputs(op)) {
      if (freed.has(inId) || !isPoolable(inId) || lastUse.get(inId) !== i) continue
      freed.add(inId)
      const b = tensorToBuffer.get(inId)!
      free.push({ bufId: b, byteSize: buffers[b]!.byteSize })
    }
    // an intermediate nothing ever reads is dead the instant it's written
    if (isPoolable(op.out) && lastUse.get(op.out) === undefined) {
      free.push({ bufId, byteSize: buffers[bufId]!.byteSize })
    }
  }

  const outputBufferIds = graph.outputs.map(id => tensorToBuffer.get(id)!)

  // Compile-time self-check: no two occupants of a shared buffer may have
  // overlapping live intervals. This guards future edits to the allocator
  // above (free-list bugs, alloc-before-free regressions) by re-deriving the
  // intervals and validating the final assignment. It CANNOT catch an
  // incomplete `getOpInputs` case for a new op — that corrupts `lastUse`
  // itself, which both the allocator and this check consume.
  const defIdx = new Map<number, number>()
  graph.ops.forEach((op, i) => defIdx.set(op.out, i))
  const occupants = new Map<number, { tid: number; def: number; end: number }[]>()
  for (const [tid, bufId] of tensorToBuffer) {
    if (!isPoolable(tid)) continue
    const def = defIdx.get(tid)
    if (def === undefined) continue
    const list = occupants.get(bufId) ?? []
    list.push({ tid, def, end: lastUse.get(tid) ?? def })   // dead-on-write: interval is just [def, def]
    occupants.set(bufId, list)
  }
  for (const [bufId, list] of occupants) {
    list.sort((a, b) => a.def - b.def)
    for (let k = 1; k < list.length; k++) {
      const prev = list[k - 1]!, cur = list[k]!
      // freed at op `end` means available from `end + 1`; equality would also
      // be the output-aliases-live-input (read + read_write bind) case
      if (cur.def <= prev.end) {
        throw new Error(
          `planBuffers: lifetime overlap on buffer ${bufId}: ` +
          `tensor #${prev.tid} live [${prev.def}, ${prev.end}] vs tensor #${cur.tid} defined at ${cur.def}`,
        )
      }
    }
  }

  const writebacks: Writeback[] = writebackDecls.map(decl => {
    const sourceBufId = tensorToBuffer.get(decl.source.id)
    if (sourceBufId === undefined) {
      throw new Error(`planBuffers: writeback source tensor #${decl.source.id} not in graph`)
    }
    const destBufId = decl.destKind === 'param'
      ? paramsByName.get(decl.destName)
      : statesByName.get(decl.destName)
    if (destBufId === undefined) {
      throw new Error(`planBuffers: writeback dest ${decl.destKind}:'${decl.destName}' not found`)
    }
    const sourceSpec = buffers[sourceBufId]!
    const destSpec = buffers[destBufId]!
    if (sourceSpec.byteSize !== destSpec.byteSize) {
      throw new Error(
        `planBuffers: writeback size mismatch for ${decl.destKind}:'${decl.destName}' ` +
        `(source ${sourceSpec.byteSize} bytes vs dest ${destSpec.byteSize})`,
      )
    }
    return { source: sourceBufId, dest: destBufId, bytes: sourceSpec.byteSize }
  })

  const capturesByName = new Map<string, number>()
  for (const [name, tensorId] of graph.captures) {
    const bufId = tensorToBuffer.get(tensorId)
    if (bufId === undefined) {
      throw new Error(`planBuffers: capture '${name}' references unknown tensor #${tensorId}`)
    }
    capturesByName.set(name, bufId)
  }

  return { buffers, tensorToBuffer, paramsByName, inputsByName, paramGradsByName, statesByName, capturesByName, outputBufferIds, writebacks }
}
