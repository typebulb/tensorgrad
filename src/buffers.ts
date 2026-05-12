// Buffer planning. v1 strategy: one GPU buffer per IR Tensor, no pooling.
// Static shapes make every buffer's size known at compile time and lifetimes
// don't overlap between steps. Total memory is the sum of every intermediate
// (~30 MB at transformer B=256 — easily fits).
//
// `BufferSpec.kind` is what the runtime branches on for allocation, upload,
// readback, and lifetime.

import type { Graph, Tensor, Dtype, Shape, OpNode } from './ir.js'
import { shapeSize } from './shape.js'

/** One entry per GPU buffer. v1: one buffer per IR tensor (no pooling — see
 *  `planBuffers` header). `kind` discriminates how the runtime should treat it
 *  (upload vs read-back vs persistent vs ephemeral). */
export interface BufferSpec {
  /** Matches `Tensor.id`. */
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
  /** Every allocation the runtime needs, indexed by `BufferSpec.id` (== tensor id). */
  buffers: BufferSpec[]
  /** Tensor id -> buffer id (currently 1:1 but kept opaque for future pooling). */
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

  const outputSet = new Set(graph.outputs)

  for (const t of graph.tensors) {
    const op = opByOutId.get(t.id)
    let kind: BufferSpec['kind'] = 'intermediate'
    let name: string | null = null
    let initValue: number | undefined

    if (op?.kind === 'param_input') {
      kind = 'param'
      name = op.name
    } else if (op?.kind === 'tensor_input') {
      kind = 'tensor_input'
      name = op.name
    } else if (op?.kind === 'state_input') {
      kind = 'state'
      name = op.name
      initValue = op.initValue
    } else if (gradTensorIdToName.has(t.id)) {
      kind = 'param_grad'
      name = gradTensorIdToName.get(t.id)!
    } else if (outputSet.has(t.id)) {
      kind = 'output'
    }

    const spec: BufferSpec = {
      id: t.id,
      byteSize: Math.max(4, shapeSize(t.shape) * dtypeBytes[t.dtype]),
      dtype: t.dtype,
      shape: t.shape,
      kind,
      name,
      ...(initValue !== undefined ? { initValue } : {}),
    }
    buffers.push(spec)
    tensorToBuffer.set(t.id, t.id)

    if (kind === 'param') paramsByName.set(name!, t.id)
    if (kind === 'tensor_input') inputsByName.set(name!, t.id)
    if (kind === 'param_grad') paramGradsByName.set(name!, t.id)
    if (kind === 'state') statesByName.set(name!, t.id)
  }

  const outputBufferIds = graph.outputs.map(id => tensorToBuffer.get(id)!)

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
