// Buffer planning: walk a Graph and decide which GPU buffer each Tensor maps to.
//
// v1 strategy: one GPU buffer per IR Tensor. Static shapes mean every buffer's
// size is known at compile time and lifetimes don't overlap between steps —
// so no pooling needed. Total memory is the sum of every intermediate tensor.
// For our transformer at B=256: ~30 MB of activations + grads. Easily fits.
//
// Categorization is what the runtime cares about:
//   * param        — uploaded by user via uploadParams; persistent across steps
//   * param_grad   — written each step by the backward pass; readable for inspection
//   * tensor_input — uploaded each step (tokens, targets, masks)
//   * intermediate — produced by an op; lifetime = within a single step
//   * output       — special intermediate that should be made readable (loss)

import type { Graph, Tensor, Dtype, Shape, OpNode } from './ir.js'

export interface BufferSpec {
  /** Matches tensor.id. */
  id: number
  byteSize: number
  dtype: Dtype
  shape: Shape
  kind: 'param' | 'param_grad' | 'tensor_input' | 'state' | 'intermediate' | 'output'
  /** External name for param/param_grad/tensor_input/state bindings. null otherwise. */
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

export interface BufferPlan {
  buffers: BufferSpec[]
  /** Tensor id -> buffer id (currently 1:1 but kept opaque for future pooling). */
  tensorToBuffer: Map<number, number>
  /** Easy lookup tables for the runtime. */
  paramsByName: Map<string, number>           // name -> buffer id
  inputsByName: Map<string, number>           // name -> buffer id
  paramGradsByName: Map<string, number>       // name -> buffer id
  statesByName: Map<string, number>           // name -> buffer id (persistent state homes)
  outputBufferIds: number[]                   // graph.outputs mapped through
  /** End-of-step writebacks (Adam updates for params, m, v, etc.) */
  writebacks: Writeback[]
}

const dtypeBytes: Record<Dtype, number> = { f32: 4, i32: 4, bool: 4 }

function shapeSize(shape: Shape): number {
  let n = 1
  for (const d of shape) n *= d
  return n
}

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

  // Build a quick reverse map: tensorId -> param name (for grads).
  const gradTensorIdToName = new Map<number, string>()
  for (const [name, tensor] of Object.entries(paramGrads)) {
    gradTensorIdToName.set(tensor.id, name)
  }
  // ...and tensorId -> param/input op (so we can name the buffer correctly).
  const opByOutId = new Map<number, OpNode>()
  for (const op of graph.ops) opByOutId.set(op.out, op)

  const outputSet = new Set(graph.outputs)

  // Walk all tensors in id order. Categorize each.
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
    tensorToBuffer.set(t.id, t.id)  // 1:1 for v1

    if (kind === 'param') paramsByName.set(name!, t.id)
    if (kind === 'tensor_input') inputsByName.set(name!, t.id)
    if (kind === 'param_grad') paramGradsByName.set(name!, t.id)
    if (kind === 'state') statesByName.set(name!, t.id)
  }

  const outputBufferIds = graph.outputs.map(id => tensorToBuffer.get(id)!)

  // Resolve writeback declarations to (source, dest) buffer-id pairs.
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

  return { buffers, tensorToBuffer, paramsByName, inputsByName, paramGradsByName, statesByName, outputBufferIds, writebacks }
}
