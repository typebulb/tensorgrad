// Dead-code elimination over a traced Graph. Every op in `graph.ops` becomes
// an executed kernel (there is no laziness downstream), so any op whose output
// nothing consumes is pure wasted GPU work. The headline producer of such ops
// is autograd: `appendGrad` emits a full adjoint for every differentiable op
// it walks, including gradients nobody reads — e.g. `conv2d_weight_grad` for
// frozen weights passed as tensor inputs (the documented freeze idiom), plus
// the cotangent-accumulate adds feeding them.
//
// Runs between `appendGrad` and the optimizer pass (see compile.ts): at that
// point the roots are exactly the loss output, capture sites, and the param
// gradients the optimizer is about to consume. Running before the optimizer
// pass keeps the optimizer's own tensors (state inputs, update ops,
// writeback sources) out of the remap entirely — they're created against the
// already-compacted graph.

import type { Graph, OpNode, Tensor } from './ir.js'
import { getOpInputs, remapOpInputs } from './ir.js'

/**
 * Remove every op whose output is not reachable (via op inputs) from the
 * graph's outputs, its capture sites, or `extraRoots`. Named leaves
 * (`param_input` / `tensor_input` / `state_input`) are always kept — params
 * must exist for upload/download regardless of gradient flow, and inputs are
 * part of the declared call surface.
 *
 * Mutates `graph` in place, renumbering tensor ids and op indices so the
 * surviving graph is dense (buffer planning allocates one GPU buffer per
 * tensor — stale tensors would still cost memory). Returns a map from old
 * tensor id to the new `Tensor` handle for every surviving tensor; callers
 * holding pre-DCE handles (param grads, the loss, materialized param tensors)
 * must re-resolve them through it.
 */
export function eliminateDeadCode(graph: Graph, extraRoots: Iterable<number> = []): Map<number, Tensor> {
  // Mark: walk backward from the roots through each producer's inputs.
  const live = new Set<number>()
  const stack: number[] = [...graph.outputs, ...graph.captures.values(), ...extraRoots]
  while (stack.length > 0) {
    const id = stack.pop()!
    if (live.has(id)) continue
    live.add(id)
    const source = graph.tensors[id]!.source
    if (source === null) continue
    for (const input of getOpInputs(graph.ops[source]!)) stack.push(input)
  }

  // Sweep: rebuild ops + tensors densely, preserving construction order (which
  // is the topological / dispatch order downstream passes rely on).
  const idMap = new Map<number, Tensor>()
  const mapId = (old: number): number => {
    const t = idMap.get(old)
    if (!t) throw new Error(`dce: op input references tensor #${old} that was eliminated (internal ordering bug)`)
    return t.id
  }
  const newOps: OpNode[] = []
  const newTensors: Tensor[] = []
  for (const op of graph.ops) {
    const isNamedLeaf = op.kind === 'param_input' || op.kind === 'tensor_input' || op.kind === 'state_input'
    if (!isNamedLeaf && !live.has(op.out)) continue
    const old = graph.tensors[op.out]!
    const remapped = remapOpInputs(op, mapId)
    const t: Tensor = { id: newTensors.length, shape: old.shape, dtype: old.dtype, source: newOps.length, site: old.site }
    newOps.push({ ...remapped, out: t.id })
    newTensors.push(t)
    idMap.set(op.out, t)
  }

  const newOutputs = graph.outputs.map(mapId)
  graph.ops.length = 0
  for (const op of newOps) graph.ops.push(op)
  graph.tensors.length = 0
  for (const t of newTensors) graph.tensors.push(t)
  graph.outputs.length = 0
  for (const id of newOutputs) graph.outputs.push(id)
  for (const [name, id] of graph.captures) graph.captures.set(name, mapId(id))
  return idMap
}
