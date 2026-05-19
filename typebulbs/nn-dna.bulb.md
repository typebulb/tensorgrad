---
format: typebulb/v1
name: "NN DNA"
---

**code.tsx**

```tsx
import { App, Component, div as divH, h1, p, a, span, button, h2, textarea, ul, li, code, strong, em } from "domeleon"
import {
  type Tensor, type Graph, type OpNode, type Shape, type CallSite, type CompiledIR,
  type ForwardFn, type InputDecls, type OptimizerConfig,
  getOpInputs,
  Module, trace, traceForward, lr, init,
  Linear, LayerNorm, RMSNorm, Embedding, Conv2d,
  crossEntropy, nllLoss,
  capture, dropout, stopGradient, singleFlight,
  add, sub, mul, div, min, max, less, greater, where,
  sqrt, rsqrt, log, exp, neg, abs, square, sin, cos,
  relu, tanh, sigmoid, gelu, silu, leakyRelu, softplus,
  clamp, randn,
  mean, sum, argmax, argmin,
  reshape, permute, swapAxes,
  splitHeads, mergeHeads,
  matmul,
  oneHot, arange, embedding, takeAlongAxis,
  zeros, ones,
  narrow, concat, stack, split,
  softmax, logSoftmax, softmaxCausal, whereCausal,
  conv2d, maxPool2d, nearestUpsample2d
} from "tensorgrad"
import { instance } from "@viz-js/viz"
import { transform as sucraseTransform } from "sucrase"

// Domeleon's `div` renamed to `divH` so tensorgrad's `div` (tensor /) keeps its natural name in pasted specs.

// ===== Spec evaluation ======================================================

type DimSpec = { size: number; name: string; desc?: string; color?: string }

// Data the bulb traces via trace() / traceForward(). The spec author's
// concrete Module / InputDecls types are erased to base types here — the
// spec's own internal types are validated when the spec is written, not
// at the bulb's consumption point.
interface IRSpec {
  label: string
  description?: string
  model: Module
  loss: ForwardFn<Module, InputDecls>
  inputs: InputDecls
  optimizer: OptimizerConfig
  predict: ForwardFn<Module, InputDecls>
  predictInputs: InputDecls
  dims?: DimSpec[]
}

// Defeats tree-shaking: tensorgrad symbols are referenced at runtime by
// eval'd spec source, which the bundler can't see. Removing this breaks
// specs with `ReferenceError: Module is not defined`.
const _tgKeepalive = {
  Module, lr, init,
  Linear, LayerNorm, RMSNorm, Embedding, Conv2d,
  crossEntropy, nllLoss,
  capture, dropout, stopGradient, singleFlight,
  add, sub, mul, div, min, max, less, greater, where,
  sqrt, rsqrt, log, exp, neg, abs, square, sin, cos,
  relu, tanh, sigmoid, gelu, silu, leakyRelu, softplus,
  clamp, randn,
  mean, sum, argmax, argmin,
  reshape, permute, swapAxes,
  splitHeads, mergeHeads,
  matmul,
  oneHot, arange, embedding, takeAlongAxis,
  zeros, ones,
  narrow, concat, stack, split,
  softmax, logSoftmax, softmaxCausal, whereCausal,
  conv2d, maxPool2d, nearestUpsample2d,
}
;(globalThis as any).__tg_keepalive = _tgKeepalive

// Cleaned source returned alongside spec so stack-frame lines map back to the original.
function evaluateSpec(source: string): { spec: IRSpec; cleanedSource: string } {
  // Multi-line imports stripped but newlines kept, so eval'd line numbers stay aligned with the original.
  let s = source.replace(
    /^\s*import\b[\s\S]*?['"][^'"]+['"];?\s*$/gm,
    (m) => m.replace(/[^\n]/g, ""),
  )
  s = s.replace(/^\s*export\s+(?=(const|let|var|class|function|async|interface|type))/gm, "")
  const { code: js } = sucraseTransform(s, { transforms: ["typescript"] })
  const spec = eval("(() => {" + js + "\n; return irSpec; })()") as IRSpec
  return { spec, cleanedSource: s }
}

// ===== IR utilities =========================================================

function forwardReachable(graph: Graph, outputs: readonly number[]): Set<number> {
  const ops = new Set<number>()
  const stack = [...outputs]
  const visited = new Set<number>()
  while (stack.length) {
    const tid = stack.pop()!
    if (visited.has(tid)) continue
    visited.add(tid)
    const t = graph.tensors[tid]!
    if (t.source === null) continue
    ops.add(t.source)
    for (const inp of getOpInputs(graph.ops[t.source]!)) stack.push(inp)
  }
  return ops
}

// ===== Source attribution ===================================================

type UserFrame = { url: string; line: number }

// Spec lives inside eval, so V8 reports its frames as `<anonymous>:LINE:COL`
// (sometimes wrapped: `eval at outer (url:N:C), <anonymous>:M:K`). Every
// other frame is bundled library/runtime — skip. Strategy: walk frames
// top-down, take the first containing an `<anonymous>:LINE:COL` token, and
// within that line take the LAST match (innermost position for nested eval).
function firstUserFrame(site: CallSite | null): UserFrame | null {
  if (!site) return null
  for (const raw of site.stack.split("\n").slice(1)) {
    const line = raw.trim()
    if (!line) continue
    const matches = [...line.matchAll(/<anonymous>:(\d+):\d+/g)]
    if (matches.length > 0) {
      return { url: "__spec__", line: parseInt(matches[matches.length - 1]![1]!, 10) }
    }
  }
  return null
}

// ===== Palettes + dim resolver =============================================

const DIM_AUTO_PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#aec7e8", "#ffbb78"] as const

// Graphviz hangs above ~3000 nodes; cap well below.
const MAX_DIAGRAM_NODES = 1500
const TENSOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"] as const
// Color keyed on node identity (group signature, or tensor_input name for
// leaves) rather than tensor id, so corresponding ops share a color across
// loop iterations and across training/inference views.
function nodeColor(node: DagNode, graph: Graph): string {
  let key: string
  if (node.group) {
    key = groupSignature(node.group, graph)
  } else {
    const t = graph.tensors[node.tensorId]!
    const op = t.source !== null ? graph.ops[t.source]! : null
    key = op?.kind === "tensor_input"
      ? `tensor_input:${(op as { name: string }).name}`
      : `tid:${node.tensorId}`
  }
  let h = 0
  for (let i = 0; i < key.length; i++) h = ((h << 5) - h + key.charCodeAt(i)) | 0
  return TENSOR_PALETTE[Math.abs(h) % TENSOR_PALETTE.length]!
}

type ResolvedDim = { name: string; color: string }
type LegendItem = { name: string; desc: string; size: number; color: string }
type DimCollision = { size: number; names: string[] }

// The IR tracks sizes, not names. When multiple declared `dims` share a size
// we use a batch heuristic to resolve the common case (axis 0 of any multi-
// axis tensor is conventionally the batch dim); anything still ambiguous
// slash-joins its candidates and surfaces an explainer below the legend.
// See specs/maybe-future/DimSemantics.md for design discussion.
function buildDimResolver(graph: Graph, dims: readonly DimSpec[] | undefined) {
  const declared = dims ?? []
  const dimsBySize = new Map<number, DimSpec[]>()
  for (const d of declared) {
    const arr = dimsBySize.get(d.size)
    if (arr) arr.push(d)
    else dimsBySize.set(d.size, [d])
  }
  // Identified by name "B" or by `batch` in the description — both conventions
  // are in the wild. Anything else opts out and slash-joins as before.
  const batchDim = declared.find(d => d.name === "B" || /\bbatch\b/i.test(d.desc ?? ""))

  // Each declared dim and each un-declared size draws from the same counter,
  // so colliding sizes (B=H=64) render as visually distinct chips/labels and
  // no auto color ever clashes with a declared one.
  let paletteIdx = 0
  const nextColor = () => DIM_AUTO_PALETTE[paletteIdx++ % DIM_AUTO_PALETTE.length]!
  const dimColors = new Map<DimSpec, string>()
  for (const d of declared) dimColors.set(d, d.color ?? nextColor())
  const autoBySize = new Map<number, string>()
  for (const t of graph.tensors) {
    for (const size of t.shape) {
      if (autoBySize.has(size) || dimsBySize.has(size)) continue
      autoBySize.set(size, nextColor())
    }
  }

  // Which declared dims could label this (size, position) of a rank-`rank`
  // tensor? Heuristic: when batch is among the candidates and the tensor has
  // rank ≥ 2, axis 0 belongs to batch and other axes to the rest.
  const candidatesAt = (size: number, position: number, rank: number): DimSpec[] => {
    const entries = dimsBySize.get(size) ?? []
    if (entries.length <= 1) return entries
    if (batchDim && batchDim.size === size && rank >= 2) {
      return position === 0 ? [batchDim] : entries.filter(e => e !== batchDim)
    }
    return entries
  }

  const resolve = (size: number, position: number, rank: number): ResolvedDim => {
    const candidates = candidatesAt(size, position, rank)
    if (candidates.length === 0) {
      return { name: String(size), color: autoBySize.get(size) ?? "#666" }
    }
    return {
      name: candidates.map(c => c.name).join("/"),
      color: dimColors.get(candidates[0]!)!,
    }
  }

  const legend: LegendItem[] = declared.map(d => ({
    name: d.name, desc: d.desc ?? "", size: d.size, color: dimColors.get(d)!,
  }))

  // A collision needs an explainer only when even the heuristic leaves
  // multiple candidates somewhere. The 2-way batch+other case is fully
  // resolved (one at axis 0, one elsewhere); 3-way or non-batch collisions
  // still slash-join at some position and need the note.
  const collisions: DimCollision[] = []
  for (const [size, entries] of dimsBySize) {
    if (entries.length <= 1) continue
    if (batchDim && batchDim.size === size && entries.length === 2) continue
    collisions.push({ size, names: entries.map(e => e.name) })
  }
  return { resolve, legend, collisions }
}

// ===== Leaf labels ==========================================================
// Only the leaf op kinds the forward DAG can actually contain. tensor_input
// nodes show as `input: <name>`; const_scalar nodes show as `const N`.
// Other op kinds are either filtered out of the forward DAG or appear via
// their group's source-line label (the user's code), not this function.

function describeLibraryOp(op: OpNode): string {
  if (op.kind === "tensor_input") return `input: ${(op as { name: string }).name}`
  if (op.kind === "const_scalar") return `const ${(op as { value: number }).value}`
  return op.kind
}

// ===== Grouping + DAG =======================================================

type Group = { ops: number[]; frame: UserFrame | null; output: number }
type DagNode = { tensorId: number; shape: Shape; group: Group | null; inputs: number[] }

// Leaf-like op kinds get a unique key per op so they each show up as a
// distinct node, even when they share a null source frame. Without this
// the source-line grouping silently absorbs every param / input / scalar
// into one big '<none>' group.
const NEVER_MERGE_KINDS = new Set<string>([
  "param_input",
  "tensor_input",
  "const_scalar",
])

function groupBySourceLine(graph: Graph, included: Set<number>): Group[] {
  const groups: Group[] = []
  let curKey = ""
  let cur: Group | null = null
  for (let i = 0; i < graph.ops.length; i++) {
    if (!included.has(i)) continue
    const op = graph.ops[i]!
    const frame = firstUserFrame(graph.tensors[op.out]!.site)
    const key = NEVER_MERGE_KINDS.has(op.kind)
      ? `${op.kind}#${i}`
      : frame ? `${frame.url}#${frame.line}` : "<none>"
    if (cur && key === curKey) { cur.ops.push(i); cur.output = op.out }
    else { cur = { ops: [i], frame, output: op.out }; groups.push(cur); curKey = key }
  }
  return groups
}

function buildDag(graph: Graph, groups: Group[]): DagNode[] {
  // Tids referenced as inputs by any included op — gates which tensor_inputs become nodes.
  const consumedTids = new Set<number>()
  for (const g of groups) {
    for (const opIdx of g.ops) {
      for (const tid of getOpInputs(graph.ops[opIdx]!)) consumedTids.add(tid)
    }
  }
  // Tensor_inputs first (in op order) so they appear at the top of the DAG, then group outputs.
  const nodes: DagNode[] = []
  const builtTids = new Set<number>()
  const pushNode = (tid: number, group: Group | null) => {
    nodes.push({ tensorId: tid, shape: graph.tensors[tid]!.shape, group, inputs: [] })
    builtTids.add(tid)
  }
  for (const op of graph.ops) {
    if (op.kind === "tensor_input" && consumedTids.has(op.out)) pushNode(op.out, null)
  }
  for (const g of groups) pushNode(g.output, g)

  // Wire inputs: tids consumed by the group but produced outside it.
  for (const node of nodes) {
    if (!node.group) continue
    const inGroupOps = new Set(node.group.ops)
    const seen = new Set<number>()
    for (const opIdx of node.group.ops) {
      for (const tid of getOpInputs(graph.ops[opIdx]!)) {
        if (seen.has(tid)) continue
        seen.add(tid)
        const src = graph.tensors[tid]!.source
        if (src !== null && inGroupOps.has(src)) continue
        if (!builtTids.has(tid)) continue
        node.inputs.push(tid)
      }
    }
  }
  return nodes
}

// ===== Loop detection =======================================================
// Iterations of a for-loop body produce groups with matching signatures in
// the same sequence, so loops are period-k repetition in the signature array.
// const_scalars/tensor_inputs are keyed by value/name (not source line) so
// they match across iterations even though their op indices differ. Greedy
// by coverage k*n; ties favor SMALLER k — tightest period = the natural body
// the user wrote. A 3-group × 8-iter loop has the same coverage as 6×4 and
// 12×2 super-periods; we want the 3×8 view, not "2 iterations" of a body
// that's actually 4 real iterations smushed together. Recurses into iter-0's
// body of each detected run; matching outer signatures guarantee identical
// inner structure across all iterations.

type LoopRun = { start: number; period: number; iterations: number; children: LoopRun[] }

function groupSignature(group: Group, graph: Graph): string {
  if (group.ops.length === 1) {
    const op = graph.ops[group.ops[0]!]!
    if (op.kind === "const_scalar") return `const_scalar:${(op as { value: number }).value}`
    if (op.kind === "tensor_input") return `tensor_input:${(op as { name: string }).name}`
  }
  const frameKey = group.frame ? `${group.frame.url}#${group.frame.line}` : "<none>"
  const opKinds = group.ops.map(i => graph.ops[i]!.kind).join(",")
  return `${frameKey}:${opKinds}`
}

function detectLoops(groups: readonly Group[], graph: Graph): LoopRun[] {
  const signatures = groups.map(g => groupSignature(g, graph))
  return detectLoopsInRange(signatures, 0, signatures.length)
}

function detectLoopsInRange(signatures: readonly string[], lo: number, hi: number): LoopRun[] {
  const loops: LoopRun[] = []
  let i = lo
  while (i < hi) {
    let bestK = 0
    let bestN = 0
    // Period >= 2: single-line iteration bodies merge into one group via
    // groupBySourceLine, so legit period-1 loops can't exist; any period-1
    // match is coincidence.
    for (let k = 2; k <= Math.floor((hi - i) / 2); k++) {
      let n = 1
      while (true) {
        const next = i + n * k
        if (next + k > hi) break
        let match = true
        for (let j = 0; j < k; j++) {
          if (signatures[i + j] !== signatures[next + j]) { match = false; break }
        }
        if (!match) break
        n++
      }
      if (n >= 2 && (k * n > bestK * bestN || (k * n === bestK * bestN && k < bestK))) {
        bestK = k
        bestN = n
      }
    }
    if (bestN >= 2) {
      const children = detectLoopsInRange(signatures, i, i + bestK)
      loops.push({ start: i, period: bestK, iterations: bestN, children })
      i += bestK * bestN
    } else {
      i++
    }
  }
  return loops
}

// Each visible outer iter contains a fresh inner-loop instance with its own
// truncation, so inner savings multiply by visibleIters.
function countHiddenGroups(loops: readonly LoopRun[]): number {
  let total = 0
  for (const loop of loops) {
    const hiddenIters = loop.iterations >= 4 ? loop.iterations - 3 : 0
    const visibleIters = loop.iterations - hiddenIters
    total += hiddenIters * loop.period
    total += visibleIters * countHiddenGroups(loop.children)
  }
  return total
}

// ===== DOT generation =======================================================

function escapeXml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

function wrapForLabel(text: string, maxChars: number): string {
  const tokens: string[] = []
  let buf = ""
  for (const ch of text) {
    buf += ch
    if (ch === " " || ch === "," || ch === "(" || ch === ".") { tokens.push(buf); buf = "" }
  }
  if (buf) tokens.push(buf)
  const lines: string[] = []
  let line = ""
  for (const t of tokens) {
    if (line.length + t.length > maxChars && line.length > 0) { lines.push(line.trimEnd()); line = t.trimStart() }
    else line += t
  }
  if (line.trim()) lines.push(line.trimEnd())
  return lines.map(escapeXml).join("<BR/>")
}

function shapeHtmlLabel(shape: Shape, resolveDim: (size: number, pos: number, rank: number) => ResolvedDim): string {
  if (shape.length === 0) return "[]"
  return "[" + shape.map((d, i) => {
    const r = resolveDim(d, i, shape.length)
    return `<FONT COLOR="${r.color}"><B>${escapeXml(r.name)}</B></FONT>`
  }).join(", ") + "]"
}

function resolveSourceLabel(graph: Graph, node: DagNode, srcCache: Map<string, string[]>): string {
  const t = graph.tensors[node.tensorId]!
  const leafOp = t.source !== null ? graph.ops[t.source]! : null
  if (leafOp && (leafOp.kind === "tensor_input" || leafOp.kind === "param_input")) {
    return describeLibraryOp(leafOp)
  }
  const frame = node.group?.frame
  if (frame) {
    const src = srcCache.get(frame.url)?.[frame.line - 1]
    return src ? src.trim() : `<spec>:${frame.line}`
  }
  return "(unattributed)"
}

function buildDOT(
  graph: Graph,
  dag: DagNode[],
  groups: readonly Group[],
  loops: readonly LoopRun[],
  resolveDim: (size: number, pos: number, rank: number) => ResolvedDim,
  srcCache: Map<string, string[]>,
): string {
  const lines: string[] = [
    "digraph G {",
    "  rankdir=TB",
    '  bgcolor="transparent"',
    "  pad=0.05",
    '  node [shape=box style=filled fillcolor=white fontname="Courier" fontsize=11 margin="0.18,0.09" penwidth=1.4]',
    "  edge [penwidth=1.5]",
    "  nodesep=0.25",
    "  ranksep=0.3",
  ]

  // Stable id per LoopRun. An inner loop has ONE id but possibly N instances
  // (one per containing outer iter); the (loopId, baseShift) pair identifies
  // a specific instance for cluster/ghost naming.
  const loopIds = new Map<LoopRun, number>()
  const assignIds = (ls: readonly LoopRun[]) => {
    for (const l of ls) { loopIds.set(l, loopIds.size); assignIds(l.children) }
  }
  assignIds(loops)

  // Truncate long loops: show first 2 + last iteration, with a "…" cluster
  // for the elided middle. Threshold ≥ 4 — 3 iterations are already brief.
  // Spec stays honest at full T; only the diagram abbreviates. Applied per
  // nesting level independently.
  const isHiddenIter = (loop: LoopRun, iter: number): boolean =>
    loop.iterations >= 4 && iter >= 2 && iter <= loop.iterations - 2

  // groupIdx → outermost-hidden ghost-key, or null when visible at every nest
  // level. Absent for groups outside any loop. "Outermost" (not nearest)
  // matters because a nearest-hidden redirect could land in an inner ghost
  // that itself doesn't render (its containing outer iter is hidden).
  const groupGhost = new Map<number, string | null>()
  const populate = (ls: readonly LoopRun[], baseShift: number, ancestor: string | null) => {
    for (const loop of ls) {
      const lid = loopIds.get(loop)!
      for (let iter = 0; iter < loop.iterations; iter++) {
        const iterStart = loop.start + baseShift + iter * loop.period
        const ghost = ancestor ?? (isHiddenIter(loop, iter) ? `${lid}_${baseShift}` : null)
        for (let g = iterStart; g < iterStart + loop.period; g++) groupGhost.set(g, ghost)
        populate(loop.children, baseShift + iter * loop.period, ghost)
      }
    }
  }
  populate(loops, 0, null)

  const groupIdx = new Map<Group, number>()
  for (let i = 0; i < groups.length; i++) groupIdx.set(groups[i]!, i)

  // Split dag nodes three ways: top-level loose (outside any loop), hidden-
  // in-loop (edge ghost-redirect targets), visible-in-loop (cluster contents).
  // buildDag emits one dag node per group → dagIdxByGroupIdx single-valued.
  const looseDagIdxs: number[] = []
  const dagIdxByGroupIdx = new Map<number, number>()
  const tidGhost = new Map<number, string>()
  for (let i = 0; i < dag.length; i++) {
    const node = dag[i]!
    const gi = node.group ? groupIdx.get(node.group) : undefined
    if (gi === undefined || !groupGhost.has(gi)) { looseDagIdxs.push(i); continue }
    const ghost = groupGhost.get(gi)
    if (ghost) tidGhost.set(node.tensorId, ghost)
    else dagIdxByGroupIdx.set(gi, i)
  }

  const tidToColor = new Map<number, string>()
  for (const node of dag) tidToColor.set(node.tensorId, nodeColor(node, graph))

  const emitCluster = (id: string, label: string, body: () => void) => {
    lines.push(`  subgraph cluster_${id} {`)
    lines.push(`    label="${label}"`)
    for (const a of [`style="dashed"`, `color="#888"`, `fontcolor="#888"`, `fontsize=10`, `fontname="sans-serif"`, `labelloc=t`, `labeljust=l`]) {
      lines.push(`    ${a}`)
    }
    body()
    lines.push(`  }`)
  }

  const emitNode = (dagIdx: number) => {
    const node = dag[dagIdx]!
    const tid = node.tensorId
    const wrapped = wrapForLabel(resolveSourceLabel(graph, node, srcCache), 36)
    const shapeHtml = shapeHtmlLabel(node.shape, resolveDim)
    const color = tidToColor.get(tid)!
    const label = `<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="3" CELLSPACING="0">` +
      `<TR><TD>${wrapped}</TD></TR><TR><TD>${shapeHtml}</TD></TR></TABLE>>`
    lines.push(`    "t${tid}" [label=${label} color="${color}" class="ti-${tid}"]`)
  }

  for (const i of looseDagIdxs) emitNode(i)

  // Within a visible iter, positions covered by a child loop's range are
  // skipped here — they'll emit inside the child's nested cluster instead.
  const emitLoopList = (ls: readonly LoopRun[], baseShift: number) => {
    for (const loop of ls) {
      const gkStr = `${loopIds.get(loop)!}_${baseShift}`
      for (let iter = 0; iter < loop.iterations; iter++) {
        if (isHiddenIter(loop, iter)) continue
        const iterStart = loop.start + baseShift + iter * loop.period
        const innerBaseShift = baseShift + iter * loop.period
        emitCluster(`loop_${gkStr}_iter_${iter}`,
          `iteration ${iter + 1} of ${loop.iterations}`, () => {
            for (let gi = iterStart; gi < iterStart + loop.period; gi++) {
              const inChild = loop.children.some(c => {
                const cs = c.start + innerBaseShift
                return gi >= cs && gi < cs + c.period * c.iterations
              })
              if (inChild) continue
              const di = dagIdxByGroupIdx.get(gi)
              if (di !== undefined) emitNode(di)
            }
            emitLoopList(loop.children, innerBaseShift)
          })
      }
      // Ghost lives in the same nesting context as the visible-iter clusters
      // above, so inner-loop ghosts sit inside their containing outer iter.
      if (loop.iterations >= 4) {
        const startIdx = 2, endIdx = loop.iterations - 2
        emitCluster(`loop_${gkStr}_ghost`,
          `iterations ${startIdx + 1}..${endIdx + 1} of ${loop.iterations}`, () => {
            lines.push(`    "tghost_${gkStr}" [label="…" shape=plaintext fontsize=22 fontcolor="#888" class="ti-ghost-${gkStr}"]`)
          })
      }
    }
  }
  emitLoopList(loops, 0)

  // Both endpoints in the same ghost → drop (internal to the elided range).
  // Ghost-touching edges dedupe by (src,dst) so many-iter fan-in (e.g. a
  // `stack` consuming every state) collapses to a single line.
  type Endpoint = { id: string; cls: string; hidden: boolean }
  const endpoint = (tid: number): Endpoint => {
    const g = tidGhost.get(tid)
    return g === undefined
      ? { id: `t${tid}`, cls: `ti-${tid}`, hidden: false }
      : { id: `tghost_${g}`, cls: `ti-ghost-${g}`, hidden: true }
  }
  const emittedGhostEdges = new Set<string>()
  for (const node of dag) {
    const dst = endpoint(node.tensorId)
    for (const inTid of node.inputs) {
      const src = endpoint(inTid)
      if (src.hidden && dst.hidden && src.id === dst.id) continue
      const color = src.hidden ? "#888" : tidToColor.get(inTid)!
      if (src.hidden || dst.hidden) {
        const key = `${src.id}->${dst.id}`
        if (emittedGhostEdges.has(key)) continue
        emittedGhostEdges.add(key)
      }
      lines.push(`  "${src.id}" -> "${dst.id}" [color="${color}" class="${src.cls} ${dst.cls}"]`)
    }
  }
  lines.push("}")
  return lines.join("\n")
}

// ===== Post-render: hover trace =============================================

function attachHoverTrace(svg: SVGElement): void {
  const elements = svg.querySelectorAll<SVGElement>('[class*="ti-"]')
  const clear = () => { for (const e of svg.querySelectorAll(".hi-highlight")) e.classList.remove("hi-highlight") }
  for (const el of elements) {
    const cls = Array.from(el.classList).find(c => c.startsWith("ti-"))
    if (!cls) continue
    el.addEventListener("mouseenter", () => {
      for (const m of svg.querySelectorAll("." + cls)) m.classList.add("hi-highlight")
    })
    el.addEventListener("mouseleave", clear)
  }
}

// ===== Stats ================================================================
// Trainable scalars only — Adam m/v moments are optimizer bookkeeping, not model params.

function countParams(graph: Graph): number {
  let total = 0
  for (const op of graph.ops) {
    if (op.kind !== "param_input") continue
    total += graph.tensors[op.out]!.shape.reduce((a, b) => a * b, 1)
  }
  return total
}

// Magnitude-scaled formatter. 1 decimal for K/M, 2 for everything bigger.
function formatScaled(n: number, units: readonly string[], base = 1000, sep = ''): string {
  let scaled = n, mag = 0
  while (scaled >= base && mag < units.length - 1) { scaled /= base; mag++ }
  if (mag === 0) return `${n}${sep}${units[0]}`
  return `${scaled.toFixed(mag >= 3 ? 2 : 1)}${sep}${units[mag]}`
}

const formatParamCount = (n: number) => formatScaled(n, ['', 'K', 'M', 'B', 'T'])

// matmul + conv2d only — everything else is single-digit ops per element,
// rounding error next to a 2·M·K·N matmul. Includes conv backward kinds;
// matmul backward is free because it still has kind 'matmul'.
function countFlops(graph: Graph, opIds: Iterable<number>): number {
  let total = 0
  for (const i of opIds) {
    const op = graph.ops[i]!
    if (op.kind === "matmul" || op.kind === "matmul_batched") {
      const K = graph.tensors[op.a]!.shape.at(-1)!
      const outN = graph.tensors[op.out]!.shape.reduce((a, b) => a * b, 1)
      total += 2 * outN * K
    } else if (op.kind === "conv2d") {
      const w = graph.tensors[op.weight]!.shape
      const outN = graph.tensors[op.out]!.shape.reduce((a, b) => a * b, 1)
      total += 2 * outN * w[1]! * w[2]! * w[3]!
    } else if (op.kind === "conv2d_input_grad") {
      const w = graph.tensors[op.weight]!.shape
      const dyN = graph.tensors[op.dy]!.shape.reduce((a, b) => a * b, 1)
      total += 2 * dyN * w[1]! * w[2]! * w[3]!
    } else if (op.kind === "conv2d_weight_grad") {
      const out = graph.tensors[op.out]!.shape
      const dyN = graph.tensors[op.dy]!.shape.reduce((a, b) => a * b, 1)
      total += 2 * dyN * out[1]! * out[2]! * out[3]!
    }
  }
  return total
}

const formatFlops = (n: number) => formatScaled(n, ['', 'K', 'M', 'G', 'T', 'P', 'E'])

// Bytes of forward intermediate tensors. Excludes params, external inputs,
// and shape-only ops (reshape/permute are views). Upper bound — the buffer
// planner aliases non-overlapping lifetimes in practice.
function countActivationBytes(graph: Graph, opIds: Iterable<number>): number {
  let total = 0
  for (const i of opIds) {
    const op = graph.ops[i]!
    if (op.kind === "param_input" || op.kind === "tensor_input") continue
    if (op.kind === "reshape" || op.kind === "permute") continue
    const n = graph.tensors[op.out]!.shape.reduce((a, b) => a * b, 1)
    total += 4 * n
  }
  return total
}

const formatBytes = (n: number) => formatScaled(n, ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB'], 1024, ' ')

// ===== Metric definitions ===================================================
// Four header metrics with definitional hover tooltips.

type MetricData = { display: string; tooltip: string }

function buildMetrics(args: {
  graph: Graph
  fwdOps: Set<number>
  isInference: boolean
  trainGraph: Graph
  kernelCount: number
}): MetricData[] {
  const { graph, fwdOps, isInference, trainGraph, kernelCount } = args
  const params = countParams(trainGraph)
  const flops = countFlops(graph, isInference ? fwdOps : graph.ops.keys())
  const flopsLabel = isInference ? "FLOPs/forward" : "FLOPs/step"
  const activations = countActivationBytes(graph, fwdOps)

  return [
    {
      display: `${formatParamCount(params)} parameters`,
      tooltip: "The model's trainable weights — what the optimizer updates each step.",
    },
    {
      display: `${formatBytes(activations)} activations`,
      tooltip: "Memory for the intermediate tensors created during a forward pass.",
    },
    {
      display: `${formatFlops(flops)} ${flopsLabel}`,
      tooltip: isInference
        ? "Arithmetic operations per forward pass (mostly matrix multiplies)."
        : "Arithmetic operations per training step (mostly matrix multiplies).",
    },
    {
      display: `${kernelCount} kernels`,
      tooltip: "Compiled GPU programs the model dispatches each step.",
    },
  ]
}

// ===== Component ============================================================

type GraphView = { graph: Graph; kernelCount: number; fwdOps: Set<number> }

function viewOf(ir: CompiledIR): GraphView {
  return {
    graph: ir.graph,
    kernelCount: ir.kernels.length,
    fwdOps: forwardReachable(ir.graph, ir.graph.outputs),
  }
}

class IRViewer extends Component {
  // Inference tab is omitted entirely when the spec has no predict function.
  activeTab: "training" | "inference" | "code" | "info" = "training"
  // `status` is the transient loading line. `errorMessage` is the persistent
  // failure state — rendered differently (constrained width, warning color)
  // so it doesn't read like a normal status line.
  status = "Loading…"
  errorMessage: string | null = null
  metrics: MetricData[] = []
  legend: LegendItem[] = []
  dimCollisions: DimCollision[] = []
  modelLabel = ""
  modelDescription = ""

  private readonly defaultSource: string = tb.insight<{ source: string }>()?.source ?? ""
  private specSource: string = this.defaultSource
  private inferring = false

  // Cached per-graph; tab switches don't recompile.
  private viz: any = null
  private train: GraphView | null = null
  private infer: GraphView | null = null
  private resolveDim: ((size: number, pos: number, rank: number) => ResolvedDim) | null = null
  private srcCache: Map<string, string[]> = new Map()
  private canvasEl: HTMLDivElement | null = null
  private textareaEl: HTMLTextAreaElement | null = null

  view() {
    const showViz = this.activeTab === "training" || this.activeTab === "inference"
    const showCode = this.activeTab === "code"
    const showInfo = this.activeTab === "info"
    return divH({ class: "ir-viewer" },
      divH({ class: "header" },
        h1("Diagram a neural network design with AI"),
        p("On the ", strong("Code"), " tab, ", em("Ask the AI"), ": balance a rod, draw a sine wave, recognize handwritten digits…"),
      ),

      divH({ class: "tabs" },
        this.tabBtn("training", "Training graph"),
        this.infer ? this.tabBtn("inference", "Inference graph") : null,
        this.tabBtn("code", "Code"),
        this.tabBtn("info", "How it works"),
      ),

      // Training and Inference share this tab-content; rerender() picks the graph from activeTab.
      divH({ class: "tab-content", style: { display: showViz ? "block" : "none" } },
        divH({ class: "panel" },
          divH({ class: "model-header" },
            this.modelLabel ? h2({ class: "model-title" }, this.modelLabel) : null,
            this.metrics.length > 0
              ? divH({ class: "model-stats" }, ...this.metrics.flatMap((m, i) => {
                  const chip = span({ class: "metric-chip", "data-tooltip": m.tooltip }, m.display)
                  return i === 0 ? [chip] : [span({ class: "metric-sep" }, "·"), chip]
                }))
              : this.errorMessage
                ? divH({ class: "model-error" }, this.errorMessage)
                : divH({ class: "model-stats" }, this.status),
            this.modelDescription ? p({ class: "model-description" }, this.modelDescription) : null,
          ),
          this.legend.length > 0
            ? divH({ class: "diagram-legend" },
                ...this.legend.map(d =>
                  span({ class: "legend-item" },
                    span({ class: "dim-chip", style: { background: d.color } }, d.name),
                    span({ class: "legend-label" }, d.desc ? `${d.desc} = ${d.size}` : `= ${d.size}`),
                  )
                ),
              )
            : null,
          this.dimCollisions.length > 0
            ? divH({ class: "diagram-legend-note" },
                "The IR tracks dim sizes, not names. When dims share a size and can't be resolved by position, the diagram joins the candidates with a slash. Affected: ",
                this.dimCollisions.map(c => `${c.names.join("/")} = ${c.size}`).join(", "),
                ".",
              )
            : null,
          divH({
            class: "canvas",
            onMounted: (el) => {
              this.canvasEl = el as HTMLDivElement
              void this.boot()
            },
          }),
        ),
      ),

      // === Code tab: buttons + textarea ===
      divH({ class: "tab-content", style: { display: showCode ? "block" : "none" } },
        divH({ class: "panel editor-panel" },
          divH({ class: "editor-buttons" },
            button({
              class: "primary-btn",
              disabled: this.inferring,
              onClick: () => this.generateFromDescription(),
            }, this.inferring ? "Asking the AI…" : "Ask the AI"),
            button({ disabled: this.inferring, onClick: () => this.applySpec() }, "Apply"),
            button({ disabled: this.inferring, onClick: () => this.resetSpec() }, "Reset to default"),
          ),
          this.errorMessage ? divH({ class: "model-error code-error" }, this.errorMessage) : null,
          textarea({
            class: "spec-editor",
            spellCheck: false,
            rows: 24,
            onMounted: (el) => {
              this.textareaEl = el as HTMLTextAreaElement
              this.textareaEl.value = this.specSource
            },
          }),
        ),
      ),

      // === How-it-works tab ===
      divH({ class: "tab-content", style: { display: showInfo ? "block" : "none" } },
        divH({ class: "panel info-panel" },
          p("On the ", strong("Code"), " tab, ", em("Ask the AI"), ":"),
          ul(
            li('"balance a rod"'),
            li('"draw a sine wave"'),
            li('"recognize handwritten digits"'),
            li("etc."),
          ),
          p("And out comes a diagram of the network."),
          p("Tensor code can be hard to follow. It shows the ops but not the shapes flowing through them. Each box is a tensor that displays its dimensionality (e.g. [B, H]) and the line of code that produced it; arrows trace dataflow."),
          p("Training and inference have different inputs and outputs because they have different goals. Training takes an input plus the correct answer, and outputs a single number measuring how wrong the model's guess was; that number drives the weight updates. Inference takes only the input and outputs the model's prediction, which is what you wanted in the first place."),
          p("Only the forward is shown. It's the part specific to the architecture (what makes a transformer different from a CNN). The backprop and optimizer are automatic."),
          p("Already have tensorgrad code? Click ", em("Ask the AI"), ", paste in your model, and again, out comes the diagram."),
        ),
      ),
    )
  }

  private tabBtn(key: typeof this.activeTab, label: string) {
    return button({
      class: ["tab-btn", this.activeTab === key && "active"],
      onClick: () => {
        if (this.activeTab === key) return
        this.activeTab = key
        this.update()
        if (key === "training" || key === "inference") this.rerender()
      },
    }, label)
  }

  private async applySpec(): Promise<void> {
    if (!this.textareaEl) return
    const source = this.textareaEl.value
    this.specSource = source
    this.update()
    await this.compileFromSource(source)
    // Flip to the diagram only on success — if compile failed, the user is
    // mid-edit on the Code tab and shouldn't be teleported to an error.
    if (!this.errorMessage) {
      this.activeTab = "training"
      this.update()
    }
  }

  private async resetSpec(): Promise<void> {
    if (!this.textareaEl) return
    this.textareaEl.value = this.defaultSource
    await this.applySpec()
  }

  // tb.infer() opens typebulb's inference modal with data.txt as the prompt.
  private async generateFromDescription(): Promise<void> {
    if (this.inferring) return
    this.inferring = true
    this.update()
    try {
      const result = await tb.infer<{ source: string }>()
      if (result && typeof result.source === "string") {
        this.specSource = result.source
        if (this.textareaEl) this.textareaEl.value = result.source
        this.update()
        await this.compileFromSource(result.source)
        if (!this.errorMessage) {
          this.activeTab = "training"
          this.update()
        }
      }
    } catch (e) {
      this.status = `inference error: ${(e as Error)?.message ?? e}`
      console.error(e)
    } finally {
      this.inferring = false
      this.update()
    }
  }

  private async boot(): Promise<void> {
    await this.compileFromSource(this.specSource)
  }

  private fail(msg: string, e: unknown): void {
    this.errorMessage = msg
    this.update()
    console.error(e)
  }

  private async compileFromSource(source: string): Promise<void> {
    if (this.canvasEl) this.canvasEl.innerHTML = ""
    this.legend = []
    this.dimCollisions = []
    this.metrics = []
    this.train = null
    this.infer = null
    // If the new spec has no predict, the Inference tab disappears — don't strand the user there.
    if (this.activeTab === "inference") this.activeTab = "training"

    this.modelLabel = ""
    this.modelDescription = ""
    this.errorMessage = null
    this.status = "Evaluating spec…"
    this.update()
    let spec: IRSpec
    let cleanedSource: string
    try {
      const ev = evaluateSpec(source)
      spec = ev.spec
      cleanedSource = ev.cleanedSource
    } catch (e) {
      this.fail(`spec error: ${(e as Error)?.message ?? e}`, e)
      return
    }

    // Set before trace so a failure displays under the new spec's metadata,
    // not the previous spec's.
    this.modelLabel = spec.label
    this.modelDescription = spec.description ?? ""
    this.status = `Tracing "${spec.label}"…`
    this.update()
    try {
      if (!this.viz) this.viz = await instance()
      const trainIR = await trace({
        model: spec.model,
        loss: spec.loss,
        inputs: spec.inputs,
        optimizer: spec.optimizer,
      })
      this.train = viewOf(trainIR)

      const { resolve, legend, collisions } = buildDimResolver(trainIR.graph, spec.dims)
      this.resolveDim = resolve
      // Spec frames have URL "__spec__" — inject the cleaned source directly.
      this.srcCache = new Map([["__spec__", cleanedSource.split("\n")]])
      this.legend = legend
      this.dimCollisions = collisions

      // Best-effort inference trace — training graph still useful if it fails.
      try {
        const inferIR = await traceForward({
          model: spec.model,
          forward: spec.predict,
          inputs: spec.predictInputs,
        })
        this.infer = viewOf(inferIR)
      } catch (e) {
        console.warn("inference trace failed:", e)
        this.infer = null
      }

      this.update()
      this.rerender()
    } catch (e) {
      this.fail(`trace error: ${(e as Error)?.message ?? e}`, e)
    }
  }

  private rerender(): void {
    const view = this.activeTab === "inference" && this.infer ? this.infer : this.train
    if (!this.viz || !view || !this.resolveDim || !this.canvasEl) return
    const { graph, fwdOps, kernelCount } = view
    // Forward-only graph: backward and optimizer are universal training
    // machinery, not architecture. Param_input leaves hidden too — Linear/
    // LayerNorm boxes already imply "has weights"; param count is in the
    // header. How-it-works tab covers what's not shown.
    const included = new Set<number>()
    for (const i of fwdOps) {
      if (graph.ops[i]!.kind !== "param_input") included.add(i)
    }
    const groups = groupBySourceLine(graph, included)
    const dag = buildDag(graph, groups)
    const loops = detectLoops(groups, graph)

    // Visible node count after loop-collapse — mirrors buildDOT's isHiddenIter
    // recursively across nested loops.
    const visibleNodes = dag.length - countHiddenGroups(loops)
    this.canvasEl.innerHTML = ""
    if (visibleNodes > MAX_DIAGRAM_NODES) {
      const notice = document.createElement("div")
      notice.className = "model-error"
      notice.textContent =
        `Diagram too large to render (${visibleNodes} visible operations after loop detection). ` +
        `Metrics above are still available — reduce loop iterations or model depth to see the diagram.`
      this.canvasEl.appendChild(notice)
    } else {
      const dot = buildDOT(graph, dag, groups, loops, this.resolveDim, this.srcCache)
      const svg = this.viz.renderSVGElement(dot)
      this.canvasEl.appendChild(svg)
      attachHoverTrace(svg)
    }
    // Params always read from train graph (shared); FLOPs cover the full
    // training step or forward-only inference depending on tab.
    const isInference = view === this.infer
    this.metrics = buildMetrics({
      graph, fwdOps, isInference,
      trainGraph: this.train!.graph,
      kernelCount,
    })
    this.update()
  }
}

new App({ root: new IRViewer(), id: "app" })
```

**styles.css**

```css
/* Theme tokens — light defaults on :root; dark overrides on
   html[data-theme="dark"]. Host doesn't always set the attribute, so the
   :root values must be a fully usable light theme on their own. */
:root {
  color-scheme: light;
  --bg-page:        #ffffff;
  --bg-panel:       #ffffff;
  --bg-canvas:      #fafbfc;
  --bg-editor:      #fafafa;
  --bg-node:        #ffffff;
  --text-primary:   #1a1a1a;
  --text-muted:     #666666;
  --text-on-accent: #ffffff;
  --border:         #dddddd;
  --border-strong:  #888888;
  --accent-bg:      #6366f1;
  --accent-bg-hover:#4f46e5;
}
html[data-theme="dark"] {
  color-scheme: dark;
  --bg-page:        #0c0c0e;
  --bg-panel:       #18181b;
  --bg-canvas:      #0c0c0e;
  --bg-editor:      #1a1a1d;
  --bg-node:        #1f1f24;
  --text-primary:   #e5e5e5;
  --text-muted:     #9ca3af;
  --text-on-accent: #ffffff;
  --border:         #2e2e34;
  --border-strong:  #555560;
  --accent-bg:      #6366f1;
  --accent-bg-hover:#7c7df0;
}

body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 0 1rem 1rem; background: var(--bg-page); color: var(--text-primary); }

.ir-viewer .header { margin: 1rem 0 0.75rem; }
.ir-viewer .header h1 { font-size: 1.25rem; margin: 0; color: var(--text-primary); }
.ir-viewer .header p { color: var(--text-muted); margin: 0.35rem 0 0; font-size: 0.9rem; }

.ir-viewer .panel { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-panel); overflow: hidden; }

/* Tab strip — flat, single-level: Training | Inference | Code | How it
   works. `position: sticky; top: 0` keeps it visible while the user
   scrolls through tall diagrams; without this, the click target can
   leave the viewport when switching between Training and Inference on
   models whose graphs have very different heights. */
.ir-viewer .tabs {
  display: flex;
  gap: 1.2rem;
  margin: 0 0 1rem;
  padding: 0.4rem 0 0;
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
  background: var(--bg-page);
}
.ir-viewer .tab-btn {
  padding: 0.4rem 0.1rem 0.35rem;
  margin-bottom: -1px;
  border: none;
  border-bottom: 2px solid transparent;
  background: transparent;
  font: inherit;
  font-size: 0.9rem;
  cursor: pointer;
  color: var(--text-muted);
  border-radius: 0;
  transition: color 0.15s, border-color 0.15s;
}
.ir-viewer .tab-btn:hover { color: var(--text-primary); }
.ir-viewer .tab-btn.active { color: var(--text-primary); border-bottom-color: var(--accent-bg); font-weight: 600; }

.ir-viewer .info-panel { padding: 0.85rem; line-height: 1.6; font-size: 0.92rem; color: var(--text-primary); }
.ir-viewer .info-panel p { margin: 0 0 0.7rem; max-width: 108ch; }
.ir-viewer .info-panel p:last-child { margin-bottom: 0; }
.ir-viewer .info-panel strong { color: var(--text-primary); }

/* `.panel` overflow: hidden gives the textarea its rounded corners — no need to round it directly. */
.ir-viewer .editor-panel { padding: 0; }
.ir-viewer .spec-editor {
  display: block;
  width: 100%;
  box-sizing: border-box;
  /* 13rem ≈ header + tabs + editor-buttons + body padding. */
  height: calc(100vh - 13rem);
  min-height: 200px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.82rem;
  line-height: 1.4;
  padding: 0.6rem 0.75rem;
  border: none;
  border-radius: 0;
  resize: vertical;
  background: var(--bg-editor);
  color: var(--text-primary);
}
.ir-viewer .spec-editor:focus { outline: 2px solid var(--accent-bg); outline-offset: -2px; }
.ir-viewer .editor-buttons { display: flex; gap: 0.5rem; padding: 0.65rem 0.85rem; border-bottom: 1px solid var(--border); }
.ir-viewer .editor-buttons button {
  padding: 0.35rem 0.85rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--bg-panel);
  font: inherit;
  font-size: 0.85rem;
  cursor: pointer;
  color: var(--text-primary);
}
.ir-viewer .editor-buttons button:hover { border-color: var(--border-strong); }
.ir-viewer .editor-buttons .primary-btn { background: var(--accent-bg); color: var(--text-on-accent); border-color: var(--accent-bg); }
.ir-viewer .editor-buttons .primary-btn:hover { background: var(--accent-bg-hover); border-color: var(--accent-bg-hover); }

.ir-viewer .diagram-legend {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 0.35rem 0.95rem;
  font-size: 0.83rem;
  padding: 0.85rem 1rem;
  background: var(--bg-canvas);
}
.ir-viewer .legend-item { display: inline-flex; align-items: center; gap: 0.3rem; }
.ir-viewer .legend-label { color: var(--text-muted); font-size: 0.82rem; }
.ir-viewer .dim-chip { padding: 0.05rem 0.45rem; border-radius: 4px; color: #fff; font-family: ui-monospace, monospace; font-size: 0.72rem; line-height: 1.4; }
.ir-viewer .diagram-legend-note {
  color: var(--text-muted);
  font-size: 0.75rem;
  line-height: 1.45;
  text-align: center;
  padding: 0 1rem 0.75rem;
  background: var(--bg-canvas);
}

/* Shares the canvas background so header + SVG read as one visual group. */
.ir-viewer .model-header { padding: 0.9rem 1rem 0.75rem; background: var(--bg-canvas); text-align: center; }
.ir-viewer .model-title { font-size: 1.18rem; margin: 0 0 0.3rem; color: var(--text-primary); font-weight: 600; }
.ir-viewer .model-stats {
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 0.6rem;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 0.15rem 0.45rem;
}
.ir-viewer .metric-sep { color: var(--text-muted); user-select: none; }

/* Warm amber rather than alarm red — failures during spec authoring are
   expected/iterative, not catastrophic. Constrained width so a long stack
   trace doesn't stretch full-tab. Monospace because errors quote code/paths. */
.ir-viewer .model-error {
  max-width: 72ch;
  margin: 0 auto 0.6rem;
  padding: 0.55rem 0.85rem;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.78rem;
  line-height: 1.5;
  color: #b86a1f;
  background: rgba(200, 122, 44, 0.08);
  border: 1px solid rgba(200, 122, 44, 0.3);
  border-radius: 6px;
  text-align: left;
  word-break: break-word;
  white-space: pre-wrap;
}
html[data-theme="dark"] .ir-viewer .model-error {
  color: #e0a86b;
  background: rgba(224, 168, 107, 0.1);
  border-color: rgba(224, 168, 107, 0.3);
}
.ir-viewer .code-error {
  max-width: none;
  margin: 0;
  border: none;
  border-radius: 0;
}

/* Below the chip, not above: `.panel` has overflow:hidden (textarea
   corners) which would clip anything extending past its top edge. */
.ir-viewer .metric-chip {
  position: relative;
  padding: 0.05rem 0.2rem;
  border-bottom: 1px dotted var(--border-strong);
  cursor: help;
}
.ir-viewer .metric-chip::after {
  content: attr(data-tooltip);
  position: absolute;
  top: calc(100% + 0.45rem);
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg-panel);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.4rem 0.65rem;
  font-size: 0.78rem;
  font-weight: 400;
  line-height: 1.4;
  white-space: normal;
  width: max-content;
  max-width: 32ch;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s;
  z-index: 20;
  text-align: left;
}
html[data-theme="dark"] .ir-viewer .metric-chip::after {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}
.ir-viewer .metric-chip:hover::after,
.ir-viewer .metric-chip:focus::after {
  opacity: 1;
}
.ir-viewer .model-description { margin: 0 auto; color: var(--text-muted); font-size: 0.9rem; line-height: 1.5; max-width: 72ch; }

.ir-viewer .canvas { background: var(--bg-canvas); width: 100%; padding: 0.85rem 0.5rem; box-sizing: border-box; }
.ir-viewer .canvas > svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }

.ir-viewer .canvas svg [class*="ti-"] { transition: filter 0.1s ease; }
.ir-viewer .hi-highlight { filter: drop-shadow(0 0 4px #f5a623); }

/* SVG content theme — Graphviz hard-codes the node fill as the DOT's
   `fillcolor=white` (emitted as `fill="#ffffff"` on the box `<polygon>`
   for `shape=box`, sometimes on a `<path>` for other shapes) and the
   default text fill as `#000000`. CSS overrides presentation attributes,
   so we re-route them through tokens. Per-dim colored labels carry their
   own non-#000000 fill and aren't matched here. */
.ir-viewer polygon[fill="#ffffff"],
.ir-viewer polygon[fill="white"],
.ir-viewer path[fill="#ffffff"],
.ir-viewer path[fill="white"]   { fill: var(--bg-node); }
.ir-viewer text:not([fill]),
.ir-viewer text[fill="#000000"],
.ir-viewer text[fill="black"]   { fill: var(--text-primary); }
```

**index.html**

```html
<div id="app"></div>
```

**data.txt**

```txt
Memorize an image.
```

**infer.md**

```md
# tensorgrad spec generator

You write **tensorgrad** model specs in TypeScript. The user describes the model they want (in data.txt); you output a working spec, ready for the IR viewer to compile and render.

## Output schema

Return JSON of the form:

    { "source": "<the full TS source of the spec, as a single string>" }

The `source` is real TypeScript — newlines and indentation included. Nothing else: no commentary, no markdown, no "scratch" calls (unused-result ops still trace into the graph).

## What the source must contain

Every tensorgrad symbol is in scope at eval time — call any op (`narrow`, `clamp`, `softmax`, etc.) directly in the body without importing. The `import { ... } from 'tensorgrad'` line at the top is optional, stripped before eval, and exists only as a hint for human readers. Don't import inline, don't `require()` — there's no need.

The source MUST end with:

    export const irSpec = {
      label: '...',
      description: '...',
      model: new MyModel(),
      loss: lossFn,
      inputs,
      optimizer,
      predict: predictFn,
      predictInputs: { /* same shape as `inputs` but without target/label fields */ },
      dims: [ { size: N, name: 'B', desc: '...' }, ... ],
    }

- **`label`**: short human title (4–8 words). Example: `'Transformer for 2-digit addition'`, `'NeRF-tiny (image INR)'`, `'MLP fits sin(x)'`. This is what shows as the visualization's headline.

- **`description`**: 2–3 sentences explaining the architecture in plain English. Cover the model type, the task, key design choices (positional encoding, causal masking, normalization, gating, etc.), and the relevant hyperparameter scales (batch size, depth, width, head count). This is what someone seeing only the rendered graph would need to know to understand what they're looking at.

- **`model`**, **`loss`**, **`inputs`**, **`optimizer`**: ingredients for tracing the training graph. Same shape as you'd pass to tensorgrad's `compile({ model, loss, inputs, optimizer })` — the bulb traces these directly without GPU work via `trace()`.

- **`predict`**: forward function returning the network's *prediction* (logits, regression output, generated sample) rather than a scalar loss. Same shape as a forward but without the loss tail. Traced separately for the Inference tab via `traceForward({ model, forward: predict, inputs: predictInputs })` — the model is shared with training.

- **`predictInputs`**: same shape as `inputs` but without target/label fields (a classifier's `{ x, y }` becomes `{ x }`). Use the same concrete batch size as `inputs`.

- **`dims`**: metadata labeling distinct integer dim sizes in the IR. One entry per architecturally meaningful dim (batch, hidden width, sequence length, head count, vocab, etc.). Use single-letter `name` for canonical dims (B, T, D, H) and a concise `desc`.

## Tensorgrad conventions (canonical, copy these)

**Model class** — extends `Module`, fields are layer modules (`Linear`, `LayerNorm`, `Conv2d`, `Embedding`) or `this.param([shape])` tensors:

    class MLP extends Module {
      l1 = new Linear(IN, HIDDEN)
      l2 = new Linear(HIDDEN, OUT)
    }

**Forward function** — free function taking `(p: Model, inputs: { ... })`, returning the output tensor:

    function modelFwd(p: MLP, x: Tensor): Tensor {
      return p.l2.fwd(relu(p.l1.fwd(x)))
    }

**Loss function** — destructures named inputs, returns a scalar tensor:

    function lossFn(p: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
      return mean(square(sub(modelFwd(p, x), y)))
    }

**Token-classification loss** — for sequence models where `logits: [B, T, V]` and `targets: [B, T]` (i32). Collapse both with `reshape` to `[-1, V]` / `[-1]`:

    function lossFn(p: GPT, { tokens, targets }: { tokens: Tensor; targets: Tensor }): Tensor {
      const logits = modelFwd(p, tokens)                  // [B, T, V]
      return crossEntropy(
        reshape(logits,  [-1, VOCAB]),                    // [B*T, V]
        reshape(targets, [-1]),                           // [B*T]
      )
    }

**Learned positional embedding** — pair an `Embedding(T, D)` with `arange(T)` for position indices. The `[T, D]` result broadcasts over batch when added to token embeddings:

    const posE = pos_emb.fwd(arange(T))                  // [T, D]
    let h = add(tok_emb.fwd(tokens), posE)               // [B, T, D]

**Sinusoidal time/position embedding** — encode a continuous index `t: [B]` (declare `f32`) as concatenated sin+cos of log-spaced frequencies. `arange(half, 'f32')` because i32 won't multiply with the f32 freqs:

    const half = D / 2
    const freqs = exp(mul(arange(half, 'f32'), -Math.log(10000) / half))   // [half]
    const angles = mul(reshape(t, [B, 1]), reshape(freqs, [1, half]))      // [B, half]
    const emb = concat([sin(angles), cos(angles)], -1)                     // [B, D]

**Predict function** — same model forward as the loss uses internally, but without the loss tail. Returns the network's prediction directly (logits, regression output, etc.):

    function predictFn(p: MLP, { x }: { x: Tensor }): Tensor {
      return modelFwd(p, x)
    }

    const predictInputs = { x: [BATCH, FEAT_DIM] } as const

**Inputs declaration** — object literal with `as const`. Tuple shape defaults to f32; non-f32 uses object form:

    const inputs = {
      x: [BATCH, FEAT_DIM],
      y: { shape: [BATCH], dtype: 'i32' },
    } as const

**Optimizer config** — discriminated union, `as const`:

    const optimizer = { kind: 'adamw', lr: 1e-3, weightDecay: 0.01 } as const
    // or { kind: 'adam', lr: ... }
    // or { kind: 'sgd', lr: ..., momentum: 0.9 }
    // `lr` may also be a schedule: `lr.linear({ peak, final, steps })` etc.

**Recurrent state in unrolled loops** — initialize with `zeros(shape)`, slice each timestep with `narrow`, collect per-step outputs into an array, `stack` at the end. `stack` *adds* the T axis; the per-step `h` stays `[B, H]` (no reshape):

    let h = zeros([B, H])
    const outs: Tensor[] = []
    for (let t = 0; t < T; t++) {
      const xt = reshape(narrow(x, 1, t, 1), [B, D])
      h = tanh(add(m.ih.fwd(xt), m.hh.fwd(h)))
      outs.push(h)
    }
    const seq = stack(outs, 1)   // [B, T, H]

**1D conv via Conv2d** — no `Conv1d` primitive; reshape sequence data `[B, C, T]` to `[B, C, 1, T]` and use a `[1, K]` kernel:

    conv = new Conv2d(C_in, C_out, [1, K], { padding: [0, K - 1] })
    // in the forward:
    const x4 = reshape(x, [B, C_in, 1, T])
    const y  = reshape(conv.fwd(x4), [B, C_out, T_out])

## Available tensorgrad symbols (already in scope at eval time)

**Layer modules** — instantiate as class fields; each instance exposes `.fwd(x)`:

    new Linear(inDim, outDim, { bias?, init?, decay? })
    new LayerNorm(dim, { eps?, bias?, decay? })
    new RMSNorm(dim, { eps?, decay? })
    new Embedding(vocab, dim, { init?, decay? })
    new Conv2d(inC, outC, k, { stride?, padding?, bias?, init?, decay? })  // dense only; no groups; k/stride/padding accept int or [kH, kW]
**Compile/lifecycle**: `Module`, `compile`, `lr`, `init`.
**Losses**: `crossEntropy(logits, targets, { reduction? })`, `nllLoss(logProbs, targets, { reduction? })` — default reduction is mean; use `'none'` for per-position output.
**Arithmetic**: `add`, `sub`, `mul`, `div`, `min`, `max` — each takes `(Tensor, Tensor)` or `(Tensor, number)`.
**Comparison**: `less`, `greater` (same scalar overload as arithmetic), `where(cond, ifTrue, ifFalse)`.
**Unary math**: `sqrt`, `rsqrt`, `log`, `exp`, `neg`, `abs`, `square`, `sin`, `cos`.
**Activations**: `relu`, `tanh`, `sigmoid`, `gelu`, `silu`, `leakyRelu(x, alpha?)` (default alpha 0.01), `softplus` (numerically-stable `log(1 + exp(x))`).
**Clamping**: `clamp(x, lo, hi)` — `lo` and `hi` are numbers.
**Reductions**: `mean(x, axis?, { keepDims? })`, `sum(x, axis?, { keepDims? })`, `argmax`, `argmin`.
**Shape**: `reshape(x, [dims])` (one `-1` allowed, inferred from total size — use `reshape(x, [B, -1])` or `reshape(x, [-1])` instead of a `flatten` op; tensorgrad doesn't have one), `permute`, `swapAxes` (= PyTorch `transpose`).
**Linear algebra**: `matmul`.
**Indexing**: `oneHot(idx, depth)`, `arange(n, dtype?)` (default `i32`; pass `'f32'` for float math like sin/cos), `embedding(table, indices)`, `takeAlongAxis(input, indices, axis)` (= PyTorch `gather`).
**Const-tensor builders**: `zeros(shape, dtype?)`, `ones(shape, dtype?)` — default `f32`. The full set is `randn` / `arange` / `oneHot` / `zeros` / `ones`; no `full`, `eye`, `linspace`, `tril`, or `like`-variants.
**Slicing/structural**: `narrow(t, axis, start, len)`, `concat([a, b, ...], axis)`, `stack([a, b, ...], axis)`, `split(t, [size1, size2, ...], axis)`.
**Fused ML**: `softmax`, `logSoftmax`, `softmaxCausal`, `whereCausal`.
**Attention layout**: `splitHeads(x, nHeads)`, `mergeHeads(x)`.
**Conv/pool**: `conv2d(input, weight, { stride?, padding? })`, `maxPool2d(x, k, { stride?, padding? })`, `nearestUpsample2d(x, factor)`.
**Stochastic/grad**: `dropout(x, p)`, `randn(shape)`, `stopGradient(x)` (= PyTorch `.detach`), `capture(name, t)`.

## Gotchas

- **`Tensor` has no methods.** Every operation is a free function from the symbols list, applied as `op(x, ...)` — e.g. `reshape(x, [B, -1])`, `sum(x, axis)`, `swapAxes(x, -2, -1)`, `narrow(x, axis, start, len)`.
- **Shape tracing — the most common compile error.** Every `Linear`/`matmul` input dim must equal the last dim of the upstream tensor. Don't pattern-match by layer name: `attn_proj = Linear(D, D)` and `mlp_proj = Linear(4*D, D)` look parallel but have different shapes because attention preserves `D` while the MLP expands then contracts. Watch any layer immediately after a dim-changing op (`Linear(D, ≠D)`, `splitHeads`, `reshape`, `narrow`, `embedding`): the next layer's input dim is the NEW last dim, not the original. Expansion-then-contraction pairs (MLP up/down, autoencoder bottleneck) always swap dim orders by design. Before returning, mentally run the forward once: every `Linear`'s first arg = the last dim of what feeds it; every `reshape` preserves total element count; every broadcast is trailing-suffix compatible.
- **Operators have no PyTorch-style optional flags.** The symbols list is the full signature. `matmul(a, b)` is two args; to transpose the rhs, write `matmul(a, swapAxes(b, -2, -1))`.
- **Module internals aren't public.** Call `embedding.fwd(idx)`, never `embedding(module.weight, idx)`. The free function `embedding(table, indices)` is for raw `this.param([V, D])` tables; on an `Embedding` instance, use `.fwd`.
- **`reshape` doesn't transpose.** It reinterprets memory layout — to reorder axes use `permute(x, [perm])` or `swapAxes(x, a, b)`. Same total element count means `reshape` won't error on a wrong-axes pass, so this is a silent correctness bug.
- **`matmul(x, codebook)` against `[N, D]` raw params errors on inner dims.** `Linear` weights are `[in, out]` so direct `matmul(x, W)` works for projections, but raw codebook / memory / prototype params shaped `[N, D]` need `matmul(x, swapAxes(codebook, -1, -2))`. Common when porting "score query against learned vectors" patterns from PyTorch.
- **Static shapes only** — every dim is a compile-time `const` in your code, not a value read from a tensor.
- **Pass raw logits to `crossEntropy`** — it fuses log-softmax internally. Don't apply `logSoftmax` first.
- **Loss must be scalar** (rank-0). Use `mean`/`sum` to reduce.
- **`splitHeads(x, nHeads)`** reshapes one tensor: `[B, T, D] → [B, H, T, D/H]`. For Q/K/V, use three independent `Linear(D, D)` projections and call `splitHeads` on each.
- **No GQA — use full Q/K/V projections (all `H` heads).** `matmul` batch dims must match exactly (no size-1 broadcasting); matching multi-dim batches like `[B, H, T, D]` work fine — don't flatten to `[B*H, ...]` defensively.
- **No RoPE — use learned `pos_emb`.** Rotate-half from scratch is fiddly; the diagram shows the same architecture either way.
- **No hand-rolled fast-GELU — use the `gelu` primitive.** `mul(x, sigmoid(mul(x, 1.702)))` is the approximation; dropping the sigmoid silently collapses the MLP to linear and the trace passes.
- **Use `new RMSNorm(D)`, don't hand-roll.** Already in the symbols list; a hand-rolled version next to `LayerNorm` reads as architecturally inconsistent.
- **Use a loop for repeated blocks, not manual unrolling.** `layers = []; for (let i = 0; i < N_LAYERS; i++) layers.push(new Block())` + `for (const layer of m.layers) ...`. Manually-named fields (`layer0`, `layer1`, …) make `N_LAYERS` decorative — editing the const won't change architecture.
- **Attention scaling**: multiply scores by `1 / Math.sqrt(D_HEAD)` before `softmaxCausal`.
- **`stack` adds a new axis; `concat` joins an existing one.** `stack([a, b, …], axis)` of N tensors of shape `[B, H]` gives `[B, N, H]` (or wherever `axis` puts it). Don't `reshape(h, [B, 1, H])` and then `stack(outs, 1)` — that double-adds the axis, producing `[B, T, 1, H]` instead of `[B, T, H]`. The bug runs: downstream Linear/reshape silently swallow the extra size-1 dim until something stricter (MSE `sub`, broadcasting) catches it.
- **No `groups` on Conv2d, no `Conv1d`.** Conv2d is dense conv only.
- **No in-forward param creation.** `this.param` is class-field only; forwards are pure tensor compositions over the already-built module.
- **No `scan` / `cumsum`.** Unroll trace-time loops; keep T small.

## Reference example (small)

Use this as a structural template. Substitute the model logic for the user's request.

    import {
      Module, Linear,
      mul, sub, mean, relu,
      type Tensor,
    } from 'tensorgrad'

    const HIDDEN = 64
    const B = 256

    class MLP extends Module {
      l1 = new Linear(1, HIDDEN)
      l2 = new Linear(HIDDEN, 1)
    }

    function modelFwd(p: MLP, x: Tensor): Tensor {
      return p.l2.fwd(relu(p.l1.fwd(x)))
    }

    function lossFn(p: MLP, { x, y }: { x: Tensor; y: Tensor }): Tensor {
      const d = sub(modelFwd(p, x), y)
      return mean(mul(d, d))
    }

    function predictFn(p: MLP, { x }: { x: Tensor }): Tensor {
      return modelFwd(p, x)
    }

    const inputs = { x: [B, 1], y: [B, 1] } as const
    const predictInputs = { x: [B, 1] } as const
    const optimizer = { kind: 'adam', lr: 0.005 } as const

    export const irSpec = {
      label: 'MLP fits sin(x)',
      description: 'A 2-layer MLP that approximates y = sin(x) over [-π, π]. Single scalar input, single scalar output, 64 hidden units with ReLU. MSE loss, Adam at 5e-3, batches of 256 random samples.',
      model: new MLP(),
      loss: lossFn,
      inputs,
      optimizer,
      predict: predictFn,
      predictInputs,
      dims: [
        { size: B,      name: 'B', desc: 'batch' },
        { size: HIDDEN, name: 'H', desc: 'hidden' },
      ],
    }

If the user pastes their own (possibly broken) code, fix it to match these conventions while preserving their intent. If the request is ambiguous, default to small (batch 64–256, hidden 64–128, Adam at 1e-3 or 5e-3). For explicit scale requests, match it — the bulb traces without GPU, so GPT-2-scale (D≈1024, 12 layers), Llama/GPT-3-scale (D≈4096+, 32+ layers, B=1), and GPT-4-scale (D≈16K+, 80+ layers) all inspect in seconds; don't shrink toward tutorial defaults.
```

**insight.json**

```json
{
  "source": "import {\n  Module, Linear,\n  sub, mul, gelu, mean, square,\n  type Tensor,\n} from 'tensorgrad'\n\nconst B = 256\nconst D = 32\nconst HIDDEN = 128\nconst N_STEPS = 12\nconst STEP_SIZE = 0.1\n\nclass Denoiser extends Module {\n  l1 = new Linear(D, HIDDEN)\n  l2 = new Linear(HIDDEN, HIDDEN)\n  l3 = new Linear(HIDDEN, D)\n}\n\n// One step of the denoiser: noisy x -> predicted noise.\nfunction denoise(m: Denoiser, x: Tensor): Tensor {\n  return m.l3.fwd(gelu(m.l2.fwd(gelu(m.l1.fwd(x)))))\n}\n\n// Training: predict the noise added to a clean signal (single forward pass).\nfunction lossFn(m: Denoiser, { clean, noisy }: { clean: Tensor; noisy: Tensor }): Tensor {\n  const predNoise = denoise(m, noisy)\n  const trueNoise = sub(noisy, clean)\n  return mean(square(sub(predNoise, trueNoise)))\n}\n\n// Inference: start from pure noise and iterate N_STEPS Euler-style denoising steps.\nfunction predictFn(m: Denoiser, { xInit }: { xInit: Tensor }): Tensor {\n  let x = xInit\n  for (let t = 0; t < N_STEPS; t++) {\n    x = sub(x, mul(denoise(m, x), STEP_SIZE))\n  }\n  return x\n}\n\nconst inputs = {\n  clean: [B, D],\n  noisy: [B, D],\n} as const\n\nconst predictInputs = { xInit: [B, D] } as const\nconst optimizer = { kind: 'adamw', lr: 1e-3, weightDecay: 0.01 } as const\n\nexport const irSpec = {\n  label: 'Iterative diffusion denoiser',\n  description: 'A 3-layer MLP trained to predict the noise added to a 32-d signal. Training is a single forward pass plus MSE on the predicted noise. Inference is the same MLP applied iteratively over 12 Euler-style denoising steps starting from pure noise.',\n  model: new Denoiser(),\n  loss: lossFn,\n  inputs,\n  optimizer,\n  predict: predictFn,\n  predictInputs,\n  dims: [\n    { size: B,      name: 'B', desc: 'batch' },\n    { size: D,      name: 'D', desc: 'signal dim' },\n    { size: HIDDEN, name: 'H', desc: 'denoiser hidden' },\n  ],\n}\n"
}
```

**config.json**

```json
{
  "description": "Diagram a neural network design with AI. Describe a model in plain English and see its architecture as a diagram. Built on tensorgrad.",
  "inference": {
    "title": "Diagram a Neural Network Design",
    "dataTitle": "What do you want the neural network to do?",
    "submitTitle": "Diagram It"
  },
  "dependencies": {
    "tensorgrad": "^0.1.6",
    "@viz-js/viz": "^3.27.0",
    "sucrase": "^3.35.0",
    "domeleon": "^0.6.0"
  }
}
```