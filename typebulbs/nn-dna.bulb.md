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
  splitHeads, mergeHeads, rope,
  matmul,
  oneHot, arange, embedding, takeAlongAxis,
  zeros, ones,
  narrow, concat, stack, split,
  softmax, logSoftmax, softmaxCausal, whereCausal, categorical,
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
  splitHeads, mergeHeads, rope,
  matmul,
  oneHot, arange, embedding, takeAlongAxis,
  zeros, ones,
  narrow, concat, stack, split,
  softmax, logSoftmax, softmaxCausal, whereCausal, categorical,
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
// other frame is bundled library/runtime — skip. Within each anonymous line
// the LAST match is the innermost position (nested eval).
//
// Returned chain is outermost-first: index 0 is the entry frame (the spec's
// top-level forward), increasing index is deeper user code. Library frames
// between user frames are filtered out — only frames in the eval'd spec source
// appear in the chain.
function userFrameChain(site: CallSite | null): UserFrame[] {
  if (!site) return []
  const innermostFirst: UserFrame[] = []
  for (const raw of site.stack.split("\n").slice(1)) {
    const line = raw.trim()
    if (!line) continue
    const matches = [...line.matchAll(/<anonymous>:(\d+):\d+/g)]
    if (matches.length > 0) {
      innermostFirst.push({ url: "__spec__", line: parseInt(matches[matches.length - 1]![1]!, 10) })
    }
  }
  return innermostFirst.reverse()
}


// ===== Palettes + dim resolver =============================================

const DIM_AUTO_PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#aec7e8", "#ffbb78"] as const

// Graphviz hangs above ~3000 nodes; cap well below.
const MAX_DIAGRAM_NODES = 1500
const TENSOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"] as const
// Color keyed on (source line, op kinds) for leaves and on input name for
// data sources — so corresponding ops share a color across loop iterations
// and between training/inference views.
function nodeColor(node: DagNode, graph: Graph): string {
  let key: string
  if (node.leaf) {
    const opKinds = node.leaf.ops.map(i => graph.ops[i]!.kind).join(",")
    key = `${node.leaf.frame.url}#${node.leaf.frame.line}:${opKinds}`
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

// ===== Grouping (tree) ======================================================
//
// Each op's user-frame chain is walked into a tree where each interior node
// is a CallCluster (the call site of a sub-call) and each leaf is a run of
// consecutive ops sharing the same source line. Clusters are pure visual
// wrappers — they render as labeled rectangles around their contents but
// don't appear as nodes in the DAG.

type LeafGroup = {
  kind: 'leaf'
  frame: UserFrame
  ops: number[]
  output: number             // tid of the leaf's last op (= node tid in the diagram)
  internalTids: Set<number>  // tids produced by ops in this leaf
}

type CallCluster = {
  kind: 'cluster'
  frame: UserFrame           // the call-site source line in the parent frame
  children: TreeNode[]       // leaves and nested clusters in trace order
}

type TreeNode = LeafGroup | CallCluster

// Leaf-like op kinds get a unique leaf each (no merging with siblings on the
// same source line). Without this every param / input / scalar collapses
// into one giant "<none>" leaf at the top of the diagram.
const NEVER_MERGE_KINDS = new Set<string>([
  "param_input",
  "tensor_input",
  "const_scalar",
])

// Walks ops in trace order, building the cluster tree. Sub-call invocation
// boundaries use a first-seen-deviated heuristic: within a parent cluster,
// when a sub-call's call-site line returns after other lines have been
// visited, treat as a new invocation. Multi-line sub-call bodies hit this
// correctly; single-line bodies (rare) merge consecutive invocations
// together — acceptable trade.
function buildGroupTree(graph: Graph, included: Set<number>): CallCluster {
  const root: CallCluster = {
    kind: 'cluster',
    frame: { url: '__spec__', line: 0 },
    children: [],
  }
  const stack: CallCluster[] = [root]
  // Per-parent tracker: which sub-call line was first seen since this parent
  // became active, and whether we've since drifted to a different line.
  const tracker = new Map<CallCluster, { firstChildLine: number | undefined; deviated: boolean }>()
  tracker.set(root, { firstChildLine: undefined, deviated: false })

  for (let opIdx = 0; opIdx < graph.ops.length; opIdx++) {
    if (!included.has(opIdx)) continue
    const op = graph.ops[opIdx]!
    const chain = userFrameChain(graph.tensors[op.out]!.site)
    // Chain layout: chain[0..length-2] are call-site frames (one per nested
    // cluster); chain[length-1] is the leaf line where the op was issued.
    const clusterFrameCount = Math.max(0, chain.length - 1)

    let matchDepth = 0
    while (matchDepth < clusterFrameCount && matchDepth + 1 < stack.length) {
      const existing = stack[matchDepth + 1]!
      if (existing.frame.line !== chain[matchDepth]!.line) break
      const parent = stack[matchDepth]!
      const t = tracker.get(parent)!
      if (t.firstChildLine === chain[matchDepth]!.line && t.deviated) {
        // First-seen sub-call line returned after deviation → new invocation.
        break
      }
      matchDepth++
    }

    while (stack.length > matchDepth + 1) {
      const popped = stack.pop()!
      tracker.delete(popped)
    }

    while (stack.length - 1 < clusterFrameCount) {
      const depth = stack.length - 1
      const parent = stack[depth]!
      const frame = chain[depth]!
      const t = tracker.get(parent)!
      if (t.firstChildLine === undefined) {
        t.firstChildLine = frame.line
      } else if (t.firstChildLine !== frame.line) {
        t.deviated = true
      } else if (t.deviated) {
        // firstChildLine returned after deviation → fresh invocation under parent.
        t.firstChildLine = frame.line
        t.deviated = false
      }
      const cluster: CallCluster = {
        kind: 'cluster',
        frame,
        children: [],
      }
      parent.children.push(cluster)
      stack.push(cluster)
      tracker.set(cluster, { firstChildLine: undefined, deviated: false })
    }

    // Add op as a leaf, merging with previous leaf if same source line.
    const inner = stack[stack.length - 1]!
    const leafFrame = chain[chain.length - 1] ?? inner.frame
    const isMergeable = !NEVER_MERGE_KINDS.has(op.kind)
    const lastChild = inner.children[inner.children.length - 1]
    if (
      isMergeable && lastChild && lastChild.kind === 'leaf' &&
      lastChild.frame.line === leafFrame.line &&
      !NEVER_MERGE_KINDS.has(graph.ops[lastChild.ops[lastChild.ops.length - 1]!]!.kind)
    ) {
      lastChild.ops.push(opIdx)
      lastChild.output = op.out
      lastChild.internalTids.add(op.out)
    } else {
      inner.children.push({
        kind: 'leaf',
        frame: leafFrame,
        ops: [opIdx],
        output: op.out,
        internalTids: new Set([op.out]),
      })
    }
  }

  return root
}

// In-order list of every leaf in the tree. Each leaf is one visible node in
// the diagram; clusters are pure visual wrappers (subgraph rectangles) that
// don't themselves appear as nodes in the DAG.
function collectLeaves(tree: CallCluster): LeafGroup[] {
  const out: LeafGroup[] = []
  const walk = (node: TreeNode) => {
    if (node.kind === 'leaf') out.push(node)
    else for (const c of node.children) walk(c)
  }
  for (const c of tree.children) walk(c)
  return out
}

type DagNode = {
  tensorId: number
  shape: Shape
  leaf: LeafGroup | null
  inputs: number[]
}

function buildDag(graph: Graph, leaves: readonly LeafGroup[]): DagNode[] {
  // Tids referenced as inputs by any leaf's ops — gates which tensor_inputs become nodes.
  const consumedTids = new Set<number>()
  for (const l of leaves) {
    for (const opIdx of l.ops) {
      for (const tid of getOpInputs(graph.ops[opIdx]!)) consumedTids.add(tid)
    }
  }
  const nodes: DagNode[] = []
  const builtTids = new Set<number>()
  // tensor_input leaves render before any user-op leaf (they're the data
  // sources). They appear as nodes with `leaf: null` and no source frame.
  for (const op of graph.ops) {
    if (op.kind === "tensor_input" && consumedTids.has(op.out)) {
      nodes.push({ tensorId: op.out, shape: graph.tensors[op.out]!.shape, leaf: null, inputs: [] })
      builtTids.add(op.out)
    }
  }
  for (const l of leaves) {
    nodes.push({ tensorId: l.output, shape: graph.tensors[l.output]!.shape, leaf: l, inputs: [] })
    builtTids.add(l.output)
  }
  // Wire inputs: external tids consumed by a leaf's ops (not produced inside).
  for (const node of nodes) {
    if (!node.leaf) continue
    const seen = new Set<number>()
    for (const opIdx of node.leaf.ops) {
      for (const tid of getOpInputs(graph.ops[opIdx]!)) {
        if (seen.has(tid)) continue
        seen.add(tid)
        if (node.leaf.internalTids.has(tid)) continue
        if (!builtTids.has(tid)) continue
        node.inputs.push(tid)
      }
    }
  }
  return nodes
}

// ===== Loop detection =======================================================
// Iterations of a for-loop body produce TreeNodes (clusters or leaves) with
// matching signatures in the same sequence. Loops are period-k repetition in
// the signature array. Greedy by coverage k*n; ties favor SMALLER k — the
// tightest period is the natural body the user wrote. Each cluster's children
// array is analyzed independently; matching outer-cluster signatures guarantee
// identical inner structure across iterations.

type LoopRun = { start: number; period: number; iterations: number; children: LoopRun[] }

function nodeSignature(node: TreeNode, graph: Graph): string {
  if (node.kind === 'leaf') {
    if (node.ops.length === 1) {
      const op = graph.ops[node.ops[0]!]!
      if (op.kind === "const_scalar") return `const_scalar:${(op as { value: number }).value}`
      if (op.kind === "tensor_input") return `tensor_input:${(op as { name: string }).name}`
    }
    const frameKey = `${node.frame.url}#${node.frame.line}`
    const opKinds = node.ops.map(i => graph.ops[i]!.kind).join(",")
    return `leaf:${frameKey}:${opKinds}`
  }
  // Sibling clusters with same call site AND structurally-identical bodies
  // produce matching signatures, which is what loop detection matches on.
  const childSigs = node.children.map(c => nodeSignature(c, graph)).join("|")
  return `cluster:${node.frame.url}#${node.frame.line}:[${childSigs}]`
}

function detectLoops(children: readonly TreeNode[], graph: Graph): LoopRun[] {
  const signatures = children.map(n => nodeSignature(n, graph))
  return detectLoopsInRange(signatures, 0, signatures.length)
}

function detectLoopsInRange(signatures: readonly string[], lo: number, hi: number): LoopRun[] {
  const loops: LoopRun[] = []
  let i = lo
  while (i < hi) {
    let bestK = 0
    let bestN = 0
    // Period >= 2: single-line iteration bodies merge into one leaf via
    // tree building, so legit period-1 loops can't exist; any period-1
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

function frameSourceText(frame: UserFrame, srcCache: Map<string, string[]>): string {
  const src = srcCache.get(frame.url)?.[frame.line - 1]
  return src ? src.trim() : `<spec>:${frame.line}`
}

// Strip a trailing "// [B, T, V]" comment when its dims match the node's
// inferred shape — the colored shape badge below already shows the same info.
// Anchored to end-of-line so prefixed/suffixed comments ("// shape: [...]",
// "// [...] later flattened") are left intact.
function stripRedundantShapeComment(src: string, shapeNames: readonly string[]): string {
  const m = src.match(/^(.*?)\s*\/\/\s*\[([^\]]*)\]\s*$/)
  if (!m) return src
  const commentDims = m[2]!.split(",").map(s => s.trim()).filter(s => s.length > 0)
  if (commentDims.length !== shapeNames.length) return src
  for (let i = 0; i < commentDims.length; i++) {
    if (commentDims[i] !== shapeNames[i]) return src
  }
  return m[1]!.trimEnd()
}

function buildDOT(
  graph: Graph,
  tree: CallCluster,
  dag: DagNode[],
  resolveDim: (size: number, pos: number, rank: number) => ResolvedDim,
  srcCache: Map<string, string[]>,
): string {
  const lines: string[] = [
    "digraph G {",
    "  rankdir=TB",
    "  compound=true",
    '  bgcolor="transparent"',
    "  pad=0.05",
    '  node [shape=box style=filled fillcolor=white fontname="Courier" fontsize=11 margin="0.18,0.09" penwidth=1.4]',
    "  edge [penwidth=1.5]",
    "  nodesep=0.25",
    "  ranksep=0.3",
  ]

  const tidToColor = new Map<number, string>()
  for (const node of dag) tidToColor.set(node.tensorId, nodeColor(node, graph))

  // Truncate long loops: show first 2 + last iteration, with a "…" cluster
  // for the elided middle. Threshold ≥ 4 — 3 iterations are already brief.
  // Applied per nesting level independently.
  const isHiddenIter = (loop: LoopRun, iter: number): boolean =>
    loop.iterations >= 4 && iter >= 2 && iter <= loop.iterations - 2

  // Walk the tree to populate ghost-tid map (tid → ghost cluster ID for leaf
  // tids inside hidden iterations at any level). Computed before emission so
  // edges can redirect through ghosts.
  const tidGhost = new Map<number, string>()
  const markGhostLeafTids = (node: TreeNode, ghostKey: string) => {
    if (node.kind === 'leaf') tidGhost.set(node.output, ghostKey)
    else for (const child of node.children) markGhostLeafTids(child, ghostKey)
  }
  const populateGhosts = (children: readonly TreeNode[], idPath: string) => {
    const loops = detectLoops(children, graph)
    for (let li = 0; li < loops.length; li++) {
      const loop = loops[li]!
      const ghostKey = `${idPath}_l${li}`
      for (let iter = 0; iter < loop.iterations; iter++) {
        const iterStart = loop.start + iter * loop.period
        if (isHiddenIter(loop, iter)) {
          for (let k = iterStart; k < iterStart + loop.period; k++) {
            markGhostLeafTids(children[k]!, ghostKey)
          }
        } else {
          populateGhosts(children.slice(iterStart, iterStart + loop.period), `${idPath}_l${li}i${iter}`)
        }
      }
    }
    const loopCovers = new Set<number>()
    for (const loop of loops) {
      for (let k = loop.start; k < loop.start + loop.period * loop.iterations; k++) loopCovers.add(k)
    }
    for (let k = 0; k < children.length; k++) {
      if (loopCovers.has(k)) continue
      const c = children[k]!
      if (c.kind === 'cluster') populateGhosts(c.children, `${idPath}_n${k}`)
    }
  }
  populateGhosts(tree.children, "r")

  // Heavier penwidth + explicit margin so the dashed border has breathing
  // room from the inner user-cluster's tint; without these, the default thin
  // gray dashes get visually absorbed by the tint.
  const LOOP_ITER_ATTRS = [`style="dashed"`, `color="#666"`, `fontcolor="#888"`, `fontsize=10`, `fontname="sans-serif"`, `labelloc=t`, `labeljust=l`, `penwidth=1.4`, `margin=14`]
  // User-cluster style: subtle background tint (no outline) so cluster
  // boundaries can't be confused with edge connectors. Tint stacks ~8% alpha
  // per nesting level.
  const USER_CLUSTER_ATTRS = [`style="filled,rounded"`, `fillcolor="#80808014"`, `color="transparent"`, `fontcolor="#888"`, `fontsize=10`, `fontname="Courier"`, `labelloc=t`, `labeljust=l`, `penwidth=0`, `margin=10`]

  const emitCluster = (id: string, label: string, attrs: readonly string[], body: () => void) => {
    lines.push(`  subgraph cluster_${id} {`)
    lines.push(`    label="${label.replace(/"/g, '\\"')}"`)
    for (const a of attrs) lines.push(`    ${a}`)
    body()
    lines.push(`  }`)
  }

  const emitGhostNode = (ghostKey: string) => {
    emitCluster(`ghost_${ghostKey}`, "…hidden iterations…", LOOP_ITER_ATTRS, () => {
      lines.push(`    "tghost_${ghostKey}" [label="…" shape=plaintext fontsize=22 fontcolor="#888" class="ti-ghost-${ghostKey}"]`)
    })
  }

  // stripRedundantShapeComment is safe on any label — the regex doesn't match
  // for input labels (`input: name`), so the call is a no-op in that case.
  const emitNodeBox = (tid: number, rawLabel: string) => {
    const shape = graph.tensors[tid]!.shape
    const shapeNames = shape.map((d, i) => resolveDim(d, i, shape.length).name)
    const wrapped = wrapForLabel(stripRedundantShapeComment(rawLabel, shapeNames), 36)
    const shapeHtml = shapeHtmlLabel(shape, resolveDim)
    const color = tidToColor.get(tid)!
    const label = `<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="3" CELLSPACING="0">` +
      `<TR><TD>${wrapped}</TD></TR><TR><TD>${shapeHtml}</TD></TR></TABLE>>`
    lines.push(`    "t${tid}" [label=${label} color="${color}" class="ti-${tid}"]`)
  }

  // Inputs render at the top of the diagram, outside any cluster.
  for (const dn of dag) {
    if (dn.leaf !== null) continue
    const op = graph.ops[graph.tensors[dn.tensorId]!.source!]!
    emitNodeBox(dn.tensorId, describeLibraryOp(op))
  }

  // Recursive walk: leaves emit as nodes; user clusters emit as labeled
  // subgraph rectangles; loop runs emit as per-iter clusters with a ghost
  // for hidden iterations.
  const emitChildren = (children: readonly TreeNode[], idPath: string) => {
    const loops = detectLoops(children, graph)
    const loopByStart = new Map<number, { loop: LoopRun; index: number }>()
    for (let li = 0; li < loops.length; li++) loopByStart.set(loops[li]!.start, { loop: loops[li]!, index: li })

    let i = 0
    while (i < children.length) {
      const entry = loopByStart.get(i)
      if (entry) {
        const { loop, index } = entry
        for (let iter = 0; iter < loop.iterations; iter++) {
          if (isHiddenIter(loop, iter)) continue
          const iterStart = loop.start + iter * loop.period
          emitCluster(`${idPath}_l${index}_iter${iter}`,
            `iteration ${iter + 1} of ${loop.iterations}`, LOOP_ITER_ATTRS, () => {
              emitChildren(children.slice(iterStart, iterStart + loop.period), `${idPath}_l${index}i${iter}`)
            })
        }
        if (loop.iterations >= 4) emitGhostNode(`${idPath}_l${index}`)
        i = loop.start + loop.period * loop.iterations
      } else {
        const child = children[i]!
        if (child.kind === 'leaf') {
          if (!tidGhost.has(child.output)) emitNodeBox(child.output, frameSourceText(child.frame, srcCache))
        } else {
          emitCluster(`${idPath}_n${i}`, frameSourceText(child.frame, srcCache), USER_CLUSTER_ATTRS, () => {
            emitChildren(child.children, `${idPath}_n${i}`)
          })
        }
        i++
      }
    }
  }
  emitChildren(tree.children, "r")

  // Edges. For each consumer→producer pair, redirect through ghosts when one
  // side sits inside a hidden iteration. Dedupe ghost-touching edges so a
  // many-iter fan-in collapses to one line.
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

// Mirrors buildDOT's loop-hiding rules so MAX_DIAGRAM_NODES matches reality.
// Leaves count as 1; clusters are pure visual wrappers and contribute their
// children's count. Hidden iters contribute 0; each loop with hidden iters
// contributes 1 for its ghost.
function countVisibleLeaves(tree: CallCluster, graph: Graph): number {
  const isHiddenIter = (loop: LoopRun, iter: number): boolean =>
    loop.iterations >= 4 && iter >= 2 && iter <= loop.iterations - 2
  const countChildren = (children: readonly TreeNode[]): number => {
    const loops = detectLoops(children, graph)
    const loopCovers = new Set<number>()
    for (const loop of loops) {
      for (let k = loop.start; k < loop.start + loop.period * loop.iterations; k++) loopCovers.add(k)
    }
    let total = 0
    for (const loop of loops) {
      for (let iter = 0; iter < loop.iterations; iter++) {
        if (isHiddenIter(loop, iter)) continue
        const iterStart = loop.start + iter * loop.period
        total += countChildren(children.slice(iterStart, iterStart + loop.period))
      }
      if (loop.iterations >= 4) total += 1  // ghost node
    }
    for (let k = 0; k < children.length; k++) {
      if (loopCovers.has(k)) continue
      const c = children[k]!
      total += c.kind === 'leaf' ? 1 : countChildren(c.children)
    }
    return total
  }
  return countChildren(tree.children)
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
            button({ disabled: this.inferring, onClick: () => this.compileSpec() }, "Compile"),
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
          p("Tensors have structure that can be invisible when looking at code that transforms them. It shows the ops but not the shapes flowing through them. Each box is a tensor that displays its dimensionality (e.g. [B, H]) and the line of code that produced it; arrows trace dataflow."),
          p("Training and inference have different inputs and outputs because they have different goals. Training takes an input plus the correct answer, and outputs a single number measuring how wrong the model's guess was; that number drives the weight updates. Inference takes only the input and outputs the model's prediction, which is what you wanted in the first place."),
          p("Only the forward is shown — the computation that produces the model's output. It's the part specific to the architecture (what makes a transformer different from a CNN). The backprop and optimizer are automatic."),
          p("How does this run? It uses a TypeScript library called tensorgrad that compiles neural networks and runs them on WebGPU. It's designed for small demos and experiments that run in your browser, like ", a({ href: "https://typebulb.com/u/samples/ne-rf/full", target: "_blank" }, "this"), ". The editor in the code tab is the actual API for defining models. In this app, we never actually ", em("run"), " the models, so you can even explore the architecture of enormous 1T-parameter networks."),
        ),
      ),
    )
  }

  private tabBtn(key: typeof this.activeTab, label: string) {
    return button({
      class: ["tab-btn", this.activeTab === key ? "active" : ""],
      onClick: () => {
        if (this.activeTab === key) return
        this.activeTab = key
        this.update()
        if (key === "training" || key === "inference") this.rerender()
      },
    }, label)
  }

  private async compileSpec(): Promise<void> {
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
    await this.compileSpec()
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
      this.status = `Ask the AI failed: ${(e as Error)?.message ?? e}`
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

      // Best-effort — training graph still useful if it fails.
      try {
        const inferIR = await traceForward({
          model: spec.model,
          forward: spec.predict,
          inputs: spec.predictInputs,
        })
        this.infer = viewOf(inferIR)
      } catch (e) {
        console.warn("compile failed:", e)
        this.infer = null
      }

      this.update()
      this.rerender()
    } catch (e) {
      this.fail(`compile error: ${(e as Error)?.message ?? e}`, e)
    }
  }

  private rerender(): void {
    const view = this.activeTab === "inference" && this.infer ? this.infer : this.train
    if (!this.viz || !view || !this.resolveDim || !this.canvasEl) return
    const { graph, fwdOps, kernelCount } = view
    // Forward-only graph: backward and optimizer are universal training
    // machinery, not architecture. Param_input leaves hidden — Linear /
    // LayerNorm boxes already imply "has weights"; param count is in the
    // header. Tensor_input leaves excluded from the tree because buildDag
    // renders them separately as data-source nodes at the top of the graph,
    // labeled via describeLibraryOp ("input: clean" / "input: noisy") rather
    // than the call-site lookup that runs on tree leaves.
    const included = new Set<number>()
    for (const i of fwdOps) {
      const kind = graph.ops[i]!.kind
      if (kind !== "param_input" && kind !== "tensor_input") included.add(i)
    }
    const tree = buildGroupTree(graph, included)
    const leaves = collectLeaves(tree)
    const dag = buildDag(graph, leaves)

    const visibleNodes = countVisibleLeaves(tree, graph)
    this.canvasEl.innerHTML = ""
    if (visibleNodes > MAX_DIAGRAM_NODES) {
      const notice = document.createElement("div")
      notice.className = "model-error"
      notice.textContent =
        `Diagram too large to render (${visibleNodes} visible operations after loop detection). ` +
        `Metrics above are still available — reduce loop iterations or model depth to see the diagram.`
      this.canvasEl.appendChild(notice)
    } else {
      const dot = buildDOT(graph, tree, dag, this.resolveDim, this.srcCache)
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

/* Cluster header labels (both user-function and loop-iter clusters) — route
   through theme tokens for proper light/dark contrast. Graphviz emits the
   color inline, which doesn't theme-switch. */
.ir-viewer .canvas svg g.cluster > text { fill: var(--text-primary); }

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

**RoPE (rotary position embedding)** — apply `rope` to the Q/K pair after `splitHeads`, before attention scores. Returns the pair rotated:

    const [q, k] = rope(splitHeads(p.q.fwd(x), H), splitHeads(p.k.fwd(x), H))
    const v = splitHeads(p.v.fwd(x), H)                  // [B, H, T, D/H]

**Tied input/output embeddings** — declare `tok_emb` as a raw `this.param([VOCAB, D])` (not `new Embedding`, which is lookup-only). Use `embedding(tok_emb, tokens)` for the input side and `matmul(h, swapAxes(tok_emb, -1, -2))` for the output tail:

    tok_emb = this.param([VOCAB, D])
    const logits = matmul(h, swapAxes(p.tok_emb, -1, -2))   // [B, T, V]

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
**Attention layout**: `splitHeads(x, nHeads)`, `mergeHeads(x)`, `rope(q, k, { base? })` (rotary position embedding on the Q/K pair; returns the pair rotated).
**Conv/pool**: `conv2d(input, weight, { stride?, padding? })`, `maxPool2d(x, k, { stride?, padding? })`, `nearestUpsample2d(x, factor)`.
**Stochastic/grad**: `dropout(x, p)`, `randn(shape)`, `categorical(logits, axis?)` (samples from logits via Gumbel-max; i32, non-diff), `stopGradient(x)` (= PyTorch `.detach`), `capture(name, t)`.

## Gotchas

- **`Tensor` has no methods.** Every operation is a free function from the symbols list, applied as `op(x, ...)` — e.g. `reshape(x, [B, -1])`, `sum(x, axis)`, `swapAxes(x, -2, -1)`, `narrow(x, axis, start, len)`.
- **Shape tracing — the most common compile error.** Every `Linear`/`matmul` input dim must equal the last dim of the upstream tensor. Don't pattern-match by layer name: `attn_proj = Linear(D, D)` and `mlp_proj = Linear(4*D, D)` look parallel but have different shapes because attention preserves `D` while the MLP expands then contracts. Watch any layer immediately after a dim-changing op (`Linear(D, ≠D)`, `splitHeads`, `reshape`, `narrow`, `embedding`): the next layer's input dim is the NEW last dim, not the original. Expansion-then-contraction pairs (MLP up/down, autoencoder bottleneck) always swap dim orders by design. Before returning, mentally run the forward once: every `Linear`'s first arg = the last dim of what feeds it; every `reshape` preserves total element count; every broadcast is trailing-suffix compatible.
- **Operators have no PyTorch-style optional flags.** The symbols list is the full signature. `matmul(a, b)` is two args; to transpose the rhs, write `matmul(a, swapAxes(b, -2, -1))`.
- **Module internals aren't public.** Call `embedding.fwd(idx)`, never `embedding(module.weight, idx)`. The free function `embedding(table, indices)` is for raw `this.param([V, D])` tables; on an `Embedding` instance, use `.fwd`. For tied input/output embeddings, declare the raw `this.param([V, D])` table (see Conventions).
- **`reshape` doesn't transpose.** It reinterprets memory layout — to reorder axes use `permute(x, [perm])` or `swapAxes(x, a, b)`. Same total element count means `reshape` won't error on a wrong-axes pass, so this is a silent correctness bug.
- **`matmul(x, codebook)` against `[N, D]` raw params errors on inner dims.** `Linear` weights are `[in, out]` so direct `matmul(x, W)` works for projections, but raw codebook / memory / prototype params shaped `[N, D]` need `matmul(x, swapAxes(codebook, -1, -2))`. Common when porting "score query against learned vectors" patterns from PyTorch.
- **Static shapes only** — every dim is a compile-time `const` in your code, not a value read from a tensor.
- **Pass raw logits to `crossEntropy`** — it fuses log-softmax internally. Don't apply `logSoftmax` first.
- **Loss must be scalar** (rank-0). Use `mean`/`sum` to reduce.
- **`splitHeads(x, nHeads)`** reshapes one tensor: `[B, T, D] → [B, H, T, D/H]`. For Q/K/V, use three independent `Linear(D, D)` projections and call `splitHeads` on each.
- **No GQA — use full Q/K/V projections (all `H` heads).** `matmul` batch dims must match exactly (no size-1 broadcasting); matching multi-dim batches like `[B, H, T, D]` work fine — don't flatten to `[B*H, ...]` defensively.
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
    "tensorgrad": "^0.2.0",
    "@viz-js/viz": "^3.27.0",
    "sucrase": "^3.35.0",
    "domeleon": "^0.6.0"
  }
}
```