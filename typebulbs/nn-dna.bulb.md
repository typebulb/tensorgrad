---
format: typebulb/v1
name: "NN DNA"
---

**code.tsx**

```tsx
import { App, Component, div as divH, h1, p, a, span, button, h2, textarea, ul, li, code, strong, em } from "domeleon"
import {
  type Tensor, type Graph, type OpNode, type Shape, type CallSite,
  isWebGPUAvailable, getOpInputs,
  Module, compile, lr, init,
  Linear, LayerNorm, RMSNorm, Embedding, Conv2d,
  crossEntropy, nllLoss,
  capture, dropout, stopGradient, singleFlight,
  add, sub, mul, div, min, max, less, greater, where,
  sqrt, rsqrt, log, exp, neg, abs, square, sin, cos,
  relu, tanh, sigmoid, gelu, silu,
  clamp, randn,
  mean, sum, argmax, argmin,
  reshape, permute, swapAxes,
  splitHeads, mergeHeads,
  matmul,
  oneHot, arange, embedding, takeAlongAxis,
  zeros, ones,
  narrow, concat, stack, split,
  softmax, logSoftmax, softmaxCausal, whereCausal,
  conv2d, maxPool2d, nearestUpsample2d, flatten
} from "tensorgrad"
import { instance } from "@viz-js/viz"
import { transform as sucraseTransform } from "sucrase"

// Domeleon's `div` (DOM helper) is renamed to `divH` so tensorgrad's `div`
// (tensor element-wise division) keeps its natural name — which is what
// specs pasted from the samples expect. The `H` suffix echoes Domeleon's
// general-purpose `h(...)` for arbitrarily-named elements.

// ===== Spec evaluation ======================================================

type DimSpec = { size: number; name: string; desc?: string; color?: string }
type PredictFn = (m: any, inputs: any) => Tensor
type PredictInputs = Record<string, unknown>

interface InspectableForward {
  graphFor(inputs: PredictInputs): Promise<{ graph: Graph; kernels: readonly unknown[] }>
  destroy(): void
}
interface InspectableTraining {
  readonly graph: Graph
  readonly kernels: readonly unknown[]
  attach(spec: { forward: PredictFn; inputs: PredictInputs }): Promise<InspectableForward>
  destroy(): void
}
interface IRSpec {
  label: string
  description?: string
  compile: () => Promise<InspectableTraining>
  /** Inference forward — same model, returns the prediction tensor instead
   *  of a scalar loss. The bulb compiles a sibling inference graph via
   *  `train.attach({ forward: predict, inputs: predictInputs })` and
   *  renders it as the Inference tab. */
  predict: PredictFn
  predictInputs: PredictInputs
  dims?: DimSpec[]
}

// Keepalive object — statically references every tensorgrad import so the
// build pipeline can't tree-shake them. The bulb's body never uses most of
// these directly; they're consumed by the eval'd spec source, which the
// tree-shaker can't see into. Without this, imports like `Module`, `Linear`,
// `add`, etc. get stripped from the bulb's compiled output, and the eval
// fails with `ReferenceError: Module is not defined`.
const _tgKeepalive = {
  Module, compile, lr, init, isWebGPUAvailable,
  Linear, LayerNorm, RMSNorm, Embedding, Conv2d,
  crossEntropy, nllLoss,
  capture, dropout, stopGradient, singleFlight,
  add, sub, mul, div, min, max, less, greater, where,
  sqrt, rsqrt, log, exp, neg, abs, square, sin, cos,
  relu, tanh, sigmoid, gelu, silu,
  clamp, randn,
  mean, sum, argmax, argmin,
  reshape, permute, swapAxes,
  splitHeads, mergeHeads,
  matmul,
  oneHot, arange, embedding, takeAlongAxis,
  zeros, ones,
  narrow, concat, stack, split,
  softmax, logSoftmax, softmaxCausal, whereCausal,
  conv2d, maxPool2d, nearestUpsample2d, flatten,
}
;(globalThis as any).__tg_keepalive = _tgKeepalive

// Returns the evaluated spec and the cleaned source (line numbers preserved
// from the original) so the viewer can look up spec lines by stack-frame
// line number after compile.
function evaluateSpec(source: string): { spec: IRSpec; cleanedSource: string } {
  // Strip imports but PRESERVE line breaks — the regex match spans multiple
  // lines for `import { ... } from '...'`, so we replace non-newline chars
  // with empty string instead of removing the match entirely. This keeps
  // line numbers in the eval'd source aligned with the original spec.
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

type UserFrame = { fn: string; url: string; file: string; line: number }

// The spec lives inside an eval, so V8 reports its frames as
// `<anonymous>:LINE:COL` (sometimes wrapped in `eval at outer (url:N:C),
// <anonymous>:M:K`). Every other frame is bundled library / runtime code
// that we want to skip. Strategy: find the topmost frame containing an
// `<anonymous>:LINE:COL` token; that's spec code.
function firstUserFrame(site: CallSite | null): UserFrame | null {
  if (!site) return null
  for (const raw of site.stack.split("\n").slice(1)) {
    const line = raw.trim()
    if (!line) continue
    // Eval'd spec frame — take the LAST `<anonymous>:N:M` on the line so
    // nested-eval cases (eval-within-eval) report the innermost position.
    const matches = [...line.matchAll(/<anonymous>:(\d+):\d+/g)]
    if (matches.length > 0) {
      const lineNo = parseInt(matches[matches.length - 1]![1]!, 10)
      return { fn: "<spec>", url: "__spec__", file: "spec", line: lineNo }
    }
  }
  return null
}

function getSourceLine(cache: Map<string, string[]>, url: string, line: number): string | null {
  const lines = cache.get(url)
  if (!lines) return null
  return lines[line - 1] ?? null
}

// ===== Palettes + dim resolver =============================================

const DIM_AUTO_PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#aec7e8", "#ffbb78"] as const
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
  const producingGroup = new Map<number, number>()
  for (let gi = 0; gi < groups.length; gi++) {
    for (const opIdx of groups[gi]!.ops) producingGroup.set(graph.ops[opIdx]!.out, gi)
  }
  const consumers = new Map<number, Set<number>>()
  for (let gi = 0; gi < groups.length; gi++) {
    for (const opIdx of groups[gi]!.ops) {
      for (const tid of getOpInputs(graph.ops[opIdx]!)) {
        let s = consumers.get(tid); if (!s) { s = new Set(); consumers.set(tid, s) }
        s.add(gi)
      }
    }
  }
  const nodeTids = new Set<number>()
  for (const g of groups) nodeTids.add(g.output)
  for (const op of graph.ops) if (op.kind === "tensor_input" && consumers.has(op.out)) nodeTids.add(op.out)
  const nodeByTid = new Map<number, DagNode>()
  const nodes: DagNode[] = []
  const ordered: number[] = []
  for (const op of graph.ops) if (op.kind === "tensor_input" && nodeTids.has(op.out)) ordered.push(op.out)
  for (const g of groups) if (nodeTids.has(g.output)) ordered.push(g.output)
  for (const tid of ordered) {
    const t = graph.tensors[tid]!
    const gi = producingGroup.get(tid)
    const node: DagNode = { tensorId: tid, shape: t.shape, group: gi !== undefined ? groups[gi]! : null, inputs: [] }
    nodes.push(node)
    nodeByTid.set(tid, node)
  }
  for (const node of nodes) {
    if (!node.group) continue
    const inGroupOps = new Set(node.group.ops)
    const seen = new Set<number>()
    for (const opIdx of node.group.ops) {
      for (const tid of getOpInputs(graph.ops[opIdx]!)) {
        if (seen.has(tid)) continue
        seen.add(tid)
        const t = graph.tensors[tid]!
        if (t.source !== null && inGroupOps.has(t.source)) continue
        if (!nodeByTid.has(tid)) continue
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
// by coverage k*n; ties favor longer bodies so 5-group × 2-iter beats a
// 1-group × 10-iter degenerate.

type LoopRun = { start: number; period: number; iterations: number }

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
  const loops: LoopRun[] = []
  let i = 0
  while (i < signatures.length) {
    let bestK = 0
    let bestN = 0
    // Period >= 2: single-line iteration bodies merge into one group via
    // groupBySourceLine, so legit period-1 loops can't exist; any period-1
    // match is coincidence.
    for (let k = 2; k <= Math.floor((signatures.length - i) / 2); k++) {
      let n = 1
      while (true) {
        const next = i + n * k
        if (next + k > signatures.length) break
        let match = true
        for (let j = 0; j < k; j++) {
          if (signatures[i + j] !== signatures[next + j]) { match = false; break }
        }
        if (!match) break
        n++
      }
      if (n >= 2 && (k * n > bestK * bestN || (k * n === bestK * bestN && k > bestK))) {
        bestK = k
        bestN = n
      }
    }
    if (bestN >= 2) {
      loops.push({ start: i, period: bestK, iterations: bestN })
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

function resolveSourceLabel(graph: Graph, node: DagNode, srcCache: Map<string, string[]>): string {
  const t = graph.tensors[node.tensorId]!
  const leafOp = t.source !== null ? graph.ops[t.source]! : null
  if (leafOp && (leafOp.kind === "tensor_input" || leafOp.kind === "param_input")) {
    return describeLibraryOp(leafOp)
  }
  const frame = node.group?.frame
  if (frame) {
    const src = getSourceLine(srcCache, frame.url, frame.line)
    return src ? src.trim() : `${frame.fn}:${frame.line}`
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

  type Membership = { loopId: number; iter: number }
  const memberOf = new Map<Group, Membership>()
  for (let li = 0; li < loops.length; li++) {
    const loop = loops[li]!
    for (let it = 0; it < loop.iterations; it++) {
      for (let gj = 0; gj < loop.period; gj++) {
        const grp = groups[loop.start + it * loop.period + gj]
        if (grp) memberOf.set(grp, { loopId: li, iter: it })
      }
    }
  }

  // Precompute tid → color so edges (which only have the source tid) can
  // look up the node-aware color without rebuilding the key.
  const tidToColor = new Map<number, string>()
  for (const node of dag) tidToColor.set(node.tensorId, nodeColor(node, graph))

  const emitNode = (dagIdx: number, indent: string) => {
    const node = dag[dagIdx]!
    const tid = node.tensorId
    const wrapped = wrapForLabel(resolveSourceLabel(graph, node, srcCache), 36)
    const shapeHtml = shapeHtmlLabel(node.shape, resolveDim)
    const color = tidToColor.get(tid)!
    const label = `<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="3" CELLSPACING="0">` +
      `<TR><TD>${wrapped}</TD></TR><TR><TD>${shapeHtml}</TD></TR></TABLE>>`
    lines.push(`${indent}"t${tid}" [label=${label} color="${color}" class="ti-${tid}"]`)
  }

  // Bucket by membership. Map insertion order = op-index order, so iteration
  // 0's cluster emits before iteration 1's.
  const flatIdxs: number[] = []
  const buckets = new Map<string, { loopId: number; iter: number; idxs: number[] }>()
  for (let i = 0; i < dag.length; i++) {
    const m = dag[i]!.group ? memberOf.get(dag[i]!.group!) : undefined
    if (!m) flatIdxs.push(i)
    else {
      const key = `${m.loopId}:${m.iter}`
      let b = buckets.get(key)
      if (!b) { b = { loopId: m.loopId, iter: m.iter, idxs: [] }; buckets.set(key, b) }
      b.idxs.push(i)
    }
  }

  for (const i of flatIdxs) emitNode(i, "  ")
  for (const b of buckets.values()) {
    const loop = loops[b.loopId]!
    lines.push(`  subgraph cluster_loop_${b.loopId}_iter_${b.iter} {`)
    lines.push(`    label="iteration ${b.iter + 1} of ${loop.iterations}"`)
    lines.push(`    style="dashed"`)
    lines.push(`    color="#888"`)
    lines.push(`    fontcolor="#888"`)
    lines.push(`    fontsize=10`)
    lines.push(`    fontname="sans-serif"`)
    lines.push(`    labelloc=t`)
    lines.push(`    labeljust=l`)
    for (const i of b.idxs) emitNode(i, "    ")
    lines.push(`  }`)
  }

  for (const node of dag) {
    for (const inTid of node.inputs) {
      const color = tidToColor.get(inTid)!
      lines.push(`  "t${inTid}" -> "t${node.tensorId}" [color="${color}" class="ti-${inTid}"]`)
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
// Total trainable scalars: walk the graph's leaf param_input ops, multiply
// their shape dims, sum. State tensors (Adam m/v moments) are excluded —
// those are optimizer bookkeeping, not model parameters.

function countParams(graph: Graph): number {
  let total = 0
  for (const op of graph.ops) {
    if (op.kind === "param_input") {
      const shape = graph.tensors[op.out]!.shape
      let n = 1
      for (const d of shape) n *= d
      total += n
    }
  }
  return total
}

function formatParamCount(n: number): string {
  if (n < 1000) return `${n}`
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}K`
  return `${(n / 1_000_000).toFixed(1)}M`
}

// ===== Component ============================================================

class IRViewer extends Component {
  // Inference tab is omitted from the strip entirely when the spec has no predict function.
  activeTab: "training" | "inference" | "code" | "info" = "training"
  status = "Loading…"
  legend: LegendItem[] = []
  dimCollisions: DimCollision[] = []
  modelLabel = ""
  modelDescription = ""

  private readonly defaultSource: string = (tb.insight<{ source: string }>()?.source ?? "")
  private specSource: string = (tb.insight<{ source: string }>()?.source ?? "")
  private inferring = false

  // Training and inference graphs cached separately so tab switches don't recompile.
  private viz: any = null
  private trainGraph: Graph | null = null
  private trainKernelCount = 0
  private trainFwdOps: Set<number> = new Set()
  private inferGraph: Graph | null = null
  private inferKernelCount = 0
  private inferFwdOps: Set<number> = new Set()
  private resolveDim: ((size: number, pos: number, rank: number) => ResolvedDim) | null = null
  private srcCache: Map<string, string[]> = new Map()
  private canvasEl: HTMLDivElement | null = null
  private textareaEl: HTMLTextAreaElement | null = null
  private currentTrain: InspectableTraining | null = null
  private currentInfer: InspectableForward | null = null

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
        this.inferGraph ? this.tabBtn("inference", "Inference graph") : null,
        this.tabBtn("code", "Code"),
        this.tabBtn("info", "How it works"),
      ),

      // Training and Inference share this tab-content; rerender() picks the graph from activeTab.
      divH({ class: "tab-content", style: { display: showViz ? "block" : "none" } },
        divH({ class: "panel" },
          divH({ class: "model-header" },
            this.modelLabel ? h2({ class: "model-title" }, this.modelLabel) : null,
            divH({ class: "model-stats" }, this.status),
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
                ...this.dimCollisions.flatMap((c, i) => {
                  const prefix = i > 0 ? ", " : ""
                  return [prefix, `${c.names.join("/")} = ${c.size}`]
                }),
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
          p("Training and inference are written as two separate forward functions. Training ends in a scalar loss; inference returns the prediction and omits training-only ops like dropout."),
          p("Only the forward is shown. It's the part specific to the architecture (what makes a transformer different from a CNN). The backprop and optimizer are automatic."),
          p("Repeated structure is auto-detected (a transformer's stacked layers, an RNN's unroll) and shown as just two iterations for brevity."),
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
        const prevTab = this.activeTab
        // Bottom-anchored users comparing training vs inference want the
        // loss/prediction tail (where the graphs diverge) to stay in view
        // across switches, even when graph heights differ. Capture pre-
        // swap so we can re-anchor below.
        const wasAtBottom = window.scrollY + window.innerHeight
          >= document.documentElement.scrollHeight - 5
        this.activeTab = key
        this.update()
        if (key === "training" || key === "inference") {
          this.rerender()
          // Anchor-to-new-bottom only when switching between viz tabs
          // (the comparison case). Entering from Code/How-it-works is a
          // context shift; let the browser preserve absolute scrollY.
          const switchedBetweenVizTabs = prevTab === "training" || prevTab === "inference"
          if (wasAtBottom && switchedBetweenVizTabs) {
            window.scrollTo(0, document.documentElement.scrollHeight)
          }
        }
      },
    }, label)
  }

  private async applySpec(): Promise<void> {
    if (!this.textareaEl) return
    const source = this.textareaEl.value
    this.specSource = source
    this.activeTab = "training"
    this.update()
    await this.compileFromSource(source)
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
        this.activeTab = "training"
        this.update()
        await this.compileFromSource(result.source)
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
    if (!isWebGPUAvailable()) {
      this.status = "WebGPU not available. Try Chrome 113+ or Safari 17.4+."
      this.update()
      return
    }
    await this.compileFromSource(this.specSource)
  }

  private async compileFromSource(source: string): Promise<void> {
    // Null both refs explicitly: a partial re-compile failure must not leave stale state.
    if (this.currentTrain) {
      try { this.currentTrain.destroy() } catch { /* ignore */ }
      this.currentTrain = null
      this.currentInfer = null
    }
    if (this.canvasEl) this.canvasEl.innerHTML = ""
    this.legend = []
    this.dimCollisions = []
    this.inferGraph = null
    // If the new spec has no predict, Inference tab disappears — don't strand the user there.
    if (this.activeTab === "inference") this.activeTab = "training"

    this.status = "Evaluating spec…"
    this.update()
    let spec: IRSpec
    let cleanedSource: string
    try {
      const ev = evaluateSpec(source)
      spec = ev.spec
      cleanedSource = ev.cleanedSource
    } catch (e) {
      this.status = `spec error: ${(e as Error)?.message ?? e}`
      this.update()
      console.error(e)
      return
    }

    this.status = `Compiling "${spec.label}"…`
    this.update()
    try {
      if (!this.viz) this.viz = await instance()
      const train = await spec.compile()
      this.currentTrain = train
      this.trainGraph = train.graph
      this.trainKernelCount = train.kernels.length
      this.trainFwdOps = forwardReachable(train.graph, train.graph.outputs)

      this.modelLabel = spec.label
      this.modelDescription = spec.description ?? ""

      const { resolve, legend, collisions } = buildDimResolver(train.graph, spec.dims)
      this.resolveDim = resolve

      // Spec frames have URL "__spec__" — inject the cleaned source directly.
      this.srcCache = new Map()
      this.srcCache.set("__spec__", cleanedSource.split("\n"))

      this.legend = legend
      this.dimCollisions = collisions

      // Inference compile: same model, prediction output instead of loss.
      // train.attach shares the worker and param buffers; the inference
      // graph is a sibling, not a clone. Best-effort — if attach fails the
      // training graph is still useful, so we warn and continue.
      try {
        const infer = await train.attach({
          forward: spec.predict,
          inputs: spec.predictInputs,
        })
        this.currentInfer = infer
        const ir = await infer.graphFor(spec.predictInputs)
        this.inferGraph = ir.graph
        this.inferKernelCount = ir.kernels.length
        this.inferFwdOps = forwardReachable(ir.graph, ir.graph.outputs)
      } catch (e) {
        console.warn("inference attach failed:", e)
        this.inferGraph = null
      }

      this.update()
      this.rerender()
    } catch (e) {
      this.status = `compile error: ${(e as Error)?.message ?? e}`
      this.update()
      console.error(e)
    }
  }

  private rerender(): void {
    // Pick the active graph based on activeTab. Training is always
    // available after a successful compile; inference is opt-in. If the
    // user is on activeTab="inference" without an inference graph (which
    // shouldn't happen — the Inference tab only renders when one exists),
    // fall back to training rather than blanking the canvas.
    const showingInference = this.activeTab === "inference" && this.inferGraph !== null
    const graph = showingInference ? this.inferGraph! : this.trainGraph
    const fwdOps = showingInference ? this.inferFwdOps : this.trainFwdOps
    const kernelCount = showingInference ? this.inferKernelCount : this.trainKernelCount
    if (!this.viz || !graph || !this.resolveDim || !this.canvasEl) return
    // Only render the forward graph — backward and optimizer are universal
    // training machinery (one mirrored gradient per forward op, one Adam
    // update per param). Showing those literally adds compiler emissions
    // not architectural insight. Also hide param_input leaves: the Linear/
    // LayerNorm/etc. boxes already imply "this has weights"; rendering W
    // and b as separate nodes adds clutter without insight. Param count
    // is in the header subtitle. The how-it-works tab covers backward
    // + optimizer in prose.
    const included = new Set<number>()
    for (const i of fwdOps) {
      if (graph.ops[i]!.kind !== "param_input") included.add(i)
    }
    const groups = groupBySourceLine(graph, included)
    const dag = buildDag(graph, groups)
    const loops = detectLoops(groups, graph)
    const dot = buildDOT(graph, dag, groups, loops, this.resolveDim, this.srcCache)
    const svg = this.viz.renderSVGElement(dot)
    this.canvasEl.innerHTML = ""
    this.canvasEl.appendChild(svg)
    attachHoverTrace(svg)
    // Param count comes from the training graph in both views — the
    // inference graph shares the same params, but the bulb intentionally
    // hides param_input leaves in the diagram so iterating the inference
    // graph's ops wouldn't double-count anything.
    const params = this.trainGraph ? countParams(this.trainGraph) : 0
    this.status = `${formatParamCount(params)} parameters · ${kernelCount} kernels`
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
.ir-viewer .info-panel h2 { font-size: 0.95rem; font-weight: 600; margin: 0.95rem 0 0.4rem; color: var(--text-primary); }
.ir-viewer .info-panel h2:first-child { margin-top: 0; }

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
.ir-viewer .model-stats { font-size: 0.88rem; font-weight: 600; color: var(--text-primary); margin: 0 0 0.6rem; }
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

The `source` is real TypeScript — newlines and indentation included. Nothing else: no commentary, no markdown.

## What the source must contain

Every tensorgrad symbol is in scope at eval time — call any op (`narrow`, `clamp`, `softmax`, etc.) directly in the body without importing. The `import { ... } from 'tensorgrad'` line at the top is optional, stripped before eval, and exists only as a hint for human readers. Don't import inline, don't `require()` — there's no need.

The source MUST end with:

    export const irSpec = {
      label: '...',
      description: '...',
      compile: () => compile({ model: new MyModel(), loss: lossFn, inputs, optimizer }),
      predict: predictFn,
      predictInputs: { /* same shape as `inputs` but without target/label fields */ },
      dims: [ { size: N, name: 'B', desc: '...' }, ... ],
    }

- **`label`**: short human title (4–8 words). Example: `'Transformer for 2-digit addition'`, `'NeRF-tiny (image INR)'`, `'MLP fits sin(x)'`. This is what shows as the visualization's headline.

- **`description`**: 2–3 sentences explaining the architecture in plain English. Cover the model type, the task, key design choices (positional encoding, causal masking, normalization, gating, etc.), and the relevant hyperparameter scales (batch size, depth, width, head count). This is what someone seeing only the rendered graph would need to know to understand what they're looking at.

- **`compile`**: zero-arg thunk returning the `compile({...})` promise.

- **`predict`**: forward function returning the network's *prediction* (logits, regression output, generated sample) rather than a scalar loss. Same shape as a forward but without the loss tail. Compiled as a sibling graph via `train.attach({ forward: predict, inputs: predictInputs })` for visualizing what the network actually produces at inference time.

- **`predictInputs`**: same shape as `inputs` but without target/label fields (a classifier's `{ x, y }` becomes `{ x }`). Use the same concrete batch size as `inputs` — the inference graph compiles at a fixed shape for rendering.

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

**Recurrent state in unrolled loops** — initialize with `zeros(shape)`, then the loop body reads as the pure recurrence:

    let h = zeros([B, H, P, N])
    for (let t = 0; t < T; t++) {
      h = add(mul(a_t, h), mul(b_t, X_t))
    }

## Available tensorgrad symbols (already in scope at eval time)

**Layer modules** — instantiate as class fields; each instance exposes `.fwd(x)`:

    new Linear(inDim, outDim, { bias?, init?, decay? })
    new LayerNorm(dim, { eps?, bias?, decay? })
    new RMSNorm(dim, { eps?, decay? })
    new Embedding(vocab, dim, { init?, decay? })
    new Conv2d(inC, outC, k, { stride?, padding?, bias?, init?, decay? })  // dense only; no groups
**Compile/lifecycle**: `Module`, `compile`, `lr`, `init`.
**Losses**: `crossEntropy(logits, targets, { reduction? })`, `nllLoss(logProbs, targets, { reduction? })` — default reduction is mean; use `'none'` for per-position output.
**Arithmetic**: `add`, `sub`, `mul`, `div`, `min`, `max` — each takes `(Tensor, Tensor)` or `(Tensor, number)`.
**Comparison**: `less`, `greater` (same scalar overload as arithmetic), `where(cond, ifTrue, ifFalse)`.
**Unary math**: `sqrt`, `rsqrt`, `log`, `exp`, `neg`, `abs`, `square`, `sin`, `cos`.
**Activations**: `relu`, `tanh`, `sigmoid`, `gelu`, `silu`.
**Clamping**: `clamp(x, lo, hi)` — `lo` and `hi` are numbers.
**Reductions**: `mean(x, axis?, { keepDims? })`, `sum(x, axis?, { keepDims? })`, `argmax`, `argmin`.
**Shape**: `reshape(x, [dims])` (one `-1` allowed, inferred from total size), `permute`, `swapAxes` (= PyTorch `transpose`), `flatten`.
**Linear algebra**: `matmul`.
**Indexing**: `oneHot(idx, depth)`, `arange(n)`, `embedding(table, indices)`, `takeAlongAxis(input, indices, axis)` (= PyTorch `gather`).
**Const-tensor builders**: `zeros(shape, dtype?)`, `ones(shape, dtype?)` — default `f32`. The full set is `randn` / `arange` / `oneHot` / `zeros` / `ones`; no `full`, `eye`, `linspace`, `tril`, or `like`-variants.
**Slicing/structural**: `narrow(t, axis, start, len)`, `concat([a, b, ...], axis)`, `stack([a, b, ...], axis)`, `split(t, [size1, size2, ...], axis)`.
**Fused ML**: `softmax`, `logSoftmax`, `softmaxCausal`, `whereCausal`.
**Attention layout**: `splitHeads(x, nHeads)`, `mergeHeads(x)`.
**Conv/pool**: `conv2d(input, weight, { stride?, padding? })`, `maxPool2d(x, k, { stride?, padding? })`, `nearestUpsample2d(x, factor)`.
**Stochastic/grad**: `dropout(x, p)`, `randn(shape)`, `stopGradient(x)` (= PyTorch `.detach`), `capture(name, t)`.

## Gotchas

- **`Tensor` has no methods.** Every operation is a free function from the symbols list, applied as `op(x, ...)` — e.g. `reshape(x, [B, -1])`, `sum(x, axis)`, `swapAxes(x, -2, -1)`, `narrow(x, axis, start, len)`.
- **Operators have no PyTorch-style optional flags.** The symbols list is the full signature. `matmul(a, b)` is two args; to transpose the rhs, write `matmul(a, swapAxes(b, -2, -1))`.
- **Static shapes only** — every dim is a compile-time `const` in your code, not a value read from a tensor.
- **Pass raw logits to `crossEntropy`** — it fuses log-softmax internally. Don't apply `logSoftmax` first.
- **Loss must be scalar** (rank-0). Use `mean`/`sum` to reduce.
- **`splitHeads(x, nHeads)`** reshapes one tensor: `[B, T, D] → [B, H, T, D/H]`. For Q/K/V, use three independent `Linear(D, D)` projections and call `splitHeads` on each.
- **Attention scaling**: multiply scores by `1 / Math.sqrt(D_HEAD)` before `softmaxCausal`.
- **Iterative architectures: small loop counts.** For RNN unrolls, diff-physics rollouts, diffusion samplers, multi-round message passing, etc., set the inner loop count to a small constant (typically `2`) — large unrolls produce diagrams the renderer can't draw. Add a comment with the production value: `const HORIZON = 2  // 32+ for actual training`.
- **No `groups` on Conv2d, no `Conv1d`.** Conv2d is dense conv only.
- **No in-forward param creation.** `this.param` is class-field only; forwards are pure tensor compositions over the already-built module.
- **No `scan` / `cumsum`.** Unroll trace-time loops; keep T small.

## Reference example (small)

Use this as a structural template. Substitute the model logic for the user's request.

    import {
      Module, compile, Linear,
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
      compile: () => compile({ model: new MLP(), loss: lossFn, inputs, optimizer }),
      predict: predictFn,
      predictInputs,
      dims: [
        { size: B,      name: 'B', desc: 'batch' },
        { size: HIDDEN, name: 'H', desc: 'hidden' },
      ],
    }

If the user pastes their own (possibly broken) code, fix it to match these conventions while preserving their intent. If the request is ambiguous, pick reasonable defaults (batch 64–256, hidden 64–128, Adam optimizer at 1e-3 or 5e-3).
```

**insight.json**

```json
{
  "source": "import {\n  Module, compile, Linear,\n  mul, sub, mean, reshape, relu, sigmoid, concat,\n  sin, cos, square,\n  type Tensor,\n} from 'tensorgrad'\n\nconst BATCH_SIZE = 1024\nconst L_FREQS = 8        // π·2^0 .. π·2^7\nconst HIDDEN = 64\n\nclass NeRFTiny extends Module {\n  l1 = new Linear(4 * L_FREQS, HIDDEN)\n  l2 = new Linear(HIDDEN, HIDDEN)\n  l3 = new Linear(HIDDEN, HIDDEN)\n  l4 = new Linear(HIDDEN, 3)\n}\n\n// Sinusoidal positional encoding (NeRF / Tancik et al.). For each input\n// coord, emit sin(π·2^k·x), cos(π·2^k·x) for k = 0..L-1; concat sin and\n// cos features. Output: [B, 4L].\nfunction posEnc(coords: Tensor, freqs: Tensor): Tensor {\n  const B = coords.shape[0]!\n  const scaled = mul(reshape(coords, [B, 2, 1]), reshape(freqs, [1, 1, L_FREQS]))\n  const sinF = reshape(sin(scaled), [B, 2 * L_FREQS])\n  const cosF = reshape(cos(scaled), [B, 2 * L_FREQS])\n  return concat([sinF, cosF], 1)\n}\n\nfunction modelFwd(m: NeRFTiny, coords: Tensor, freqs: Tensor): Tensor {\n  let h = posEnc(coords, freqs)\n  h = relu(m.l1.fwd(h))\n  h = relu(m.l2.fwd(h))\n  h = relu(m.l3.fwd(h))\n  return sigmoid(m.l4.fwd(h))\n}\n\nfunction lossFn(\n  m: NeRFTiny,\n  { coords, rgb, freqs }: { coords: Tensor; rgb: Tensor; freqs: Tensor },\n): Tensor {\n  return mean(square(sub(modelFwd(m, coords, freqs), rgb)))\n}\n\nfunction predictFn(\n  m: NeRFTiny,\n  { coords, freqs }: { coords: Tensor; freqs: Tensor },\n): Tensor {\n  return modelFwd(m, coords, freqs)\n}\n\nconst inputs = {\n  coords: [BATCH_SIZE, 2],\n  rgb:    [BATCH_SIZE, 3],\n  freqs:  [L_FREQS],\n} as const\n\nconst predictInputs = {\n  coords: [BATCH_SIZE, 2],\n  freqs:  [L_FREQS],\n} as const\n\nconst optimizer = { kind: 'adam', lr: 1e-3 } as const\n\nexport const irSpec = {\n  label: 'An MLP that learns an image',\n  description: 'A 4-layer MLP that fits a single image as an implicit function (x, y) → (r, g, b). Uses sinusoidal positional encoding with 8 frequency bands (π·2^0 through π·2^7) before the MLP, giving the network the high-frequency basis it needs to represent fine image detail. 1024 random pixels per batch, MSE loss in RGB space, Adam at 1e-3.',\n  compile: () => compile({ model: new NeRFTiny(), loss: lossFn, inputs, optimizer }),\n  predict: predictFn,\n  predictInputs,\n  dims: [\n    { size: BATCH_SIZE,  name: 'B',  desc: 'batch (random pixels)' },\n    { size: 2,           name: '2',  desc: 'xy coords' },\n    { size: 3,           name: '3',  desc: 'RGB' },\n    { size: L_FREQS,     name: 'L',  desc: 'frequency bands' },\n    { size: 4 * L_FREQS, name: '4L', desc: 'pos-enc features' },\n    { size: 2 * L_FREQS, name: '2L', desc: 'sin/cos features' },\n    { size: HIDDEN,      name: 'H',  desc: 'hidden' },\n  ],\n}\n"
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
    "tensorgrad": "0.1.0",
    "@viz-js/viz": "^3.27.0",
    "sucrase": "^3.35.0",
    "domeleon": "^0.6.0"
  }
}
```
