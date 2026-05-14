// tensorgrad-viewer: renders a compiled `Graph` as an interactive DAG, using
// Graphviz (via @viz-js/viz, a WASM port) for layout AND rendering. We just
// emit a DOT description from the IR; Graphviz returns a fully-rendered SVG
// with edges that actually connect to nodes.
//
// Usage:
//   import { renderIRViewer } from 'tensorgrad-viewer'
//   const train = await compile({ ... })
//   await renderIRViewer({
//     container: document.getElementById('ir')!,
//     graph: train.graph,
//     kernelCount: train.kernels.length,
//     dims: [{ size: B, name: 'B', desc: 'batch' }, ...],
//   })

import { instance } from '@viz-js/viz'
import type { Graph, OpNode, CallSite, Shape } from 'tensorgrad'

export type DimSpec = {
  size: number
  name: string
  desc?: string
  color?: string
}

// Minimum structural shape of a CompiledTraining the IR viewer needs.
// `tensorgrad`'s CompiledTraining<T> satisfies this; we type it structurally
// so an IRSpec doesn't have to leak the model generic.
export interface InspectableTraining {
  readonly graph: Graph
  readonly kernels: readonly unknown[]
  destroy(): void
}

// A sample's contract for being inspectable: a label, a thunk that returns a
// compiled training graph, and optional dim metadata for the visualization.
// `compile` should return a fresh worker each call — the picker calls
// `.destroy()` when switching specs.
export interface IRSpec {
  label: string
  compile: () => Promise<InspectableTraining>
  dims?: DimSpec[]
}

export type RenderIRViewerOptions = {
  container: HTMLElement
  graph: Graph
  /** From `train.kernels.length`; shown in the stats line if provided. */
  kernelCount?: number
  /** Human-readable labels for dim sizes — drives both shape text and the legend. */
  dims?: DimSpec[]
  /** Default 'forward'. The user toggles between this and 'full' (forward + autograd + optimizer). */
  initialMode?: 'forward' | 'full'
}

// ===== IR utilities =========================================================

function opInputs(op: OpNode): number[] {
  switch (op.kind) {
    case 'param_input': case 'tensor_input': case 'state_input':
    case 'arange': case 'const_scalar': case 'randn':
      return []
    case 'add': case 'sub': case 'mul': case 'div': case 'min': case 'max':
    case 'less': case 'greater':
    case 'matmul': case 'matmul_batched':
      return [op.a, op.b]
    case 'mul_scalar': case 'add_scalar':
    case 'sqrt': case 'rsqrt': case 'log': case 'exp': case 'relu':
    case 'neg': case 'abs': case 'tanh': case 'sigmoid': case 'sin': case 'cos':
    case 'mean_last': case 'sum_last': case 'argmax_last':
    case 'reshape': case 'permute':
    case 'softmax_causal_last': case 'log_softmax_last':
    case 'where_causal': case 'stop_gradient':
    case 'slice_last_range': case 'slice_range': case 'scatter_axis':
    case 'broadcast_to': case 'sum_to_shape':
      return [op.a]
    case 'dropout': return [op.a, op.seed]
    case 'one_hot': return [op.indices]
    case 'where': return [op.cond, op.a, op.b]
    case 'concat': return [...op.inputs]
    case 'relu_grad': return [op.x, op.dy]
    case 'adam_update_m': return [op.m, op.g]
    case 'adam_update_v': return [op.v, op.g]
    case 'adam_update_p':
      return [op.p, op.mNew, op.vNew, op.lrt, ...(op.decayShrinkTensor !== null ? [op.decayShrinkTensor as number] : [])]
    case 'conv2d': return [op.input, op.weight]
    case 'conv2d_input_grad': return [op.weight, op.dy]
    case 'conv2d_weight_grad': return [op.input, op.dy]
    case 'max_pool_2d': return [op.input]
    case 'max_pool_2d_grad': return [op.input, op.dy]
    default: return []
  }
}

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
    for (const inp of opInputs(graph.ops[t.source]!)) stack.push(inp)
  }
  return ops
}

// ===== Source attribution ===================================================

type UserFrame = { fn: string; url: string; file: string; line: number }

function firstUserFrame(site: CallSite | null): UserFrame | null {
  if (!site) return null
  for (const raw of site.stack.split('\n').slice(1)) {
    const line = raw.trim()
    if (!line) continue
    // Skip library frames — tensorgrad itself and node_modules. The viewer
    // package never appears in op-construction stacks (it doesn't call ops).
    if (/tensorgrad[\/\\](?:src|dist)[\/\\]/.test(line)) continue
    if (/node_modules/.test(line)) continue
    const m1 = line.match(/at\s+([^\s(]+)\s+\(([^)]+):(\d+):\d+\)/)
    if (m1) {
      const url = m1[2]!
      const file = url.split(/[/\\]/).pop() ?? ''
      return { fn: m1[1] ?? '<anon>', url, file, line: parseInt(m1[3] ?? '0', 10) }
    }
    const m2 = line.match(/at\s+([^:)]+):(\d+):\d+/)
    if (m2) {
      const url = m2[1]!
      const file = url.split(/[/\\]/).pop() ?? ''
      return { fn: '<top>', url, file, line: parseInt(m2[2] ?? '0', 10) }
    }
  }
  return null
}

async function loadSources(urls: Set<string>): Promise<Map<string, string[]>> {
  const cache = new Map<string, string[]>()
  await Promise.all(Array.from(urls).map(async url => {
    try {
      const resp = await fetch(url)
      if (!resp.ok) { cache.set(url, []); return }
      const txt = await resp.text()
      cache.set(url, txt.split('\n'))
    } catch { cache.set(url, []) }
  }))
  return cache
}
function getSourceLine(cache: Map<string, string[]>, url: string, line: number): string | null {
  const lines = cache.get(url)
  if (!lines) return null
  return lines[line - 1] ?? null
}

// ===== Dim naming + palette =================================================

// Auto-color palette for unique dim sizes the user didn't name. Stable per
// session: assigned in encounter order over the graph's tensors.
const DIM_AUTO_PALETTE = [
  '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b',
  '#e377c2', '#17becf', '#bcbd22', '#aec7e8', '#ffbb78',
] as const

type ResolvedDim = { name: string; color: string }

function buildDimResolver(graph: Graph, dims: readonly DimSpec[] | undefined): {
  resolve: (size: number) => ResolvedDim
  legend: { name: string; desc: string; size: number; color: string }[]
} {
  const userBySize = new Map<number, DimSpec>()
  for (const d of dims ?? []) if (!userBySize.has(d.size)) userBySize.set(d.size, d)
  // Walk every tensor shape so the auto-color assignment is deterministic
  // and matches the visual order dims appear in the graph. Assign auto colors
  // to *every* unique size, including ones the user named — so user dims
  // that omitted an explicit `color` still get one for free, instead of
  // falling back to the gray '#666'.
  const autoBySize = new Map<number, string>()
  let next = 0
  for (const t of graph.tensors) {
    for (const d of t.shape) {
      if (autoBySize.has(d)) continue
      autoBySize.set(d, DIM_AUTO_PALETTE[next % DIM_AUTO_PALETTE.length]!)
      next++
    }
  }
  const resolve = (size: number): ResolvedDim => {
    const u = userBySize.get(size)
    if (u) return { name: u.name, color: u.color ?? autoBySize.get(size) ?? '#666' }
    return { name: String(size), color: autoBySize.get(size) ?? '#666' }
  }
  const legend = (dims ?? []).map(d => ({
    name: d.name,
    desc: d.desc ?? '',
    size: d.size,
    color: d.color ?? autoBySize.get(d.size) ?? '#666',
  }))
  return { resolve, legend }
}

// ===== Per-tensor palette ===================================================
// Stable color per tensor — picked for edges so overlapping/crossing edges
// stay distinguishable.

const TENSOR_PALETTE = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8',
  '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
] as const
function tensorColor(tid: number): string {
  return TENSOR_PALETTE[tid % TENSOR_PALETTE.length]!
}

// ===== Library-op descriptions ==============================================
// Library-emitted groups (autograd, Adam, leaf inputs) get described by op
// kind rather than by their captured site — which would walk back to the
// user's `await compile(...)` call and be useless.

function describeLibraryOp(op: OpNode): string {
  if (op.kind === 'const_scalar') return `const ${(op as { value: number }).value}`
  if (op.kind === 'state_input') return `state: ${(op as { name: string }).name}`
  if (op.kind === 'tensor_input') return `input: ${(op as { name: string }).name}`
  if (op.kind === 'param_input') return `param: ${(op as { name: string }).name}`
  if (op.kind === 'adam_update_p') return 'Adam: param update'
  if (op.kind === 'adam_update_m') return 'Adam: m moment'
  if (op.kind === 'adam_update_v') return 'Adam: v moment'
  if (op.kind === 'relu_grad') return 'grad: relu'
  if (op.kind === 'broadcast_to') return 'broadcast'
  if (op.kind === 'sum_to_shape') return 'sum → shape'
  if (op.kind === 'scatter_axis') return 'scatter'
  return `backward: ${op.kind}`
}

// ===== Node category (dye highlighting) =====================================
// Each DAG node is one of these kinds. The chip row lets the user dye one or
// more kinds — selected kinds keep their per-tensor color, the rest go
// grayscale (still readable, no saturation).

type NodeCategory = 'user' | 'input' | 'param' | 'backward' | 'optimizer'

function categorize(graph: Graph, node: DagNode, forwardOps: Set<number>): NodeCategory {
  const t = graph.tensors[node.tensorId]!
  const leafOp = t.source !== null ? graph.ops[t.source]! : null
  if (leafOp) {
    if (leafOp.kind === 'tensor_input') return 'input'
    if (leafOp.kind === 'param_input') return 'param'
    // State inputs are the optimizer's persistent storage (Adam m/v); group
    // them with optimizer so the dye for "optimizer" lights up the whole
    // moment-read → update → moment-write chain.
    if (leafOp.kind === 'state_input') return 'optimizer'
  }
  if (node.group && node.group.ops.every(idx => !forwardOps.has(idx))) {
    const lastOp = graph.ops[node.group.ops[node.group.ops.length - 1]!]!
    if (lastOp.kind === 'adam_update_p' || lastOp.kind === 'adam_update_m' || lastOp.kind === 'adam_update_v') {
      return 'optimizer'
    }
    return 'backward'
  }
  return 'user'
}

const CATEGORY_LABELS: { key: NodeCategory; label: string }[] = [
  { key: 'user',      label: 'User code' },
  { key: 'input',     label: 'Inputs' },
  { key: 'param',     label: 'Params' },
  { key: 'backward',  label: 'Backward' },
  { key: 'optimizer', label: 'Optimizer' },
]

// ===== Grouping by source line =============================================

type Group = {
  ops: number[]
  frame: UserFrame | null
  output: number
}

function groupBySourceLine(graph: Graph, included: Set<number>): Group[] {
  const groups: Group[] = []
  let curKey = ''
  let cur: Group | null = null
  for (let i = 0; i < graph.ops.length; i++) {
    if (!included.has(i)) continue
    const op = graph.ops[i]!
    const t = graph.tensors[op.out]!
    const frame = firstUserFrame(t.site)
    const key = frame ? `${frame.url}#${frame.line}` : '<none>'
    if (cur && key === curKey) {
      cur.ops.push(i)
      cur.output = op.out
    } else {
      cur = { ops: [i], frame, output: op.out }
      groups.push(cur)
      curKey = key
    }
  }
  return groups
}

// ===== DAG building =========================================================
// Pick which tensors become DAG nodes (one per source-line group output, plus
// graph inputs). For each non-leaf node, compute which other DAG-node tensors
// its producing group consumes.

type DagNode = {
  tensorId: number
  shape: Shape
  group: Group | null
  inputs: number[]
}

function buildDag(graph: Graph, groups: Group[]): DagNode[] {
  const producingGroup = new Map<number, number>()
  for (let gi = 0; gi < groups.length; gi++) {
    for (const opIdx of groups[gi]!.ops) producingGroup.set(graph.ops[opIdx]!.out, gi)
  }
  const consumers = new Map<number, Set<number>>()
  for (let gi = 0; gi < groups.length; gi++) {
    for (const opIdx of groups[gi]!.ops) {
      for (const tid of opInputs(graph.ops[opIdx]!)) {
        let s = consumers.get(tid)
        if (!s) { s = new Set(); consumers.set(tid, s) }
        s.add(gi)
      }
    }
  }
  const nodeTids = new Set<number>()
  for (const g of groups) nodeTids.add(g.output)
  for (const op of graph.ops) {
    if (op.kind === 'tensor_input' && consumers.has(op.out)) nodeTids.add(op.out)
  }
  const nodeByTid = new Map<number, DagNode>()
  const nodes: DagNode[] = []
  const ordered: number[] = []
  for (const op of graph.ops) {
    if (op.kind === 'tensor_input' && nodeTids.has(op.out)) ordered.push(op.out)
  }
  for (const g of groups) {
    if (nodeTids.has(g.output)) ordered.push(g.output)
  }
  for (const tid of ordered) {
    const t = graph.tensors[tid]!
    const gi = producingGroup.get(tid)
    const node: DagNode = {
      tensorId: tid,
      shape: t.shape,
      group: gi !== undefined ? groups[gi]! : null,
      inputs: [],
    }
    nodes.push(node)
    nodeByTid.set(tid, node)
  }
  for (const node of nodes) {
    if (!node.group) continue
    const inGroupOps = new Set(node.group.ops)
    const seen = new Set<number>()
    for (const opIdx of node.group.ops) {
      for (const tid of opInputs(graph.ops[opIdx]!)) {
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

// ===== DOT generation =======================================================

function escapeXml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

// Greedy word-wrap at code break-points so HTML labels render multi-line.
function wrapForLabel(text: string, maxChars: number): string {
  const tokens: string[] = []
  let buf = ''
  for (const ch of text) {
    buf += ch
    if (ch === ' ' || ch === ',' || ch === '(' || ch === '.') {
      tokens.push(buf)
      buf = ''
    }
  }
  if (buf) tokens.push(buf)
  const lines: string[] = []
  let line = ''
  for (const t of tokens) {
    if (line.length + t.length > maxChars && line.length > 0) {
      lines.push(line.trimEnd())
      line = t.trimStart()
    } else {
      line += t
    }
  }
  if (line.trim()) lines.push(line.trimEnd())
  return lines.map(escapeXml).join('<BR/>')
}

function shapeHtmlLabel(shape: Shape, resolveDim: (size: number) => ResolvedDim): string {
  if (shape.length === 0) return '[]'
  const parts = shape.map(d => {
    const r = resolveDim(d)
    return `<FONT COLOR="${r.color}"><B>${escapeXml(r.name)}</B></FONT>`
  })
  return `[${parts.join(', ')}]`
}

function resolveSourceLabel(
  graph: Graph,
  node: DagNode,
  forwardOps: Set<number>,
  srcCache: Map<string, string[]>,
): string {
  const t = graph.tensors[node.tensorId]!
  const leafOp = t.source !== null ? graph.ops[t.source]! : null
  if (leafOp && (leafOp.kind === 'tensor_input' || leafOp.kind === 'state_input' || leafOp.kind === 'param_input')) {
    return describeLibraryOp(leafOp)
  }
  if (node.group && node.group.ops.every(idx => !forwardOps.has(idx))) {
    const lastOp = graph.ops[node.group.ops[node.group.ops.length - 1]!]!
    return describeLibraryOp(lastOp)
  }
  const frame = node.group?.frame
  if (frame) {
    const src = getSourceLine(srcCache, frame.url, frame.line)
    return src ? src.trim() : `${frame.fn}:${frame.line}`
  }
  return '(unattributed)'
}

function buildDOT(
  graph: Graph,
  dag: DagNode[],
  forwardOps: Set<number>,
  resolveDim: (size: number) => ResolvedDim,
  srcCache: Map<string, string[]>,
): string {
  const lines: string[] = [
    'digraph G {',
    '  rankdir=TB',
    '  bgcolor="transparent"',
    // Tight outer padding — Graphviz's default leaves visible blank space at
    // the top and bottom of the SVG bounding box; 0.05" is just a hair of
    // breathing room.
    '  pad=0.05',
    // fontname is "Courier" so Graphviz's bundled font metrics match what the
    // browser renders (the WASM build doesn't ship ui-monospace/Menlo/etc., so
    // those names cause width-mismatch — text rendered wider than the cell).
    // Sharp corners — Graphviz aims arrow tips at the unrounded box edge, so
    // rounded corners leave a visible gap between the curve and the arrowhead.
    '  node [shape=box style=filled fillcolor=white fontname="Courier" fontsize=11 margin="0.18,0.09" penwidth=1.4]',
    '  edge [penwidth=1.5]',
    '  nodesep=0.25',
    '  ranksep=0.3',
  ]
  for (const node of dag) {
    const tid = node.tensorId
    const sourceText = resolveSourceLabel(graph, node, forwardOps, srcCache)
    const wrapped = wrapForLabel(sourceText, 36)
    const shapeHtml = shapeHtmlLabel(node.shape, resolveDim)
    const color = tensorColor(tid)
    const label = `<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="3" CELLSPACING="0">` +
      `<TR><TD>${wrapped}</TD></TR>` +
      `<TR><TD>${shapeHtml}</TD></TR>` +
      `</TABLE>>`
    lines.push(`  "t${tid}" [label=${label} color="${color}" class="ti-${tid}"]`)
  }
  for (const node of dag) {
    const to = node.tensorId
    for (const inTid of node.inputs) {
      const color = tensorColor(inTid)
      lines.push(`  "t${inTid}" -> "t${to}" [color="${color}" class="ti-${inTid}"]`)
    }
  }
  lines.push('}')
  return lines.join('\n')
}

// ===== Hover trace ==========================================================
// Graphviz emits `class="ti-N"` on every node/edge for tensor N. On hover we
// add `tg-viewer-highlight` to every element sharing the same class.

function tagCategories(svg: SVGElement, graph: Graph, dag: DagNode[], forwardOps: Set<number>): void {
  for (const node of dag) {
    const cat = categorize(graph, node, forwardOps)
    for (const el of svg.querySelectorAll(`.ti-${node.tensorId}`)) {
      el.setAttribute('data-cat', cat)
    }
  }
}

function attachHoverTrace(svg: SVGElement): void {
  const elements = svg.querySelectorAll<SVGElement>('[class*="ti-"]')
  const clear = (): void => {
    for (const e of svg.querySelectorAll('.tg-viewer-highlight')) e.classList.remove('tg-viewer-highlight')
  }
  for (const el of elements) {
    const cls = Array.from(el.classList).find(c => c.startsWith('ti-'))
    if (!cls) continue
    el.addEventListener('mouseenter', () => {
      for (const m of svg.querySelectorAll('.' + cls)) m.classList.add('tg-viewer-highlight')
    })
    el.addEventListener('mouseleave', clear)
  }
}

// ===== Chrome ===============================================================
// Scoped CSS injected once per document; selectors live under `.tg-viewer`.

const CSS = `
.tg-viewer { font-family: ui-sans-serif, system-ui, sans-serif; }
.tg-viewer .tg-viewer-panel { border: 1px solid #ddd; border-radius: 8px; background: #fff; }
.tg-viewer .tg-viewer-controls { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; padding: 0.75rem 1rem; }
.tg-viewer .tg-viewer-controls label { font-size: 0.9rem; color: #444; display: flex; align-items: center; gap: 0.35rem; }
.tg-viewer .tg-viewer-legend-title { font-size: 0.8rem; margin: 0 1rem 0.4rem; color: #888; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }
.tg-viewer .tg-viewer-legend { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; font-size: 0.85rem; padding: 0 1rem 0.75rem; }
.tg-viewer .tg-viewer-legend-item { display: inline-flex; align-items: center; gap: 0.35rem; }
.tg-viewer .tg-viewer-legend-label { color: #555; font-size: 0.82rem; }
.tg-viewer .tg-viewer-dim-chip { padding: 0.05rem 0.45rem; border-radius: 4px; color: #fff; font-family: ui-monospace, monospace; font-size: 0.72rem; line-height: 1.4; }
.tg-viewer .tg-viewer-status { color: #888; font-size: 0.85rem; padding: 0.5rem 1rem; }
.tg-viewer .tg-viewer-canvas { border-top: 1px solid #eee; background: #fafbfc; width: 100%; }
.tg-viewer .tg-viewer-canvas > svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }
.tg-viewer .tg-viewer-canvas svg [class*="ti-"] { transition: filter 0.1s ease; }
.tg-viewer .tg-viewer-highlight { filter: drop-shadow(0 0 4px #f5a623); }

.tg-viewer .tg-viewer-filters { display: flex; gap: 0.4rem; align-items: center; padding: 0 1rem 0.6rem; flex-wrap: wrap; }
.tg-viewer .tg-viewer-filters-label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; margin-right: 0.2rem; }
.tg-viewer .tg-viewer-filter-chip {
  padding: 0.15rem 0.55rem;
  border-radius: 4px;
  border: 1px solid #ccc;
  background: #fff;
  font: inherit;
  font-size: 0.78rem;
  cursor: pointer;
  color: #444;
}
.tg-viewer .tg-viewer-filter-chip:hover { border-color: #888; }
.tg-viewer .tg-viewer-filter-chip[aria-pressed="true"] {
  background: #1e293b; color: #fff; border-color: #1e293b;
}

/* Dye highlighting: non-matching nodes get [data-greyed] set by JS. The
   rules below override every Graphviz-emitted color (stroke / fill / text)
   to a uniform gray so all non-selected nodes look identical regardless of
   their original hue. White node backgrounds shift to light gray for added
   contrast against the selected (full-color, white-bg) nodes. CSS overrides
   SVG presentation attributes naturally — no !important needed. */
.tg-viewer [data-greyed] path[fill="#ffffff"]    { fill:   #d8d8d8; }
.tg-viewer [data-greyed] path[stroke]            { stroke: #a8a8a8; }
.tg-viewer [data-greyed] polygon[stroke]         { stroke: #a8a8a8; }
.tg-viewer [data-greyed] polygon[fill]:not([fill="#ffffff"]):not([fill="none"]) { fill: #a8a8a8; }
.tg-viewer [data-greyed] text                    { fill:   #666666; }
`

function ensureStyles(): void {
  if (document.getElementById('tg-viewer-styles')) return
  const style = document.createElement('style')
  style.id = 'tg-viewer-styles'
  style.textContent = CSS
  document.head.appendChild(style)
}

type Chrome = {
  forwardRadio: HTMLInputElement
  fullRadio: HTMLInputElement
  status: HTMLElement
  canvas: HTMLElement
  active: Set<NodeCategory>
}

function applyDye(svg: SVGElement, active: Set<NodeCategory>): void {
  // Toggle `data-greyed` on every tagged element. CSS does the rest.
  for (const el of svg.querySelectorAll<SVGElement>('[data-cat]')) {
    const cat = el.getAttribute('data-cat') as NodeCategory
    if (active.size === 0 || active.has(cat)) el.removeAttribute('data-greyed')
    else el.setAttribute('data-greyed', '')
  }
}

function buildChrome(
  container: HTMLElement,
  legend: { name: string; desc: string; size: number; color: string }[],
  initialMode: 'forward' | 'full',
): Chrome {
  container.classList.add('tg-viewer')
  container.innerHTML = ''

  // Pre-create the canvas element so chip handlers can refer to it.
  const canvas = document.createElement('div')
  canvas.className = 'tg-viewer-canvas'

  const top = document.createElement('div')
  top.className = 'tg-viewer-panel'
  container.appendChild(top)

  const controls = document.createElement('div')
  controls.className = 'tg-viewer-controls'
  top.appendChild(controls)

  const mkRadio = (value: 'forward' | 'full', label: string): HTMLInputElement => {
    const wrap = document.createElement('label')
    const input = document.createElement('input')
    input.type = 'radio'
    input.name = 'tg-viewer-mode-' + Math.random().toString(36).slice(2, 8)
    input.value = value
    input.checked = value === initialMode
    wrap.appendChild(input)
    wrap.appendChild(document.createTextNode(' ' + label))
    controls.appendChild(wrap)
    return input
  }
  const forwardRadio = mkRadio('forward', 'Forward only')
  const fullRadio = mkRadio('full', 'Full (incl. autograd + optimizer)')
  // Same name so they're mutually exclusive.
  fullRadio.name = forwardRadio.name

  // Filter chips row — dye highlighting for node categories.
  const filtersRow = document.createElement('div')
  filtersRow.className = 'tg-viewer-filters'
  const filtersLabel = document.createElement('span')
  filtersLabel.className = 'tg-viewer-filters-label'
  filtersLabel.textContent = 'Highlight:'
  filtersRow.appendChild(filtersLabel)
  const active = new Set<NodeCategory>()
  const refreshDye = (): void => {
    const svg = canvas.querySelector('svg')
    if (svg) applyDye(svg as unknown as SVGElement, active)
  }
  for (const { key, label } of CATEGORY_LABELS) {
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'tg-viewer-filter-chip'
    btn.dataset.cat = key
    btn.setAttribute('aria-pressed', 'false')
    btn.textContent = label
    btn.addEventListener('click', () => {
      if (active.has(key)) { active.delete(key); btn.setAttribute('aria-pressed', 'false') }
      else { active.add(key); btn.setAttribute('aria-pressed', 'true') }
      refreshDye()
    })
    filtersRow.appendChild(btn)
  }
  top.appendChild(filtersRow)

  if (legend.length > 0) {
    const legendTitle = document.createElement('h2')
    legendTitle.className = 'tg-viewer-legend-title'
    legendTitle.textContent = 'Dim legend'
    top.appendChild(legendTitle)

    const legendEl = document.createElement('div')
    legendEl.className = 'tg-viewer-legend'
    top.appendChild(legendEl)
    for (const d of legend) {
      const item = document.createElement('span')
      item.className = 'tg-viewer-legend-item'
      const chip = document.createElement('span')
      chip.className = 'tg-viewer-dim-chip'
      chip.style.background = d.color
      chip.textContent = d.name
      item.appendChild(chip)
      const label = document.createElement('span')
      label.className = 'tg-viewer-legend-label'
      label.textContent = d.desc ? `${d.desc} = ${d.size}` : `= ${d.size}`
      item.appendChild(label)
      legendEl.appendChild(item)
    }
  }

  const bottom = document.createElement('div')
  bottom.className = 'tg-viewer-panel'
  bottom.style.marginTop = '1rem'
  bottom.style.padding = '0'
  container.appendChild(bottom)

  const status = document.createElement('div')
  status.className = 'tg-viewer-status'
  status.textContent = 'Rendering...'
  bottom.appendChild(status)

  bottom.appendChild(canvas)

  return { forwardRadio, fullRadio, status, canvas, active }
}

// ===== Entry point ==========================================================

export async function renderIRViewer(opts: RenderIRViewerOptions): Promise<void> {
  ensureStyles()
  const initialMode = opts.initialMode ?? 'forward'
  const { resolve, legend } = buildDimResolver(opts.graph, opts.dims)
  const chrome = buildChrome(opts.container, legend, initialMode)
  chrome.status.textContent = 'Loading Graphviz...'

  const [viz, srcCache] = await Promise.all([
    instance(),
    (async () => {
      const urls = new Set<string>()
      for (const op of opts.graph.ops) {
        const f = firstUserFrame(opts.graph.tensors[op.out]!.site)
        if (f) urls.add(f.url)
      }
      return loadSources(urls)
    })(),
  ])

  const fwd = forwardReachable(opts.graph, opts.graph.outputs)
  const all = new Set<number>(opts.graph.ops.map((_, i) => i))

  const showFor = (mode: 'forward' | 'full'): void => {
    const included = mode === 'forward' ? fwd : all
    const groups = groupBySourceLine(opts.graph, included)
    const dag = buildDag(opts.graph, groups)
    const dot = buildDOT(opts.graph, dag, fwd, resolve, srcCache)
    const svg = viz.renderSVGElement(dot)
    chrome.canvas.innerHTML = ''
    chrome.canvas.appendChild(svg)
    tagCategories(svg, opts.graph, dag, fwd)
    applyDye(svg, chrome.active)
    attachHoverTrace(svg)
    const kernelsPart = opts.kernelCount !== undefined ? ` / ${opts.kernelCount} kernels` : ''
    chrome.status.textContent =
      `${groups.length} statements / ${dag.length} tensor nodes / ${opts.graph.tensors.length} tensors total${kernelsPart}`
  }
  chrome.forwardRadio.addEventListener('change', () => { if (chrome.forwardRadio.checked) showFor('forward') })
  chrome.fullRadio.addEventListener('change', () => { if (chrome.fullRadio.checked) showFor('full') })
  showFor(initialMode)
}
