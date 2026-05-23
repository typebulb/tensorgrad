---
format: typebulb/v1
name: Vectors and Tensors
---

**code.tsx**

```tsx
import {
  App, Component, div, h1, h3, p, span, pre, button, input, a, ul, li,
  svg, line, polygon, text, g, rect,
  type VElement, type HValues,
} from 'domeleon'

// ---------- Shared helpers ----------
const range = (n: number) => Array.from({ length: n }, (_, i) => i)

const fmt = (x: number, d = 2): string => {
  if (Number.isNaN(x)) return 'NaN'
  if (Math.abs(x) < 1e-9) return '0'
  return x.toFixed(d)
}

const mono   = (...c: HValues[]) => span({ class: 'mono' }, c)
const italic = (...c: HValues[]) => span({ class: 'italic' }, c)

// Centered monospace formula, the standard way to surface an equation inside .intro prose.
const formula = (text: string) => div({ class: 'formula-display' }, mono(text))

// Vector dot product: Σᵢ aᵢ·bᵢ. Works on any number[] (Vec2, Vec5, vec, ...).
const dotVec = (a: number[], b: number[]) =>
  a.reduce((s, x, i) => s + x * b[i]!, 0)

// Tri-state color from a signed value (eps avoids floor-noise flicker at zero).
const signColor = (v: number, eps = 0.01) =>
  v > eps ? 'var(--positive)' : v < -eps ? 'var(--negative)' : 'var(--zero)'

// CSS color-mix tint: blend `color` at `pct`% with the cell background.
const tintBg = (color: string, pct: number) =>
  `color-mix(in srgb, ${color} ${pct}%, var(--cell-bg))`

// ---------- Number-cell coloring + grid helpers (shared across Shapes / Embedding / Matmul) ----------
const numClass = (v: number) => v === 0 ? 'num-zero' : v > 0 ? 'num-pos' : 'num-neg'

const numCell = (v: number) => div({ class: ['mat-cell', numClass(v)] }, String(v))

// CSS-grid container of mat-cells with cols equal-width columns. Callers
// supply the cells, so highlight/selection variants can build their own.
function matGrid(cols: number, children: HValues) {
  return div({
    class: 'mat-grid',
    style: { gridTemplateColumns: `repeat(${cols}, var(--cell-w))` },
  }, children)
}

// Labeled-matrix primitive shared across the Attention X / scores / attn
// diagrams and the Softmax stepwise grid. Caller supplies pre-rendered value
// cells; this wraps them in the same bordered .mat-grid used by the other
// tabs, then places optional col-labels (top), row-labels (left), and
// per-row annotations (right) in an outer 3-col grid that auto-sizes to
// align with the cell tracks.
function labeledMatrix(opts: {
  rowLabels?: HValues[]
  colLabels?: HValues[]
  rowAnnotations?: HValues[]
  rows: HValues[][]
}) {
  const { rowLabels, colLabels, rowAnnotations, rows } = opts
  const nCols = rows[0]!.length
  const nRows = rows.length
  const cellRow = colLabels ? '2' : '1'

  const items: HValues[] = []

  if (colLabels) {
    items.push(div({
      class: 'lm-col-labels',
      style: {
        gridColumn: '2', gridRow: '1',
        gridTemplateColumns: `repeat(${nCols}, var(--cell-w))`,
      },
    }, colLabels.map(c => div({ class: 'lm-col-label' }, c))))
  }

  if (rowLabels) {
    items.push(div({
      class: 'lm-row-labels',
      style: {
        gridColumn: '1', gridRow: cellRow,
        gridTemplateRows: `repeat(${nRows}, var(--cell-h))`,
      },
    }, rowLabels.map(l => div({ class: 'lm-row-label' }, l))))
  }

  items.push(div({
    class: 'lm-cells-wrap',
    style: { gridColumn: '2', gridRow: cellRow },
  }, matGrid(nCols, rows.flatMap(r => r))))

  if (rowAnnotations) {
    items.push(div({
      class: 'lm-row-annotations',
      style: {
        gridColumn: '3', gridRow: cellRow,
        gridTemplateRows: `repeat(${nRows}, var(--cell-h))`,
      },
    }, rowAnnotations.map(a => div({ class: 'lm-row-annotation' }, a))))
  }

  return div({ class: 'labeled-matrix' },
    div({ class: 'lm-grid' }, items),
  )
}

// Plain matrix of numbers, sign-colored.
function numGrid(rows: number[][]) {
  return matGrid(rows[0]!.length, rows.flatMap(r => r.map(numCell)))
}

function rowCells(values: number[]) {
  return div({ class: 'cell-row' }, values.map(numCell))
}

// 5 tokens × 4 dims. Four named axes used:
//   dim 0 = pet         (cat, dog)
//   dim 1 = cat-ness    (cat, catwoman; dog mildly anti)
//   dim 2 = temperature (hot positive, cold negative — same axis, opposite signs)
//   dim 3 = sexy        (hot, catwoman)
// dog's -1 on cat-ness ensures dog·dog > dog·cat: without it, a vector that's
// a directional subset of another ties its raw self-dot, which reads weird in
// the widget's diagonal.
// catwoman bridges two clusters (cat-ness AND sexy) — a polysemous token.
const NAMED_EMBED_TABLE: { name: string; vec: number[] }[] = [
  { name: 'cat',       vec: [ 3,  2,  0,  0] },
  { name: 'dog',       vec: [ 3, -1,  0,  0] },
  { name: 'hot',       vec: [ 0,  0,  3,  2] },
  { name: 'cold',      vec: [ 0,  0, -3,  0] },
  { name: 'catwoman',  vec: [ 0,  3,  0,  2] },
]

// Shared worked example for the matmul tab and the Shapes tab's matmul-stack
// viz. Lives at module scope so both sections reference the same source of
// truth instead of one section reaching into the other.
const MATMUL_EXAMPLE = (() => {
  const A: number[][] = [
    [1,  2, -1],
    [0,  1,  1],
  ]
  const B: number[][] = [
    [ 1,  0,  2, -1],
    [-1,  2,  0,  1],
    [ 1,  1, -1,  0],
  ]
  const C = A.map(rowA => range(B[0]!.length).map(j =>
    dotVec(rowA, B.map(r => r[j]!)),
  ))
  return { A, B, C }
})()

type Vec2 = [number, number]

// A and B from angle θ (degrees, between them); A lies on the +x axis.
// C = A + B is the parallelogram-rule sum, returned for callers that want it.
function vectorsFromAngle(angleDeg: number, aLen: number, bLen: number): { A: Vec2; B: Vec2; C: Vec2 } {
  const θ = angleDeg * Math.PI / 180
  const A: Vec2 = [aLen, 0]
  const B: Vec2 = [bLen * Math.cos(θ), bLen * Math.sin(θ)]
  return { A, B, C: [A[0] + B[0], A[1] + B[1]] }
}

// Standard "label + value above, slider below" control. Returns an array of
// two VElements (auto-flattened by domeleon).
function sliderRow(opts: {
  label: string
  value: string
  min: number; max: number; step: number
  current: number
  onChange: (v: number) => void
}) {
  return [
    div({ class: 'control-row' },
      div({ class: 'control-label' }, opts.label),
      div({ class: 'control-value' }, opts.value),
    ),
    input({
      type: 'range',
      min: String(opts.min), max: String(opts.max), step: String(opts.step),
      value: String(opts.current),
      class: 'slider',
      onInput: (e: any) => opts.onChange(Number(e.target.value)),
    }),
  ]
}

// Background tint by relative magnitude in row: winner = green, loser = red.
function relMagBg(value: number, rowMin: number, rowMax: number): string | undefined {
  if (rowMax === rowMin) return undefined
  const t = (value - rowMin) / (rowMax - rowMin)
  if (t > 0.55) return tintBg('var(--positive)', Math.round((t - 0.5) * 2 * 35))
  if (t < 0.45) return tintBg('var(--negative)', Math.round((0.5 - t) * 2 * 35))
  return undefined
}

// A row of value cells, tinted by each value's relative position in the row
// (green toward max, red toward min). Used by the Softmax stepwise grid.
function magnitudeCells(values: number[], digits: number, cellClass: string | string[] = 'mat-cell') {
  const rowMin = Math.min(...values)
  const rowMax = Math.max(...values)
  return values.map(v => {
    const bg = relMagBg(v, rowMin, rowMax)
    return div({
      class: cellClass,
      style: bg ? { background: bg } : undefined,
    }, fmt(v, digits))
  })
}

// One "label = value" row in a side panel, with optional colors / weight.
function detailRow(opts: {
  label: HValues
  value: HValues
  lblColor?: string
  valColor?: string
  valWeight?: string
}) {
  return div({ class: 'detail-row' },
    span({
      class: 'detail-lbl',
      style: opts.lblColor ? { color: opts.lblColor, fontWeight: '600' } : undefined,
    }, opts.label),
    span('='),
    span({
      class: 'detail-val',
      style: (opts.valColor || opts.valWeight)
        ? { color: opts.valColor, fontWeight: opts.valWeight }
        : undefined,
    }, opts.value),
  )
}

// ---------- Plot geometry ----------
const PLOT = 360
const ORIGIN = PLOT / 2
const UNIT = 36   // pixels per unit; ~5 units visible in each direction

function plotToScreen(x: number, y: number): [number, number] {
  return [ORIGIN + x * UNIT, ORIGIN - y * UNIT]
}

function axes() {
  return [
    line({ x1: 0, y1: ORIGIN, x2: PLOT, y2: ORIGIN, stroke: 'var(--axis)', strokeWidth: 1 }),
    line({ x1: ORIGIN, y1: 0, x2: ORIGIN, y2: PLOT, stroke: 'var(--axis)', strokeWidth: 1 }),
  ]
}

type ArrowOpts = {
  color: string
  label?: string
  labelOffset?: { x: number; y: number }
  labelAtMidpoint?: boolean
  labelAnchor?: 'start' | 'middle' | 'end'
  dashed?: boolean
}

const ARROW_HEAD = 10

function arrow(from: [number, number], to: [number, number], opts: ArrowOpts) {
  const [x1, y1] = plotToScreen(from[0], from[1])
  const [x2, y2] = plotToScreen(to[0], to[1])
  const dx = x2 - x1, dy = y2 - y1
  const len = Math.sqrt(dx * dx + dy * dy)
  const ux = len > 0 ? dx / len : 0
  const uy = len > 0 ? dy / len : 0
  const baseX = x2 - ux * ARROW_HEAD
  const baseY = y2 - uy * ARROW_HEAD
  const perpX = -uy * (ARROW_HEAD * 0.45)
  const perpY = ux * (ARROW_HEAD * 0.45)
  return g(
    line({
      x1, y1, x2: baseX, y2: baseY,
      stroke: opts.color,
      strokeWidth: 2.5,
      strokeDashArray: opts.dashed ? '6 4' : undefined,
    }),
    polygon({
      points: `${x2},${y2} ${baseX + perpX},${baseY + perpY} ${baseX - perpX},${baseY - perpY}`,
      fill: opts.color,
    }),
    opts.label ? text({
      x: (opts.labelAtMidpoint ? (x1 + x2) / 2 : x2) + (opts.labelOffset?.x ?? 8),
      y: (opts.labelAtMidpoint ? (y1 + y2) / 2 : y2) + (opts.labelOffset?.y ?? -8),
      textAnchor: opts.labelAnchor,
      fill: opts.color,
      fontSize: 14,
      fontFamily: 'monospace',
      fontWeight: 600,
    }, opts.label) : null,
  )
}

// Shared scaffold for the Vectors / Dot vizzes: plot on the left
// (axes + caller-supplied svg content), side panel on the right (angle slider
// + caller-supplied detail + optional caption). All three drive the same
// 0..180° angle parameter.
function angleViz(opts: {
  angle: number
  setAngle: (a: number) => void
  sliderLabel?: string
  sliderMin?: number
  sliderMax?: number
  svgContent: HValues
  detail: HValues
  caption?: HValues
}) {
  return div({ class: 'viz-row' },
    svg({ class: 'plot', viewBox: `0 0 ${PLOT} ${PLOT}`, width: PLOT, height: PLOT },
      axes(),
      opts.svgContent,
    ),
    div({ class: 'side-panel' },
      sliderRow({
        label: opts.sliderLabel ?? 'angle between A and B',
        value: `${opts.angle}°`,
        min: opts.sliderMin ?? 0,
        max: opts.sliderMax ?? 180,
        step: 1,
        current: opts.angle,
        onChange: opts.setAngle,
      }),
      opts.detail,
      opts.caption,
    ),
  )
}

// ---------- Spec-excerpt block ----------
// `paragraphs` items alternate plain text / monospace fragments: indices 0,2,4 are
// plain prose; 1,3,5 are wrapped in <span class="mono">. Lets us write inline code
// references without nesting span() calls everywhere.
function specBox(code: string, paragraphs: string[][]) {
  return div({ class: 'spec-anchor' },
    pre({ class: 'spec-code' }, code),
    paragraphs.map(parts =>
      p({ class: 'spec-caption' },
        parts.map((s, i) => i % 2 === 1 ? mono(s) : s),
      ),
    ),
  )
}

// ---------- TabControl ----------
type TabDef = { key: string; label: string; content: HValues; hideSubBar?: boolean }

class TabControl extends Component {
  selected: string | null = null

  view(props: { tabs: TabDef[]; subBar?: HValues }) {
    const { tabs, subBar } = props
    const sel = this.selected ?? tabs[0]!.key
    const showSubBar = !tabs.find(t => t.key === sel)?.hideSubBar
    return div({ class: 'tab-control' },
      div({ class: 'tab-bar' },
        tabs.map(t => button({
          class: ['tab-btn', sel === t.key && 'active'],
          onClick: () => {
            if (this.selected === t.key) return
            this.selected = t.key
            this.update()
          },
        }, t.label)),
      ),
      showSubBar ? subBar : null,
      tabs.map(t => div({
        class: 'tab-pane',
        style: { display: sel === t.key ? 'block' : 'none' },
      }, t.content)),
    )
  }
}

// ---------- Root contract (sections read this) ----------
type ViewMode = 'conceptual' | 'tensorgrad'

interface IRoot {
  viewMode: ViewMode
}

// Every content section has the same root-accessor + conceptual/tensorgrad
// dispatcher; this base class factors both out so sections only have to
// implement the two view methods.
abstract class ModeSection extends Component {
  get root() { return this.ctx.root as any as IRoot }
  view() {
    return this.root.viewMode === 'tensorgrad' ? this.tensorgradView() : this.conceptualView()
  }
  abstract conceptualView(): VElement
  abstract tensorgradView(): VElement
}

// Shared by sections whose conceptual view is driven by an angle slider in [0, 180].
abstract class AngleSection extends ModeSection {
  abstract angle: number
  setAngle(a: number) {
    this.angle = a
    this.update()
  }
}

// ---------- Shapes section ----------
class ShapesSection extends ModeSection {
  conceptualView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'A ',
        italic('tensor'),
        ' is an N-dimensional array of numbers with a fixed shape. A scalar is rank 0, a vector is rank 1, a matrix is rank 2, and from rank 3 upward you stack matrices, then stacks of matrices, and so on. Shapes are written as bracketed dim lists: ',
        mono('[3]'),
        ', ',
        mono('[3, 4]'),
        ', ',
        mono('[B, T, D]'),
        '.',
      ),

      div({ class: 'shape-ladder' },
        this.shapeRung('scalar', 'rank 0', numCell(5)),
        this.shapeRung('vector', 'rank 1, shape [4]',
          rowCells([1, 4, -2, 3]),
        ),
        this.shapeRung('matrix', 'rank 2, shape [3, 4]',
          numGrid([[1, 2, -1, 0], [0, 1, 2, -1], [-1, 0, 1, 2]]),
        ),
        this.shapeRung('3-tensor', 'rank 3, shape [3, 2, 3]',
          div({ class: 'shape-stack' },
            [
              [[ 1, -1,  0], [ 0,  2,  1]],
              [[ 2,  0, -1], [ 1, -1,  2]],
              [[-1,  1,  2], [ 0,  1, -1]],
            ].map(numGrid),
          ),
        ),
      ),

      this.matmulStackViz(),

    )
  }

  matmulStackViz() {
    // Reuses the matmul tab's worked example (MATMUL_EXAMPLE). The front
    // layer is the real matmul; the two blank layers behind each grid stand in
    // for outer dimensions — the same matmul running at every outer index.
    const { A, B, C } = MATMUL_EXAMPLE

    const blankGrid = (rows: number, cols: number) =>
      matGrid(cols, range(rows * cols).map(() => div({ class: 'mat-cell' })))

    const stack = (data: number[][]) => {
      const rows = data.length, cols = data[0]!.length
      return div({
        class: 'mm-stack',
        style: {
          width:  `calc(var(--cell-w) * ${cols} + 24px)`,
          height: `calc(var(--cell-h) * ${rows} + 24px)`,
        },
      },
        numGrid(data),
        blankGrid(rows, cols),
        blankGrid(rows, cols),
      )
    }

    const item = (name: string, data: number[][]) => div({ class: 'mm-item' },
      div({ class: 'mm-label' }, mono(`${name}   [${data.length}, ${data[0]!.length}]`)),
      stack(data),
    )

    return div({ class: 'mm-viz' },
      div({ class: 'intro' },
        'Below is a 2D matmul: ',
        mono('[2, 3] · [3, 4] = [2, 4]'),
        '. The matmul tab will walk through exactly this example, fully expanded — but only the front layer of each grid, where the math happens.',
      ),

      div({ class: 'intro' },
        'The two blank layers behind each grid stand in for the extra outer dimensions you find in real tensor code. The matmul itself is only defined on a pair of 2D matrices; when the inputs have more dimensions, the same matmul runs at every outer index, independently. Most ops follow this rule — the exceptions are ops designed to operate on dimensions themselves, like ',
        mono('swapAxes'),
        '.',
      ),

      div({ class: 'mm-row' },
        item('A', A),
        div({ class: 'mm-op-glyph' }, '·'),
        item('B', B),
        div({ class: 'mm-op-glyph' }, '='),
        item('C', C),
      ),

      div({ class: 'intro' },
        'In practice, the outermost dimension is usually a batch — independent inputs run together. The rule above is what makes batching fast: operations across outer indices are independent, so tensor libraries parallelize them across GPU cores.',
      ),
    )
  }

  shapeRung(label: string, notation: string, viz: HValues) {
    return div({ class: 'shape-rung' },
      div({ class: 'shape-rung-label' }, label),
      viz,
      div({ class: 'shape-rung-notation' }, notation),
    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'In tensorgrad, every parameter declares its shape. Shapes are part of the type — downstream ops are checked against them when the model is compiled to a GPU kernel.',
      ),

      specBox(
        `class MyModel extends Module {\n  W = this.param([3, 4])   // a learnable 2D tensor: 3 rows of 4 numbers\n}`,
        [
          [
            '',
            'this.param([...])', ' declares a learnable tensor inside a Module subclass. Training fills in the values via gradient descent. The shape is fixed at compile time.',
          ],
        ],
      ),

      specBox(
        `const x = ones([3, 4])    // shape [3, 4]\nconst y = ones([4])       // shape [4]\nconst z = add(x, y)       // shape [3, 4]; the [4] vector is reused across all 3 rows`,
        [
          [
            'When two tensors of different ranks meet, the shorter shape lines up against the end of the longer one, and the missing leading axes are repeated implicitly. The shorter tensor is said to ',
            'broadcast', ' across those axes. Here ',
            'y', ' broadcasts across the leading axis of ',
            'x', '.',
          ],
        ],
      ),
    )
  }
}

// ---------- Embedding section ----------
class EmbeddingSection extends ModeSection {
  selectedTokenIdx: number = 1

  selectToken(i: number) {
    this.selectedTokenIdx = i
    this.update()
  }

  conceptualView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'An embedding is a table of vectors, with one row per item in a vocabulary. In LLMs that\'s tokens, usually word fragments, but for simplicity just think of them as words. Because a vector has many dimensions, it can carry several aspects of a word at once. This allows vectors to encode the relationships between words.',
      ),

      this.embeddingExplorer(),

      div({ class: 'intro' },
        'Each row is a list of numbers — a direction in 4-dimensional space. The dot products above measure their alignment: related words point in similar directions, unrelated ones land near-perpendicular, opposites point oppositely.',
      ),

      div({ class: 'intro' },
        'Real LLMs go further: they encode more signals than there are dimensions — sometimes by an order of magnitude or more. This works because the signals are sparse: any one input activates only a tiny fraction of them, so two signals that almost never co-occur can share a direction with negligible interference. The phenomenon is called ',
        italic('superposition'),
        '.',
      ),

    )
  }

  embeddingExplorer() {
    const TABLE = NAMED_EMBED_TABLE
    const sel = this.selectedTokenIdx
    const selRow = TABLE[sel]!
    const dots = TABLE.map(t => dotVec(t.vec, selRow.vec))
    // Fixed scale across all possible selections so the same dot value
    // produces the same bar width when the user clicks between tokens.
    const allDots = TABLE.flatMap(a => TABLE.map(b => dotVec(a.vec, b.vec)))
    const maxAbsDot = Math.max(1, ...allDots.map(Math.abs))
    const otherProds = TABLE.flatMap((t, i) =>
      i === sel ? [] : t.vec.map((v, j) => selRow.vec[j]! * v),
    )
    const maxAbsProd = Math.max(1, ...otherProds.map(Math.abs))

    const productBg = (prod: number) => {
      if (prod === 0) return undefined
      const pct = Math.round((Math.abs(prod) / maxAbsProd) * 55)
      return tintBg(prod > 0 ? 'var(--positive)' : 'var(--negative)', pct)
    }

    return div({ class: 'embed-block' },
      div({ class: 'embed-prose' },
        'Click a token to look up its row — that\'s the embedding op. Each other row\'s similarity to it is shown on the right; the vector cells tint by how much each dim contributes to that similarity.',
      ),
      div({ class: 'embed-grid' },
        div({ class: 'embed-grid-head' },
          div(),
          div(),
          div({ class: 'embed-sim-head' },
            'Similarity to ',
            span({ class: 'embed-h-name' }, selRow.name),
          ),
        ),
        TABLE.map((t, i) => {
          const isSel = i === sel
          return div({
            class: ['embed-grid-row', isSel && 'is-sel'],
            onClick: () => this.selectToken(i),
          },
            div({ class: 'embed-token-name' }, t.name),
            div({ class: 'cell-row' },
              t.vec.map((v, j) => {
                const bg = isSel ? undefined : productBg(selRow.vec[j]! * v)
                return div({
                  class: ['mat-cell', numClass(v)],
                  style: bg ? { background: bg } : undefined,
                }, String(v))
              }),
            ),
            this.simBar(dots[i]!, maxAbsDot),
          )
        }),
      ),
    )
  }

  simBar(dot: number, maxAbs: number) {
    const t = maxAbs > 0 ? dot / maxAbs : 0
    const widthPct = Math.abs(t) * 50
    const offsetPct = dot >= 0 ? 50 : 50 - widthPct
    const color = signColor(dot)
    return div({ class: 'embed-sim' },
      div({ class: 'embed-sim-num', style: { color } }, String(dot)),
      div({ class: 'embed-sim-track' },
        div({ class: 'embed-sim-axis' }),
        widthPct > 0 ? div({
          class: 'embed-sim-bar',
          style: {
            left: `${offsetPct}%`,
            width: `${widthPct}%`,
            background: color,
          },
        }) : null,
      ),
    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        mono('embedding(table, indices)'),
        ' is an indexed lookup: every integer in ',
        mono('indices'),
        ' is replaced by the corresponding row from ',
        mono('table'),
        '. The shape gets promoted from integers to D-dim vectors in one op.',
      ),

      specBox(
        `class Embed extends Module {\n  table = this.param([12, 64])     // 12 learnable rows, each a 64-dim vector\n}\n\n// In a forward function, given ids as an i32 Tensor of shape [3]:\nconst out = embedding(p.table, ids)   // shape [3, 64] — the three fetched rows`,
        [
          [
            'Each integer in ',
            'ids', ' becomes the corresponding row of ',
            'table', '. The output shape is the input shape with one extra trailing axis equal to ',
            'D', '.',
          ],
        ],
      ),

      specBox(
        `// indices can have any leading shape — e.g., a batch of token sequences:\n// ids: i32 Tensor of shape [B, T]\nconst out = embedding(p.table, ids)   // shape [B, T, 64]`,
        [
          [
            'The same op vectorizes across any leading shape. A ',
            '[B, T]', ' tensor of integers becomes a ',
            '[B, T, 64]', ' tensor of vectors — exactly what a transformer needs to feed a batch of token sequences into the rest of the model.',
          ],
        ],
      ),
    )
  }
}

// ---------- Add section ----------
class AddSection extends AngleSection {
  angle: number = 60   // degrees between A and B

  conceptualView() {
    // A_LEN + B_LEN must fit inside the plot's ~5-unit half-extent — even at
    // θ=0° where B is colinear with A and the sum lands at A_LEN + B_LEN on
    // the x-axis. 2.5 + 2.0 = 4.5, comfortably inside.
    const { A, B, C } = vectorsFromAngle(this.angle, 2.5, 2.0)
    const fmtV = (v: Vec2) => `[${fmt(v[0])}, ${fmt(v[1])}]`

    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'A vector is a point in n-dimensional space. You can visualize this as an arrow from the origin, and express it as a list of numbers. Given two vectors of the same size, you can add them component-wise:',
        formula('A + B = [a₁+b₁, a₂+b₂]'),
        'Geometrically, this is tip-to-tail addition, as shown in the picture below. Tensorgrad code (and PyTorch, JAX) ultimately works on these, especially stacks of them.',
      ),

      angleViz({
        angle: this.angle,
        setAngle: a => this.setAngle(a),
        svgContent: [
          arrow([0, 0], A, { color: 'var(--a-color)', label: 'A' }),
          arrow(A, C, { color: 'var(--b-color)', label: 'B', labelOffset: { x: 8, y: -6 } }),
          arrow([0, 0], C, {
            color: 'var(--c-color)',
            dashed: true,
            label: 'A + B',
            labelAtMidpoint: true,
            labelAnchor: 'middle',
            labelOffset: { x: 0, y: 24 },
          }),
        ],
        detail: div({ class: 'detail' },
          detailRow({ label: 'A',     value: fmtV(A), lblColor: 'var(--a-color)' }),
          detailRow({ label: 'B',     value: fmtV(B), lblColor: 'var(--b-color)' }),
          detailRow({ label: 'A + B', value: fmtV(C), lblColor: 'var(--c-color)' }),
        ),
      }),

    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        mono('add(a, b)'),
        ' is element-wise addition over two same-shape tensors.',
      ),
      specBox(
        `const c = add(a, b)   // a: [64], b: [64]  →  c: [64]\n                      // c[i] = a[i] + b[i] for every i`,
        [
          [
            'Each output element is the sum of corresponding inputs. The shape is preserved.',
          ],
        ],
      ),
    )
  }
}

// ---------- Dot section ----------
class DotSection extends AngleSection {
  angle: number = 45   // degrees

  conceptualView() {
    const { A, B } = vectorsFromAngle(this.angle, 3, 2.5)
    const dot = dotVec(A, B)
    const Amag = Math.sqrt(dotVec(A, A))
    const Bmag = Math.sqrt(dotVec(B, B))
    const cosθ = Math.cos(this.angle * Math.PI / 180)
    const projX = B[0]

    const dotColor = signColor(dot)

    const [ox, oy] = plotToScreen(0, 0)
    const [bx, by] = plotToScreen(B[0], B[1])
    const [px, py] = plotToScreen(projX, 0)

    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'We compare the directions of two vectors with a dot product. This produces a single number measuring their alignment. Abstractly, this is relatedness. The same operation has a component-wise definition and a geometric one:',
        formula('A · B = a₁b₁ + a₂b₂ + ... + aₙbₙ'),
        formula('A · B = |A| |B| cos(θ)'),
        'where ',
        mono('|A|'),
        ' is the length of A and ',
        mono('θ'),
        ' is the angle between A and B. The length is Pythagoras generalized to n dimensions:',
        formula('|A| = √(a₁² + a₂² + ... + aₙ²)'),
        'In 1D this reduces to ordinary absolute value, which is why the bar notation extends. Setting B = A in either form gives the self-case: ',
        mono('A · A = |A|²'),
        ' — a vector dotted with itself is its squared magnitude.',
      ),

      angleViz({
        angle: this.angle,
        setAngle: a => this.setAngle(a),
        sliderLabel: 'angle θ',
        svgContent: [
          line({
            x1: ox, y1: oy, x2: px, y2: py,
            stroke: 'var(--accent)', strokeWidth: 5, strokeOpacity: 0.35, strokeLineCap: 'round',
          }),
          line({
            x1: bx, y1: by, x2: px, y2: py,
            stroke: 'var(--axis)', strokeWidth: 1, strokeDashArray: '4 3',
          }),
          arrow([0, 0], A, { color: 'var(--a-color)', label: 'A' }),
          arrow([0, 0], B, { color: 'var(--b-color)', label: 'B' }),
        ],
        detail: div({ class: 'detail' },
          detailRow({ label: 'A · B',         value: fmt(dot), valColor: dotColor, valWeight: '700' }),
          detailRow({ label: '|A|',           value: fmt(Amag) }),
          detailRow({ label: '|B|',           value: fmt(Bmag) }),
          detailRow({ label: 'cos θ',         value: fmt(cosθ, 3) }),
          detailRow({ label: '|A||B| cos θ',  value: fmt(Amag * Bmag * cosθ) }),
        ),
        caption: [
          div({ class: 'caption' }, this.stateDescription()),
          div({ class: 'caption' },
            'The faint indigo segment along A is the projection of B onto A\'s direction. Its signed length, scaled by |A|, equals A · B.',
          ),
        ],
      }),

      div({ class: 'intro' },
        'Dot product is the math of relatedness: positive for aligned, near-zero for unrelated, negative for opposite. The next tab uses this to give directions meaning.',
      ),

    )
  }

  stateDescription() {
    const θ = this.angle
    if (θ <= 10) {
      return [
        'A and B are nearly aligned. ',
        mono('A · B'),
        ' is near the maximum of ',
        mono('|A||B|'),
        '.',
      ]
    }
    if (θ >= 170) {
      return [
        'A and B are nearly opposite. ',
        mono('A · B'),
        ' is near the minimum of ',
        mono('−|A||B|'),
        '.',
      ]
    }
    if (θ >= 80 && θ <= 100) {
      return [
        'A and B are near-perpendicular. ',
        mono('A · B'),
        ' is small (only exactly 0 at 90°).',
      ]
    }
    if (θ < 80) {
      return [
        'A and B partially overlap. ',
        mono('A · B'),
        ' is positive — between 0 and |A||B|.',
      ]
    }
    return [
      'A and B partially oppose. ',
      mono('A · B'),
      ' is negative — between −|A||B| and 0.',
    ]
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'tensorgrad does not expose a standalone ',
        mono('dot'),
        ' op. Element-wise multiply, then sum:',
      ),
      specBox(
        `const d = sum(mul(a, b))   // a: [N], b: [N]  →  d: scalar\n                           // d = a[0]*b[0] + a[1]*b[1] + ... + a[N-1]*b[N-1]`,
        [
          [
            '',
            'mul', ' multiplies element-wise (',
            'mul(a, b)[i] = a[i] * b[i]', '); ',
            'sum', ' collapses the result to a single number. Together: the dot product.',
          ],
        ],
      ),
      div({ class: 'intro' },
        'In practice ML almost always wants many dot products at once, and those naturally live inside ',
        mono('matmul'),
        ' (explained later). That\'s why tensorgrad doesn\'t bother with a standalone one; just use ',
        mono('sum(mul(a, b))'),
        '. Some frameworks like PyTorch (',
        mono('torch.dot'),
        ') and JAX/NumPy (',
        mono('jnp.dot'),
        ' / ',
        mono('np.dot'),
        ') expose a standalone one, to cross the Ts and · the Is.',
      ),
    )
  }
}

// ---------- matmul section ----------
class MatmulSection extends ModeSection {
  selected: { row: number; col: number } | null = { row: 0, col: 0 }

  selectCell(row: number, col: number) {
    const s = this.selected
    this.selected = (s && s.row === row && s.col === col) ? null : { row, col }
    this.update()
  }

  conceptualView() {
    const { A, B, C } = MATMUL_EXAMPLE
    const M = A.length, K = A[0]!.length, N = B[0]!.length
    const sel = this.selected

    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'matmul is bulk dot product. The output at row i, column j is the dot product of A\'s row i with B\'s column j — so every cell of the output is one dot. Click any C cell to see one expanded.',
      ),

      div({ class: 'matmul-row' },
        this.labeledMatrix('A', `${M}×${K}`, A, { rowHl: sel?.row }),
        div({ class: 'mat-op' }, '·'),
        this.labeledMatrix('B', `${K}×${N}`, B, { colHl: sel?.col }),
        div({ class: 'mat-op' }, '='),
        this.labeledMatrix('C', `${M}×${N}`, C, {
          cellHl: sel ?? undefined,
          onClick: (r, c) => this.selectCell(r, c),
        }),
      ),

      sel ? this.expand(A, B, C, sel) : null,

      div({ class: 'intro' },
        'The shape rule:',
        formula('[M × K] · [K × N] = [M × N]'),
        'The inner K dims must agree — K is the length of every dot product. M and N control how many output cells there are.',
      ),

      div({ class: 'intro' },
        'Matmul is the workhorse of neural networks: over 90% of the compute in a typical model. It earns that share by doing two conceptually different jobs, though always as the same operation. The first is the textbook neural-net picture: stacks of neurons connected by wires, every wire a learned weight, all packed into the matrix ',
        mono('W'),
        '. Every output is a weighted sum of the inputs. This is called a ',
        italic('projection'),
        '; libraries call the building block a ',
        mono('Linear'),
        ' layer, and the formula is ',
        mono('matmul(x, W) + b'),
        ', where ',
        mono('x'),
        ' is the input vector (a single-row matrix, in matmul\'s shape rule), ',
        mono('W'),
        ' is the weight matrix, and ',
        mono('b'),
        ' is a learned bias added to each output.',
      ),

      div({ class: 'intro' },
        'The second use of matmul is bulk reading — many dot products at once, where each row of one matrix samples a different feature from another. Attention uses this heavily.',
      ),
    )
  }

  labeledMatrix(name: string, shape: string, data: number[][], opts: {
    rowHl?: number; colHl?: number; cellHl?: { row: number; col: number };
    onClick?: (r: number, c: number) => void;
  }) {
    const cells = data.flatMap((row, i) => row.map((v, j) => div({
      class: [
        'mat-cell',
        numClass(v),
        opts.rowHl === i && 'cell-row-hl',
        opts.colHl === j && 'cell-col-hl',
        opts.cellHl?.row === i && opts.cellHl?.col === j && 'cell-sel',
        opts.onClick && 'cell-clickable',
      ],
      onClick: opts.onClick ? () => opts.onClick!(i, j) : undefined,
    }, String(v))))
    return div({ class: 'mat-col' },
      div({ class: 'mat-label' }, `${name}   [${shape}]`),
      matGrid(data[0]!.length, cells),
    )
  }

  expand(A: number[][], B: number[][], C: number[][], sel: { row: number; col: number }) {
    const i = sel.row, j = sel.col
    const K = A[0]!.length
    const indexTerms = range(K).map(p => `A[${i},${p}]·B[${p},${j}]`).join('  +  ')
    const valueTerms = range(K).map(p => {
      const a = A[i]![p]!, b = B[p]![j]!
      const w = (n: number) => n < 0 ? `(${n})` : String(n)
      return `${w(a)}·${w(b)}`
    }).join('  +  ')
    const lhs = `C[${i},${j}]`
    const indent = { paddingLeft: `${lhs.length + 2}ch` }
    return div({ class: 'expr' },
      div(
        span({ class: 'expr-lhs' }, lhs),
        '  =  ', indexTerms,
      ),
      div({ style: indent }, '=  ', valueTerms),
      div({ style: indent }, '=  ', span({ class: 'expr-result' }, String(C[i]![j]!))),
    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        mono('matmul(a, b)'),
        ' is bulk dot product: every row of ',
        mono('a'),
        ' dotted against every column of ',
        mono('b'),
        '.',
      ),

      specBox(
        `const c = matmul(a, b)   // a: [3, 4], b: [4, 5]  →  c: [3, 5]\n                         // c[i, j] = sum(mul(a[i, :], b[:, j]))`,
        [
          [
            'Shape rule: ',
            '[M, K] · [K, N] = [M, N]', '. The inner K must agree — it is the length of every dot product.',
          ],
        ],
      ),

      specBox(
        `const c = matmul(a, b)   // a: [B, T, D], b: [D, V]  →  c: [B, T, V]\n                         // any leading axes (here, B) carry through unchanged;\n                         // the [T, D] · [D, V] matmul runs across each B slot`,
        [
          [
            'matmul works the same whether the input is a single matrix or a stack of them. The shape rule applies to the trailing two axes; anything earlier is along for the ride.',
          ],
        ],
      ),

      div({ class: 'intro' },
        'When the data you have is shaped wrong for matmul\'s ',
        mono('[M, K] · [K, N]'),
        ' rule, you rearrange axes first with ',
        mono('swapAxes'),
        '. Inside attention you will see ',
        mono('matmul(q, swapAxes(k, -1, -2))'),
        ' — K\'s last two axes get swapped so K\'s rows of features become the columns matmul wants to dot against. Pure shape-rearrangement, no math.',
      ),
    )
  }
}

// ---------- softmax section ----------
class SoftmaxSection extends ModeSection {
  peakLogit: number = 2.0

  setPeak(v: number) {
    this.peakLogit = v
    this.update()
  }

  conceptualView() {
    const logits = [this.peakLogit, 1.0, 0.1, -0.5]
    const max = Math.max(...logits)
    const shifted = logits.map(x => x - max)
    const exped = shifted.map(x => Math.exp(x))
    const sumExp = exped.reduce((a, b) => a + b, 0)
    const probs = exped.map(x => x / sumExp)
    const N = logits.length

    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'Softmax turns a row of arbitrary numbers into a probability distribution: every output lands in ',
        mono('[0, 1]'),
        ', the whole row sums to 1, and the biggest input becomes the biggest output. The raw numbers going in are called ',
        italic('logits'),
        ' — short for "log-odds." They sit on the log-probability scale, which is why softmax starts by exponentiating: ',
        mono('exp'),
        ' inverts the log, turning logits into positive numbers.',
        formula('softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)'),
        'Small differences between logits become large gaps in probability. Cells below are tinted green toward the row max, red toward the row min.',
      ),

      div({ class: 'softmax-controls' },
        sliderRow({
          label: 'first score (leftmost)',
          value: fmt(this.peakLogit, 2),
          min: -2, max: 5, step: 0.1,
          current: this.peakLogit,
          onChange: v => this.setPeak(v),
        }),
      ),

      labeledMatrix({
        colLabels: range(N).map(i => String(i)),
        rowLabels: [
          [span({ class: 'sm-lbl-full' }, 'scores'),        span({ class: 'sm-lbl-short' }, 's')],
          [span({ class: 'sm-lbl-full' }, 'probabilities'), span({ class: 'sm-lbl-short' }, 'p')],
        ],
        rows: [
          magnitudeCells(logits, 2),
          magnitudeCells(probs, 3, ['mat-cell', 'sm-prob-cell']),
        ],
      }),

      div({ class: 'intro' },
        'Differences in logits compound exponentially: a logit 1 unit higher becomes ~2.7× more probable (',
        mono('exp(1) ≈ 2.718'),
        '); 5 units higher, ~148×. softmax is "winner-take-most" by default.',
      ),

      div({ class: 'intro' },
        'In language models, softmax is often applied with a ',
        italic('causal mask'),
        ': future positions in each row are set to −∞ before normalizing, so they become 0 after exp. Why mask at all? A language model generates one token at a time, based on prior ones — future tokens haven\'t been generated yet. But training computes loss for every position in parallel over the whole sequence. If position t could see t+1 during training, it would just copy the answer and the task collapses. Causal masking forces each training position to see only its past, so training and generation see the same information. The same idea applies to any model that generates a sequence one position at a time: audio, code, raster-order images.',
      ),

    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        mono('softmax(x, axis?)'),
        ' applies along one axis (default: last). For numerical stability, tensorgrad subtracts the row max before exp — the answer is identical because the same constant gets exp\'d into every term and cancels in the divide.',
      ),
      specBox(
        `const logits = ...               // [V]\nconst probs  = softmax(logits)   // [V] — all in [0, 1], summing to 1`,
        [
          [
            '',
            'softmax', ' acts on the last axis by default. The output values are non-negative and sum to 1 across that axis.',
          ],
        ],
      ),
      specBox(
        `// Causal variant\nconst attn = softmaxCausal(scores)   // future positions masked to −∞ before softmax`,
        [
          [
            '',
            'softmaxCausal', ' is softmax with the causal mask fused in. See the conceptual view for why.',
          ],
        ],
      ),
    )
  }
}

// ---------- Composition / capstone section ----------
class CompositionSection extends Component {
  view() {
    return div({ class: 'tab-content-inner' },
      // Hook: linked bulb has the animated visualisations; this tab does the
      // primitive-by-primitive math for one head.
      div({ class: 'intro' },
        'Here\'s a ',
        a({ href: 'https://typebulb.com/u/samples/transformer/full', target: '_blank' },
          'small transformer training live'),
        ' with animated visualisations. We want to couple the visual intuitions in that demo with actual transformer code. To do that, we\'ll build an attention head with the tensor primitives we\'ve explained in the previous tabs.',
      ),

      // The motivation: why this op exists. Uses dog + hot from the Embedding
      // tab's vocabulary rather than introducing fresh words (bank, river).
      div({ class: 'intro' },
        'Embedding gives "dog" a single fixed vector. But "hot dog" is a sausage, not a warm pet — meaning depends on context, and a static lookup can\'t carry that. Attention is the operation that lets each token\'s vector update based on the other tokens around it.',
      ),

      // The missing concept: a sequence of token vectors stacked as a matrix.
      // Uses the hot dog example continuously: same two words, real embedding
      // values from the table, now arranged as X.
      div({ class: 'intro' },
        'Take that ',
        mono('hot dog'),
        ' example: 2 tokens. Embedding turns each one into a vector; stacked in order they form a matrix ',
        mono('X'),
        ' of shape ',
        mono('[T, D]'),
        '. ',
        mono('T'),
        ' is the number of tokens (here, 2). ',
        mono('D'),
        ' is the embedding dimension — the length of each row\'s vector (here, 4, matching the Embedding tab). Each row is one ',
        italic('position'),
        ' in the sequence. That\'s the input to attention.',
      ),

      // Diagram: real embedding values for "hot" and "dog" stacked as X.
      this.xDiagram(),

      // Bridge: ground X as the residual stream so the section headings below
      // have a concrete substrate to operate on. The structure (project →
      // score → mix → write) shows up in the headings themselves.
      div({ class: 'intro' },
        'What follows operates on the residual stream ',
        mono('X'),
        ' — which starts as token embeddings (added to position embeddings) and is updated by each block in the transformer stack.',
      ),

      // Project: X into Q, K, V via three learned linear projections.
      h3('Route X into queries, keys, values'),
      div({ class: 'intro' },
        'Each weight matrix is a learned linear ',
        italic('routing'),
        ': it takes each row of ',
        mono('X'),
        ' and re-expresses it as another vector, tuned for what the projection will be used for downstream. Apply this three times with three different weight matrices to get ',
        mono('Q'), ', ',
        mono('K'), ', ',
        mono('V'),
        ' — same shape as ',
        mono('X'),
        ', each row holding one position\'s projected view.',
      ),

      pre({ class: 'pipeline-code-block' },
        `Q = matmul(X, Wq)   // [T, D]\nK = matmul(X, Wk)   // [T, D]\nV = matmul(X, Wv)   // [T, D]`,
      ),

      // Score: pairwise dot products of Q rows with K rows. Matchmaking
      // metaphor + concrete hot/dog example anchors the asymmetric Q-vs-K
      // roles.
      h3('Score every (Q, K) pair'),
      div({ class: 'intro' },
        'Think of this step as pairwise matchmaking: every position pitches what it ',
        italic('offers'),
        ' (its key) and scans the others for what it ',
        italic('wants'),
        ' (its query). For example, in ',
        mono('hot dog'),
        ', position ',
        mono('dog'),
        ' might be looking for a modifier to disambiguate it (its Q), while position ',
        mono('hot'),
        ' offers "I modify the noun next to me" (its K). The (dog, hot) pair scores high — so ',
        mono('dog'),
        ' ends up attending to ',
        mono('hot'),
        '.',
      ),

      div({ class: 'intro' },
        'That\'s exactly what the dot product from the Dot tab does — for any pair (i, j):',
        formula('score(i, j) = q[i] · k[j]'),
        'One matmul produces all ',
        mono('T × T'),
        ' of them at once (',
        mono('swapAxes'),
        ' transposes K so the shapes line up).',
      ),

      pre({ class: 'pipeline-code-block' },
        `scores = matmul(Q, swapAxes(K, -1, -2))   // [T, T]`,
      ),

      this.scoresDiagram(),

      // Softmax: turn raw scores into a per-position weight distribution.
      h3('Softmax each row of scores'),
      div({ class: 'intro' },
        'Softmax over each row of ',
        mono('scores'),
        ' turns it into ',
        mono('attn'),
        ': a per-position distribution where each row sums to 1 (covered in the softmax tab).',
      ),

      div({ class: 'intro' },
        'We need these to act as weights in the next step (Average), and raw scores wouldn\'t work — they could be negative, unbounded, or near-identical. Softmax fixes that: non-negative outputs that sum to 1 per row, giving each position a fixed attention budget to spread across the others. And because softmax is winner-take-most, each position can focus that budget on the few keys that best matched its query.',
      ),

      div({ class: 'intro' },
        'When the data has causal or temporal structure (language, audio, time series, video), future positions are masked to −∞ first to keep them from influencing earlier ones (see the softmax tab).',
      ),

      pre({ class: 'pipeline-code-block' },
        `attn = softmaxCausal(scores)   // [T, T]`,
      ),

      this.attnDiagram(),

      // Weighted average of value vectors using attn weights.
      h3('Average values using attention weights'),
      div({ class: 'intro' },
        'For each position, take a weighted average of all the value vectors, with the weights coming from that position\'s row of ',
        mono('attn'),
        '. One matmul does this for every position at once, producing ',
        mono('context'),
        ' (shape ',
        mono('[T, D]'),
        ').',
      ),

      pre({ class: 'pipeline-code-block' },
        `context = matmul(attn, V)   // [T, D]`,
      ),

      div({ class: 'intro' },
        'Notice we\'re using ',
        mono('V'),
        ' here, not ',
        mono('K'),
        '. Take ',
        mono('hot'),
        ' as the example: ',
        mono('K[hot]'),
        ' was just hot\'s match label ("I modify the noun next to me") that ',
        mono('dog'),
        ' used to find it during scoring. ',
        mono('V[hot]'),
        ' is hot\'s actual content (heat, temperature, often paired with food) — what flows to anyone who attends to it. ',
        mono('K'),
        ' is how you get found; ',
        mono('V'),
        ' is what you give.',
      ),

      div({ class: 'intro' },
        'Since ',
        mono('dog'),
        ' attended mostly to ',
        mono('hot'),
        ' from the scoring step, ',
        mono('context[dog]'),
        ' is mostly ',
        mono('V[hot]'),
        ' — ',
        mono('dog'),
        '\'s vector now carries information from ',
        mono('hot'),
        '. That\'s the payoff of the whole machinery: each position has gathered context from the positions it cared about.',
      ),

      // Write: add context back into X (residual update). Closes the
      // opening hot dog arc — dog's vector now carries hot-info, resolving
      // the pet/sausage disambiguation the tab opened with.
      h3('Write context back into X'),
      div({ class: 'intro' },
        mono('add(X, context)'),
        ' adds each position\'s context onto its original ',
        mono('X'),
        ' row. Every position now carries its original vector plus its context-aware contribution.',
      ),

      pre({ class: 'pipeline-code-block' },
        `X = add(X, context)   // [T, D]`,
      ),

      div({ class: 'intro' },
        mono('add'),
        ' instead of replace: the original ',
        mono('X'),
        ' survives. Context layers on top — what attention found is added alongside, not in place of.',
      ),

      div({ class: 'intro' },
        'Transformers also include a normalization step somewhere in each block — RMSNorm or LayerNorm — that keeps ',
        mono('X'),
        '\'s magnitudes bounded across many stacked layers. See this ',
        a({ href: 'https://tinyurl.com/44ayrzfp', target: '_blank' }, 'transformer architecture diagrammed by nn-dna'),
        ' for where it sits alongside the attention and MLP sub-layers.',
      ),

      div({ class: 'intro' },
        'And the opening puzzle closes. ',
        mono('dog'),
        '\'s vector started as just the embedding for "dog" — ambiguous between the pet and the food. After this attention head, it also carries information from ',
        mono('hot'),
        ', so downstream processing can finally tell this is "hot dog" the sausage, not "dog" the pet. The static lookup we opened with has become context-aware. That\'s one attention head.',
      ),

    )
  }

  // Raw scores matrix (pre-softmax, pre-mask) for the running hot/dog
  // example. Values are illustrative — chosen so that softmaxCausal of this
  // matrix produces the attnDiagram numbers (Q[dog] row 2.0 vs 1.15 → 0.7
  // vs 0.3). Asymmetry of score[i,j] vs score[j,i] is visible by design.
  scoresDiagram() {
    const cell = (val: string) => div({ class: ['mat-cell', 'num-pos'] }, val)
    return labeledMatrix({
      colLabels: [mono('K[hot]'), mono('K[dog]')],
      rowLabels: [mono('Q[hot]'), mono('Q[dog]')],
      rowAnnotations: ['raw scores', 'raw scores'],
      rows: [
        [cell('0.3'), cell('1.2')],
        [cell('2.0'), cell('1.15')],
      ],
    })
  }

  // attn matrix for the running hot/dog example, after softmaxCausal.
  // Hot at pos 0 sees only itself (future masked); dog at pos 1 sees both
  // and weights mostly toward hot — the disambiguation flow the tab sets up.
  attnDiagram() {
    const cell = (val: string, cls: string) =>
      div({ class: ['mat-cell', cls] }, val)
    return labeledMatrix({
      colLabels: [mono('K[hot]'), mono('K[dog]')],
      rowLabels: [mono('Q[hot]'), mono('Q[dog]')],
      rowAnnotations: ['row sums to 1', 'row sums to 1'],
      rows: [
        [cell('1.0', 'num-pos'), cell('—', 'lm-masked')],
        [cell('0.7', 'num-pos'), cell('0.3', 'num-pos')],
      ],
    })
  }

  // X for the attention example: two rows pulled directly from the Embedding
  // table — "hot" and "dog" — making the continuity from the motivating
  // paragraph above to the matrix shape below literal rather than implied.
  xDiagram() {
    const rows = ['hot', 'dog'].map(name => ({
      name,
      vec: NAMED_EMBED_TABLE.find(t => t.name === name)!.vec,
    }))
    return div({ class: 'x-diagram' },
      labeledMatrix({
        colLabels: rows[0]!.vec.map((_, j) => mono(`d${j}`)),
        rowLabels: rows.map(r => mono(r.name)),
        rowAnnotations: rows.map((_, i) => `pos ${i}`),
        rows: rows.map(r => r.vec.map(numCell)),
      }),
      p(mono('X'), ' shape ', mono('[T, D] = [2, 4]')),
    )
  }
}

// ---------- Root ----------
class Root extends Component implements IRoot {
  viewMode: ViewMode = 'conceptual'

  tabs = new TabControl()
  shapesSection      = new ShapesSection()
  embeddingSection   = new EmbeddingSection()
  addSection         = new AddSection()
  dotSection         = new DotSection()
  matmulSection      = new MatmulSection()
  softmaxSection     = new SoftmaxSection()
  compositionSection = new CompositionSection()

  setViewMode(m: ViewMode) {
    if (this.viewMode === m) return
    this.viewMode = m
    this.update()
  }

  viewModeBar() {
    const mk = (key: ViewMode, label: string) => button({
      class: ['view-mode-btn', this.viewMode === key && 'active'],
      onClick: () => this.setViewMode(key),
    }, label)
    return div({ class: 'view-mode-bar' },
      div({ class: 'view-mode-group' },
        mk('conceptual', 'Conceptual'),
        mk('tensorgrad', 'In tensorgrad'),
      ),
    )
  }

  view() {
    return div({ class: 'app' },
      div({ class: 'header' },
        div({ class: 'header-inner' },
          h1('Vectors and Tensors'),
          p({ class: 'subtitle' },
            'The building blocks of neural networks.',
          ),
        ),
      ),
      div({ class: 'main' },
        this.tabs.view({
          tabs: [
            { key: 'add',         label: 'Vectors',     content: this.addSection.view() },
            { key: 'dot',         label: 'Dot',         content: this.dotSection.view() },
            { key: 'embedding',   label: 'Embedding',   content: this.embeddingSection.view() },
            { key: 'shapes',      label: 'Tensors',     content: this.shapesSection.view() },
            { key: 'matmul',      label: 'matmul',      content: this.matmulSection.view() },
            { key: 'softmax',     label: 'softmax',     content: this.softmaxSection.view() },
            { key: 'composition', label: 'Attention',    content: this.compositionSection.view(), hideSubBar: true },
          ],
          subBar: this.viewModeBar(),
        }),
      ),
    )
  }
}

new App({ root: new Root(), id: 'app' })
```

**styles.css**

```css
:root {
  --bg: rgb(255, 255, 255);
  --surface: rgb(248, 248, 248);
  --text: rgb(20, 20, 20);
  --text-muted: rgb(95, 95, 95);
  --border: rgb(220, 220, 220);
  --cell-border: rgb(220, 220, 220);
  --cell-bg: rgb(255, 255, 255);
  --accent: rgb(99, 102, 241);
  --positive: rgb(0, 120, 0);
  --negative: rgb(170, 30, 30);
  --zero: rgb(140, 140, 140);
  --a-color: rgb(40, 100, 200);
  --b-color: rgb(220, 110, 40);
  --c-color: rgb(0, 120, 0);
  --axis: rgb(200, 200, 200);
  --code-bg: rgb(245, 245, 245);
  --code-border: rgb(220, 220, 220);
  --cell-w: 52px;
  --cell-h: 44px;
  --content-max-width: 780px;
  --code-font-size: 14px;
  --row-hl: rgba(50, 130, 255, 0.12);
  --selected-bg: rgba(0, 170, 80, 0.20);
  --selected-border: rgb(0, 150, 60);
}

html[data-theme="dark"] {
  --bg: rgb(22, 22, 24);
  --surface: rgb(34, 34, 38);
  --text: rgb(235, 235, 235);
  --text-muted: rgb(160, 160, 165);
  --border: rgb(50, 50, 55);
  --cell-border: rgb(55, 55, 60);
  --cell-bg: rgb(28, 28, 32);
  --accent: rgb(129, 140, 248);
  --positive: rgb(110, 220, 110);
  --negative: rgb(255, 115, 115);
  --zero: rgb(130, 130, 135);
  --a-color: rgb(110, 170, 255);
  --b-color: rgb(255, 175, 100);
  --c-color: rgb(110, 220, 110);
  --axis: rgb(70, 70, 75);
  --code-bg: rgb(15, 15, 18);
  --code-border: rgb(50, 50, 55);
  --row-hl: rgba(90, 165, 255, 0.18);
  --selected-bg: rgba(40, 210, 110, 0.25);
  --selected-border: rgb(80, 220, 130);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.app { min-height: 100vh; }

.header {
  padding: 12px 0 12px;
}

.header-inner,
.main {
  max-width: 960px;
  margin: 0 auto;
  padding-left: 24px;
  padding-right: 24px;
}

.header h1 {
  margin: 0;
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

/* Muted body text — same role across sections; the per-class rules below
   only add positional extras (margins, max-widths, italic). */
.subtitle,
.caption,
.spec-caption,
.embed-prose {
  font-size: 16px;
  line-height: 1.6;
  color: var(--text-muted);
}

.subtitle {
  margin: 4px 0 0;
  max-width: 720px;
}

.main {
  padding-top: 0;
  padding-bottom: 56px;
}

/* Main tab bar */
.tab-control .tab-bar {
  display: flex;
  gap: 0 24px;
  margin: 0 0 14px;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}

.tab-control .tab-btn {
  padding: 8px 2px 7px;
  margin-bottom: -1px;
  border: none;
  border-bottom: 2px solid transparent;
  background: transparent;
  font: inherit;
  font-size: 16px;
  cursor: pointer;
  color: var(--text-muted);
  border-radius: 0;
  transition: color 150ms, border-color 150ms;
}

.tab-control .tab-btn:hover { color: var(--text); }

.tab-control .tab-btn.active {
  color: var(--text);
  border-bottom-color: var(--accent);
  font-weight: 600;
}

/* Sub-bar: view mode toggle */
.view-mode-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 22px;
}

.view-mode-group {
  display: inline-flex;
}

.view-mode-btn {
  padding: 5px 14px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text-muted);
  font: inherit;
  font-size: 14px;
  cursor: pointer;
  position: relative;
  border-radius: 0;
  transition: background 120ms, color 120ms, border-color 120ms;
}

.view-mode-btn:first-of-type {
  border-top-left-radius: 6px;
  border-bottom-left-radius: 6px;
}

.view-mode-btn:last-of-type {
  border-top-right-radius: 6px;
  border-bottom-right-radius: 6px;
}

.view-mode-btn + .view-mode-btn { margin-left: -1px; }

.view-mode-btn:hover { color: var(--text); }

.view-mode-btn.active {
  background: color-mix(in srgb, var(--accent) 14%, transparent);
  color: var(--accent);
  border-color: var(--accent);
  z-index: 1;
}

/* Per-tab content shell — structural width cap. Every direct/indirect child
   inherits this bound, so individual rules don't need their own max-widths. */
.tab-content-inner {
  max-width: var(--content-max-width);
}

.tab-content-inner > * + * { margin-top: 22px; }

.intro {
  color: var(--text);
  font-size: 16px;
  line-height: 1.65;
}

/* Section headings inside tab content (e.g. the project/score/mix/write
   spine in the Attention tab). Generous top margin marks the section break;
   tighter gap below keeps the heading attached to its paragraph. */
.tab-content-inner h3 {
  font-size: 21px;
  font-weight: 600;
  margin: 36px 0 0;
  color: var(--text);
}
.tab-content-inner h3 + * { margin-top: 10px; }

.italic { font-style: italic; }

/* Display formulas inside .intro paragraphs */
.formula-display {
  text-align: center;
  margin: 14px 0;
}

/* Plot + side panel row */
.viz-row {
  display: flex;
  gap: 28px;
  flex-wrap: wrap;
  align-items: flex-start;
}

.plot {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  display: block;
  max-width: 100%;
  height: auto;
}

.side-panel {
  flex: 1 1 240px;
  min-width: 240px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.detail {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-family: monospace;
  font-size: 17px;
}

.detail-row {
  display: flex;
  gap: 6px;
  align-items: baseline;
  flex-wrap: wrap;
}

.detail-lbl { min-width: 13ch; }
.detail-val { color: var(--text); }

.control-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 14px;
  color: var(--text-muted);
}

.control-value {
  font-family: monospace;
  color: var(--text);
  font-weight: 600;
}

.slider {
  width: 100%;
  accent-color: var(--accent);
}

.caption { font-style: italic; }

/* Spec excerpt */
.spec-anchor {
  background: var(--code-bg);
  border: 1px solid var(--code-border);
  border-radius: 8px;
  padding: 16px 18px;
}

.spec-anchor + .spec-anchor { margin-top: 14px; }

.spec-code {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 14px;
  font-family: monospace;
  font-size: var(--code-font-size);
  line-height: 1.6;
  margin: 0 0 12px;
  overflow-x: auto;
  white-space: pre;
  color: var(--text);
}

.spec-caption { margin: 0; }

.spec-caption + .spec-caption { margin-top: 10px; }

.mono {
  font-family: monospace;
  color: var(--accent);
}

/* Matrix cells (matmul + sum) */
.mat-col {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}

.mat-label,
.mm-label {
  font-family: monospace;
  font-size: 14px;
  color: var(--text-muted);
}

.mat-grid {
  display: grid;
  gap: 1px;
  background: var(--cell-border);
  border: 1px solid var(--cell-border);
  border-radius: 4px;
  overflow: hidden;
}

.mat-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  width: var(--cell-w);
  height: var(--cell-h);
  background: var(--cell-bg);
  font-family: monospace;
  font-size: 16px;
  font-weight: 500;
  user-select: none;
  transition: background 120ms, box-shadow 120ms;
}

.num-pos  { color: var(--positive); }
.num-neg  { color: var(--negative); }
.num-zero { color: var(--zero); }

.cell-clickable { cursor: pointer; }
.cell-clickable:hover:not(.cell-sel) { background: var(--row-hl); }

.cell-row-hl:not(.cell-sel),
.cell-col-hl:not(.cell-sel) { background: var(--row-hl); }

.cell-sel {
  background: var(--selected-bg);
  box-shadow: inset 0 0 0 2px var(--selected-border);
}

.mat-op {
  font-size: 22px;
  font-family: monospace;
  color: var(--text-muted);
  padding: 0 6px;
  align-self: center;
  margin-top: 16px;
}

/* matmul layout */
.matmul-row {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  flex-wrap: wrap;
}

.expr {
  font-family: monospace;
  font-size: var(--code-font-size);
  line-height: 1.85;
  color: var(--text);
  overflow-x: auto;
}

.expr-lhs    { font-weight: 600; }
.expr-result { font-weight: 700; font-size: 17px; }

/* Tensors tab */
.shape-ladder {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  flex-wrap: wrap;
  padding: 12px 0;
}

.shape-rung {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.shape-rung-label {
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.shape-rung-notation {
  font-family: monospace;
  font-size: 14px;
  color: var(--text-muted);
}

/* 3-layer matrix stack — used both for the rank-3 tensor (opaque slices)
   and the matmul-stack viz (faded ghost layers behind the real matmul). */
.shape-stack,
.mm-stack { position: relative; }

.shape-stack { width: calc(var(--cell-w) * 3 + 24px); height: calc(var(--cell-h) * 2 + 24px); }

.shape-stack > .mat-grid,
.mm-stack > .mat-grid { position: absolute; }

.shape-stack > .mat-grid:nth-child(1),
.mm-stack    > .mat-grid:nth-child(1) { top: 0;    left: 24px; }
.shape-stack > .mat-grid:nth-child(2),
.mm-stack    > .mat-grid:nth-child(2) { top: 12px; left: 12px; }
.shape-stack > .mat-grid:nth-child(3),
.mm-stack    > .mat-grid:nth-child(3) { top: 24px; left: 0;    }

.mm-stack > .mat-grid:nth-child(1) { z-index: 3; }
.mm-stack > .mat-grid:nth-child(2) { z-index: 2; opacity: 0.55; }
.mm-stack > .mat-grid:nth-child(3) { z-index: 1; opacity: 0.35; }

/* Matmul-stack viz layout. */
.mm-viz {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.mm-row {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  flex-wrap: wrap;
  padding: 8px 0 4px;
}

.mm-item {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
}

.mm-op-glyph {
  font-family: monospace;
  font-size: 28px;
  color: var(--text-muted);
  align-self: center;
  padding: 0 4px;
}

/* Surface-tinted box. Shared with .expr below. */
.embed-block,
.expr {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
}

.embed-prose {
  margin-bottom: 14px;
  max-width: 640px;
}

/* Inline row of cells (rank-1 vector in the shape ladder; token row in the
   embedding explorer). */
.cell-row {
  display: inline-flex;
  gap: 1px;
  background: var(--cell-border);
  border: 1px solid var(--cell-border);
  border-radius: 4px;
  overflow: hidden;
  width: fit-content;
}

/* Attention tab: X-as-matrix diagram. Wraps a labeledMatrix plus the shape
   caption (<p>) and centers the whole stack. */
.x-diagram {
  margin: 12px 0 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.x-diagram > p { margin: 4px 0 0; font-size: 14px; color: var(--text-muted); }

/* Labeled-matrix primitive: cells in the middle (rendered as a normal
   .mat-grid so borders match the embedding/tensors tabs), optional col-labels
   on top, optional row-labels on the left, optional row-annotations on the
   right. Outer 3-col grid auto-sizes column 2 to the .mat-grid width and
   sizes the inner col-labels / row-labels grids to match the cell tracks. */
.labeled-matrix {
  display: flex;
  justify-content: center;
  margin: 16px 0 4px;
}
.lm-grid {
  display: inline-grid;
  grid-template-columns: auto auto auto;
  align-items: center;
}
.lm-cells-wrap { display: flex; }
.lm-col-labels {
  display: grid;
  gap: 1px;
  /* match the 1px outer border of the inner .mat-grid so col labels align
     with cells in column 2 */
  border: 1px solid transparent;
  padding-bottom: 6px;
  align-self: end;
}
.lm-row-labels,
.lm-row-annotations {
  display: grid;
  gap: 1px;
  border: 1px solid transparent;
}
.lm-col-label,
.lm-row-label,
.lm-row-annotation {
  display: flex;
  align-items: center;
  font-size: 14px;
  color: var(--text-muted);
}
.lm-col-label { justify-content: center; }
.lm-row-label { justify-content: flex-end; padding-right: 10px; white-space: nowrap; }
.lm-row-annotation {
  justify-content: flex-start;
  padding-left: 10px;
  font-size: 12px;
  font-style: italic;
}
/* Labels stay in the muted gray of the surrounding chrome — the .mono helper
   defaults to var(--accent) which would tint them blue. */
.lm-col-label .mono,
.lm-row-label .mono,
.lm-row-annotation .mono { color: inherit; }

.lm-masked {
  color: var(--text-muted);
  background: var(--surface);
}

/* Token × vector × similarity explorer */
.embed-grid {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.embed-grid-head,
.embed-grid-row {
  display: grid;
  grid-template-columns: 8ch auto 1fr;
  gap: 18px;
  align-items: center;
  padding: 5px 10px 5px 9px;
  border-left: 3px solid transparent;
  border-radius: 4px;
}

.embed-grid-head {
  font-size: 13px;
  color: var(--text-muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding-bottom: 4px;
  cursor: default;
}

.embed-sim-head {
  display: flex;
  align-items: baseline;
  gap: 6px;
}

.embed-h-name {
  font-family: monospace;
  text-transform: none;
  font-size: 16px;
  color: var(--text);
  font-weight: 400;
  letter-spacing: 0;
}

.embed-grid-row {
  cursor: pointer;
  transition: background 120ms, border-color 120ms;
}

.embed-grid-row:hover:not(.is-sel) { background: var(--row-hl); }

.embed-grid-row.is-sel {
  background: var(--selected-bg);
  border-left-color: var(--selected-border);
}

.embed-token-name {
  font-family: monospace;
  font-size: 16px;
  color: var(--text);
}

.embed-sim {
  display: flex;
  align-items: center;
  gap: 12px;
}

.embed-sim-num {
  font-family: monospace;
  font-size: 16px;
  font-weight: 600;
  min-width: 3ch;
  text-align: right;
}

.embed-sim-track {
  position: relative;
  flex: 1 1 auto;
  max-width: 160px;
  height: 10px;
}

.embed-sim-axis {
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 0;
  width: 1px;
  background: var(--axis);
}

.embed-sim-bar {
  position: absolute;
  top: 1px;
  bottom: 1px;
  border-radius: 2px;
}

/* Mobile: shrink the explorer to fit narrow viewports */
@media (max-width: 600px) {
  .embed-block { padding: 12px 14px; }
  .embed-grid-head,
  .embed-grid-row {
    gap: 8px;
    padding: 4px 6px;
  }
  .embed-token-name { font-size: 14px; }
  .embed-h-name { font-size: 14px; }
  .embed-grid .mat-cell {
    width: 28px;
    height: 32px;
    font-size: 12px;
  }
  .embed-sim-num { font-size: 14px; }
  .embed-sim-track { display: none; }
}

/* Code blocks in the Attention tab (one per stage of the attention spine). */
.pipeline-code-block {
  font-family: monospace;
  font-size: var(--code-font-size);
  line-height: 1.5;
  color: var(--text);
  background: var(--code-bg);
  border: 1px solid var(--code-border);
  border-radius: 4px;
  padding: 8px 12px;
  margin: 0;
  overflow-x: auto;
  white-space: pre;
}

/* Code blocks sit closer to surrounding prose than the default
   intro-to-intro gap — symmetric 14px above and below, replacing the
   .tab-content-inner > * + * 22px rule for this case. */
.tab-content-inner > .pipeline-code-block,
.tab-content-inner > .pipeline-code-block + * { margin-top: 14px; }

/* softmax tab */
.softmax-controls {
  max-width: 360px;
  margin-left: auto;
  margin-right: auto;
}

.sm-prob-cell {
  font-weight: 700;
}

.sm-lbl-short { display: none; }

@media (max-width: 480px) {
  .sm-lbl-full { display: none; }
  .sm-lbl-short { display: inline; }
}
```

**index.html**

```html
<div id="app"></div>
```

**config.json**

```json
{
  "dependencies": {
    "domeleon": "^0.6.0"
  },
  "description": "The building blocks of neural networks explained interactively. Vectors, tensors, matmul, softmax, embedding, and more."
}
```
