---
format: typebulb/v1
name: Tensors
---

**code.tsx**

```tsx
import {
  App, Component, div, h1, p, span, pre, button, input,
  svg, line, circle, polygon, text, g,
  type VElement, type HValues,
} from 'domeleon'

// ---------- Shared helpers ----------
const range = (n: number) => Array.from({ length: n }, (_, i) => i)

const fmt = (x: number, d = 2): string => {
  if (Number.isNaN(x)) return 'NaN'
  if (Math.abs(x) < 1e-9) return '0'
  return x.toFixed(d)
}

// ---------- Number-cell coloring + small row viz (shared across Shapes / Embedding) ----------
const numClass = (v: number) => v === 0 ? 'num-zero' : v > 0 ? 'num-pos' : 'num-neg'

function rowCells(values: number[]) {
  return div({ class: 'embed-cells' },
    values.map(v => div({ class: ['mat-cell', numClass(v)] }, String(v))),
  )
}

const SAMPLE_EMBED_TABLE: number[][] = [
  [ 1,  0, -1,  2],
  [ 0,  2,  1, -1],
  [-1,  1,  2,  0],
]

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

// One row in the softmax stepwise grid: label cell + a row of value cells.
function stepRow(label: HValues, values: number[], digits: number, cellClass: string | string[] = 'mat-cell') {
  return [
    div({ class: 'sm-step-label' }, label),
    values.map(v => div({ class: cellClass }, fmt(v, digits))),
  ]
}

// One "label = value" row in a side panel, with optional colors / weight / aside.
function detailRow(opts: {
  label: HValues
  value: HValues
  lblColor?: string
  valColor?: string
  valWeight?: string
  aside?: string
}) {
  return div({ class: 'detail-row' },
    span({
      class: 'detail-lbl',
      style: opts.lblColor ? { color: opts.lblColor } : undefined,
    }, opts.label),
    span('='),
    span({
      class: 'detail-val',
      style: (opts.valColor || opts.valWeight)
        ? { color: opts.valColor, fontWeight: opts.valWeight }
        : undefined,
    }, opts.value),
    opts.aside ? span({ class: 'sm-aside' }, opts.aside) : null,
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
  width?: number
  label?: string
  labelOffset?: { x: number; y: number }
  dashed?: boolean
  headSize?: number
}

function arrow(from: [number, number], to: [number, number], opts: ArrowOpts) {
  const [x1, y1] = plotToScreen(from[0], from[1])
  const [x2, y2] = plotToScreen(to[0], to[1])
  const dx = x2 - x1, dy = y2 - y1
  const len = Math.sqrt(dx * dx + dy * dy)
  const ux = len > 0 ? dx / len : 0
  const uy = len > 0 ? dy / len : 0
  const headSize = opts.headSize ?? 10
  const baseX = x2 - ux * headSize
  const baseY = y2 - uy * headSize
  const perpX = -uy * (headSize * 0.45)
  const perpY = ux * (headSize * 0.45)
  return g(
    line({
      x1, y1, x2: baseX, y2: baseY,
      stroke: opts.color,
      strokeWidth: opts.width ?? 2.5,
      strokeDashArray: opts.dashed ? '6 4' : undefined,
    }),
    polygon({
      points: `${x2},${y2} ${baseX + perpX},${baseY + perpY} ${baseX - perpX},${baseY - perpY}`,
      fill: opts.color,
    }),
    opts.label ? text({
      x: x2 + (opts.labelOffset?.x ?? 8),
      y: y2 + (opts.labelOffset?.y ?? -8),
      fill: opts.color,
      fontSize: 14,
      fontFamily: 'monospace',
      fontWeight: 600,
    }, opts.label) : null,
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
        parts.map((s, i) => i % 2 === 1 ? span({ class: 'mono' }, s) : s),
      ),
    ),
  )
}

// ---------- TabControl ----------
type TabDef = { key: string; label: string; content: HValues }

class TabControl extends Component {
  selected: string | null = null

  view(props: { tabs: TabDef[]; subBar?: HValues }) {
    const { tabs, subBar } = props
    const sel = this.selected ?? tabs[0]!.key
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
      subBar,
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

// ---------- Shapes section ----------
class ShapesSection extends ModeSection {
  conceptualView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'Neural-network code in PyTorch, JAX, or tensorgrad does not work on individual numbers or vectors; it works on ',
        span({ class: 'italic' }, 'tensors'),
        ': N-dimensional arrays of numbers, with a fixed shape. A scalar is rank 0, a vector is rank 1, a matrix is rank 2, and from rank 3 upward you stack matrices, then stacks of matrices, and so on. Shapes are written as bracketed dim lists: ',
        span({ class: 'mono' }, '[3]'),
        ', ',
        span({ class: 'mono' }, '[3, 4]'),
        ', ',
        span({ class: 'mono' }, '[B, T, D]'),
        '.',
      ),

      div({ class: 'shape-ladder' },
        this.shapeRung('scalar', 'rank 0',
          div({ class: 'mat-cell num-pos' }, '5'),
        ),
        div({ class: 'shape-arrow' }, '→'),
        this.shapeRung('vector', 'rank 1, shape [4]',
          rowCells([1, 4, -2, 3]),
        ),
        div({ class: 'shape-arrow' }, '→'),
        this.shapeRung('matrix', 'rank 2, shape [3, 4]',
          div({
            class: 'mat-grid',
            style: { gridTemplateColumns: 'repeat(4, var(--cell-w))' },
          },
            [[1, 2, -1, 0], [0, 1, 2, -1], [-1, 0, 1, 2]].flatMap(r =>
              r.map(v => div({ class: ['mat-cell', numClass(v)] }, String(v))),
            ),
          ),
        ),
      ),

      div({ class: 'intro' },
        'A typical transformer tensor is shaped ',
        span({ class: 'mono' }, '[B, T, D]'),
        ': ',
        span({ class: 'mono' }, 'B'),
        ' = batch size (independent examples processed together), ',
        span({ class: 'mono' }, 'T'),
        ' = sequence length (positions in the prompt), ',
        span({ class: 'mono' }, 'D'),
        ' = model dim (the vector dim every position carries). Think of it as a ',
        span({ class: 'mono' }, '[B, T]'),
        ' grid of cells, each holding one ',
        span({ class: 'mono' }, 'D'),
        '-dim vector. Most ops act on the last axis (the vector) and treat the leading axes as batching — the same per-vector computation happens in parallel across every (b, t) cell.',
      ),

      div({ class: 'intro' },
        'When two tensors of different ranks meet — for example an ',
        span({ class: 'mono' }, '[B, T, D]'),
        ' tensor added to a ',
        span({ class: 'mono' }, '[T, D]'),
        ' tensor — the shorter shape lines up against the end of the longer one and the missing leading axes get repeated implicitly. ',
        span({ class: 'mono' }, '[T, D] + [B, T, D]'),
        ' works; the ',
        span({ class: 'mono' }, '[T, D]'),
        ' tensor is "broadcast" across the batch dim. This is how the same position-embedding vector applies to every example in a batch without copying.',
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
        `const W = param([3, 4])   // a learnable 2D tensor: 3 rows of 4 numbers`,
        [
          [
            'param([...])', ' creates a learnable tensor of the given shape. Training fills in the values via gradient descent. The shape is fixed at compile time.',
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
        'Embedding is a 2D learnable table of shape ',
        span({ class: 'mono' }, '[VOCAB, D]'),
        ' that holds one ',
        span({ class: 'mono' }, 'D'),
        '-dim row per token (a token is an integer ID, one per position in the input sequence). ',
        span({ class: 'mono' }, 'embedding(table, idx)'),
        ' just fetches the row at position ',
        span({ class: 'mono' }, 'idx'),
        '. The table starts random and trains to place related tokens in similar directions.',
      ),

      this.embeddingLookup(),

      div({ class: 'intro' },
        'There is more than enough space in 64D. This transformer\'s vocab has only 12 tokens (digits 0–9 plus "+" and "="); the embedding table has D=64 dimensions per row. That is far more dimensions than tokens, and the surplus is the whole point. In 2D, only two strictly orthogonal directions exist (the axes); in 64D, many vectors can point in near-orthogonal directions and not collide. Real transformers go much further: GPT-class models have VOCAB ≈ 50,000 and D ≈ 4096. Even then, there is room to fit every token in a distinct direction, plus everything else the model picks up through its layers.',
      ),

      div({ class: 'intro' },
        'This is the property that retroactively rescues ⊕. Adding two vectors in 2D looked lossy because 2D is cramped; in 64D it is not lossy at all. A and B keep their identity inside A + B, recoverable by anyone reading along their respective directions. Every layer of the transformer ⊕\'s into a shared 64D vector, and the contributions accumulate without collapsing into mush.',
      ),
    )
  }

  embeddingLookup() {
    const TABLE = SAMPLE_EMBED_TABLE
    const D = TABLE[0]!.length
    const VOCAB = TABLE.length
    const sel = this.selectedTokenIdx
    const out = TABLE[sel]!

    return div({ class: 'embed-block' },
      div({ class: 'embed-title' },
        span({ class: 'mono' }, 'embedding'),
        ' — lookup by integer index',
      ),
      div({ class: 'embed-prose' },
        'Click any row of the table to fetch its vector. The output is exactly the row you picked — nothing more, just an indexed read.',
      ),
      div({ class: 'embed-row-host' },
        div(
          div({ class: 'mat-label' }, `table  [VOCAB=${VOCAB} × D=${D}]`),
          div({ class: 'embed-table' },
            TABLE.map((row, i) => div({
              class: ['embed-row', i === sel && 'embed-row-sel'],
              onClick: () => this.selectToken(i),
            },
              div({ class: 'embed-idx' }, String(i)),
              rowCells(row),
            )),
          ),
        ),
        div({ class: 'embed-out-col' },
          div({ class: 'embed-out-call' },
            span({ class: 'mono' }, `embedding(table, ${sel})`),
          ),
          div({ class: 'embed-out-arrow' }, '↓'),
          div({ class: 'mat-label' }, `out  [D=${D}]`),
          rowCells(out),
        ),
      ),
    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        span({ class: 'mono' }, 'embedding(table, indices)'),
        ' is an indexed lookup: every integer in ',
        span({ class: 'mono' }, 'indices'),
        ' is replaced by the corresponding row from ',
        span({ class: 'mono' }, 'table'),
        '. The shape gets promoted from integers to D-dim vectors in one op.',
      ),

      specBox(
        `const table = param([12, 64])           // 12 learnable rows, each a 64-dim vector\nconst ids   = [3, 7, 0]                 // pick three IDs; shape [3]\nconst out   = embedding(table, ids)     // shape [3, 64] — the three fetched rows`,
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
        `// indices can have any leading shape:\nconst ids = ...                           // shape [B, T] of integer token IDs\nconst out = embedding(table, ids)         // shape [B, T, 64]`,
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
class AddSection extends ModeSection {
  angle: number = 60   // degrees between A and B

  setAngle(a: number) {
    this.angle = Math.max(0, Math.min(180, a))
    this.update()
  }

  conceptualView() {
    // A_LEN + B_LEN must fit inside the plot's ~5-unit half-extent — even at
    // θ=0° where B is colinear with A and the sum lands at A_LEN + B_LEN on
    // the x-axis. 2.5 + 2.0 = 4.5, comfortably inside.
    const { A, B, C } = vectorsFromAngle(this.angle, 2.5, 2.0)
    const fmtV = (v: Vec2) => `[${fmt(v[0])}, ${fmt(v[1])}]`

    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'A vector is a list of numbers — equivalently, an arrow from the origin to a point in space. Tensorgrad code (and PyTorch, JAX) ultimately works on these and stacks of them. The first thing you do with two vectors is add them. We write vector addition as ⊕: ',
        span({ class: 'mono' }, 'A ⊕ B = [a₁+b₁, a₂+b₂]'),
        ' — component-wise; tip-to-tail in the picture below.',
      ),

      div({ class: 'viz-row' },
        svg({ class: 'plot', viewBox: `0 0 ${PLOT} ${PLOT}`, width: PLOT, height: PLOT },
          axes(),
          arrow([0, 0], A, { color: 'var(--a-color)', label: 'A' }),
          arrow(A, C, { color: 'var(--b-color)', label: 'B', labelOffset: { x: 8, y: -6 } }),
          arrow([0, 0], C, { color: 'var(--c-color)', dashed: true, label: 'A + B', labelOffset: { x: 8, y: 16 } }),
        ),
        div({ class: 'side-panel' },
          sliderRow({
            label: 'angle between A and B',
            value: `${this.angle}°`,
            min: 0, max: 180, step: 1,
            current: this.angle,
            onChange: v => this.setAngle(v),
          }),
          div({ class: 'detail' },
            detailRow({ label: 'A',     value: fmtV(A), lblColor: 'var(--a-color)' }),
            detailRow({ label: 'B',     value: fmtV(B), lblColor: 'var(--b-color)' }),
            detailRow({ label: 'A + B', value: fmtV(C), lblColor: 'var(--c-color)' }),
          ),
        ),
      ),

    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        span({ class: 'mono' }, 'add(a, b)'),
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
class DotSection extends ModeSection {
  angle: number = 45   // degrees

  setAngle(a: number) {
    this.angle = Math.max(0, Math.min(180, a))
    this.update()
  }

  conceptualView() {
    const { A, B } = vectorsFromAngle(this.angle, 3, 2.5)
    const dot = A[0] * B[0] + A[1] * B[1]
    const Amag = Math.sqrt(A[0] * A[0] + A[1] * A[1])
    const Bmag = Math.sqrt(B[0] * B[0] + B[1] * B[1])
    const cosθ = Math.cos(this.angle * Math.PI / 180)
    const projX = B[0]

    const dotColor = dot > 0.01 ? 'var(--positive)' : dot < -0.01 ? 'var(--negative)' : 'var(--zero)'

    const [ox, oy] = plotToScreen(0, 0)
    const [bx, by] = plotToScreen(B[0], B[1])
    const [px, py] = plotToScreen(projX, 0)

    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'Two vectors can be added; they can also be compared. The dot product turns two vectors into a single number measuring how aligned they are. Component-wise: ',
        span({ class: 'mono' }, 'A · B = a₁b₁ + a₂b₂ + ... + aₙbₙ'),
        '. Geometrically: ',
        span({ class: 'mono' }, 'A · B = |A| |B| cos(θ)'),
        ', where ',
        span({ class: 'mono' }, '|A|'),
        ' is the length of A (',
        span({ class: 'mono' }, '√(a₁² + a₂² + ... + aₙ²)'),
        ' — Pythagoras generalized to n dimensions; in 1D it is ordinary absolute value, which is why the bar notation extends) and ',
        span({ class: 'mono' }, 'θ'),
        ' is the angle between A and B.',
      ),

      div({ class: 'viz-row' },
        svg({ class: 'plot', viewBox: `0 0 ${PLOT} ${PLOT}`, width: PLOT, height: PLOT },
          axes(),
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
        ),
        div({ class: 'side-panel' },
          sliderRow({
            label: 'angle θ',
            value: `${this.angle}°`,
            min: 0, max: 180, step: 1,
            current: this.angle,
            onChange: v => this.setAngle(v),
          }),
          div({ class: 'detail' },
            detailRow({ label: 'A · B',         value: fmt(dot), valColor: dotColor, valWeight: '700' }),
            detailRow({ label: '|A|',           value: fmt(Amag) }),
            detailRow({ label: '|B|',           value: fmt(Bmag) }),
            detailRow({ label: 'cos θ',         value: fmt(cosθ, 3) }),
            detailRow({ label: '|A||B| cos θ',  value: fmt(Amag * Bmag * cosθ) }),
          ),
          div({ class: 'caption' }, this.stateDescription()),
          div({ class: 'caption' },
            'The faint indigo segment along A is the projection of B onto A\'s direction. Its signed length, scaled by |A|, equals A · B.',
          ),
        ),
      ),

      div({ class: 'intro' },
        'And the self-case: ',
        span({ class: 'mono' }, 'A · A = |A|²'),
        '. A vector dotted with itself recovers its squared magnitude.',
      ),
    )
  }

  stateDescription() {
    const θ = this.angle
    if (θ <= 3) {
      return [
        'A and B point the same way. ',
        span({ class: 'mono' }, 'A · B = |A||B|'),
        ' — the maximum value the dot product can reach.',
      ]
    }
    if (θ >= 177) {
      return [
        'A and B point opposite directions. ',
        span({ class: 'mono' }, 'A · B = −|A||B|'),
        ' — the minimum value.',
      ]
    }
    if (θ >= 87 && θ <= 93) {
      return [
        'A and B are perpendicular. ',
        span({ class: 'mono' }, 'A · B = 0'),
        ' — they share no information along each other\'s direction. This is the orthogonality property that lets ⊕ stack contributions without destroying them.',
      ]
    }
    if (θ < 90) {
      return [
        'A and B partially overlap. ',
        span({ class: 'mono' }, 'A · B'),
        ' is positive — between 0 and |A||B|.',
      ]
    }
    return [
      'A and B partially oppose. ',
      span({ class: 'mono' }, 'A · B'),
      ' is negative — between −|A||B| and 0.',
    ]
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'tensorgrad does not expose a standalone ',
        span({ class: 'mono' }, 'dot'),
        ' op. Element-wise multiply, then sum:',
      ),
      specBox(
        `const d = sum(mul(a, b))   // a: [N], b: [N]  →  d: scalar\n                           // d = a₁·b₁ + a₂·b₂ + ... + aₙ·bₙ`,
        [
          [
            'mul', ' multiplies element-wise (',
            'mul(a, b)[i] = a[i] * b[i]', '); ',
            'sum', ' collapses the result to a single number. Together: the dot product.',
          ],
        ],
      ),
      div({ class: 'intro' },
        'In practice ML almost always wants many dot products at once, and those naturally live inside ',
        span({ class: 'mono' }, 'matmul'),
        ' (next tab). That\'s why tensorgrad doesn\'t bother with a standalone one; just use ',
        span({ class: 'mono' }, 'sum(mul(a, b))'),
        '. Some frameworks like PyTorch (',
        span({ class: 'mono' }, 'torch.dot'),
        ') and JAX/NumPy (',
        span({ class: 'mono' }, 'jnp.dot'),
        ' / ',
        span({ class: 'mono' }, 'np.dot'),
        ') expose a standalone one, to cross the Ts and · the Is.',
      ),
    )
  }
}

// ---------- matmul section ----------
class MatmulSection extends ModeSection {
  selected: { row: number; col: number } | null = { row: 0, col: 0 }

  // Worked example: 2x3 · 3x2 = 2x2
  static readonly A: number[][] = [
    [1,  2, -1],
    [0,  1,  1],
  ]
  static readonly B: number[][] = [
    [ 1,  0],
    [-1,  2],
    [ 1,  1],
  ]
  static readonly C: number[][] = MatmulSection.A.map(rowA =>
    MatmulSection.B[0]!.map((_, j) =>
      rowA.reduce((s, a, k) => s + a * MatmulSection.B[k]![j]!, 0),
    ),
  )

  selectCell(row: number, col: number) {
    const s = this.selected
    this.selected = (s && s.row === row && s.col === col) ? null : { row, col }
    this.update()
  }

  conceptualView() {
    const { A, B, C } = MatmulSection
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
        'The shape rule is ',
        span({ class: 'mono' }, '[M × K] · [K × N] = [M × N]'),
        '. The inner K dims must agree — K is the length of every dot product. M and N control how many output cells there are.',
      ),

      div({ class: 'intro' },
        'Inside a transformer, matmul does two conceptually distinct things but it is always the same op. As a ',
        span({ class: 'italic' }, 'projection'),
        ', it takes a vector ',
        span({ class: 'mono' }, 'x'),
        ' and passes it through a learned matrix to produce a new vector in a different direction — a ',
        span({ class: 'mono' }, 'Linear'),
        ' layer is just ',
        span({ class: 'mono' }, 'matmul(x, W) + b'),
        '. As a ',
        span({ class: 'italic' }, 'readout'),
        ', it takes a stacked vector and dots it against a set of learned "reader" directions to recover the individual contributions stacked into it.',
      ),
    )
  }

  labeledMatrix(name: string, shape: string, data: number[][], opts: {
    rowHl?: number; colHl?: number; cellHl?: { row: number; col: number };
    onClick?: (r: number, c: number) => void;
  }) {
    const cols = data[0]!.length
    return div({ class: 'mat-col' },
      div({ class: 'mat-label' }, `${name}   [${shape}]`),
      div({
        class: 'mat-grid',
        style: { gridTemplateColumns: `repeat(${cols}, var(--cell-w))` },
      }, data.flatMap((row, i) => row.map((v, j) => div({
        class: [
          'mat-cell',
          numClass(v),
          opts.rowHl === i && 'cell-row-hl',
          opts.colHl === j && 'cell-col-hl',
          opts.cellHl?.row === i && opts.cellHl?.col === j && 'cell-sel',
          opts.onClick && 'cell-clickable',
        ],
        onClick: opts.onClick ? () => opts.onClick!(i, j) : undefined,
      }, String(v))))),
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
    return div({ class: 'expr' },
      div(
        span({ class: 'expr-lhs' }, lhs),
        '  =  ', indexTerms,
      ),
      div({ class: 'expr-line', style: { paddingLeft: `${lhs.length + 2}ch` } },
        '=  ', valueTerms,
      ),
      div({ class: 'expr-line', style: { paddingLeft: `${lhs.length + 2}ch` } },
        '=  ', span({ class: 'expr-result' }, String(C[i]![j]!)),
      ),
    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        span({ class: 'mono' }, 'matmul(a, b)'),
        ' is bulk dot product: every row of ',
        span({ class: 'mono' }, 'a'),
        ' dotted against every column of ',
        span({ class: 'mono' }, 'b'),
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
        span({ class: 'mono' }, '[M, K] · [K, N]'),
        ' rule, you rearrange axes first with ',
        span({ class: 'mono' }, 'swapAxes'),
        '. Inside attention you will see ',
        span({ class: 'mono' }, 'matmul(q, swapAxes(k, -1, -2))'),
        ' — K\'s last two axes get swapped so K\'s rows of features become the columns matmul wants to dot against. Pure shape-rearrangement, no math.',
      ),
    )
  }
}

// ---------- Duality section ----------
class DualitySection extends ModeSection {
  angle: number = 90   // degrees between A and B; default to orthogonal (clean recovery)

  setAngle(a: number) {
    this.angle = Math.max(0, Math.min(180, a))
    this.update()
  }

  conceptualView() {
    const aLen = 2.5, bLen = 2.0
    const { A, B, C } = vectorsFromAngle(this.angle, aLen, bLen)
    // Unit-length "readers" aligned with A's and B's directions.
    const θ = this.angle * Math.PI / 180
    const rA: Vec2 = [1, 0]
    const rB: Vec2 = [Math.cos(θ), Math.sin(θ)]
    const got_A = C[0] * rA[0] + C[1] * rA[1]   // C · r_A
    const got_B = C[0] * rB[0] + C[1] * rB[1]   // C · r_B
    const isOrth = this.angle === 90
    const readColor = isOrth ? 'var(--positive)' : 'var(--negative)'

    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        '⊕ writes contributions into a shared vector. To pull one back out, you dot the shared vector against a ',
        span({ class: 'italic' }, 'reader'),
        ' — another vector that points in the direction of the contribution you want to recover. If A and B are orthogonal, a reader pointed along A recovers ',
        span({ class: 'mono' }, '|A|'),
        ' exactly from ',
        span({ class: 'mono' }, 'C = A + B'),
        ' — the B contribution drops to zero. When A and B overlap, the readers leak: each reading picks up some of the wrong vector.',
      ),
      div({ class: 'intro' },
        'In a transformer, the shared vector that every layer ⊕\'s contributions into is called the ',
        span({ class: 'italic' }, 'residual stream'),
        '. It is just a tensor — one D-dim vector at each sequence position, passed forward through the layers and added to at each one.',
      ),

      div({ class: 'viz-row' },
        svg({ class: 'plot', viewBox: `0 0 ${PLOT} ${PLOT}`, width: PLOT, height: PLOT },
          axes(),
          // C first (dashed, drawn under)
          arrow([0, 0], C, { color: 'var(--c-color)', dashed: true, label: 'A + B', labelOffset: { x: 8, y: 16 } }),
          // Readers (unit length, thinner, muted)
          arrow([0, 0], rA, { color: 'var(--a-color)', width: 1.5, headSize: 7, label: 'r_A', labelOffset: { x: 6, y: 18 } }),
          this.angle > 0
            ? arrow([0, 0], rB, { color: 'var(--b-color)', width: 1.5, headSize: 7, label: 'r_B', labelOffset: { x: this.angle < 30 ? 6 : -28, y: this.angle < 30 ? -18 : -8 } })
            : null,
          // Primals on top
          arrow([0, 0], A, { color: 'var(--a-color)', label: 'A' }),
          arrow([0, 0], B, { color: 'var(--b-color)', label: 'B' }),
        ),
        div({ class: 'side-panel' },
          sliderRow({
            label: 'angle between A and B',
            value: `${this.angle}°`,
            min: 0, max: 180, step: 1,
            current: this.angle,
            onChange: v => this.setAngle(v),
          }),
          div({ class: 'detail' },
            detailRow({ label: 'C · r_A', value: fmt(got_A), valColor: readColor, valWeight: '700', aside: `want ${fmt(aLen)}` }),
            detailRow({ label: 'C · r_B', value: fmt(got_B), valColor: readColor, valWeight: '700', aside: `want ${fmt(bLen)}` }),
          ),
          div({ class: 'caption' },
            isOrth
              ? 'A ⊥ B: both readings recover the magnitudes exactly. The two contributions in C are perfectly separable.'
              : 'A and B overlap. Each reading picks up some of the other vector\'s contribution — the recovery degrades by exactly the amount of overlap.',
          ),
        ),
      ),

      div({ class: 'intro' },
        'Through the transformer\'s layers, the residual stream accumulates contributions: token embedding, position embedding, every attention head\'s output, every MLP\'s output. All ⊕\'d into the same 64D vector at each position. Reader directions are learned during training — separate matmul weights for each downstream use — and become approximately orthogonal where the model needs to keep contributions separate.',
      ),

    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'Duality is not a primitive — it is the pattern of writing with ',
        span({ class: 'mono' }, 'add'),
        ' and reading with ',
        span({ class: 'mono' }, 'matmul'),
        '. In isolation:',
      ),

      specBox(
        `// Write: stack two contributions into one shared vector\nconst c = add(a, b)              // a, b, c: [D]`,
        [
          [
            'Both contributions are present in ',
            'c', '. If ',
            'a', ' and ',
            'b', ' point in near-orthogonal directions in high-dimensional D, each remains recoverable from the sum.',
          ],
        ],
      ),

      specBox(
        `// Read: dot the shared vector against learned "reader" vectors\nconst W = param([D, 3])           // a [D × 3] learnable matrix — 3 readers, each a D-dim column\nconst reads = matmul(c, W)        // reads: [3] — one dot product per reader`,
        [
          [
            'Each column of ',
            'W', ' is a "reader" from above. Dotting ',
            'c', ' with each column gives the contribution recovered along that direction. The matmul performs all three reads in one op. The transformer uses this pattern over and over with different W matrices for different downstream jobs.',
          ],
        ],
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
        'Softmax turns a row of arbitrary numbers (logits, for instance the raw scores from a matmul-read) into a probability distribution: every output lands in [0, 1], the whole row sums to 1, and the biggest logit becomes the biggest probability. Differences between logits get exponentially amplified into gaps in probability.',
      ),

      div({ class: 'softmax-controls' },
        sliderRow({
          label: 'logit 0 (leftmost)',
          value: fmt(this.peakLogit, 2),
          min: -2, max: 5, step: 0.1,
          current: this.peakLogit,
          onChange: v => this.setPeak(v),
        }),
      ),

      div({
        class: 'softmax-steps',
        style: { gridTemplateColumns: `auto repeat(${N}, var(--cell-w))` },
      },
        div({ class: 'sm-step-label sm-header-row' }, ''),
        range(N).map(i => div({ class: 'sm-header' }, String(i))),

        stepRow('logits', logits, 2),
        stepRow(
          [span('− max'), span({ class: 'sm-aside' }, `max = ${fmt(max, 2)}`)],
          shifted, 2),
        stepRow('exp', exped, 3),
        stepRow(
          [span('÷ sum'), span({ class: 'sm-aside' }, `sum = ${fmt(sumExp, 3)}`)],
          probs, 3, ['mat-cell', 'sm-prob-cell']),
      ),

      div({ class: 'intro' },
        'The bare formula is ',
        span({ class: 'mono' }, 'softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)'),
        '. But ',
        span({ class: 'mono' }, 'exp(800)'),
        ' overflows float32 long before that matters mathematically. Subtracting the row max before exp guarantees the largest input becomes 0 and every other input is ≤ 0, so every exp lands in (0, 1]. The final probabilities are identical to the bare formula: the same constant gets exp\'d into every term, so it cancels in the divide.',
      ),

      div({ class: 'intro' },
        'Differences in logits compound exponentially: a logit 1 unit higher becomes ~2.7× more probable (',
        span({ class: 'mono' }, 'exp(1) ≈ 2.718'),
        '); 5 units higher, ~148×. softmax is "winner-take-most" by default. Note the sum in step 3 — softmax internally uses a sum over exp\'d logits to normalize. That is how a vector of arbitrary numbers becomes a distribution that sums to exactly 1.',
      ),

      div({ class: 'intro' },
        'For attention specifically, future positions get masked to −∞ before softmax (they become 0 after exp) so each query only attends to itself and earlier positions. The fused variant ',
        span({ class: 'mono' }, 'softmaxCausal'),
        ' does this in one op.',
      ),

    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        span({ class: 'mono' }, 'softmax(x, axis?)'),
        ' applies along one axis (default: last). tensorgrad fuses it for numerical stability (the max-subtraction trick happens inside the op).',
      ),
      specBox(
        `const logits = ...               // [V]\nconst probs  = softmax(logits)   // [V] — all in [0, 1], summing to 1`,
        [
          [
            'softmax', ' acts on the last axis by default. The output values are non-negative and sum to 1 across that axis.',
          ],
        ],
      ),
      specBox(
        `// Attention-specific fused variant\nconst attn = softmaxCausal(scores)   // future positions masked to −∞ before softmax`,
        [
          [
            'softmaxCausal', ' is a fused variant tensorgrad ships for attention. It sets future positions to −∞ (which become 0 after exp) before softmaxing each row, so each query attends only to earlier positions.',
          ],
        ],
      ),
    )
  }
}

// ---------- Composition / capstone section ----------
class CompositionSection extends ModeSection {
  conceptualView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'One block of attention — the central operation of a transformer — composes the primitives ⊕, dot, matmul, softmax into five stages.',
      ),

      div({ class: 'intro' },
        'One mechanism to introduce first: attention runs in parallel across ',
        span({ class: 'italic' }, 'heads'),
        ' — each head operates on its own slice of the D-dim vector at each position. In this transformer D = 64 with 4 heads, so each head gets a 16-dim slice (64 ÷ 4). ',
        span({ class: 'mono' }, 'splitHeads'),
        ' rearranges shape ',
        span({ class: 'mono' }, '[B, T, 64]'),
        ' to ',
        span({ class: 'mono' }, '[B, 4, T, 16]'),
        ' — each head\'s slice on its own axis. ',
        span({ class: 'mono' }, 'mergeHeads'),
        ' concatenates them back. Neither op does any math; they only rearrange axes.',
      ),

      div({ class: 'pipeline' },
        this.stage(1, 'Project Q, K, V',
          `const q = splitHeads(matmul(x, W_q), H)\nconst k = splitHeads(matmul(x, W_k), H)\nconst v = splitHeads(matmul(x, W_v), H)`,
          'Three matmuls — each through its own learned matrix (W_q, W_k, W_v) — turn x into Q, K, V vectors at each position. splitHeads slices each into H heads.',
          'x [B, T, D]  →  q, k, v  [B, H, T, d]'),
        div({ class: 'pipeline-arrow' }, '↓'),

        this.stage(2, 'Score every query against every key',
          `const scores = mul(matmul(q, swapAxes(k, -1, -2)), 1 / Math.sqrt(d))`,
          'One matmul produces every (query, key) dot product at once. swapAxes flips K\'s last two axes so the shape rule lines up. The 1/√d scaling keeps the numbers from going extreme before softmax.',
          'q, k  →  scores  [B, H, T, T]'),
        div({ class: 'pipeline-arrow' }, '↓'),

        this.stage(3, 'Turn scores into attention weights',
          `const attn = softmaxCausal(scores)`,
          'softmax turns each row of scores into a probability distribution. The "causal" variant masks future positions to −∞ first, so each query attends only to itself and earlier positions.',
          'scores  →  attn  [B, H, T, T]'),
        div({ class: 'pipeline-arrow' }, '↓'),

        this.stage(4, 'Weighted sum of values',
          `const headOut = matmul(attn, v)`,
          'Same matmul op as step 2, used differently. Each row of attn weights the earlier V vectors. matmul produces, for each query, the sum of those V vectors scaled by their attention weights. The duality pattern in action: matmul reading what ⊕ wrote.',
          'attn, v  →  headOut  [B, H, T, d]'),
        div({ class: 'pipeline-arrow' }, '↓'),

        this.stage(5, 'Merge heads, project, ⊕ into residual',
          `const blockOut = matmul(mergeHeads(headOut), W_o)\nconst newX     = add(x, blockOut)`,
          'mergeHeads concatenates the per-head outputs back into one D-dim vector per position. One more matmul projects through learned matrix W_o. The result ⊕\'s into x — this block\'s contribution to the residual stream, visible to every later block.',
          'headOut  →  blockOut  [B, T, D]   ⊕ x  →  newX  [B, T, D]'),
      ),

      div({ class: 'intro' },
        'That is one attention block. The transformer stacks several of these (a typical size is anywhere from 6 to 100+ layers). Each block reads the residual ',
        span({ class: 'mono' }, 'x'),
        ', computes its contribution from the primitives above, and ⊕\'s back. The load-bearing math is matmul, softmax, and add — the other names you will see in attention code (',
        span({ class: 'mono' }, 'splitHeads'),
        ', ',
        span({ class: 'mono' }, 'mergeHeads'),
        ', ',
        span({ class: 'mono' }, 'Linear'),
        ', ',
        span({ class: 'mono' }, 'LayerNorm'),
        ') are shape rearrangement and pre-packaged combinations of these same primitives.',
      ),

      div({ class: 'intro' },
        'After attention each block also runs an MLP and ⊕\'s its output too. The MLP is just ',
        span({ class: 'mono' }, 'matmul → relu → matmul → add'),
        ' — the same primitives, simpler shape.',
      ),
    )
  }

  stage(num: number, title: string, code: string, desc: string, shapes: string) {
    return div({ class: 'pipeline-stage' },
      div({ class: 'pipeline-stage-num' }, String(num)),
      div({ class: 'pipeline-stage-body' },
        div({ class: 'pipeline-stage-title' }, title),
        pre({ class: 'pipeline-stage-code' }, code),
        div({ class: 'pipeline-stage-desc' }, desc),
        div({ class: 'pipeline-stage-shapes' }, shapes),
      ),
    )
  }

  tensorgradView() {
    return div({ class: 'tab-content-inner' },
      div({ class: 'intro' },
        'Here is an attention block, top to bottom. The load-bearing math is the matmul / softmax / add you just learned. The rest — ',
        span({ class: 'mono' }, 'splitHeads'),
        ', ',
        span({ class: 'mono' }, 'mergeHeads'),
        ', and the ',
        span({ class: 'mono' }, '.fwd'),
        ' wrappers on ',
        span({ class: 'mono' }, 'Linear'),
        ' and ',
        span({ class: 'mono' }, 'LayerNorm'),
        ' — is shape management and module abstraction layered on top.',
      ),

      specBox(
        `function attentionFwd(p: Attention, x: Tensor): Tensor {\n  const q = splitHeads(p.q.fwd(x), N_HEADS)\n  const k = splitHeads(p.k.fwd(x), N_HEADS)\n  const v = splitHeads(p.v.fwd(x), N_HEADS)\n  const scores = mul(matmul(q, swapAxes(k, -1, -2)), 1 / Math.sqrt(D_HEAD))\n  const attn = softmaxCausal(scores)\n  return p.o.fwd(mergeHeads(matmul(attn, v)))\n}`,
        [
          [
            'Six lines of body. Three projections produce Q, K, V; one matmul scores; one softmax normalizes; one matmul reads weighted values; one final projection (',
            'p.o.fwd', ') mixes the heads. The core math is matmul, softmax (causal variant), and add; ',
            'splitHeads', ' and ', 'mergeHeads', ' are pure shape rearrangement.',
          ],
        ],
      ),

      specBox(
        `function blockFwd(p: Block, x: Tensor): Tensor {\n  const a = attentionFwd(p.attn, p.ln1.fwd(x))\n  const x1 = add(x, a)\n  return add(x1, mlpFwd(p.mlp, p.ln2.fwd(x1)))\n}`,
        [
          [
            'And the surrounding block. Attention runs over a normalized copy of ',
            'x', ', the result ⊕\'s into ',
            'x', '; MLP runs over a normalized copy of ',
            'x1', ' and ⊕\'s into that. Every transformer block is two ⊕\'s into the residual stream — one from attention, one from the MLP. Stack ',
            'N_LAYERS', ' of these and you have the transformer.',
          ],
        ],
      ),
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
  dualitySection     = new DualitySection()
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
          h1('Tensors'),
          p({ class: 'subtitle' },
            'Geometric intuition for the basic operations a transformer uses — vectors, dot product, shapes, embedding, matmul, duality, softmax — and how they compose into one block of attention. Each tab has a Conceptual view (the geometric story) and an In tensorgrad view (real lines from a transformer that learns 2-digit addition).',
          ),
        ),
      ),
      div({ class: 'main' },
        this.tabs.view({
          tabs: [
            { key: 'add',         label: 'Vectors',     content: this.addSection.view() },
            { key: 'dot',         label: 'Dot',         content: this.dotSection.view() },
            { key: 'shapes',      label: 'Shapes',      content: this.shapesSection.view() },
            { key: 'embedding',   label: 'Embedding',   content: this.embeddingSection.view() },
            { key: 'matmul',      label: 'matmul',      content: this.matmulSection.view() },
            { key: 'duality',     label: 'Duality',     content: this.dualitySection.view() },
            { key: 'softmax',     label: 'softmax',     content: this.softmaxSection.view() },
            { key: 'composition', label: 'Altogether now', content: this.compositionSection.view() },
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
  --c-color: rgb(140, 60, 180);
  --axis: rgb(200, 200, 200);
  --code-bg: rgb(245, 245, 245);
  --code-border: rgb(220, 220, 220);
  --cell-w: 52px;
  --cell-h: 44px;
  --row-hl: rgba(50, 130, 255, 0.12);
  --col-hl: rgba(50, 130, 255, 0.12);
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
  --c-color: rgb(200, 130, 240);
  --axis: rgb(70, 70, 75);
  --code-bg: rgb(15, 15, 18);
  --code-border: rgb(50, 50, 55);
  --row-hl: rgba(90, 165, 255, 0.18);
  --col-hl: rgba(90, 165, 255, 0.18);
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
  padding: 28px 0 22px;
  border-bottom: 1px solid var(--border);
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

.subtitle {
  color: var(--text-muted);
  margin: 8px 0 0;
  font-size: 14.5px;
  line-height: 1.6;
  max-width: 720px;
}

.main {
  padding-top: 28px;
  padding-bottom: 56px;
}

/* Main tab bar */
.tab-control .tab-bar {
  display: flex;
  gap: 24px;
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
  font-size: 14px;
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
  font-size: 13px;
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

/* Per-tab content shell */
.tab-content-inner > * + * { margin-top: 22px; }

.intro {
  color: var(--text);
  font-size: 14.5px;
  line-height: 1.65;
  max-width: 720px;
}

.italic { font-style: italic; }

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
  flex-shrink: 0;
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
  font-size: 13.5px;
}

.detail-row {
  display: flex;
  gap: 6px;
  align-items: baseline;
  flex-wrap: wrap;
}

.detail-lbl { font-weight: 600; min-width: 80px; }
.detail-val { color: var(--text); }

.control-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 13px;
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

.caption {
  font-size: 12.5px;
  color: var(--text-muted);
  line-height: 1.5;
  font-style: italic;
}

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
  font-size: 13.5px;
  line-height: 1.6;
  margin: 0 0 12px;
  overflow-x: auto;
  white-space: pre;
  color: var(--text);
}

.spec-caption {
  color: var(--text-muted);
  font-size: 13.5px;
  line-height: 1.65;
  margin: 0;
}

.spec-caption + .spec-caption { margin-top: 10px; }

.mono {
  font-family: monospace;
  color: var(--text);
}

/* Matrix cells (matmul + sum) */
.mat-col {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.mat-label {
  font-family: monospace;
  font-size: 12.5px;
  color: var(--text-muted);
  margin-bottom: 6px;
  letter-spacing: 0.02em;
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
  font-size: 14px;
  font-weight: 500;
  user-select: none;
  transition: background 120ms, box-shadow 120ms;
}

.num-pos  { color: var(--positive); }
.num-neg  { color: var(--negative); }
.num-zero { color: var(--zero); }

.cell-clickable { cursor: pointer; }
.cell-clickable:hover:not(.cell-sel) { background: var(--row-hl); }

.cell-row-hl:not(.cell-sel) { background: var(--row-hl); }
.cell-col-hl:not(.cell-sel) { background: var(--col-hl); }

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
  padding: 16px 20px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-family: monospace;
  font-size: 14px;
  line-height: 1.85;
  color: var(--text);
  overflow-x: auto;
}

.expr-lhs    { font-weight: 600; }
.expr-result { font-weight: 700; font-size: 16px; }

/* Tensors tab */
.shape-ladder {
  display: flex;
  align-items: center;
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
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.shape-rung-notation {
  font-family: monospace;
  font-size: 11.5px;
  color: var(--text-muted);
}

.shape-arrow {
  font-size: 20px;
  color: var(--text-muted);
}

.embed-block {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
}

.embed-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 6px;
}

.embed-prose {
  font-size: 13.5px;
  line-height: 1.6;
  color: var(--text-muted);
  margin-bottom: 14px;
  max-width: 640px;
}

.embed-row-host {
  display: flex;
  gap: 28px;
  align-items: flex-start;
  flex-wrap: wrap;
}

.embed-table {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.embed-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 3px 6px;
  border-radius: 4px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: background 120ms, border-color 120ms;
}

.embed-row:hover:not(.embed-row-sel) { background: var(--row-hl); }

.embed-row-sel {
  background: var(--selected-bg);
  border-color: var(--selected-border);
}

.embed-idx {
  font-family: monospace;
  font-size: 13px;
  color: var(--text-muted);
  min-width: 22px;
  text-align: right;
}

.embed-cells {
  display: flex;
  gap: 1px;
  background: var(--cell-border);
  border: 1px solid var(--cell-border);
  border-radius: 4px;
  overflow: hidden;
}

.embed-out-col {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}

.embed-out-call {
  font-family: monospace;
  font-size: 13px;
  color: var(--text);
  padding: 4px 8px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 4px;
}

.embed-out-arrow {
  font-size: 18px;
  color: var(--text-muted);
  align-self: center;
}

/* Composition tab */
.pipeline {
  display: flex;
  flex-direction: column;
  gap: 0;
  max-width: 720px;
}

.pipeline-stage {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 18px;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 14px;
}

.pipeline-stage-num {
  font-family: monospace;
  font-size: 22px;
  font-weight: 700;
  color: var(--accent);
  min-width: 28px;
  line-height: 1.1;
}

.pipeline-stage-body {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}

.pipeline-stage-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.pipeline-stage-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
}

.pipeline-stage-ops {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.pipeline-op-chip {
  font-family: monospace;
  font-size: 11.5px;
  background: var(--code-bg);
  border: 1px solid var(--code-border);
  border-radius: 4px;
  padding: 2px 8px;
  color: var(--text);
  white-space: nowrap;
}

.pipeline-stage-code {
  font-family: monospace;
  font-size: 12.5px;
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

.pipeline-stage-desc {
  font-size: 13.5px;
  color: var(--text-muted);
  line-height: 1.55;
}

.pipeline-stage-shapes {
  font-family: monospace;
  font-size: 12.5px;
  color: var(--text);
  padding: 6px 10px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow-x: auto;
  white-space: nowrap;
}

.pipeline-arrow {
  text-align: center;
  font-size: 18px;
  color: var(--text-muted);
  padding: 4px 0;
}

/* softmax tab */
.softmax-controls {
  max-width: 360px;
}

.softmax-steps {
  display: grid;
  gap: 1px;
  background: var(--cell-border);
  border: 1px solid var(--cell-border);
  border-radius: 6px;
  overflow: hidden;
}

.sm-step-label,
.sm-header {
  display: flex;
  align-items: center;
  background: var(--cell-bg);
  height: var(--cell-h);
  font-family: monospace;
  color: var(--text);
}

.sm-step-label {
  padding: 0 14px;
  color: var(--text-muted);
  font-size: 13px;
  justify-content: flex-start;
  gap: 8px;
  white-space: nowrap;
}

.sm-header {
  justify-content: center;
  font-size: 11.5px;
  color: var(--text-muted);
  letter-spacing: 0.04em;
}

.sm-header-row { background: var(--cell-bg); }

.sm-aside {
  font-size: 11.5px;
  color: var(--text-muted);
  opacity: 0.85;
}

.sm-prob-cell {
  font-weight: 700;
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
  "description": "Foundational tensor primitives — vectors, dot product, shapes, embedding, matmul, duality, softmax — and how they compose into one block of attention. Each tab has a Conceptual view (the geometric story) and an In-tensorgrad view drawn from a transformer that learns 2-digit addition."
}
```
