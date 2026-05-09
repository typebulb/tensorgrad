# Activation capture — design spec

Status: design pass, **pre-implementation**.
Authors: Claude (this window) + Ben.
Scope: tensorgrad 0.0.4+ (deferred from the 0.0.3 ergonomics bundle).

## Problem

The compiled graph already computes every intermediate tensor on the GPU.
The IR has each one allocated as a buffer in the plan. But today the
runtime only reads the loss scalar back to JS. Anything else — attention
maps, per-layer residuals, MLP hidden activations, logits at any position —
is invisible to the caller.

Concrete cost: the transformer-tensorgrad bulb has ~250 lines of
hand-written JS forward (`forwardCpu`, `predictAddition`,
`logitLensAtPosition`, plus matmul/softmax/layernorm helpers) that exist
*only* to compute these intermediates a second time on CPU for the viz
panels. Two failure modes:

1. **Silent drift.** The JS forward and the WGSL forward can disagree
   numerically (different reduction order, different epsilons, missing
   masks). The viz looks coherent and is silently wrong.
2. **Maintenance burden.** Every change to the model architecture has to
   be reflected in two places.

Both go away if tensorgrad can hand back any tensor the user names.

## Goals

1. **Pay-as-you-go.** Reading the loss alone should cost what it costs
   today. Adding capture should add cost only for the buffers actually
   read back.
2. **Per-step opt-in.** Some steps want captures (UI tick), most don't
   (training loop). Toggle without recompile.
3. **Locality.** A user writing `attentionFwd(p, x)` should be able to
   mark `softmaxCausalLast(scores)` for capture without bubbling its
   value back through the call stack.
4. **Type-safe results** where possible. `Float32Array` (or `Float32Array[]`)
   per registered name, keyed off the user's declarations.
5. **No silent buffer reuse.** A captured tensor's buffer must survive
   to the end of the step; the buffer planner needs to know which IDs
   are pinned.

## Design options

### Option A — return-shape declaration

Forward function signs the contract by returning more than a `Tensor`:

```ts
function forward(m: Transformer, tokens: Tensor): {
  loss: Tensor
  captures: {
    residuals: Tensor[]      // one per layer
    attnMaps:  Tensor[]
    mlpHiddens: Tensor[]
  }
}
```

`compileModule` reads the return shape; runtime returns matching shape:

```ts
const { loss, captures } = await compiled.step({ tokens, ... })
// loss: number
// captures.residuals: Float32Array[]
// captures.attnMaps:  Float32Array[]
// captures.mlpHiddens: Float32Array[]
```

**Pros:**
- Static. The function signature *is* the spec.
- TypeScript can infer capture shape automatically.
- No globals during trace.
- "Toggle off captures per step" is a flag on `step()`.

**Cons:**
- **Threading problem.** A helper like `attentionFwd` can't register a
  capture without changing its return type. You either restructure
  the helper to return `{ output, attnMap }`, or build the captures
  object in the top-level forward by hand (which means the helper has
  to expose the intermediate as a return value).
- The "what gets captured" decision lives at the top of the forward,
  often far from where the tensor exists.
- Mixing a model's logical return (loss) with a diagnostic concern
  (captures) at the same return shape.

### Option B — side-channel `capture()` call

A `capture(name, tensor)` function recorded during trace adds the
tensor to a per-graph capture map. The forward function still returns
just the loss tensor.

```ts
import { capture } from 'tensorgrad'

function attentionFwd(p: Attention, x: Tensor): Tensor {
  const scores = mul(matmulBatched(q, kT), SCALE_QK)
  const attn = softmaxCausalLast(scores)
  capture('attn', attn)         // marks for readback
  return matmulBatched(attn, v)
}
```

`compile` collects all `capture()` calls; runtime returns them keyed by
name:

```ts
const { loss, captures } = await compiled.step({ tokens }, { withCaptures: true })
// captures: Record<string, Float32Array>
```

**Pros:**
- **Locality.** Capture at the source. No threading, no helper-signature
  changes.
- Existing forward functions migrate trivially (sprinkle `capture()`
  calls; everything else unchanged).
- Naturally handles per-layer capture via name templating
  (`capture(\`attn.${i}\`, ...)` inside a loop).

**Cons:**
- Implicit. To know what's captured you grep, not read the function
  signature.
- Capture names live in user-space and aren't statically typed —
  `captures['atnn.0']` typo silently returns undefined unless we throw
  on unknown keys.
- Globally-scoped trace state (already exists for the graph itself, but
  this is more user-visible).

### Option C — hybrid: capture-as-pass-through

Same as B mechanically, but `capture()` returns its argument:

```ts
const attn = capture('attn', softmaxCausalLast(scores))
//      ^ Tensor — returns the same tensor, side-effect registers it
```

This makes the capture inline-able at the point of computation without
adding a separate statement. Otherwise identical to B.

## Recommendation

**Option C (capture-as-pass-through).** Reasoning:

- The threading problem from A is real. The bulb's existing JS forward
  literally exists to dodge this — every helper threads back
  `(output, attnMap)`. Forcing the same threading on the WGSL forward
  rebuilds that pain.
- B/C captures are cheap to add to existing forwards without rewrites.
  The bulb's transformer can sprinkle `capture()` calls where it
  currently calls `recordCapture()` in its JS forward; same pattern.
- C's pass-through form keeps the call site readable (`const attn =
  capture('attn', softmaxCausalLast(...))`).
- Type safety on capture names: solved by a separate runtime check
  ("unknown capture key") at step time, not by the type system.

Letter from prior session preferred A. Pushing back: A's threading
overhead in helpers is the same overhead the JS-forward already has,
which is what we're trying to delete. C avoids that.

If users hit issues with C (unclear what's captured, name collisions
across modules), we can layer A on top later — both can coexist.

## Implementation outline

### 1. `capture()` API

```ts
// In ops.ts (or new src/capture.ts):
export function capture<T extends Tensor>(name: string, t: T): T {
  const g = currentGraph()
  if (g.captures.has(name)) {
    throw new Error(`capture: name '${name}' already registered`)
  }
  g.captures.set(name, t.id)
  return t
}
```

`Graph` gains a `captures: Map<string, tensorId>` field. Mutation is
guarded against duplicate names; can relax later if pattern requires
(e.g., `capture(\`L${i}.attn\`, ...)` produces unique names already).

### 2. Buffer planning

Capture tensors must:
- Not be elided by intermediate-buffer reuse (their buffer must persist
  through the step).
- Be readable post-dispatch (need `COPY_SRC` usage flag, which all
  `STORAGE_RW` buffers already have today).

Implementation: `planBuffers` consults `graph.captures`; any tensorId
in that set is pinned (kept as its own buffer, not folded with another
intermediate's).

**Pinning is write-output exclusive, not read exclusive.** A captured
tensor's buffer can still be read as an input by other ops — pinning
only forbids *another op writing into the same buffer slot*. If
`planBuffers` already only fuses output slots (which it does today),
pin handling is a one-line check; ops that read the captured tensor
keep working unchanged.

### 2a. `capture()` is a no-op outside the user's forward trace

`appendGrad` and `appendAdam` both re-enter the graph via `traceInto`.
Their backward / optimizer ops should *not* register captures —
nobody wants accidental gradient-tensor readbacks driving the UI.

Implementation: the trace context carries a `captureEnabled` flag, set
to `true` only during the user's forward trace and `false` during
`appendGrad` / `appendAdam`. `capture()` is a no-op when the flag is
off. Cheaper than try/catch and explicit at the call site that
captures are forward-only.

### 3. Runtime: extra readbacks per step

`step()` gains an optional `{ withCaptures?: boolean }` flag. When set:

- Before submit: for each capture buffer, copyBufferToBuffer to a
  staging buffer (created at runtime construction; one staging per
  capture, sized to the capture tensor).
- After submit + loss readback: mapAsync each staging, slice into
  `Float32Array`, return alongside loss.

```ts
step(
  inputs: Record<string, Int32Array | Float32Array>,
  opts?: { withCaptures?: boolean },
): Promise<number> | Promise<{ loss: number; captures: Record<string, Float32Array> }>
```

Return-type discrimination via overload (TS supports this cleanly when
a literal `true` is passed).

### 4. `compileModule` signature

No change. Forward function still returns `Tensor`. The `capture()`
calls inside it register tensors via the active trace context.

### 5. Cost when not used

- Compile: identical (no captures registered → graph.captures empty →
  buffer planner short-circuits → no staging buffers allocated).
- Step: identical for `step({ ... })` calls (no `withCaptures` flag →
  no extra copyBufferToBuffer, no extra mapAsync).
- Reading `loss` in normal training loop is byte-for-byte the same.

## Migration impact

### samples/transformer
No mandatory change. Optionally: add `capture('attn.${i}', attn)` inside
`attentionFwd` if a viz panel ever wants those.

### bulb (transformer-tensorgrad.bulb.md)
Significant: ~250-line CPU forward goes away. New flow:

```ts
function attentionFwd(p: Attention, x: Tensor, layerIdx: number): Tensor {
  // ... build attn ...
  capture(`attn.${layerIdx}`, attn)
  return matmulBatched(attn, v)
}

function forward(m: Transformer, tokens: Tensor): Tensor {
  let x = embed(...)
  for (let i = 0; i < N_LAYERS; i++) {
    capture(`residual.${i}`, x)
    x = blockFwd(m.layers[i], x, i)
  }
  return loss(...)
}

// Per UI tick:
const { loss, captures } = await compiled.step(batch, { withCaptures: true })
const attnMaps = Array.from({ length: N_LAYERS }, (_, i) => captures[`attn.${i}`])
```

### `predictAddition` (bulb only)
This currently runs a *separate* forward pass with batch=1 to generate
predictions. With capture, it's a different problem: the compiled graph
has fixed batch B=512, so we can't run with B=1. Two options here:

1. Compile a second, prediction-only graph at compile time (B=1 for
   the prediction path; same params reused via shared state buffers).
2. Pad the B=1 prediction up to B=512 and only consume the first row.

**This is orthogonal to capture and the bulb's CPU code does not drop
to zero.** Capture solves "watch a training step's intermediates."
Prediction is "what does the model say about this *specific* input,"
which still needs a B=1 inference path. The bulb may keep a small
CPU forward (or compile a second graph) for that. Don't expect the
~250 lines to become 0; expect the *training-tap* portion to vanish.

## Open questions

1. **Capture name collisions across calls.** If `attentionFwd` is called
   twice (e.g., two layers) with the same `capture('attn', ...)`, we
   throw. The user has to disambiguate (`capture(\`attn.${i}\`, ...)`).
   Acceptable footgun? Or auto-suffix?
   *Lean: throw, force explicit naming. Auto-suffix hides the issue.*

2. **`capture` with array shapes.** Should `capture` allow registering
   `Tensor[]` so the user can read it as `Float32Array[]`?
   *Lean: no. Single tensors only; arrays are user-side
   `Array.from({ length: N }, (_, i) => captures[\`x.${i}\`])`.*

3. **Multiple step() shapes.** With overloaded return type, TS
   inference at the call site can be touchy.
   *Plan: define two overloads on the `step` interface, narrowed by
   the literal type of `withCaptures`.*

4. **Cost of always-allocated staging buffers.** Even with no captures
   in the graph, the runtime currently allocates one staging buffer
   per parameter for `downloadParams`. Captures would add more. For
   models with many captures (per-layer attention maps × N_LAYERS),
   memory grows.
   *Lean: lazy allocation — staging buffers created on first
   `withCaptures: true` call, reused thereafter.*

5. **Should capture register on a *path* (auto-derived from the trace
   call stack), not a name?** That would eliminate collision concerns
   entirely. But the trace context doesn't currently know which
   `Module.method` is on the stack; threading that through is its own
   feature.
   *Lean: name-based for v0.4. Revisit auto-paths if name collisions
   become a real pattern. **If/when paths land, they should derive
   from the `Module` instance path the capture fires from
   (`layers.0.attn`), not a generic call stack** — same
   auto-derivation logic as `materializeParams`. That keeps capture
   names consistent with the param-naming convention.*

## Verification

### Numerical equivalence
Before declaring the feature done, prove the capture path is order-
irrelevant: train one step with `withCaptures: false`, train an
identical step (same seed, same batch) with `withCaptures: true`,
confirm the loss returns are bit-identical. The capture path adds
`copyBufferToBuffer` calls to the encoder; those should not affect
the kernel outputs at all, but a one-line check catches mistakes
(e.g., capture inadvertently writing into the wrong buffer).

### Bulb shrinkage
- Delete the *training-tap* portion of `forwardCpu` and the
  matmul/softmax/layernorm helper duplications it uses for that.
  (Most of the ~250 lines.)
- Keep whatever's needed for `predictAddition`'s B=1 inference path
  (see Migration impact above).
- Sprinkle `capture()` calls in the GPU-traced forward.
- Per-tick, switch to `step(batch, { withCaptures: true })`.
- Verify viz panels render the same output as before.
- Held-out accuracy still hits >90% (no regression in the train path).

If the training-tap code is gone and viz output is visually identical,
the feature is done.

## Order of work

1. Add `Graph.captures` field + `capture()` op + trace integration
   (with the `captureEnabled` flag for backward/optimizer passes).
2. Buffer planner: pin capture tensors.
3. Runtime: lazy staging + extra readback per `withCaptures: true`
   step.
4. **Bulb first**: replace the training-tap JS forward with
   `capture()` calls and `withCaptures: true`. The bulb is the real
   verification surface — if it shortens meaningfully and the viz
   matches, the API delivers.
5. Sample (transformer): add `capture()` calls in attention/blocks
   for kernel-coverage in the smoke test. Lower priority than (4) —
   the smoke test only checks structural soundness, doesn't prove
   ergonomics.
6. Bump `0.0.4` and publish.
