// Wire format for the main-thread ↔ worker postMessage channel.
//
// All requests carry a numeric `id` assigned by the main thread; responses
// echo it back so the proxy can match concurrent in-flight calls. Every
// response is either `{ ok: true, result }` or `{ ok: false, error }`.
// Errors carry serialized name/message/stack so the proxy can reconstitute
// an Error with a working `instanceof` check on the receiving side.
//
// Inputs (typed arrays) and outputs (typed arrays, captures) are transferred
// rather than copied — see the per-request notes for which fields go on the
// transfer list. A single worker may host multiple compiled graphs (a train
// graph plus sibling forward graphs); each has a `graphId` issued by the
// main thread at compile time.

import type { Graph } from './ir.js'
import type { BufferPlan } from './buffers.js'
import type { KernelSpec } from './codegen.js'
import type { LR } from './adam.js'

// ============================================================================
// Serializable config (subset of AdamResolvedConfig that crosses the wire).
// `decayFilter` (a function, used only at compile time) is NOT part of this —
// the per-param decay decision is already baked into the IR by appendAdam
// before the IR ships to the worker.
// ============================================================================

export interface WireAdamConfig {
  lr: LR
  b1: number
  b2: number
  eps: number
  weightDecay: number
  lrIsScheduled: boolean
  /** Names of the per-step scalar inputs the worker must populate before
   *  every step (`_adam_lrt`, optionally `_adam_decay_shrink`). Mirrors
   *  AdamResult so the worker can update them without re-deriving. */
  lrtInputName: string
  decayShrinkInputName: string | null
}

/** Serializable subset of SGDResolvedConfig for the wire. `decayFilter` is
 *  baked into the IR at appendSGD time, so it doesn't cross. */
export interface WireSGDConfig {
  lr: LR
  momentum: number
  nesterov: boolean
  weightDecay: number
  lrIsScheduled: boolean
  /** Name of the per-step lr scalar input the worker must populate before
   *  every step (`_sgd_lr`). */
  lrInputName: string
}

/** Discriminated optimizer config that crosses the wire. Exactly one branch
 *  is populated when training; `null` for forward-only compiles. */
export type WireOptimizerConfig =
  | { kind: 'adam'; config: WireAdamConfig }
  | { kind: 'sgd'; config: WireSGDConfig }

/** Compile output that crosses to the worker. Same fields as CompiledIR
 *  minus the `loss` tensor (carried by graph.outputs[0]). */
export interface WireIR {
  graph: Graph
  plan: BufferPlan
  kernels: KernelSpec[]
}

// ============================================================================
// Requests (main → worker)
// ============================================================================

export type Req =
  | { id: number; kind: 'createRuntime'; payload: CreateRuntimePayload }
  | { id: number; kind: 'compileForward'; payload: CompileForwardPayload }
  | { id: number; kind: 'step'; payload: StepPayload }
  | { id: number; kind: 'run'; payload: RunPayload }
  | { id: number; kind: 'uploadParams'; payload: UploadParamsPayload }
  | { id: number; kind: 'downloadParams'; payload: { graphId: number } }
  | { id: number; kind: 'downloadParamGrads'; payload: { graphId: number } }
  | { id: number; kind: 'resetOptimizer'; payload: { graphId: number } }
  | { id: number; kind: 'setOptimizerConfig'; payload: SetOptimizerConfigPayload }
  | { id: number; kind: 'destroy'; payload: { graphId: number } }

/** Build the training runtime. Always graphId=0 for a fresh worker. */
export interface CreateRuntimePayload {
  graphId: number
  ir: WireIR
  /** Initial param values per name. Transferred (zero-copy) — the main
   *  thread loses access after postMessage. */
  initialParams: Record<string, Float32Array>
  /** Optimizer config when training; `null` for forward-only compiles. */
  optimizer: WireOptimizerConfig | null
}

/** Build a sibling forward-only graph that shares param buffers with an
 *  existing graph (typically the training graph at graphId=0). */
export interface CompileForwardPayload {
  graphId: number
  parentGraphId: number
  ir: WireIR
}

/** One training step. Inputs are transferred; the caller's typed arrays
 *  become detached after postMessage. */
export interface StepPayload {
  graphId: number
  inputs: Record<string, Int32Array | Float32Array>
  withCaptures: boolean
}

/** Forward-only run. Same transfer semantics as `step`. */
export interface RunPayload {
  graphId: number
  inputs: Record<string, Int32Array | Float32Array>
  withCaptures: boolean
}

export interface UploadParamsPayload {
  graphId: number
  params: Record<string, Float32Array>  // transferred
  partial: boolean
}

/** Update one or more Adam hyperparameters on a training graph at runtime,
 *  without recompiling. The step counter is preserved. Only the fields
 *  present are updated; absent fields stay unchanged. Note that the set
 *  of decayed params is baked into the IR at compile time — adjusting
 *  weightDecay here changes the shrink magnitude on already-decayed
 *  params, not which params receive decay.
 *
 *  When `update.lr` is a non-constant schedule with no explicit `startStep`,
 *  the worker auto-rebases it so step 1 aligns with the next training step
 *  ("decay from now"). Numbers, `constant`, and schedules with an explicit
 *  `startStep` pass through unchanged. */
export interface SetOptimizerConfigPayload {
  graphId: number
  update: {
    lr?: LR
    weightDecay?: number
    b1?: number
    b2?: number
  }
}

// ============================================================================
// Responses (worker → main)
// ============================================================================

export type Res<R = unknown> =
  | { id: number; ok: true; result: R }
  | { id: number; ok: false; error: WireError }

export interface WireError {
  name: string
  message: string
  stack: string
}

// Per-request result shapes:

export interface CreateRuntimeResult {
  paramNames: string[]
  outputShape: number[]
  kernelCount: number
  captureShapes: Record<string, number[]>
}

export interface CompileForwardResult {
  paramNames: string[]
  outputShape: number[]
  kernelCount: number
  captureShapes: Record<string, number[]>
}

/** Step without `withCaptures` returns just `loss`. With captures, also
 *  populates `captures` (per-name Float32Array, all transferred back). */
export interface StepResultWire {
  loss: number
  captures: Record<string, Float32Array> | null
}

/** Run without `withCaptures` returns `{ output, captures: null }`.
 *  With captures, also populates `captures`. */
export interface RunResultWire {
  output: Float32Array
  captures: Record<string, Float32Array> | null
}

export interface DownloadParamsResult {
  params: Record<string, Float32Array>  // transferred
}

// ============================================================================
// Transfer-list helpers
// ============================================================================

/** Collect the underlying ArrayBuffers from a Record of typed arrays so we
 *  can pass them on `postMessage`'s transfer list. The values themselves
 *  stay in the Record; only their backing buffers move. */
export function transferablesOfRecord(
  rec: Record<string, Int32Array | Float32Array>,
): ArrayBuffer[] {
  const out: ArrayBuffer[] = []
  for (const v of Object.values(rec)) out.push(v.buffer as ArrayBuffer)
  return out
}

/** Serialize an Error to a wire-friendly shape, preserving stack + name so
 *  the receiving side can reconstitute an Error that an `instanceof`-aware
 *  caller (e.g., for `ShapeError`) can still pattern-match by name. */
export function wireError(e: unknown): WireError {
  if (e instanceof Error) {
    return { name: e.name, message: e.message, stack: e.stack ?? '' }
  }
  return { name: 'Error', message: String(e), stack: '' }
}

/** Reconstitute an Error from the wire shape on the receiving (main) side. */
export function reconstituteError(w: WireError): Error {
  const err = new Error(w.message)
  err.name = w.name
  err.stack = w.stack
  return err
}
