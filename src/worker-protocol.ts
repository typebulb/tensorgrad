// Wire format for the main-thread ↔ worker postMessage channel. Requests
// carry a numeric `id`; responses echo it back. Inputs and outputs (typed
// arrays, captures) are transferred zero-copy. A single worker may host
// multiple graphs (training + sibling forwards) keyed by `graphId`.

import type { Graph } from './ir.js'
import type { BufferPlan } from './buffers.js'
import type { KernelSpec } from './codegen.js'
import type { LR } from './adam.js'

// Per-param decay flags are baked into the IR by appendAdam/appendSGD before
// it ships to the worker, so `decayFilter` isn't part of the wire types.

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

/** CompiledIR minus the `loss` tensor (carried by graph.outputs[0]). */
export interface WireIR {
  graph: Graph
  plan: BufferPlan
  kernels: KernelSpec[]
}

// ---- Requests (main → worker) -------------------------------------------

export type Req =
  | { id: number; kind: 'createRuntime'; payload: CreateRuntimePayload }
  | { id: number; kind: 'compileForward'; payload: CompileForwardPayload }
  | { id: number; kind: 'step'; payload: StepPayload }
  | { id: number; kind: 'queueStep'; payload: StepPayload }
  | { id: number; kind: 'readLoss'; payload: { graphId: number } }
  | { id: number; kind: 'run'; payload: RunPayload }
  | { id: number; kind: 'uploadParams'; payload: UploadParamsPayload }
  | { id: number; kind: 'downloadParams'; payload: { graphId: number } }
  | { id: number; kind: 'downloadParamGrads'; payload: { graphId: number } }
  | { id: number; kind: 'resetOptimizer'; payload: { graphId: number } }
  | { id: number; kind: 'setLR'; payload: SetLRPayload }
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
 *  become detached after postMessage. Captures are always read back for
 *  graphs that registered any during the trace (no-op for graphs that
 *  didn't). */
export interface StepPayload {
  graphId: number
  inputs: Record<string, Int32Array | Float32Array>
}

/** Forward-only run. Same transfer semantics as `step`. */
export interface RunPayload {
  graphId: number
  inputs: Record<string, Int32Array | Float32Array>
}

export interface UploadParamsPayload {
  graphId: number
  params: Record<string, Float32Array>  // transferred
}

/** Update the lr on a training graph at runtime, without recompiling. The
 *  step counter is preserved. Non-constant schedules without an explicit
 *  `startStep` auto-rebase so step 1 = the next training step. */
export interface SetLRPayload {
  graphId: number
  lr: LR
}

// ---- Responses (worker → main) ------------------------------------------

export type Res<R = unknown> =
  | { id: number; ok: true; result: R }
  | { id: number; ok: false; error: WireError }

export interface WireError {
  name: string
  message: string
  stack: string
}

/** Reply for both `createRuntime` and `compileForward`: the shape metadata
 *  the main thread needs to expose on the runtime handle (param order,
 *  output shape, kernel count for status, capture shape index). */
export interface CompileResult {
  paramNames: string[]
  outputShape: number[]
  kernelCount: number
  captureShapes: Record<string, number[]>
}

export interface StepResultWire {
  loss: number
  /** Null when the graph registered no captures. */
  captures: Record<string, Float32Array> | null
}

export interface RunResultWire {
  output: Float32Array
  captures: Record<string, Float32Array> | null
}

/** Loss readback. Pair with `queueStep`'s fire-and-forget training pattern. */
export interface ReadLossResult {
  loss: number
}

export interface DownloadParamsResult {
  params: Record<string, Float32Array>
}

/** Collect the ArrayBuffers from a Record of typed arrays for postMessage's
 *  transfer list. The values stay in the Record; their buffers move. */
export function transferablesOfRecord(
  rec: Record<string, Int32Array | Float32Array>,
): ArrayBuffer[] {
  const out: ArrayBuffer[] = []
  for (const v of Object.values(rec)) out.push(v.buffer as ArrayBuffer)
  return out
}

/** Serialize an Error, preserving `name` so callers can still match on it
 *  after reconstitution (e.g., for `ShapeError`). */
export function wireError(e: unknown): WireError {
  if (e instanceof Error) {
    return { name: e.name, message: e.message, stack: e.stack ?? '' }
  }
  return { name: 'Error', message: String(e), stack: '' }
}

export function reconstituteError(w: WireError): Error {
  const err = new Error(w.message)
  err.name = w.name
  err.stack = w.stack
  return err
}
