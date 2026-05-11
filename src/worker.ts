// Worker entry point. Holds the GPUDevice + CompiledRuntime for one or more
// graphs and proxies main-thread requests via postMessage. See
// specs/WorkerArchitecture.md for the rationale.
//
// Keep this file dependency-free of anything DOM-y: it bundles into a Blob
// URL and runs in a Web Worker context where `window`/`document` don't
// exist. WebGPU IS available in workers (Chrome 113+, Safari 17.4+).

import { createRuntime, type CompiledRuntime, type RuntimeOpts } from './runtime.js'
import { resolveLR, rebaseLR, type LR } from './adam.js'
import { DROPOUT_SEED_INPUT } from './ops.js'
import type { Req, Res, WireIR, WireAdamConfig, WireSGDConfig, WireOptimizerConfig, WireError } from './worker-protocol.js'
import { wireError } from './worker-protocol.js'

// ----------------------------------------------------------------------------
// Per-graph state
// ----------------------------------------------------------------------------

interface GraphSlot {
  runtime: CompiledRuntime
  paramNames: readonly string[]
  outputShape: number[]
  kernelCount: number
  captureShapes: Record<string, number[]>
  /** Optimizer state for this graph, if it's a training graph. The wrapped
   *  step uses these to populate per-step scalars (Adam's lrt + decayShrink,
   *  or SGD's lr). Exactly one branch is populated when training; `null`
   *  for forward-only graphs (compileForward siblings). */
  optimizer: OptimizerState | null
  /** Parent graph id for sibling forward graphs (those that share params via
   *  `sharedParams`). Set during `compileForward`; `null` for the training
   *  graph itself. Used by `handleDestroy` to cascade-destroy children when
   *  the parent goes away. */
  parentGraphId: number | null
  /** Per-step dropout seed state, when the graph contains any `dropout` op.
   *  `null` otherwise. The injected seed input is named DROPOUT_SEED_INPUT
   *  ('__dropoutSeed') and updated before every step()/run(). */
  dropout: DropoutState | null
}

type OptimizerState =
  | { kind: 'adam'; state: AdamState }
  | { kind: 'sgd'; state: SGDState }

interface AdamState {
  config: WireAdamConfig
  t: number
  lrtBuf: Float32Array
  decayShrinkBuf: Float32Array | null
}

interface SGDState {
  config: WireSGDConfig
  t: number
  lrBuf: Float32Array
}

interface DropoutState {
  /** Per-graph step counter; mixes into the kernel's PCG hash. Starts at 1;
   *  every step()/run() increments before dispatch so each call has a fresh
   *  seed and reset() back to 1 reproduces an earlier run from the same
   *  starting point. */
  counter: number
  /** Reused i32 single-element buffer for the per-step seed. */
  seedBuf: Int32Array
}

const graphs = new Map<number, GraphSlot>()

// Worker holds one device shared across all graphs (sibling forward graphs
// must share param GPUBuffers, which means sharing a device).
let device: GPUDevice | null = null

async function ensureDevice(): Promise<GPUDevice> {
  if (device) return device
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('tensorgrad worker: WebGPU not available in this environment')
  }
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('tensorgrad worker: no WebGPU adapter')
  device = await adapter.requestDevice()
  return device
}

// ----------------------------------------------------------------------------
// Request handlers
// ----------------------------------------------------------------------------

async function handleCreateRuntime(payload: {
  graphId: number
  ir: WireIR
  initialParams: Record<string, Float32Array>
  optimizer: WireOptimizerConfig | null
}): Promise<{ paramNames: string[]; outputShape: number[]; kernelCount: number; captureShapes: Record<string, number[]> }> {
  const dev = await ensureDevice()
  const { graph, plan, kernels } = payload.ir
  const outputTensorId = graph.outputs[0]!
  const outputBufferId = plan.tensorToBuffer.get(outputTensorId)!
  const opts: RuntimeOpts = { device: dev }
  const runtime = await createRuntime(plan, kernels, outputBufferId, opts)

  // Upload initial params.
  if (Object.keys(payload.initialParams).length > 0) {
    runtime.uploadParams(payload.initialParams)
  }

  // Capture shape metadata for return.
  const captureShapes: Record<string, number[]> = {}
  for (const [name, bufId] of plan.capturesByName) {
    captureShapes[name] = [...plan.buffers[bufId]!.shape]
  }

  const slot: GraphSlot = {
    runtime,
    paramNames: [...plan.paramsByName.keys()],
    outputShape: [...runtime.outputShape],
    kernelCount: kernels.filter(k => k.wgsl).length,
    captureShapes,
    optimizer: createOptimizerState(payload.optimizer),
    parentGraphId: null,
    dropout: graphUsesDropout(graph) ? { counter: 0, seedBuf: new Int32Array(1) } : null,
  }
  graphs.set(payload.graphId, slot)

  return {
    paramNames: [...slot.paramNames],
    outputShape: slot.outputShape,
    kernelCount: slot.kernelCount,
    captureShapes: slot.captureShapes,
  }
}

function createOptimizerState(cfg: WireOptimizerConfig | null): OptimizerState | null {
  if (!cfg) return null
  if (cfg.kind === 'adam') return { kind: 'adam', state: createAdamState(cfg.config) }
  return { kind: 'sgd', state: createSGDState(cfg.config) }
}

/** True iff the graph contains any `dropout` op. Drives whether the slot
 *  carries a `DropoutState` and whether step/run inject the seed input. */
function graphUsesDropout(graph: WireIR['graph']): boolean {
  for (const op of graph.ops) if (op.kind === 'dropout') return true
  return false
}

async function handleCompileForward(payload: {
  graphId: number
  parentGraphId: number
  ir: WireIR
}): Promise<{ paramNames: string[]; outputShape: number[]; kernelCount: number; captureShapes: Record<string, number[]> }> {
  const dev = await ensureDevice()
  const parent = graphs.get(payload.parentGraphId)
  if (!parent) throw new Error(`compileForward: parent graph ${payload.parentGraphId} not found`)

  const { graph, plan, kernels } = payload.ir
  const outputTensorId = graph.outputs[0]!
  const outputBufferId = plan.tensorToBuffer.get(outputTensorId)!
  const opts: RuntimeOpts = { device: dev, sharedParams: parent.runtime.params }
  const runtime = await createRuntime(plan, kernels, outputBufferId, opts)
  // No initial-param upload — sharedParams covers everything.

  const captureShapes: Record<string, number[]> = {}
  for (const [name, bufId] of plan.capturesByName) {
    captureShapes[name] = [...plan.buffers[bufId]!.shape]
  }

  const slot: GraphSlot = {
    runtime,
    paramNames: [...plan.paramsByName.keys()],
    outputShape: [...runtime.outputShape],
    kernelCount: kernels.filter(k => k.wgsl).length,
    captureShapes,
    optimizer: null,
    parentGraphId: payload.parentGraphId,
    dropout: graphUsesDropout(graph) ? { counter: 0, seedBuf: new Int32Array(1) } : null,
  }
  graphs.set(payload.graphId, slot)

  return {
    paramNames: [...slot.paramNames],
    outputShape: slot.outputShape,
    kernelCount: slot.kernelCount,
    captureShapes: slot.captureShapes,
  }
}

function createAdamState(cfg: WireAdamConfig): AdamState {
  return {
    config: cfg,
    t: 0,
    lrtBuf: new Float32Array(1),
    decayShrinkBuf: cfg.decayShrinkInputName ? new Float32Array(1) : null,
  }
}

function createSGDState(cfg: WireSGDConfig): SGDState {
  return { config: cfg, t: 0, lrBuf: new Float32Array(1) }
}

/** Inject the optimizer's per-step scalars into the inputs map. For Adam:
 *  the bias-corrected effective LR (`lrt`) and optionally `decayShrink`.
 *  For SGD: the per-step LR. Buffers are reused across steps. */
function injectOptimizerScalars(slot: GraphSlot, inputs: Record<string, Int32Array | Float32Array>): Record<string, Int32Array | Float32Array> {
  const o = slot.optimizer
  if (!o) return inputs
  if (o.kind === 'adam') {
    const a = o.state
    a.t++
    const lrNow = resolveLR(a.config.lr as LR, a.t)
    a.lrtBuf[0] = lrNow * Math.sqrt(1 - Math.pow(a.config.b2, a.t)) / (1 - Math.pow(a.config.b1, a.t))
    const merged: Record<string, Int32Array | Float32Array> = { ...inputs, [a.config.lrtInputName]: a.lrtBuf }
    if (a.decayShrinkBuf && a.config.decayShrinkInputName) {
      a.decayShrinkBuf[0] = 1 - lrNow * a.config.weightDecay
      merged[a.config.decayShrinkInputName] = a.decayShrinkBuf
    }
    return merged
  }
  // SGD
  const s = o.state
  s.t++
  s.lrBuf[0] = resolveLR(s.config.lr as LR, s.t)
  return { ...inputs, [s.config.lrInputName]: s.lrBuf }
}

/** Inject the per-step dropout seed into `inputs` when this graph uses
 *  dropout. Counter advances before every step()/run() so each call
 *  produces a different mask, while same-counter dispatches reproduce. */
function injectDropoutSeed(slot: GraphSlot, inputs: Record<string, Int32Array | Float32Array>): Record<string, Int32Array | Float32Array> {
  const d = slot.dropout
  if (!d) return inputs
  d.counter++
  d.seedBuf[0] = d.counter | 0
  return { ...inputs, [DROPOUT_SEED_INPUT]: d.seedBuf }
}

async function handleStep(payload: {
  graphId: number
  inputs: Record<string, Int32Array | Float32Array>
  withCaptures: boolean
}): Promise<{ loss: number; captures: Record<string, Float32Array> | null }> {
  const slot = mustGet(payload.graphId)
  const merged = injectDropoutSeed(slot, injectOptimizerScalars(slot, payload.inputs))
  try {
    if (payload.withCaptures) {
      const r = await slot.runtime.step(merged, { withCaptures: true })
      return { loss: r.loss, captures: capturesToRecord(r.captures, slot.captureShapes) }
    }
    const loss = await slot.runtime.step(merged)
    return { loss, captures: null }
  } catch (e) {
    // If the graph was destroyed mid-flight (e.g. replaceModel ran while we
    // were awaiting mapAsync), surface a clean AbortError instead of the raw
    // WebGPU "buffer is destroyed" or similar — callers can branch on it
    // (or pass { onAbort: 'value' } to get a discriminated result).
    if (!graphs.has(payload.graphId)) throw abortErr('step aborted: graph destroyed mid-flight')
    throw e
  }
}

async function handleRun(payload: {
  graphId: number
  inputs: Record<string, Int32Array | Float32Array>
  withCaptures: boolean
}): Promise<{ output: Float32Array; captures: Record<string, Float32Array> | null }> {
  const slot = mustGet(payload.graphId)
  const merged = injectDropoutSeed(slot, payload.inputs)
  try {
    if (payload.withCaptures) {
      const r = await slot.runtime.run(merged, { withCaptures: true })
      return { output: r.output, captures: capturesToRecord(r.captures, slot.captureShapes) }
    }
    const output = await slot.runtime.run(merged)
    return { output, captures: null }
  } catch (e) {
    if (!graphs.has(payload.graphId)) throw abortErr('run aborted: graph destroyed mid-flight')
    throw e
  }
}

function abortErr(msg: string): Error {
  const e = new Error(msg)
  e.name = 'AbortError'
  return e
}

/** Captures (a class instance with a private Map) → a plain Record so the
 *  worker can transfer Float32Arrays back without serializing the class. */
function capturesToRecord(
  captures: { get(name: string): Float32Array; has(name: string): boolean; names(): string[] },
  // captureShapes available but not used directly — capture names from
  // shapes in case captures.names() is filtered (it isn't, but be safe).
  shapes: Record<string, number[]>,
): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {}
  for (const name of Object.keys(shapes)) {
    if (captures.has(name)) out[name] = captures.get(name)
  }
  return out
}

function handleUploadParams(payload: {
  graphId: number
  params: Record<string, Float32Array>
  partial: boolean
}): void {
  const slot = mustGet(payload.graphId)
  slot.runtime.uploadParams(payload.params, { partial: payload.partial })
}

async function handleDownloadParams(payload: { graphId: number }): Promise<{ params: Record<string, Float32Array> }> {
  const slot = mustGet(payload.graphId)
  return { params: await slot.runtime.downloadParams() }
}

async function handleDownloadParamGrads(payload: { graphId: number }): Promise<{ params: Record<string, Float32Array> }> {
  const slot = mustGet(payload.graphId)
  return { params: await slot.runtime.downloadParamGrads() }
}

function handleResetOptimizer(payload: { graphId: number }): void {
  const slot = mustGet(payload.graphId)
  slot.runtime.resetOptimizerState()
  if (slot.optimizer) {
    if (slot.optimizer.kind === 'adam') slot.optimizer.state.t = 0
    else slot.optimizer.state.t = 0
  }
}

function handleSetOptimizerConfig(payload: {
  graphId: number
  update: { lr?: LR }
}): void {
  const slot = mustGet(payload.graphId)
  if (!slot.optimizer) {
    throw new Error(`setOptimizerConfig: graph ${payload.graphId} has no optimizer (compileForward graphs don't take optimizer state)`)
  }
  if (payload.update.lr === undefined) return
  // The next step will increment t from its current value, so the schedule
  // takes effect at t+1 — that's the step we rebase against.
  const state = slot.optimizer.state
  const nextStep = state.t + 1
  const newLR = rebaseLR(payload.update.lr, nextStep)
  if (slot.optimizer.kind === 'adam') {
    state.config = { ...(state.config as WireAdamConfig), lr: newLR }
  } else {
    state.config = { ...(state.config as WireSGDConfig), lr: newLR }
  }
}

function handleDestroy(payload: { graphId: number }): void {
  // Cascade: any graph that listed this one as parent (a sibling forward
  // sharing params) loses its backing buffers when we destroy the parent.
  // Destroy children first so their bind groups release before parent buffers.
  for (const [id, slot] of graphs) {
    if (slot.parentGraphId === payload.graphId) {
      slot.runtime.destroy()
      graphs.delete(id)
    }
  }
  const slot = graphs.get(payload.graphId)
  if (!slot) return
  slot.runtime.destroy()
  graphs.delete(payload.graphId)
}

function mustGet(graphId: number): GraphSlot {
  const slot = graphs.get(graphId)
  if (!slot) throw new Error(`tensorgrad worker: graph ${graphId} not found`)
  return slot
}

// ----------------------------------------------------------------------------
// Message dispatch
// ----------------------------------------------------------------------------

self.onmessage = async (ev: MessageEvent<Req>) => {
  const req = ev.data
  try {
    let result: unknown
    let transferList: ArrayBuffer[] = []
    switch (req.kind) {
      case 'createRuntime':     result = await handleCreateRuntime(req.payload); break
      case 'compileForward':    result = await handleCompileForward(req.payload); break
      case 'step':              result = await handleStep(req.payload); transferList = collectTransfers((result as any).captures); break
      case 'run':               { const r = await handleRun(req.payload); result = r; transferList = [r.output.buffer as ArrayBuffer, ...collectTransfers(r.captures)]; break }
      case 'uploadParams':      handleUploadParams(req.payload); result = null; break
      case 'downloadParams':    { const r = await handleDownloadParams(req.payload); result = r; transferList = collectTransfers(r.params); break }
      case 'downloadParamGrads':{ const r = await handleDownloadParamGrads(req.payload); result = r; transferList = collectTransfers(r.params); break }
      case 'resetOptimizer':    handleResetOptimizer(req.payload); result = null; break
      case 'setOptimizerConfig':handleSetOptimizerConfig(req.payload); result = null; break
      case 'destroy':           handleDestroy(req.payload); result = null; break
      default: throw new Error(`unknown request kind: ${(req as { kind: string }).kind}`)
    }
    const reply: Res = { id: req.id, ok: true, result }
    self.postMessage(reply, { transfer: transferList })
  } catch (e) {
    const error: WireError = wireError(e)
    const reply: Res = { id: req.id, ok: false, error }
    self.postMessage(reply)
  }
}

function collectTransfers(rec: Record<string, Float32Array> | null | undefined): ArrayBuffer[] {
  if (!rec) return []
  const out: ArrayBuffer[] = []
  for (const v of Object.values(rec)) out.push(v.buffer as ArrayBuffer)
  return out
}
