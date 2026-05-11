// Worker entry point. Holds the GPUDevice + CompiledRuntime for one or more
// graphs and proxies main-thread requests via postMessage. See
// specs/WorkerArchitecture.md for the rationale.
//
// Keep this file dependency-free of anything DOM-y: it bundles into a Blob
// URL and runs in a Web Worker context where `window`/`document` don't
// exist. WebGPU IS available in workers (Chrome 113+, Safari 17.4+).

import { createRuntime, type CompiledRuntime, type RuntimeOpts } from './runtime.js'
import { resolveLR, type LRSchedule } from './adam.js'
import type { Req, Res, WireIR, WireAdamConfig, WireError } from './worker-protocol.js'
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
  /** Adam state for this graph, if it's a training graph. The wrapped step
   *  uses these to populate the per-step lrt and decayShrink scalars. */
  adam: AdamState | null
}

interface AdamState {
  config: WireAdamConfig
  t: number
  lrtBuf: Float32Array
  decayShrinkBuf: Float32Array | null
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
  adam: WireAdamConfig | null
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
    adam: payload.adam ? createAdamState(payload.adam) : null,
  }
  graphs.set(payload.graphId, slot)

  return {
    paramNames: [...slot.paramNames],
    outputShape: slot.outputShape,
    kernelCount: slot.kernelCount,
    captureShapes: slot.captureShapes,
  }
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
    adam: null,
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

/** Inject Adam's per-step lrt + decayShrink scalars into the inputs map.
 *  Called before every step on a training graph. The buffers are reused
 *  across steps to avoid allocation. */
function injectAdamScalars(slot: GraphSlot, inputs: Record<string, Int32Array | Float32Array>): Record<string, Int32Array | Float32Array> {
  const a = slot.adam
  if (!a) return inputs
  a.t++
  const lrNow = resolveLR(a.config.lr as LRSchedule, a.t)
  a.lrtBuf[0] = lrNow * Math.sqrt(1 - Math.pow(a.config.b2, a.t)) / (1 - Math.pow(a.config.b1, a.t))
  const merged: Record<string, Int32Array | Float32Array> = { ...inputs, [a.config.lrtInputName]: a.lrtBuf }
  if (a.decayShrinkBuf && a.config.decayShrinkInputName) {
    a.decayShrinkBuf[0] = 1 - lrNow * a.config.weightDecay
    merged[a.config.decayShrinkInputName] = a.decayShrinkBuf
  }
  return merged
}

async function handleStep(payload: {
  graphId: number
  inputs: Record<string, Int32Array | Float32Array>
  withCaptures: boolean
}): Promise<{ loss: number; captures: Record<string, Float32Array> | null }> {
  const slot = mustGet(payload.graphId)
  const merged = injectAdamScalars(slot, payload.inputs)
  if (payload.withCaptures) {
    const r = await slot.runtime.step(merged, { withCaptures: true })
    return { loss: r.loss, captures: capturesToRecord(r.captures, slot.captureShapes) }
  }
  const loss = await slot.runtime.step(merged)
  return { loss, captures: null }
}

async function handleRun(payload: {
  graphId: number
  inputs: Record<string, Int32Array | Float32Array>
  withCaptures: boolean
}): Promise<{ output: Float32Array; captures: Record<string, Float32Array> | null }> {
  const slot = mustGet(payload.graphId)
  if (payload.withCaptures) {
    const r = await slot.runtime.run(payload.inputs, { withCaptures: true })
    return { output: r.output, captures: capturesToRecord(r.captures, slot.captureShapes) }
  }
  const output = await slot.runtime.run(payload.inputs)
  return { output, captures: null }
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
  if (slot.adam) slot.adam.t = 0
}

function handleSetLR(payload: { graphId: number; lr: LRSchedule }): void {
  const slot = mustGet(payload.graphId)
  if (!slot.adam) {
    throw new Error(`setLR: graph ${payload.graphId} has no Adam optimizer (compileForward graphs don't take an LR)`)
  }
  slot.adam.config = { ...slot.adam.config, lr: payload.lr }
}

function handleDestroy(payload: { graphId: number }): void {
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
      case 'setLR':             handleSetLR(req.payload); result = null; break
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
