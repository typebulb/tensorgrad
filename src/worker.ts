// Worker entry point. Runs in a Web Worker context (no `window`/`document`);
// see specs/WorkerArchitecture.md for the rationale.

import { createRuntime, type CompiledRuntime, type RuntimeOpts } from './runtime.js'
import { resolveLR, rebaseLR, type LR } from './adam.js'
import { PRNG_SEED_INPUT } from './ops.js'
import type { Req, Res, WireIR, WireAdamConfig, WireSGDConfig, WireOptimizerConfig, WireError } from './worker-protocol.js'
import { wireError } from './worker-protocol.js'

interface GraphSlot {
  runtime: CompiledRuntime
  paramNames: readonly string[]
  outputShape: number[]
  kernelCount: number
  captureShapes: Record<string, number[]>
  /** Null for forward-only graphs (compileForward siblings). */
  optimizer: OptimizerState | null
  /** Set for sibling forward graphs (those sharing params via `sharedParams`);
   *  null for the training graph itself. Used to cascade-destroy children. */
  parentGraphId: number | null
  /** Per-step PRNG state, set when the graph contains any stochastic op
   *  (`dropout` or `randn`). Null otherwise — saves the per-step inject. */
  prng: PrngState | null
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

interface PrngState {
  /** Per-graph step counter that mixes into the kernel's PCG hash. Bumped
   *  before each dispatch so successive calls draw different masks /
   *  noise while staying reproducible from a known starting counter. */
  counter: number
  seedBuf: Int32Array
}

const graphs = new Map<number, GraphSlot>()

// One device shared across all graphs (siblings must share param buffers).
let device: GPUDevice | null = null

async function ensureDevice(): Promise<GPUDevice> {
  if (device) return device
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('tensorgrad worker: WebGPU not available in this environment')
  }
  // requestAdapter() resolves with null (not throws) when the GPU process
  // is transiently unhealthy — recent crash, power-state transition, sandboxed
  // iframe+blob worker quirk, etc. Retry with backoff before giving up.
  let adapter: GPUAdapter | null = null
  const delays = [0, 100, 400]
  for (const ms of delays) {
    if (ms > 0) await new Promise(r => setTimeout(r, ms))
    adapter = await navigator.gpu.requestAdapter()
    if (adapter) break
  }
  if (!adapter) throw new Error('tensorgrad worker: no WebGPU adapter')
  device = await adapter.requestDevice()
  return device
}

// ---- Request handlers ---------------------------------------------------

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
    prng: graphUsesPrng(graph) ? { counter: 0, seedBuf: new Int32Array(1) } : null,
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

function graphUsesPrng(graph: WireIR['graph']): boolean {
  for (const op of graph.ops) {
    if (op.kind === 'dropout' || op.kind === 'randn' || op.kind === 'categorical_last') return true
  }
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
    prng: graphUsesPrng(graph) ? { counter: 0, seedBuf: new Int32Array(1) } : null,
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

/** Inject the optimizer's per-step scalars: Adam's `lrt` (bias-corrected
 *  effective LR) and optional `decayShrink`, or SGD's `lr`. */
function injectOptimizerScalars(slot: GraphSlot, inputs: Record<string, Int32Array | Float32Array>): Record<string, Int32Array | Float32Array> {
  const o = slot.optimizer
  if (!o) return inputs
  if (o.kind === 'adam') {
    const a = o.state
    a.t++
    const lrNow = resolveLR(a.config.lr, a.t)
    a.lrtBuf[0] = lrNow * Math.sqrt(1 - Math.pow(a.config.beta2, a.t)) / (1 - Math.pow(a.config.beta1, a.t))
    const merged: Record<string, Int32Array | Float32Array> = { ...inputs, [a.config.lrtInputName]: a.lrtBuf }
    if (a.decayShrinkBuf && a.config.decayShrinkInputName) {
      a.decayShrinkBuf[0] = 1 - lrNow * a.config.weightDecay
      merged[a.config.decayShrinkInputName] = a.decayShrinkBuf
    }
    return merged
  }
  const s = o.state
  s.t++
  s.lrBuf[0] = resolveLR(s.config.lr, s.t)
  return { ...inputs, [s.config.lrInputName]: s.lrBuf }
}

function injectPrngSeed(slot: GraphSlot, inputs: Record<string, Int32Array | Float32Array>): Record<string, Int32Array | Float32Array> {
  const p = slot.prng
  if (!p) return inputs
  p.counter++
  p.seedBuf[0] = p.counter | 0
  return { ...inputs, [PRNG_SEED_INPUT]: p.seedBuf }
}

async function handleStep(payload: {
  graphId: number
  inputs: Record<string, Int32Array | Float32Array>
}): Promise<{ loss: number; captures: Record<string, Float32Array> | null }> {
  const slot = mustGet(payload.graphId)
  const merged = injectPrngSeed(slot, injectOptimizerScalars(slot, payload.inputs))
  const hasCaptures = Object.keys(slot.captureShapes).length > 0
  try {
    const r = await slot.runtime.step(merged)
    return {
      loss: r.loss,
      captures: hasCaptures ? capturesToRecord(r.captures, slot.captureShapes) : null,
    }
  } catch (e) {
    // If the graph was destroyed mid-flight (replaceModel while awaiting
    // mapAsync), translate WebGPU's "buffer is destroyed" into a clean
    // AbortError so callers can branch on it.
    if (!graphs.has(payload.graphId)) throw abortErr('step aborted: graph destroyed mid-flight')
    throw e
  }
}

async function handleRun(payload: {
  graphId: number
  inputs: Record<string, Int32Array | Float32Array>
}): Promise<{ output: Float32Array; captures: Record<string, Float32Array> | null }> {
  const slot = mustGet(payload.graphId)
  const merged = injectPrngSeed(slot, payload.inputs)
  const hasCaptures = Object.keys(slot.captureShapes).length > 0
  try {
    const r = await slot.runtime.run(merged)
    return {
      output: r.output,
      captures: hasCaptures ? capturesToRecord(r.captures, slot.captureShapes) : null,
    }
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

/** Captures class instance → plain Record so we can transfer Float32Arrays
 *  back without serializing the class. */
function capturesToRecord(
  captures: { get(name: string): Float32Array; has(name: string): boolean; names(): string[] },
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
}): void {
  const slot = mustGet(payload.graphId)
  slot.runtime.uploadParams(payload.params)
}

async function handleDownloadParams(payload: { graphId: number }): Promise<{ params: Record<string, Float32Array> }> {
  const slot = mustGet(payload.graphId)
  return { params: await slot.runtime.downloadParams() }
}

function handleResetOptimizer(payload: { graphId: number }): void {
  const slot = mustGet(payload.graphId)
  slot.runtime.resetOptimizerState()
  if (slot.optimizer) {
    if (slot.optimizer.kind === 'adam') slot.optimizer.state.t = 0
    else slot.optimizer.state.t = 0
  }
}

function handleSetLR(payload: { graphId: number; lr: LR }): void {
  const slot = mustGet(payload.graphId)
  if (!slot.optimizer) {
    throw new Error(`setLR: graph ${payload.graphId} has no optimizer (compileForward graphs don't take optimizer state)`)
  }
  // injectOptimizerScalars increments t before each step, so the new schedule
  // takes effect at t+1 — that's the step we rebase against.
  const state = slot.optimizer.state
  const nextStep = state.t + 1
  const newLR = rebaseLR(payload.lr, nextStep)
  if (slot.optimizer.kind === 'adam') {
    state.config = { ...(state.config as WireAdamConfig), lr: newLR }
  } else {
    state.config = { ...(state.config as WireSGDConfig), lr: newLR }
  }
}

function handleDestroy(payload: { graphId: number }): void {
  // Cascade: destroy children first so their bind groups release before the
  // parent's param buffers (which they were sharing) go away.
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

// ---- Message dispatch ---------------------------------------------------

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
