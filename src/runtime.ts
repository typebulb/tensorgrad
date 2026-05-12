// WebGPU runtime. Browser-only — needs `navigator.gpu` at runtime.

import type { BufferPlan } from './buffers.js'
import type { KernelSpec } from './codegen.js'

// lib.dom declares the WebGPU types but not this runtime constant.
declare const GPUMapMode: { readonly READ: number; readonly WRITE: number }

export interface UploadParamsOptions {
  /** Skip the "missing param" check, allowing the caller to update only some
   *  params and leave the rest at their current GPU values. Extra (unknown)
   *  keys are still rejected — that's always a typo. Default: false. */
  partial?: boolean
}

/**
 * Activation readbacks for one `step()`/`run()` call. Keyed by the names
 * passed to `capture(name, t)` during the trace. `get(name)` throws if the
 * name wasn't registered during the trace. `shape(name)` returns the
 * static-after-compile shape — works whether or not captures were populated.
 */
export class Captures {
  constructor(
    private readonly shapes: Record<string, readonly number[]>,
    private readonly data: Map<string, Float32Array>,
  ) {}
  get(name: string): Float32Array {
    const d = this.data.get(name)
    if (!d) {
      const known = [...this.data.keys()].sort().join(', ') || '(none registered)'
      throw new Error(`Captures.get: '${name}' not present. Known: ${known}`)
    }
    return d
  }
  shape(name: string): readonly number[] {
    const s = this.shapes[name]
    if (!s) {
      const known = Object.keys(this.shapes).sort().join(', ') || '(none registered)'
      throw new Error(`Captures.shape: '${name}' not registered. Known: ${known}`)
    }
    return s
  }
  has(name: string): boolean { return this.data.has(name) }
  names(): string[] { return [...this.data.keys()].sort() }
}

/** Result of `run(inputs)`: the output tensor as a flat `Float32Array`
 *  plus a `Captures` instance. When the traced graph has no `capture(...)`
 *  sites, `captures` is empty (calling `.get` on any name throws). */
export interface RunCompletion {
  output: Float32Array
  captures: Captures
}

/** Result of `step(inputs)`: the scalar loss plus a `Captures` instance.
 *  Empty captures when the graph has no `capture(...)` sites. */
export interface StepCompletion {
  loss: number
  captures: Captures
}

/** Common surface for both training and forward-only compiled runtimes. */
export interface CompiledBase {
  /** The GPUDevice this runtime is bound to. Pass to sibling compiles to
   *  share the device, or use directly for other GPU work. */
  device: GPUDevice
  /** Param name -> the underlying GPUBuffer. Pass to a sibling compile via
   *  `sharedParams` to share without copies. */
  params: Map<string, GPUBuffer>
  /** Shape of the graph's output (loss scalar `[]` for training; the user's
   *  returned tensor for forward-only compiles). */
  outputShape: number[]
  /** Upload parameter Float32Arrays to their GPU buffers. By default, requires
   *  *all* params to be present; throws on any unknown or missing key. Pass
   *  `{ partial: true }` to skip the missing-key check. */
  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): void
  /** Read all parameters back as Float32Arrays — used for UI panels. */
  downloadParams(): Promise<Record<string, Float32Array>>
  /** Free GPU resources. */
  destroy(): void
}

/** Run a dispatch and read back the full output tensor plus any registered
 *  captures. */
export type RunFn = (
  inputs: Record<string, Int32Array | Float32Array>,
) => Promise<RunCompletion>

export interface CompiledRuntime extends CompiledBase {
  /** Read all parameter gradients back. Mostly for verification / debugging. */
  downloadParamGrads(): Promise<Record<string, Float32Array>>
  /**
   * One full forward+backward step.
   *   1. Uploads `inputs` (tokens, targets, masks) to input buffers.
   *   2. Dispatches every kernel in order.
   *   3. Reads back the loss scalar plus any captures the graph registered.
   * Always awaits the loss `mapAsync`. For fire-and-forget training (e.g.
   * mobile UI responsiveness), use `queueStep` + `readLoss` instead.
   */
  step(inputs: Record<string, Int32Array | Float32Array>): Promise<StepCompletion>
  /** Submit a training step without awaiting the loss readback. Each
   *  `mapAsync` round-trip is ~1 ms on desktop but 10–30 ms on Android
   *  Chrome; for mobile responsiveness, queueStep + periodic `readLoss`
   *  for UI display beats awaiting every step. */
  queueStep(inputs: Record<string, Int32Array | Float32Array>): Promise<void>
  /** Same dispatch as step() but returns the full output array; for
   *  training graphs the output is a scalar loss, so step() is usually
   *  more convenient. Provided for parity with `compileForward`. */
  run: RunFn
  /** Read the latest loss value from the GPU. Pair with `queueStep` for
   *  fire-and-forget training: queue most iterations, call `readLoss()`
   *  every Nth for the UI. */
  readLoss(): Promise<number>
  /** Re-zero all optimizer state buffers (Adam's m/v) in place. Pair with
   *  `uploadParams` (or call the proxy's `reset()` which does both) for a
   *  full training reset without recompile. */
  resetOptimizerState(): void
}

/** Forward-only compiled runtime — produced by `compileForward`. No optimizer,
 *  no backward. Returns the output tensor (not just a scalar) per `run()` call. */
export interface CompiledForward extends CompiledBase {
  run: RunFn
}

export interface RuntimeOpts {
  /** Pre-acquired GPUDevice. If omitted, runtime requests its own. */
  device?: GPUDevice
  /** External param buffers to bind in place of allocating fresh ones, keyed
   *  by param name. Used to share params between a training compile and a
   *  sibling forward-only compile (e.g., a B=1 inference graph). When a name
   *  is in this map, the runtime reuses the provided GPUBuffer; otherwise it
   *  allocates as usual. */
  sharedParams?: Map<string, GPUBuffer>
}

// Spec-inlined so this module is importable in Node for codegen-only use.
// `GPUBufferUsage` is a browser global; referencing it at module scope would
// crash on import in Node.
const STORAGE_RW = 0x80 /*STORAGE*/ | 0x8 /*COPY_DST*/ | 0x4 /*COPY_SRC*/
const READBACK = 0x1 /*MAP_READ*/ | 0x8 /*COPY_DST*/

export async function createRuntime(
  plan: BufferPlan,
  kernels: KernelSpec[],
  lossBufferId: number,
  opts: RuntimeOpts = {},
): Promise<CompiledRuntime> {
  const device = opts.device ?? await acquireDevice()
  const queue = device.queue

  // Allocate one GPUBuffer per BufferSpec. State buffers are filled with
  // their initValue here. Param buffers in `sharedParams` are reused as-is
  // (the sibling that owns them is responsible for upload + lifetime);
  // `ownedBufferIds` tracks which buffers `.destroy()` must release.
  const buffers = new Map<number, GPUBuffer>()
  const ownedBufferIds = new Set<number>()
  const sharedParams = opts.sharedParams
  for (const spec of plan.buffers) {
    const shared = spec.kind === 'param' ? sharedParams?.get(spec.name!) : undefined
    if (shared) {
      if (shared.size !== spec.byteSize) {
        throw new Error(
          `sharedParams: size mismatch for '${spec.name}' — supplied ${shared.size} bytes, ` +
          `compiled graph expects ${spec.byteSize}.`,
        )
      }
      buffers.set(spec.id, shared)
      continue
    }
    const buf = device.createBuffer({
      size: spec.byteSize,
      usage: STORAGE_RW,
      label: spec.name ?? `t${spec.id}-${spec.kind}`,
    })
    buffers.set(spec.id, buf)
    ownedBufferIds.add(spec.id)
    if (spec.kind === 'state') fillStateBuffer(spec, buf)
  }

  // Per-kernel pipelines, cached by WGSL source. Error scope around each
  // creation surfaces the real shader compile error instead of the cryptic
  // "previous error" you'd otherwise get at dispatch time.
  const moduleCache = new Map<string, GPUShaderModule>()
  const pipelines: (GPUComputePipeline | null)[] = []
  type ErrorProbe = Promise<{ k: KernelSpec; module: GPUShaderModule; err: GPUError } | null>
  const probes: ErrorProbe[] = []
  for (const k of kernels) {
    if (!k.wgsl) { pipelines.push(null); continue }
    let module = moduleCache.get(k.wgsl)
    if (!module) {
      module = device.createShaderModule({ code: k.wgsl, label: k.opKind })
      moduleCache.set(k.wgsl, module)
    }
    device.pushErrorScope('validation')
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
      label: k.opKind,
    })
    pipelines.push(pipeline)
    probes.push(device.popErrorScope().then(err => err ? { k, module: module!, err } : null))
  }
  const probeResults = await Promise.all(probes)
  const failures = probeResults.filter((p): p is { k: KernelSpec; module: GPUShaderModule; err: GPUError } => p != null)
  if (failures.length > 0) {
    const reports: string[] = []
    for (const { k, module, err } of failures) {
      const info = await module.getCompilationInfo()
      const messages = info.messages
        .map(m => `  L${m.lineNum}:${m.linePos} [${m.type}] ${m.message}`)
        .join('\n')
      reports.push(
        `[shader compile error] ${k.opKind} (op #${k.opIndex}): ${err.message}\n` +
        (messages || '  (no compilation messages)') +
        `\n--- WGSL ---\n${k.wgsl}\n-----------`,
      )
    }
    // eslint-disable-next-line no-console
    console.error(reports.join('\n\n'))
    throw new Error(`tensorgrad: ${failures.length} shader(s) failed to compile (see console).`)
  }

  // Static bind groups — buffer ids don't change per step.
  const bindGroups: (GPUBindGroup | null)[] = kernels.map((k, i) => {
    const pipeline = pipelines[i]
    if (!pipeline) return null
    return device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: k.bindings.map((bufId, idx) => ({
        binding: idx,
        resource: { buffer: buffers.get(bufId)! },
      })),
    })
  })

  // step() reads buf[0] for the scalar loss; run() returns the full array.
  const outputSpec = plan.buffers[lossBufferId]!
  const outputReadback = device.createBuffer({ size: outputSpec.byteSize, usage: READBACK })

  // One concatenated staging buffer for ALL captures. mapAsync round-trips
  // on Android Chrome cost 10–30 ms each (vs ~1 ms on desktop) — N captures
  // means N stalls without batching. Allocated lazily on first
  // step({ withCaptures: true }) call.
  type CaptureLayout = {
    buffer: GPUBuffer
    slices: { name: string; bufId: number; offset: number; byteSize: number }[]
  }
  let captureStaging: CaptureLayout | null = null
  function ensureCaptureStaging(): CaptureLayout {
    if (captureStaging) return captureStaging
    let totalBytes = 0
    const slices: CaptureLayout['slices'] = []
    for (const [name, bufId] of plan.capturesByName) {
      const spec = plan.buffers[bufId]!
      // copyBufferToBuffer offsets must be 4-aligned. byteSizes are always
      // shape-product × 4 (every dtype is 4 bytes), so offsets stay aligned.
      slices.push({ name, bufId, offset: totalBytes, byteSize: spec.byteSize })
      totalBytes += spec.byteSize
    }
    const buffer = device.createBuffer({ size: totalBytes, usage: READBACK, label: 'captures-staging' })
    captureStaging = { buffer, slices }
    return captureStaging
  }

  // Shared core for step() and run(): upload inputs, dispatch every kernel
  // in order, queue writebacks, copy output (and captures, if requested) into
  // staging, submit, read back.
  //
  // Concurrent calls auto-serialize. Two step/run calls would otherwise both
  // mapAsync the shared output staging buffer and trip "Buffer already has
  // an outstanding map pending." Chaining via `pending` makes independent
  // async paths (e.g. training loop + aux `refreshPrediction`) run in turn.
  let pending: Promise<unknown> = Promise.resolve()
  type DispatchOpts = { wantCaptures: boolean; readback: boolean }
  type DispatchResult = { output: Float32Array; captures: Map<string, Float32Array> } | null
  async function dispatch(
    inputs: Record<string, Int32Array | Float32Array>,
    opts: DispatchOpts,
  ): Promise<DispatchResult> {
    const turn = pending.catch(() => {}).then(() => dispatchUnsynchronized(inputs, opts))
    pending = turn
    return turn
  }
  async function dispatchUnsynchronized(
    inputs: Record<string, Int32Array | Float32Array>,
    opts: DispatchOpts,
  ): Promise<DispatchResult> {
    // Captures are populated whenever the graph has any capture() sites.
    // If the user asked for captures but the trace registered none, that's
    // a no-op (the Captures bag will be empty) — not an error.
    const wantCaptures = opts.wantCaptures && plan.capturesByName.size > 0
    for (const [name, bufId] of plan.inputsByName) {
      const data = inputs[name]
      if (!data) throw new Error(`tensorgrad: missing input '${name}'`)
      const expectedBytes = plan.buffers[bufId]!.byteSize
      if (data.byteLength !== expectedBytes) {
        throw new Error(`tensorgrad: input '${name}' has ${data.byteLength} bytes, expected ${expectedBytes}`)
      }
      // Cast to BufferSource: typed arrays are accepted by writeBuffer at runtime
      // but TS may infer ArrayBufferLike (vs ArrayBuffer) under strict configs.
      queue.writeBuffer(buffers.get(bufId)!, 0, data as unknown as BufferSource)
    }

    const encoder = device.createCommandEncoder({ label: 'tensorgrad-step' })
    for (let i = 0; i < kernels.length; i++) {
      const k = kernels[i]!
      if (!k.wgsl || k.threads === 0) continue
      const pipeline = pipelines[i]!
      const bindGroup = bindGroups[i]!
      const pass = encoder.beginComputePass({ label: k.opKind })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      // WebGPU caps each dispatch dim at 65535 workgroups; split into 2D when
      // the X axis overflows. Kernels compute the global index as
      // `gid.x + gid.y * (65535 * workgroup_size)` to match this stride.
      const wgCount = Math.max(1, Math.ceil(k.threads / k.workgroupSize))
      const MAX_X = 65535
      const wgX = Math.min(wgCount, MAX_X)
      const wgY = Math.ceil(wgCount / MAX_X)
      pass.dispatchWorkgroups(wgX, wgY, 1)
      pass.end()
    }
    // Writebacks for Adam state + updated params. Empty for forward-only.
    for (const wb of plan.writebacks) {
      encoder.copyBufferToBuffer(buffers.get(wb.source)!, 0, buffers.get(wb.dest)!, 0, wb.bytes)
    }
    encoder.copyBufferToBuffer(buffers.get(lossBufferId)!, 0, outputReadback, 0, outputSpec.byteSize)
    let layout: CaptureLayout | null = null
    if (wantCaptures) {
      layout = ensureCaptureStaging()
      for (const s of layout.slices) {
        encoder.copyBufferToBuffer(buffers.get(s.bufId)!, 0, layout.buffer, s.offset, s.byteSize)
      }
    }
    queue.submit([encoder.finish()])

    // Fire-and-forget: the encoder already copied loss → outputReadback,
    // but we skip mapAsync. The caller can read it later via readLoss().
    if (!opts.readback) return null

    await outputReadback.mapAsync(GPUMapMode.READ)
    const output = new Float32Array(outputReadback.getMappedRange().slice(0))
    outputReadback.unmap()

    const captures = new Map<string, Float32Array>()
    if (layout) {
      await layout.buffer.mapAsync(GPUMapMode.READ)
      const range = layout.buffer.getMappedRange()
      for (const s of layout.slices) {
        // .slice() copies before unmap — the ArrayBuffer detaches on unmap.
        captures.set(s.name, new Float32Array(range, s.offset, s.byteSize / 4).slice())
      }
      layout.buffer.unmap()
    }
    return { output, captures }
  }

  async function step(
    inputs: Record<string, Int32Array | Float32Array>,
  ): Promise<StepCompletion> {
    const r = (await dispatch(inputs, { wantCaptures: true, readback: true }))!
    return { loss: r.output[0]!, captures: new Captures(captureShapes, r.captures) }
  }

  async function queueStep(
    inputs: Record<string, Int32Array | Float32Array>,
  ): Promise<void> {
    await dispatch(inputs, { wantCaptures: false, readback: false })
  }

  // Goes through the same `pending` serialization as step()/run() so two
  // readLoss() calls don't both try to mapAsync the same buffer.
  async function readLoss(): Promise<number> {
    const turn = pending.catch(() => {}).then(async () => {
      await outputReadback.mapAsync(GPUMapMode.READ)
      const v = new Float32Array(outputReadback.getMappedRange())[0]!
      outputReadback.unmap()
      return v
    })
    pending = turn
    return turn
  }

  async function run(
    inputs: Record<string, Int32Array | Float32Array>,
  ): Promise<RunCompletion> {
    const r = (await dispatch(inputs, { wantCaptures: true, readback: true }))!
    return { output: r.output, captures: new Captures(captureShapes, r.captures) }
  }

  // ---- uploadParams ---------------------------------------------------------
  function uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions) {
    const partial = opts?.partial ?? false
    for (const name of Object.keys(params)) {
      if (!plan.paramsByName.has(name)) {
        throw new Error(
          `uploadParams: unknown param '${name}'. ` +
          `Known: ${[...plan.paramsByName.keys()].sort().join(', ')}`,
        )
      }
    }
    if (!partial) {
      for (const name of plan.paramsByName.keys()) {
        if (!(name in params)) {
          throw new Error(
            `uploadParams: missing param '${name}'. ` +
            `Pass { partial: true } if you mean to update only some params.`,
          )
        }
      }
    }
    for (const [name, bufId] of plan.paramsByName) {
      const data = params[name]
      if (!data) continue
      const expected = plan.buffers[bufId]!.byteSize / 4
      if (data.length !== expected) {
        throw new Error(`uploadParams: '${name}' has ${data.length} elements, expected ${expected}`)
      }
      queue.writeBuffer(buffers.get(bufId)!, 0, data as unknown as BufferSource)
    }
  }

  async function downloadFromMap(map: Map<string, number>): Promise<Record<string, Float32Array>> {
    const stagings: { name: string; buf: GPUBuffer; bytes: number }[] = []
    const encoder = device.createCommandEncoder({ label: 'tensorgrad-download' })
    for (const [name, bufId] of map) {
      const spec = plan.buffers[bufId]!
      const staging = device.createBuffer({ size: spec.byteSize, usage: READBACK })
      encoder.copyBufferToBuffer(buffers.get(bufId)!, 0, staging, 0, spec.byteSize)
      stagings.push({ name, buf: staging, bytes: spec.byteSize })
    }
    queue.submit([encoder.finish()])
    const out: Record<string, Float32Array> = {}
    for (const s of stagings) {
      await s.buf.mapAsync(GPUMapMode.READ)
      out[s.name] = new Float32Array(s.buf.getMappedRange().slice(0))
      s.buf.unmap()
      s.buf.destroy()
    }
    return out
  }

  // Fill a state buffer with its declared initValue. Used at allocation and
  // on resetOptimizerState().
  function fillStateBuffer(spec: { byteSize: number; dtype: 'f32' | 'i32' | 'bool'; initValue?: number }, target: GPUBuffer): void {
    const elements = spec.byteSize / 4
    const init = spec.dtype === 'f32'
      ? new Float32Array(elements).fill(spec.initValue ?? 0)
      : new Int32Array(elements).fill(Math.trunc(spec.initValue ?? 0))
    queue.writeBuffer(target, 0, init as unknown as BufferSource)
  }

  function resetOptimizerState() {
    for (const spec of plan.buffers) {
      if (spec.kind === 'state') fillStateBuffer(spec, buffers.get(spec.id)!)
    }
  }

  // Built after allocation so it points at the actual (possibly shared) buffers.
  const params = new Map<string, GPUBuffer>()
  for (const [name, bufId] of plan.paramsByName) {
    params.set(name, buffers.get(bufId)!)
  }
  // Static shape per capture, surfaced so callers don't recompute strides.
  const captureShapes: Record<string, number[]> = {}
  for (const [name, bufId] of plan.capturesByName) {
    captureShapes[name] = [...plan.buffers[bufId]!.shape]
  }
  const outputShape = [...plan.buffers[lossBufferId]!.shape]

  const destroy = () => {
    for (const [id, b] of buffers) {
      if (ownedBufferIds.has(id)) b.destroy()
    }
    outputReadback.destroy()
    if (captureStaging) captureStaging.buffer.destroy()
  }

  return {
    device,
    params,
    outputShape,
    uploadParams,
    downloadParams: () => downloadFromMap(plan.paramsByName),
    downloadParamGrads: () => downloadFromMap(plan.paramGradsByName),
    step,
    queueStep,
    run,
    readLoss,
    resetOptimizerState,
    destroy,
  }
}

/** Same machinery as `createRuntime`, narrower public type: a forward-only
 *  graph exposes `run()` instead of `step()` (no optimizer state, no scalar-
 *  loss readback). The full runtime object is built once and projected by
 *  `compileForward` to the public shape. */
export async function createForwardRuntime(
  plan: BufferPlan,
  kernels: KernelSpec[],
  outputBufferId: number,
  opts: RuntimeOpts = {},
): Promise<CompiledForward> {
  return await createRuntime(plan, kernels, outputBufferId, opts)
}

async function acquireDevice(): Promise<GPUDevice> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('tensorgrad: WebGPU not available in this environment')
  }
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('tensorgrad: no WebGPU adapter')
  return await adapter.requestDevice()
}
