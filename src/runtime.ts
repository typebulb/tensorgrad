// WebGPU runtime. Reads a BufferPlan + KernelSpec[] (produced by codegen),
// allocates real GPU buffers and pipelines, and provides a `step()` method
// that uploads inputs, dispatches all kernels, and reads back outputs.
//
// Browser-only: this module needs `navigator.gpu` at runtime.

import type { BufferPlan } from './buffers.js'
import type { KernelSpec } from './codegen.js'

// TS lib.dom defines WebGPU types but not the GPUMapMode runtime constant.
// Provided by the browser per WebGPU spec; declare just what we use.
declare const GPUMapMode: { readonly READ: number; readonly WRITE: number }

export interface UploadParamsOptions {
  /** Skip the "missing param" check, allowing the caller to update only some
   *  params and leave the rest at their current GPU values. Extra (unknown)
   *  keys are still rejected — that's always a typo. Default: false. */
  partial?: boolean
}

export interface StepOptions {
  /** Read back tensors registered via `capture(name, t)` during the trace.
   *  When false/unset, `step()` returns just the loss number; staging buffers
   *  for captures are not allocated. When true, returns `{ loss, captures }`
   *  and lazily allocates one staging buffer per capture on first use. */
  withCaptures?: boolean
}

export interface StepWithCaptures {
  loss: number
  captures: Record<string, Float32Array>
}

export interface RunOptions {
  /** Read back tensors registered via `capture(name, t)` during the trace.
   *  When false/unset, `run()` returns just the output Float32Array.
   *  When true, returns `{ output, captures }`. */
  withCaptures?: boolean
}

export interface RunWithCaptures {
  output: Float32Array
  captures: Record<string, Float32Array>
}

/** Common surface for both training and forward-only compiled runtimes. */
export interface CompiledBase {
  /** Param name -> the underlying GPUBuffer. Pass to a sibling compile via
   *  `sharedParams` to share without copies. */
  params: Map<string, GPUBuffer>
  /** Shape of each tensor registered via `capture(name, t)`. Static after
   *  compile — reshape readbacks without recomputing strides. */
  captureShapes: Record<string, number[]>
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

/** Run a dispatch and read back the full output tensor (and any registered
 *  captures if requested). Forward-only compiles use this as their primary
 *  surface; training compiles also expose it but `step()` is more convenient
 *  there because the output is a scalar loss. */
export interface RunFn {
  (inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  (inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunWithCaptures>
  (inputs: Record<string, Int32Array | Float32Array>, opts: RunOptions): Promise<Float32Array | RunWithCaptures>
}

export interface CompiledRuntime extends CompiledBase {
  /** Read all parameter gradients back. Mostly for verification / debugging. */
  downloadParamGrads(): Promise<Record<string, Float32Array>>
  /**
   * One full forward+backward step.
   *   1. Uploads `inputs` (tokens, targets, masks) to input buffers.
   *   2. Dispatches every kernel in order.
   *   3. Reads back the loss scalar (and any registered captures, if requested).
   * Default returns the loss as a JS number; with `{ withCaptures: true }`
   * returns `{ loss, captures }` where `captures` is keyed by the names passed
   * to `capture(...)` during the trace.
   */
  step(inputs: Record<string, Int32Array | Float32Array>): Promise<number>
  step(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<StepWithCaptures>
  step(inputs: Record<string, Int32Array | Float32Array>, opts: StepOptions): Promise<number | StepWithCaptures>
  /** Same dispatch as step() but returns the full output Float32Array — for
   *  training graphs the output is a scalar loss, so step() is usually more
   *  convenient. Provided for parity with `compileForward`. */
  run: RunFn
  /** Re-zero all optimizer state buffers (Adam's m/v) in place. Pair with
   *  `uploadInitialParams()` for a full training reset without recompile. */
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

// Inlined numeric values (per WebGPU spec) so this module is importable in Node
// for codegen-only usage. The browser provides GPUBufferUsage as a global, but
// referencing it at module scope would crash before any browser code runs.
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

  // ---- Allocate one GPUBuffer per BufferSpec --------------------------------
  // State buffers also get filled with their initValue at allocation time.
  // Param buffers may be supplied externally via opts.sharedParams; in that
  // case we reuse the provided GPUBuffer instead of allocating, and the
  // sibling compile that owns it is responsible for upload + lifetime.
  const buffers = new Map<number, GPUBuffer>()
  const sharedParams = opts.sharedParams
  for (const spec of plan.buffers) {
    if (spec.kind === 'param' && sharedParams?.has(spec.name!)) {
      const shared = sharedParams.get(spec.name!)!
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
    if (spec.kind === 'state') fillStateBuffer(spec, buf)
  }
  // Track which params are externally owned — those are skipped on destroy().
  const ownedBufferIds = new Set<number>()
  for (const spec of plan.buffers) {
    if (!(spec.kind === 'param' && sharedParams?.has(spec.name!))) {
      ownedBufferIds.add(spec.id)
    }
  }

  // ---- Compile pipelines per kernel; cache by WGSL source -------------------
  // Push an error scope around each shader+pipeline creation so we can surface
  // the actual compile error rather than the cryptic "previous error" that
  // comes from using an invalid pipeline at dispatch time.
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

  // ---- Pre-build bind groups (static — buffer ids don't change per step) ---
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

  // ---- Output readback staging buffer ---------------------------------------
  // `outputBufferId` is the graph's main output (loss for training, the user's
  // returned tensor for forward-only). step() reads back its first element;
  // run() reads back the full Float32Array.
  const outputSpec = plan.buffers[lossBufferId]!
  const outputReadback = device.createBuffer({ size: outputSpec.byteSize, usage: READBACK })

  // ---- Capture readback staging buffers (lazy) ------------------------------
  // Allocated on first `step({ withCaptures: true })` call and reused across
  // subsequent calls. When the graph has no captures registered or when the
  // caller never opts in, no extra GPU memory is allocated.
  let captureStagings: Map<string, GPUBuffer> | null = null
  function ensureCaptureStagings(): Map<string, GPUBuffer> {
    if (captureStagings) return captureStagings
    captureStagings = new Map()
    for (const [name, bufId] of plan.capturesByName) {
      const spec = plan.buffers[bufId]!
      const staging = device.createBuffer({ size: spec.byteSize, usage: READBACK, label: `cap-${name}` })
      captureStagings.set(name, staging)
    }
    return captureStagings
  }

  // ---- dispatch() — shared core for step() and run() -----------------------
  // Uploads inputs, dispatches all kernels (in order), queues writebacks, copies
  // the output buffer into its staging, optionally copies captures into theirs,
  // submits, and reads back. Returns the full output Float32Array; step() takes
  // [0] for scalar loss, run() returns it whole.
  async function dispatch(
    inputs: Record<string, Int32Array | Float32Array>,
    wantCaptures: boolean,
  ): Promise<{ output: Float32Array; captures: Record<string, Float32Array> | null }> {
    if (wantCaptures && plan.capturesByName.size === 0) {
      throw new Error(
        `withCaptures=true but no capture(...) calls were registered during ` +
        `the trace. Add capture('name', tensor) inside your forward pass for ` +
        `the intermediates you want read back.`,
      )
    }
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
      // WebGPU caps each dispatch dimension at 65535 workgroups. Split into 2D
      // when a kernel needs more than that on the X axis. Kernels compute their
      // global index as `gid.x + gid.y * (65535 * workgroup_size)`, matching the
      // stride we set here. For dispatches that fit in one row, gid.y is 0.
      const wgCount = Math.max(1, Math.ceil(k.threads / k.workgroupSize))
      const MAX_X = 65535
      const wgX = Math.min(wgCount, MAX_X)
      const wgY = Math.ceil(wgCount / MAX_X)
      pass.dispatchWorkgroups(wgX, wgY, 1)
      pass.end()
    }
    // After all dispatches: writebacks (Adam state, updated params). Empty for
    // forward-only compiles.
    for (const wb of plan.writebacks) {
      encoder.copyBufferToBuffer(buffers.get(wb.source)!, 0, buffers.get(wb.dest)!, 0, wb.bytes)
    }
    encoder.copyBufferToBuffer(buffers.get(lossBufferId)!, 0, outputReadback, 0, outputSpec.byteSize)
    // Capture readbacks (only when opted in). Queued before submit so they
    // observe the same kernel outputs as the main output.
    let stagings: Map<string, GPUBuffer> | null = null
    if (wantCaptures) {
      stagings = ensureCaptureStagings()
      for (const [name, bufId] of plan.capturesByName) {
        const spec = plan.buffers[bufId]!
        encoder.copyBufferToBuffer(buffers.get(bufId)!, 0, stagings.get(name)!, 0, spec.byteSize)
      }
    }
    queue.submit([encoder.finish()])

    await outputReadback.mapAsync(GPUMapMode.READ)
    const output = new Float32Array(outputReadback.getMappedRange().slice(0))
    outputReadback.unmap()

    if (!wantCaptures) return { output, captures: null }

    const captures: Record<string, Float32Array> = {}
    for (const [name, staging] of stagings!) {
      await staging.mapAsync(GPUMapMode.READ)
      captures[name] = new Float32Array(staging.getMappedRange().slice(0))
      staging.unmap()
    }
    return { output, captures }
  }

  // ---- step() — training-mode wrapper, returns scalar [0] of output ---------
  function step(inputs: Record<string, Int32Array | Float32Array>): Promise<number>
  function step(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<StepWithCaptures>
  function step(inputs: Record<string, Int32Array | Float32Array>, opts: StepOptions): Promise<number | StepWithCaptures>
  async function step(
    inputs: Record<string, Int32Array | Float32Array>,
    opts?: StepOptions,
  ): Promise<number | StepWithCaptures> {
    const r = await dispatch(inputs, opts?.withCaptures === true)
    if (opts?.withCaptures) return { loss: r.output[0]!, captures: r.captures! }
    return r.output[0]!
  }

  // ---- run() — forward-mode wrapper, returns full output Float32Array -------
  function run(inputs: Record<string, Int32Array | Float32Array>): Promise<Float32Array>
  function run(inputs: Record<string, Int32Array | Float32Array>, opts: { withCaptures: true }): Promise<RunWithCaptures>
  function run(inputs: Record<string, Int32Array | Float32Array>, opts: RunOptions): Promise<Float32Array | RunWithCaptures>
  async function run(
    inputs: Record<string, Int32Array | Float32Array>,
    opts?: RunOptions,
  ): Promise<Float32Array | RunWithCaptures> {
    const r = await dispatch(inputs, opts?.withCaptures === true)
    if (opts?.withCaptures) return { output: r.output, captures: r.captures! }
    return r.output
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

  // ---- download helpers -----------------------------------------------------
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

  // Fill a state buffer with its declared initValue (typically 0). Float and
  // int both serialize to 4 bytes per element. Used at allocation time and on
  // resetOptimizerState() — same logic, two callers.
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

  // Build the params map AFTER buffer allocation so it points at the actual
  // GPUBuffers (shared or freshly allocated).
  const params = new Map<string, GPUBuffer>()
  for (const [name, bufId] of plan.paramsByName) {
    params.set(name, buffers.get(bufId)!)
  }
  // Static-after-compile shape metadata so users don't have to recompute
  // strides to interpret a flat capture readback.
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
    if (captureStagings) for (const b of captureStagings.values()) b.destroy()
  }

  return {
    params,
    captureShapes,
    outputShape,
    uploadParams,
    downloadParams: () => downloadFromMap(plan.paramsByName),
    downloadParamGrads: () => downloadFromMap(plan.paramGradsByName),
    step,
    run,
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
