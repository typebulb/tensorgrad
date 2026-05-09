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

export interface CompiledRuntime {
  /** Upload parameter Float32Arrays to their GPU buffers. By default, requires
   *  *all* params to be present; throws on any unknown or missing key. Pass
   *  `{ partial: true }` to skip the missing-key check. */
  uploadParams(params: Record<string, Float32Array>, opts?: UploadParamsOptions): void
  /** Read all parameters back as Float32Arrays — used for UI panels. */
  downloadParams(): Promise<Record<string, Float32Array>>
  /** Read all parameter gradients back. Mostly for verification / debugging. */
  downloadParamGrads(): Promise<Record<string, Float32Array>>
  /**
   * One full forward+backward step.
   *   1. Uploads `inputs` (tokens, targets, masks) to input buffers.
   *   2. Dispatches every kernel in order.
   *   3. Reads back the loss scalar.
   * Returns the loss as a JS number.
   */
  step(inputs: Record<string, Int32Array | Float32Array>): Promise<number>
  /** Re-zero all optimizer state buffers (Adam's m/v) in place. Pair with
   *  `uploadInitialParams()` for a full training reset without recompile. */
  resetOptimizerState(): void
  /** Free GPU resources. */
  destroy(): void
}

export interface RuntimeOpts {
  /** Pre-acquired GPUDevice. If omitted, runtime requests its own. */
  device?: GPUDevice
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
  const buffers = new Map<number, GPUBuffer>()
  for (const spec of plan.buffers) {
    const buf = device.createBuffer({
      size: spec.byteSize,
      usage: STORAGE_RW,
      label: spec.name ?? `t${spec.id}-${spec.kind}`,
    })
    buffers.set(spec.id, buf)
    if (spec.kind === 'state') {
      // Fill with initValue (typically 0). Float and int both 4 bytes per element.
      const elements = spec.byteSize / 4
      const init = spec.dtype === 'f32'
        ? new Float32Array(elements).fill(spec.initValue ?? 0)
        : new Int32Array(elements).fill(Math.trunc(spec.initValue ?? 0))
      queue.writeBuffer(buf, 0, init as unknown as BufferSource)
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

  // ---- Loss readback staging buffer -----------------------------------------
  const lossSpec = plan.buffers[lossBufferId]!
  const lossReadback = device.createBuffer({ size: lossSpec.byteSize, usage: READBACK })

  // ---- step() ---------------------------------------------------------------
  async function step(inputs: Record<string, Int32Array | Float32Array>): Promise<number> {
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
    // After all dispatches: writebacks (Adam state, updated params).
    // copyBufferToBuffer is queued onto the same encoder so it's ordered after
    // all kernel dispatches.
    for (const wb of plan.writebacks) {
      encoder.copyBufferToBuffer(buffers.get(wb.source)!, 0, buffers.get(wb.dest)!, 0, wb.bytes)
    }
    encoder.copyBufferToBuffer(buffers.get(lossBufferId)!, 0, lossReadback, 0, lossSpec.byteSize)
    queue.submit([encoder.finish()])

    await lossReadback.mapAsync(GPUMapMode.READ)
    const view = new Float32Array(lossReadback.getMappedRange().slice(0))
    lossReadback.unmap()
    return view[0]!
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

  function resetOptimizerState() {
    for (const spec of plan.buffers) {
      if (spec.kind !== 'state') continue
      const elements = spec.byteSize / 4
      const init = spec.dtype === 'f32'
        ? new Float32Array(elements).fill(spec.initValue ?? 0)
        : new Int32Array(elements).fill(Math.trunc(spec.initValue ?? 0))
      queue.writeBuffer(buffers.get(spec.id)!, 0, init as unknown as BufferSource)
    }
  }

  return {
    uploadParams,
    downloadParams: () => downloadFromMap(plan.paramsByName),
    downloadParamGrads: () => downloadFromMap(plan.paramGradsByName),
    step,
    resetOptimizerState,
    destroy: () => {
      for (const b of buffers.values()) b.destroy()
      lossReadback.destroy()
    },
  }
}

async function acquireDevice(): Promise<GPUDevice> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('tensorgrad: WebGPU not available in this environment')
  }
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('tensorgrad: no WebGPU adapter')
  return await adapter.requestDevice()
}
