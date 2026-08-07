// WebGPU availability probing. Shared by the public pre-flight API
// (`checkWebGPU`) and the worker's device bring-up so both report the same
// diagnosis when a machine has no usable GPU.

/** Result of `checkWebGPU`. One flat shape — no union, no narrowing needed:
 *  every property is reachable without a type guard, so it works under any
 *  checker configuration. */
export interface WebGPUSupport {
  /** True when a real adapter was obtained. */
  ok: boolean
  /** Empty when `ok`; otherwise a user-showable diagnosis naming the fix
   *  when one is known. */
  message: string
  /** Why the probe failed; absent when `ok`. */
  reason?: 'no-webgpu' | 'no-adapter'
}

/** requestAdapter() resolves with null (not throws) when the GPU process
 *  is transiently unhealthy — recent crash, power-state transition, sandboxed
 *  iframe+blob worker quirk, etc. Retry with backoff before giving up. */
export async function requestAdapterWithRetry(
  options?: GPURequestAdapterOptions,
): Promise<GPUAdapter | null> {
  for (const ms of [0, 100, 400]) {
    if (ms > 0) await new Promise(r => setTimeout(r, ms))
    const adapter = await navigator.gpu.requestAdapter(options)
    if (adapter) return adapter
  }
  return null
}

/** User-showable diagnosis for a failed WebGPU probe. */
export function webGPUFailureMessage(reason: 'no-webgpu' | 'no-adapter'): string {
  if (reason === 'no-webgpu') {
    return 'This browser does not support WebGPU. It needs a recent version of Chrome, Edge, Firefox, or Safari.'
  }
  return 'WebGPU is supported but no GPU adapter is available. On Linux Chrome, WebGPU is behind flags: enable chrome://flags/#enable-unsafe-webgpu and chrome://flags/#enable-vulkan, then relaunch (Firefox on Linux has it on by default). Otherwise, the GPU may be blocklisted by the browser, or its drivers may need updating.'
}

/**
 * The honest WebGPU pre-flight: probes for an actual adapter rather than just
 * the API surface, since browsers can expose `navigator.gpu` while refusing
 * to hand out adapters (Chrome on Linux without its flags, blocklisted
 * drivers). On failure, `message` is ready to render to the user. Await this
 * before `compile` / `compileForward` so a machine without WebGPU gets a
 * friendly message rather than a crash deep inside the worker.
 */
export async function checkWebGPU(): Promise<WebGPUSupport> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    return { ok: false, reason: 'no-webgpu', message: webGPUFailureMessage('no-webgpu') }
  }
  const adapter = await requestAdapterWithRetry()
  if (!adapter) {
    return { ok: false, reason: 'no-adapter', message: webGPUFailureMessage('no-adapter') }
  }
  return { ok: true, message: '' }
}
