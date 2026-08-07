// WebGPU availability probing. Shared by the public pre-flight API
// (`checkWebGPU`) and the worker's device bring-up so both report the same
// diagnosis when a machine has no usable GPU.

/** Result of `checkWebGPU`: `ok`, or a reason plus a `message` ready to show
 *  an end user, naming the fix when one is known. */
export type WebGPUSupport =
  | { ok: true }
  | { ok: false; reason: 'no-webgpu' | 'no-adapter'; message: string }

/** True when this environment exposes the WebGPU API — the surface only, not
 *  a usable GPU: Chrome on Linux exposes `navigator.gpu` yet hands out no
 *  adapters unless the user enables its flags.
 *  @deprecated Prefer awaiting `checkWebGPU()`, which probes for a real
 *  adapter and returns a user-showable message on failure. Reach for this
 *  only in synchronous contexts that cannot await. */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator
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

// The one no-adapter case with a known user-side fix: Chromium browsers on
// desktop Linux expose `navigator.gpu` but gate adapters behind flags.
function isChromiumOnLinux(): boolean {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : ''
  return ua.includes('Linux') && !ua.includes('Android') && ua.includes('Chrome/')
}

/** User-showable diagnosis for a failed WebGPU probe. */
export function webGPUFailureMessage(reason: 'no-webgpu' | 'no-adapter'): string {
  if (reason === 'no-webgpu') {
    return 'This browser does not support WebGPU. It needs a recent version of Chrome, Edge, Firefox, or Safari.'
  }
  if (isChromiumOnLinux()) {
    return 'This browser ships WebGPU behind a flag on Linux: enable chrome://flags/#enable-unsafe-webgpu and chrome://flags/#enable-vulkan, then relaunch. Firefox on Linux enables WebGPU by default.'
  }
  return 'This browser supports WebGPU, but no GPU adapter is available. The GPU may be blocklisted by the browser, or its drivers may need updating.'
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
  if (!isWebGPUAvailable()) {
    return { ok: false, reason: 'no-webgpu', message: webGPUFailureMessage('no-webgpu') }
  }
  const adapter = await requestAdapterWithRetry()
  if (!adapter) {
    return { ok: false, reason: 'no-adapter', message: webGPUFailureMessage('no-adapter') }
  }
  return { ok: true }
}
