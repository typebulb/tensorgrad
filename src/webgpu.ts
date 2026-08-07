// WebGPU availability probing. Shared by the public pre-flight API
// (`checkWebGPU`) and the worker's device bring-up so both report the same
// diagnosis when a machine has no usable GPU.

/** Why a WebGPU probe failed. */
export type WebGPUFailureReason = 'no-webgpu' | 'no-adapter'

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
  reason?: WebGPUFailureReason
}

/** Retry backoff for a null adapter, shared with the worker probe source. */
const retryDelaysMs = [0, 100, 400]

/** requestAdapter() resolves with null (not throws) when the GPU process
 *  is transiently unhealthy — recent crash, power-state transition, sandboxed
 *  iframe+blob worker quirk, etc. Retry with backoff before giving up. */
export async function requestAdapterWithRetry(
  options?: GPURequestAdapterOptions,
): Promise<GPUAdapter | null> {
  for (const ms of retryDelaysMs) {
    if (ms > 0) await new Promise(r => setTimeout(r, ms))
    const adapter = await navigator.gpu.requestAdapter(options)
    if (adapter) return adapter
  }
  return null
}

/** User-showable diagnosis for a failed WebGPU probe. */
export function webGPUFailureMessage(reason: WebGPUFailureReason): string {
  if (reason === 'no-webgpu') {
    return 'This browser does not support WebGPU. It needs a recent version of Chrome, Edge, Firefox, or Safari.'
  }
  return 'WebGPU is supported but no GPU adapter is available. On Linux Chrome, WebGPU is behind flags: enable chrome://flags/#enable-unsafe-webgpu and chrome://flags/#enable-vulkan, then relaunch (Firefox on Linux has it on by default). Otherwise, the GPU may be blocklisted by the browser, or its drivers may need updating.'
}

const supported: WebGPUSupport = { ok: true, message: '' }

function unsupported(reason: WebGPUFailureReason): WebGPUSupport {
  return { ok: false, reason, message: webGPUFailureMessage(reason) }
}

/** Probe for an adapter inside a throwaway worker. The main thread's first
 *  `requestAdapter` stalls frame production for its whole duration (observed
 *  160ms–seconds: the renderer's GPU-channel/adapter bring-up), so the probe
 *  must not run there — and the worker is also the honest context: tensorgrad
 *  only ever creates its device inside one. Resolves null when workers are
 *  unusable here (construction throws, or the worker errors), so the caller
 *  can fall back to a main-thread probe. */
function probeAdapterInWorker(): Promise<'ok' | WebGPUFailureReason | null> {
  const src = `(async () => {
    if (!('gpu' in navigator)) { postMessage('no-webgpu'); return }
    for (const ms of ${JSON.stringify(retryDelaysMs)}) {
      if (ms > 0) await new Promise(r => setTimeout(r, ms))
      try { if (await navigator.gpu.requestAdapter()) { postMessage('ok'); return } } catch {}
    }
    postMessage('no-adapter')
  })()`
  return new Promise(resolve => {
    let worker: Worker
    try {
      const url = URL.createObjectURL(new Blob([src], { type: 'application/javascript' }))
      worker = new Worker(url, { type: 'module' })
      URL.revokeObjectURL(url)
    } catch {
      resolve(null)
      return
    }
    const settle = (v: 'ok' | WebGPUFailureReason | null) => {
      worker.terminate()
      resolve(v)
    }
    worker.onmessage = ev => settle(ev.data)
    worker.onerror = () => settle(null)
  })
}

/** The worker-first probe behind `checkWebGPU`, uncached. */
async function probeSupport(): Promise<WebGPUSupport> {
  const viaWorker = await probeAdapterInWorker()
  if (viaWorker === 'ok') return supported
  if (viaWorker !== null) {
    // The worker lacking WebGPU IS a failed pre-flight: the device is
    // created in a worker, so main-thread-only WebGPU can't run tensorgrad.
    return unsupported(viaWorker)
  }
  // No usable workers in this context: probe here rather than report a
  // failure that compile (which needs workers anyway) will diagnose itself.
  return (await requestAdapterWithRetry()) ? supported : unsupported('no-adapter')
}

/** Cache of the last successful probe, so repeat `checkWebGPU` calls don't
 *  respawn workers. Failures are never cached — a null adapter can be a
 *  transient GPU-process state, and a re-check should see the recovery. */
let okProbe: Promise<WebGPUSupport> | null = null

/**
 * The honest WebGPU pre-flight: probes for an actual adapter rather than just
 * the API surface, since browsers can expose `navigator.gpu` while refusing
 * to hand out adapters (Chrome on Linux without its flags, blocklisted
 * drivers). On failure, `message` is ready to render to the user. Await this
 * before `compile` / `compileForward` so a machine without WebGPU gets a
 * friendly message rather than a crash deep inside the worker. The probe runs
 * in a worker so it never stalls the caller's rendering (and because the
 * worker is where the device will actually live).
 */
export async function checkWebGPU(): Promise<WebGPUSupport> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    return unsupported('no-webgpu')
  }
  okProbe ??= probeSupport().then(
    r => { if (!r.ok) okProbe = null; return r },
    e => { okProbe = null; throw e },
  )
  return okProbe
}
