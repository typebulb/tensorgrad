// Main-thread half of the worker channel: request/response correlation,
// promise wiring, error reconstitution. Knows nothing about Adam, captures,
// IR, etc. — just shuttles typed messages.

import type { Req, Res, WireError } from './worker-protocol.js'
import { reconstituteError } from './worker-protocol.js'

interface PendingHandlers {
  resolve: (v: unknown) => void
  reject: (e: Error) => void
}

/** Spawn a worker from an inlined source string and provide a typed
 *  request/response channel. One WorkerProxy = one Worker = one GPUDevice
 *  on the worker side. Sibling graphs share the same WorkerProxy. */
export class WorkerProxy {
  private worker: Worker
  private nextId = 1
  private pending = new Map<number, PendingHandlers>()
  private terminated = false

  constructor(workerSource: string) {
    const blob = new Blob([workerSource], { type: 'application/javascript' })
    const url = URL.createObjectURL(blob)
    this.worker = new Worker(url, { type: 'module' })
    // The Blob URL keeps memory alive as long as it's referenced; revoke
    // once the worker has loaded its source. Browsers tolerate revoke
    // immediately after construction in practice.
    URL.revokeObjectURL(url)

    this.worker.onmessage = (ev: MessageEvent<Res>) => {
      const reply = ev.data
      const handlers = this.pending.get(reply.id)
      if (!handlers) return  // stale reply; ignore
      this.pending.delete(reply.id)
      if (reply.ok) handlers.resolve(reply.result)
      else handlers.reject(reconstituteError(reply.error))
    }

    this.worker.onerror = (ev: ErrorEvent) => {
      const err = new Error(`tensorgrad worker error: ${ev.message || 'unknown'}`)
      const wire: WireError = { name: 'WorkerError', message: err.message, stack: err.stack ?? '' }
      // Reject everything in flight; subsequent calls will fail too.
      for (const handlers of this.pending.values()) handlers.reject(reconstituteError(wire))
      this.pending.clear()
    }
  }

  /** Send a request and await its matching response. `transfer` lists the
   *  ArrayBuffers to move (zero-copy) into the worker. */
  request<R>(req: Omit<Req, 'id'>, transfer: ArrayBuffer[] = []): Promise<R> {
    if (this.terminated) return Promise.reject(new Error('tensorgrad: worker has been terminated'))
    const id = this.nextId++
    return new Promise<R>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject })
      this.worker.postMessage({ ...req, id } as Req, transfer)
    })
  }

  /** Fire-and-forget variant for cases where the caller doesn't need a reply
   *  (currently unused; keep for symmetry / future use). */
  send(req: Omit<Req, 'id'>, transfer: ArrayBuffer[] = []): void {
    if (this.terminated) return
    const id = this.nextId++
    this.worker.postMessage({ ...req, id } as Req, transfer)
  }

  terminate(): void {
    if (this.terminated) return
    this.terminated = true
    this.worker.terminate()
    const err = new Error('tensorgrad: worker terminated')
    for (const handlers of this.pending.values()) handlers.reject(err)
    this.pending.clear()
  }
}
