interface PendingCall<A, R> {
  arg: A
  resolver: { resolve: (r: R) => void; reject: (e: Error) => void }
}

/**
 * Wrap an async function so overlapping calls coalesce to "latest wins."
 * Matches RxJS `switchMap` / p-debounce: at most one call is in flight,
 * at most one is queued, and any displaced waiter rejects with `AbortError`.
 *
 * ```ts
 * const predict = singleFlight(b => infer.run({ x: b }))
 * try { updateUI(await predict(bytes)) }
 * catch (e) { if (e?.name !== 'AbortError') throw e }
 * ```
 *
 * Single-argument only — pack multiple args into one object.
 */
export function singleFlight<A, R>(fn: (arg: A) => Promise<R>): (arg: A) => Promise<R> {
  let active: Promise<R> | null = null
  let pending: PendingCall<A, R> | null = null

  function call(arg: A): Promise<R> {
    if (!active) {
      const p = fn(arg)
      active = p
      p.finally(() => {
        active = null
        if (pending) {
          const next = pending
          pending = null
          call(next.arg).then(next.resolver.resolve, next.resolver.reject)
        }
      })
      return p
    }
    if (pending) pending.resolver.reject(abortErr('singleFlight: superseded by newer call'))
    return new Promise<R>((resolve, reject) => {
      pending = { arg, resolver: { resolve, reject } }
    })
  }

  return call
}

function abortErr(msg: string): Error {
  const e = new Error(msg)
  e.name = 'AbortError'
  return e
}
