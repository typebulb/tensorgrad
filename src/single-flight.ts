/** Discriminated result of a `singleFlight`-wrapped call. `'completed'`
 *  carries the wrapped function's value; `'aborted'` means a newer call
 *  superseded this one before it resolved. Same vocabulary as
 *  `step` / `run` so callers use one cancellation pattern across the
 *  library. */
export type SingleFlightResult<R> =
  | { kind: 'completed'; value: R }
  | { kind: 'aborted' }

interface PendingCall<A, R> {
  arg: A
  resolve: (r: SingleFlightResult<R>) => void
}

/**
 * Wrap an async function so overlapping calls coalesce to "latest wins."
 * Matches RxJS `switchMap` / p-debounce: at most one call is in flight,
 * at most one is queued, and any displaced waiter resolves with
 * `{ kind: 'aborted' }`.
 *
 * ```ts
 * const predict = singleFlight(b => infer.run({ x: b }))
 * const r = await predict(bytes)
 * if (r.kind === 'completed') updateUI(r.value)
 * ```
 *
 * Single-argument only — pack multiple args into one object.
 */
export function singleFlight<A, R>(
  fn: (arg: A) => Promise<R>,
): (arg: A) => Promise<SingleFlightResult<R>> {
  let active: Promise<R> | null = null
  let pending: PendingCall<A, R> | null = null

  function call(arg: A): Promise<SingleFlightResult<R>> {
    if (!active) {
      const p = fn(arg)
      active = p
      const result = p.then<SingleFlightResult<R>>(value => ({ kind: 'completed', value }))
      p.finally(() => {
        active = null
        if (pending) {
          const next = pending
          pending = null
          call(next.arg).then(next.resolve)
        }
      })
      return result
    }
    if (pending) pending.resolve({ kind: 'aborted' })
    return new Promise<SingleFlightResult<R>>(resolve => {
      pending = { arg, resolve }
    })
  }

  return call
}
