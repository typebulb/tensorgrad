// Generic single-flight wrapper for async functions. Use it when you want
// "latest-wins" semantics: rapid successive calls supersede earlier queued
// calls, only the most recent argument actually runs.
//
// Matches the convention of RxJS `switchMap` and p-debounce: a queued caller
// rejects with `AbortError` when a newer call supersedes it. Only one call
// is in flight at a time; only one is queued. Caller's idiomatic pattern:
//
//   const predict = singleFlight(b => infer.run({ x: b }))
//
//   try {
//     const out = await predict(bytes)
//     updateUI(out)
//   } catch (e) {
//     if (e?.name === 'AbortError') return  // newer call superseded
//     throw e
//   }
//
// This is a plain utility — nothing tensorgrad-specific. It lives in the
// public surface because the pattern is common enough (drawing-canvas
// previews, hover-based inference, debounced search) that re-implementing
// the state machine at every call site is wasteful.

interface PendingCall<A, R> {
  arg: A
  resolver: { resolve: (r: R) => void; reject: (e: Error) => void }
}

/** Wrap an async function so that overlapping calls coalesce to "latest
 *  wins." See module header for semantics. The wrapped function takes the
 *  same single argument as the original; if your underlying function takes
 *  multiple arguments, pack them into one object first. */
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
          // Re-enter for the queued waiter. Only this single resolver gets
          // the result; any callers that arrived before it were already
          // displaced (rejected with AbortError) by their successors.
          call(next.arg).then(next.resolver.resolve, next.resolver.reject)
        }
      })
      return p
    }
    // Already in flight: this call replaces any older queued waiter. The
    // older waiter rejects with AbortError.
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
