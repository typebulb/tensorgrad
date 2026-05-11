// Minimal test helpers — no framework. Each test file runs as `tsx test/X.ts`
// and exits non-zero on the first failure. The runner in package.json chains
// them, so a failure halts the suite at the offending file.
//
// Style matches the existing smoke.ts: console-driven, fail-fast, no asserts
// that swallow context. Helpers exist to deduplicate the prefix/format.

// Minimal Node typings the test harness needs. We don't pull in `@types/node`
// to keep the dev-dep tree small; the few symbols the runner touches are
// declared here.
declare const process: { exit(code: number): never }

let sectionDepth = 0

export function section(name: string): void {
  console.log((sectionDepth === 0 ? '\n' : '') + name)
  sectionDepth = 1
}

export function ok(msg: string): void {
  console.log(`  ✓ ${msg}`)
}

export function fail(msg: string): never {
  console.error(`  ✗ ${msg}`)
  process.exit(1)
}

export function assert(cond: boolean, msg: string): void {
  if (!cond) fail(msg)
  else ok(msg)
}

export function assertEq<T extends string | number | boolean | null | undefined>(
  actual: T, expected: T, msg: string,
): void {
  if (actual !== expected) fail(`${msg}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  else ok(`${msg} = ${JSON.stringify(actual)}`)
}

export function assertShape(actual: readonly number[], expected: readonly number[], msg: string): void {
  if (actual.length !== expected.length || actual.some((v, i) => v !== expected[i])) {
    fail(`${msg}: expected [${expected.join(', ')}], got [${actual.join(', ')}]`)
  }
  ok(`${msg} = [${actual.join(', ')}]`)
}

export function assertThrows(fn: () => unknown, expectedFragment: string, msg: string): void {
  try {
    fn()
  } catch (e: unknown) {
    const got = String((e as { message?: string })?.message ?? e)
    if (!got.includes(expectedFragment)) {
      fail(`${msg}: error didn't include "${expectedFragment}". Got: ${got}`)
    }
    ok(`${msg} → threw containing "${expectedFragment}"`)
    return
  }
  fail(`${msg}: expected throw including "${expectedFragment}", but no error was raised`)
}

export function done(name: string): void {
  console.log(`\n${name} — all assertions passed.`)
}
