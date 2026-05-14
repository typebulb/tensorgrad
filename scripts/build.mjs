// Build orchestrator. Two-stage:
//   1. Bundle src/worker.ts into a single self-contained JS string.
//   2. Bundle src/index.ts with `__WORKER_SOURCE__` substituted to that string
//      via esbuild's `define`. The main bundle then instantiates the worker
//      at runtime via `new Worker(URL.createObjectURL(new Blob([__WORKER_SOURCE__], ...)))`.
//
// Single-file output: consumers `import 'tensorgrad'` and the worker is
// invisible. See specs/WorkerArchitecture.md.
//
// Run: `node scripts/build.mjs`. Watch mode: `node scripts/build.mjs --watch`.

import { build, context } from 'esbuild'
import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const watch = process.argv.includes('--watch')

const sharedEsbuildOpts = {
  bundle: true,
  format: 'esm',
  target: 'es2022',
  platform: 'browser',
  // Tensorgrad has no runtime dependencies; everything in src/ should bundle.
}

async function bundleWorker() {
  const result = await build({
    ...sharedEsbuildOpts,
    entryPoints: ['src/worker.ts'],
    write: false,
    sourcemap: false,
  })
  return result.outputFiles[0].text
}

async function bundleMain(workerSource) {
  const opts = {
    ...sharedEsbuildOpts,
    // Two entry points: the public barrel and the extension barrel. Both fully
    // tree-shaken bundles; users pay for what they import. `__WORKER_SOURCE__`
    // only resolves inside the public bundle (internal doesn't spawn workers).
    entryPoints: ['src/index.ts', 'src/internal.ts'],
    outdir: 'dist',
    sourcemap: true,
    define: {
      // The main bundle reads `__WORKER_SOURCE__` as a string literal at the
      // point it spawns a worker; esbuild inlines the JSON-stringified worker
      // source verbatim (including all of worker.ts's transitively imported
      // module code, which esbuild tree-shook into a single bundle in step 1).
      __WORKER_SOURCE__: JSON.stringify(workerSource),
    },
  }
  if (watch) {
    const ctx = await context(opts)
    await ctx.watch()
    console.log('build.mjs: watching src/{index,internal}.ts (worker source baked at startup; restart on worker.ts changes)')
  } else {
    await build(opts)
  }
}

async function main() {
  await mkdir('dist', { recursive: true })
  await mkdir('build-artifacts', { recursive: true })
  const workerSource = await bundleWorker()
  // Optional debug artifact, written outside dist/ so it doesn't ship: lets us
  // inspect / source-map the worker bundle if something blows up at runtime.
  await writeFile('build-artifacts/worker.debug.js', workerSource)
  await bundleMain(workerSource)
  if (!watch) {
    const sizeKB = (workerSource.length / 1024).toFixed(1)
    console.log(`build.mjs: worker bundle ${sizeKB} KB inlined into dist/index.js`)
  }
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
