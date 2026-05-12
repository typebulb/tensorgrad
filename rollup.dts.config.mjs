// Bundles the per-file .d.ts in dist/types-temp/ into a single dist/index.d.ts.
// Why: some type resolvers (notably typebulb's Monaco loader) don't follow
// `export … from './module.js'` re-export chains across multiple .d.ts files,
// and report missing exports for symbols that live in sibling files.
// Inlining everything into the entry .d.ts works around this — same trick
// domeleon uses.
import dts from 'rollup-plugin-dts'

export default [
  {
    input: 'dist/types-temp/index.d.ts',
    output: { file: 'dist/index.d.ts', format: 'es' },
    plugins: [dts()],
  },
  {
    input: 'dist/types-temp/internal.d.ts',
    output: { file: 'dist/internal.d.ts', format: 'es' },
    plugins: [dts()],
  },
]
