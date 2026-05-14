// IR viewer picker: discovers `spec.ts` in every sibling sample folder via
// Vite's import.meta.glob, lists them in a dropdown, and renders the chosen
// sample's compiled training graph. Each sample stays standalone — its spec
// is the import-safe shared kernel that both its own boot and this picker
// consume.

import { renderIRViewer, type IRSpec, type InspectableTraining } from 'tensorgrad-viewer'

// Eager — specs are pure declarations + a compile thunk, no boot side
// effects, so loading them all up front is cheap and lets us populate the
// dropdown with real labels.
const modules = import.meta.glob<{ irSpec: IRSpec }>('../*/spec.ts', { eager: true })

type Entry = { key: string; spec: IRSpec }
const entries: Entry[] = Object.entries(modules)
  .map(([path, mod]) => ({
    key: path.split('/').slice(-2, -1)[0]!,
    spec: mod.irSpec,
  }))
  .sort((a, b) => a.spec.label.localeCompare(b.spec.label))

const pickerMount = document.getElementById('picker-mount')!
const viewerMount = document.getElementById('ir-mount')!

if (entries.length === 0) {
  viewerMount.textContent = 'No sample specs found. Add a `spec.ts` to a sample folder that exports an `irSpec`.'
} else {
  boot()
}

function boot(): void {
  // Dropdown chrome lives in the page's sticky header so it stays visible
  // while scrolling through tall graphs.
  pickerMount.innerHTML = ''
  const controls = document.createElement('div')
  controls.style.cssText = 'display: flex; gap: 0.5rem; align-items: center; font-size: 0.9rem;'
  const label = document.createElement('label')
  label.htmlFor = 'sample-select'
  label.textContent = 'Sample:'
  controls.appendChild(label)
  const select = document.createElement('select')
  select.id = 'sample-select'
  select.style.cssText = 'padding: 0.2rem 0.4rem; font-size: 0.9rem;'
  for (const e of entries) {
    const opt = document.createElement('option')
    opt.value = e.key
    opt.textContent = e.spec.label
    select.appendChild(opt)
  }
  controls.appendChild(select)
  const status = document.createElement('span')
  status.style.cssText = 'color: #888; font-size: 0.82rem;'
  controls.appendChild(status)
  pickerMount.appendChild(controls)

  const mount = viewerMount

  let current: InspectableTraining | null = null
  let inflight = 0

  async function show(key: string): Promise<void> {
    const entry = entries.find(e => e.key === key)
    if (!entry) return
    const myToken = ++inflight
    status.textContent = `Compiling ${entry.spec.label}…`
    mount.innerHTML = ''

    // Destroy previous compile's worker before spinning up the next one. If
    // another switch already raced ahead, bail.
    const prev = current
    current = null
    if (prev) prev.destroy()
    if (myToken !== inflight) return

    let train: InspectableTraining
    try {
      train = await entry.spec.compile()
    } catch (e) {
      status.textContent = `error: ${(e as Error)?.message ?? e}`
      console.error(e)
      return
    }
    if (myToken !== inflight) {
      train.destroy()
      return
    }
    current = train
    status.textContent = ''

    await renderIRViewer({
      container: mount,
      graph: train.graph,
      kernelCount: train.kernels.length,
      ...(entry.spec.dims !== undefined ? { dims: entry.spec.dims } : {}),
    })
  }

  select.addEventListener('change', () => { void show(select.value) })
  void show(entries[0]!.key)
}
