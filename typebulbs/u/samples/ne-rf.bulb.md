---
format: typebulb/v1
name: NeRF
---

**code.tsx**

```tsx
import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'

import {
  Module, compile, isWebGPUAvailable, Linear,
  mul, sub, mean, reshape, relu, sigmoid, concat,
  sin, cos, square,
  type Tensor, type CompiledTraining, type CompiledForward,
} from 'tensorgrad'

const IMG_W = 64
const IMG_H = 64
const N_PIXELS = IMG_W * IMG_H
const BATCH_SIZE = 1024
const L_FREQS = 8
const HIDDEN = 64

const FREQS = new Float32Array(L_FREQS)
for (let k = 0; k < L_FREQS; k++) FREQS[k] = Math.PI * Math.pow(2, k)

const GRID_COORDS = new Float32Array(N_PIXELS * 2)
for (let row = 0; row < IMG_H; row++) {
  for (let col = 0; col < IMG_W; col++) {
    const i = row * IMG_W + col
    GRID_COORDS[i * 2]     = 2 * (col + 0.5) / IMG_W - 1
    GRID_COORDS[i * 2 + 1] = 2 * (row + 0.5) / IMG_H - 1
  }
}

class NeRFTiny extends Module {
  l1 = new Linear(4 * L_FREQS, HIDDEN)
  l2 = new Linear(HIDDEN, HIDDEN)
  l3 = new Linear(HIDDEN, HIDDEN)
  l4 = new Linear(HIDDEN, 3)
}

function posEnc(coords: Tensor, freqs: Tensor): Tensor {
  const B = coords.shape[0]!
  const scaled = mul(reshape(coords, [B, 2, 1]), reshape(freqs, [1, 1, L_FREQS]))
  const sinF = reshape(sin(scaled), [B, 2 * L_FREQS])
  const cosF = reshape(cos(scaled), [B, 2 * L_FREQS])
  return concat([sinF, cosF], 1)
}

function modelFwd(m: NeRFTiny, coords: Tensor, freqs: Tensor): Tensor {
  let h = posEnc(coords, freqs)
  h = relu(m.l1.fwd(h))
  h = relu(m.l2.fwd(h))
  h = relu(m.l3.fwd(h))
  return sigmoid(m.l4.fwd(h))
}

function lossFn(
  m: NeRFTiny,
  { coords, rgb, freqs }: { coords: Tensor; rgb: Tensor; freqs: Tensor },
): Tensor {
  return mean(square(sub(modelFwd(m, coords, freqs), rgb)))
}

function predictFn(m: NeRFTiny, { coords, freqs }: { coords: Tensor; freqs: Tensor }): Tensor {
  return modelFwd(m, coords, freqs)
}

const PIXEL_SCALE = 4

function App() {
  const [status, setStatus]   = useState('initializing…')
  const [ready, setReady]     = useState(false)
  const [running, setRunning] = useState(false)

  // Long-lived (non-render) state lives in refs so updates don't trigger re-renders.
  const trainRef    = useRef<CompiledTraining<NeRFTiny> | null>(null)
  const inferRef       = useRef<CompiledForward<NeRFTiny> | null>(null)
  const targetRgbRef   = useRef<Float32Array | null>(null)
  const runningRef     = useRef(false)
  const stepRef        = useRef(0)

  // Canvas refs.
  const targetCanvasRef = useRef<HTMLCanvasElement>(null)
  const reconCanvasRef  = useRef<HTMLCanvasElement>(null)

  // Offscreen 64×64 buffer for pixel writes; drawImage upscales to visible canvas.
  const smallRef    = useRef<HTMLCanvasElement | null>(null)
  const smallCtxRef = useRef<CanvasRenderingContext2D | null>(null)
  const imgDataRef  = useRef<ImageData | null>(null)

  function rgbToCanvas(rgb: Float32Array, dstCanvas: HTMLCanvasElement) {
    const imgData = imgDataRef.current!
    const smallCtx = smallCtxRef.current!
    const small = smallRef.current!
    for (let i = 0; i < N_PIXELS; i++) {
      const off = i * 4
      imgData.data[off]     = Math.max(0, Math.min(255, Math.round(rgb[i * 3]!     * 255)))
      imgData.data[off + 1] = Math.max(0, Math.min(255, Math.round(rgb[i * 3 + 1]! * 255)))
      imgData.data[off + 2] = Math.max(0, Math.min(255, Math.round(rgb[i * 3 + 2]! * 255)))
      imgData.data[off + 3] = 255
    }
    smallCtx.putImageData(imgData, 0, 0)
    const dstCtx = dstCanvas.getContext('2d')!
    dstCtx.drawImage(small, 0, 0, dstCanvas.width, dstCanvas.height)
  }

  function nextBatch() {
    const targetRgb = targetRgbRef.current!
    const coords = new Float32Array(BATCH_SIZE * 2)
    const rgb = new Float32Array(BATCH_SIZE * 3)
    for (let b = 0; b < BATCH_SIZE; b++) {
      const idx = Math.floor(Math.random() * N_PIXELS)
      coords[b * 2]     = GRID_COORDS[idx * 2]!
      coords[b * 2 + 1] = GRID_COORDS[idx * 2 + 1]!
      rgb[b * 3]     = targetRgb[idx * 3]!
      rgb[b * 3 + 1] = targetRgb[idx * 3 + 1]!
      rgb[b * 3 + 2] = targetRgb[idx * 3 + 2]!
    }
    return { coords, rgb, freqs: FREQS }
  }

  async function renderReconstruction() {
    const infer = inferRef.current
    if (!infer) return
    const r = await infer.run({ coords: GRID_COORDS, freqs: FREQS })
    if (r.kind === 'completed') rgbToCanvas(r.output, reconCanvasRef.current!)
  }

  async function runTraining() {
    let lastRender = 0
    let lastLoss = 0
    while (runningRef.current && trainRef.current) {
      const sr = await trainRef.current.step(nextBatch())
      if (sr.kind !== 'completed') return
      lastLoss = sr.loss
      stepRef.current += 1
      if (!Number.isFinite(lastLoss)) {
        setStatus(`step ${stepRef.current}: loss is ${lastLoss} — NaN, aborting.`)
        runningRef.current = false
        setRunning(false)
        return
      }
      const now = Date.now()
      if (now - lastRender > 250) {
        lastRender = now
        setStatus(`step ${stepRef.current}  loss ${lastLoss.toFixed(5)}`)
        await renderReconstruction()
      }
      if (stepRef.current % 4 === 0) await new Promise(r => setTimeout(r, 0))
    }
  }

  async function buildGraphs() {
    setStatus('compiling…')
    const t0 = performance.now()
    const model = new NeRFTiny()
    const train = await compile({
      model,
      loss: lossFn,
      optimizer: { kind: 'adam', lr: 1e-3 },
      inputs: {
        coords: [BATCH_SIZE, 2],
        rgb:    [BATCH_SIZE, 3],
        freqs:  [L_FREQS],
      },
    })
    const infer = await train.attach({
      forward: predictFn,
      inputs: {
        coords: [N_PIXELS, 2],
        freqs:  [L_FREQS],
      },
    })
    trainRef.current = train
    inferRef.current = infer
    stepRef.current = 0
    setStatus(`compiled (${train.kernels.length} kernels, ${(performance.now() - t0).toFixed(0)} ms)`)
  }

  function makeDefaultImage(): Float32Array {
    const c = document.createElement('canvas')
    c.width = IMG_W
    c.height = IMG_H
    const ctx = c.getContext('2d')!
    const grad = ctx.createRadialGradient(IMG_W / 2, IMG_H / 2, 4, IMG_W / 2, IMG_H / 2, IMG_W * 0.7)
    grad.addColorStop(0, '#f97316')
    grad.addColorStop(1, '#1e293b')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, IMG_W, IMG_H)
    ctx.fillStyle = '#ffffff'
    ctx.font = 'bold 10px ui-sans-serif, system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText('tensorgrad', IMG_W / 2, IMG_H / 2)
    const px = ctx.getImageData(0, 0, IMG_W, IMG_H).data
    const rgb = new Float32Array(N_PIXELS * 3)
    for (let i = 0; i < N_PIXELS; i++) {
      rgb[i * 3]     = px[i * 4]!     / 255
      rgb[i * 3 + 1] = px[i * 4 + 1]! / 255
      rgb[i * 3 + 2] = px[i * 4 + 2]! / 255
    }
    return rgb
  }

  async function loadUploadedImage(file: File): Promise<Float32Array> {
    const bitmap = await createImageBitmap(file)
    const smallCtx = smallCtxRef.current!
    smallCtx.clearRect(0, 0, IMG_W, IMG_H)
    smallCtx.drawImage(bitmap, 0, 0, IMG_W, IMG_H)
    const px = smallCtx.getImageData(0, 0, IMG_W, IMG_H).data
    const rgb = new Float32Array(N_PIXELS * 3)
    for (let i = 0; i < N_PIXELS; i++) {
      rgb[i * 3]     = px[i * 4]!     / 255
      rgb[i * 3 + 1] = px[i * 4 + 1]! / 255
      rgb[i * 3 + 2] = px[i * 4 + 2]! / 255
    }
    return rgb
  }

  async function boot() {
    if (!isWebGPUAvailable()) {
      setStatus('WebGPU not available. Try Chrome 113+ or Safari 17.4+.')
      return
    }
    const rgb = makeDefaultImage()
    targetRgbRef.current = rgb
    rgbToCanvas(rgb, targetCanvasRef.current!)
    await buildGraphs()
    await renderReconstruction()
    setReady(true)
  }

  // Mount: size canvases, build offscreen buffer, boot.
  useEffect(() => {
    const target = targetCanvasRef.current!
    const recon  = reconCanvasRef.current!
    target.width  = recon.width  = IMG_W * PIXEL_SCALE
    target.height = recon.height = IMG_H * PIXEL_SCALE
    target.getContext('2d')!.imageSmoothingEnabled = false
    recon.getContext('2d')!.imageSmoothingEnabled  = false

    const small = document.createElement('canvas')
    small.width = IMG_W
    small.height = IMG_H
    const smallCtx = small.getContext('2d')!
    smallRef.current    = small
    smallCtxRef.current = smallCtx
    imgDataRef.current  = smallCtx.createImageData(IMG_W, IMG_H)

    boot().catch((e: unknown) => {
      const msg = (e as { message?: string })?.message ?? String(e)
      setStatus(`error: ${msg}`)
      console.error(e)
    })
  }, [])

  // Event handlers.
  const onTrain = () => {
    if (!trainRef.current || runningRef.current) return
    runningRef.current = true
    setRunning(true)
    void runTraining().then(() => { setRunning(false) })
  }

  const onStop = () => {
    runningRef.current = false
    setRunning(false)
  }

  const onReset = async () => {
    const train = trainRef.current
    if (!train) return
    const wasRunning = runningRef.current
    runningRef.current = false
    setRunning(false)
    await new Promise<void>(r => setTimeout(r, 0))
    await train.reset()
    stepRef.current = 0
    await renderReconstruction()
    setStatus(`weights re-initialized (seed ${train.seed})`)
    if (wasRunning) {
      runningRef.current = true
      setRunning(true)
      void runTraining().then(() => { setRunning(false) })
    }
  }

  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const wasRunning = runningRef.current
    runningRef.current = false
    setRunning(false)
    await new Promise<void>(r => setTimeout(r, 0))
    const rgb = await loadUploadedImage(file)
    targetRgbRef.current = rgb
    rgbToCanvas(rgb, targetCanvasRef.current!)
    if (trainRef.current) {
      await trainRef.current.reset()
      stepRef.current = 0
      await renderReconstruction()
    }
    e.target.value = ''
    setStatus(`loaded ${file.name} (${rgb.length / 3} px) — weights reset`)
    if (wasRunning) {
      runningRef.current = true
      setRunning(true)
      void runTraining().then(() => { setRunning(false) })
    }
  }

  return (
    <div className="page">
      <h1>NeRF-tiny</h1>
      <p className="subtitle">
        A 4-layer MLP learns the mapping <code>(x, y) → (r, g, b)</code> for one
        64×64 image. Watch the reconstruction sharpen as the network fits the
        target. The headline trick is sinusoidal positional encoding —
        <code>sin</code> / <code>cos</code> at <em>L</em> frequency bands — without
        which a plain MLP can only fit the low-frequency part of the signal.
      </p>

      <div id="status">{status}</div>

      <div className="controls">
        <button className="primary" disabled={!ready || running} onClick={onTrain}>Train</button>
        <button disabled={!running} onClick={onStop}>Stop</button>
        <button disabled={!ready} onClick={onReset}>Reset weights</button>
        <label className="upload">
          <span>Upload image…</span>
          <input type="file" accept="image/*" disabled={!ready} onChange={onUpload} />
        </label>
      </div>

      <div className="canvases">
        <figure>
          <canvas ref={targetCanvasRef} />
          <figcaption>target</figcaption>
        </figure>
        <figure>
          <canvas ref={reconCanvasRef} />
          <figcaption>reconstruction</figcaption>
        </figure>
      </div>

      <p className="help">
        Architecture: positional encoding (<code>L = 8</code> bands, 32 features)
        → 3 × Linear(64) + ReLU → Linear(3) + sigmoid. Adam, MSE loss in pixel
        space, batches of 1024 random pixels. ~10K parameters; converges to a
        recognisable image in a few seconds.
      </p>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<App />)

```
**styles.css**

```css
:root {
  color-scheme: light;
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #666666;
  --border: #dddddd;
  --accent: #0d9488;
  --accent-fg: #ffffff;
  --btn-bg: #ffffff;
  --btn-border: #bbbbbb;
  --canvas-bg: #000000;
}

html[data-theme="dark"] {
  color-scheme: dark;
  --bg: #1c1c1c;
  --fg: #ececec;
  --muted: #a0a0a0;
  --border: #333333;
  --accent: #14b8a6;
  --accent-fg: #0f1f1d;
  --btn-bg: #2a2a2a;
  --btn-border: #444444;
  --canvas-bg: #000000;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.page {
  max-width: 900px;
  margin: 2rem auto;
  padding: 0 1rem;
}

h1 {
  font-size: 1.4rem;
  margin-bottom: 0.25rem;
  color: var(--accent);
}

.subtitle {
  color: var(--muted);
  margin-top: 0;
  margin-bottom: 1.5rem;
}

#status {
  font-family: ui-monospace, monospace;
  font-size: 0.9rem;
  color: var(--muted);
  background: var(--btn-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.6rem 0.9rem;
  min-height: 1.4em;
  margin-bottom: 1rem;
}

.controls {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}

button, label.upload {
  font: inherit;
  padding: 0.5rem 1rem;
  font-size: 0.95rem;
  cursor: pointer;
  background: var(--btn-bg);
  color: var(--fg);
  border: 1px solid var(--btn-border);
  border-radius: 4px;
}

button.primary {
  background: var(--accent);
  color: var(--accent-fg);
  border-color: var(--accent);
}

button:disabled, label.upload:has(input:disabled) {
  opacity: 0.5;
  cursor: default;
}

label.upload input { display: none; }

.canvases {
  display: flex;
  gap: 1.5rem;
  align-items: flex-start;
  flex-wrap: wrap;
}

.canvases figure { margin: 0; }

.canvases figcaption {
  font-size: 0.85rem;
  color: var(--muted);
  margin-top: 0.4rem;
  text-align: center;
}

canvas {
  display: block;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--canvas-bg);
  image-rendering: pixelated;
  max-width: 100%;
  height: auto;
}

.help {
  color: var(--muted);
  font-size: 0.9rem;
  margin-top: 1rem;
}

code {
  font-family: ui-monospace, monospace;
  font-size: 0.9em;
  background: var(--border);
  padding: 0.05rem 0.3rem;
  border-radius: 3px;
}

```
**index.html**

```html
<div id="root"></div>

```
**config.json**

```json
{
  "description": "NeRF is a coordinate-network MLP learns the (x, y) → RGB mapping for one 64×64 image, using sinusoidal positional encoding so it can fit high-frequency detail.",
  "dependencies": {
    "tensorgrad": "^0.3.0",
    "react": "^19.2.6",
    "react-dom": "^19.2.6"
  }
}
```