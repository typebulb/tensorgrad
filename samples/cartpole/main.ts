// CartPole REINFORCE — a tiny policy MLP learns to balance an inverted
// pendulum via discounted-return-weighted log-policy gradients. The classic
// RL "hello world": no supervised target, the gradient comes from the
// agent's own actions weighted by how well things went after taking them.
//
// Architecture: state [4] → tanh-MLP(16) → logits [2]. Two actions: push
// the cart left or right. Reward is +1 per step until the pole tips past
// 12° or the cart drives off the ±2.4 track.
//
// Rollout: K = 16 parallel environments stepped in lockstep for MAX_T
// timesteps. After each rollout we compute discounted returns per
// trajectory, normalize them across the batch (variance reduction), and
// take one Adam step. Static shapes — episodes that end early are masked
// out of the loss rather than truncating the graph.
//
// Visualisation: env[0] is drawn live on a canvas while the rollout
// proceeds — you'll see the cart wobble and the pole fall over and over
// at first, then start to stabilize as the policy improves.
//
// File layout: ML + app logic at the top, UI at the bottom. Same convention
// as the other samples.

import {
  Module, compile, trainingSpec, forwardSpec, isWebGPUAvailable, nn,
  mul, sum,
  tanh, oneHot, logSoftmax, softmax,
  type Tensor, type CompiledTraining, type CompiledForward,
} from 'tensorgrad'

// ============================================================================
//                          MODEL / TRAINING
// ============================================================================

const K = 16                                // parallel envs per rollout
const MAX_T = 200                           // max steps per rollout
const STATE_DIM = 4
const N_ACTIONS = 2
const HIDDEN = 16
const GAMMA = 0.99                          // reward discount
const LR = 5e-3

// CartPole physics constants (OpenAI gym defaults).
const GRAVITY = 9.8
const MASSCART = 1.0
const MASSPOLE = 0.1
const POLE_HALF_LENGTH = 0.5
const TOTAL_MASS = MASSCART + MASSPOLE
const POLEMASS_LENGTH = MASSPOLE * POLE_HALF_LENGTH
const FORCE_MAG = 10.0
const TAU = 0.02                            // physics dt (seconds)
const X_BOUND = 2.4
const THETA_BOUND = 12 * Math.PI / 180

// ---------------------------------------------------------------------------
// Environment: pure CPU. Stateful struct, semi-implicit Euler step.
// ---------------------------------------------------------------------------

interface Env {
  x: number; xDot: number; theta: number; thetaDot: number
  done: boolean
}

function resetEnv(): Env {
  return {
    x:        (Math.random() - 0.5) * 0.1,
    xDot:     (Math.random() - 0.5) * 0.1,
    theta:    (Math.random() - 0.5) * 0.1,
    thetaDot: (Math.random() - 0.5) * 0.1,
    done: false,
  }
}

function stepEnv(e: Env, action: number): Env {
  if (e.done) return e
  const force = action === 1 ? FORCE_MAG : -FORCE_MAG
  const ct = Math.cos(e.theta)
  const st = Math.sin(e.theta)
  // Standard cartpole equations (Florian, 2007).
  const temp = (force + POLEMASS_LENGTH * e.thetaDot * e.thetaDot * st) / TOTAL_MASS
  const thetaAcc = (GRAVITY * st - ct * temp) / (POLE_HALF_LENGTH * (4 / 3 - MASSPOLE * ct * ct / TOTAL_MASS))
  const xAcc = temp - POLEMASS_LENGTH * thetaAcc * ct / TOTAL_MASS
  const x        = e.x        + TAU * e.xDot
  const xDot     = e.xDot     + TAU * xAcc
  const theta    = e.theta    + TAU * e.thetaDot
  const thetaDot = e.thetaDot + TAU * thetaAcc
  const done = Math.abs(x) > X_BOUND || Math.abs(theta) > THETA_BOUND
  return { x, xDot, theta, thetaDot, done }
}

// ---------------------------------------------------------------------------
// Policy model
// ---------------------------------------------------------------------------

class Policy extends Module {
  l1 = new nn.Linear(STATE_DIM, HIDDEN)
  l2 = new nn.Linear(HIDDEN, N_ACTIONS)
}

function policyLogits(m: Policy, state: Tensor): Tensor {
  return m.l2.fwd(tanh(m.l1.fwd(state)))
}

// Training loss: -E[log π(a | s) · normalized_return · mask]. Masked steps
// (past episode-done) contribute zero. We divide by total batch size (MAX_T·K)
// rather than the unmasked count — biases the gradient magnitude slightly
// but doesn't change direction, and Adam absorbs the rescaling.
function lossFn(
  m: Policy,
  { states, actions, returns, mask }:
    { states: Tensor; actions: Tensor; returns: Tensor; mask: Tensor },
): Tensor {
  const logits = policyLogits(m, states)              // [N, A]
  const logProbs = logSoftmax(logits, -1)              // [N, A]
  // Per-step log-prob of the action that was actually taken.
  const taken = sum(mul(logProbs, oneHot(actions, N_ACTIONS, 'f32')), -1)  // [N]
  return mul(sum(mul(mul(taken, returns), mask)), -1 / (MAX_T * K))
}

// Inference: softmax over logits, sampling happens CPU-side. The proxy is
// polymorphic over batch — same graph serves the K-wide rollout and the
// (unused here, but easy to add) B=1 case.
function predictFn(m: Policy, { state }: { state: Tensor }): Tensor {
  return softmax(policyLogits(m, state), -1)
}

// ---------------------------------------------------------------------------
// State + lifecycle
// ---------------------------------------------------------------------------

let train: CompiledTraining<Policy> | null = null
let infer: CompiledForward<Policy> | null = null
let running = false
let rolloutCount = 0
let recentEpLens: number[] = []                       // rolling window for status

let onStatus: (msg: string) => void = () => {}
let onEnvFrame: (e: Env) => void = () => {}

// Sample an action ∈ {0..n-1} from a slice of the K-wide softmax output.
function sampleCategorical(probs: Float32Array, offset: number, n: number): number {
  let u = Math.random()
  for (let i = 0; i < n; i++) {
    u -= probs[offset + i]!
    if (u < 0) return i
  }
  return n - 1
}

// One K-parallel rollout. Returns the flattened training batch plus per-env
// episode lengths (for the status display).
async function rollout(): Promise<{
  states: Float32Array; actions: Int32Array; returns: Float32Array; mask: Float32Array
  epLens: number[]
}> {
  if (!infer) throw new Error('rollout: no inference graph')
  const envs: Env[] = Array.from({ length: K }, resetEnv)
  const stateBuf  = new Float32Array(K * STATE_DIM)
  const statesAll = new Float32Array(MAX_T * K * STATE_DIM)
  const actionsAll = new Int32Array(MAX_T * K)
  const rewardsAll = new Float32Array(MAX_T * K)
  const maskAll    = new Float32Array(MAX_T * K)
  const epLens = new Array<number>(K).fill(0)

  for (let t = 0; t < MAX_T; t++) {
    // Pack the K current states.
    for (let k = 0; k < K; k++) {
      const o = k * STATE_DIM
      stateBuf[o]     = envs[k]!.x
      stateBuf[o + 1] = envs[k]!.xDot
      stateBuf[o + 2] = envs[k]!.theta
      stateBuf[o + 3] = envs[k]!.thetaDot
    }
    const probsR = await infer.run({ state: stateBuf })  // [K, A]
    if (probsR.kind === 'aborted') break
    const probs = probsR.output

    for (let k = 0; k < K; k++) {
      const idx = (t * K + k)
      if (envs[k]!.done) continue
      const a = sampleCategorical(probs, k * N_ACTIONS, N_ACTIONS)
      statesAll.set(stateBuf.subarray(k * STATE_DIM, (k + 1) * STATE_DIM), idx * STATE_DIM)
      actionsAll[idx] = a
      rewardsAll[idx] = 1
      maskAll[idx]    = 1
      epLens[k]!++
      envs[k] = stepEnv(envs[k]!, a)
    }

    // Live-render env[0] as the rollout plays out.
    onEnvFrame(envs[0]!)
    if (t % 4 === 0) await new Promise(r => setTimeout(r, 0))
  }

  // Discounted returns per trajectory, reverse scan.
  const returnsAll = new Float32Array(MAX_T * K)
  for (let k = 0; k < K; k++) {
    let acc = 0
    for (let t = MAX_T - 1; t >= 0; t--) {
      const idx = t * K + k
      if (maskAll[idx] === 0) continue
      acc = rewardsAll[idx]! + GAMMA * acc
      returnsAll[idx] = acc
    }
  }

  // Normalize returns over unmasked entries: subtract mean, divide by std.
  // Standard REINFORCE variance reduction — same gradient direction, smaller
  // variance per step.
  let count = 0, sum1 = 0
  for (let i = 0; i < MAX_T * K; i++) if (maskAll[i]) { sum1 += returnsAll[i]!; count++ }
  const meanR = count > 0 ? sum1 / count : 0
  let varSum = 0
  for (let i = 0; i < MAX_T * K; i++) if (maskAll[i]) varSum += (returnsAll[i]! - meanR) ** 2
  const stdR = Math.sqrt(varSum / Math.max(1, count)) + 1e-8
  for (let i = 0; i < MAX_T * K; i++) if (maskAll[i]) returnsAll[i] = (returnsAll[i]! - meanR) / stdR

  return { states: statesAll, actions: actionsAll, returns: returnsAll, mask: maskAll, epLens }
}

async function runTraining(): Promise<void> {
  while (running && train) {
    const r = await rollout()
    const stepR = await train.step({
      states: r.states, actions: r.actions, returns: r.returns, mask: r.mask,
    })
    if (stepR.kind === 'aborted') return
    const loss = stepR.loss
    if (!Number.isFinite(loss)) {
      onStatus(`rollout ${rolloutCount}: loss is ${loss} — NaN, aborting.`)
      running = false
      return
    }
    rolloutCount++
    // Rolling-window mean of episode lengths over the last ~5 rollouts
    // (5 * K = 80 episodes). Smoother signal than a single rollout.
    for (const l of r.epLens) recentEpLens.push(l)
    while (recentEpLens.length > 5 * K) recentEpLens.shift()
    const avg = recentEpLens.reduce((a, b) => a + b, 0) / recentEpLens.length
    onStatus(`rollout ${rolloutCount}  loss ${loss.toFixed(4)}  avg ep len ${avg.toFixed(1)} / ${MAX_T}`)
  }
}

async function buildGraphs(): Promise<void> {
  onStatus('compiling…')
  const t0 = performance.now()
  const model = new Policy()
  train = await compile(trainingSpec({
    model,
    loss: lossFn,
    optimizer: { kind: 'adam', lr: LR },
    inputs: {
      states:  [MAX_T * K, STATE_DIM],
      actions: { shape: [MAX_T * K], dtype: 'i32' },
      returns: [MAX_T * K],
      mask:    [MAX_T * K],
    },
  }))
  infer = await train.attach(forwardSpec({
    model,
    forward: predictFn,
    inputs: { state: [K, STATE_DIM] },
  }))
  rolloutCount = 0
  recentEpLens = []
  onStatus(`compiled (${train.kernels.length} kernels, ${(performance.now() - t0).toFixed(0)} ms)`)
}

function startTraining(): void {
  if (running) return
  running = true
  void runTraining()
}

function stopTraining(): void {
  running = false
}

async function resetWeights(): Promise<void> {
  if (!train) return
  const wasRunning = running
  running = false
  await new Promise<void>(r => setTimeout(r, 0))
  await train.reset()
  rolloutCount = 0
  recentEpLens = []
  onStatus(`weights re-initialized (seed ${train.seed})`)
  if (wasRunning) { running = true; void runTraining() }
}

// ============================================================================
//                                   UI
// ============================================================================

const statusEl  = document.getElementById('status') as HTMLDivElement
const trainBtn  = document.getElementById('train')  as HTMLButtonElement
const stopBtn   = document.getElementById('stop')   as HTMLButtonElement
const resetBtn  = document.getElementById('reset')  as HTMLButtonElement
const canvas    = document.getElementById('cart-canvas') as HTMLCanvasElement

const CANVAS_W = 640
const CANVAS_H = 200
canvas.width  = CANVAS_W
canvas.height = CANVAS_H
const ctx = canvas.getContext('2d')!

// World-space → screen mapping. X_BOUND is ±2.4; we map to ±(half width − margin).
const SCREEN_MARGIN = 40
const TRACK_Y = CANVAS_H - 50
const CART_W = 40
const CART_H = 20
const POLE_LEN_SCREEN = 80   // visual length of pole

function worldToScreenX(xWorld: number): number {
  const halfTrack = CANVAS_W / 2 - SCREEN_MARGIN
  return CANVAS_W / 2 + (xWorld / X_BOUND) * halfTrack
}

function renderEnv(e: Env): void {
  ctx.fillStyle = '#fafafa'
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H)

  // Track + boundary markers.
  ctx.strokeStyle = '#bbb'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(0, TRACK_Y)
  ctx.lineTo(CANVAS_W, TRACK_Y)
  ctx.stroke()
  ctx.strokeStyle = '#ddd'
  ctx.lineWidth = 1
  for (const xw of [-X_BOUND, X_BOUND]) {
    const sx = worldToScreenX(xw)
    ctx.beginPath()
    ctx.moveTo(sx, TRACK_Y - 10)
    ctx.lineTo(sx, TRACK_Y + 10)
    ctx.stroke()
  }

  // Cart.
  const cx = worldToScreenX(e.x)
  ctx.fillStyle = e.done ? '#ef4444' : '#1e293b'
  ctx.fillRect(cx - CART_W / 2, TRACK_Y - CART_H, CART_W, CART_H)

  // Pole. theta = 0 points up; positive theta tips right.
  const px = cx + Math.sin(e.theta) * POLE_LEN_SCREEN
  const py = (TRACK_Y - CART_H) - Math.cos(e.theta) * POLE_LEN_SCREEN
  ctx.strokeStyle = '#f97316'
  ctx.lineWidth = 6
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(cx, TRACK_Y - CART_H)
  ctx.lineTo(px, py)
  ctx.stroke()
}

onStatus = (msg) => { statusEl.textContent = msg }
onEnvFrame = (e) => renderEnv(e)

trainBtn.addEventListener('click', () => {
  trainBtn.disabled = true
  stopBtn.disabled = false
  startTraining()
})

stopBtn.addEventListener('click', () => {
  stopTraining()
  trainBtn.disabled = false
  stopBtn.disabled = true
})

resetBtn.addEventListener('click', () => { void resetWeights() })

// Show an idle initial frame so the canvas isn't blank before training starts.
renderEnv(resetEnv())

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  if (!isWebGPUAvailable()) {
    onStatus('WebGPU not available. Try Chrome 113+ or Safari 17.4+.')
    return
  }
  await buildGraphs()
  trainBtn.disabled = false
  resetBtn.disabled = false
}

boot().catch((e: unknown) => {
  const msg = (e as { message?: string })?.message ?? String(e)
  onStatus(`error: ${msg}`)
  console.error(e)
})
