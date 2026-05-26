// safetensors loader — parse the safetensors container into the flat
// `Record<string, Float32Array>` that `uploadParams` consumes. Pure TS, no deps.
//
// Format: an 8-byte little-endian u64 header length N, then an N-byte JSON
// header, then one contiguous data blob. The header maps each tensor's
// `name -> { dtype, shape, data_offsets: [start, end] }`, where the offsets are
// byte ranges into the blob (relative to the blob start, i.e. after the
// header). A reserved `__metadata__` key carries free-form string metadata, not
// a tensor.
//
// f32-only: tensorgrad's math path is f32 (see README "Constraints"), so the
// loader produces `Float32Array` directly. Two kinds of non-F32 tensor are
// handled differently:
//   - integer / bool tensors (e.g. BatchNorm's `num_batches_tracked` I64
//     counters) are checkpoint bookkeeping, not params you upload — they're
//     *skipped* and reported in `skipped`, never silently dropped.
//   - other float dtypes (F16/BF16/F64) ARE real weights; the loader *throws*
//     rather than mis-read or silently drop them. Convert to f32 offline (the
//     same self-host/convert step that bakes in the PyTorch->tensorgrad layout
//     transposes; see the transfer-learning spec). f16/bf16 dequant-on-read is
//     deferred (no consumer — the candidate backbones ship f32).

export interface SafetensorsData {
  /** F32 tensor name -> values, ready for `uploadParams` (after any
   *  per-backbone key remap / layout transform the importer applies). */
  tensors: Record<string, Float32Array>
  /** Tensor name -> shape (every tensor, including skipped ones), for
   *  verifying a port matches the target Module's param shapes. */
  shapes: Record<string, number[]>
  /** Non-F32 integer/bool tensors that were skipped, name -> dtype. Surfaced
   *  (not silently dropped) so a missing param is debuggable. Typically just
   *  the `*.num_batches_tracked` (I64) counters from BatchNorm checkpoints. */
  skipped: Record<string, string>
}

/** safetensors integer/bool dtypes — checkpoint bookkeeping, not f32 params.
 *  Skipped (and reported) rather than thrown on. */
const INTEGER_DTYPES = new Set(['I64', 'I32', 'I16', 'I8', 'U8', 'U16', 'U32', 'U64', 'BOOL'])

interface HeaderEntry {
  dtype: string
  shape: number[]
  data_offsets: [number, number]
}

/** Parse a safetensors `ArrayBuffer` (e.g. from `await (await fetch(url)).arrayBuffer()`)
 *  into f32 tensors + their shapes. Throws on a malformed container or any
 *  non-F32 tensor. */
export function loadSafetensors(buf: ArrayBuffer): SafetensorsData {
  if (buf.byteLength < 8) {
    throw new Error('loadSafetensors: buffer too small to hold the 8-byte header length')
  }
  const view = new DataView(buf)
  const headerLen = Number(view.getBigUint64(0, true))
  const headerEnd = 8 + headerLen
  if (headerEnd > buf.byteLength) {
    throw new Error(`loadSafetensors: header length ${headerLen} exceeds buffer (${buf.byteLength} bytes)`)
  }

  const headerJson = new TextDecoder().decode(new Uint8Array(buf, 8, headerLen))
  let header: Record<string, unknown>
  try {
    header = JSON.parse(headerJson)
  } catch (e) {
    throw new Error(`loadSafetensors: header is not valid JSON: ${(e as Error).message}`)
  }

  const tensors: Record<string, Float32Array> = {}
  const shapes: Record<string, number[]> = {}
  const skipped: Record<string, string> = {}
  for (const [name, raw] of Object.entries(header)) {
    if (name === '__metadata__') continue
    const entry = raw as HeaderEntry
    shapes[name] = entry.shape
    if (entry.dtype !== 'F32') {
      // Integer/bool: bookkeeping (e.g. BatchNorm counters) — skip + report.
      if (INTEGER_DTYPES.has(entry.dtype)) { skipped[name] = entry.dtype; continue }
      // Other float dtypes are real weights — refuse rather than mis-read.
      throw new Error(
        `loadSafetensors: tensor '${name}' has dtype '${entry.dtype}', but tensorgrad is f32-only. ` +
        `Convert the checkpoint to f32 offline before loading (see README "Constraints").`,
      )
    }
    const [start, end] = entry.data_offsets
    if (start > end || headerEnd + end > buf.byteLength) {
      throw new Error(`loadSafetensors: tensor '${name}' data_offsets [${start}, ${end}] out of range`)
    }
    const numel = entry.shape.reduce((a, b) => a * b, 1)
    const byteLen = end - start
    if (byteLen !== numel * 4) {
      throw new Error(
        `loadSafetensors: tensor '${name}' byte length ${byteLen} != ${numel} f32 elements * 4 ` +
        `(shape [${entry.shape.join(', ')}])`,
      )
    }
    // Copy into a fresh (4-aligned) buffer: the blob's absolute byte offset
    // isn't guaranteed to be 4-aligned, so a direct Float32Array *view* over
    // `buf` can throw. The copy also keeps the result independent of the input
    // buffer (which `uploadParams` may later transfer).
    tensors[name] = new Float32Array(buf.slice(headerEnd + start, headerEnd + end))
  }
  return { tensors, shapes, skipped }
}
