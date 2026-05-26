// safetensors loader tests — pure TS, no GPU. Builds in-memory safetensors
// containers and round-trips them through loadSafetensors, plus the malformed /
// non-f32 error paths.
//
// Run with:  tsx test/safetensors.ts

import { loadSafetensors } from '../src/index.js'
import { section, assert, assertShape, assertThrows, done } from './_assert.js'

declare const TextEncoder: { new (): { encode(s: string): Uint8Array } }

/** Build a valid safetensors ArrayBuffer from named f32 tensors. Mirrors the
 *  reference writer: 8-byte LE u64 header length, JSON header, then the blob. */
function buildSafetensors(
  entries: { name: string; shape: number[]; data: Float32Array }[],
  metadata?: Record<string, string>,
): ArrayBuffer {
  const header: Record<string, unknown> = {}
  if (metadata) header['__metadata__'] = metadata
  let offset = 0
  for (const e of entries) {
    const bytes = e.data.length * 4
    header[e.name] = { dtype: 'F32', shape: e.shape, data_offsets: [offset, offset + bytes] }
    offset += bytes
  }
  const headerJson = new TextEncoder().encode(JSON.stringify(header))
  const blobBytes = offset
  const buf = new ArrayBuffer(8 + headerJson.length + blobBytes)
  const view = new DataView(buf)
  view.setBigUint64(0, BigInt(headerJson.length), true)
  new Uint8Array(buf, 8, headerJson.length).set(headerJson)
  // Byte-wise copy (not a Float32Array view): the blob start (8 + headerLen)
  // isn't 4-aligned here, so an f32 view would throw. Real writers pad the
  // header to keep the blob aligned; loadSafetensors copies to tolerate either.
  let o = 8 + headerJson.length
  for (const e of entries) {
    const src = new Uint8Array(e.data.buffer, e.data.byteOffset, e.data.length * 4)
    new Uint8Array(buf, o, src.length).set(src)
    o += src.length
  }
  return buf
}

function arraysEqual(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

// 1. Round-trips multiple tensors with correct values + shapes.
section('safetensors: round-trip')
{
  const w = new Float32Array([1, 2, 3, 4, 5, 6])
  const b = new Float32Array([0.5, -0.5])
  const buf = buildSafetensors([
    { name: 'l1.W', shape: [3, 2], data: w },
    { name: 'l1.b', shape: [2], data: b },
  ])
  const { tensors, shapes, skipped } = loadSafetensors(buf)
  assert(arraysEqual(tensors['l1.W']!, w), 'l1.W values round-trip')
  assert(arraysEqual(tensors['l1.b']!, b), 'l1.b values round-trip')
  assertShape(shapes['l1.W']!, [3, 2], 'l1.W shape')
  assertShape(shapes['l1.b']!, [2], 'l1.b shape')
  assert(Object.keys(tensors).length === 2, 'exactly 2 tensors parsed')
  assert(Object.keys(skipped).length === 0, 'nothing skipped for an all-f32 file')
}

// 2. __metadata__ is skipped (not surfaced as a tensor).
section('safetensors: __metadata__ skipped')
{
  const buf = buildSafetensors(
    [{ name: 'w', shape: [2], data: new Float32Array([7, 8]) }],
    { format: 'pt', note: 'hello' },
  )
  const { tensors } = loadSafetensors(buf)
  assert(!('__metadata__' in tensors), '__metadata__ not present as a tensor')
  assert(Object.keys(tensors).length === 1, 'only the real tensor parsed')
}

// 3. A scalar (shape []) is one element.
section('safetensors: scalar shape []')
{
  const buf = buildSafetensors([{ name: 's', shape: [], data: new Float32Array([42]) }])
  const { tensors, shapes } = loadSafetensors(buf)
  assert(tensors['s']!.length === 1 && tensors['s']![0] === 42, 'scalar value round-trips')
  assertShape(shapes['s']!, [], 'scalar shape is []')
}

// 4. Non-F32 FLOAT dtype throws (it's a real weight — refuse, don't drop).
section('safetensors: non-f32 float rejected')
{
  const header = { x: { dtype: 'F16', shape: [2], data_offsets: [0, 4] } }
  const headerJson = new TextEncoder().encode(JSON.stringify(header))
  const buf = new ArrayBuffer(8 + headerJson.length + 4)
  new DataView(buf).setBigUint64(0, BigInt(headerJson.length), true)
  new Uint8Array(buf, 8, headerJson.length).set(headerJson)
  assertThrows(() => loadSafetensors(buf), 'f32-only', 'F16 tensor rejected')
}

// 4b. Integer/bool dtype (BatchNorm's num_batches_tracked etc.) is SKIPPED +
//     reported, not thrown — and the f32 tensors alongside it still load.
section('safetensors: integer tensors skipped + reported')
{
  // An F32 weight [2] (8 bytes) followed by an I64 scalar counter (8 bytes).
  const wData = new Float32Array([1.5, -2.5])
  const header = {
    'bn.weight': { dtype: 'F32', shape: [2], data_offsets: [0, 8] },
    'bn.num_batches_tracked': { dtype: 'I64', shape: [], data_offsets: [8, 16] },
  }
  const hj = new TextEncoder().encode(JSON.stringify(header))
  const buf = new ArrayBuffer(8 + hj.length + 16)
  new DataView(buf).setBigUint64(0, BigInt(hj.length), true)
  new Uint8Array(buf, 8, hj.length).set(hj)
  const wbytes = new Uint8Array(wData.buffer, wData.byteOffset, 8)
  new Uint8Array(buf, 8 + hj.length, 8).set(wbytes)
  // I64 counter bytes left as zero — they must never be read as f32.
  const { tensors, shapes, skipped } = loadSafetensors(buf)
  assert(arraysEqual(tensors['bn.weight']!, wData), 'f32 tensor loads alongside an int one')
  assert(!('bn.num_batches_tracked' in tensors), 'I64 tensor not in tensors')
  assert(skipped['bn.num_batches_tracked'] === 'I64', 'I64 tensor reported in skipped with its dtype')
  assertShape(shapes['bn.num_batches_tracked']!, [], 'skipped tensor still has its shape recorded')
}

// 5. Truncated buffer (header length past EOF) throws.
section('safetensors: malformed buffers')
{
  const tooSmall = new ArrayBuffer(4)
  assertThrows(() => loadSafetensors(tooSmall), 'too small', 'sub-8-byte buffer rejected')

  const buf = new ArrayBuffer(8)
  new DataView(buf).setBigUint64(0, BigInt(9999), true)  // claims a 9999-byte header
  assertThrows(() => loadSafetensors(buf), 'exceeds buffer', 'oversized header length rejected')
}

// 6. data_offsets / shape mismatch throws (guards a bad port, not silent read).
section('safetensors: offset/shape mismatch')
{
  // Header says shape [4] (16 bytes) but data_offsets span only 8 bytes.
  const header = { x: { dtype: 'F32', shape: [4], data_offsets: [0, 8] } }
  const headerJson = new TextEncoder().encode(JSON.stringify(header))
  const buf = new ArrayBuffer(8 + headerJson.length + 8)
  new DataView(buf).setBigUint64(0, BigInt(headerJson.length), true)
  new Uint8Array(buf, 8, headerJson.length).set(headerJson)
  assertThrows(() => loadSafetensors(buf), 'byte length', 'shape/offset mismatch rejected')
}

done('test/safetensors.ts')
