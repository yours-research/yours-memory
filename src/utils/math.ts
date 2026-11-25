/** Vector math and symmetric int8 scalar quantization utilities. */

export interface QuantizedInt8Vector {
  q: Int8Array;
  scale: number;
}

export function l2Norm(vec: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    const v = vec[i] ?? 0;
    sum += v * v;
  }
  return Math.sqrt(sum);
}

export function normalizeVector(vec: ArrayLike<number>): Float32Array {
  const out = new Float32Array(vec.length);
  const norm = l2Norm(vec);
  if (norm <= 1e-12) {
    return out;
  }
  const inv = 1 / norm;
  for (let i = 0; i < vec.length; i++) {
    out[i] = (vec[i] ?? 0) * inv;
  }
  return out;
}

export function dotProduct(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return sum;
}

export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    const va = a[i] ?? 0;
    const vb = b[i] ?? 0;
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom <= 1e-12) return 0;
  const sim = dot / denom;
  return sim > 1 ? 1 : sim < -1 ? -1 : sim;
}

export function cosineDistance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return 1 - cosineSimilarity(a, b);
}

/**
 * Symmetric int8 scalar quantization: maps [-maxAbs, +maxAbs] to [-127, +127].
 * Provides ~4x memory and OPFS footprint reduction versus Float32Array.
 */
export function quantizeInt8(vec: Float32Array): QuantizedInt8Vector {
  let maxAbs = 0;
  for (let i = 0; i < vec.length; i++) {
    const abs = Math.abs(vec[i] ?? 0);
    if (abs > maxAbs) maxAbs = abs;
  }
  const q = new Int8Array(vec.length);
  if (maxAbs <= 1e-12) {
    return { q, scale: 1 };
  }
  const scale = maxAbs / 127;
  const invScale = 127 / maxAbs;
  for (let i = 0; i < vec.length; i++) {
    const scaled = Math.round((vec[i] ?? 0) * invScale);
    q[i] = scaled > 127 ? 127 : scaled < -127 ? -127 : scaled;
  }
  return { q, scale };
}

export function dequantizeInt8(q: Int8Array, scale: number): Float32Array {
  const out = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) {
    out[i] = (q[i] ?? 0) * scale;
  }
  return out;
}

/**
 * Fast cosine similarity between two int8-quantized vectors using integer MACs.
 * Because scale factors cancel out in cosine normalization, we only need integer
 * arithmetic on the Int8Arrays!
 */
export function quantizedCosineSimilarity(a: Int8Array, b: Int8Array): number {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    const va = a[i] ?? 0;
    const vb = b[i] ?? 0;
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  if (normA === 0 || normB === 0) return 0;
  const sim = dot / (Math.sqrt(normA) * Math.sqrt(normB));
  return sim > 1 ? 1 : sim < -1 ? -1 : sim;
}

/** Asymmetric cosine similarity: float32 query against int8 stored vector. */
export function asymmetricQuantizedCosineSimilarity(
  query: Float32Array,
  qNorm: number,
  stored: Int8Array
): number {
  if (qNorm <= 1e-12) return 0;
  const len = Math.min(query.length, stored.length);
  let dot = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    const vb = stored[i] ?? 0;
    dot += (query[i] ?? 0) * vb;
    normB += vb * vb;
  }
  if (normB === 0) return 0;
  const sim = dot / (qNorm * Math.sqrt(normB));
  return sim > 1 ? 1 : sim < -1 ? -1 : sim;
}

/** FNV-1a 32-bit hash for deterministic feature hashing and canonical keys. */
export function fnv1a32(input: string, seed = 0x811c9dc5): number {
  let hash = seed >>> 0;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}
