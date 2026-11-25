import { describe, expect, it } from 'vitest';
import {
  asymmetricQuantizedCosineSimilarity,
  cosineSimilarity,
  dequantizeInt8,
  HnswIndex,
  l2Norm,
  normalizeVector,
  quantizeInt8,
} from '../src/index.js';

function makeDeterministicVectors(count: number, dims: number, seed = 42): Float32Array[] {
  let state = seed >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state / 0xffffffff) * 2 - 1;
  };

  const vectors: Float32Array[] = [];
  for (let i = 0; i < count; i++) {
    const v = new Float32Array(dims);
    for (let d = 0; d < dims; d++) {
      v[d] = next();
    }
    vectors.push(normalizeVector(v));
  }
  return vectors;
}

describe('HnswIndex & int8 quantization', () => {
  it('quantizeInt8 preserves cosine direction and shrinks payload 4x', () => {
    const [vec] = makeDeterministicVectors(1, 128, 7);
    const { q, scale } = quantizeInt8(vec!);
    expect(q.byteLength).toBe(128);
    expect(vec!.byteLength).toBe(512);

    const restored = dequantizeInt8(q, scale);
    expect(cosineSimilarity(vec!, restored)).toBeGreaterThan(0.995);
    expect(asymmetricQuantizedCosineSimilarity(vec!, l2Norm(vec!), q)).toBeGreaterThan(0.995);
  });

  it('achieves high recall@10 with int8 quantization vs exact brute-force kNN', () => {
    const dims = 64;
    const n = 220;
    const corpus = makeDeterministicVectors(n, dims, 101);
    const queries = makeDeterministicVectors(12, dims, 999);

    const hnswInt8 = new HnswIndex({
      dimensions: dims,
      M: 16,
      efConstruction: 120,
      efSearch: 80,
      quantization: 'int8',
      seed: 1234,
    });
    const hnswF32 = new HnswIndex({
      dimensions: dims,
      M: 16,
      efConstruction: 120,
      efSearch: 80,
      quantization: 'f32',
      seed: 1234,
    });

    for (let i = 0; i < n; i++) {
      hnswInt8.insert(`doc_${i}`, corpus[i]!);
      hnswF32.insert(`doc_${i}`, corpus[i]!);
    }

    let recallSumInt8 = 0;
    for (const q of queries) {
      const exactTop10 = corpus
        .map((v, idx) => ({ id: `doc_${idx}`, sim: cosineSimilarity(q, v) }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, 10)
        .map((x) => x.id);
      const exactSet = new Set(exactTop10);

      const annHits = hnswInt8.search(q, 10);
      const overlap = annHits.filter((h) => exactSet.has(h.id)).length;
      recallSumInt8 += overlap / 10;
    }

    const avgRecall = recallSumInt8 / queries.length;
    expect(avgRecall).toBeGreaterThanOrEqual(0.88);
    // Serialized int8 index is substantially smaller than f32
    expect(hnswInt8.serialize().byteLength).toBeLessThan(hnswF32.serialize().byteLength);
  });

  it('roundtrips binary serialization and supports removal + filtered search', () => {
    const dims = 32;
    const corpus = makeDeterministicVectors(40, dims, 55);
    const index = new HnswIndex({ dimensions: dims, quantization: 'int8', seed: 77 });
    corpus.forEach((v, i) => index.insert(`id_${i}`, v));

    expect(index.remove('id_0')).toBe(true);
    expect(index.has('id_0')).toBe(false);
    expect(index.size).toBe(39);

    const bytes = index.serialize();
    const restored = HnswIndex.deserialize(bytes);
    expect(restored.size).toBe(39);
    expect(restored.has('id_0')).toBe(false);
    expect(restored.has('id_1')).toBe(true);

    const onlyEven = restored.search(corpus[2]!, 5, (id) => Number(id.split('_')[1]) % 2 === 0);
    expect(onlyEven.length).toBe(5);
    expect(onlyEven[0]?.id).toBe('id_2');
    for (const hit of onlyEven) {
      expect(Number(hit.id.split('_')[1]) % 2).toBe(0);
    }
  });
});
