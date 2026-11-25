import type { EmbeddingProvider } from '../contracts.js';
import { fnv1a32, normalizeVector } from '../utils/math.js';
import { tokenize } from '../utils/text.js';

export interface DeterministicEmbedderOptions {
  dimensions?: number;
}

/**
 * Built-in zero-weight deterministic embedder using signed feature hashing over
 * normalized word tokens, word bigrams, and character trigrams.
 * Ideal for instant bootstrapping, offline fallback, and deterministic tests.
 */
export class DeterministicHashEmbedder implements EmbeddingProvider {
  readonly id = 'deterministic-hash-trigram-v1';
  readonly dimensions: number;

  constructor(options: DeterministicEmbedderOptions = {}) {
    this.dimensions = options.dimensions ?? 256;
    if (this.dimensions < 16) {
      throw new Error('Embedding dimensions must be >= 16');
    }
  }

  async embed(texts: string[], signal?: AbortSignal): Promise<Float32Array[]> {
    if (signal?.aborted) {
      throw new DOMException('Embedding aborted', 'AbortError');
    }
    return texts.map((text) => this.embedSingle(text));
  }

  async embedQuery(text: string, signal?: AbortSignal): Promise<Float32Array> {
    if (signal?.aborted) {
      throw new DOMException('Embedding aborted', 'AbortError');
    }
    return this.embedSingle(text);
  }

  private embedSingle(text: string): Float32Array {
    const vec = new Float32Array(this.dimensions);
    const tokens = tokenize(text, { removeStopwords: true, stem: true, splitIdentifiers: true });

    // 1. Word unigrams (strongest weight)
    for (const token of tokens) {
      this.projectFeature(vec, `w:${token}`, 2.0);
      // Character trigrams within each token for morphological / typo resilience
      const padded = `^${token}$`;
      for (let i = 0; i <= padded.length - 3; i++) {
        this.projectFeature(vec, `c:${padded.slice(i, i + 3)}`, 0.75);
      }
    }

    // 2. Word bigrams for local phrase structure
    for (let i = 0; i < tokens.length - 1; i++) {
      this.projectFeature(vec, `b:${tokens[i]}_${tokens[i + 1]}`, 1.4);
    }

    // Fallback for punctuation-only or empty queries
    if (tokens.length === 0 && text.trim().length > 0) {
      const compact = text.trim().toLowerCase();
      for (let i = 0; i < compact.length; i++) {
        this.projectFeature(vec, `r:${compact[i]}`, 1.0);
      }
    }

    return normalizeVector(vec);
  }

  private projectFeature(vec: Float32Array, feature: string, weight: number): void {
    // Dual-hash projection reduces collision variance cleanly.
    const h1 = fnv1a32(feature, 0x811c9dc5);
    const h2 = fnv1a32(feature, 0x9e3779b9);
    const idx1 = h1 % this.dimensions;
    const sign1 = (h1 >>> 31) === 0 ? 1 : -1;
    const idx2 = h2 % this.dimensions;
    const sign2 = (h2 >>> 31) === 0 ? 1 : -1;

    vec[idx1] = (vec[idx1] ?? 0) + sign1 * weight;
    vec[idx2] = (vec[idx2] ?? 0) + sign2 * (weight * 0.5);
  }
}

export interface WebGpuEmbedderOptions {
  id?: string;
  dimensions: number;
  maxBatchSize?: number;
  normalize?: boolean;
  /** Underlying local inference runner (e.g., WebGPU WGSL runtime, Transformers.js pipeline, or ORT session). */
  embedBatch: (texts: string[], signal?: AbortSignal) => Promise<Array<Float32Array | number[]>>;
}

/**
 * Pluggable adapter for WebGPU / WASM local embedding models (Transformers.js, ORT WebGPU,
 * or `@yours/harness` / Rust WGSL embedding workers).
 */
export class WebGpuEmbedderAdapter implements EmbeddingProvider {
  readonly id: string;
  readonly dimensions: number;
  private readonly maxBatchSize: number;
  private readonly normalize: boolean;
  private readonly embedBatchFn: WebGpuEmbedderOptions['embedBatch'];

  constructor(options: WebGpuEmbedderOptions) {
    this.id = options.id ?? 'webgpu-local-embedder';
    this.dimensions = options.dimensions;
    this.maxBatchSize = Math.max(1, options.maxBatchSize ?? 32);
    this.normalize = options.normalize ?? true;
    this.embedBatchFn = options.embedBatch;
  }

  async embed(texts: string[], signal?: AbortSignal): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const results: Float32Array[] = [];

    for (let i = 0; i < texts.length; i += this.maxBatchSize) {
      if (signal?.aborted) {
        throw new DOMException('Embedding aborted', 'AbortError');
      }
      const slice = texts.slice(i, i + this.maxBatchSize);
      const rawBatch = await this.embedBatchFn(slice, signal);
      if (rawBatch.length !== slice.length) {
        throw new Error(`Embedder returned ${rawBatch.length} vectors for batch of ${slice.length}`);
      }
      for (const raw of rawBatch) {
        if (raw.length !== this.dimensions) {
          throw new Error(`Expected embedding dimension ${this.dimensions}, received ${raw.length}`);
        }
        const f32 = raw instanceof Float32Array ? new Float32Array(raw) : Float32Array.from(raw);
        results.push(this.normalize ? normalizeVector(f32) : f32);
      }
    }

    return results;
  }

  async embedQuery(text: string, signal?: AbortSignal): Promise<Float32Array> {
    const [vec] = await this.embed([text], signal);
    if (!vec) {
      throw new Error('Failed to embed query');
    }
    return vec;
  }
}
