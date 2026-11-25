import type { QuantizationMode } from '../contracts.js';
import {
  asymmetricQuantizedCosineSimilarity,
  cosineSimilarity,
  dequantizeInt8,
  l2Norm,
  normalizeVector,
  quantizeInt8,
} from '../utils/math.js';

export interface HnswOptions {
  dimensions: number;
  /** Maximum bi-directional links per node on layers >= 1 (default 16). Layer 0 allows 2*M. */
  M?: number;
  /** Dynamic candidate list size during index construction (default 100). */
  efConstruction?: number;
  /** Dynamic candidate list size during query search (default 64). */
  efSearch?: number;
  /** Vector storage precision inside the index: 'int8' (~4x smaller) or 'f32' (default 'int8'). */
  quantization?: QuantizationMode;
  /** Optional seed for deterministic level assignment. */
  seed?: number;
}

export interface HnswSearchResult {
  id: string;
  similarity: number;
  distance: number;
  rank: number;
}

interface HnswNode {
  id: string;
  internalId: number;
  level: number;
  deleted: boolean;
  /** Present when quantization === 'f32' */
  f32?: Float32Array;
  /** Present when quantization === 'int8' */
  q8?: Int8Array;
  scale: number;
  /** neighbors[layer] = array of internalIds */
  neighbors: number[][];
}

interface Candidate {
  internalId: number;
  dist: number;
}

const MAGIC = 0x574e4859; // 'YHNW' little-endian
const FORMAT_VERSION = 1;

/**
 * Native TypeScript Hierarchical Navigable Small World (HNSW) ANN index
 * supporting symmetric int8 scalar quantization (4x smaller in OPFS/RAM) and cosine distance.
 */
export class HnswIndex {
  readonly dimensions: number;
  readonly M: number;
  readonly M0: number;
  readonly efConstruction: number;
  readonly efSearch: number;
  readonly quantization: QuantizationMode;

  private readonly mL: number;
  private rngState: number;

  private nodes: HnswNode[] = [];
  private idToInternal = new Map<string, number>();
  private entryPoint: number | null = null;
  private maxLevel = -1;
  private activeCount = 0;

  constructor(options: HnswOptions) {
    if (options.dimensions <= 0) {
      throw new Error('HnswIndex requires dimensions > 0');
    }
    this.dimensions = options.dimensions;
    this.M = Math.max(4, options.M ?? 16);
    this.M0 = this.M * 2;
    this.efConstruction = Math.max(this.M, options.efConstruction ?? 100);
    this.efSearch = Math.max(16, options.efSearch ?? 64);
    this.quantization = options.quantization ?? 'int8';
    this.mL = 1 / Math.log(this.M);
    this.rngState = (options.seed ?? 0x9e3779b9) >>> 0 || 1;
  }

  get size(): number {
    return this.activeCount;
  }

  has(id: string): boolean {
    const idx = this.idToInternal.get(id);
    if (idx === undefined) return false;
    return !this.nodes[idx]?.deleted;
  }

  private nextRandom(): number {
    // xorshift32 PRNG
    let x = this.rngState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rngState = x >>> 0;
    return (this.rngState & 0x7fffffff) / 0x80000000;
  }

  private sampleLevel(): number {
    const r = Math.max(1e-9, this.nextRandom());
    return Math.min(16, Math.floor(-Math.log(r) * this.mL));
  }

  private distanceNodeToQuery(
    internalId: number,
    queryF32: Float32Array,
    queryNorm: number
  ): number {
    const node = this.nodes[internalId];
    if (!node) return 2;
    if (node.q8) {
      return 1 - asymmetricQuantizedCosineSimilarity(queryF32, queryNorm, node.q8);
    }
    if (node.f32) {
      return 1 - cosineSimilarity(queryF32, node.f32);
    }
    return 2;
  }

  private distanceBetweenNodes(aId: number, bId: number): number {
    const a = this.nodes[aId];
    const b = this.nodes[bId];
    if (!a || !b) return 2;
    if (a.f32 && b.f32) {
      return 1 - cosineSimilarity(a.f32, b.f32);
    }
    if (a.q8 && b.q8) {
      // Integer MAC dot product for fast internal graph construction
      const len = a.q8.length;
      let dot = 0;
      let nA = 0;
      let nB = 0;
      for (let i = 0; i < len; i++) {
        const va = a.q8[i] ?? 0;
        const vb = b.q8[i] ?? 0;
        dot += va * vb;
        nA += va * va;
        nB += vb * vb;
      }
      if (nA === 0 || nB === 0) return 1;
      const sim = dot / (Math.sqrt(nA) * Math.sqrt(nB));
      return 1 - (sim > 1 ? 1 : sim < -1 ? -1 : sim);
    }
    return 2;
  }

  /** Insert or update a vector for `id`. */
  insert(id: string, vector: Float32Array | number[]): void {
    if (vector.length !== this.dimensions) {
      throw new Error(`Expected vector of dimension ${this.dimensions}, got ${vector.length}`);
    }

    if (this.idToInternal.has(id)) {
      this.remove(id);
    }

    const normalized = normalizeVector(vector);
    const level = this.sampleLevel();
    const internalId = this.nodes.length;
    const neighbors: number[][] = Array.from({ length: level + 1 }, () => []);

    let node: HnswNode;
    if (this.quantization === 'int8') {
      const { q, scale } = quantizeInt8(normalized);
      node = {
        id,
        internalId,
        level,
        deleted: false,
        q8: q,
        scale,
        neighbors,
      };
    } else {
      node = {
        id,
        internalId,
        level,
        deleted: false,
        f32: normalized,
        scale: 1,
        neighbors,
      };
    }

    this.nodes.push(node);
    this.idToInternal.set(id, internalId);
    this.activeCount++;

    if (this.entryPoint === null) {
      this.entryPoint = internalId;
      this.maxLevel = level;
      return;
    }

    let currObj = this.entryPoint;
    let currDist = this.distanceBetweenNodes(internalId, currObj);

    // Greedy descent on upper layers > level
    for (let lc = this.maxLevel; lc > level; lc--) {
      let changed = true;
      while (changed) {
        changed = false;
        const currNeighbors = this.nodes[currObj]?.neighbors[lc] ?? [];
        for (const nb of currNeighbors) {
          const d = this.distanceBetweenNodes(internalId, nb);
          if (d < currDist) {
            currDist = d;
            currObj = nb;
            changed = true;
          }
        }
      }
    }

    // Search & connect from min(level, maxLevel) down to layer 0
    const startLayer = Math.min(level, this.maxLevel);
    for (let lc = startLayer; lc >= 0; lc--) {
      const candidates = this.searchLayerForNode(internalId, currObj, this.efConstruction, lc);
      const maxM = lc === 0 ? this.M0 : this.M;
      const selected = this.selectNeighborsHeuristic(internalId, candidates, maxM);

      node.neighbors[lc] = selected.map((c) => c.internalId);

      for (const cand of selected) {
        const nbNode = this.nodes[cand.internalId];
        if (!nbNode) continue;
        const nbList = nbNode.neighbors[lc];
        if (!nbList) continue;
        if (!nbList.includes(internalId)) {
          nbList.push(internalId);
          if (nbList.length > maxM) {
            const nbCandidates: Candidate[] = nbList.map((nId) => ({
              internalId: nId,
              dist: this.distanceBetweenNodes(cand.internalId, nId),
            }));
            nbCandidates.sort((a, b) => a.dist - b.dist);
            const pruned = this.selectNeighborsHeuristic(cand.internalId, nbCandidates, maxM);
            nbNode.neighbors[lc] = pruned.map((p) => p.internalId);
          }
        }
      }

      if (candidates[0]) {
        currObj = candidates[0].internalId;
      }
    }

    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.entryPoint = internalId;
    }
  }

  /** Tombstone-delete an id from the index. */
  remove(id: string): boolean {
    const internalId = this.idToInternal.get(id);
    if (internalId === undefined) return false;
    const node = this.nodes[internalId];
    if (!node || node.deleted) return false;

    node.deleted = true;
    this.idToInternal.delete(id);
    this.activeCount = Math.max(0, this.activeCount - 1);

    if (this.entryPoint === internalId) {
      // Pick next highest-level non-deleted node as entry point
      let bestId: number | null = null;
      let bestLevel = -1;
      for (const n of this.nodes) {
        if (!n.deleted && n.level > bestLevel) {
          bestLevel = n.level;
          bestId = n.internalId;
        }
      }
      this.entryPoint = bestId;
      this.maxLevel = bestLevel;
    }

    return true;
  }

  /** Return reconstructed Float32 vector for an active id, if present. */
  getVector(id: string): Float32Array | null {
    const internalId = this.idToInternal.get(id);
    if (internalId === undefined) return null;
    const node = this.nodes[internalId];
    if (!node || node.deleted) return null;
    if (node.f32) return new Float32Array(node.f32);
    if (node.q8) return dequantizeInt8(node.q8, node.scale);
    return null;
  }

  search(
    query: Float32Array | number[],
    k = 10,
    filterFn?: (id: string) => boolean,
    efOverride?: number
  ): HnswSearchResult[] {
    if (this.entryPoint === null || this.activeCount === 0 || k <= 0) {
      return [];
    }
    if (query.length !== this.dimensions) {
      throw new Error(`Expected query of dimension ${this.dimensions}, got ${query.length}`);
    }

    const qF32 = query instanceof Float32Array ? query : Float32Array.from(query);
    const qNorm = l2Norm(qF32);
    if (qNorm <= 1e-12) return [];

    let currObj = this.entryPoint;
    let currDist = this.distanceNodeToQuery(currObj, qF32, qNorm);

    for (let lc = this.maxLevel; lc > 0; lc--) {
      let changed = true;
      while (changed) {
        changed = false;
        const neighbors = this.nodes[currObj]?.neighbors[lc] ?? [];
        for (const nb of neighbors) {
          const d = this.distanceNodeToQuery(nb, qF32, qNorm);
          if (d < currDist) {
            currDist = d;
            currObj = nb;
            changed = true;
          }
        }
      }
    }

    // Expand ef when a predicate filter is provided so filtered recall stays high.
    const baseEf = Math.max(efOverride ?? this.efSearch, k);
    const ef = filterFn ? Math.min(this.nodes.length, Math.max(baseEf * 2, k * 4)) : baseEf;
    const layer0 = this.searchLayerForQuery(qF32, qNorm, currObj, ef, 0);

    const results: HnswSearchResult[] = [];
    for (const cand of layer0) {
      const node = this.nodes[cand.internalId];
      if (!node || node.deleted) continue;
      if (filterFn && !filterFn(node.id)) continue;
      const similarity = Math.max(-1, Math.min(1, 1 - cand.dist));
      results.push({
        id: node.id,
        similarity,
        distance: cand.dist,
        rank: results.length + 1,
      });
      if (results.length >= k) break;
    }

    return results;
  }

  private searchLayerForNode(
    targetInternalId: number,
    entryId: number,
    ef: number,
    layer: number
  ): Candidate[] {
    const visited = new Set<number>([entryId]);
    const entryDist = this.distanceBetweenNodes(targetInternalId, entryId);
    const candidates: Candidate[] = [{ internalId: entryId, dist: entryDist }];
    const results: Candidate[] = [{ internalId: entryId, dist: entryDist }];

    while (candidates.length > 0) {
      candidates.sort((a, b) => a.dist - b.dist);
      const curr = candidates.shift()!;
      const worstResult = results[results.length - 1];
      if (worstResult && curr.dist > worstResult.dist && results.length >= ef) {
        break;
      }

      const neighbors = this.nodes[curr.internalId]?.neighbors[layer] ?? [];
      for (const nb of neighbors) {
        if (visited.has(nb)) continue;
        visited.add(nb);
        const d = this.distanceBetweenNodes(targetInternalId, nb);
        const worst = results[results.length - 1];
        if (results.length < ef || !worst || d < worst.dist) {
          const item = { internalId: nb, dist: d };
          candidates.push(item);
          results.push(item);
          results.sort((a, b) => a.dist - b.dist);
          if (results.length > ef) {
            results.pop();
          }
        }
      }
    }

    return results;
  }

  private searchLayerForQuery(
    qF32: Float32Array,
    qNorm: number,
    entryId: number,
    ef: number,
    layer: number
  ): Candidate[] {
    const visited = new Set<number>([entryId]);
    const entryDist = this.distanceNodeToQuery(entryId, qF32, qNorm);
    const candidates: Candidate[] = [{ internalId: entryId, dist: entryDist }];
    const results: Candidate[] = [{ internalId: entryId, dist: entryDist }];

    while (candidates.length > 0) {
      candidates.sort((a, b) => a.dist - b.dist);
      const curr = candidates.shift()!;
      const worstResult = results[results.length - 1];
      if (worstResult && curr.dist > worstResult.dist && results.length >= ef) {
        break;
      }

      const neighbors = this.nodes[curr.internalId]?.neighbors[layer] ?? [];
      for (const nb of neighbors) {
        if (visited.has(nb)) continue;
        visited.add(nb);
        const d = this.distanceNodeToQuery(nb, qF32, qNorm);
        const worst = results[results.length - 1];
        if (results.length < ef || !worst || d < worst.dist) {
          const item = { internalId: nb, dist: d };
          candidates.push(item);
          results.push(item);
          results.sort((a, b) => a.dist - b.dist);
          if (results.length > ef) {
            results.pop();
          }
        }
      }
    }

    return results;
  }

  /**
   * HNSW diverse neighbor selection heuristic (Malkov & Yashunin, Alg. 4).
   * Keeps diverse directional edges so graph connectivity remains navigable across clusters.
   */
  private selectNeighborsHeuristic(
    baseInternalId: number,
    candidates: Candidate[],
    M: number
  ): Candidate[] {
    const sorted = [...candidates]
      .filter((c) => !this.nodes[c.internalId]?.deleted && c.internalId !== baseInternalId)
      .sort((a, b) => a.dist - b.dist);

    if (sorted.length <= M) return sorted;

    const selected: Candidate[] = [];
    const discarded: Candidate[] = [];

    for (const cand of sorted) {
      if (selected.length >= M) break;
      let closerToSelected = false;
      for (const sel of selected) {
        const dCandSel = this.distanceBetweenNodes(cand.internalId, sel.internalId);
        if (dCandSel < cand.dist) {
          closerToSelected = true;
          break;
        }
      }
      if (!closerToSelected) {
        selected.push(cand);
      } else {
        discarded.push(cand);
      }
    }

    for (const rem of discarded) {
      if (selected.length >= M) break;
      selected.push(rem);
    }

    return selected;
  }

  /**
   * Compact binary serialization (`YHNW` v1) suitable for OPFS SyncAccessHandle or IndexedDB.
   * Strips deleted nodes cleanly by re-indexing active nodes or preserving internal topology.
   */
  serialize(): Uint8Array {
    const encoder = new TextEncoder();
    const idBytesList = this.nodes.map((n) => encoder.encode(n.id));
    const vecByteSize = this.quantization === 'int8' ? this.dimensions : this.dimensions * 4;

    let totalBytes = 32; // Header
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i]!;
      const idLen = idBytesList[i]!.byteLength;
      // 4 (idLen) + idLen + 1 (deleted) + 1 (level) + 4 (scale) + vecByteSize
      totalBytes += 10 + idLen + vecByteSize;
      for (let lc = 0; lc <= n.level; lc++) {
        const nbs = n.neighbors[lc] ?? [];
        totalBytes += 4 + nbs.length * 4;
      }
    }

    const buffer = new ArrayBuffer(totalBytes);
    const view = new DataView(buffer);
    const u8 = new Uint8Array(buffer);
    let offset = 0;

    view.setUint32(offset, MAGIC, true); offset += 4;
    view.setUint16(offset, FORMAT_VERSION, true); offset += 2;
    view.setUint8(offset, this.quantization === 'int8' ? 1 : 0); offset += 1;
    view.setUint8(offset, 0); offset += 1; // reserved
    view.setUint32(offset, this.dimensions, true); offset += 4;
    view.setUint32(offset, this.M, true); offset += 4;
    view.setUint32(offset, this.efConstruction, true); offset += 4;
    view.setUint32(offset, this.efSearch, true); offset += 4;
    view.setInt32(offset, this.entryPoint ?? -1, true); offset += 4;
    view.setUint32(offset, this.nodes.length, true); offset += 4;

    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i]!;
      const idBytes = idBytesList[i]!;
      view.setUint32(offset, idBytes.byteLength, true); offset += 4;
      u8.set(idBytes, offset); offset += idBytes.byteLength;
      view.setUint8(offset, n.deleted ? 1 : 0); offset += 1;
      view.setUint8(offset, n.level); offset += 1;
      view.setFloat32(offset, n.scale, true); offset += 4;

      if (this.quantization === 'int8') {
        const q = n.q8 ?? new Int8Array(this.dimensions);
        u8.set(new Uint8Array(q.buffer, q.byteOffset, q.byteLength), offset);
        offset += this.dimensions;
      } else {
        const f = n.f32 ?? new Float32Array(this.dimensions);
        for (let d = 0; d < this.dimensions; d++) {
          view.setFloat32(offset + d * 4, f[d] ?? 0, true);
        }
        offset += this.dimensions * 4;
      }

      for (let lc = 0; lc <= n.level; lc++) {
        const nbs = n.neighbors[lc] ?? [];
        view.setUint32(offset, nbs.length, true); offset += 4;
        for (const nb of nbs) {
          view.setUint32(offset, nb, true); offset += 4;
        }
      }
    }

    return u8;
  }

  static deserialize(bytes: Uint8Array): HnswIndex {
    if (bytes.byteLength < 32) {
      throw new Error('Invalid HNSW snapshot: header too short');
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let offset = 0;

    const magic = view.getUint32(offset, true); offset += 4;
    if (magic !== MAGIC) {
      throw new Error('Invalid HNSW snapshot: magic mismatch');
    }
    const version = view.getUint16(offset, true); offset += 2;
    if (version !== FORMAT_VERSION) {
      throw new Error(`Unsupported HNSW snapshot version: ${version}`);
    }
    const quantFlag = view.getUint8(offset); offset += 1;
    offset += 1; // reserved
    const dimensions = view.getUint32(offset, true); offset += 4;
    const M = view.getUint32(offset, true); offset += 4;
    const efConstruction = view.getUint32(offset, true); offset += 4;
    const efSearch = view.getUint32(offset, true); offset += 4;
    const entryPointRaw = view.getInt32(offset, true); offset += 4;
    const nodeCount = view.getUint32(offset, true); offset += 4;

    const quantization: QuantizationMode = quantFlag === 1 ? 'int8' : 'f32';
    const index = new HnswIndex({ dimensions, M, efConstruction, efSearch, quantization });
    const decoder = new TextDecoder();

    let maxLevel = -1;
    let activeCount = 0;

    for (let i = 0; i < nodeCount; i++) {
      const idLen = view.getUint32(offset, true); offset += 4;
      const id = decoder.decode(bytes.subarray(offset, offset + idLen)); offset += idLen;
      const deleted = view.getUint8(offset) === 1; offset += 1;
      const level = view.getUint8(offset); offset += 1;
      const scale = view.getFloat32(offset, true); offset += 4;

      let q8: Int8Array | undefined;
      let f32: Float32Array | undefined;

      if (quantization === 'int8') {
        q8 = new Int8Array(dimensions);
        for (let d = 0; d < dimensions; d++) {
          q8[d] = view.getInt8(offset + d);
        }
        offset += dimensions;
      } else {
        f32 = new Float32Array(dimensions);
        for (let d = 0; d < dimensions; d++) {
          f32[d] = view.getFloat32(offset + d * 4, true);
        }
        offset += dimensions * 4;
      }

      const neighbors: number[][] = [];
      for (let lc = 0; lc <= level; lc++) {
        const nbLen = view.getUint32(offset, true); offset += 4;
        const list: number[] = [];
        for (let j = 0; j < nbLen; j++) {
          list.push(view.getUint32(offset, true));
          offset += 4;
        }
        neighbors.push(list);
      }

      const node: HnswNode = {
        id,
        internalId: i,
        level,
        deleted,
        q8,
        f32,
        scale,
        neighbors,
      };
      index.nodes.push(node);
      if (!deleted) {
        index.idToInternal.set(id, i);
        activeCount++;
        if (level > maxLevel) maxLevel = level;
      }
    }

    index.entryPoint = entryPointRaw >= 0 && entryPointRaw < nodeCount ? entryPointRaw : null;
    index.maxLevel = maxLevel;
    index.activeCount = activeCount;
    return index;
  }
}
