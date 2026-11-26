import type {
  BlockedDomainRule,
  DomainRuleKind,
  EmbeddingProvider,
  HarnessMessage,
  InspectOptions,
  MemoryEdge,
  MemoryProvenanceNode,
  MemoryProvider,
  MemoryQuery,
  MemoryHit,
  MemoryRecord,
  MemoryRecordInput,
  MemoryStorageStats,
  MemoryTier,
  QuantizationMode,
  RankingConfig,
  RememberOptions,
} from './contracts.js';
import { MemoryError, SensitiveDomainBlockedError } from './errors.js';
import { planConsolidation } from './lifecycle/consolidator.js';
import { findContradictions } from './lifecycle/contradiction.js';
import { findDuplicateRecord } from './lifecycle/dedup.js';
import { SensitiveDomainFilter } from './privacy/blocklist.js';
import { decryptPayload, encryptPayload } from './privacy/crypto.js';
import { assertTaintMutationAllowed, resolveTaintFlag } from './privacy/taint.js';
import { DeterministicHashEmbedder } from './retrieval/embeddings.js';
import { HnswIndex } from './retrieval/hnsw.js';
import { DEFAULT_RANKING_CONFIG, fuseHybridRanks } from './retrieval/ranking.js';
import { createBestStorageBackend } from './storage/auto.js';
import type { MemoryStorageBackend, StoredVectorRecord } from './storage/types.js';
import { generateId } from './utils/id.js';
import { quantizeInt8 } from './utils/math.js';

export interface HybridMemoryEngineOptions {
  storage?: MemoryStorageBackend;
  embedder?: EmbeddingProvider;
  quantization?: QuantizationMode;
  hnswM?: number;
  hnswEfConstruction?: number;
  hnswEfSearch?: number;
  ranking?: Partial<RankingConfig>;
  nearDuplicateThreshold?: number;
  includeBuiltinBlocklist?: boolean;
  pbkdf2Iterations?: number;
  dbName?: string;
}

interface ExportEnvelopeV1 {
  format: 'yours-memory-export-v1';
  exportedAt: number;
  records: MemoryRecord[];
  edges: MemoryEdge[];
  blockedRules: BlockedDomainRule[];
}

function serializeVectorForStorage(
  memoryId: string,
  vec: Float32Array,
  quantization: QuantizationMode,
  nowMs: number
): StoredVectorRecord {
  if (quantization === 'int8') {
    const { q, scale } = quantizeInt8(vec);
    return {
      memoryId,
      dims: vec.length,
      quantization: 'int8',
      scale,
      data: new Uint8Array(q.buffer, q.byteOffset, q.byteLength).slice(),
      updatedAt: nowMs,
    };
  }
  const bytes = new Uint8Array(vec.buffer, vec.byteOffset, vec.byteLength).slice();
  return {
    memoryId,
    dims: vec.length,
    quantization: 'f32',
    scale: 1,
    data: bytes,
    updatedAt: nowMs,
  };
}

function deserializeStoredVector(stored: StoredVectorRecord): Float32Array {
  if (stored.quantization === 'int8') {
    const q = new Int8Array(stored.data.buffer, stored.data.byteOffset, stored.dims);
    const out = new Float32Array(stored.dims);
    for (let i = 0; i < stored.dims; i++) {
      out[i] = (q[i] ?? 0) * stored.scale;
    }
    return out;
  }
  const view = new DataView(stored.data.buffer, stored.data.byteOffset, stored.data.byteLength);
  const out = new Float32Array(stored.dims);
  for (let i = 0; i < stored.dims; i++) {
    out[i] = view.getFloat32(i * 4, true);
  }
  return out;
}

/**
 * Production local hybrid memory engine combining:
 * - Multi-tier relational storage (OPFS SQLite-WASM / IndexedDB / InMemory)
 * - FTS5 / BM25 lexical search + int8-quantized HNSW ANN vector search
 * - Reciprocal Rank Fusion (RRF) + tier-aware temporal salience decay
 * - Deduplication, non-destructive contradiction superseding, taint enforcement,
 *   sensitive-domain blocking, and AES-256-GCM encrypted export/import.
 */
export class HybridMemoryEngine implements MemoryProvider {
  private storage: MemoryStorageBackend | null;
  private readonly embedder: EmbeddingProvider;
  private readonly quantization: QuantizationMode;
  private readonly hnswM: number;
  private readonly hnswEfConstruction: number;
  private readonly hnswEfSearch: number;
  private readonly rankingConfig: Partial<RankingConfig>;
  private readonly nearDuplicateThreshold: number;
  private readonly includeBuiltinBlocklist: boolean;
  private readonly pbkdf2Iterations: number;
  private readonly dbName?: string;

  private hnsw: HnswIndex;
  private domainFilter: SensitiveDomainFilter;
  private initPromise: Promise<void> | null = null;
  private initialized = false;

  constructor(options: HybridMemoryEngineOptions = {}) {
    this.storage = options.storage ?? null;
    this.embedder = options.embedder ?? new DeterministicHashEmbedder();
    this.quantization = options.quantization ?? 'int8';
    this.hnswM = options.hnswM ?? 16;
    this.hnswEfConstruction = options.hnswEfConstruction ?? 100;
    this.hnswEfSearch = options.hnswEfSearch ?? 64;
    this.rankingConfig = options.ranking ?? {};
    this.nearDuplicateThreshold = options.nearDuplicateThreshold ?? 0.94;
    this.includeBuiltinBlocklist = options.includeBuiltinBlocklist ?? true;
    this.pbkdf2Iterations = options.pbkdf2Iterations ?? 210_000;
    this.dbName = options.dbName;

    this.hnsw = new HnswIndex({
      dimensions: this.embedder.dimensions,
      M: this.hnswM,
      efConstruction: this.hnswEfConstruction,
      efSearch: this.hnswEfSearch,
      quantization: this.quantization,
    });
    this.domainFilter = new SensitiveDomainFilter({
      includeBuiltins: this.includeBuiltinBlocklist,
    });
  }

  async init(): Promise<void> {
    if (this.initialized) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      if (!this.storage) {
        this.storage = await createBestStorageBackend({ dbName: this.dbName });
      } else {
        await this.storage.init();
      }

      const persistedRules = await this.storage.listBlockedRules();
      this.domainFilter = new SensitiveDomainFilter({
        includeBuiltins: this.includeBuiltinBlocklist,
        initialRules: persistedRules,
      });

      const snapshot = await this.storage.loadIndexSnapshot('hnsw');
      let loadedFromSnapshot = false;
      if (snapshot && snapshot.byteLength > 0) {
        try {
          const restored = HnswIndex.deserialize(snapshot);
          if (
            restored.dimensions === this.embedder.dimensions &&
            restored.quantization === this.quantization
          ) {
            this.hnsw = restored;
            loadedFromSnapshot = true;
          }
        } catch {
          loadedFromSnapshot = false;
        }
      }

      if (!loadedFromSnapshot) {
        await this.rebuildIndex();
      }
      this.initialized = true;
    })();

    try {
      await this.initPromise;
    } finally {
      this.initPromise = null;
    }
  }

  private async ensureReady(): Promise<MemoryStorageBackend> {
    await this.init();
    if (!this.storage) {
      throw new MemoryError('NOT_INITIALIZED', 'Memory storage backend unavailable');
    }
    return this.storage;
  }

  async remember(input: MemoryRecordInput, options: RememberOptions = {}): Promise<MemoryRecord> {
    const storage = await this.ensureReady();
    const content = input.content.trim();
    if (!content) {
      throw new MemoryError('INVALID_INPUT', 'Memory content must not be empty');
    }

    const blockCheck = this.domainFilter.check(input.sourceUrl);
    if (blockCheck.blocked) {
      throw new SensitiveDomainBlockedError(
        input.sourceUrl ?? '',
        blockCheck.rule ?? 'unknown',
        blockCheck.reason ?? 'Sensitive URL'
      );
    }

    const nowMs = options.nowMs ?? Date.now();
    const incomingTainted = resolveTaintFlag(input);
    const normalizedInput: MemoryRecordInput = {
      ...input,
      content,
      importance: Math.max(0, Math.min(1, input.importance ?? 0.5)),
      tainted: incomingTainted ? true : input.tainted,
    };

    const [embedding] = await this.embedder.embed([content]);
    if (!embedding) {
      throw new MemoryError('EMBEDDING_FAILED', 'Embedder returned no vector');
    }

    const activeInTier = await storage.listRecords({
      tier: normalizedInput.tier,
      includeSuperseded: false,
    });

    // 1. Check exact / near-duplicate before contradiction or insert
    if (!options.skipDeduplication) {
      const candidates = activeInTier.map((record) => ({
        record,
        vector: this.hnsw.getVector(record.id),
      }));
      const dup = findDuplicateRecord(normalizedInput, embedding, candidates, {
        nearDuplicateThreshold: this.nearDuplicateThreshold,
      });
      if (dup) {
        const mergedTags = Array.from(
          new Set([...(dup.record.tags ?? []), ...(normalizedInput.tags ?? [])])
        );
        const reinforced: MemoryRecord = {
          ...dup.record,
          tags: mergedTags.length > 0 ? mergedTags : undefined,
          importance: Math.max(dup.record.importance ?? 0.5, normalizedInput.importance ?? 0.5),
          updatedAt: nowMs,
          lastAccessedAt: nowMs,
          accessCount: dup.record.accessCount + 1,
        };
        await storage.upsertRecord(reinforced);
        return reinforced;
      }
    }

    // 2. Check contradictions & taint policy BEFORE mutating any records
    const contradictions = options.skipContradictionCheck
      ? normalizedInput.supersedesId
        ? findContradictions(normalizedInput, activeInTier).filter(
            (m) => m.reason === 'explicit-supersede'
          )
        : []
      : findContradictions(normalizedInput, activeInTier);

    for (const match of contradictions) {
      assertTaintMutationAllowed(
        match.record,
        incomingTainted,
        options.confirmTaintedProfileMutation
      );
    }

    // Also guard explicit supersedesId if target is outside activeInTier (e.g., cross-tier)
    if (normalizedInput.supersedesId && !contradictions.some((c) => c.record.id === normalizedInput.supersedesId)) {
      const explicitTarget = await storage.getRecord(normalizedInput.supersedesId);
      if (explicitTarget && !explicitTarget.supersededById) {
        assertTaintMutationAllowed(
          explicitTarget,
          incomingTainted,
          options.confirmTaintedProfileMutation
        );
        contradictions.push({ record: explicitTarget, reason: 'explicit-supersede' });
      }
    }

    const newId = generateId('mem');
    const newRecord: MemoryRecord = {
      ...normalizedInput,
      id: newId,
      supersedesId: normalizedInput.supersedesId ?? contradictions[0]?.record.id,
      createdAt: nowMs,
      updatedAt: nowMs,
      lastAccessedAt: nowMs,
      accessCount: 1,
    };

    await storage.upsertRecord(newRecord);

    const storedVec = serializeVectorForStorage(newId, embedding, this.quantization, nowMs);
    await storage.upsertVector(storedVec);
    this.hnsw.insert(newId, embedding);

    // 3. Apply non-destructive superseding + provenance edges
    for (const match of contradictions) {
      const updatedOld: MemoryRecord = {
        ...match.record,
        supersededById: newId,
        updatedAt: nowMs,
      };
      await storage.upsertRecord(updatedOld);
      this.hnsw.remove(match.record.id);

      await storage.upsertEdge({
        id: generateId('edge'),
        sourceId: newId,
        targetId: match.record.id,
        relation: 'supersedes',
        weight: 1,
        metadata: { reason: match.reason },
        createdAt: nowMs,
      });
    }

    if (!options.deferIndexSave) {
      await storage.saveIndexSnapshot('hnsw', this.hnsw.serialize());
    }
    return newRecord;
  }

  async recall(query: MemoryQuery): Promise<MemoryHit[]> {
    const storage = await this.ensureReady();
    const qText = query.query.trim();
    if (!qText) return [];

    const nowMs = query.nowMs ?? Date.now();
    const limit = Math.max(1, query.limit ?? 10);
    const minScore = query.minScore ?? 0;
    const candidateLimit = Math.max(limit * 4, 32);

    const activeRecords = await storage.listRecords({
      includeSuperseded: false,
    });
    if (activeRecords.length === 0) return [];

    const tierSet = query.tiers && query.tiers.length > 0 ? new Set<MemoryTier>(query.tiers) : null;
    const recordsById = new Map<string, MemoryRecord>();
    for (const rec of activeRecords) {
      if (!tierSet || tierSet.has(rec.tier)) {
        recordsById.set(rec.id, rec);
      }
    }
    if (recordsById.size === 0) return [];

    const queryVec = this.embedder.embedQuery
      ? await this.embedder.embedQuery(qText)
      : (await this.embedder.embed([qText]))[0]!;

    const [lexicalHits, vectorHits] = await Promise.all([
      storage.searchLexical(qText, {
        tiers: query.tiers,
        includeSuperseded: false,
        limit: candidateLimit,
      }),
      Promise.resolve(
        this.hnsw.search(queryVec, candidateLimit, (id) => recordsById.has(id))
      ),
    ]);

    const fused = fuseHybridRanks({
      recordsById,
      lexicalHits,
      vectorHits,
      nowMs,
      limit,
      minScore,
      config: this.rankingConfig,
    });

    // Reinforce access timestamps & counts for recalled hits
    for (const hit of fused) {
      const updated: MemoryRecord = {
        ...hit.record,
        lastAccessedAt: Math.max(hit.record.lastAccessedAt, nowMs),
        accessCount: hit.record.accessCount + 1,
      };
      hit.record = updated;
      await storage.upsertRecord(updated);
    }

    return fused;
  }

  async consolidate(
    threadId: string,
    messages: HarnessMessage[]
  ): Promise<{ added: number; superseded: number }> {
    const storage = await this.ensureReady();
    const plan = planConsolidation(threadId, messages);
    let added = 0;
    let superseded = 0;

    const createdProfileIds: string[] = [];

    for (const fact of plan.profileFacts) {
      const beforeActive = await storage.listRecords({ tier: 'profile', includeSuperseded: false });
      const activeIdsBefore = new Set(beforeActive.map((r) => r.id));
      const res = await this.remember(fact, { deferIndexSave: true });
      if (!activeIdsBefore.has(res.id)) {
        added++;
        createdProfileIds.push(res.id);
      }
      if (res.supersedesId) {
        superseded++;
      }
    }

    if (plan.episodicSummary) {
      const existingEpisodic = await storage.listRecords({
        tier: 'episodic',
        threadId,
        includeSuperseded: false,
      });
      const prevSummary = existingEpisodic[0];
      const summaryRecord = await this.remember(
        {
          ...plan.episodicSummary,
          supersedesId: prevSummary?.id,
        },
        { skipDeduplication: Boolean(prevSummary), deferIndexSave: true }
      );
      if (!existingEpisodic.some((e) => e.id === summaryRecord.id)) {
        added++;
      }
      if (prevSummary && summaryRecord.supersedesId === prevSummary.id) {
        superseded++;
      }
      for (const pid of createdProfileIds) {
        await storage.upsertEdge({
          id: generateId('edge'),
          sourceId: pid,
          targetId: summaryRecord.id,
          relation: 'derived_from',
          weight: 0.9,
          createdAt: Date.now(),
        });
      }
    }

    if (added > 0 || superseded > 0) {
      await storage.saveIndexSnapshot('hnsw', this.hnsw.serialize());
    }

    return { added, superseded };
  }

  async inspect(filter?: InspectOptions): Promise<MemoryRecord[]> {
    const storage = await this.ensureReady();
    return storage.listRecords(filter);
  }

  async inspectGraph(recordId: string): Promise<MemoryProvenanceNode | null> {
    const storage = await this.ensureReady();
    const record = await storage.getRecord(recordId);
    if (!record) return null;

    const edges = await storage.listEdgesForRecord(recordId);
    const supersedes: MemoryRecord[] = [];
    for (const edge of edges) {
      if (edge.relation === 'supersedes' && edge.sourceId === recordId) {
        const target = await storage.getRecord(edge.targetId);
        if (target) supersedes.push(target);
      }
    }
    if (record.supersedesId && !supersedes.some((r) => r.id === record.supersedesId)) {
      const direct = await storage.getRecord(record.supersedesId);
      if (direct) supersedes.push(direct);
    }

    const supersededBy = record.supersededById
      ? (await storage.getRecord(record.supersededById)) ?? undefined
      : undefined;

    return {
      record,
      edges,
      supersedes,
      supersededBy,
    };
  }

  async forget(id: string): Promise<boolean> {
    const storage = await this.ensureReady();
    this.hnsw.remove(id);
    const deleted = await storage.deleteRecord(id);
    if (deleted) {
      await storage.saveIndexSnapshot('hnsw', this.hnsw.serialize());
    }
    return deleted;
  }

  async exportEncrypted(passphrase: string): Promise<Uint8Array> {
    const storage = await this.ensureReady();
    const [records, edges, blockedRules] = await Promise.all([
      storage.listRecords({ includeSuperseded: true }),
      storage.listAllEdges(),
      storage.listBlockedRules(),
    ]);

    const envelope: ExportEnvelopeV1 = {
      format: 'yours-memory-export-v1',
      exportedAt: Date.now(),
      records,
      edges,
      blockedRules: blockedRules.filter((r) => !r.builtin),
    };

    const jsonBytes = new TextEncoder().encode(JSON.stringify(envelope));
    return encryptPayload(jsonBytes, passphrase, this.pbkdf2Iterations);
  }

  async importEncrypted(blob: Uint8Array, passphrase: string): Promise<{ imported: number }> {
    const storage = await this.ensureReady();
    const plainBytes = await decryptPayload(blob, passphrase);
    let parsed: ExportEnvelopeV1;
    try {
      parsed = JSON.parse(new TextDecoder().decode(plainBytes)) as ExportEnvelopeV1;
    } catch (err) {
      throw new MemoryError('INVALID_BACKUP', 'Decrypted backup is not valid JSON', { cause: err });
    }

    if (parsed.format !== 'yours-memory-export-v1' || !Array.isArray(parsed.records)) {
      throw new MemoryError('INVALID_BACKUP', 'Unsupported backup envelope format');
    }

    let imported = 0;
    const activeToEmbed: MemoryRecord[] = [];

    for (const rec of parsed.records) {
      await storage.upsertRecord(rec);
      imported++;
      if (!rec.supersededById) {
        activeToEmbed.push(rec);
      }
    }

    if (activeToEmbed.length > 0) {
      const vectors = await this.embedder.embed(activeToEmbed.map((r) => r.content));
      for (let i = 0; i < activeToEmbed.length; i++) {
        const rec = activeToEmbed[i]!;
        const vec = vectors[i]!;
        await storage.upsertVector(
          serializeVectorForStorage(rec.id, vec, this.quantization, rec.updatedAt)
        );
        this.hnsw.insert(rec.id, vec);
      }
    }

    for (const edge of parsed.edges ?? []) {
      await storage.upsertEdge(edge);
    }

    for (const rule of parsed.blockedRules ?? []) {
      if (!rule.builtin) {
        await storage.upsertBlockedRule(rule);
        this.domainFilter.addRuleObject(rule);
      }
    }

    await storage.saveIndexSnapshot('hnsw', this.hnsw.serialize());
    return { imported };
  }

  async wipeAll(): Promise<void> {
    const storage = await this.ensureReady();
    await storage.clearAll();
    this.hnsw = new HnswIndex({
      dimensions: this.embedder.dimensions,
      M: this.hnswM,
      efConstruction: this.hnswEfConstruction,
      efSearch: this.hnswEfSearch,
      quantization: this.quantization,
    });
    this.domainFilter = new SensitiveDomainFilter({
      includeBuiltins: this.includeBuiltinBlocklist,
    });
  }

  async addBlockedDomain(
    pattern: string,
    kind: DomainRuleKind = 'glob',
    reason = 'User-configured privacy rule'
  ): Promise<BlockedDomainRule> {
    const storage = await this.ensureReady();
    const rule = this.domainFilter.addRule(pattern, kind, reason);
    await storage.upsertBlockedRule(rule);
    return rule;
  }

  async removeBlockedDomain(id: string): Promise<boolean> {
    const storage = await this.ensureReady();
    const removed = this.domainFilter.removeRule(id);
    if (removed) {
      await storage.deleteBlockedRule(id);
    }
    return removed;
  }

  async listBlockedDomains(): Promise<BlockedDomainRule[]> {
    await this.ensureReady();
    return this.domainFilter.listRules();
  }

  isUrlBlocked(url: string): { blocked: boolean; reason?: string; rule?: string } {
    return this.domainFilter.check(url);
  }

  async rebuildIndex(): Promise<number> {
    if (!this.storage) return 0;
    this.hnsw = new HnswIndex({
      dimensions: this.embedder.dimensions,
      M: this.hnswM,
      efConstruction: this.hnswEfConstruction,
      efSearch: this.hnswEfSearch,
      quantization: this.quantization,
    });

    const activeRecords = await this.storage.listRecords({ includeSuperseded: false });
    if (activeRecords.length === 0) {
      await this.storage.saveIndexSnapshot('hnsw', this.hnsw.serialize());
      return 0;
    }

    const storedVectors = await this.storage.listVectors(activeRecords.map((r) => r.id));
    const vecById = new Map<string, StoredVectorRecord>();
    for (const v of storedVectors) {
      vecById.set(v.memoryId, v);
    }

    const missingRecords: MemoryRecord[] = [];
    for (const rec of activeRecords) {
      const sv = vecById.get(rec.id);
      if (sv && sv.dims === this.embedder.dimensions) {
        this.hnsw.insert(rec.id, deserializeStoredVector(sv));
      } else {
        missingRecords.push(rec);
      }
    }

    if (missingRecords.length > 0) {
      const embedded = await this.embedder.embed(missingRecords.map((r) => r.content));
      for (let i = 0; i < missingRecords.length; i++) {
        const rec = missingRecords[i]!;
        const vec = embedded[i]!;
        await this.storage.upsertVector(
          serializeVectorForStorage(rec.id, vec, this.quantization, rec.updatedAt)
        );
        this.hnsw.insert(rec.id, vec);
      }
    }

    await this.storage.saveIndexSnapshot('hnsw', this.hnsw.serialize());
    return this.hnsw.size;
  }

  async getStats(): Promise<MemoryStorageStats> {
    const storage = await this.ensureReady();
    const all = await storage.listRecords({ includeSuperseded: true });
    const edges = await storage.listAllEdges();
    const byTier: Record<MemoryTier, number> = {
      working: 0,
      profile: 0,
      episodic: 0,
      document: 0,
    };
    let activeRecords = 0;
    let supersededRecords = 0;
    for (const r of all) {
      byTier[r.tier]++;
      if (r.supersededById) {
        supersededRecords++;
      } else {
        activeRecords++;
      }
    }

    return {
      backend: storage.kind,
      totalRecords: all.length,
      activeRecords,
      supersededRecords,
      byTier,
      vectorCount: this.hnsw.size,
      edgeCount: edges.length,
      blockedRuleCount: this.domainFilter.listRules().length,
      quantization: this.quantization,
      dimensions: this.embedder.dimensions,
    };
  }

  getRankingDefaults(): RankingConfig {
    return {
      ...DEFAULT_RANKING_CONFIG,
      ...this.rankingConfig,
      decayByTier: {
        ...DEFAULT_RANKING_CONFIG.decayByTier,
        ...this.rankingConfig.decayByTier,
      },
    };
  }

  async close(): Promise<void> {
    if (!this.storage) return;
    if (this.initialized) {
      await this.storage.saveIndexSnapshot('hnsw', this.hnsw.serialize());
    }
    await this.storage.close();
    this.initialized = false;
  }
}
