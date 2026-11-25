import type { BlockedDomainRule, InspectOptions, MemoryEdge, MemoryRecord } from '../contracts.js';
import { Bm25Index } from '../retrieval/bm25.js';
import type { LexicalSearchOptions, LexicalSearchResult, MemoryStorageBackend, StoredVectorRecord } from './types.js';

export function cloneRecord(record: MemoryRecord): MemoryRecord {
  return {
    ...record,
    tags: record.tags ? [...record.tags] : undefined,
  };
}

export function cloneEdge(edge: MemoryEdge): MemoryEdge {
  return {
    ...edge,
    metadata: edge.metadata ? { ...edge.metadata } : undefined,
  };
}

export function cloneVector(vec: StoredVectorRecord): StoredVectorRecord {
  return {
    ...vec,
    data: new Uint8Array(vec.data),
  };
}

export function matchesInspectFilter(record: MemoryRecord, filter?: InspectOptions): boolean {
  if (!filter) return !record.supersededById;
  if (!filter.includeSuperseded && record.supersededById) return false;
  if (filter.tier && record.tier !== filter.tier) return false;
  if (filter.threadId && record.threadId !== filter.threadId) return false;
  if (filter.tainted !== undefined && Boolean(record.tainted) !== filter.tainted) return false;
  return true;
}

/**
 * Pure TypeScript In-Memory storage backend.
 * Serves as a crisp reference backend for Node/Vitest and as a zero-dependency fallback.
 */
export class InMemoryStorageBackend implements MemoryStorageBackend {
  readonly kind = 'in-memory' as const;

  private readonly records = new Map<string, MemoryRecord>();
  private readonly vectors = new Map<string, StoredVectorRecord>();
  private readonly edges = new Map<string, MemoryEdge>();
  private readonly blockedRules = new Map<string, BlockedDomainRule>();
  private readonly snapshots = new Map<string, Uint8Array>();
  private readonly bm25 = new Bm25Index();

  async init(): Promise<void> {
    // No async setup required.
  }

  async upsertRecord(record: MemoryRecord): Promise<void> {
    const copy = cloneRecord(record);
    this.records.set(copy.id, copy);
    this.bm25.upsert({
      id: copy.id,
      text: copy.content,
      boostText: [copy.sourceTitle ?? '', ...(copy.tags ?? [])].join(' '),
    });
  }

  async getRecord(id: string): Promise<MemoryRecord | null> {
    const found = this.records.get(id);
    return found ? cloneRecord(found) : null;
  }

  async listRecords(filter?: InspectOptions): Promise<MemoryRecord[]> {
    const matched = Array.from(this.records.values())
      .filter((r) => matchesInspectFilter(r, filter))
      .sort((a, b) => b.updatedAt - a.updatedAt || b.createdAt - a.createdAt || a.id.localeCompare(b.id));

    const offset = Math.max(0, filter?.offset ?? 0);
    const limit = filter?.limit !== undefined ? Math.max(0, filter.limit) : matched.length;
    return matched.slice(offset, offset + limit).map(cloneRecord);
  }

  async deleteRecord(id: string): Promise<boolean> {
    const existed = this.records.delete(id);
    this.bm25.remove(id);
    this.vectors.delete(id);
    for (const [edgeId, edge] of this.edges) {
      if (edge.sourceId === id || edge.targetId === id) {
        this.edges.delete(edgeId);
      }
    }
    return existed;
  }

  async upsertVector(vector: StoredVectorRecord): Promise<void> {
    this.vectors.set(vector.memoryId, cloneVector(vector));
  }

  async getVector(memoryId: string): Promise<StoredVectorRecord | null> {
    const found = this.vectors.get(memoryId);
    return found ? cloneVector(found) : null;
  }

  async listVectors(memoryIds?: string[]): Promise<StoredVectorRecord[]> {
    if (memoryIds) {
      const out: StoredVectorRecord[] = [];
      for (const id of memoryIds) {
        const v = this.vectors.get(id);
        if (v) out.push(cloneVector(v));
      }
      return out;
    }
    return Array.from(this.vectors.values()).map(cloneVector);
  }

  async deleteVector(memoryId: string): Promise<boolean> {
    return this.vectors.delete(memoryId);
  }

  async searchLexical(query: string, options: LexicalSearchOptions = {}): Promise<LexicalSearchResult[]> {
    const limit = options.limit ?? 20;
    const tierSet = options.tiers && options.tiers.length > 0 ? new Set(options.tiers) : null;
    return this.bm25.search(query, limit, (id) => {
      const rec = this.records.get(id);
      if (!rec) return false;
      if (!options.includeSuperseded && rec.supersededById) return false;
      if (tierSet && !tierSet.has(rec.tier)) return false;
      return true;
    });
  }

  async upsertEdge(edge: MemoryEdge): Promise<void> {
    this.edges.set(edge.id, cloneEdge(edge));
  }

  async listEdgesForRecord(recordId: string): Promise<MemoryEdge[]> {
    return Array.from(this.edges.values())
      .filter((e) => e.sourceId === recordId || e.targetId === recordId)
      .sort((a, b) => a.createdAt - b.createdAt)
      .map(cloneEdge);
  }

  async listAllEdges(): Promise<MemoryEdge[]> {
    return Array.from(this.edges.values())
      .sort((a, b) => a.createdAt - b.createdAt)
      .map(cloneEdge);
  }

  async deleteEdgesForRecord(recordId: string): Promise<void> {
    for (const [id, edge] of this.edges) {
      if (edge.sourceId === recordId || edge.targetId === recordId) {
        this.edges.delete(id);
      }
    }
  }

  async upsertBlockedRule(rule: BlockedDomainRule): Promise<void> {
    this.blockedRules.set(rule.id, { ...rule });
  }

  async listBlockedRules(): Promise<BlockedDomainRule[]> {
    return Array.from(this.blockedRules.values())
      .sort((a, b) => a.createdAt - b.createdAt)
      .map((r) => ({ ...r }));
  }

  async deleteBlockedRule(id: string): Promise<boolean> {
    return this.blockedRules.delete(id);
  }

  async saveIndexSnapshot(name: string, bytes: Uint8Array): Promise<void> {
    this.snapshots.set(name, new Uint8Array(bytes));
  }

  async loadIndexSnapshot(name: string): Promise<Uint8Array | null> {
    const found = this.snapshots.get(name);
    return found ? new Uint8Array(found) : null;
  }

  async clearAll(): Promise<void> {
    this.records.clear();
    this.vectors.clear();
    this.edges.clear();
    this.blockedRules.clear();
    this.snapshots.clear();
    this.bm25.clear();
  }

  async close(): Promise<void> {
    // Nothing to release.
  }
}
