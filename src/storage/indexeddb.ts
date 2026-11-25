import type { BlockedDomainRule, InspectOptions, MemoryEdge, MemoryRecord } from '../contracts.js';
import { Bm25Index } from '../retrieval/bm25.js';
import { cloneEdge, cloneRecord, cloneVector, InMemoryStorageBackend, matchesInspectFilter } from './in-memory.js';
import type { LexicalSearchOptions, LexicalSearchResult, MemoryStorageBackend, StoredVectorRecord } from './types.js';

const STORES = ['memories', 'memory_vectors', 'memory_edges', 'blocked_domains', 'index_snapshots'] as const;
type StoreName = (typeof STORES)[number];

export interface IndexedDbBackendOptions {
  dbName?: string;
  version?: number;
  /** Explicit IDBFactory override (useful for tests or custom environments). */
  indexedDbFactory?: IDBFactory;
  /** If true, throw when IndexedDB is unavailable instead of falling back to InMemoryStorageBackend. */
  requireIndexedDb?: boolean;
}

function idbRequest<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB request failed'));
  });
}

function idbTransactionDone(tx: IDBTransaction): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted'));
  });
}

/**
 * IndexedDB storage backend with identical semantics to OPFS SQLite-WASM.
 * Automatically hydrates an in-memory BM25 index for fast lexical ranking, and falls
 * back gracefully to InMemoryStorageBackend when `indexedDB` is unavailable.
 */
export class IndexedDbStorageBackend implements MemoryStorageBackend {
  readonly kind = 'indexeddb' as const;

  private readonly dbName: string;
  private readonly version: number;
  private readonly factory?: IDBFactory;
  private readonly requireIndexedDb: boolean;

  private db: IDBDatabase | null = null;
  private fallback: InMemoryStorageBackend | null = null;
  private readonly bm25 = new Bm25Index();
  private readonly recordIndex = new Map<string, MemoryRecord>();

  constructor(options: IndexedDbBackendOptions = {}) {
    this.dbName = options.dbName ?? 'yours-memory-db';
    this.version = options.version ?? 1;
    this.factory = options.indexedDbFactory ?? (typeof globalThis.indexedDB !== 'undefined' ? globalThis.indexedDB : undefined);
    this.requireIndexedDb = options.requireIndexedDb ?? false;
  }

  get isUsingInMemoryFallback(): boolean {
    return this.fallback !== null;
  }

  async init(): Promise<void> {
    if (this.db || this.fallback) return;
    if (!this.factory) {
      if (this.requireIndexedDb) {
        throw new Error('IndexedDB is not available in this environment');
      }
      this.fallback = new InMemoryStorageBackend();
      await this.fallback.init();
      return;
    }

    const factory = this.factory;
    this.db = await new Promise<IDBDatabase>((resolve, reject) => {
      const openReq = factory.open(this.dbName, this.version);
      openReq.onupgradeneeded = () => {
        const db = openReq.result;
        if (!db.objectStoreNames.contains('memories')) {
          db.createObjectStore('memories', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('memory_vectors')) {
          db.createObjectStore('memory_vectors', { keyPath: 'memoryId' });
        }
        if (!db.objectStoreNames.contains('memory_edges')) {
          db.createObjectStore('memory_edges', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('blocked_domains')) {
          db.createObjectStore('blocked_domains', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('index_snapshots')) {
          db.createObjectStore('index_snapshots', { keyPath: 'name' });
        }
      };
      openReq.onsuccess = () => resolve(openReq.result);
      openReq.onerror = () => reject(openReq.error ?? new Error('Failed to open IndexedDB'));
    });

    // Hydrate BM25 index & record metadata cache from persisted records.
    const existing = await this.getAllFromStore<MemoryRecord>('memories');
    for (const rec of existing) {
      const copy = cloneRecord(rec);
      this.recordIndex.set(copy.id, copy);
      this.bm25.upsert({
        id: copy.id,
        text: copy.content,
        boostText: [copy.sourceTitle ?? '', ...(copy.tags ?? [])].join(' '),
      });
    }
  }

  private getDb(): IDBDatabase {
    if (!this.db) {
      throw new Error('IndexedDbStorageBackend is not initialized');
    }
    return this.db;
  }

  private async getAllFromStore<T>(storeName: StoreName): Promise<T[]> {
    const db = this.getDb();
    const tx = db.transaction(storeName, 'readonly');
    const store = tx.objectStore(storeName);
    const result = await idbRequest(store.getAll() as IDBRequest<T[]>);
    await idbTransactionDone(tx);
    return result;
  }

  async upsertRecord(record: MemoryRecord): Promise<void> {
    if (this.fallback) return this.fallback.upsertRecord(record);
    const copy = cloneRecord(record);
    const tx = this.getDb().transaction('memories', 'readwrite');
    tx.objectStore('memories').put(copy);
    await idbTransactionDone(tx);

    this.recordIndex.set(copy.id, copy);
    this.bm25.upsert({
      id: copy.id,
      text: copy.content,
      boostText: [copy.sourceTitle ?? '', ...(copy.tags ?? [])].join(' '),
    });
  }

  async getRecord(id: string): Promise<MemoryRecord | null> {
    if (this.fallback) return this.fallback.getRecord(id);
    const cached = this.recordIndex.get(id);
    if (cached) return cloneRecord(cached);
    const tx = this.getDb().transaction('memories', 'readonly');
    const found = await idbRequest(tx.objectStore('memories').get(id) as IDBRequest<MemoryRecord | undefined>);
    await idbTransactionDone(tx);
    return found ? cloneRecord(found) : null;
  }

  async listRecords(filter?: InspectOptions): Promise<MemoryRecord[]> {
    if (this.fallback) return this.fallback.listRecords(filter);
    const matched = Array.from(this.recordIndex.values())
      .filter((r) => matchesInspectFilter(r, filter))
      .sort((a, b) => b.updatedAt - a.updatedAt || b.createdAt - a.createdAt || a.id.localeCompare(b.id));

    const offset = Math.max(0, filter?.offset ?? 0);
    const limit = filter?.limit !== undefined ? Math.max(0, filter.limit) : matched.length;
    return matched.slice(offset, offset + limit).map(cloneRecord);
  }

  async deleteRecord(id: string): Promise<boolean> {
    if (this.fallback) return this.fallback.deleteRecord(id);
    const existed = this.recordIndex.has(id);
    if (!existed) return false;

    const allEdges = await this.listEdgesForRecord(id);
    const tx = this.getDb().transaction(['memories', 'memory_vectors', 'memory_edges'], 'readwrite');
    tx.objectStore('memories').delete(id);
    tx.objectStore('memory_vectors').delete(id);
    const edgeStore = tx.objectStore('memory_edges');
    for (const edge of allEdges) {
      edgeStore.delete(edge.id);
    }
    await idbTransactionDone(tx);

    this.recordIndex.delete(id);
    this.bm25.remove(id);
    return true;
  }

  async upsertVector(vector: StoredVectorRecord): Promise<void> {
    if (this.fallback) return this.fallback.upsertVector(vector);
    const copy = cloneVector(vector);
    const tx = this.getDb().transaction('memory_vectors', 'readwrite');
    tx.objectStore('memory_vectors').put(copy);
    await idbTransactionDone(tx);
  }

  async getVector(memoryId: string): Promise<StoredVectorRecord | null> {
    if (this.fallback) return this.fallback.getVector(memoryId);
    const tx = this.getDb().transaction('memory_vectors', 'readonly');
    const found = await idbRequest(
      tx.objectStore('memory_vectors').get(memoryId) as IDBRequest<StoredVectorRecord | undefined>
    );
    await idbTransactionDone(tx);
    return found ? cloneVector(found) : null;
  }

  async listVectors(memoryIds?: string[]): Promise<StoredVectorRecord[]> {
    if (this.fallback) return this.fallback.listVectors(memoryIds);
    const all = await this.getAllFromStore<StoredVectorRecord>('memory_vectors');
    if (memoryIds) {
      const set = new Set(memoryIds);
      return all.filter((v) => set.has(v.memoryId)).map(cloneVector);
    }
    return all.map(cloneVector);
  }

  async deleteVector(memoryId: string): Promise<boolean> {
    if (this.fallback) return this.fallback.deleteVector(memoryId);
    const existing = await this.getVector(memoryId);
    if (!existing) return false;
    const tx = this.getDb().transaction('memory_vectors', 'readwrite');
    tx.objectStore('memory_vectors').delete(memoryId);
    await idbTransactionDone(tx);
    return true;
  }

  async searchLexical(query: string, options: LexicalSearchOptions = {}): Promise<LexicalSearchResult[]> {
    if (this.fallback) return this.fallback.searchLexical(query, options);
    const limit = options.limit ?? 20;
    const tierSet = options.tiers && options.tiers.length > 0 ? new Set(options.tiers) : null;
    return this.bm25.search(query, limit, (id) => {
      const rec = this.recordIndex.get(id);
      if (!rec) return false;
      if (!options.includeSuperseded && rec.supersededById) return false;
      if (tierSet && !tierSet.has(rec.tier)) return false;
      return true;
    });
  }

  async upsertEdge(edge: MemoryEdge): Promise<void> {
    if (this.fallback) return this.fallback.upsertEdge(edge);
    const tx = this.getDb().transaction('memory_edges', 'readwrite');
    tx.objectStore('memory_edges').put(cloneEdge(edge));
    await idbTransactionDone(tx);
  }

  async listEdgesForRecord(recordId: string): Promise<MemoryEdge[]> {
    if (this.fallback) return this.fallback.listEdgesForRecord(recordId);
    const all = await this.getAllFromStore<MemoryEdge>('memory_edges');
    return all
      .filter((e) => e.sourceId === recordId || e.targetId === recordId)
      .sort((a, b) => a.createdAt - b.createdAt)
      .map(cloneEdge);
  }

  async listAllEdges(): Promise<MemoryEdge[]> {
    if (this.fallback) return this.fallback.listAllEdges();
    const all = await this.getAllFromStore<MemoryEdge>('memory_edges');
    return all.sort((a, b) => a.createdAt - b.createdAt).map(cloneEdge);
  }

  async deleteEdgesForRecord(recordId: string): Promise<void> {
    if (this.fallback) return this.fallback.deleteEdgesForRecord(recordId);
    const related = await this.listEdgesForRecord(recordId);
    if (related.length === 0) return;
    const tx = this.getDb().transaction('memory_edges', 'readwrite');
    const store = tx.objectStore('memory_edges');
    for (const edge of related) {
      store.delete(edge.id);
    }
    await idbTransactionDone(tx);
  }

  async upsertBlockedRule(rule: BlockedDomainRule): Promise<void> {
    if (this.fallback) return this.fallback.upsertBlockedRule(rule);
    const tx = this.getDb().transaction('blocked_domains', 'readwrite');
    tx.objectStore('blocked_domains').put({ ...rule });
    await idbTransactionDone(tx);
  }

  async listBlockedRules(): Promise<BlockedDomainRule[]> {
    if (this.fallback) return this.fallback.listBlockedRules();
    const all = await this.getAllFromStore<BlockedDomainRule>('blocked_domains');
    return all.sort((a, b) => a.createdAt - b.createdAt).map((r) => ({ ...r }));
  }

  async deleteBlockedRule(id: string): Promise<boolean> {
    if (this.fallback) return this.fallback.deleteBlockedRule(id);
    const all = await this.listBlockedRules();
    if (!all.some((r) => r.id === id)) return false;
    const tx = this.getDb().transaction('blocked_domains', 'readwrite');
    tx.objectStore('blocked_domains').delete(id);
    await idbTransactionDone(tx);
    return true;
  }

  async saveIndexSnapshot(name: string, bytes: Uint8Array): Promise<void> {
    if (this.fallback) return this.fallback.saveIndexSnapshot(name, bytes);
    const tx = this.getDb().transaction('index_snapshots', 'readwrite');
    tx.objectStore('index_snapshots').put({
      name,
      bytes: new Uint8Array(bytes),
      updatedAt: Date.now(),
    });
    await idbTransactionDone(tx);
  }

  async loadIndexSnapshot(name: string): Promise<Uint8Array | null> {
    if (this.fallback) return this.fallback.loadIndexSnapshot(name);
    const tx = this.getDb().transaction('index_snapshots', 'readonly');
    const row = await idbRequest(
      tx.objectStore('index_snapshots').get(name) as IDBRequest<{ name: string; bytes: Uint8Array } | undefined>
    );
    await idbTransactionDone(tx);
    return row?.bytes ? new Uint8Array(row.bytes) : null;
  }

  async clearAll(): Promise<void> {
    if (this.fallback) return this.fallback.clearAll();
    const tx = this.getDb().transaction([...STORES], 'readwrite');
    for (const name of STORES) {
      tx.objectStore(name).clear();
    }
    await idbTransactionDone(tx);
    this.recordIndex.clear();
    this.bm25.clear();
  }

  async close(): Promise<void> {
    if (this.fallback) {
      await this.fallback.close();
      this.fallback = null;
      return;
    }
    this.db?.close();
    this.db = null;
    this.recordIndex.clear();
    this.bm25.clear();
  }
}
