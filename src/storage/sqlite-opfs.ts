import type {
  BlockedDomainRule,
  InspectOptions,
  MemoryEdge,
  MemoryEdgeRelation,
  MemoryRecord,
  MemoryTier,
  QuantizationMode,
} from '../contracts.js';
import { buildFts5MatchQuery } from '../utils/text.js';
import { SCHEMA_VERSION, SQLITE_MIGRATIONS } from './migrations.js';
import type { LexicalSearchOptions, LexicalSearchResult, MemoryStorageBackend, StoredVectorRecord } from './types.js';

export type SqlValue = string | number | Uint8Array | null;

/**
 * Minimal synchronous/async-compatible SQLite database driver contract.
 * Compatible with `@sqlite.org/sqlite-wasm` (`oo1.OpfsDb` / `oo1.DB`) via a lightweight adapter.
 */
export interface SqliteDatabase {
  exec(sql: string, bind?: SqlValue[]): void;
  query<T extends Record<string, SqlValue> = Record<string, SqlValue>>(sql: string, bind?: SqlValue[]): T[];
  close?(): void;
}

/**
 * Wraps an `@sqlite.org/sqlite-wasm` `oo1.DB` / `oo1.OpfsDb` instance into `SqliteDatabase`.
 */
export interface SqliteWasmDbLike {
  exec(options: {
    sql: string;
    bind?: SqlValue[];
    rowMode?: 'object' | 'array';
    returnValue?: 'resultRows' | 'this';
  }): unknown;
  close?(): void;
}

export function adaptSqliteWasmDb(db: SqliteWasmDbLike): SqliteDatabase {
  return {
    exec(sql: string, bind?: SqlValue[]): void {
      db.exec({ sql, bind });
    },
    query<T extends Record<string, SqlValue>>(sql: string, bind?: SqlValue[]): T[] {
      const rows = db.exec({
        sql,
        bind,
        rowMode: 'object',
        returnValue: 'resultRows',
      });
      return Array.isArray(rows) ? (rows as T[]) : [];
    },
    close(): void {
      db.close?.();
    },
  };
}

/** Minimal OPFS FileSystemSyncAccessHandle contract for Dedicated Worker binary persistence. */
export interface OpfsSyncAccessHandleLike {
  getSize(): number;
  truncate(newSize: number): void;
  read(buffer: ArrayBufferView, options?: { at?: number }): number;
  write(buffer: ArrayBufferView, options?: { at?: number }): number;
  flush(): void;
  close(): void;
}

export interface OpfsFileHandleLike {
  createSyncAccessHandle(): Promise<OpfsSyncAccessHandleLike>;
}

export interface OpfsDirectoryHandleLike {
  getFileHandle(name: string, options?: { create?: boolean }): Promise<OpfsFileHandleLike>;
  removeEntry?(name: string): Promise<void>;
}

/**
 * Single-writer OPFS SyncAccessHandle binary file helper for Dedicated Workers.
 * Persists WAL / HNSW snapshots without IndexedDB serialization overhead.
 */
export class OpfsSyncFileStore {
  private handle: OpfsSyncAccessHandleLike | null = null;

  constructor(
    private readonly dir: OpfsDirectoryHandleLike,
    private readonly fileName: string
  ) {}

  async open(): Promise<void> {
    if (this.handle) return;
    const fileHandle = await this.dir.getFileHandle(this.fileName, { create: true });
    this.handle = await fileHandle.createSyncAccessHandle();
  }

  readAll(): Uint8Array | null {
    if (!this.handle) {
      throw new Error(`OPFS file "${this.fileName}" is not open`);
    }
    const size = this.handle.getSize();
    if (size === 0) return null;
    const buf = new Uint8Array(size);
    this.handle.read(buf, { at: 0 });
    return buf;
  }

  writeAll(bytes: Uint8Array): void {
    if (!this.handle) {
      throw new Error(`OPFS file "${this.fileName}" is not open`);
    }
    this.handle.truncate(0);
    if (bytes.byteLength > 0) {
      this.handle.write(bytes, { at: 0 });
    }
    this.handle.flush();
  }

  clear(): void {
    if (!this.handle) return;
    this.handle.truncate(0);
    this.handle.flush();
  }

  close(): void {
    if (!this.handle) return;
    try {
      this.handle.flush();
    } finally {
      this.handle.close();
      this.handle = null;
    }
  }
}

export interface OpfsSqliteOptions {
  /** Pre-opened SQLite database (e.g., adapted from `@sqlite.org/sqlite-wasm` `oo1.OpfsDb`). */
  db?: SqliteDatabase;
  /** Factory invoked on `init()` to construct the SQLite database inside the worker. */
  openDb?: () => Promise<SqliteDatabase> | SqliteDatabase;
  /** Optional OPFS directory handle for SyncAccessHandle sidecar snapshots (e.g. quantized HNSW index). */
  opfsDirectory?: OpfsDirectoryHandleLike;
  /** File name for the optional HNSW OPFS sidecar snapshot. */
  hnswSidecarFileName?: string;
}

function parseJsonStringArray(raw: SqlValue): string[] | undefined {
  if (typeof raw !== 'string' || !raw) return undefined;
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(String) : undefined;
  } catch {
    return undefined;
  }
}

function parseJsonObject(raw: SqlValue): Record<string, unknown> | undefined {
  if (typeof raw !== 'string' || !raw) return undefined;
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : undefined;
  } catch {
    return undefined;
  }
}

function toUint8Array(val: SqlValue): Uint8Array {
  if (val instanceof Uint8Array) return new Uint8Array(val);
  return new Uint8Array(0);
}

/**
 * OPFS SQLite-WASM + FTS5 + quantized vector storage backend.
 * Designed for single-owner Dedicated Web Workers to avoid MV3 multi-writer OPFS lock contention.
 */
export class OpfsSqliteStorageBackend implements MemoryStorageBackend {
  readonly kind = 'opfs-sqlite' as const;

  private db: SqliteDatabase | null;
  private readonly openDb?: () => Promise<SqliteDatabase> | SqliteDatabase;
  private readonly sidecarStore?: OpfsSyncFileStore;
  private initialized = false;

  constructor(options: OpfsSqliteOptions) {
    this.db = options.db ?? null;
    this.openDb = options.openDb;
    if (options.opfsDirectory) {
      this.sidecarStore = new OpfsSyncFileStore(
        options.opfsDirectory,
        options.hnswSidecarFileName ?? 'yours-memory-hnsw.bin'
      );
    }
  }

  private getDb(): SqliteDatabase {
    if (!this.db) {
      throw new Error('OpfsSqliteStorageBackend is not initialized');
    }
    return this.db;
  }

  async init(): Promise<void> {
    if (this.initialized) return;
    if (!this.db && this.openDb) {
      this.db = await this.openDb();
    }
    if (!this.db) {
      throw new Error('OpfsSqliteStorageBackend requires either `db` or `openDb`');
    }
    if (this.sidecarStore) {
      await this.sidecarStore.open();
    }

    const db = this.db;
    db.exec('PRAGMA foreign_keys = ON;');
    db.exec(`CREATE TABLE IF NOT EXISTS schema_migrations (
      version INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      applied_at INTEGER NOT NULL
    );`);

    const appliedRows = db.query<{ version: number }>('SELECT version FROM schema_migrations;');
    const applied = new Set(appliedRows.map((r) => Number(r.version)));

    for (const migration of SQLITE_MIGRATIONS) {
      if (applied.has(migration.version)) continue;
      db.exec('BEGIN IMMEDIATE;');
      try {
        for (const stmt of migration.statements) {
          db.exec(stmt);
        }
        db.exec('INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?);', [
          migration.version,
          migration.name,
          Date.now(),
        ]);
        db.exec('COMMIT;');
      } catch (err) {
        try {
          db.exec('ROLLBACK;');
        } catch {
          // Ignore rollback failure.
        }
        throw err;
      }
    }

    const maxRow = db.query<{ v: number }>('SELECT MAX(version) AS v FROM schema_migrations;');
    const currentVersion = Number(maxRow[0]?.v ?? 0);
    if (currentVersion < SCHEMA_VERSION) {
      throw new Error(`SQLite migration incomplete: expected ${SCHEMA_VERSION}, got ${currentVersion}`);
    }
    this.initialized = true;
  }

  async upsertRecord(record: MemoryRecord): Promise<void> {
    const db = this.getDb();
    db.exec(
      `INSERT INTO memories (
        id, tier, content, source_url, source_title, thread_id, tags_json,
        importance, tainted, supersedes_id, superseded_by_id,
        created_at, updated_at, last_accessed_at, access_count
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        tier = excluded.tier,
        content = excluded.content,
        source_url = excluded.source_url,
        source_title = excluded.source_title,
        thread_id = excluded.thread_id,
        tags_json = excluded.tags_json,
        importance = excluded.importance,
        tainted = excluded.tainted,
        supersedes_id = excluded.supersedes_id,
        superseded_by_id = excluded.superseded_by_id,
        updated_at = excluded.updated_at,
        last_accessed_at = excluded.last_accessed_at,
        access_count = excluded.access_count;`,
      [
        record.id,
        record.tier,
        record.content,
        record.sourceUrl ?? null,
        record.sourceTitle ?? null,
        record.threadId ?? null,
        JSON.stringify(record.tags ?? []),
        record.importance ?? 0.5,
        record.tainted ? 1 : 0,
        record.supersedesId ?? null,
        record.supersededById ?? null,
        record.createdAt,
        record.updatedAt,
        record.lastAccessedAt,
        record.accessCount,
      ]
    );
  }

  private mapRowToRecord(row: Record<string, SqlValue>): MemoryRecord {
    const tags = parseJsonStringArray(row.tags_json ?? null);
    return {
      id: String(row.id),
      tier: String(row.tier) as MemoryTier,
      content: String(row.content),
      sourceUrl: row.source_url != null ? String(row.source_url) : undefined,
      sourceTitle: row.source_title != null ? String(row.source_title) : undefined,
      threadId: row.thread_id != null ? String(row.thread_id) : undefined,
      tags: tags && tags.length > 0 ? tags : undefined,
      importance: row.importance != null ? Number(row.importance) : undefined,
      tainted: Number(row.tainted ?? 0) === 1 ? true : undefined,
      supersedesId: row.supersedes_id != null ? String(row.supersedes_id) : undefined,
      supersededById: row.superseded_by_id != null ? String(row.superseded_by_id) : undefined,
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
      lastAccessedAt: Number(row.last_accessed_at),
      accessCount: Number(row.access_count),
    };
  }

  async getRecord(id: string): Promise<MemoryRecord | null> {
    const rows = this.getDb().query('SELECT * FROM memories WHERE id = ? LIMIT 1;', [id]);
    const row = rows[0];
    return row ? this.mapRowToRecord(row) : null;
  }

  async listRecords(filter?: InspectOptions): Promise<MemoryRecord[]> {
    const clauses: string[] = [];
    const params: SqlValue[] = [];

    if (!filter?.includeSuperseded) {
      clauses.push('superseded_by_id IS NULL');
    }
    if (filter?.tier) {
      clauses.push('tier = ?');
      params.push(filter.tier);
    }
    if (filter?.threadId) {
      clauses.push('thread_id = ?');
      params.push(filter.threadId);
    }
    if (filter?.tainted !== undefined) {
      clauses.push('tainted = ?');
      params.push(filter.tainted ? 1 : 0);
    }

    const where = clauses.length > 0 ? `WHERE ${clauses.join(' AND ')}` : '';
    const limitSql = filter?.limit !== undefined ? 'LIMIT ?' : '';
    const offsetSql = filter?.offset !== undefined ? 'OFFSET ?' : '';
    if (filter?.limit !== undefined) params.push(Math.max(0, filter.limit));
    if (filter?.offset !== undefined) {
      if (filter.limit === undefined) {
        // SQLite requires LIMIT before OFFSET.
        params.push(-1);
      }
      params.push(Math.max(0, filter.offset));
    }

    const sql = `SELECT * FROM memories ${where} ORDER BY updated_at DESC, created_at DESC, id ASC ${
      filter?.offset !== undefined && filter.limit === undefined ? 'LIMIT ? OFFSET ?' : `${limitSql} ${offsetSql}`
    };`;
    const rows = this.getDb().query(sql, params);
    return rows.map((r) => this.mapRowToRecord(r));
  }

  async deleteRecord(id: string): Promise<boolean> {
    const db = this.getDb();
    const existing = await this.getRecord(id);
    if (!existing) return false;
    db.exec('DELETE FROM memory_edges WHERE source_id = ? OR target_id = ?;', [id, id]);
    db.exec('DELETE FROM memory_vectors WHERE memory_id = ?;', [id]);
    db.exec('DELETE FROM memories WHERE id = ?;', [id]);
    return true;
  }

  async upsertVector(vector: StoredVectorRecord): Promise<void> {
    this.getDb().exec(
      `INSERT INTO memory_vectors (memory_id, dims, quantization, scale, vector_blob, updated_at)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(memory_id) DO UPDATE SET
         dims = excluded.dims,
         quantization = excluded.quantization,
         scale = excluded.scale,
         vector_blob = excluded.vector_blob,
         updated_at = excluded.updated_at;`,
      [
        vector.memoryId,
        vector.dims,
        vector.quantization,
        vector.scale,
        new Uint8Array(vector.data),
        vector.updatedAt,
      ]
    );
  }

  async getVector(memoryId: string): Promise<StoredVectorRecord | null> {
    const rows = this.getDb().query(
      'SELECT memory_id, dims, quantization, scale, vector_blob, updated_at FROM memory_vectors WHERE memory_id = ? LIMIT 1;',
      [memoryId]
    );
    const row = rows[0];
    if (!row) return null;
    return {
      memoryId: String(row.memory_id),
      dims: Number(row.dims),
      quantization: String(row.quantization) as QuantizationMode,
      scale: Number(row.scale),
      data: toUint8Array(row.vector_blob ?? null),
      updatedAt: Number(row.updated_at),
    };
  }

  async listVectors(memoryIds?: string[]): Promise<StoredVectorRecord[]> {
    const db = this.getDb();
    let rows: Record<string, SqlValue>[];
    if (memoryIds && memoryIds.length > 0) {
      const placeholders = memoryIds.map(() => '?').join(', ');
      rows = db.query(
        `SELECT memory_id, dims, quantization, scale, vector_blob, updated_at FROM memory_vectors WHERE memory_id IN (${placeholders});`,
        memoryIds
      );
    } else if (memoryIds && memoryIds.length === 0) {
      return [];
    } else {
      rows = db.query(
        'SELECT memory_id, dims, quantization, scale, vector_blob, updated_at FROM memory_vectors;'
      );
    }
    return rows.map((row) => ({
      memoryId: String(row.memory_id),
      dims: Number(row.dims),
      quantization: String(row.quantization) as QuantizationMode,
      scale: Number(row.scale),
      data: toUint8Array(row.vector_blob ?? null),
      updatedAt: Number(row.updated_at),
    }));
  }

  async deleteVector(memoryId: string): Promise<boolean> {
    const existing = await this.getVector(memoryId);
    if (!existing) return false;
    this.getDb().exec('DELETE FROM memory_vectors WHERE memory_id = ?;', [memoryId]);
    return true;
  }

  async searchLexical(query: string, options: LexicalSearchOptions = {}): Promise<LexicalSearchResult[]> {
    const matchExpr = buildFts5MatchQuery(query);
    if (!matchExpr) return [];

    const db = this.getDb();
    const limit = Math.max(1, options.limit ?? 20);
    const clauses: string[] = ['memories_fts MATCH ?'];
    const params: SqlValue[] = [matchExpr];

    if (!options.includeSuperseded) {
      clauses.push('m.superseded_by_id IS NULL');
    }
    if (options.tiers && options.tiers.length > 0) {
      clauses.push(`m.tier IN (${options.tiers.map(() => '?').join(', ')})`);
      params.push(...options.tiers);
    }
    params.push(limit);

    // SQLite FTS5 bm25() returns negative scores (more negative = better match).
    const rows = db.query<{ id: string; raw_bm25: number }>(
      `SELECT m.id AS id, bm25(memories_fts) AS raw_bm25
       FROM memories_fts
       JOIN memories m ON m.id = memories_fts.id
       WHERE ${clauses.join(' AND ')}
       ORDER BY raw_bm25 ASC, m.id ASC
       LIMIT ?;`,
      params
    );

    return rows.map((row, idx) => ({
      id: String(row.id),
      score: Math.abs(Number(row.raw_bm25)),
      rank: idx + 1,
    }));
  }

  async upsertEdge(edge: MemoryEdge): Promise<void> {
    this.getDb().exec(
      `INSERT INTO memory_edges (id, source_id, target_id, relation, weight, metadata_json, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         source_id = excluded.source_id,
         target_id = excluded.target_id,
         relation = excluded.relation,
         weight = excluded.weight,
         metadata_json = excluded.metadata_json;`,
      [
        edge.id,
        edge.sourceId,
        edge.targetId,
        edge.relation,
        edge.weight,
        edge.metadata ? JSON.stringify(edge.metadata) : null,
        edge.createdAt,
      ]
    );
  }

  private mapEdge(row: Record<string, SqlValue>): MemoryEdge {
    return {
      id: String(row.id),
      sourceId: String(row.source_id),
      targetId: String(row.target_id),
      relation: String(row.relation) as MemoryEdgeRelation,
      weight: Number(row.weight),
      metadata: parseJsonObject(row.metadata_json ?? null),
      createdAt: Number(row.created_at),
    };
  }

  async listEdgesForRecord(recordId: string): Promise<MemoryEdge[]> {
    const rows = this.getDb().query(
      'SELECT * FROM memory_edges WHERE source_id = ? OR target_id = ? ORDER BY created_at ASC;',
      [recordId, recordId]
    );
    return rows.map((r) => this.mapEdge(r));
  }

  async listAllEdges(): Promise<MemoryEdge[]> {
    const rows = this.getDb().query('SELECT * FROM memory_edges ORDER BY created_at ASC;');
    return rows.map((r) => this.mapEdge(r));
  }

  async deleteEdgesForRecord(recordId: string): Promise<void> {
    this.getDb().exec('DELETE FROM memory_edges WHERE source_id = ? OR target_id = ?;', [recordId, recordId]);
  }

  async upsertBlockedRule(rule: BlockedDomainRule): Promise<void> {
    this.getDb().exec(
      `INSERT INTO blocked_domains (id, pattern, kind, reason, builtin, created_at)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         pattern = excluded.pattern,
         kind = excluded.kind,
         reason = excluded.reason,
         builtin = excluded.builtin;`,
      [rule.id, rule.pattern, rule.kind, rule.reason, rule.builtin ? 1 : 0, rule.createdAt]
    );
  }

  async listBlockedRules(): Promise<BlockedDomainRule[]> {
    const rows = this.getDb().query('SELECT * FROM blocked_domains ORDER BY created_at ASC;');
    return rows.map((row) => ({
      id: String(row.id),
      pattern: String(row.pattern),
      kind: String(row.kind) as BlockedDomainRule['kind'],
      reason: String(row.reason),
      builtin: Number(row.builtin ?? 0) === 1,
      createdAt: Number(row.created_at),
    }));
  }

  async deleteBlockedRule(id: string): Promise<boolean> {
    const existing = this.getDb().query('SELECT id FROM blocked_domains WHERE id = ? LIMIT 1;', [id]);
    if (existing.length === 0) return false;
    this.getDb().exec('DELETE FROM blocked_domains WHERE id = ?;', [id]);
    return true;
  }

  async saveIndexSnapshot(name: string, bytes: Uint8Array): Promise<void> {
    if (this.sidecarStore && name === 'hnsw') {
      this.sidecarStore.writeAll(bytes);
    }
    this.getDb().exec(
      `INSERT INTO index_snapshots (name, data_blob, updated_at)
       VALUES (?, ?, ?)
       ON CONFLICT(name) DO UPDATE SET
         data_blob = excluded.data_blob,
         updated_at = excluded.updated_at;`,
      [name, new Uint8Array(bytes), Date.now()]
    );
  }

  async loadIndexSnapshot(name: string): Promise<Uint8Array | null> {
    if (this.sidecarStore && name === 'hnsw') {
      const sidecarBytes = this.sidecarStore.readAll();
      if (sidecarBytes && sidecarBytes.byteLength > 0) {
        return sidecarBytes;
      }
    }
    const rows = this.getDb().query<{ data_blob: Uint8Array }>(
      'SELECT data_blob FROM index_snapshots WHERE name = ? LIMIT 1;',
      [name]
    );
    const row = rows[0];
    return row ? toUint8Array(row.data_blob) : null;
  }

  async clearAll(): Promise<void> {
    const db = this.getDb();
    db.exec('DELETE FROM memory_edges;');
    db.exec('DELETE FROM memory_vectors;');
    db.exec('DELETE FROM memories;');
    db.exec('DELETE FROM blocked_domains;');
    db.exec('DELETE FROM index_snapshots;');
    this.sidecarStore?.clear();
  }

  async close(): Promise<void> {
    this.sidecarStore?.close();
    this.db?.close?.();
    this.db = null;
    this.initialized = false;
  }
}
