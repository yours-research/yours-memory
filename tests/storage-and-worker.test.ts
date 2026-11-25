import { describe, expect, it } from 'vitest';
import {
  createInMemoryWorkerPair,
  HybridMemoryEngine,
  IndexedDbStorageBackend,
  InMemoryStorageBackend,
  MemoryWorkerClient,
  MemoryWorkerHost,
  OpfsSqliteStorageBackend,
  OpfsSyncFileStore,
  SCHEMA_VERSION,
  WorkerCancelledError,
  type OpfsDirectoryHandleLike,
  type SqliteDatabase,
  type SqlValue,
} from '../src/index.js';

/** Minimal SQLite contract stub verifying DDL migrations, FTS5 queries, vectors, and edges. */
function createContractSqliteStub(): {
  db: SqliteDatabase;
  executedSql: string[];
} {
  const executedSql: string[] = [];
  const migrations: Array<{ version: number; name: string; applied_at: number }> = [];
  const memories = new Map<string, Record<string, SqlValue>>();
  const vectors = new Map<string, Record<string, SqlValue>>();
  const edges = new Map<string, Record<string, SqlValue>>();
  const blocked = new Map<string, Record<string, SqlValue>>();
  const snapshots = new Map<string, Record<string, SqlValue>>();

  const db: SqliteDatabase = {
    exec(sql: string, bind: SqlValue[] = []): void {
      executedSql.push(sql.trim());
      if (sql.includes('INSERT INTO schema_migrations')) {
        migrations.push({
          version: Number(bind[0]),
          name: String(bind[1]),
          applied_at: Number(bind[2]),
        });
      } else if (sql.includes('INSERT INTO memories ')) {
        memories.set(String(bind[0]), {
          id: bind[0]!,
          tier: bind[1]!,
          content: bind[2]!,
          source_url: bind[3]!,
          source_title: bind[4]!,
          thread_id: bind[5]!,
          tags_json: bind[6]!,
          importance: bind[7]!,
          tainted: bind[8]!,
          supersedes_id: bind[9]!,
          superseded_by_id: bind[10]!,
          created_at: bind[11]!,
          updated_at: bind[12]!,
          last_accessed_at: bind[13]!,
          access_count: bind[14]!,
        });
      } else if (sql.includes('DELETE FROM memories WHERE id = ?')) {
        memories.delete(String(bind[0]));
      } else if (sql.includes('DELETE FROM memories;')) {
        memories.clear();
      } else if (sql.includes('INSERT INTO memory_vectors')) {
        vectors.set(String(bind[0]), {
          memory_id: bind[0]!,
          dims: bind[1]!,
          quantization: bind[2]!,
          scale: bind[3]!,
          vector_blob: bind[4]!,
          updated_at: bind[5]!,
        });
      } else if (sql.includes('DELETE FROM memory_vectors WHERE memory_id = ?')) {
        vectors.delete(String(bind[0]));
      } else if (sql.includes('DELETE FROM memory_vectors;')) {
        vectors.clear();
      } else if (sql.includes('INSERT INTO memory_edges')) {
        edges.set(String(bind[0]), {
          id: bind[0]!,
          source_id: bind[1]!,
          target_id: bind[2]!,
          relation: bind[3]!,
          weight: bind[4]!,
          metadata_json: bind[5]!,
          created_at: bind[6]!,
        });
      } else if (sql.includes('DELETE FROM memory_edges WHERE source_id = ?')) {
        for (const [k, e] of edges) {
          if (e.source_id === bind[0] || e.target_id === bind[1]) edges.delete(k);
        }
      } else if (sql.includes('DELETE FROM memory_edges;')) {
        edges.clear();
      } else if (sql.includes('INSERT INTO blocked_domains')) {
        blocked.set(String(bind[0]), {
          id: bind[0]!,
          pattern: bind[1]!,
          kind: bind[2]!,
          reason: bind[3]!,
          builtin: bind[4]!,
          created_at: bind[5]!,
        });
      } else if (sql.includes('DELETE FROM blocked_domains WHERE id = ?')) {
        blocked.delete(String(bind[0]));
      } else if (sql.includes('DELETE FROM blocked_domains;')) {
        blocked.clear();
      } else if (sql.includes('INSERT INTO index_snapshots')) {
        snapshots.set(String(bind[0]), {
          name: bind[0]!,
          data_blob: bind[1]!,
          updated_at: bind[2]!,
        });
      } else if (sql.includes('DELETE FROM index_snapshots;')) {
        snapshots.clear();
      }
    },
    query<T extends Record<string, SqlValue>>(sql: string, bind: SqlValue[] = []): T[] {
      if (sql.includes('SELECT version FROM schema_migrations')) {
        return migrations.map((m) => ({ version: m.version }) as unknown as T);
      }
      if (sql.includes('SELECT MAX(version) AS v FROM schema_migrations')) {
        const max = migrations.reduce((acc, m) => Math.max(acc, m.version), 0);
        return [{ v: max } as unknown as T];
      }
      if (sql.includes('FROM memories WHERE id = ?')) {
        const row = memories.get(String(bind[0]));
        return row ? [row as unknown as T] : [];
      }
      if (sql.includes('FROM memories_fts')) {
        const matchTokens = String(bind[0] ?? '')
          .toLowerCase()
          .replace(/["*]/g, '')
          .split(/\s+or\s+/)
          .filter(Boolean);
        const rows = Array.from(memories.values())
          .filter((m) => m.superseded_by_id == null)
          .filter((m) =>
            matchTokens.some((tok) => String(m.content).toLowerCase().includes(tok))
          )
          .map((m) => ({ id: m.id, raw_bm25: -2.5 }) as unknown as T);
        return rows;
      }
      if (sql.includes('FROM memories')) {
        let rows = Array.from(memories.values());
        if (sql.includes('superseded_by_id IS NULL')) {
          rows = rows.filter((r) => r.superseded_by_id == null);
        }
        return rows as unknown as T[];
      }
      if (sql.includes('FROM memory_vectors WHERE memory_id = ?')) {
        const row = vectors.get(String(bind[0]));
        return row ? [row as unknown as T] : [];
      }
      if (sql.includes('FROM memory_vectors')) {
        return Array.from(vectors.values()) as unknown as T[];
      }
      if (sql.includes('FROM memory_edges')) {
        return Array.from(edges.values()) as unknown as T[];
      }
      if (sql.includes('FROM blocked_domains')) {
        return Array.from(blocked.values()) as unknown as T[];
      }
      if (sql.includes('FROM index_snapshots WHERE name = ?')) {
        const row = snapshots.get(String(bind[0]));
        return row ? [row as unknown as T] : [];
      }
      return [];
    },
    close(): void {},
  };

  return { db, executedSql };
}

function createFakeOpfsDirectory(): OpfsDirectoryHandleLike {
  const files = new Map<string, Uint8Array>();
  return {
    async getFileHandle(name: string) {
      if (!files.has(name)) {
        files.set(name, new Uint8Array(0));
      }
      return {
        async createSyncAccessHandle() {
          return {
            getSize() {
              return files.get(name)?.byteLength ?? 0;
            },
            truncate(newSize: number) {
              const cur = files.get(name) ?? new Uint8Array(0);
              const next = new Uint8Array(newSize);
              next.set(cur.subarray(0, Math.min(cur.byteLength, newSize)));
              files.set(name, next);
            },
            read(buffer: ArrayBufferView, options?: { at?: number }) {
              const at = options?.at ?? 0;
              const src = files.get(name) ?? new Uint8Array(0);
              const dst = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
              const slice = src.subarray(at, at + dst.byteLength);
              dst.set(slice);
              return slice.byteLength;
            },
            write(buffer: ArrayBufferView, options?: { at?: number }) {
              const at = options?.at ?? 0;
              const src = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
              const cur = files.get(name) ?? new Uint8Array(0);
              const needed = at + src.byteLength;
              const next = new Uint8Array(Math.max(cur.byteLength, needed));
              next.set(cur);
              next.set(src, at);
              files.set(name, next);
              return src.byteLength;
            },
            flush() {},
            close() {},
          };
        },
      };
    },
  };
}

describe('OPFS SQLite-WASM, IndexedDB fallback & Dedicated Worker RPC', () => {
  it('runs SQLite migrations, FTS5 lexical queries, and OPFS SyncAccessHandle sidecar snapshots', async () => {
    const { db, executedSql } = createContractSqliteStub();
    const opfsDir = createFakeOpfsDirectory();
    const backend = new OpfsSqliteStorageBackend({ db, opfsDirectory: opfsDir });

    const engine = new HybridMemoryEngine({ storage: backend });
    await engine.init();

    expect(executedSql.some((s) => s.includes('CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5'))).toBe(true);
    expect(db.query<{ v: number }>('SELECT MAX(version) AS v FROM schema_migrations;')[0]?.v).toBe(SCHEMA_VERSION);

    const saved = await engine.remember({
      tier: 'document',
      content: 'OPFS SyncAccessHandle gives synchronous zero-copy SQLite persistence inside a Dedicated Worker.',
      sourceTitle: 'MV3 OPFS Architecture',
    });

    const hits = await engine.recall({ query: 'OPFS SyncAccessHandle' });
    expect(hits[0]?.record.id).toBe(saved.id);
    expect((await engine.getStats()).backend).toBe('opfs-sqlite');

    // Verify OPFS sidecar file directly persisted the quantized HNSW snapshot
    const sidecar = new OpfsSyncFileStore(opfsDir, 'yours-memory-hnsw.bin');
    await sidecar.open();
    const sidecarBytes = sidecar.readAll();
    expect(sidecarBytes).not.toBeNull();
    expect(sidecarBytes!.byteLength).toBeGreaterThan(32);
    sidecar.close();

    await engine.close();
  });

  it('IndexedDbStorageBackend falls back cleanly in Node without global indexedDB', async () => {
    const backend = new IndexedDbStorageBackend();
    await backend.init();
    expect(backend.isUsingInMemoryFallback).toBe(true);

    const engine = new HybridMemoryEngine({ storage: backend });
    await engine.remember({ tier: 'working', content: 'Current task: implement IndexedDB fallback' });
    const hits = await engine.recall({ query: 'IndexedDB fallback' });
    expect(hits).toHaveLength(1);
    await engine.close();
  });

  it('Worker RPC executes full MemoryProvider lifecycle, sequentializes writes, and supports cancellation', async () => {
    const { clientEndpoint, hostEndpoint } = createInMemoryWorkerPair();
    const engine = new HybridMemoryEngine({
      storage: new InMemoryStorageBackend(),
      pbkdf2Iterations: 5_000,
    });
    const host = new MemoryWorkerHost({ engine, endpoint: hostEndpoint });
    const client = new MemoryWorkerClient(clientEndpoint, { timeoutMs: 5_000 });

    await host.notifyReady();

    // Concurrent writes over RPC are sequentialized cleanly
    const [r1, r2] = await Promise.all([
      client.remember({ tier: 'profile', content: 'My favorite language is Rust' }),
      client.remember({ tier: 'working', content: 'Exploring Chrome MV3 offscreen documents' }),
    ]);

    expect(r1.id).toBeTruthy();
    expect(r2.id).toBeTruthy();

    const hits = await client.recall({ query: 'favorite language Rust' });
    expect(hits[0]?.record.id).toBe(r1.id);

    // Pre-aborted signal rejects immediately with WorkerCancelledError
    const preAbort = new AbortController();
    preAbort.abort();
    await expect(client.recall({ query: 'Rust' }, preAbort.signal)).rejects.toBeInstanceOf(
      WorkerCancelledError
    );

    // Mid-flight cancellation rejects cleanly
    const midAbort = new AbortController();
    const pendingCall = client.inspect(undefined, midAbort.signal);
    midAbort.abort();
    await expect(pendingCall).rejects.toBeInstanceOf(WorkerCancelledError);

    // Encrypted export/import roundtrip across worker boundary (Transferable Uint8Array)
    const blob = await client.exportEncrypted('worker-passphrase');
    await client.wipeAll();
    expect(await client.inspect()).toHaveLength(0);

    const imported = await client.importEncrypted(blob, 'worker-passphrase');
    expect(imported.imported).toBe(2);

    const stats = await client.getStats();
    expect(stats.activeRecords).toBe(2);

    await client.close(true);
    host.detach();
  });
});
