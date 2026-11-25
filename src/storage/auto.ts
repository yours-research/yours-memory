import { InMemoryStorageBackend } from './in-memory.js';
import { IndexedDbStorageBackend } from './indexeddb.js';
import { OpfsSqliteStorageBackend, type OpfsDirectoryHandleLike, type SqliteDatabase } from './sqlite-opfs.js';
import type { MemoryStorageBackend } from './types.js';

export interface StorageEnvironmentCapabilities {
  isDedicatedWorker: boolean;
  hasOpfs: boolean;
  hasSyncAccessHandle: boolean;
  hasIndexedDb: boolean;
}

export function detectStorageCapabilities(): StorageEnvironmentCapabilities {
  const g = globalThis as Record<string, unknown>;
  const isDedicatedWorker =
    typeof g.DedicatedWorkerGlobalScope !== 'undefined' ||
    (typeof g.importScripts === 'function' && typeof g.document === 'undefined');

  const nav = g.navigator as { storage?: { getDirectory?: unknown } } | undefined;
  const hasOpfs = Boolean(nav?.storage && typeof nav.storage.getDirectory === 'function');
  const hasSyncAccessHandle =
    isDedicatedWorker &&
    hasOpfs &&
    typeof g.FileSystemFileHandle !== 'undefined' &&
    typeof (g.FileSystemFileHandle as { prototype?: { createSyncAccessHandle?: unknown } }).prototype
      ?.createSyncAccessHandle === 'function';

  const hasIndexedDb = typeof g.indexedDB !== 'undefined';

  return {
    isDedicatedWorker,
    hasOpfs,
    hasSyncAccessHandle,
    hasIndexedDb,
  };
}

export interface AutoStorageOptions {
  dbName?: string;
  openSqliteDb?: () => Promise<SqliteDatabase> | SqliteDatabase;
  opfsDirectory?: OpfsDirectoryHandleLike;
  preferBackend?: 'auto' | 'opfs-sqlite' | 'indexeddb' | 'in-memory';
}

/**
 * Selects the strongest available local storage backend:
 * 1. OPFS SQLite-WASM (`SyncAccessHandle` in Dedicated Worker) when `openSqliteDb` is provided
 * 2. IndexedDB (when available in extension/browser contexts)
 * 3. InMemoryStorageBackend (Node/Vitest or ephemeral mode)
 */
export async function createBestStorageBackend(options: AutoStorageOptions = {}): Promise<MemoryStorageBackend> {
  const prefer = options.preferBackend ?? 'auto';
  const caps = detectStorageCapabilities();

  if (prefer === 'in-memory') {
    const backend = new InMemoryStorageBackend();
    await backend.init();
    return backend;
  }

  if (prefer === 'opfs-sqlite' || (prefer === 'auto' && options.openSqliteDb && (caps.hasSyncAccessHandle || caps.isDedicatedWorker))) {
    if (options.openSqliteDb) {
      let dir = options.opfsDirectory;
      if (!dir && caps.hasOpfs) {
        dir = (await navigator.storage.getDirectory()) as unknown as OpfsDirectoryHandleLike;
      }
      const backend = new OpfsSqliteStorageBackend({
        openDb: options.openSqliteDb,
        opfsDirectory: dir,
      });
      await backend.init();
      return backend;
    }
    if (prefer === 'opfs-sqlite') {
      throw new Error('opfs-sqlite backend requested without an `openSqliteDb` factory');
    }
  }

  if (prefer === 'indexeddb' || (prefer === 'auto' && caps.hasIndexedDb)) {
    const backend = new IndexedDbStorageBackend({
      dbName: options.dbName,
      requireIndexedDb: prefer === 'indexeddb',
    });
    await backend.init();
    return backend;
  }

  const backend = new InMemoryStorageBackend();
  await backend.init();
  return backend;
}
