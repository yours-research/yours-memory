/** Versioned SQLite schema migrations for OPFS SQLite-WASM storage. */

export const SCHEMA_VERSION = 1;

export interface SqlMigration {
  version: number;
  name: string;
  statements: readonly string[];
}

export const SQLITE_MIGRATIONS: readonly SqlMigration[] = [
  {
    version: 1,
    name: 'init_hybrid_memory_schema',
    statements: [
      `CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at INTEGER NOT NULL
      );`,
      `CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        tier TEXT NOT NULL CHECK (tier IN ('working', 'profile', 'episodic', 'document')),
        content TEXT NOT NULL,
        source_url TEXT,
        source_title TEXT,
        thread_id TEXT,
        tags_json TEXT NOT NULL DEFAULT '[]',
        importance REAL NOT NULL DEFAULT 0.5,
        tainted INTEGER NOT NULL DEFAULT 0,
        supersedes_id TEXT,
        superseded_by_id TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        last_accessed_at INTEGER NOT NULL,
        access_count INTEGER NOT NULL DEFAULT 1
      );`,
      `CREATE INDEX IF NOT EXISTS idx_memories_tier_active ON memories (tier, superseded_by_id, updated_at DESC);`,
      `CREATE INDEX IF NOT EXISTS idx_memories_thread ON memories (thread_id, created_at DESC);`,
      `CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        id UNINDEXED,
        content,
        source_title,
        tags,
        tokenize='unicode61'
      );`,
      `CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts (id, content, source_title, tags)
        VALUES (new.id, new.content, COALESCE(new.source_title, ''), new.tags_json);
      END;`,
      `CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
        DELETE FROM memories_fts WHERE id = old.id;
        INSERT INTO memories_fts (id, content, source_title, tags)
        VALUES (new.id, new.content, COALESCE(new.source_title, ''), new.tags_json);
      END;`,
      `CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        DELETE FROM memories_fts WHERE id = old.id;
      END;`,
      `CREATE TABLE IF NOT EXISTS memory_vectors (
        memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
        dims INTEGER NOT NULL,
        quantization TEXT NOT NULL CHECK (quantization IN ('int8', 'f32')),
        scale REAL NOT NULL DEFAULT 1.0,
        vector_blob BLOB NOT NULL,
        updated_at INTEGER NOT NULL
      );`,
      `CREATE TABLE IF NOT EXISTS memory_edges (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        relation TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        metadata_json TEXT,
        created_at INTEGER NOT NULL
      );`,
      `CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges (source_id);`,
      `CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges (target_id);`,
      `CREATE TABLE IF NOT EXISTS blocked_domains (
        id TEXT PRIMARY KEY,
        pattern TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('glob', 'regex', 'exact')),
        reason TEXT NOT NULL,
        builtin INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL
      );`,
      `CREATE TABLE IF NOT EXISTS index_snapshots (
        name TEXT PRIMARY KEY,
        data_blob BLOB NOT NULL,
        updated_at INTEGER NOT NULL
      );`,
    ],
  },
];
