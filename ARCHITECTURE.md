# `@yours/memory` Architecture

`@yours/memory` implements the `MemoryProvider` contract defined in [docs/ECOSYSTEM_CONTRACTS.md](./docs/ECOSYSTEM_CONTRACTS.md) with zero external network calls and strict Chrome Manifest V3 isolation.

## 1. Module Layout

```text
src/
├── contracts.ts              # Canonical MemoryProvider, MemoryRecord, MemoryQuery, MemoryHit types
├── errors.ts                 # Typed MemoryError subclasses (SensitiveDomainBlockedError, TaintViolationError, ...)
├── engine.ts                 # HybridMemoryEngine orchestrating storage, HNSW, RRF, lifecycle & privacy
├── index.ts                  # Public ESM exports
├── utils/
│   ├── id.ts                 # WebCrypto collision-resistant IDs
│   ├── math.ts               # Vector norms, cosine distance, int8 symmetric scalar quantization
│   └── text.ts               # Unicode61 + camelCase/snake_case tokenizer & FTS5 query builder
├── retrieval/
│   ├── embeddings.ts         # DeterministicHashEmbedder & WebGpuEmbedderAdapter
│   ├── bm25.ts               # Pure TypeScript BM25 inverted index
│   ├── hnsw.ts               # Native HNSW ANN graph with int8/f32 storage & YHNW binary serialization
│   └── ranking.ts            # Reciprocal Rank Fusion (RRF) + per-tier temporal salience decay
├── storage/
│   ├── types.ts              # Unified MemoryStorageBackend interface
│   ├── migrations.ts         # Versioned SQLite DDL (memories, memories_fts, memory_vectors, memory_edges, blocked_domains)
│   ├── sqlite-opfs.ts        # OpfsSqliteStorageBackend + OpfsSyncFileStore (Dedicated Worker SyncAccessHandle)
│   ├── indexeddb.ts          # IndexedDbStorageBackend with hydrated BM25 & InMemory fallback
│   ├── in-memory.ts          # Reference InMemoryStorageBackend
│   └── auto.ts               # Runtime environment detector & backend selector
├── lifecycle/
│   ├── dedup.ts              # Canonical + cosine near-duplicate detection & reinforcement
│   ├── contradiction.ts      # Slot, polarity-flip & antonym contradiction detector + non-destructive superseding
│   └── consolidator.ts       # Thread consolidation into untainted profile facts & episodic summaries
├── privacy/
│   ├── blocklist.ts          # Built-in & custom glob/regex/exact sensitive URL filter
│   ├── taint.ts              # Untrusted content taint boundary enforcement
│   └── crypto.ts             # PBKDF2-SHA256 + AES-256-GCM authenticated backup envelope (YMEM v1)
└── worker/
    ├── protocol.ts           # Typed RPC request/response/cancel message protocol
    ├── host.ts               # MemoryWorkerHost with sequentialized write queue & AbortController cancellation
    ├── client.ts             # MemoryWorkerClient implementing MemoryProvider over Worker/MessagePort
    └── mv3.ts                # MV3 worker bootstrapper & in-memory MessageChannel pair for testing
```

## 2. Chrome MV3 Single-Owner Worker Topology

In Chrome MV3:
1. OPFS `FileSystemFileHandle.createSyncAccessHandle()` is synchronous and **only available inside Dedicated Web Workers** (not Service Workers or Window pages).
2. Opening the same OPFS SQLite file from multiple contexts causes `NoModificationAllowedError` lock contention.

Therefore, `@yours/memory` routes all reads and writes through a **single Dedicated Worker** (`MemoryWorkerHost`) typically spawned by an **MV3 Offscreen Document** or **Extension Sidepanel**. `MemoryWorkerHost` serializes mutating operations (`remember`, `consolidate`, `forget`, `importEncrypted`, `wipeAll`) through an async write queue (`enqueueWrite`), while reads (`recall`, `inspect`, `inspectGraph`) run safely against consistent state.

## 3. Hybrid Retrieval & Quantization

1. **Lexical Branch:**
   - On OPFS SQLite-WASM: `memories_fts` (`USING fts5(..., tokenize='unicode61')`) kept synchronized via `AFTER INSERT/UPDATE/DELETE` triggers, queried with `bm25(memories_fts)`.
   - On IndexedDB / InMemory: `Bm25Index` with matching code-aware identifier splitting (`camelCase` / `snake_case`) and `k1=1.2, b=0.75`.
2. **Semantic Branch (`int8` HNSW):**
   - Vectors are normalized and stored with symmetric `int8` scalar quantization (`4×` smaller than `Float32Array`, storing `Int8Array` + `float32` scale per node).
   - Query evaluation uses asymmetric `float32 × int8` dot products, preserving high recall while cutting OPFS snapshot size and worker RAM by ~4×.
3. **Fusion & Temporal Salience Decay:**
   - Ranks from both branches are fused via **Reciprocal Rank Fusion (`RRF`)**: $\sum w_i / (k + \text{rank}_i)$ with $k=60$.
   - Scores are modulated by tier-specific exponential half-life decay:
     - `profile`: 180-day half-life, `0.65` floor (durable facts remain salient)
     - `document`: 60-day half-life, `0.25` floor
     - `episodic`: 30-day half-life, `0.15` floor
     - `working`: 12-hour half-life, `0.05` floor (ephemeral tab/task context fades rapidly)
   - Reinforced logarithmically by `accessCount` and linearly by `importance`, with a penalty factor for `tainted` records.

## 4. Provenance, Contradictions & Security

- **Non-Destructive Superseding:** Updating a preference (e.g., *"My favorite editor is Vim"* $\rightarrow$ *"My favorite editor is VS Code"*) marks the old record's `supersededById`, adds a `supersedes` edge in `memory_edges`, and removes the stale vector from active HNSW search while keeping the full provenance chain inspectable via `inspectGraph()`.
- **Taint Isolation:** Content from untrusted web pages or tool calls (`tainted: true`) cannot silently overwrite or supersede untainted `profile` records (`TaintViolationError`) unless explicitly confirmed by the user (`confirmTaintedProfileMutation: true`).
- **Sensitive-Domain Filtering:** Built-in rules reject automatic capture from `chrome://`, `chrome-extension://`, `edge://`, `about:`, `devtools://`, banking/auth/password-manager domains, localhost secret/admin endpoints, and URLs containing token/credential query parameters.
- **Zero-Knowledge Local Backups:** `exportEncrypted(passphrase)` derives a 256-bit AES-GCM key via PBKDF2-SHA256 (`210,000` iterations, random 16-byte salt, random 12-byte IV) with WebCrypto `crypto.subtle`.
