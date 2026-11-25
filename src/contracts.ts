/**
 * Canonical @yours/memory & @yours/harness TypeScript contracts.
 * Kept strictly synchronized with docs/ECOSYSTEM_CONTRACTS.md (v1.0).
 */

export type MemoryTier = 'working' | 'profile' | 'episodic' | 'document';

export interface HarnessMessage {
  id: string;
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  toolName?: string;
  toolCallId?: string;
  toolArgs?: Record<string, unknown>;
  tainted?: boolean;
  createdAt: number;
}

export interface MemoryRecordInput {
  tier: MemoryTier;
  content: string;
  sourceUrl?: string;
  sourceTitle?: string;
  threadId?: string;
  tags?: string[];
  importance?: number; // [0..1]
  tainted?: boolean;
  supersedesId?: string;
}

export interface MemoryRecord extends MemoryRecordInput {
  id: string;
  createdAt: number;
  updatedAt: number;
  lastAccessedAt: number;
  accessCount: number;
  supersededById?: string;
}

export interface MemoryQuery {
  query: string;
  tiers?: MemoryTier[];
  limit?: number;
  minScore?: number;
  nowMs?: number;
}

export interface MemoryHit {
  record: MemoryRecord;
  score: number;
  bm25Rank?: number;
  vectorRank?: number;
  recencyWeight: number;
}

export interface MemoryProvider {
  remember(input: MemoryRecordInput): Promise<MemoryRecord>;
  recall(query: MemoryQuery): Promise<MemoryHit[]>;
  consolidate(threadId: string, messages: HarnessMessage[]): Promise<{ added: number; superseded: number }>;
  inspect(filter?: { tier?: MemoryTier; limit?: number; offset?: number }): Promise<MemoryRecord[]>;
  forget(id: string): Promise<boolean>;
  exportEncrypted(passphrase: string): Promise<Uint8Array>;
  importEncrypted(blob: Uint8Array, passphrase: string): Promise<{ imported: number }>;
  wipeAll(): Promise<void>;
}

/** Extended @yours/memory types for graph provenance, privacy rules, quantization, and lifecycle. */

export type QuantizationMode = 'int8' | 'f32';

export type MemoryEdgeRelation =
  | 'supersedes'
  | 'derived_from'
  | 'related_to'
  | 'same_thread'
  | 'mentions';

export interface MemoryEdge {
  id: string;
  sourceId: string;
  targetId: string;
  relation: MemoryEdgeRelation;
  weight: number;
  metadata?: Record<string, unknown>;
  createdAt: number;
}

export type DomainRuleKind = 'glob' | 'regex' | 'exact';

export interface BlockedDomainRule {
  id: string;
  pattern: string;
  kind: DomainRuleKind;
  reason: string;
  builtin: boolean;
  createdAt: number;
}

export interface RememberOptions {
  /** Explicitly allow a tainted record to supersede a trusted profile record after user confirmation. */
  confirmTaintedProfileMutation?: boolean;
  /** Skip automatic deduplication and force creating a distinct record. */
  skipDeduplication?: boolean;
  /** Skip automatic contradiction superseding. */
  skipContradictionCheck?: boolean;
  /** Explicit timestamp override (ms since epoch) for deterministic tests. */
  nowMs?: number;
}

export interface InspectOptions {
  tier?: MemoryTier;
  limit?: number;
  offset?: number;
  includeSuperseded?: boolean;
  threadId?: string;
  tainted?: boolean;
}

export interface MemoryProvenanceNode {
  record: MemoryRecord;
  edges: MemoryEdge[];
  supersedes: MemoryRecord[];
  supersededBy?: MemoryRecord;
}

export interface TierDecayConfig {
  /** Half-life in milliseconds for temporal salience decay. */
  halfLifeMs: number;
  /** Minimum recency floor in (0..1] so durable facts never vanish completely. */
  floor: number;
}

export type TierDecayMap = Record<MemoryTier, TierDecayConfig>;

export interface RankingConfig {
  /** Reciprocal Rank Fusion constant k (default 60). */
  rrfK: number;
  /** Weight applied to BM25 RRF component (default 1.0). */
  lexicalWeight: number;
  /** Weight applied to HNSW vector RRF component (default 1.0). */
  vectorWeight: number;
  /** Weight applied to temporal salience multiplier (default 0.35). */
  recencyMix: number;
  /** Weight applied to importance [0..1] boost (default 0.25). */
  importanceBoost: number;
  /** Logarithmic access count reinforcement factor (default 0.08). */
  accessBoost: number;
  /** Multiplicative penalty for tainted records (default 0.85). */
  taintedPenalty: number;
  /** Per-tier temporal half-life configuration. */
  decayByTier: TierDecayMap;
}

export interface EmbeddingProvider {
  readonly id: string;
  readonly dimensions: number;
  embed(texts: string[], signal?: AbortSignal): Promise<Float32Array[]>;
  embedQuery?(text: string, signal?: AbortSignal): Promise<Float32Array>;
}

export interface MemoryStorageStats {
  backend: 'opfs-sqlite' | 'indexeddb' | 'in-memory';
  totalRecords: number;
  activeRecords: number;
  supersededRecords: number;
  byTier: Record<MemoryTier, number>;
  vectorCount: number;
  edgeCount: number;
  blockedRuleCount: number;
  quantization: QuantizationMode;
  dimensions: number;
}
