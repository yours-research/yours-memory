import type {
  BlockedDomainRule,
  InspectOptions,
  MemoryEdge,
  MemoryRecord,
  MemoryTier,
  QuantizationMode,
} from '../contracts.js';

export interface StoredVectorRecord {
  memoryId: string;
  dims: number;
  quantization: QuantizationMode;
  scale: number;
  data: Uint8Array;
  updatedAt: number;
}

export interface LexicalSearchOptions {
  tiers?: MemoryTier[];
  includeSuperseded?: boolean;
  limit?: number;
}

export interface LexicalSearchResult {
  id: string;
  score: number;
  rank: number;
}

export interface MemoryStorageBackend {
  readonly kind: 'opfs-sqlite' | 'indexeddb' | 'in-memory';

  init(): Promise<void>;

  upsertRecord(record: MemoryRecord): Promise<void>;
  getRecord(id: string): Promise<MemoryRecord | null>;
  listRecords(filter?: InspectOptions): Promise<MemoryRecord[]>;
  deleteRecord(id: string): Promise<boolean>;

  upsertVector(vector: StoredVectorRecord): Promise<void>;
  getVector(memoryId: string): Promise<StoredVectorRecord | null>;
  listVectors(memoryIds?: string[]): Promise<StoredVectorRecord[]>;
  deleteVector(memoryId: string): Promise<boolean>;

  searchLexical(query: string, options?: LexicalSearchOptions): Promise<LexicalSearchResult[]>;

  upsertEdge(edge: MemoryEdge): Promise<void>;
  listEdgesForRecord(recordId: string): Promise<MemoryEdge[]>;
  listAllEdges(): Promise<MemoryEdge[]>;
  deleteEdgesForRecord(recordId: string): Promise<void>;

  upsertBlockedRule(rule: BlockedDomainRule): Promise<void>;
  listBlockedRules(): Promise<BlockedDomainRule[]>;
  deleteBlockedRule(id: string): Promise<boolean>;

  saveIndexSnapshot(name: string, bytes: Uint8Array): Promise<void>;
  loadIndexSnapshot(name: string): Promise<Uint8Array | null>;

  clearAll(): Promise<void>;
  close(): Promise<void>;
}
