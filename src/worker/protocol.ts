import type {
  BlockedDomainRule,
  DomainRuleKind,
  HarnessMessage,
  InspectOptions,
  MemoryHit,
  MemoryProvenanceNode,
  MemoryQuery,
  MemoryRecord,
  MemoryRecordInput,
  MemoryStorageStats,
  RememberOptions,
} from '../contracts.js';

export type MemoryRpcMethod =
  | 'init'
  | 'remember'
  | 'recall'
  | 'consolidate'
  | 'inspect'
  | 'inspectGraph'
  | 'forget'
  | 'exportEncrypted'
  | 'importEncrypted'
  | 'wipeAll'
  | 'addBlockedDomain'
  | 'removeBlockedDomain'
  | 'listBlockedDomains'
  | 'isUrlBlocked'
  | 'getStats'
  | 'rebuildIndex'
  | 'close';

export interface MemoryRpcParamsMap {
  init: undefined;
  remember: { input: MemoryRecordInput; options?: RememberOptions };
  recall: { query: MemoryQuery };
  consolidate: { threadId: string; messages: HarnessMessage[] };
  inspect: { filter?: InspectOptions };
  inspectGraph: { recordId: string };
  forget: { id: string };
  exportEncrypted: { passphrase: string };
  importEncrypted: { blob: Uint8Array; passphrase: string };
  wipeAll: undefined;
  addBlockedDomain: { pattern: string; kind?: DomainRuleKind; reason?: string };
  removeBlockedDomain: { id: string };
  listBlockedDomains: undefined;
  isUrlBlocked: { url: string };
  getStats: undefined;
  rebuildIndex: undefined;
  close: undefined;
}

export interface MemoryRpcResultMap {
  init: void;
  remember: MemoryRecord;
  recall: MemoryHit[];
  consolidate: { added: number; superseded: number };
  inspect: MemoryRecord[];
  inspectGraph: MemoryProvenanceNode | null;
  forget: boolean;
  exportEncrypted: Uint8Array;
  importEncrypted: { imported: number };
  wipeAll: void;
  addBlockedDomain: BlockedDomainRule;
  removeBlockedDomain: boolean;
  listBlockedDomains: BlockedDomainRule[];
  isUrlBlocked: { blocked: boolean; reason?: string; rule?: string };
  getStats: MemoryStorageStats;
  rebuildIndex: number;
  close: void;
}

export interface MemoryRpcRequest<M extends MemoryRpcMethod = MemoryRpcMethod> {
  type: 'yours-memory:request';
  callId: string;
  method: M;
  params: MemoryRpcParamsMap[M];
}

export interface MemoryRpcCancel {
  type: 'yours-memory:cancel';
  callId: string;
}

export interface MemoryRpcResponse<M extends MemoryRpcMethod = MemoryRpcMethod> {
  type: 'yours-memory:response';
  callId: string;
  ok: boolean;
  result?: MemoryRpcResultMap[M];
  error?: {
    name: string;
    code?: string;
    message: string;
  };
}

export interface MemoryRpcReadyEvent {
  type: 'yours-memory:ready';
  backend: MemoryStorageStats['backend'];
}

export type WorkerInboundMessage = MemoryRpcRequest | MemoryRpcCancel;
export type WorkerOutboundMessage = MemoryRpcResponse | MemoryRpcReadyEvent;

/** Abstract Worker / MessagePort / test-channel endpoint interface. */
export interface WorkerEndpointLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(type: 'message', listener: (event: { data: unknown }) => void): void;
  removeEventListener(type: 'message', listener: (event: { data: unknown }) => void): void;
  terminate?(): void;
}
