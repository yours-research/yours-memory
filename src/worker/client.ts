import type {
  BlockedDomainRule,
  DomainRuleKind,
  HarnessMessage,
  InspectOptions,
  MemoryHit,
  MemoryProvenanceNode,
  MemoryProvider,
  MemoryQuery,
  MemoryRecord,
  MemoryRecordInput,
  MemoryStorageStats,
  RememberOptions,
} from '../contracts.js';
import { MemoryError, WorkerCancelledError } from '../errors.js';
import { generateId } from '../utils/id.js';
import type {
  MemoryRpcMethod,
  MemoryRpcParamsMap,
  MemoryRpcRequest,
  MemoryRpcResponse,
  MemoryRpcResultMap,
  WorkerEndpointLike,
} from './protocol.js';

export interface MemoryWorkerClientOptions {
  /** Default RPC request timeout in ms (default 30_000). */
  timeoutMs?: number;
}

interface PendingCall {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  timer?: ReturnType<typeof setTimeout>;
  cleanupAbort?: () => void;
}

function isRpcResponse(msg: unknown): msg is MemoryRpcResponse {
  return (
    typeof msg === 'object' &&
    msg !== null &&
    (msg as { type?: unknown }).type === 'yours-memory:response' &&
    typeof (msg as { callId?: unknown }).callId === 'string'
  );
}

function hydrateRemoteError(err?: { name: string; code?: string; message: string }): Error {
  if (!err) return new MemoryError('WORKER_ERROR', 'Unknown worker error');
  if (err.code === 'WORKER_CANCELLED') {
    return new WorkerCancelledError(err.message);
  }
  if (err.code) {
    const mErr = new MemoryError(err.code, err.message);
    mErr.name = err.name;
    return mErr;
  }
  const e = new Error(err.message);
  e.name = err.name;
  return e;
}

/**
 * RPC client implementing `MemoryProvider` against a Dedicated Web Worker hosting `MemoryWorkerHost`.
 * Supports `AbortSignal` cancellation, timeouts, and clean disposal.
 */
export class MemoryWorkerClient implements MemoryProvider {
  private readonly pending = new Map<string, PendingCall>();
  private readonly timeoutMs: number;
  private disposed = false;

  private readonly onMessage = (event: { data: unknown }) => {
    if (!isRpcResponse(event.data)) return;
    const resp = event.data;
    const entry = this.pending.get(resp.callId);
    if (!entry) return;

    this.pending.delete(resp.callId);
    if (entry.timer) clearTimeout(entry.timer);
    entry.cleanupAbort?.();

    if (resp.ok) {
      entry.resolve(resp.result);
    } else {
      entry.reject(hydrateRemoteError(resp.error));
    }
  };

  constructor(
    private readonly endpoint: WorkerEndpointLike,
    options: MemoryWorkerClientOptions = {}
  ) {
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.endpoint.addEventListener('message', this.onMessage);
  }

  private call<M extends MemoryRpcMethod>(
    method: M,
    params: MemoryRpcParamsMap[M],
    signal?: AbortSignal,
    transfer?: Transferable[]
  ): Promise<MemoryRpcResultMap[M]> {
    if (this.disposed) {
      return Promise.reject(new MemoryError('WORKER_DISPOSED', 'MemoryWorkerClient has been closed'));
    }
    if (signal?.aborted) {
      return Promise.reject(new WorkerCancelledError());
    }

    const callId = generateId('rpc');

    return new Promise<MemoryRpcResultMap[M]>((resolve, reject) => {
      const entry: PendingCall = {
        resolve: resolve as (value: unknown) => void,
        reject,
      };

      if (this.timeoutMs > 0) {
        entry.timer = setTimeout(() => {
          if (this.pending.delete(callId)) {
            entry.cleanupAbort?.();
            this.endpoint.postMessage({ type: 'yours-memory:cancel', callId });
            reject(new MemoryError('WORKER_TIMEOUT', `Worker RPC "${method}" timed out after ${this.timeoutMs}ms`));
          }
        }, this.timeoutMs);
      }

      if (signal) {
        const abortHandler = () => {
          if (this.pending.delete(callId)) {
            if (entry.timer) clearTimeout(entry.timer);
            this.endpoint.postMessage({ type: 'yours-memory:cancel', callId });
            reject(new WorkerCancelledError());
          }
        };
        signal.addEventListener('abort', abortHandler, { once: true });
        entry.cleanupAbort = () => signal.removeEventListener('abort', abortHandler);
      }

      this.pending.set(callId, entry);

      const request: MemoryRpcRequest<M> = {
        type: 'yours-memory:request',
        callId,
        method,
        params,
      };
      this.endpoint.postMessage(request, transfer);
    });
  }

  init(signal?: AbortSignal): Promise<void> {
    return this.call('init', undefined, signal);
  }

  remember(input: MemoryRecordInput, options?: RememberOptions, signal?: AbortSignal): Promise<MemoryRecord> {
    return this.call('remember', { input, options }, signal);
  }

  recall(query: MemoryQuery, signal?: AbortSignal): Promise<MemoryHit[]> {
    return this.call('recall', { query }, signal);
  }

  consolidate(
    threadId: string,
    messages: HarnessMessage[],
    signal?: AbortSignal
  ): Promise<{ added: number; superseded: number }> {
    return this.call('consolidate', { threadId, messages }, signal);
  }

  inspect(filter?: InspectOptions, signal?: AbortSignal): Promise<MemoryRecord[]> {
    return this.call('inspect', { filter }, signal);
  }

  inspectGraph(recordId: string, signal?: AbortSignal): Promise<MemoryProvenanceNode | null> {
    return this.call('inspectGraph', { recordId }, signal);
  }

  forget(id: string, signal?: AbortSignal): Promise<boolean> {
    return this.call('forget', { id }, signal);
  }

  exportEncrypted(passphrase: string, signal?: AbortSignal): Promise<Uint8Array> {
    return this.call('exportEncrypted', { passphrase }, signal);
  }

  importEncrypted(blob: Uint8Array, passphrase: string, signal?: AbortSignal): Promise<{ imported: number }> {
    const copy = new Uint8Array(blob);
    return this.call('importEncrypted', { blob: copy, passphrase }, signal, [copy.buffer]);
  }

  wipeAll(signal?: AbortSignal): Promise<void> {
    return this.call('wipeAll', undefined, signal);
  }

  addBlockedDomain(
    pattern: string,
    kind?: DomainRuleKind,
    reason?: string,
    signal?: AbortSignal
  ): Promise<BlockedDomainRule> {
    return this.call('addBlockedDomain', { pattern, kind, reason }, signal);
  }

  removeBlockedDomain(id: string, signal?: AbortSignal): Promise<boolean> {
    return this.call('removeBlockedDomain', { id }, signal);
  }

  listBlockedDomains(signal?: AbortSignal): Promise<BlockedDomainRule[]> {
    return this.call('listBlockedDomains', undefined, signal);
  }

  isUrlBlocked(
    url: string,
    signal?: AbortSignal
  ): Promise<{ blocked: boolean; reason?: string; rule?: string }> {
    return this.call('isUrlBlocked', { url }, signal);
  }

  getStats(signal?: AbortSignal): Promise<MemoryStorageStats> {
    return this.call('getStats', undefined, signal);
  }

  rebuildIndex(signal?: AbortSignal): Promise<number> {
    return this.call('rebuildIndex', undefined, signal);
  }

  async close(terminateWorker = false): Promise<void> {
    if (this.disposed) return;
    try {
      await this.call('close', undefined);
    } finally {
      this.disposed = true;
      for (const [callId, entry] of this.pending) {
        if (entry.timer) clearTimeout(entry.timer);
        entry.cleanupAbort?.();
        entry.reject(new WorkerCancelledError('Client disposed'));
        this.pending.delete(callId);
      }
      this.endpoint.removeEventListener('message', this.onMessage);
      if (terminateWorker) {
        this.endpoint.terminate?.();
      }
    }
  }
}
