import type { HybridMemoryEngine } from '../engine.js';
import { MemoryError, WorkerCancelledError } from '../errors.js';
import type {
  MemoryRpcMethod,
  MemoryRpcParamsMap,
  MemoryRpcRequest,
  MemoryRpcResponse,
  MemoryRpcResultMap,
  WorkerEndpointLike,
  WorkerInboundMessage,
  WorkerOutboundMessage,
} from './protocol.js';

export interface MemoryWorkerHostOptions {
  engine: HybridMemoryEngine;
  /** Optional custom postMessage endpoint (defaults to Dedicated Worker `globalThis`). */
  endpoint?: WorkerEndpointLike;
}

function isRpcRequest(msg: unknown): msg is MemoryRpcRequest {
  return (
    typeof msg === 'object' &&
    msg !== null &&
    (msg as { type?: unknown }).type === 'yours-memory:request' &&
    typeof (msg as { callId?: unknown }).callId === 'string'
  );
}

function isRpcCancel(msg: unknown): msg is { type: 'yours-memory:cancel'; callId: string } {
  return (
    typeof msg === 'object' &&
    msg !== null &&
    (msg as { type?: unknown }).type === 'yours-memory:cancel' &&
    typeof (msg as { callId?: unknown }).callId === 'string'
  );
}

function serializeError(err: unknown): { name: string; code?: string; message: string } {
  if (err instanceof MemoryError) {
    return { name: err.name, code: err.code, message: err.message };
  }
  if (err instanceof Error) {
    return { name: err.name, message: err.message };
  }
  return { name: 'Error', message: String(err) };
}

/**
 * Single-owner Dedicated Web Worker host for `@yours/memory`.
 * Sequentializes mutating operations so OPFS `SyncAccessHandle` / SQLite-WASM never suffers
 * multi-writer contention, and supports per-call cancellation.
 */
export class MemoryWorkerHost {
  private readonly engine: HybridMemoryEngine;
  private readonly inflight = new Map<string, AbortController>();
  private readonly cancelledBeforeStart = new Set<string>();
  private writeQueue: Promise<void> = Promise.resolve();
  private endpoint: WorkerEndpointLike | null = null;
  private listener: ((event: { data: unknown }) => void) | null = null;

  constructor(options: MemoryWorkerHostOptions) {
    this.engine = options.engine;
    if (options.endpoint) {
      this.attach(options.endpoint);
    }
  }

  attach(endpoint: WorkerEndpointLike): void {
    this.detach();
    this.endpoint = endpoint;
    this.listener = (event: { data: unknown }) => {
      void this.handleMessage(event.data);
    };
    endpoint.addEventListener('message', this.listener);
  }

  detach(): void {
    if (this.endpoint && this.listener) {
      this.endpoint.removeEventListener('message', this.listener);
    }
    this.endpoint = null;
    this.listener = null;
  }

  async notifyReady(): Promise<void> {
    await this.engine.init();
    const stats = await this.engine.getStats();
    this.post({
      type: 'yours-memory:ready',
      backend: stats.backend,
    });
  }

  private post(msg: WorkerOutboundMessage, transfer?: Transferable[]): void {
    this.endpoint?.postMessage(msg, transfer);
  }

  private enqueueWrite<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.writeQueue.then(fn, fn);
    this.writeQueue = next.then(
      () => undefined,
      () => undefined
    );
    return next;
  }

  async handleMessage(raw: unknown): Promise<void> {
    const msg = raw as WorkerInboundMessage;
    if (isRpcCancel(msg)) {
      const ctrl = this.inflight.get(msg.callId);
      if (ctrl) {
        ctrl.abort(new WorkerCancelledError());
      } else {
        this.cancelledBeforeStart.add(msg.callId);
      }
      return;
    }

    if (!isRpcRequest(msg)) return;

    const controller = new AbortController();
    this.inflight.set(msg.callId, controller);

    if (this.cancelledBeforeStart.delete(msg.callId)) {
      controller.abort(new WorkerCancelledError());
    }

    try {
      const result = await this.dispatch(msg.method, msg.params, controller.signal);
      if (controller.signal.aborted) {
        throw new WorkerCancelledError();
      }
      const response: MemoryRpcResponse = {
        type: 'yours-memory:response',
        callId: msg.callId,
        ok: true,
        result,
      };
      if (result instanceof Uint8Array) {
        const copy = new Uint8Array(result);
        response.result = copy;
        this.post(response, [copy.buffer]);
      } else {
        this.post(response);
      }
    } catch (err) {
      this.post({
        type: 'yours-memory:response',
        callId: msg.callId,
        ok: false,
        error: serializeError(err),
      });
    } finally {
      this.inflight.delete(msg.callId);
    }
  }

  private async dispatch<M extends MemoryRpcMethod>(
    method: M,
    params: MemoryRpcParamsMap[M],
    signal: AbortSignal
  ): Promise<MemoryRpcResultMap[M]> {
    if (signal.aborted) {
      throw new WorkerCancelledError();
    }

    switch (method) {
      case 'init':
        await this.engine.init();
        return undefined as MemoryRpcResultMap[M];

      case 'remember': {
        const p = params as MemoryRpcParamsMap['remember'];
        return (await this.enqueueWrite(async () => {
          if (signal.aborted) throw new WorkerCancelledError();
          return this.engine.remember(p.input, p.options);
        })) as MemoryRpcResultMap[M];
      }

      case 'recall': {
        const p = params as MemoryRpcParamsMap['recall'];
        return (await this.engine.recall(p.query)) as MemoryRpcResultMap[M];
      }

      case 'consolidate': {
        const p = params as MemoryRpcParamsMap['consolidate'];
        return (await this.enqueueWrite(async () => {
          if (signal.aborted) throw new WorkerCancelledError();
          return this.engine.consolidate(p.threadId, p.messages);
        })) as MemoryRpcResultMap[M];
      }

      case 'inspect': {
        const p = params as MemoryRpcParamsMap['inspect'];
        return (await this.engine.inspect(p?.filter)) as MemoryRpcResultMap[M];
      }

      case 'inspectGraph': {
        const p = params as MemoryRpcParamsMap['inspectGraph'];
        return (await this.engine.inspectGraph(p.recordId)) as MemoryRpcResultMap[M];
      }

      case 'forget': {
        const p = params as MemoryRpcParamsMap['forget'];
        return (await this.enqueueWrite(async () => {
          if (signal.aborted) throw new WorkerCancelledError();
          return this.engine.forget(p.id);
        })) as MemoryRpcResultMap[M];
      }

      case 'exportEncrypted': {
        const p = params as MemoryRpcParamsMap['exportEncrypted'];
        return (await this.engine.exportEncrypted(p.passphrase)) as MemoryRpcResultMap[M];
      }

      case 'importEncrypted': {
        const p = params as MemoryRpcParamsMap['importEncrypted'];
        return (await this.enqueueWrite(async () => {
          if (signal.aborted) throw new WorkerCancelledError();
          return this.engine.importEncrypted(p.blob, p.passphrase);
        })) as MemoryRpcResultMap[M];
      }

      case 'wipeAll':
        await this.enqueueWrite(async () => {
          if (signal.aborted) throw new WorkerCancelledError();
          await this.engine.wipeAll();
        });
        return undefined as MemoryRpcResultMap[M];

      case 'addBlockedDomain': {
        const p = params as MemoryRpcParamsMap['addBlockedDomain'];
        return (await this.enqueueWrite(() =>
          this.engine.addBlockedDomain(p.pattern, p.kind, p.reason)
        )) as MemoryRpcResultMap[M];
      }

      case 'removeBlockedDomain': {
        const p = params as MemoryRpcParamsMap['removeBlockedDomain'];
        return (await this.enqueueWrite(() =>
          this.engine.removeBlockedDomain(p.id)
        )) as MemoryRpcResultMap[M];
      }

      case 'listBlockedDomains':
        return (await this.engine.listBlockedDomains()) as MemoryRpcResultMap[M];

      case 'isUrlBlocked': {
        const p = params as MemoryRpcParamsMap['isUrlBlocked'];
        return this.engine.isUrlBlocked(p.url) as MemoryRpcResultMap[M];
      }

      case 'getStats':
        return (await this.engine.getStats()) as MemoryRpcResultMap[M];

      case 'rebuildIndex':
        return (await this.enqueueWrite(() => this.engine.rebuildIndex())) as MemoryRpcResultMap[M];

      case 'close':
        await this.enqueueWrite(() => this.engine.close());
        return undefined as MemoryRpcResultMap[M];
    }
  }
}
