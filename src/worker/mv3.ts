import { HybridMemoryEngine, type HybridMemoryEngineOptions } from '../engine.js';
import { MemoryWorkerClient, type MemoryWorkerClientOptions } from './client.js';
import { MemoryWorkerHost } from './host.js';
import type { WorkerEndpointLike } from './protocol.js';

/**
 * Attaches a `MemoryWorkerHost` inside a Chrome MV3 Dedicated Web Worker (launched by
 * an Offscreen Document, Sidepanel, Popup, or Web App).
 */
export async function startMemoryWorkerHost(
  engineOrOptions: HybridMemoryEngine | HybridMemoryEngineOptions = {}
): Promise<MemoryWorkerHost> {
  const engine =
    engineOrOptions instanceof HybridMemoryEngine
      ? engineOrOptions
      : new HybridMemoryEngine(engineOrOptions);

  const scope = globalThis as unknown as {
    postMessage?: (msg: unknown, transfer?: Transferable[]) => void;
    addEventListener?: (type: 'message', listener: (ev: { data: unknown }) => void) => void;
    removeEventListener?: (type: 'message', listener: (ev: { data: unknown }) => void) => void;
  };

  if (
    typeof scope.postMessage !== 'function' ||
    typeof scope.addEventListener !== 'function' ||
    typeof scope.removeEventListener !== 'function'
  ) {
    throw new Error('startMemoryWorkerHost must be invoked inside a Web Worker global scope');
  }

  const endpoint: WorkerEndpointLike = {
    postMessage: (msg, transfer) => scope.postMessage!(msg, transfer),
    addEventListener: (type, listener) => scope.addEventListener!(type, listener),
    removeEventListener: (type, listener) => scope.removeEventListener!(type, listener),
  };

  const host = new MemoryWorkerHost({ engine, endpoint });
  await host.notifyReady();
  return host;
}

/**
 * Creates a `MemoryWorkerClient` connected to a Dedicated Web Worker URL
 * (e.g., `chrome.runtime.getURL('memory-worker.js')` in MV3 Offscreen/Sidepanel).
 */
export function createMv3WorkerClient(
  workerOrUrl: WorkerEndpointLike | string | URL,
  options?: MemoryWorkerClientOptions
): MemoryWorkerClient {
  if (typeof workerOrUrl === 'string' || workerOrUrl instanceof URL) {
    if (typeof Worker === 'undefined') {
      throw new Error('Global Web Worker constructor is unavailable in this environment');
    }
    const worker = new Worker(workerOrUrl, { type: 'module', name: 'yours-memory-worker' });
    return new MemoryWorkerClient(worker as unknown as WorkerEndpointLike, options);
  }
  return new MemoryWorkerClient(workerOrUrl, options);
}

/**
 * Paired in-process MessageChannel-compatible transport for testing Worker RPC,
 * async scheduling, and cancellation without mocking global Worker constructors.
 */
export function createInMemoryWorkerPair(): {
  clientEndpoint: WorkerEndpointLike;
  hostEndpoint: WorkerEndpointLike;
} {
  const clientListeners = new Set<(event: { data: unknown }) => void>();
  const hostListeners = new Set<(event: { data: unknown }) => void>();
  let terminated = false;

  const clientEndpoint: WorkerEndpointLike = {
    postMessage(message: unknown) {
      if (terminated) return;
      queueMicrotask(() => {
        if (terminated) return;
        for (const listener of hostListeners) {
          listener({ data: message });
        }
      });
    },
    addEventListener(_type, listener) {
      clientListeners.add(listener);
    },
    removeEventListener(_type, listener) {
      clientListeners.delete(listener);
    },
    terminate() {
      terminated = true;
      clientListeners.clear();
      hostListeners.clear();
    },
  };

  const hostEndpoint: WorkerEndpointLike = {
    postMessage(message: unknown) {
      if (terminated) return;
      queueMicrotask(() => {
        if (terminated) return;
        for (const listener of clientListeners) {
          listener({ data: message });
        }
      });
    },
    addEventListener(_type, listener) {
      hostListeners.add(listener);
    },
    removeEventListener(_type, listener) {
      hostListeners.delete(listener);
    },
  };

  return { clientEndpoint, hostEndpoint };
}
