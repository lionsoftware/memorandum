import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Logger } from 'pino';
import type { Config, ResolvedPaths } from '../src/config.js';
import { MemoryStore } from '../src/store.js';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';

function mockLogger(): Logger {
  return {
    info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(),
    child: vi.fn().mockReturnThis(), fatal: vi.fn(), trace: vi.fn(),
    silent: vi.fn(), level: 'debug',
  } as unknown as Logger;
}

function makeConfig(): Config {
  return {
    max_entries: 100, autosave_interval_seconds: 0, storage_dir: '.memorandum',
    max_document_size: 16 * 1024 * 1024, semantic_model: 'Xenova/multilingual-e5-small',
    semantic_model_dtype: 'q8', semantic_enabled: false, semantic_debounce_seconds: 10,
    semantic_max_queue_size: 200, semantic_max_retries: 3,
  };
}

function makePaths(): ResolvedPaths {
  return {
    storageDir: '.memorandum',
    factsPath: join(tmpdir(), `test-memory-${randomUUID()}.json`),
    documentsPath: join(tmpdir(), `test-docs-${randomUUID()}`),
    cachePath: join(tmpdir(), `test-cache-${randomUUID()}`),
    configPath: '.memorandum/config.yaml',
  };
}

describe('MemoryStore - concurrent write deduplication via queue', () => {
  let store: MemoryStore;

  beforeEach(() => {
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
  });

  it('two rapid writes to the same key — both call enqueueFact', () => {
    const enqueueFactCalls: Array<{ key: string; namespace: string; value: unknown }> = [];
    const mockIndex = {
      enqueueFact: vi.fn((key: string, namespace: string, value: unknown) => {
        enqueueFactCalls.push({ key, namespace, value });
      }),
    };
    store.setSemanticIndex(mockIndex as never);

    store.write('my-key', 'value-1', 'default');
    store.write('my-key', 'value-2', 'default');

    expect(enqueueFactCalls).toHaveLength(2);
    expect(enqueueFactCalls[0]!.value).toBe('value-1');
    expect(enqueueFactCalls[1]!.value).toBe('value-2');
  });

  it('writes to different keys — both are enqueued independently', () => {
    const mockIndex = { enqueueFact: vi.fn() };
    store.setSemanticIndex(mockIndex as never);

    store.write('key-alpha', 'value-a', 'default');
    store.write('key-beta', 'value-b', 'default');

    expect(mockIndex.enqueueFact).toHaveBeenCalledTimes(2);
  });

  it('single write calls enqueueFact once', () => {
    const mockIndex = { enqueueFact: vi.fn() };
    store.setSemanticIndex(mockIndex as never);

    store.write('solo-key', 'solo-value', 'default');

    expect(mockIndex.enqueueFact).toHaveBeenCalledOnce();
    expect(mockIndex.enqueueFact).toHaveBeenCalledWith('solo-key', 'default', 'solo-value');
  });
});
