import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { existsSync, rmSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { Logger } from 'pino';
import type { VectorEntry, SearchFilterOptions } from '../src/semantic-types.js';
import { VectorStore } from '../src/vector-store.js';

function mockLogger(): Logger {
  return {
    info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(),
    child: vi.fn().mockReturnThis(), fatal: vi.fn(), trace: vi.fn(),
    silent: vi.fn(), level: 'debug',
  } as unknown as Logger;
}

function makeFact(id: string, vector: number[], namespace = 'default', key = 'test-key'): VectorEntry {
  return { id, vector, source: 'fact', metadata: { key, namespace, preview: `preview for ${key}` }, indexedAt: Date.now() };
}

function makeDoc(id: string, vector: number[], title = 'Test Doc', tags: string[] = [], topic?: string, contentType = 'text/plain'): VectorEntry {
  return { id, vector, source: 'document', metadata: { documentId: id, title, topic, tags, contentType }, indexedAt: Date.now() };
}

describe('VectorStore - CRUD', () => {
  let store: VectorStore;

  beforeEach(() => {
    const indexPath = join(tmpdir(), `vector-${randomUUID()}.json`);
    store = new VectorStore(indexPath, 'test-model', 3, mockLogger());
  });

  it('upsert adds new entry', () => {
    const entry = makeFact('f1', [1, 0, 0]);
    store.upsert(entry);
    expect(store.entryCount).toBe(1);
  });

  it('upsert replaces existing entry with same id', () => {
    store.upsert(makeFact('f1', [1, 0, 0]));
    store.upsert(makeFact('f1', [0, 1, 0]));
    expect(store.entryCount).toBe(1);
    const entries = store.allEntries();
    expect(entries[0].vector).toEqual([0, 1, 0]);
  });

  it('remove returns true for existing entry', () => {
    store.upsert(makeFact('f1', [1, 0, 0]));
    expect(store.remove('f1')).toBe(true);
    expect(store.entryCount).toBe(0);
  });

  it('remove returns false for non-existent entry', () => {
    expect(store.remove('nonexistent')).toBe(false);
  });

  it('rebuild replaces all entries', () => {
    store.upsert(makeFact('f1', [1, 0, 0]));
    store.upsert(makeFact('f2', [0, 1, 0]));
    store.rebuild([makeFact('f3', [0, 0, 1])]);
    expect(store.entryCount).toBe(1);
    expect(store.allEntries()[0].id).toBe('f3');
  });
});

describe('VectorStore - Cosine similarity query', () => {
  let store: VectorStore;

  beforeEach(() => {
    store = new VectorStore(join(tmpdir(), `vector-${randomUUID()}.json`), 'test-model', 3, mockLogger());
    store.upsert(makeFact('f1', [1, 0, 0], 'default', 'key1'));
    store.upsert(makeFact('f2', [0, 1, 0], 'default', 'key2'));
    store.upsert(makeFact('f3', [0, 0, 1], 'other', 'key3'));
  });

  it('returns results sorted by score descending', () => {
    const results = store.query([1, 0, 0]);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].score).toBeGreaterThanOrEqual(results[results.length - 1].score);
  });

  it('identical vector returns score ~1.0', () => {
    const results = store.query([1, 0, 0]);
    expect(results[0].score).toBeCloseTo(1.0, 5);
  });

  it('orthogonal vector returns score ~0.0 (filtered by threshold)', () => {
    const results = store.query([1, 0, 0], { threshold: 0.5 });
    expect(results.length).toBe(1); // only f1 matches
  });

  it('respects limit', () => {
    const results = store.query([0.5, 0.5, 0.5], { limit: 1 });
    expect(results.length).toBeLessThanOrEqual(1);
  });

  it('filters by source=facts', () => {
    store.upsert(makeDoc('d1', [1, 0, 0]));
    const results = store.query([1, 0, 0], { source: 'facts' });
    expect(results.every((r) => r.source === 'fact')).toBe(true);
  });

  it('filters by source=documents', () => {
    store.upsert(makeDoc('d1', [1, 0, 0]));
    const results = store.query([1, 0, 0], { source: 'documents' });
    expect(results.every((r) => r.source === 'document')).toBe(true);
  });

  it('filters by namespace', () => {
    const results = store.query([0.5, 0.5, 0.5], { namespace: 'other' });
    expect(results.every((r) => r.source === 'fact')).toBe(true);
  });
});

describe('VectorStore - Persistence', () => {
  let indexPath: string;

  beforeEach(() => {
    indexPath = join(tmpdir(), `vector-${randomUUID()}.json`);
  });

  afterEach(() => {
    if (existsSync(indexPath)) rmSync(indexPath, { force: true });
  });

  it('save and load round-trip preserves entries', async () => {
    const store1 = new VectorStore(indexPath, 'test-model', 3, mockLogger());
    store1.upsert(makeFact('f1', [1, 0, 0]));
    await store1.save();

    const store2 = new VectorStore(indexPath, 'test-model', 3, mockLogger());
    store2.load();
    expect(store2.entryCount).toBe(1);
    expect(store2.allEntries()[0].id).toBe('f1');
  });

  it('save returns false when not dirty', async () => {
    const store = new VectorStore(indexPath, 'test-model', 3, mockLogger());
    expect(await store.save()).toBe(false);
  });

  it('save returns true after upsert', async () => {
    const store = new VectorStore(indexPath, 'test-model', 3, mockLogger());
    store.upsert(makeFact('f1', [1, 0, 0]));
    expect(await store.save()).toBe(true);
  });

  it('load with missing file starts empty', () => {
    const store = new VectorStore(indexPath, 'test-model', 3, mockLogger());
    store.load();
    expect(store.entryCount).toBe(0);
  });

  it('tracks stored model ID', async () => {
    const store1 = new VectorStore(indexPath, 'model-v1', 3, mockLogger());
    store1.upsert(makeFact('f1', [1, 0, 0]));
    await store1.save();

    const store2 = new VectorStore(indexPath, 'model-v2', 3, mockLogger());
    store2.load();
    expect(store2.indexModelId).toBe('model-v1');
    expect(store2.currentModelId).toBe('model-v2');
  });
});

describe('VectorStore - Failure tracking', () => {
  let store: VectorStore;

  beforeEach(() => {
    store = new VectorStore(join(tmpdir(), `vector-${randomUUID()}.json`), 'test-model', 3, mockLogger());
  });

  it('recordFailure increments retry count', () => {
    store.recordFailure('item-1', 'error msg', 3);
    const failures = store.getFailedEmbeddings();
    expect(failures['item-1'].retries).toBe(1);
    expect(failures['item-1'].status).toBe('pending');
  });

  it('recordFailure marks as failed when retries reach max', () => {
    store.recordFailure('item-1', 'err', 2);
    store.recordFailure('item-1', 'err', 2);
    expect(store.getFailedEmbeddings()['item-1'].status).toBe('failed');
  });

  it('clearFailure removes the record', () => {
    store.recordFailure('item-1', 'err', 3);
    store.clearFailure('item-1');
    expect(store.getFailedEmbeddings()['item-1']).toBeUndefined();
  });
});
