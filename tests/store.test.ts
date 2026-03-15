import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { Logger } from 'pino';
import type { Config, ResolvedPaths } from '../src/config.js';
import { MemoryStore } from '../src/store.js';

function mockLogger(): Logger {
  return {
    info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(),
    child: vi.fn().mockReturnThis(), fatal: vi.fn(), trace: vi.fn(),
    silent: vi.fn(), level: 'debug',
  } as unknown as Logger;
}

function makeConfig(overrides: Partial<Config> = {}): Config {
  return {
    max_entries: 100,
    autosave_interval_seconds: 0,
    storage_dir: '.memorandum',
    max_document_size: 16 * 1024 * 1024,
    semantic_model: 'Xenova/multilingual-e5-small',
    semantic_model_dtype: 'q8',
    semantic_enabled: false,
    semantic_debounce_seconds: 10,
    semantic_max_queue_size: 200,
    semantic_max_retries: 3,
    ...overrides,
  };
}

function makePaths(storagePath?: string): ResolvedPaths {
  const factsPath = storagePath ?? join(tmpdir(), `test-memory-${randomUUID()}.json`);
  return {
    storageDir: '.memorandum',
    factsPath,
    documentsPath: join(tmpdir(), `test-docs-${randomUUID()}`),
    cachePath: join(tmpdir(), `test-cache-${randomUUID()}`),
    configPath: '.memorandum/config.yaml',
  };
}

describe('MemoryStore - Composite key helpers', () => {
  it('makeKey creates correct composite key with NUL separator', () => {
    expect(MemoryStore.makeKey('servers', 'web01')).toBe('servers\0web01');
  });

  it('parseKey correctly splits composite key', () => {
    expect(MemoryStore.parseKey('servers\0web01')).toEqual({ namespace: 'servers', key: 'web01' });
  });

  it('parseKey handles key without NUL separator (fallback to "default")', () => {
    expect(MemoryStore.parseKey('legacy-key')).toEqual({ namespace: 'default', key: 'legacy-key' });
  });
});

describe('MemoryStore - write/read round-trip', () => {
  let store: MemoryStore;

  beforeEach(() => {
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
  });

  it('write a fact and read it back', () => {
    const result = store.write('web01', { ip: '192.168.1.10' }, 'servers');
    expect(result.created).toBe(true);
    const fact = store.read('web01', 'servers');
    expect(fact?.value).toEqual({ ip: '192.168.1.10' });
    expect(fact?.key).toBe('web01');
    expect(fact?.namespace).toBe('servers');
  });

  it('read non-existent key returns null', () => {
    expect(store.read('nonexistent', 'servers')).toBeNull();
  });

  it('write same key twice — second write returns {created: false}', () => {
    expect(store.write('web01', { ip: '192.168.1.10' }, 'servers').created).toBe(true);
    expect(store.write('web01', { ip: '192.168.1.11' }, 'servers').created).toBe(false);
    expect(store.read('web01', 'servers')?.value).toEqual({ ip: '192.168.1.11' });
  });

  it('update preserves original createdAt and sets updatedAt', async () => {
    store.write('ts-key', 'v1', 'default');
    const originalCreatedAt = store.read('ts-key', 'default')!.createdAt;
    await new Promise((r) => setTimeout(r, 10));
    store.write('ts-key', 'v2', 'default');
    const after2 = store.read('ts-key', 'default');
    expect(after2!.createdAt).toBe(originalCreatedAt);
    expect(after2!.updatedAt).toBeGreaterThan(originalCreatedAt);
  });
});

describe('MemoryStore - Namespace isolation', () => {
  let store: MemoryStore;

  beforeEach(() => {
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
  });

  it('facts in different namespaces with same key are independent', () => {
    store.write('web01', { ip: '192.168.1.10' }, 'servers');
    store.write('web01', { subnet: '10.0.0.0/24' }, 'networks');
    expect(store.read('web01', 'servers')?.value).toEqual({ ip: '192.168.1.10' });
    expect(store.read('web01', 'networks')?.value).toEqual({ subnet: '10.0.0.0/24' });
  });
});

describe('MemoryStore - LRU eviction', () => {
  it('create store with max_entries=3, write 4 facts — oldest is evicted', () => {
    const store = new MemoryStore(makeConfig({ max_entries: 3 }), makePaths(), mockLogger());
    store.write('key1', 'value1', 'default');
    store.write('key2', 'value2', 'default');
    store.write('key3', 'value3', 'default');
    store.write('key4', 'value4', 'default');
    expect(store.read('key1', 'default')).toBeNull();
    expect(store.read('key4', 'default')?.value).toBe('value4');
  });

  it('read() promotes key in LRU', () => {
    const store = new MemoryStore(makeConfig({ max_entries: 3 }), makePaths(), mockLogger());
    store.write('key1', 'value1', 'default');
    store.write('key2', 'value2', 'default');
    store.write('key3', 'value3', 'default');
    store.read('key1', 'default');
    store.write('key4', 'value4', 'default');
    expect(store.read('key1', 'default')?.value).toBe('value1');
    expect(store.read('key2', 'default')).toBeNull();
  });

  it('eviction triggers warn log', () => {
    const logger = mockLogger();
    const store = new MemoryStore(makeConfig({ max_entries: 2 }), makePaths(), logger);
    store.write('key1', 'value1', 'servers');
    store.write('key2', 'value2', 'servers');
    store.write('key3', 'value3', 'servers');
    expect(logger.warn).toHaveBeenCalledWith({ key: 'key1', namespace: 'servers' }, 'Fact evicted from memory (LRU)');
  });
});

describe('MemoryStore - TTL', () => {
  let store: MemoryStore;
  beforeEach(() => { store = new MemoryStore(makeConfig(), makePaths(), mockLogger()); });

  it('write fact with ttl_seconds — returns value with ttl info', () => {
    store.write('temp-key', 'temp-value', 'default', 10);
    const fact = store.read('temp-key', 'default');
    expect(fact?.ttl).toBeGreaterThan(0);
    expect(fact?.remainingTtl).toBeGreaterThan(0);
  });

  it('write fact without TTL — ttl fields are null', () => {
    store.write('permanent', 'forever', 'default');
    const fact = store.read('permanent', 'default');
    expect(fact?.ttl).toBeNull();
    expect(fact?.remainingTtl).toBeNull();
  });
});

describe('MemoryStore - List', () => {
  let store: MemoryStore;

  beforeEach(() => {
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    store.write('web01', { ip: '192.168.1.10' }, 'servers');
    store.write('web02', { ip: '192.168.1.11' }, 'servers');
    store.write('db01', { ip: '192.168.1.20' }, 'databases');
    store.write('net-core', { subnet: '10.0.0.0/24' }, 'networks');
  });

  it('list() returns all facts when no filters', () => {
    expect(store.list({ limit: 100, includeValues: false, includeStats: false }).facts).toHaveLength(4);
  });

  it('list() filters by namespace', () => {
    const result = store.list({ namespace: 'servers', limit: 100, includeValues: false, includeStats: false });
    expect(result.facts).toHaveLength(2);
    expect(result.facts.every((f) => f.namespace === 'servers')).toBe(true);
  });

  it('list() filters by pattern', () => {
    const result = store.list({ pattern: '*web*', limit: 100, includeValues: false, includeStats: false });
    expect(result.facts).toHaveLength(2);
  });

  it('list() respects limit', () => {
    expect(store.list({ limit: 2, includeValues: false, includeStats: false }).facts.length).toBeLessThanOrEqual(2);
  });

  it('list() includeValues=true includes values', () => {
    const result = store.list({ namespace: 'servers', limit: 100, includeValues: true, includeStats: false });
    expect(result.facts[0].value).toBeDefined();
  });

  it('list() includeStats=true includes stats', () => {
    const result = store.list({ limit: 100, includeValues: false, includeStats: true });
    expect(result.stats?.totalFacts).toBe(4);
    expect(result.stats?.namespaceCount).toBe(3);
  });
});

describe('MemoryStore - Search', () => {
  let store: MemoryStore;

  beforeEach(() => {
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    store.write('web01', { ip: '192.168.1.10', role: 'webserver' }, 'servers');
    store.write('web02', { ip: '192.168.1.11', role: 'webserver' }, 'servers');
    store.write('db01', { ip: '192.168.1.20', role: 'database' }, 'databases');
  });

  it('search() finds facts by key match', () => {
    const result = store.search({ query: 'web', limit: 10 });
    expect(result.results.length).toBe(2);
    expect(result.total_matches).toBe(2);
  });

  it('search() finds facts by value match', () => {
    const result = store.search({ query: 'database', limit: 10 });
    expect(result.results.length).toBe(1);
  });

  it('search() filters by namespace', () => {
    const result = store.search({ query: 'web', namespace: 'servers', limit: 10 });
    expect(result.results.every((r) => r.namespace === 'servers')).toBe(true);
  });

  it('search() respects limit', () => {
    const result = store.search({ query: 'web', limit: 1 });
    expect(result.results).toHaveLength(1);
    expect(result.total_matches).toBe(2);
  });

  it('search() throws on unsafe regexp', () => {
    expect(() => store.search({ query: '(a+)+b', limit: 10 })).toThrow('Unsafe regexp');
  });
});

describe('MemoryStore - Delete', () => {
  let store: MemoryStore;

  beforeEach(() => {
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    store.write('web01', { ip: '192.168.1.10' }, 'servers');
    store.write('web02', { ip: '192.168.1.11' }, 'servers');
    store.write('db01', { ip: '192.168.1.20' }, 'databases');
  });

  it('delete() removes existing fact, returns true', () => {
    expect(store.delete('web01', 'servers')).toBe(true);
    expect(store.read('web01', 'servers')).toBeNull();
  });

  it('delete() for non-existent fact returns false', () => {
    expect(store.delete('nonexistent', 'servers')).toBe(false);
  });

  it('deleteNamespace() removes all facts in namespace', () => {
    expect(store.deleteNamespace('servers')).toBe(2);
    expect(store.read('web01', 'servers')).toBeNull();
    expect(store.read('db01', 'databases')).not.toBeNull();
  });
});

describe('MemoryStore - Persistence (save/load)', () => {
  let storagePath: string;
  let config: Config;
  let paths: ResolvedPaths;
  let logger: Logger;

  beforeEach(() => {
    storagePath = join(tmpdir(), `test-memory-${randomUUID()}.json`);
    config = makeConfig();
    paths = makePaths(storagePath);
    logger = mockLogger();
  });

  afterEach(() => {
    if (existsSync(storagePath)) rmSync(storagePath, { force: true });
  });

  it('save() creates JSON file on disk', async () => {
    const store = new MemoryStore(config, paths, logger);
    store.write('web01', { ip: '192.168.1.10' }, 'servers');
    await store.save();
    expect(existsSync(storagePath)).toBe(true);
  });

  it('load() restores facts from disk', async () => {
    const store1 = new MemoryStore(config, paths, logger);
    store1.write('web01', { ip: '192.168.1.10' }, 'servers');
    await store1.save();

    const store2 = new MemoryStore(config, paths, logger);
    await store2.load();
    expect(store2.read('web01', 'servers')?.value).toEqual({ ip: '192.168.1.10' });
  });

  it('load() with missing file starts empty', async () => {
    const store = new MemoryStore(config, paths, logger);
    await expect(store.load()).resolves.toBeUndefined();
    expect(store.size).toBe(0);
  });

  it('load() with corrupt file starts empty', async () => {
    writeFileSync(storagePath, 'invalid json', 'utf-8');
    const store = new MemoryStore(config, paths, logger);
    await expect(store.load()).resolves.toBeUndefined();
    expect(store.size).toBe(0);
  });
});

describe('MemoryStore - Export/Import', () => {
  let store: MemoryStore;

  beforeEach(() => {
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    store.write('web01', { ip: '192.168.1.10' }, 'servers');
    store.write('db01', { ip: '192.168.1.20' }, 'databases');
  });

  it('export() returns object with version=1', () => {
    const exported = store.export();
    expect(exported.version).toBe(1);
    expect((exported.data as unknown[]).length).toBe(2);
  });

  it('import() loads facts from exported JSON string', () => {
    const store2 = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    const result = store2.import(JSON.stringify(store.export()), true);
    expect(result.imported_count).toBe(2);
    expect(store2.read('web01', 'servers')?.value).toEqual({ ip: '192.168.1.10' });
  });

  it('import() with merge=false clears existing facts', () => {
    const store2 = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    store2.write('existing', 'value', 'default');
    store2.import(JSON.stringify(store.export()), false);
    expect(store2.read('existing', 'default')).toBeNull();
    expect(store2.size).toBe(2);
  });

  it('import() throws on invalid JSON', () => {
    const store2 = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    expect(() => store2.import('invalid json', true)).toThrow('not valid JSON');
  });
});

describe('MemoryStore - Dirty tracking', () => {
  let storagePath: string;
  let config: Config;
  let paths: ResolvedPaths;
  let logger: Logger;

  beforeEach(() => {
    storagePath = join(tmpdir(), `test-memory-${randomUUID()}.json`);
    config = makeConfig();
    paths = makePaths(storagePath);
    logger = mockLogger();
  });

  afterEach(() => {
    if (existsSync(storagePath)) rmSync(storagePath, { force: true });
  });

  it('save() returns false when no mutations', async () => {
    const store = new MemoryStore(config, paths, logger);
    expect(await store.save()).toBe(false);
  });

  it('save() returns true after write()', async () => {
    const store = new MemoryStore(config, paths, logger);
    store.write('key1', 'value1', 'default');
    expect(await store.save()).toBe(true);
  });

  it('save() returns false on second call without new mutations', async () => {
    const store = new MemoryStore(config, paths, logger);
    store.write('key1', 'value1', 'default');
    await store.save();
    expect(await store.save()).toBe(false);
  });
});

describe('MemoryStore - Autosave', () => {
  it('startAutosave schedules periodic saves', async () => {
    const store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    const saveSpy = vi.spyOn(store, 'save');
    store.startAutosave(100);
    await new Promise((r) => setTimeout(r, 250));
    expect(saveSpy).toHaveBeenCalled();
    store.stopAutosave();
  });

  it('stopAutosave clears the interval', async () => {
    const store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    const saveSpy = vi.spyOn(store, 'save');
    store.startAutosave(100);
    store.stopAutosave();
    await new Promise((r) => setTimeout(r, 250));
    expect(saveSpy).not.toHaveBeenCalled();
  });
});
