import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { Logger } from 'pino';
import type { Config, ResolvedPaths } from '../src/config.js';
import { MemoryStore } from '../src/store.js';
import { registerMemoryTools } from '../src/tools.js';
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
    max_entries: 1000, autosave_interval_seconds: 60, storage_dir: '.memorandum',
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

type ToolHandler = (input: Record<string, unknown>, extra: unknown) => Promise<unknown>;

interface RegisteredToolCall {
  name: string;
  config: Record<string, unknown>;
  handler: ToolHandler;
}

function createMockServer() {
  const tools: RegisteredToolCall[] = [];
  return {
    registerTool: vi.fn((name: string, config: Record<string, unknown>, handler: ToolHandler) => {
      tools.push({ name, config, handler });
      return { enable: vi.fn(), disable: vi.fn(), remove: vi.fn() };
    }),
    tools,
  };
}

function getToolHandler(tools: RegisteredToolCall[], name: string): ToolHandler {
  const tool = tools.find((t) => t.name === name);
  if (!tool) throw new Error(`Tool '${name}' not registered`);
  return tool.handler;
}

function parseResponse(response: unknown): Record<string, unknown> {
  const resp = response as { content: Array<{ type: string; text: string }>; isError?: boolean };
  return JSON.parse(resp.content[0].text) as Record<string, unknown>;
}

describe('registerMemoryTools - Tool registration', () => {
  it('registers 3 tools: memory_write, memory_read, memory_manage', () => {
    const mockServer = createMockServer();
    const store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    registerMemoryTools(mockServer as any, store, mockLogger());
    expect(mockServer.registerTool).toHaveBeenCalledTimes(3);
    const toolNames = mockServer.tools.map((t) => t.name);
    expect(toolNames).toContain('memory_write');
    expect(toolNames).toContain('memory_read');
    expect(toolNames).toContain('memory_manage');
  });
});

describe('memory_write tool handler', () => {
  let mockServer: ReturnType<typeof createMockServer>;
  let store: MemoryStore;
  let handler: ToolHandler;

  beforeEach(() => {
    mockServer = createMockServer();
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    registerMemoryTools(mockServer as any, store, mockLogger());
    handler = getToolHandler(mockServer.tools, 'memory_write');
  });

  it('writes a fact and returns success', async () => {
    const response = await handler({ key: 'test-key', value: 'test-value' }, undefined);
    const output = parseResponse(response);
    expect(output.success).toBe(true);
    expect(output.key).toBe('test-key');
    expect(output.namespace).toBe('default');
    expect(output.created).toBe(true);
  });

  it('returns created: false when updating existing key', async () => {
    await handler({ key: 'key2', value: 'value1' }, undefined);
    const response = await handler({ key: 'key2', value: 'value2' }, undefined);
    expect(parseResponse(response).created).toBe(false);
  });

  it('accepts custom namespace', async () => {
    const response = await handler({ key: 'key3', value: 'value3', namespace: 'servers' }, undefined);
    expect(parseResponse(response).namespace).toBe('servers');
  });

  it('accepts ttl_seconds parameter', async () => {
    await handler({ key: 'key4', value: 'value4', ttl_seconds: 300 }, undefined);
    const fact = store.read('key4', 'default');
    expect(fact?.ttl).toBeGreaterThan(0);
    expect(fact?.ttl).toBeLessThanOrEqual(300);
  });

  it('stores complex object values', async () => {
    const complexValue = { server: 'web-01', tags: ['production'], metadata: { region: 'us-west' } };
    await handler({ key: 'server-web-01', value: complexValue, namespace: 'servers' }, undefined);
    expect(store.read('server-web-01', 'servers')?.value).toEqual(complexValue);
  });
});

describe('memory_read tool handler', () => {
  let mockServer: ReturnType<typeof createMockServer>;
  let store: MemoryStore;
  let handler: ToolHandler;

  beforeEach(() => {
    mockServer = createMockServer();
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    registerMemoryTools(mockServer as any, store, mockLogger());
    handler = getToolHandler(mockServer.tools, 'memory_read');
  });

  it('returns fact metadata for existing key', async () => {
    store.write('read-key1', { data: 'test' }, 'default');
    const output = parseResponse(await handler({ key: 'read-key1' }, undefined));
    expect(output.key).toBe('read-key1');
    expect(output.value).toEqual({ data: 'test' });
  });

  it('returns {found: false} for non-existent key', async () => {
    const output = parseResponse(await handler({ key: 'non-existent' }, undefined));
    expect(output.found).toBe(false);
  });

  it('returns ttl info for facts with TTL', async () => {
    store.write('key4', 'value4', 'default', 300);
    const output = parseResponse(await handler({ key: 'key4' }, undefined));
    expect(output.ttl).toBeGreaterThan(0);
  });

  it('returns null ttl for facts without TTL', async () => {
    store.write('key5', 'value5', 'default');
    const output = parseResponse(await handler({ key: 'key5' }, undefined));
    expect(output.ttl).toBeNull();
  });
});

describe('memory_manage tool handler - list action', () => {
  let handler: ToolHandler;
  let store: MemoryStore;

  beforeEach(() => {
    const mockServer = createMockServer();
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    registerMemoryTools(mockServer as any, store, mockLogger());
    handler = getToolHandler(mockServer.tools, 'memory_manage');
    store.write('key1', 'value1', 'default');
    store.write('key2', 'value2', 'servers');
    store.write('key3', 'value3', 'servers');
  });

  it('returns facts list', async () => {
    const output = parseResponse(await handler({ action: 'list' }, undefined));
    expect(Array.isArray(output.facts)).toBe(true);
    expect((output.facts as unknown[]).length).toBeGreaterThanOrEqual(3);
  });

  it('filters by namespace', async () => {
    const output = parseResponse(await handler({ action: 'list', namespace: 'servers' }, undefined));
    const facts = output.facts as Array<Record<string, unknown>>;
    expect(facts).toHaveLength(2);
    expect(facts.every((f) => f.namespace === 'servers')).toBe(true);
  });

  it('includes stats when requested', async () => {
    const output = parseResponse(await handler({ action: 'list', include_stats: true }, undefined));
    expect(output.stats).toBeDefined();
    expect((output.stats as Record<string, unknown>).maxEntries).toBe(1000);
  });

  it('filters by pattern', async () => {
    store.write('server-web-01', 'v', 'default');
    store.write('client-app-01', 'v', 'default');
    const output = parseResponse(await handler({ action: 'list', pattern: 'server-*' }, undefined));
    const facts = output.facts as Array<Record<string, unknown>>;
    expect(facts.every((f) => (f.key as string).startsWith('server-'))).toBe(true);
  });
});

describe('memory_manage tool handler - namespaces action', () => {
  it('returns namespaces with counts', async () => {
    const mockServer = createMockServer();
    const store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    registerMemoryTools(mockServer as any, store, mockLogger());
    const handler = getToolHandler(mockServer.tools, 'memory_manage');

    store.write('key1', 'value1', 'default');
    store.write('key2', 'value2', 'default');
    store.write('key3', 'value3', 'servers');

    const output = parseResponse(await handler({ action: 'namespaces' }, undefined));
    const namespaces = output.namespaces as Array<Record<string, unknown>>;
    expect(namespaces.length).toBe(2);
    expect(namespaces.find((ns) => ns.namespace === 'default')?.count).toBe(2);
  });
});

describe('memory_manage tool handler - search action', () => {
  let handler: ToolHandler;
  let store: MemoryStore;

  beforeEach(() => {
    const mockServer = createMockServer();
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    registerMemoryTools(mockServer as any, store, mockLogger());
    handler = getToolHandler(mockServer.tools, 'memory_manage');
    store.write('server-web-01', { hostname: 'web-01.example.com' }, 'servers');
    store.write('server-db-01', { hostname: 'db-01.example.com' }, 'servers');
  });

  it('returns search results with match_in field', async () => {
    const output = parseResponse(await handler({ action: 'search', query: 'server' }, undefined));
    const results = output.results as Array<Record<string, unknown>>;
    expect(results.length).toBe(2);
    expect(results[0].match_in).toBeDefined();
  });

  it('returns total_matches count', async () => {
    const output = parseResponse(await handler({ action: 'search', query: 'server' }, undefined));
    expect(output.total_matches).toBe(2);
  });
});

describe('memory_manage tool handler - delete action', () => {
  let handler: ToolHandler;
  let store: MemoryStore;

  beforeEach(() => {
    const mockServer = createMockServer();
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    registerMemoryTools(mockServer as any, store, mockLogger());
    handler = getToolHandler(mockServer.tools, 'memory_manage');
    store.write('delete-key1', 'value1', 'default');
  });

  it('deletes existing fact', async () => {
    const output = parseResponse(await handler({ action: 'delete', key: 'delete-key1' }, undefined));
    expect(output.success).toBe(true);
    expect(output.deleted).toBe(true);
    expect(store.read('delete-key1', 'default')).toBeNull();
  });

  it('returns deleted: false for non-existent fact', async () => {
    const output = parseResponse(await handler({ action: 'delete', key: 'non-existent' }, undefined));
    expect(output.deleted).toBe(false);
  });
});

describe('memory_manage tool handler - delete_namespace action', () => {
  it('deletes all facts in namespace', async () => {
    const mockServer = createMockServer();
    const store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    registerMemoryTools(mockServer as any, store, mockLogger());
    const handler = getToolHandler(mockServer.tools, 'memory_manage');

    store.write('key1', 'v', 'servers');
    store.write('key2', 'v', 'servers');
    store.write('key3', 'v', 'default');

    const output = parseResponse(await handler({ action: 'delete_namespace', namespace: 'servers' }, undefined));
    expect(output.deleted_count).toBe(2);
    expect(store.read('key3', 'default')).toBeDefined();
  });
});

describe('memory_manage tool handler - export/import actions', () => {
  let handler: ToolHandler;
  let store: MemoryStore;

  beforeEach(() => {
    const mockServer = createMockServer();
    store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    registerMemoryTools(mockServer as any, store, mockLogger());
    handler = getToolHandler(mockServer.tools, 'memory_manage');
    store.write('key1', 'value1', 'default');
  });

  it('export returns version 1 format', async () => {
    const output = parseResponse(await handler({ action: 'export' }, undefined));
    expect(output.version).toBe(1);
    expect(Array.isArray(output.data)).toBe(true);
  });

  it('import loads facts successfully', async () => {
    const exportData = {
      version: 1, exported_at: Date.now(), facts_count: 1,
      data: [{ key: 'import-key1', namespace: 'default', value: 'value1', createdAt: Date.now() }],
    };
    const output = parseResponse(await handler({ action: 'import', data: JSON.stringify(exportData) }, undefined));
    expect(output.success).toBe(true);
    expect(output.imported_count).toBe(1);
  });
});

describe('memory_manage tool handler - sync action', () => {
  it('calls syncFn and returns its result', async () => {
    const mockServer = createMockServer();
    const store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    const syncFn = vi.fn().mockResolvedValue({ flushed_embeddings: 42, saved_stores: ['facts', 'vectors'] });
    registerMemoryTools(mockServer as any, store, mockLogger(), syncFn);
    const handler = getToolHandler(mockServer.tools, 'memory_manage');

    const output = parseResponse(await handler({ action: 'sync' }, undefined));
    expect(syncFn).toHaveBeenCalledOnce();
    expect(output.flushed_embeddings).toBe(42);
  });

  it('falls back to store.save() when syncFn is absent', async () => {
    const mockServer = createMockServer();
    const store = new MemoryStore(makeConfig(), makePaths(), mockLogger());
    registerMemoryTools(mockServer as any, store, mockLogger());
    const handler = getToolHandler(mockServer.tools, 'memory_manage');
    vi.spyOn(store, 'save').mockResolvedValue(true);

    const output = parseResponse(await handler({ action: 'sync' }, undefined));
    expect(output.flushed_embeddings).toBe(0);
    expect(output.saved_stores).toEqual(['facts']);
  });
});
