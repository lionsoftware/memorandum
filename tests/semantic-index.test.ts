import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Logger } from 'pino';
import type { SearchResult } from '../src/semantic-types.js';
import { makeFactVectorId, makeDocVectorId, INDEX_STATUS_HINTS } from '../src/semantic-types.js';
import { SemanticIndex } from '../src/semantic-index.js';

// ============================================================================
// Test helpers
// ============================================================================

function mockLogger(): Logger {
  return {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
    child: vi.fn().mockReturnThis(),
    fatal: vi.fn(),
    trace: vi.fn(),
    silent: vi.fn(),
    level: 'debug',
  } as unknown as Logger;
}

function mockEmbedder() {
  let callCount = 0;
  return {
    embedQuery: vi.fn().mockImplementation(async () => {
      callCount++;
      return [callCount * 0.1, callCount * 0.2, callCount * 0.3];
    }),
    embedPassage: vi.fn().mockImplementation(async () => {
      callCount++;
      return [callCount * 0.1, callCount * 0.2, callCount * 0.3];
    }),
    get isLoaded() { return true; },
    get isLoading() { return false; },
    get isUnavailable() { return false; },
    get unavailableReason() { return null; },
    repairAndLoad: vi.fn().mockResolvedValue(undefined),
  };
}

function mockVectorStore() {
  const entries: Map<string, any> = new Map();
  return {
    upsert: vi.fn().mockImplementation((entry: any) => entries.set(entry.id, entry)),
    remove: vi.fn().mockImplementation((id: string) => {
      const existed = entries.has(id);
      entries.delete(id);
      return existed;
    }),
    query: vi.fn().mockReturnValue([]),
    rebuild: vi.fn().mockImplementation((newEntries: any[]) => {
      entries.clear();
      newEntries.forEach((e) => entries.set(e.id, e));
    }),
    save: vi.fn().mockResolvedValue(undefined),
    load: vi.fn(),
    allEntries: vi.fn().mockImplementation(() => Array.from(entries.values())),
    getFailedEmbeddings: vi.fn().mockReturnValue({}),
    setFailedEmbeddings: vi.fn(),
    clearFailure: vi.fn(),
    recordFailure: vi.fn(),
    get entryCount() { return entries.size; },
    get currentModelId() { return 'test-model'; },
    get indexModelId() { return 'test-model'; },
  };
}

function mockMemoryStoreForReindex() {
  return {
    list: vi.fn().mockReturnValue({
      facts: [
        { key: 'k1', namespace: 'ns1', value: 'val1' },
        { key: 'k2', namespace: 'ns2', value: 'val2' },
      ],
    }),
    get size() { return 2; },
    get maxEntries() { return 1000; },
  };
}

function mockDocumentStoreForReindex() {
  return {
    list: vi.fn().mockReturnValue({
      documents: [
        {
          id: 'doc-001',
          title: 'Doc1',
          content_type: 'text/plain',
          tags: ['tag1'],
          topic: 'topic1',
        },
      ],
      total: 1,
    }),
    read: vi.fn().mockReturnValue({ body: 'Document body text' }),
    getIndex: vi.fn().mockReturnValue({ documents: [{ id: 'doc-001' }] }),
  };
}

// ============================================================================
// US1: Document Indexing
// ============================================================================

describe('SemanticIndex - document indexing (US1)', () => {
  let embedder: ReturnType<typeof mockEmbedder>;
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let logger: Logger;
  let index: InstanceType<typeof SemanticIndex>;

  beforeEach(() => {
    embedder = mockEmbedder();
    vectorStore = mockVectorStore();
    logger = mockLogger();
    index = new SemanticIndex(embedder as any, vectorStore as any, logger);
  });

  it('indexDocument calls embedder.embedPassage with the document body for text content types', async () => {
    await index.indexDocument(
      'doc-001',
      { title: 'Test Doc', tags: ['t1'], contentType: 'text/plain' },
      'The body text',
    );

    expect(embedder.embedPassage).toHaveBeenCalledOnce();
    expect(embedder.embedPassage).toHaveBeenCalledWith('The body text');
  });

  it('indexDocument calls vectorStore.upsert with id prefixed by "doc:"', async () => {
    await index.indexDocument(
      'doc-001',
      { title: 'Test Doc', tags: [], contentType: 'text/plain' },
      'body',
    );

    expect(vectorStore.upsert).toHaveBeenCalledOnce();
    const entry = vectorStore.upsert.mock.calls[0][0];
    expect(entry.id).toBe(makeDocVectorId('doc-001'));
    expect(entry.id).toBe('doc:doc-001');
    expect(entry.source).toBe('document');
  });

  it('indexDocument with text content type uses body text for embedding', async () => {
    await index.indexDocument(
      'doc-txt',
      { title: 'Ignored Title', tags: [], contentType: 'text/plain' },
      'Actual body content',
    );

    expect(embedder.embedPassage).toHaveBeenCalledWith('Actual body content');
  });

  it('indexDocument with application/json content type uses metadata for embedding (json is binary)', async () => {
    await index.indexDocument(
      'doc-json',
      { title: 'JSON Doc', tags: [], contentType: 'application/json' },
      '{"key": "value"}',
    );

    // application/json is binary — body is NOT used, only metadata (title)
    expect(embedder.embedPassage).toHaveBeenCalledWith('JSON Doc');
  });

  it('indexDocument with application/yaml content type uses metadata for embedding (yaml is binary)', async () => {
    await index.indexDocument(
      'doc-yaml',
      { title: 'YAML Doc', tags: [], contentType: 'application/yaml' },
      'key: value',
    );

    // application/yaml is binary — body is NOT used, only metadata (title)
    expect(embedder.embedPassage).toHaveBeenCalledWith('YAML Doc');
  });

  it('indexDocument with binary content type uses metadata (title + description + tags)', async () => {
    await index.indexDocument(
      'doc-bin',
      {
        title: 'Binary Doc',
        description: 'A binary file',
        tags: ['binaries', 'archive'],
        contentType: 'application/octet-stream',
      },
      'some binary body that should be ignored',
    );

    expect(embedder.embedPassage).toHaveBeenCalledWith(
      'Binary Doc A binary file binaries archive',
    );
  });

  it('indexDocument with binary content type and no body uses metadata only', async () => {
    await index.indexDocument(
      'doc-img',
      { title: 'Image', tags: ['photo'], contentType: 'image/png' },
    );

    expect(embedder.embedPassage).toHaveBeenCalledWith('Image photo');
  });

  it('indexDocument stores correct metadata in the vector entry', async () => {
    await index.indexDocument(
      'doc-meta',
      {
        title: 'Meta Doc',
        topic: 'testing',
        tags: ['unit', 'test'],
        contentType: 'text/plain',
        description: 'A test doc',
      },
      'body content',
    );

    const entry = vectorStore.upsert.mock.calls[0][0];
    expect(entry.metadata).toMatchObject({
      documentId: 'doc-meta',
      title: 'Meta Doc',
      topic: 'testing',
      tags: ['unit', 'test'],
      contentType: 'text/plain',
    });
  });

  it('indexDocument does not throw when embedder fails — logs warning instead', async () => {
    embedder.embedPassage.mockRejectedValueOnce(new Error('Embedding failed'));

    await expect(
      index.indexDocument('doc-fail', { title: 'Fail', tags: [], contentType: 'text/plain' }, 'body'),
    ).resolves.toBeUndefined();

    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({ documentId: 'doc-fail' }),
      'Failed to index document',
    );
    expect(vectorStore.upsert).not.toHaveBeenCalled();
  });

  it('removeDocument calls vectorStore.remove with the correct prefixed id', () => {
    index.removeDocument('doc-001');

    expect(vectorStore.remove).toHaveBeenCalledOnce();
    expect(vectorStore.remove).toHaveBeenCalledWith(makeDocVectorId('doc-001'));
    expect(vectorStore.remove).toHaveBeenCalledWith('doc:doc-001');
  });

  it('removeDocument triggers save when entry existed', () => {
    // remove returns true (entry existed)
    vectorStore.remove.mockReturnValueOnce(true);

    index.removeDocument('doc-001');

    expect(vectorStore.save).toHaveBeenCalled();
  });

  it('removeDocument does not trigger save when entry did not exist', () => {
    // remove returns false (entry not found)
    vectorStore.remove.mockReturnValueOnce(false);

    index.removeDocument('doc-ghost');

    expect(vectorStore.save).not.toHaveBeenCalled();
  });

  it('index consistency: indexDocument → vectorStore has entry → removeDocument → entry gone', async () => {
    await index.indexDocument(
      'doc-cycle',
      { title: 'Cycle Doc', tags: [], contentType: 'text/plain' },
      'body',
    );
    expect(vectorStore.entryCount).toBe(1);

    index.removeDocument('doc-cycle');
    expect(vectorStore.entryCount).toBe(0);
  });
});

// ============================================================================
// US2: Fact Indexing
// ============================================================================

describe('SemanticIndex - fact indexing (US2)', () => {
  let embedder: ReturnType<typeof mockEmbedder>;
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let logger: Logger;
  let index: InstanceType<typeof SemanticIndex>;

  beforeEach(() => {
    embedder = mockEmbedder();
    vectorStore = mockVectorStore();
    logger = mockLogger();
    index = new SemanticIndex(embedder as any, vectorStore as any, logger);
  });

  it('indexFact calls embedder.embedPassage with "namespace: key — value"', async () => {
    await index.indexFact('my-key', 'my-ns', 'my-value');

    expect(embedder.embedPassage).toHaveBeenCalledOnce();
    expect(embedder.embedPassage).toHaveBeenCalledWith('my-ns: my-key — my-value');
  });

  it('indexFact with non-string value JSON-stringifies it', async () => {
    await index.indexFact('obj-key', 'ns', { foo: 'bar', num: 42 });

    expect(embedder.embedPassage).toHaveBeenCalledWith(
      'ns: obj-key — {"foo":"bar","num":42}',
    );
  });

  it('indexFact calls vectorStore.upsert with id "fact:{namespace}\\0{key}"', async () => {
    await index.indexFact('the-key', 'the-ns', 'the-value');

    expect(vectorStore.upsert).toHaveBeenCalledOnce();
    const entry = vectorStore.upsert.mock.calls[0][0];
    expect(entry.id).toBe(makeFactVectorId('the-ns', 'the-key'));
    expect(entry.id).toBe('fact:the-ns\0the-key');
    expect(entry.source).toBe('fact');
  });

  it('indexFact stores key and namespace in metadata', async () => {
    await index.indexFact('meta-key', 'meta-ns', 'meta-value');

    const entry = vectorStore.upsert.mock.calls[0][0];
    expect(entry.metadata).toMatchObject({
      key: 'meta-key',
      namespace: 'meta-ns',
    });
  });

  it('indexFact truncates preview to 200 chars when value is longer', async () => {
    const longValue = 'x'.repeat(300);
    await index.indexFact('long-key', 'ns', longValue);

    const entry = vectorStore.upsert.mock.calls[0][0];
    expect(entry.metadata.preview).toHaveLength(200);
    expect(entry.metadata.preview).toBe(longValue.slice(0, 200));
  });

  it('indexFact preserves full preview when value is exactly 200 chars', async () => {
    const exactValue = 'a'.repeat(200);
    await index.indexFact('exact-key', 'ns', exactValue);

    const entry = vectorStore.upsert.mock.calls[0][0];
    expect(entry.metadata.preview).toHaveLength(200);
    expect(entry.metadata.preview).toBe(exactValue);
  });

  it('indexFact preserves full preview when value is shorter than 200 chars', async () => {
    const shortValue = 'short value';
    await index.indexFact('short-key', 'ns', shortValue);

    const entry = vectorStore.upsert.mock.calls[0][0];
    expect(entry.metadata.preview).toBe(shortValue);
  });

  it('indexFact does not throw when embedder fails — logs warning instead', async () => {
    embedder.embedPassage.mockRejectedValueOnce(new Error('Embedder down'));

    await expect(index.indexFact('fail-key', 'ns', 'value')).resolves.toBeUndefined();

    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({ key: 'fail-key', namespace: 'ns' }),
      'Failed to index fact',
    );
    expect(vectorStore.upsert).not.toHaveBeenCalled();
  });

  it('removeFact calls vectorStore.remove with "fact:{namespace}\\0{key}"', () => {
    index.removeFact('rem-key', 'rem-ns');

    expect(vectorStore.remove).toHaveBeenCalledOnce();
    expect(vectorStore.remove).toHaveBeenCalledWith(makeFactVectorId('rem-ns', 'rem-key'));
    expect(vectorStore.remove).toHaveBeenCalledWith('fact:rem-ns\0rem-key');
  });

  it('removeFact triggers save when entry existed', () => {
    vectorStore.remove.mockReturnValueOnce(true);

    index.removeFact('rem-key', 'rem-ns');

    expect(vectorStore.save).toHaveBeenCalled();
  });

  it('removeFact does not trigger save when entry did not exist', () => {
    vectorStore.remove.mockReturnValueOnce(false);

    index.removeFact('ghost-key', 'ns');

    expect(vectorStore.save).not.toHaveBeenCalled();
  });

  it('re-indexing a fact replaces previous entry (upsert with same id)', async () => {
    await index.indexFact('key', 'ns', 'old-value');
    await index.indexFact('key', 'ns', 'new-value');

    // Two calls to embedPassage (index + re-index)
    expect(embedder.embedPassage).toHaveBeenCalledTimes(2);
    // Two calls to upsert, both with same id
    expect(vectorStore.upsert).toHaveBeenCalledTimes(2);
    const firstId = vectorStore.upsert.mock.calls[0][0].id;
    const secondId = vectorStore.upsert.mock.calls[1][0].id;
    expect(firstId).toBe(secondId);
  });

  it('index consistency: indexFact → entry in store → removeFact → entry gone', async () => {
    await index.indexFact('cycle-key', 'ns', 'value');
    expect(vectorStore.entryCount).toBe(1);

    index.removeFact('cycle-key', 'ns');
    expect(vectorStore.entryCount).toBe(0);
  });
});

// ============================================================================
// Search
// ============================================================================

describe('SemanticIndex - search', () => {
  let embedder: ReturnType<typeof mockEmbedder>;
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let logger: Logger;
  let index: InstanceType<typeof SemanticIndex>;

  beforeEach(() => {
    embedder = mockEmbedder();
    vectorStore = mockVectorStore();
    logger = mockLogger();
    index = new SemanticIndex(embedder as any, vectorStore as any, logger);
  });

  it('search returns empty results and status="empty" when vectorStore has no entries', async () => {
    // entryCount is 0 by default in mockVectorStore
    const result = await index.search('any query');

    expect(result.status).toBe('empty');
    expect(result.results).toEqual([]);
    expect(result.total).toBe(0);
    expect(embedder.embedQuery).not.toHaveBeenCalled();
  });

  it('search returns empty results and status="model_unavailable" when embedder is unavailable', async () => {
    Object.defineProperty(embedder, 'isUnavailable', { value: true, configurable: true });
    Object.defineProperty(embedder, 'isLoaded', { value: false, configurable: true });

    const result = await index.search('any query');

    expect(result.status).toBe('model_unavailable');
    expect(result.results).toEqual([]);
    expect(embedder.embedQuery).not.toHaveBeenCalled();
  });

  it('search returns empty results and status="model_loading" when embedder is loading', async () => {
    Object.defineProperty(embedder, 'isLoading', { value: true, configurable: true });
    Object.defineProperty(embedder, 'isLoaded', { value: false, configurable: true });

    const result = await index.search('any query');

    expect(result.status).toBe('model_loading');
    expect(result.results).toEqual([]);
    expect(embedder.embedQuery).not.toHaveBeenCalled();
  });

  it('search calls embedder.embedQuery with the query text when index has entries', async () => {
    // Add an entry so vectorStore.entryCount > 0
    await index.indexFact('k', 'ns', 'v');
    // embedPassage call consumed; reset for clear assertion on embedQuery
    embedder.embedQuery.mockClear();

    await index.search('my search query');

    expect(embedder.embedQuery).toHaveBeenCalledOnce();
    expect(embedder.embedQuery).toHaveBeenCalledWith('my search query');
  });

  it('search calls vectorStore.query with the query vector', async () => {
    await index.indexFact('k', 'ns', 'v');
    const queryVector = [0.9, 1.8, 2.7];
    embedder.embedQuery.mockResolvedValueOnce(queryVector);

    await index.search('query');

    expect(vectorStore.query).toHaveBeenCalledWith(
      queryVector,
      expect.objectContaining({ limit: Number.MAX_SAFE_INTEGER }),
    );
  });

  it('search passes filter options through to vectorStore.query', async () => {
    await index.indexFact('k', 'ns', 'v');

    const options = { source: 'facts' as const, namespace: 'ns', limit: 5 };
    await index.search('query', options);

    expect(vectorStore.query).toHaveBeenCalledWith(
      expect.any(Array),
      expect.objectContaining({ source: 'facts', namespace: 'ns', limit: Number.MAX_SAFE_INTEGER }),
    );
  });

  it('search returns results from vectorStore.query', async () => {
    await index.indexFact('k', 'ns', 'v');

    const mockResults: SearchResult[] = [
      {
        id: 'fact:ns\0k',
        source: 'fact',
        score: 0.95,
        metadata: { key: 'k', namespace: 'ns', preview: 'v' },
      },
    ];
    vectorStore.query.mockReturnValueOnce(mockResults);

    const result = await index.search('query');

    expect(result.results).toEqual(mockResults);
    expect(result.total).toBe(1);
  });

  it('search respects limit option by slicing results', async () => {
    await index.indexFact('k', 'ns', 'v');

    const manyResults: SearchResult[] = Array.from({ length: 20 }, (_, i) => ({
      id: `fact:ns\0k${i}`,
      source: 'fact' as const,
      score: 1 - i * 0.01,
      metadata: { key: `k${i}`, namespace: 'ns', preview: `v${i}` },
    }));
    vectorStore.query.mockReturnValueOnce(manyResults);

    const result = await index.search('query', { limit: 5 });

    expect(result.results).toHaveLength(5);
    expect(result.total).toBe(20);
  });

  it('search defaults to limit=10 when not specified', async () => {
    await index.indexFact('k', 'ns', 'v');

    const manyResults: SearchResult[] = Array.from({ length: 15 }, (_, i) => ({
      id: `fact:ns\0k${i}`,
      source: 'fact' as const,
      score: 1 - i * 0.01,
      metadata: { key: `k${i}`, namespace: 'ns', preview: `v${i}` },
    }));
    vectorStore.query.mockReturnValueOnce(manyResults);

    const result = await index.search('query');

    expect(result.results).toHaveLength(10);
    expect(result.total).toBe(15);
  });

  it('search includes status and hint in the response', async () => {
    await index.indexFact('k', 'ns', 'v');
    vectorStore.query.mockReturnValueOnce([]);

    const result = await index.search('query');

    expect(result.status).toBe('ready');
    expect(result).toHaveProperty('hint');
  });

  it('search returns status="model_unavailable" and empty results when embedQuery throws', async () => {
    await index.indexFact('k', 'ns', 'v');
    embedder.embedQuery.mockRejectedValueOnce(new Error('Embedder failure'));

    const result = await index.search('query');

    expect(result.status).toBe('model_unavailable');
    expect(result.results).toEqual([]);
    expect(result.total).toBe(0);
    expect(logger.warn).toHaveBeenCalled();
  });
});

// ============================================================================
// Status Diagnostics
// ============================================================================

describe('SemanticIndex - getStatus diagnostics', () => {
  let embedder: ReturnType<typeof mockEmbedder>;
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let logger: Logger;
  let index: InstanceType<typeof SemanticIndex>;

  beforeEach(() => {
    embedder = mockEmbedder();
    vectorStore = mockVectorStore();
    logger = mockLogger();
    index = new SemanticIndex(embedder as any, vectorStore as any, logger);
  });

  it('getStatus returns "empty" when vectorStore.entryCount is 0', () => {
    expect(index.getStatus()).toBe('empty');
  });

  it('getStatus returns "model_unavailable" when embedder.isUnavailable is true', () => {
    Object.defineProperty(embedder, 'isUnavailable', { value: true, configurable: true });
    Object.defineProperty(embedder, 'isLoaded', { value: false, configurable: true });

    expect(index.getStatus()).toBe('model_unavailable');
  });

  it('getStatus returns "model_loading" when embedder.isLoading is true', () => {
    Object.defineProperty(embedder, 'isLoading', { value: true, configurable: true });
    Object.defineProperty(embedder, 'isLoaded', { value: false, configurable: true });
    // isUnavailable must be false for loading to take precedence
    Object.defineProperty(embedder, 'isUnavailable', { value: false, configurable: true });

    expect(index.getStatus()).toBe('model_loading');
  });

  it('getStatus returns "model_unavailable" before "model_loading" (unavailable takes priority)', () => {
    Object.defineProperty(embedder, 'isUnavailable', { value: true, configurable: true });
    Object.defineProperty(embedder, 'isLoading', { value: true, configurable: true });

    expect(index.getStatus()).toBe('model_unavailable');
  });

  it('getStatus returns "ready" when index has entries and model matches', async () => {
    await index.indexFact('k', 'ns', 'v');
    // entryCount is now 1; indexModelId === currentModelId === 'test-model'

    expect(index.getStatus()).toBe('ready');
  });

  it('getStatus returns "stale_model" when indexModelId !== currentModelId', async () => {
    await index.indexFact('k', 'ns', 'v');

    // Override indexModelId to differ from currentModelId
    Object.defineProperty(vectorStore, 'indexModelId', {
      value: 'old-model',
      configurable: true,
    });
    Object.defineProperty(vectorStore, 'currentModelId', {
      value: 'new-model',
      configurable: true,
    });

    expect(index.getStatus()).toBe('stale_model');
  });

  it('getStatus returns "ready" when indexModelId is null (no saved index yet)', async () => {
    await index.indexFact('k', 'ns', 'v');

    Object.defineProperty(vectorStore, 'indexModelId', {
      value: null,
      configurable: true,
    });

    // null indexModelId means no stale check applies → ready
    expect(index.getStatus()).toBe('ready');
  });

  it('getStatus returns "partial" when entryCount < total items in stores', async () => {
    // Add one entry to the vector store
    await index.indexFact('k', 'ns', 'v');

    // Attach stores with 2 facts and 1 document = 3 total, but entryCount is 1
    const memStore = { list: vi.fn(), size: 2, maxEntries: 1000 };
    const docStore = {
      list: vi.fn(),
      read: vi.fn(),
      getIndex: vi.fn().mockReturnValue({ documents: [{ id: 'd1' }] }),
    };

    index.setMemoryStore(memStore as any);
    index.setDocumentStore(docStore as any);

    // entryCount = 1, totalItems = 2 + 1 = 3 → partial
    expect(index.getStatus()).toBe('partial');
  });

  it('getStatus returns "ready" when entryCount matches total items in stores', async () => {
    // Add entries matching the total count
    await index.indexFact('k1', 'ns', 'v1');
    await index.indexFact('k2', 'ns', 'v2');

    const memStore = { list: vi.fn(), size: 2, maxEntries: 1000 };
    const docStore = {
      list: vi.fn(),
      read: vi.fn(),
      getIndex: vi.fn().mockReturnValue({ documents: [] }),
    };

    index.setMemoryStore(memStore as any);
    index.setDocumentStore(docStore as any);

    // entryCount = 2, totalItems = 2 + 0 = 2 → ready
    expect(index.getStatus()).toBe('ready');
  });

  it('getStatus returns "ready" when stores are not attached and there are entries', async () => {
    await index.indexFact('k', 'ns', 'v');
    // No memoryStore or documentStore attached → totalItems = 0
    // entryCount = 1, totalItems = 0 → partial check (0 > 0 is false) → ready

    expect(index.getStatus()).toBe('ready');
  });
});

// ============================================================================
// modelId getter
// ============================================================================

describe('SemanticIndex - modelId getter', () => {
  it('modelId returns vectorStore.currentModelId', () => {
    const embedder = mockEmbedder();
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    expect(index.modelId).toBe('test-model');
  });
});

// ============================================================================
// Reindex
// ============================================================================

describe('SemanticIndex - reindex("all")', () => {
  let embedder: ReturnType<typeof mockEmbedder>;
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let logger: Logger;
  let index: InstanceType<typeof SemanticIndex>;
  let memStore: ReturnType<typeof mockMemoryStoreForReindex>;
  let docStore: ReturnType<typeof mockDocumentStoreForReindex>;

  beforeEach(() => {
    embedder = mockEmbedder();
    vectorStore = mockVectorStore();
    logger = mockLogger();
    index = new SemanticIndex(embedder as any, vectorStore as any, logger);
    memStore = mockMemoryStoreForReindex();
    docStore = mockDocumentStoreForReindex();
    index.setMemoryStore(memStore as any);
    index.setDocumentStore(docStore as any);
  });

  it('reindex("all") calls vectorStore.rebuild (not upsert)', async () => {
    await index.reindex('all');

    expect(vectorStore.rebuild).toHaveBeenCalledOnce();
    expect(vectorStore.upsert).not.toHaveBeenCalled();
  });

  it('reindex("all") iterates all facts from memoryStore.list', async () => {
    const result = await index.reindex('all');

    expect(memStore.list).toHaveBeenCalledWith(
      expect.objectContaining({ includeValues: true }),
    );
    expect(result.facts).toBe(2);
  });

  it('reindex("all") iterates all documents from documentStore.list', async () => {
    const result = await index.reindex('all');

    expect(docStore.list).toHaveBeenCalledOnce();
    expect(result.documents).toBe(1);
  });

  it('reindex("all") returns correct total count', async () => {
    const result = await index.reindex('all');

    expect(result.total).toBe(result.facts + result.documents);
    expect(result.total).toBe(3);
  });

  it('reindex("all") returns skipped=0 when all embeddings succeed', async () => {
    const result = await index.reindex('all');

    expect(result.skipped).toBe(0);
  });

  it('reindex("all") returns a non-negative duration_ms', async () => {
    const result = await index.reindex('all');

    expect(result.duration_ms).toBeGreaterThanOrEqual(0);
  });

  it('reindex("all") calls vectorStore.save after rebuilding', async () => {
    await index.reindex('all');

    expect(vectorStore.save).toHaveBeenCalledOnce();
  });

  it('reindex("all") calls embedder.embedPassage for each fact and document', async () => {
    // 2 facts + 1 document = 3 embeddings
    await index.reindex('all');

    expect(embedder.embedPassage).toHaveBeenCalledTimes(3);
  });

  it('reindex("all") embeds facts with "namespace: key — value" format', async () => {
    await index.reindex('all');

    const calls = embedder.embedPassage.mock.calls.map((c) => c[0] as string);
    expect(calls).toContain('ns1: k1 — val1');
    expect(calls).toContain('ns2: k2 — val2');
  });

  it('reindex("all") uses document body for inline content types', async () => {
    await index.reindex('all');

    // doc-001 is text/plain (inline), so body from read() should be used
    const calls = embedder.embedPassage.mock.calls.map((c) => c[0] as string);
    expect(calls).toContain('Document body text');
    expect(docStore.read).toHaveBeenCalledWith({ id: 'doc-001', include_body: true });
  });

  it('reindex("all") passes limit=maxEntries to memoryStore.list', async () => {
    await index.reindex('all');

    expect(memStore.list).toHaveBeenCalledWith(
      expect.objectContaining({ limit: memStore.maxEntries }),
    );
  });

  it('reindex("all") increments skipped when embedding a fact throws', async () => {
    embedder.embedPassage
      .mockRejectedValueOnce(new Error('embed fail for fact'))
      .mockResolvedValue([0.1, 0.2, 0.3]);

    const result = await index.reindex('all');

    expect(result.skipped).toBe(1);
    expect(result.facts).toBe(1); // only 1 of 2 facts succeeded
  });

  it('reindex("all") increments skipped when embedding a document throws', async () => {
    embedder.embedPassage
      .mockResolvedValueOnce([0.1, 0.2, 0.3]) // fact 1 ok
      .mockResolvedValueOnce([0.1, 0.2, 0.3]) // fact 2 ok
      .mockRejectedValueOnce(new Error('embed fail for doc')); // doc 1 fails

    const result = await index.reindex('all');

    expect(result.skipped).toBe(1);
    expect(result.documents).toBe(0);
  });

  it('reindex("all") logs completion info', async () => {
    await index.reindex('all');

    expect(logger.info).toHaveBeenCalledWith(
      expect.objectContaining({ factsIndexed: 2, docsIndexed: 1 }),
      'Reindex completed',
    );
  });

  it('reindex with no stores attached returns zeros', async () => {
    const freshIndex = new SemanticIndex(embedder as any, vectorStore as any, logger);
    // No memoryStore or documentStore attached

    const result = await freshIndex.reindex('all');

    expect(result.facts).toBe(0);
    expect(result.documents).toBe(0);
    expect(result.total).toBe(0);
    expect(result.skipped).toBe(0);
    expect(vectorStore.rebuild).toHaveBeenCalledWith([]);
  });
});

describe('SemanticIndex - reindex("facts")', () => {
  let embedder: ReturnType<typeof mockEmbedder>;
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let logger: Logger;
  let index: InstanceType<typeof SemanticIndex>;
  let memStore: ReturnType<typeof mockMemoryStoreForReindex>;
  let docStore: ReturnType<typeof mockDocumentStoreForReindex>;

  beforeEach(() => {
    embedder = mockEmbedder();
    vectorStore = mockVectorStore();
    logger = mockLogger();
    index = new SemanticIndex(embedder as any, vectorStore as any, logger);
    memStore = mockMemoryStoreForReindex();
    docStore = mockDocumentStoreForReindex();
    index.setMemoryStore(memStore as any);
    index.setDocumentStore(docStore as any);
  });

  it('reindex("facts") only indexes facts, not documents', async () => {
    const result = await index.reindex('facts');

    expect(result.facts).toBe(2);
    expect(result.documents).toBe(0);
    expect(docStore.list).not.toHaveBeenCalled();
  });

  it('reindex("facts") uses upsert instead of rebuild', async () => {
    await index.reindex('facts');

    expect(vectorStore.upsert).toHaveBeenCalled();
    expect(vectorStore.rebuild).not.toHaveBeenCalled();
  });

  it('reindex("facts") calls upsert once per fact', async () => {
    await index.reindex('facts');

    // 2 facts in mockMemoryStoreForReindex
    expect(vectorStore.upsert).toHaveBeenCalledTimes(2);
  });

  it('reindex("facts") calls vectorStore.save after upserting', async () => {
    await index.reindex('facts');

    expect(vectorStore.save).toHaveBeenCalledOnce();
  });

  it('reindex("facts") increments skipped on embedding error', async () => {
    embedder.embedPassage.mockRejectedValueOnce(new Error('fail'));

    const result = await index.reindex('facts');

    expect(result.skipped).toBe(1);
  });
});

describe('SemanticIndex - reindex("documents")', () => {
  let embedder: ReturnType<typeof mockEmbedder>;
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let logger: Logger;
  let index: InstanceType<typeof SemanticIndex>;
  let memStore: ReturnType<typeof mockMemoryStoreForReindex>;
  let docStore: ReturnType<typeof mockDocumentStoreForReindex>;

  beforeEach(() => {
    embedder = mockEmbedder();
    vectorStore = mockVectorStore();
    logger = mockLogger();
    index = new SemanticIndex(embedder as any, vectorStore as any, logger);
    memStore = mockMemoryStoreForReindex();
    docStore = mockDocumentStoreForReindex();
    index.setMemoryStore(memStore as any);
    index.setDocumentStore(docStore as any);
  });

  it('reindex("documents") only indexes documents, not facts', async () => {
    const result = await index.reindex('documents');

    expect(result.documents).toBe(1);
    expect(result.facts).toBe(0);
    expect(memStore.list).not.toHaveBeenCalled();
  });

  it('reindex("documents") uses upsert instead of rebuild', async () => {
    await index.reindex('documents');

    expect(vectorStore.upsert).toHaveBeenCalled();
    expect(vectorStore.rebuild).not.toHaveBeenCalled();
  });

  it('reindex("documents") calls upsert once per document', async () => {
    await index.reindex('documents');

    // 1 document in mockDocumentStoreForReindex
    expect(vectorStore.upsert).toHaveBeenCalledTimes(1);
  });

  it('reindex("documents") calls vectorStore.save after upserting', async () => {
    await index.reindex('documents');

    expect(vectorStore.save).toHaveBeenCalledOnce();
  });

  it('reindex("documents") increments skipped on embedding error', async () => {
    embedder.embedPassage.mockRejectedValueOnce(new Error('fail'));

    const result = await index.reindex('documents');

    expect(result.skipped).toBe(1);
    expect(result.documents).toBe(0);
  });

  it('reindex("documents") uses metadata for binary content type (no read call)', async () => {
    docStore.list.mockReturnValueOnce({
      documents: [
        {
          id: 'doc-bin',
          title: 'Binary Doc',
          content_type: 'image/png',
          tags: ['photo'],
          topic: undefined,
          description: 'A photo',
        },
      ],
      total: 1,
    });

    await index.reindex('documents');

    // Should NOT call read() for binary content type
    expect(docStore.read).not.toHaveBeenCalled();
    const call = embedder.embedPassage.mock.calls[0][0] as string;
    expect(call).toContain('Binary Doc');
  });

  it('reindex("documents") default source is "all" when called with no argument', async () => {
    // Default reindex() call should process both facts and documents
    const result = await index.reindex();

    expect(result.facts).toBe(2);
    expect(result.documents).toBe(1);
    expect(vectorStore.rebuild).toHaveBeenCalledOnce();
  });
});

// ============================================================================
// INDEX_STATUS_HINTS coverage
// ============================================================================

describe('SemanticIndex - status hints in search response', () => {
  it('search includes the correct hint for "empty" status', async () => {
    const embedder = mockEmbedder();
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    const result = await index.search('query');

    expect(result.status).toBe('empty');
    expect(result.hint).toBe(INDEX_STATUS_HINTS.empty);
  });

  it('search includes the correct hint for "model_unavailable" status', async () => {
    const embedder = mockEmbedder();
    Object.defineProperty(embedder, 'isUnavailable', { value: true, configurable: true });
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    const result = await index.search('query');

    expect(result.status).toBe('model_unavailable');
    expect(result.hint).toBe(INDEX_STATUS_HINTS.model_unavailable);
  });

  it('search hint is undefined for "ready" status', async () => {
    const embedder = mockEmbedder();
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    await index.indexFact('k', 'ns', 'v');
    vectorStore.query.mockReturnValueOnce([]);

    const result = await index.search('query');

    expect(result.status).toBe('ready');
    expect(result.hint).toBeUndefined();
  });
});

// ============================================================================
// setMemoryStore / setDocumentStore
// ============================================================================

describe('SemanticIndex - store injection', () => {
  it('setMemoryStore attaches store used during reindex', async () => {
    const embedder = mockEmbedder();
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    const memStore = mockMemoryStoreForReindex();
    index.setMemoryStore(memStore as any);

    const result = await index.reindex('facts');

    expect(memStore.list).toHaveBeenCalled();
    expect(result.facts).toBe(2);
  });

  it('setDocumentStore attaches store used during reindex', async () => {
    const embedder = mockEmbedder();
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    const docStore = mockDocumentStoreForReindex();
    index.setDocumentStore(docStore as any);

    const result = await index.reindex('documents');

    expect(docStore.list).toHaveBeenCalled();
    expect(result.documents).toBe(1);
  });

  it('setDocumentStore attaches store used in getStatus partial check', async () => {
    const embedder = mockEmbedder();
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    // Inject one entry so we're not "empty"
    await index.indexFact('k', 'ns', 'v');

    // Inject a docStore with 5 docs → totalItems = 5, entryCount = 1 → partial
    const docStore = {
      list: vi.fn(),
      read: vi.fn(),
      getIndex: vi.fn().mockReturnValue({
        documents: Array.from({ length: 5 }, (_, i) => ({ id: `doc-${i}` })),
      }),
    };
    index.setDocumentStore(docStore as any);

    expect(index.getStatus()).toBe('partial');
  });
});

// ============================================================================
// enqueueMissing — startup recovery
// ============================================================================

describe('SemanticIndex - enqueueMissing (startup recovery)', () => {
  let embedder: ReturnType<typeof mockEmbedder>;
  let vectorStore: ReturnType<typeof mockVectorStore> & {
    allEntries: ReturnType<typeof vi.fn>;
    getFailedEmbeddings: ReturnType<typeof vi.fn>;
  };
  let logger: Logger;
  let index: InstanceType<typeof SemanticIndex>;
  let mockQueue: { enqueue: ReturnType<typeof vi.fn>; dequeue: ReturnType<typeof vi.fn> };
  let mockMemStore: {
    read: ReturnType<typeof vi.fn>;
    list: ReturnType<typeof vi.fn>;
    size: number;
    maxEntries: number;
  };
  let mockDocStore: {
    getIndex: ReturnType<typeof vi.fn>;
    read: ReturnType<typeof vi.fn>;
    list: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    embedder = mockEmbedder();
    vectorStore = mockVectorStore() as any;
    // Add enqueueMissing-specific methods to the mock vectorStore
    (vectorStore as any).allEntries = vi.fn().mockReturnValue([]);
    (vectorStore as any).getFailedEmbeddings = vi.fn().mockReturnValue({});
    logger = mockLogger();

    mockQueue = {
      enqueue: vi.fn(),
      dequeue: vi.fn(),
    };

    mockMemStore = {
      read: vi.fn().mockReturnValue(null),
      list: vi.fn().mockReturnValue({ facts: [] }),
      size: 0,
      maxEntries: 100,
    };

    mockDocStore = {
      getIndex: vi.fn().mockReturnValue({ documents: [] }),
      read: vi.fn().mockReturnValue({ body: 'document body' }),
      list: vi.fn().mockReturnValue({ documents: [] }),
    };

    index = new SemanticIndex(embedder as any, vectorStore as any, logger);
    index.setEmbeddingQueue(mockQueue as any);
    index.setMemoryStore(mockMemStore as any);
    index.setDocumentStore(mockDocStore as any);
  });

  it('enqueueMissing re-enqueues items with pending failure status', () => {
    const factVectorId = makeFactVectorId('ns', 'key1');
    (vectorStore as any).getFailedEmbeddings.mockReturnValue({
      [factVectorId]: { retries: 1, status: 'pending', lastAttempt: Date.now() },
    });
    mockMemStore.read.mockImplementation((key: string, ns: string) => {
      if (key === 'key1' && ns === 'ns') return { value: 'test-value' };
      return null;
    });

    index.enqueueMissing();

    expect(mockQueue.enqueue).toHaveBeenCalledWith(
      expect.objectContaining({ id: factVectorId }),
    );
  });

  it('enqueueMissing skips items with failed status', () => {
    const factVectorId = makeFactVectorId('ns', 'key1');
    (vectorStore as any).getFailedEmbeddings.mockReturnValue({
      [factVectorId]: { retries: 3, status: 'failed', lastAttempt: Date.now() },
    });
    mockMemStore.read.mockReturnValue({ value: 'test-value' });

    index.enqueueMissing();

    expect(mockQueue.enqueue).not.toHaveBeenCalled();
  });

  it('enqueueMissing enqueues facts missing from vector index', () => {
    (vectorStore as any).allEntries.mockReturnValue([]);
    mockMemStore.list.mockReturnValue({
      facts: [{ key: 'missing-key', namespace: 'ns', value: 'missing-value' }],
    });

    index.enqueueMissing();

    expect(mockQueue.enqueue).toHaveBeenCalledWith(
      expect.objectContaining({
        id: makeFactVectorId('ns', 'missing-key'),
        source: 'fact',
      }),
    );
  });

  it('enqueueMissing enqueues documents missing from vector index', () => {
    (vectorStore as any).allEntries.mockReturnValue([]);
    mockDocStore.getIndex.mockReturnValue({
      documents: [
        {
          id: 'doc-missing',
          title: 'Missing Doc',
          content_type: 'text/plain',
          tags: ['tag1'],
          topic: 'testing',
        },
      ],
    });
    mockDocStore.read.mockReturnValue({ body: 'doc body text' });

    index.enqueueMissing();

    expect(mockQueue.enqueue).toHaveBeenCalledWith(
      expect.objectContaining({
        id: makeDocVectorId('doc-missing'),
        source: 'document',
      }),
    );
  });

  it('enqueueMissing returns count of enqueued items', () => {
    (vectorStore as any).allEntries.mockReturnValue([]);
    mockMemStore.list.mockReturnValue({
      facts: [
        { key: 'key1', namespace: 'ns', value: 'val1' },
        { key: 'key2', namespace: 'ns', value: 'val2' },
      ],
    });
    mockDocStore.getIndex.mockReturnValue({
      documents: [
        { id: 'doc-1', title: 'Doc 1', content_type: 'text/plain', tags: [] },
      ],
    });
    mockDocStore.read.mockReturnValue({ body: 'body' });

    const count = index.enqueueMissing();

    expect(count).toBe(3);
  });

  it('enqueueMissing returns 0 when everything is already indexed', () => {
    const factVectorId = makeFactVectorId('ns', 'key1');
    const docVectorId = makeDocVectorId('doc-1');

    // All entries already present in the vector store
    (vectorStore as any).allEntries.mockReturnValue([
      { id: factVectorId, vector: [], source: 'fact', metadata: {}, indexedAt: 0 },
      { id: docVectorId, vector: [], source: 'document', metadata: {}, indexedAt: 0 },
    ]);

    mockMemStore.list.mockReturnValue({
      facts: [{ key: 'key1', namespace: 'ns', value: 'val1' }],
    });
    mockDocStore.getIndex.mockReturnValue({
      documents: [{ id: 'doc-1', title: 'Doc 1', content_type: 'text/plain', tags: [] }],
    });

    const count = index.enqueueMissing();

    expect(count).toBe(0);
    expect(mockQueue.enqueue).not.toHaveBeenCalled();
  });

  it('enqueueMissing returns 0 when no embeddingQueue is set', () => {
    const indexWithoutQueue = new SemanticIndex(embedder as any, vectorStore as any, logger);
    indexWithoutQueue.setMemoryStore(mockMemStore as any);
    indexWithoutQueue.setDocumentStore(mockDocStore as any);

    mockMemStore.list.mockReturnValue({
      facts: [{ key: 'key1', namespace: 'ns', value: 'val1' }],
    });

    const count = indexWithoutQueue.enqueueMissing();

    expect(count).toBe(0);
    expect(mockQueue.enqueue).not.toHaveBeenCalled();
  });
});

// ============================================================================
// Queue Integration
// ============================================================================

describe('SemanticIndex - queue integration', () => {
  let embedder: ReturnType<typeof mockEmbedder>;
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let logger: Logger;
  let index: InstanceType<typeof SemanticIndex>;
  let mockQueue: { enqueue: ReturnType<typeof vi.fn>; dequeue: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    embedder = mockEmbedder();
    vectorStore = mockVectorStore();
    logger = mockLogger();

    mockQueue = {
      enqueue: vi.fn(),
      dequeue: vi.fn(),
    };

    index = new SemanticIndex(embedder as any, vectorStore as any, logger);
    index.setEmbeddingQueue(mockQueue as any);
  });

  it('indexDocument delegates to enqueueDocument when queue is set', async () => {
    await index.indexDocument(
      'doc-queued',
      { title: 'Queued Doc', tags: ['t1'], contentType: 'text/plain' },
      'body text',
    );

    expect(mockQueue.enqueue).toHaveBeenCalledOnce();
    expect(mockQueue.enqueue).toHaveBeenCalledWith(
      expect.objectContaining({
        id: makeDocVectorId('doc-queued'),
        source: 'document',
      }),
    );
    expect(embedder.embedPassage).not.toHaveBeenCalled();
  });

  it('indexFact delegates to enqueueFact when queue is set', async () => {
    await index.indexFact('queued-key', 'queued-ns', 'queued-value');

    expect(mockQueue.enqueue).toHaveBeenCalledOnce();
    expect(mockQueue.enqueue).toHaveBeenCalledWith(
      expect.objectContaining({
        id: makeFactVectorId('queued-ns', 'queued-key'),
        source: 'fact',
      }),
    );
    expect(embedder.embedPassage).not.toHaveBeenCalled();
  });

  it('removeDocument calls dequeue before vectorStore.remove', () => {
    index.removeDocument('doc-to-remove');

    expect(mockQueue.dequeue).toHaveBeenCalledWith(makeDocVectorId('doc-to-remove'));
    expect(vectorStore.remove).toHaveBeenCalledWith(makeDocVectorId('doc-to-remove'));
  });

  it('removeFact calls dequeue before vectorStore.remove', () => {
    index.removeFact('key-to-remove', 'ns-to-remove');

    expect(mockQueue.dequeue).toHaveBeenCalledWith(makeFactVectorId('ns-to-remove', 'key-to-remove'));
    expect(vectorStore.remove).toHaveBeenCalledWith(makeFactVectorId('ns-to-remove', 'key-to-remove'));
  });
});

// ============================================================================
// Hint text validation
// ============================================================================

describe('SemanticIndex - model_unavailable hint text', () => {
  it('model_unavailable hint mentions semantic_reindex and memory_manage', () => {
    const hint = INDEX_STATUS_HINTS.model_unavailable;
    expect(hint).toContain('semantic_reindex');
    expect(hint).toContain('memory_manage');
  });

  it('search returns hint with semantic_reindex and memory_manage when model is unavailable', async () => {
    const embedder = mockEmbedder();
    Object.defineProperty(embedder, 'isUnavailable', { value: true, configurable: true });
    Object.defineProperty(embedder, 'isLoaded', { value: false, configurable: true });
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    const result = await index.search('query');

    expect(result.status).toBe('model_unavailable');
    expect(result.hint).toContain('semantic_reindex');
    expect(result.hint).toContain('memory_manage');
  });
});

// ============================================================================
// Reindex with repairAndLoad
// ============================================================================

describe('SemanticIndex - reindex calls repairAndLoad', () => {
  it('reindex calls embedder.repairAndLoad() before indexing', async () => {
    const embedder = mockEmbedder();
    (embedder as any).repairAndLoad = vi.fn().mockResolvedValue(undefined);
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    await index.reindex();

    expect((embedder as any).repairAndLoad).toHaveBeenCalledTimes(1);
  });

  it('reindex propagates repairAndLoad failure', async () => {
    const embedder = mockEmbedder();
    (embedder as any).repairAndLoad = vi.fn().mockRejectedValue(new Error('Model download failed'));
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    await expect(index.reindex()).rejects.toThrow('Model download failed');
  });

  it('reindex succeeds after repairAndLoad succeeds', async () => {
    const embedder = mockEmbedder();
    (embedder as any).repairAndLoad = vi.fn().mockResolvedValue(undefined);
    const vectorStore = mockVectorStore();
    const logger = mockLogger();
    const index = new SemanticIndex(embedder as any, vectorStore as any, logger);

    const result = await index.reindex();

    expect(result.facts).toBe(0);
    expect(result.documents).toBe(0);
    expect(result.total).toBe(0);
  });
});
