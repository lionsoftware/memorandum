import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Logger } from 'pino';
import type { VectorEntry } from '../src/semantic-types.js';
import { makeFactVectorId, makeDocVectorId } from '../src/semantic-types.js';
import { SemanticIndex } from '../src/semantic-index.js';

function mockLogger(): Logger {
  return {
    info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(),
    child: vi.fn().mockReturnThis(), fatal: vi.fn(), trace: vi.fn(),
    silent: vi.fn(), level: 'debug',
  } as unknown as Logger;
}

function mockEmbedder() {
  return {
    embedPassage: vi.fn().mockResolvedValue([1, 0, 0]),
    embedQuery: vi.fn(),
    get isUnavailable() { return false; },
    get isLoading() { return false; },
  };
}

function mockVectorStore(initialEntries: VectorEntry[] = []) {
  const entries: VectorEntry[] = [...initialEntries];
  return {
    allEntries: vi.fn<[], ReadonlyArray<VectorEntry>>().mockImplementation(() => entries),
    remove: vi.fn<[string], boolean>().mockReturnValue(true),
    save: vi.fn<[], Promise<void>>().mockResolvedValue(undefined),
    upsert: vi.fn<[VectorEntry], void>(),
    get entryCount() { return entries.length; },
    get currentModelId() { return 'test'; },
    get indexModelId() { return 'test'; },
    _pushEntry(e: VectorEntry) { entries.push(e); },
  };
}

function makeFactEntry(namespace: string, key: string): VectorEntry {
  return {
    id: makeFactVectorId(namespace, key), vector: [1, 0, 0], source: 'fact',
    metadata: { key, namespace, preview: `${namespace}:${key}` }, indexedAt: Date.now(),
  };
}

function makeDocEntry(documentId: string): VectorEntry {
  return {
    id: makeDocVectorId(documentId), vector: [1, 0, 0], source: 'document',
    metadata: { documentId, title: `Document ${documentId}`, tags: [], contentType: 'text/plain' },
    indexedAt: Date.now(),
  };
}

describe('SemanticIndex.pruneOrphans - fact entries', () => {
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let index: SemanticIndex;

  beforeEach(() => {
    vectorStore = mockVectorStore();
    index = new SemanticIndex(mockEmbedder() as any, vectorStore as any, mockLogger());
  });

  it('removes orphaned fact entry', async () => {
    const entry = makeFactEntry('project', 'roadmap');
    vectorStore._pushEntry(entry);
    index.setMemoryStore({ read: vi.fn().mockReturnValue(null) } as any);
    await index.pruneOrphans();
    expect(vectorStore.remove).toHaveBeenCalledWith(entry.id);
  });

  it('keeps valid fact entry', async () => {
    vectorStore._pushEntry(makeFactEntry('project', 'existing'));
    index.setMemoryStore({ read: vi.fn().mockReturnValue({ value: 'data' }) } as any);
    await index.pruneOrphans();
    expect(vectorStore.remove).not.toHaveBeenCalled();
  });

  it('removes malformed ID (no NUL separator)', async () => {
    const malformed: VectorEntry = {
      id: 'fact:no-separator', vector: [1, 0, 0], source: 'fact',
      metadata: { key: 'x', namespace: 'y', preview: '' }, indexedAt: Date.now(),
    };
    vectorStore._pushEntry(malformed);
    index.setMemoryStore({ read: vi.fn().mockReturnValue({ value: 'some' }) } as any);
    await index.pruneOrphans();
    expect(vectorStore.remove).toHaveBeenCalledWith('fact:no-separator');
  });
});

describe('SemanticIndex.pruneOrphans - document entries', () => {
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let index: SemanticIndex;

  beforeEach(() => {
    vectorStore = mockVectorStore();
    index = new SemanticIndex(mockEmbedder() as any, vectorStore as any, mockLogger());
  });

  it('removes orphaned document entry', async () => {
    vectorStore._pushEntry(makeDocEntry('doc-001'));
    index.setDocumentStore({ getIndex: vi.fn().mockReturnValue({ documents: [] }) } as any);
    await index.pruneOrphans();
    expect(vectorStore.remove).toHaveBeenCalled();
  });

  it('keeps valid document entry', async () => {
    vectorStore._pushEntry(makeDocEntry('doc-001'));
    index.setDocumentStore({ getIndex: vi.fn().mockReturnValue({ documents: [{ id: 'doc-001' }] }) } as any);
    await index.pruneOrphans();
    expect(vectorStore.remove).not.toHaveBeenCalled();
  });
});

describe('SemanticIndex.pruneOrphans - save behaviour', () => {
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let logger: Logger;
  let index: SemanticIndex;

  beforeEach(() => {
    vectorStore = mockVectorStore();
    logger = mockLogger();
    index = new SemanticIndex(mockEmbedder() as any, vectorStore as any, logger);
  });

  it('calls save after removing orphans', async () => {
    vectorStore._pushEntry(makeFactEntry('ns', 'orphan'));
    index.setMemoryStore({ read: vi.fn().mockReturnValue(null) } as any);
    await index.pruneOrphans();
    expect(vectorStore.save).toHaveBeenCalledOnce();
  });

  it('does not call save when no orphans found', async () => {
    vectorStore._pushEntry(makeFactEntry('ns', 'live'));
    index.setMemoryStore({ read: vi.fn().mockReturnValue({ value: 'ok' }) } as any);
    await index.pruneOrphans();
    expect(vectorStore.save).not.toHaveBeenCalled();
  });

  it('logs pruned count', async () => {
    vectorStore._pushEntry(makeFactEntry('ns', 'k1'));
    vectorStore._pushEntry(makeFactEntry('ns', 'k2'));
    index.setMemoryStore({ read: vi.fn().mockReturnValue(null) } as any);
    await index.pruneOrphans();
    expect(logger.info).toHaveBeenCalledWith(expect.objectContaining({ pruned: 2 }), 'Pruned orphaned vector entries');
  });
});

describe('SemanticIndex.removeFact - fire-and-forget save', () => {
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let index: SemanticIndex;

  beforeEach(() => {
    vectorStore = mockVectorStore();
    index = new SemanticIndex(mockEmbedder() as any, vectorStore as any, mockLogger());
  });

  it('calls save after removing existing fact', async () => {
    vectorStore.remove.mockReturnValueOnce(true);
    index.removeFact('key', 'ns');
    await new Promise(r => setTimeout(r, 0));
    expect(vectorStore.save).toHaveBeenCalledOnce();
  });

  it('does not call save when fact did not exist', async () => {
    vectorStore.remove.mockReturnValueOnce(false);
    index.removeFact('ghost', 'ns');
    await new Promise(r => setTimeout(r, 0));
    expect(vectorStore.save).not.toHaveBeenCalled();
  });
});

describe('SemanticIndex.removeDocument - fire-and-forget save', () => {
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let index: SemanticIndex;

  beforeEach(() => {
    vectorStore = mockVectorStore();
    index = new SemanticIndex(mockEmbedder() as any, vectorStore as any, mockLogger());
  });

  it('calls save after removing existing document', async () => {
    vectorStore.remove.mockReturnValueOnce(true);
    index.removeDocument('doc-001');
    await new Promise(r => setTimeout(r, 0));
    expect(vectorStore.save).toHaveBeenCalledOnce();
  });

  it('does not call save when document did not exist', async () => {
    vectorStore.remove.mockReturnValueOnce(false);
    index.removeDocument('doc-ghost');
    await new Promise(r => setTimeout(r, 0));
    expect(vectorStore.save).not.toHaveBeenCalled();
  });
});
