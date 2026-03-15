import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { Logger } from 'pino';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { DocumentStore } from '../src/document-store.js';
import { MemorandumError } from '../src/errors.js';
import { registerDocumentTools } from '../src/document-tools.js';

// ---------------------------------------------------------------------------
// Shared test infrastructure
// ---------------------------------------------------------------------------

function makeMockStore(): DocumentStore {
  return {
    write: vi.fn(),
    read: vi.fn(),
    list: vi.fn(),
    delete: vi.fn(),
    restore: vi.fn(),
  } as unknown as DocumentStore;
}

function makeMockLogger(): Logger {
  return {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    child: vi.fn().mockReturnThis(),
    fatal: vi.fn(),
    trace: vi.fn(),
    silent: vi.fn(),
    level: 'debug',
  } as unknown as Logger;
}

/**
 * Captures tool handlers registered via server.registerTool() and returns
 * them in a Map keyed by tool name.
 */
function captureHandlers(
  store: DocumentStore,
  logger: Logger,
): Map<string, (input: Record<string, unknown>) => Promise<unknown>> {
  const handlers = new Map<string, (input: Record<string, unknown>) => Promise<unknown>>();
  const mockServer = {
    registerTool: vi.fn(
      (
        name: string,
        _config: unknown,
        handler: (input: Record<string, unknown>) => Promise<unknown>,
      ) => {
        handlers.set(name, handler);
      },
    ),
  } as unknown as McpServer;

  registerDocumentTools(mockServer, store, logger);
  return handlers;
}

// ---------------------------------------------------------------------------
// document_write
// ---------------------------------------------------------------------------

describe('document_write handler', () => {
  let mockStore: DocumentStore;
  let handlers: Map<string, (input: Record<string, unknown>) => Promise<unknown>>;

  beforeEach(() => {
    vi.resetAllMocks();
    mockStore = makeMockStore();
    handlers = captureHandlers(mockStore, makeMockLogger());
  });

  it('registers the document_write tool', () => {
    expect(handlers.has('document_write')).toBe(true);
  });

  it('valid create (title + body) returns { success: true, id, created: true }', async () => {
    (mockStore.write as ReturnType<typeof vi.fn>).mockReturnValue({
      id: 'doc-001',
      created: true,
    });

    const result = await handlers.get('document_write')!({
      title: 'My Document',
      body: 'Hello world',
    });

    expect(mockStore.write).toHaveBeenCalledOnce();
    // Handler injects _mode: 'create' when no id is present
    expect((mockStore.write as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      _mode: 'create',
      title: 'My Document',
    });

    expect(result).toEqual({
      content: [{ type: 'text', text: JSON.stringify({ success: true, id: 'doc-001', created: true }) }],
      structuredContent: { success: true, id: 'doc-001', created: true },
    });
  });

  it('valid update (id + title) returns { success: true, id, created: false }', async () => {
    (mockStore.write as ReturnType<typeof vi.fn>).mockReturnValue({
      id: 'doc-001',
      created: false,
    });

    const result = await handlers.get('document_write')!({
      id: 'doc-001',
      title: 'Updated Title',
    });

    expect(mockStore.write).toHaveBeenCalledOnce();
    // Handler injects _mode: 'update' when id is present
    expect((mockStore.write as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      _mode: 'update',
      id: 'doc-001',
      title: 'Updated Title',
    });

    expect(result).toEqual({
      content: [{ type: 'text', text: JSON.stringify({ success: true, id: 'doc-001', created: false }) }],
      structuredContent: { success: true, id: 'doc-001', created: false },
    });
  });

  it('file_path is passed through to store.write() on create', async () => {
    (mockStore.write as ReturnType<typeof vi.fn>).mockReturnValue({
      id: 'doc-002',
      created: true,
    });

    await handlers.get('document_write')!({
      file_path: '/tmp/test-file.txt',
    });

    expect(mockStore.write).toHaveBeenCalledOnce();
    expect((mockStore.write as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      _mode: 'create',
      file_path: '/tmp/test-file.txt',
    });
  });

  it('file_path is passed through to store.write() on update', async () => {
    (mockStore.write as ReturnType<typeof vi.fn>).mockReturnValue({
      id: 'doc-001',
      created: false,
    });

    await handlers.get('document_write')!({
      id: 'doc-001',
      file_path: '/tmp/updated-file.txt',
    });

    expect(mockStore.write).toHaveBeenCalledOnce();
    expect((mockStore.write as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      _mode: 'update',
      id: 'doc-001',
      file_path: '/tmp/updated-file.txt',
    });
  });

  it('missing title on create causes store.write to throw and returns MCP error response', async () => {
    (mockStore.write as ReturnType<typeof vi.fn>).mockImplementation(() => {
      throw new MemorandumError('validation_error', 'title is required for document creation');
    });

    const result = await handlers.get('document_write')!({
      body: 'some body without title',
    }) as { isError: boolean };

    expect(result.isError).toBe(true);
  });

  it('propagates store error as MCP error response', async () => {
    (mockStore.write as ReturnType<typeof vi.fn>).mockImplementation(() => {
      throw new MemorandumError('document_too_large', 'Document body too large');
    });

    const result = await handlers.get('document_write')!({
      title: 'Doc',
      body: 'x'.repeat(100),
    }) as { isError: boolean };

    expect(result.isError).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// document_read
// ---------------------------------------------------------------------------

describe('document_read handler', () => {
  let mockStore: DocumentStore;
  let handlers: Map<string, (input: Record<string, unknown>) => Promise<unknown>>;

  beforeEach(() => {
    vi.resetAllMocks();
    mockStore = makeMockStore();
    handlers = captureHandlers(mockStore, makeMockLogger());
  });

  it('registers the document_read tool', () => {
    expect(handlers.has('document_read')).toBe(true);
  });

  it('valid read by id returns document data', async () => {
    const docData = {
      id: 'doc-001',
      title: 'My Document',
      content_type: 'text/plain',
      tags: [],
      created_at: '2025-01-01T00:00:00.000Z',
      updated_at: '2025-01-01T00:00:00.000Z',
      body: 'Hello world',
    };
    (mockStore.read as ReturnType<typeof vi.fn>).mockReturnValue(docData);

    const result = await handlers.get('document_read')!({ id: 'doc-001' });

    expect(mockStore.read).toHaveBeenCalledOnce();
    expect((mockStore.read as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      id: 'doc-001',
    });
    expect(result).toEqual({
      content: [{ type: 'text', text: JSON.stringify(docData) }],
      structuredContent: docData,
    });
  });

  it('read with include_body: false passes flag to store', async () => {
    const docData = {
      id: 'doc-001',
      title: 'My Document',
      content_type: 'text/plain',
      tags: [],
      created_at: '2025-01-01T00:00:00.000Z',
      updated_at: '2025-01-01T00:00:00.000Z',
    };
    (mockStore.read as ReturnType<typeof vi.fn>).mockReturnValue(docData);

    await handlers.get('document_read')!({ id: 'doc-001', include_body: false });

    expect(mockStore.read).toHaveBeenCalledOnce();
    expect((mockStore.read as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      id: 'doc-001',
      include_body: false,
    });
  });

  it('document not found returns MCP error response', async () => {
    (mockStore.read as ReturnType<typeof vi.fn>).mockImplementation(() => {
      throw new MemorandumError('document_not_found', "Document 'doc-999' not found");
    });

    const result = await handlers.get('document_read')!({ id: 'doc-999' }) as { isError: boolean };

    expect(result.isError).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// document_list
// ---------------------------------------------------------------------------

describe('document_list handler', () => {
  let mockStore: DocumentStore;
  let handlers: Map<string, (input: Record<string, unknown>) => Promise<unknown>>;

  beforeEach(() => {
    vi.resetAllMocks();
    mockStore = makeMockStore();
    handlers = captureHandlers(mockStore, makeMockLogger());
  });

  it('registers the document_list tool', () => {
    expect(handlers.has('document_list')).toBe(true);
  });

  it('valid list with no filters returns { documents: [...], total: N }', async () => {
    const listResult = {
      documents: [
        {
          id: 'doc-001',
          title: 'Doc 1',
          content_type: 'text/plain',
          tags: [],
          created_at: '2025-01-01T00:00:00.000Z',
          updated_at: '2025-01-01T00:00:00.000Z',
        },
        {
          id: 'doc-002',
          title: 'Doc 2',
          content_type: 'text/plain',
          tags: [],
          created_at: '2025-01-02T00:00:00.000Z',
          updated_at: '2025-01-02T00:00:00.000Z',
        },
      ],
      total: 2,
    };
    (mockStore.list as ReturnType<typeof vi.fn>).mockReturnValue(listResult);

    const result = await handlers.get('document_list')!({});

    expect(mockStore.list).toHaveBeenCalledOnce();
    expect(result).toEqual({
      content: [{ type: 'text', text: JSON.stringify(listResult) }],
      structuredContent: listResult,
    });
  });

  it('valid list with tag filter passes tag to store', async () => {
    const listResult = {
      documents: [
        {
          id: 'doc-001',
          title: 'Tagged Doc',
          content_type: 'text/plain',
          tags: ['infrastructure'],
          created_at: '2025-01-01T00:00:00.000Z',
          updated_at: '2025-01-01T00:00:00.000Z',
        },
      ],
      total: 1,
    };
    (mockStore.list as ReturnType<typeof vi.fn>).mockReturnValue(listResult);

    const result = await handlers.get('document_list')!({ tag: 'infrastructure' });

    expect(mockStore.list).toHaveBeenCalledOnce();
    expect((mockStore.list as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      tag: 'infrastructure',
    });
    expect(result).toEqual({
      content: [{ type: 'text', text: JSON.stringify(listResult) }],
      structuredContent: listResult,
    });
  });

  it('valid list with topic filter passes topic to store', async () => {
    (mockStore.list as ReturnType<typeof vi.fn>).mockReturnValue({ documents: [], total: 0 });

    await handlers.get('document_list')!({ topic: 'networking' });

    expect(mockStore.list).toHaveBeenCalledOnce();
    expect((mockStore.list as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      topic: 'networking',
    });
  });

  it('valid list with search filter passes search to store', async () => {
    (mockStore.list as ReturnType<typeof vi.fn>).mockReturnValue({ documents: [], total: 0 });

    await handlers.get('document_list')!({ search: 'runbook' });

    expect(mockStore.list).toHaveBeenCalledOnce();
    expect((mockStore.list as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      search: 'runbook',
    });
  });

  it('valid list with limit passes limit to store', async () => {
    (mockStore.list as ReturnType<typeof vi.fn>).mockReturnValue({ documents: [], total: 0 });

    await handlers.get('document_list')!({ limit: 10 });

    expect(mockStore.list).toHaveBeenCalledOnce();
    expect((mockStore.list as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      limit: 10,
    });
  });

  it('empty result set is returned as-is', async () => {
    const emptyResult = { documents: [], total: 0 };
    (mockStore.list as ReturnType<typeof vi.fn>).mockReturnValue(emptyResult);

    const result = await handlers.get('document_list')!({});

    expect(result).toEqual({
      content: [{ type: 'text', text: JSON.stringify(emptyResult) }],
      structuredContent: emptyResult,
    });
  });
});

// ---------------------------------------------------------------------------
// document_delete
// ---------------------------------------------------------------------------

describe('document_delete handler', () => {
  let mockStore: DocumentStore;
  let handlers: Map<string, (input: Record<string, unknown>) => Promise<unknown>>;

  beforeEach(() => {
    vi.resetAllMocks();
    mockStore = makeMockStore();
    handlers = captureHandlers(mockStore, makeMockLogger());
  });

  it('registers the document_delete tool', () => {
    expect(handlers.has('document_delete')).toBe(true);
  });

  it('valid delete of existing document returns { success: true, deleted: true }', async () => {
    (mockStore.delete as ReturnType<typeof vi.fn>).mockReturnValue({ deleted: true });

    const result = await handlers.get('document_delete')!({ id: 'doc-001' });

    expect(mockStore.delete).toHaveBeenCalledOnce();
    expect((mockStore.delete as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      id: 'doc-001',
    });
    expect(result).toEqual({
      content: [{ type: 'text', text: JSON.stringify({ success: true, deleted: true }) }],
      structuredContent: { success: true, deleted: true },
    });
  });

  it('delete non-existent document returns { success: true, deleted: false }', async () => {
    (mockStore.delete as ReturnType<typeof vi.fn>).mockReturnValue({ deleted: false });

    const result = await handlers.get('document_delete')!({ id: 'doc-999' });

    expect(mockStore.delete).toHaveBeenCalledOnce();
    expect(result).toEqual({
      content: [{ type: 'text', text: JSON.stringify({ success: true, deleted: false }) }],
      structuredContent: { success: true, deleted: false },
    });
  });

  it('propagates store error as MCP error response', async () => {
    (mockStore.delete as ReturnType<typeof vi.fn>).mockImplementation(() => {
      throw new MemorandumError('internal_error', 'Unexpected deletion failure');
    });

    const result = await handlers.get('document_delete')!({ id: 'doc-001' }) as { isError: boolean };

    expect(result.isError).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// registerDocumentTools — registration coverage
// ---------------------------------------------------------------------------

describe('registerDocumentTools — tool registration', () => {
  it('registers exactly 5 tools: document_write, document_read, document_list, document_delete, document_restore', () => {
    vi.resetAllMocks();
    const mockStore = makeMockStore();
    const registeredNames: string[] = [];
    const mockServer = {
      registerTool: vi.fn((name: string) => {
        registeredNames.push(name);
      }),
    } as unknown as McpServer;

    registerDocumentTools(mockServer, mockStore, makeMockLogger());

    expect(registeredNames).toHaveLength(5);
    expect(registeredNames).toContain('document_write');
    expect(registeredNames).toContain('document_read');
    expect(registeredNames).toContain('document_list');
    expect(registeredNames).toContain('document_delete');
    expect(registeredNames).toContain('document_restore');
  });
});
