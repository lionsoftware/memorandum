/**
 * Unit tests for semantic-tools MCP tool handlers.
 *
 * Tests the registration and behavior of semantic_search and semantic_reindex tools.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { Logger } from 'pino';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { MemorandumError } from '../src/errors.js';
import { registerSemanticTools } from '../src/semantic-tools.js';

// ---------------------------------------------------------------------------
// Shared test infrastructure
// ---------------------------------------------------------------------------

type ToolHandler = (input: Record<string, unknown>) => Promise<unknown>;

function makeMockLogger(): Logger {
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

function makeMockSemanticIndex() {
  return {
    search: vi.fn().mockResolvedValue({
      status: 'ready',
      hint: undefined,
      results: [
        {
          source: 'document',
          id: 'doc-001',
          score: 0.85,
          metadata: {
            documentId: 'doc-001',
            title: 'Test',
            tags: [],
            contentType: 'text/plain',
          },
        },
      ],
      total: 1,
    }),
    reindex: vi.fn().mockResolvedValue({
      facts: 5,
      documents: 3,
      total: 8,
      skipped: 0,
      duration_ms: 1234,
    }),
    modelId: 'test-model',
  };
}

/**
 * Captures tool handlers registered via server.registerTool() and returns
 * them in a Map keyed by tool name.
 */
function captureHandlers(
  semanticIndex: ReturnType<typeof makeMockSemanticIndex>,
  logger: Logger,
): {
  handlers: Map<string, ToolHandler>;
  registeredConfigs: Map<string, Record<string, unknown>>;
  mockServer: McpServer;
} {
  const handlers = new Map<string, ToolHandler>();
  const registeredConfigs = new Map<string, Record<string, unknown>>();

  const mockServer = {
    registerTool: vi.fn(
      (
        name: string,
        config: Record<string, unknown>,
        handler: ToolHandler,
      ) => {
        handlers.set(name, handler);
        registeredConfigs.set(name, config);
      },
    ),
  } as unknown as McpServer;

  registerSemanticTools(mockServer, semanticIndex as any, logger);

  return { handlers, registeredConfigs, mockServer };
}

/**
 * Parse the JSON text from the first content item of an MCP response.
 */
function parseResponse(response: unknown): Record<string, unknown> {
  const resp = response as { content: Array<{ type: string; text: string }>; isError?: boolean };
  return JSON.parse(resp.content[0].text) as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Tool registration
// ---------------------------------------------------------------------------

describe('registerSemanticTools — tool registration', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('registers exactly 2 tools: semantic_search and semantic_reindex', () => {
    const index = makeMockSemanticIndex();
    const registeredNames: string[] = [];
    const mockServer = {
      registerTool: vi.fn((name: string) => {
        registeredNames.push(name);
      }),
    } as unknown as McpServer;

    registerSemanticTools(mockServer, index as any, makeMockLogger());

    expect(registeredNames).toHaveLength(2);
    expect(registeredNames).toContain('semantic_search');
    expect(registeredNames).toContain('semantic_reindex');
  });

  it('semantic_search tool has a title and description', () => {
    const index = makeMockSemanticIndex();
    const { registeredConfigs } = captureHandlers(index, makeMockLogger());

    const config = registeredConfigs.get('semantic_search');
    expect(config).toBeDefined();
    expect(typeof config!.title).toBe('string');
    expect((config!.title as string).length).toBeGreaterThan(0);
    expect(typeof config!.description).toBe('string');
    expect((config!.description as string).length).toBeGreaterThan(0);
  });

  it('semantic_reindex tool has a title and description', () => {
    const index = makeMockSemanticIndex();
    const { registeredConfigs } = captureHandlers(index, makeMockLogger());

    const config = registeredConfigs.get('semantic_reindex');
    expect(config).toBeDefined();
    expect(typeof config!.title).toBe('string');
    expect((config!.title as string).length).toBeGreaterThan(0);
    expect(typeof config!.description).toBe('string');
    expect((config!.description as string).length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// semantic_search handler
// ---------------------------------------------------------------------------

describe('semantic_search handler', () => {
  let index: ReturnType<typeof makeMockSemanticIndex>;
  let handlers: Map<string, ToolHandler>;

  beforeEach(() => {
    vi.resetAllMocks();
    index = makeMockSemanticIndex();
    ({ handlers } = captureHandlers(index, makeMockLogger()));
  });

  it('registers the semantic_search tool', () => {
    expect(handlers.has('semantic_search')).toBe(true);
  });

  it('calls semanticIndex.search with the correct query', async () => {
    await handlers.get('semantic_search')!({ query: 'test query' });

    expect(index.search).toHaveBeenCalledOnce();
    expect((index.search as ReturnType<typeof vi.fn>).mock.calls[0][0]).toBe('test query');
  });

  it('returns proper output shape {status, hint, results, total, query}', async () => {
    const result = await handlers.get('semantic_search')!({ query: 'my search' });
    const resp = result as { structuredContent: Record<string, unknown> };

    expect(resp.structuredContent).toHaveProperty('status');
    expect(resp.structuredContent).toHaveProperty('results');
    expect(resp.structuredContent).toHaveProperty('total');
    expect(resp.structuredContent).toHaveProperty('query');
    expect(resp.structuredContent.query).toBe('my search');
  });

  it('output includes results from semanticIndex.search', async () => {
    const result = await handlers.get('semantic_search')!({ query: 'test' });
    const output = parseResponse(result);

    expect(output.status).toBe('ready');
    expect(Array.isArray(output.results)).toBe(true);
    expect((output.results as unknown[]).length).toBe(1);
    expect(output.total).toBe(1);
  });

  it('passes source filter to semanticIndex.search options', async () => {
    await handlers.get('semantic_search')!({ query: 'test', source: 'documents' });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ source: 'documents' });
  });

  it('passes source=facts filter to semanticIndex.search options', async () => {
    await handlers.get('semantic_search')!({ query: 'test', source: 'facts' });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ source: 'facts' });
  });

  it('passes source=all (default) when not specified', async () => {
    await handlers.get('semantic_search')!({ query: 'test' });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ source: 'all' });
  });

  it('passes namespace metadata filter to semanticIndex.search options', async () => {
    await handlers.get('semantic_search')!({ query: 'test', namespace: 'servers' });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ namespace: 'servers' });
  });

  it('passes tag metadata filter to semanticIndex.search options', async () => {
    await handlers.get('semantic_search')!({ query: 'test', tag: 'infrastructure' });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ tag: 'infrastructure' });
  });

  it('passes topic metadata filter to semanticIndex.search options', async () => {
    await handlers.get('semantic_search')!({ query: 'test', topic: 'networking' });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ topic: 'networking' });
  });

  it('passes content_type metadata filter to semanticIndex.search options', async () => {
    await handlers.get('semantic_search')!({ query: 'test', content_type: 'text/markdown' });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ content_type: 'text/markdown' });
  });

  it('passes limit to semanticIndex.search options', async () => {
    await handlers.get('semantic_search')!({ query: 'test', limit: 25 });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ limit: 25 });
  });

  it('passes threshold to semanticIndex.search options', async () => {
    await handlers.get('semantic_search')!({ query: 'test', threshold: 0.7 });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ threshold: 0.7 });
  });

  it('returns empty results when status is "empty"', async () => {
    (index.search as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      status: 'empty',
      hint: 'Index is empty. Run semantic_reindex to populate it.',
      results: [],
      total: 0,
    });

    const result = await handlers.get('semantic_search')!({ query: 'test' });
    const output = parseResponse(result);

    expect(output.status).toBe('empty');
    expect(Array.isArray(output.results)).toBe(true);
    expect((output.results as unknown[]).length).toBe(0);
    expect(output.total).toBe(0);
  });

  it('returns hint when status is "empty"', async () => {
    (index.search as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      status: 'empty',
      hint: 'Index is empty. Run semantic_reindex to populate it.',
      results: [],
      total: 0,
    });

    const result = await handlers.get('semantic_search')!({ query: 'test' });
    const output = parseResponse(result);

    expect(typeof output.hint).toBe('string');
    expect((output.hint as string).length).toBeGreaterThan(0);
  });

  it('returns hint=undefined when status is "ready" and no hint provided', async () => {
    const result = await handlers.get('semantic_search')!({ query: 'test' });
    const resp = result as { structuredContent: Record<string, unknown> };

    expect(resp.structuredContent.status).toBe('ready');
    expect(resp.structuredContent.hint).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// semantic_reindex handler
// ---------------------------------------------------------------------------

describe('semantic_reindex handler', () => {
  let index: ReturnType<typeof makeMockSemanticIndex>;
  let handlers: Map<string, ToolHandler>;

  beforeEach(() => {
    vi.resetAllMocks();
    index = makeMockSemanticIndex();
    ({ handlers } = captureHandlers(index, makeMockLogger()));
  });

  it('registers the semantic_reindex tool', () => {
    expect(handlers.has('semantic_reindex')).toBe(true);
  });

  it('calls semanticIndex.reindex with the source', async () => {
    await handlers.get('semantic_reindex')!({ source: 'all' });

    expect(index.reindex).toHaveBeenCalledOnce();
    expect((index.reindex as ReturnType<typeof vi.fn>).mock.calls[0][0]).toBe('all');
  });

  it('uses default source "all" when not specified', async () => {
    await handlers.get('semantic_reindex')!({});

    expect(index.reindex).toHaveBeenCalledOnce();
    expect((index.reindex as ReturnType<typeof vi.fn>).mock.calls[0][0]).toBe('all');
  });

  it('passes source=documents to semanticIndex.reindex', async () => {
    await handlers.get('semantic_reindex')!({ source: 'documents' });

    expect((index.reindex as ReturnType<typeof vi.fn>).mock.calls[0][0]).toBe('documents');
  });

  it('passes source=facts to semanticIndex.reindex', async () => {
    await handlers.get('semantic_reindex')!({ source: 'facts' });

    expect((index.reindex as ReturnType<typeof vi.fn>).mock.calls[0][0]).toBe('facts');
  });

  it('returns proper output shape {success, indexed, skipped, duration_ms, model}', async () => {
    const result = await handlers.get('semantic_reindex')!({});
    const output = parseResponse(result);

    expect(output).toHaveProperty('success');
    expect(output).toHaveProperty('indexed');
    expect(output).toHaveProperty('skipped');
    expect(output).toHaveProperty('duration_ms');
    expect(output).toHaveProperty('model');
  });

  it('output.success is true on success', async () => {
    const result = await handlers.get('semantic_reindex')!({});
    const output = parseResponse(result);

    expect(output.success).toBe(true);
  });

  it('output.indexed contains facts, documents, total counts', async () => {
    const result = await handlers.get('semantic_reindex')!({});
    const output = parseResponse(result);

    const indexed = output.indexed as Record<string, unknown>;
    expect(indexed).toHaveProperty('facts');
    expect(indexed).toHaveProperty('documents');
    expect(indexed).toHaveProperty('total');
    expect(indexed.facts).toBe(5);
    expect(indexed.documents).toBe(3);
    expect(indexed.total).toBe(8);
  });

  it('output.skipped reflects the reindex result', async () => {
    (index.reindex as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      facts: 2,
      documents: 1,
      total: 3,
      skipped: 2,
      duration_ms: 500,
    });

    const result = await handlers.get('semantic_reindex')!({});
    const output = parseResponse(result);

    expect(output.skipped).toBe(2);
  });

  it('output.duration_ms reflects the reindex result', async () => {
    const result = await handlers.get('semantic_reindex')!({});
    const output = parseResponse(result);

    expect(output.duration_ms).toBe(1234);
  });

  it('output.model comes from semanticIndex.modelId', async () => {
    const result = await handlers.get('semantic_reindex')!({});
    const output = parseResponse(result);

    expect(output.model).toBe('test-model');
  });

  it('propagates reindex error as MCP error response', async () => {
    (index.reindex as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new MemorandumError('internal_error', 'Reindex failed unexpectedly'),
    );

    const result = await handlers.get('semantic_reindex')!({}) as { isError: boolean };

    expect(result.isError).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Input validation — semantic_search
// ---------------------------------------------------------------------------

describe('semantic_search — input validation', () => {
  let index: ReturnType<typeof makeMockSemanticIndex>;
  let handlers: Map<string, ToolHandler>;

  beforeEach(() => {
    vi.resetAllMocks();
    index = makeMockSemanticIndex();
    ({ handlers } = captureHandlers(index, makeMockLogger()));
  });

  it('requires non-empty query — empty string returns error', async () => {
    const result = await handlers.get('semantic_search')!({ query: '' }) as { isError: boolean };
    expect(result.isError).toBe(true);
    expect(index.search).not.toHaveBeenCalled();
  });

  it('requires query to be present — missing query returns error', async () => {
    const result = await handlers.get('semantic_search')!({}) as { isError: boolean };
    expect(result.isError).toBe(true);
    expect(index.search).not.toHaveBeenCalled();
  });

  it('validates limit minimum (must be >= 1) — limit=0 returns error', async () => {
    const result = await handlers.get('semantic_search')!({ query: 'test', limit: 0 }) as { isError: boolean };
    expect(result.isError).toBe(true);
    expect(index.search).not.toHaveBeenCalled();
  });

  it('validates limit maximum (must be <= 100) — limit=101 returns error', async () => {
    const result = await handlers.get('semantic_search')!({ query: 'test', limit: 101 }) as { isError: boolean };
    expect(result.isError).toBe(true);
    expect(index.search).not.toHaveBeenCalled();
  });

  it('accepts limit at the boundary minimum (limit=1)', async () => {
    await handlers.get('semantic_search')!({ query: 'test', limit: 1 });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ limit: 1 });
  });

  it('accepts limit at the boundary maximum (limit=100)', async () => {
    await handlers.get('semantic_search')!({ query: 'test', limit: 100 });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ limit: 100 });
  });

  it('validates threshold minimum (must be >= 0) — threshold=-0.1 returns error', async () => {
    const result = await handlers.get('semantic_search')!({ query: 'test', threshold: -0.1 }) as { isError: boolean };
    expect(result.isError).toBe(true);
    expect(index.search).not.toHaveBeenCalled();
  });

  it('validates threshold maximum (must be <= 1) — threshold=1.1 returns error', async () => {
    const result = await handlers.get('semantic_search')!({ query: 'test', threshold: 1.1 }) as { isError: boolean };
    expect(result.isError).toBe(true);
    expect(index.search).not.toHaveBeenCalled();
  });

  it('accepts threshold at boundary minimum (threshold=0)', async () => {
    await handlers.get('semantic_search')!({ query: 'test', threshold: 0 });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ threshold: 0 });
  });

  it('accepts threshold at boundary maximum (threshold=1)', async () => {
    await handlers.get('semantic_search')!({ query: 'test', threshold: 1 });

    const callOptions = (index.search as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(callOptions).toMatchObject({ threshold: 1 });
  });

  it('rejects unknown extra fields (strict schema)', async () => {
    const result = await handlers.get('semantic_search')!({ query: 'test', unknownField: 'value' }) as { isError: boolean };
    expect(result.isError).toBe(true);
    expect(index.search).not.toHaveBeenCalled();
  });

  it('rejects invalid source enum value', async () => {
    const result = await handlers.get('semantic_search')!({ query: 'test', source: 'invalid-source' }) as { isError: boolean };
    expect(result.isError).toBe(true);
    expect(index.search).not.toHaveBeenCalled();
  });

  it('accepts all valid source enum values without throwing', async () => {
    for (const source of ['all', 'documents', 'facts']) {
      vi.clearAllMocks();

      await handlers.get('semantic_search')!({ query: 'test', source });
      expect(index.search).toHaveBeenCalledOnce();
    }
  });
});

// ---------------------------------------------------------------------------
// Input validation — semantic_reindex
// ---------------------------------------------------------------------------

describe('semantic_reindex — input validation', () => {
  let index: ReturnType<typeof makeMockSemanticIndex>;
  let handlers: Map<string, ToolHandler>;

  beforeEach(() => {
    vi.resetAllMocks();
    index = makeMockSemanticIndex();
    ({ handlers } = captureHandlers(index, makeMockLogger()));
  });

  it('rejects invalid source enum value', async () => {
    const result = await handlers.get('semantic_reindex')!({ source: 'invalid' }) as { isError: boolean };
    expect(result.isError).toBe(true);
    expect(index.reindex).not.toHaveBeenCalled();
  });

  it('rejects unknown extra fields (strict schema)', async () => {
    const result = await handlers.get('semantic_reindex')!({ unknownField: 'value' }) as { isError: boolean };
    expect(result.isError).toBe(true);
    expect(index.reindex).not.toHaveBeenCalled();
  });

  it('accepts empty input and defaults source to "all"', async () => {
    await handlers.get('semantic_reindex')!({});

    expect(index.reindex).toHaveBeenCalledOnce();
    expect((index.reindex as ReturnType<typeof vi.fn>).mock.calls[0][0]).toBe('all');
  });
});
