import { z } from 'zod';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { Logger } from 'pino';
import { toMcpErrorResponse } from './errors.js';
import type { SemanticIndex } from './semantic-index.js';

const SemanticSearchDisplaySchema = z.object({
  query: z.string().optional().describe('Search query in natural language'),
  source: z.string().optional().describe('Filter by source: all (default), documents, facts'),
  limit: z.number().optional().describe('Maximum results (default: 10)'),
  threshold: z.number().optional().describe('Minimum relevance threshold 0.0-1.0 (default: 0.3)'),
  namespace: z.string().optional().describe('Filter by namespace (facts only)'),
  tag: z.string().optional().describe('Filter by tag (documents only)'),
  topic: z.string().optional().describe('Filter by topic (documents only)'),
  content_type: z.string().optional().describe('Filter by content type (documents only)'),
});

const SemanticReindexDisplaySchema = z.object({
  source: z.string().optional().describe('What to reindex: all (default), documents, facts'),
});

const SemanticSearchInputSchema = z.strictObject({
  query: z.string().min(1).max(1000),
  source: z.enum(['all', 'documents', 'facts']).default('all'),
  limit: z.number().int().min(1).max(100).default(10),
  threshold: z.number().min(0).max(1).default(0.3),
  namespace: z.string().optional(),
  tag: z.string().optional(),
  topic: z.string().optional(),
  content_type: z.string().optional(),
});

const SemanticReindexInputSchema = z.strictObject({
  source: z.enum(['all', 'documents', 'facts']).default('all'),
});

/**
 * Registers semantic_search and semantic_reindex tools on the MCP server.
 * @param server - The MCP server instance to register tools on.
 * @param semanticIndex - The semantic index used for embedding-based search and reindexing.
 * @param _logger - Logger instance (reserved for future use).
 */
export function registerSemanticTools(
  server: McpServer,
  semanticIndex: SemanticIndex,
  _logger: Logger,
): void {
  server.registerTool(
    'semantic_search',
    {
      title: 'Semantic Search',
      description: 'Search documents and facts by meaning using natural language queries. Returns ranked results with relevance scores.',
      inputSchema: SemanticSearchDisplaySchema,
    },
    async (rawInput) => {
      try {
        const input = SemanticSearchInputSchema.parse(rawInput);
        const result = await semanticIndex.search(input.query, {
          source: input.source, limit: input.limit, threshold: input.threshold,
          namespace: input.namespace, tag: input.tag, topic: input.topic, content_type: input.content_type,
        });
        const output = { status: result.status, hint: result.hint, results: result.results, total: result.total, query: input.query };
        return {
          content: [{ type: 'text' as const, text: JSON.stringify(output) }],
          structuredContent: output,
        };
      } catch (error) {
        return toMcpErrorResponse(error);
      }
    },
  );

  server.registerTool(
    'semantic_reindex',
    {
      title: 'Semantic Reindex',
      description: 'Rebuild the semantic search index for all documents and/or facts.',
      inputSchema: SemanticReindexDisplaySchema,
    },
    async (rawInput) => {
      try {
        const input = SemanticReindexInputSchema.parse(rawInput);
        const result = await semanticIndex.reindex(input.source);
        const output = {
          success: true,
          indexed: { facts: result.facts, documents: result.documents, total: result.total },
          skipped: result.skipped, duration_ms: result.duration_ms, model: semanticIndex.modelId,
        };
        return {
          content: [{ type: 'text' as const, text: JSON.stringify(output) }],
          structuredContent: output,
        };
      } catch (error) {
        return toMcpErrorResponse(error);
      }
    },
  );
}
