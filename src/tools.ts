import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { Logger } from 'pino';
import { toMcpErrorResponse } from './errors.js';
import { MemoryStore } from './store.js';
import {
  MemoryWriteDisplaySchema, MemoryWriteInputSchema,
  MemoryReadDisplaySchema, MemoryReadInputSchema,
  MemoryManageDisplaySchema, MemoryManageInputSchema,
} from './types.js';

export type SyncFn = () => Promise<{ flushed_embeddings: number; saved_stores: string[] }>;

/**
 * Registers memory_write, memory_read, and memory_manage tools on the MCP server.
 * @param server - The MCP server instance to register tools on.
 * @param store - The memory store used for reading, writing, and managing facts.
 * @param _logger - Logger instance (reserved for future use).
 * @param syncFn - Optional custom sync function; falls back to `store.save()` if not provided.
 */
export function registerMemoryTools(
  server: McpServer,
  store: MemoryStore,
  _logger: Logger,
  syncFn?: SyncFn,
): void {
  server.registerTool(
    'memory_write',
    {
      title: 'Write Memory',
      description: 'Write a fact to memory. The key and namespace should be meaningful.',
      inputSchema: MemoryWriteDisplaySchema,
    },
    async (rawInput) => {
      try {
        const input = MemoryWriteInputSchema.parse(rawInput);
        const result = store.write(input.key, input.value, input.namespace, input.ttl_seconds);
        const output = { success: true, key: input.key, namespace: input.namespace, created: result.created };
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
    'memory_read',
    {
      title: 'Read Memory',
      description: 'Read a fact by key and namespace. Returns content with metadata or null if not found.',
      inputSchema: MemoryReadDisplaySchema,
    },
    async (rawInput) => {
      try {
        const input = MemoryReadInputSchema.parse(rawInput);
        const result = store.read(input.key, input.namespace);
        const output: Record<string, unknown> = result ? { ...result } : { found: false };
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
    'memory_manage',
    {
      title: 'Manage Memory',
      description: 'Manage memory: delete, list, search, namespaces, export, import, sync.',
      inputSchema: MemoryManageDisplaySchema,
    },
    async (rawInput) => {
      try {
        const input = MemoryManageInputSchema.parse(rawInput);
        let output: Record<string, unknown> = {};

        switch (input.action) {
          case 'list':
            output = store.list({
              namespace: input.namespace, pattern: input.pattern,
              limit: input.limit, includeValues: input.include_values, includeStats: input.include_stats,
            });
            break;
          case 'namespaces':
            output = { namespaces: store.getNamespaces() };
            break;
          case 'search':
            output = store.search({ query: input.query, namespace: input.namespace, limit: input.limit });
            break;
          case 'delete': {
            const deleted = store.delete(input.key, input.namespace);
            output = { success: true, deleted };
            break;
          }
          case 'delete_namespace': {
            const deletedCount = store.deleteNamespace(input.namespace);
            output = { success: true, deleted_count: deletedCount };
            break;
          }
          case 'export':
            output = store.export();
            break;
          case 'import':
            output = store.import(input.data, input.merge);
            break;
          case 'sync':
            if (syncFn) {
              output = await syncFn();
            } else {
              const written = await store.save();
              output = { flushed_embeddings: 0, saved_stores: written ? ['facts'] : [] };
            }
            break;
        }

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
