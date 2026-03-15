import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { Logger } from 'pino';
import { toMcpErrorResponse } from './errors.js';
import { DocumentStore } from './document-store.js';
import {
  DocumentWriteDisplaySchema, DocumentWriteInputSchema,
  DocumentReadDisplaySchema, DocumentReadInputSchema,
  DocumentListDisplaySchema, DocumentListInputSchema,
  DocumentDeleteDisplaySchema, DocumentDeleteInputSchema,
  DocumentRestoreDisplaySchema, DocumentRestoreInputSchema,
} from './document-types.js';

/**
 * Registers document_write, document_read, document_list, document_delete, and document_restore tools on the MCP server.
 * @param server - The MCP server instance to register tools on.
 * @param store - The document store used for CRUD operations on documents.
 * @param _logger - Logger instance (reserved for future use).
 */
export function registerDocumentTools(
  server: McpServer,
  store: DocumentStore,
  _logger: Logger,
): void {
  server.registerTool(
    'document_write',
    {
      title: 'Write Document',
      description: 'Create or update a document. Supports text (Markdown, JSON, YAML), binary (base64), or file import. Use id to update.',
      inputSchema: DocumentWriteDisplaySchema,
    },
    async (rawInput) => {
      try {
        const args = rawInput as Record<string, unknown>;
        args._mode = args.id ? 'update' : 'create';
        const input = DocumentWriteInputSchema.parse(args);
        const result = store.write(input);
        const output = { success: true, id: result.id, created: result.created };
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
    'document_read',
    {
      title: 'Read Document',
      description: 'Read a document by ID. Returns metadata and optionally body content.',
      inputSchema: DocumentReadDisplaySchema,
    },
    async (rawInput) => {
      try {
        const input = DocumentReadInputSchema.parse(rawInput);
        const result = store.read(input);
        return {
          content: [{ type: 'text' as const, text: JSON.stringify(result) }],
          structuredContent: result,
        };
      } catch (error) {
        return toMcpErrorResponse(error);
      }
    },
  );

  server.registerTool(
    'document_list',
    {
      title: 'List Documents',
      description: 'List and search documents. Filter by tag, topic, content_type, or search in title/description.',
      inputSchema: DocumentListDisplaySchema,
    },
    async (rawInput) => {
      try {
        const input = DocumentListInputSchema.parse(rawInput);
        const result = store.list(input);
        return {
          content: [{ type: 'text' as const, text: JSON.stringify(result) }],
          structuredContent: result,
        };
      } catch (error) {
        return toMcpErrorResponse(error);
      }
    },
  );

  server.registerTool(
    'document_delete',
    {
      title: 'Delete Document',
      description: 'Delete a document by ID. Removes file, blob, and index entry.',
      inputSchema: DocumentDeleteDisplaySchema,
    },
    async (rawInput) => {
      try {
        const input = DocumentDeleteInputSchema.parse(rawInput);
        const result = store.delete(input);
        const output = { success: true, deleted: result.deleted };
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
    'document_restore',
    {
      title: 'Restore Document',
      description: 'Restore a document from the store to a file on disk. Verifies SHA256 integrity for binary documents. Refuses to overwrite existing files unless force=true.',
      inputSchema: DocumentRestoreDisplaySchema,
    },
    async (rawInput) => {
      try {
        const input = DocumentRestoreInputSchema.parse(rawInput);
        const result = store.restore(input);
        const output: Record<string, unknown> = { ...result };
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
