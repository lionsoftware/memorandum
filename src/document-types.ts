import { z } from 'zod';

// ============================================================================
// Constants
// ============================================================================

export const RESERVED_FIELDS = [
  'id', 'title', 'content_type', 'topic', 'description',
  'tags', 'created_at', 'updated_at', 'blob_path', 'blob_size', 'blob_sha256',
] as const;

export const DEFAULT_MAX_DOCUMENT_SIZE = 16 * 1024 * 1024;
export const MAX_TAGS = 50;
export const DEFAULT_LIST_LIMIT = 50;
export const INLINE_CONTENT_TYPES = ['text/', 'application/json', 'application/yaml'] as const;

export const EXTENSION_MIME_MAP: Record<string, string> = {
  '.txt': 'text/plain', '.md': 'text/markdown',
  '.json': 'application/json', '.yaml': 'application/yaml', '.yml': 'application/yaml',
  '.xml': 'text/xml', '.html': 'text/html', '.htm': 'text/html',
  '.css': 'text/css', '.js': 'text/javascript', '.ts': 'text/typescript',
  '.sh': 'text/x-shellscript', '.py': 'text/x-python',
  '.log': 'text/plain', '.csv': 'text/csv',
  '.conf': 'text/plain', '.ini': 'text/plain', '.toml': 'text/plain',
  '.env': 'text/plain', '.sql': 'text/x-sql',
  '.pdf': 'application/pdf',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.gif': 'image/gif', '.svg': 'image/svg+xml',
  '.zip': 'application/zip', '.tar': 'application/x-tar', '.gz': 'application/gzip',
};

export const DEFAULT_MIME_TYPE = 'application/octet-stream';

// ============================================================================
// Helper Functions
// ============================================================================

export function isInlineContentType(contentType: string): boolean {
  return (
    contentType.startsWith('text/') ||
    contentType === 'application/json' ||
    contentType === 'application/yaml'
  );
}

export function mimeTypeFromExtension(ext: string): string {
  return EXTENSION_MIME_MAP[ext.toLowerCase()] ?? DEFAULT_MIME_TYPE;
}

// ============================================================================
// Entity Types
// ============================================================================

export interface DocumentMetadata {
  id: string;
  title: string;
  content_type: string;
  topic?: string;
  description?: string;
  tags: string[];
  created_at: string;
  updated_at: string;
  blob_path?: string;
  blob_size?: number;
  blob_sha256?: string;
  [key: string]: unknown;
}

export interface IndexFile {
  version: number;
  next_id: number;
  documents: DocumentMetadata[];
}

export interface FileValidationResult {
  resolvedPath: string;
  size: number;
  contentType: string;
  basename: string;
  nameWithoutExt: string;
}

// ============================================================================
// Display Schemas
// ============================================================================

export const DocumentWriteDisplaySchema = z.object({
  id: z.string().optional().describe('Document ID for update (omit for create)'),
  title: z.string().optional().describe('Document title (required for create, auto-derived from filename if omitted with file_path)'),
  content_type: z.string().optional().describe('MIME type (default: "text/plain", auto-detected from file extension when using file_path)'),
  topic: z.string().optional().describe('Topic/category'),
  description: z.string().optional().describe('Description'),
  tags: z.array(z.string()).optional().describe('Tags (normalized: lowercase, trimmed)'),
  body: z.string().optional().describe('Content body (text or base64 for binary). Mutually exclusive with file_path.'),
  file_path: z.string().optional().describe('Path to a local file. Mutually exclusive with body.'),
  metadata: z.record(z.string(), z.unknown()).optional().describe('Custom metadata fields'),
});

export const DocumentReadDisplaySchema = z.object({
  id: z.string().describe('Document ID (e.g., "doc-001")'),
  include_body: z.boolean().optional().describe('Include body content (default: true)'),
});

export const DocumentListDisplaySchema = z.object({
  tag: z.string().optional().describe('Filter by tag (exact match)'),
  topic: z.string().optional().describe('Filter by topic (exact match)'),
  content_type: z.string().optional().describe('Filter by content_type (exact match)'),
  search: z.string().optional().describe('Substring search in title+description (case-insensitive)'),
  metadata: z.record(z.string(), z.unknown()).optional().describe('Filter by custom metadata (shallow exact match)'),
  limit: z.number().optional().describe('Max results (default: 50)'),
});

export const DocumentDeleteDisplaySchema = z.object({
  id: z.string().describe('Document ID to delete'),
});

// ============================================================================
// Input Schemas
// ============================================================================

export const DocumentWriteCreateInputSchema = z.strictObject({
  _mode: z.literal('create'),
  title: z.string().min(1).max(500).optional(),
  content_type: z.string().min(1).default('text/plain'),
  topic: z.string().min(1).max(100).optional(),
  description: z.string().min(1).max(2000).optional(),
  tags: z.array(z.string().min(1).max(100)).max(MAX_TAGS).optional(),
  body: z.string().optional(),
  file_path: z.string().min(1).optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
}).refine(
  (data) => !(data.body !== undefined && data.file_path !== undefined),
  { message: "Cannot specify both 'file_path' and 'body'", path: ['file_path'] },
).refine(
  (data) => data.title !== undefined || data.file_path !== undefined,
  { message: "Title is required for create (or provide file_path for auto-detection)", path: ['title'] },
);

export const DocumentWriteUpdateInputSchema = z.strictObject({
  _mode: z.literal('update'),
  id: z.string().min(1),
  title: z.string().min(1).max(500).optional(),
  content_type: z.string().min(1).optional(),
  topic: z.string().min(1).max(100).optional(),
  description: z.string().min(1).max(2000).optional(),
  tags: z.array(z.string().min(1).max(100)).max(MAX_TAGS).optional(),
  body: z.string().optional(),
  file_path: z.string().min(1).optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
}).refine(
  (data) => !(data.body !== undefined && data.file_path !== undefined),
  { message: "Cannot specify both 'file_path' and 'body'", path: ['file_path'] },
);

export const DocumentWriteInputSchema = z.union([
  DocumentWriteCreateInputSchema,
  DocumentWriteUpdateInputSchema,
]);

export const DocumentReadInputSchema = z.strictObject({
  id: z.string().min(1),
  include_body: z.boolean().default(true),
});

export const DocumentListInputSchema = z.strictObject({
  tag: z.string().min(1).optional(),
  topic: z.string().min(1).optional(),
  content_type: z.string().min(1).optional(),
  search: z.string().min(1).optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
  limit: z.number().int().positive().default(DEFAULT_LIST_LIMIT),
});

export const DocumentDeleteInputSchema = z.strictObject({
  id: z.string().min(1),
});

// ============================================================================
// Inferred Types
// ============================================================================

export type DocumentWriteInput = z.infer<typeof DocumentWriteInputSchema>;
export type DocumentReadInput = z.infer<typeof DocumentReadInputSchema>;
export type DocumentListInput = z.infer<typeof DocumentListInputSchema>;
export type DocumentDeleteInput = z.infer<typeof DocumentDeleteInputSchema>;
