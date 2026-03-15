import { z } from 'zod';

// ============================================================================
// Constants
// ============================================================================

export const VECTOR_INDEX_VERSION = 2;
export const DEFAULT_SEARCH_LIMIT = 10;
export const DEFAULT_SEARCH_THRESHOLD = 0.3;
export const FACT_ID_PREFIX = 'fact:';
export const DOC_ID_PREFIX = 'doc:';

// ============================================================================
// Index Status
// ============================================================================

export const IndexStatusEnum = z.enum([
  'ready', 'empty', 'stale_model', 'partial', 'model_loading', 'model_unavailable',
]);

export type IndexStatus = z.infer<typeof IndexStatusEnum>;

export const INDEX_STATUS_HINTS: Record<IndexStatus, string | undefined> = {
  ready: undefined,
  empty: 'Index is empty. Call semantic_reindex to index existing data.',
  stale_model: 'Model has changed. Call semantic_reindex to re-index with the new model.',
  partial: 'Index is incomplete. Call semantic_reindex for full indexing.',
  model_loading: 'Model is loading, retry in a few seconds.',
  model_unavailable: 'Embedding model could not be loaded. Run semantic_reindex to attempt repair, or use text-based search via memory_manage.',
};

// ============================================================================
// Vector Metadata
// ============================================================================

export const FactVectorMetadataSchema = z.object({
  key: z.string(),
  namespace: z.string(),
  preview: z.string(),
});

export const DocumentVectorMetadataSchema = z.object({
  documentId: z.string(),
  title: z.string(),
  topic: z.string().optional(),
  tags: z.array(z.string()),
  contentType: z.string(),
});

export const VectorMetadataSchema = z.union([
  FactVectorMetadataSchema,
  DocumentVectorMetadataSchema,
]);

export type FactVectorMetadata = z.infer<typeof FactVectorMetadataSchema>;
export type DocumentVectorMetadata = z.infer<typeof DocumentVectorMetadataSchema>;
export type VectorMetadata = z.infer<typeof VectorMetadataSchema>;

// ============================================================================
// Vector Entry
// ============================================================================

export const VectorEntrySchema = z.object({
  id: z.string(),
  vector: z.array(z.number()),
  source: z.enum(['fact', 'document']),
  metadata: VectorMetadataSchema,
  indexedAt: z.number(),
});

export type VectorEntry = z.infer<typeof VectorEntrySchema>;

// ============================================================================
// Embedding Failure Record
// ============================================================================

export const EmbeddingFailureRecordSchema = z.object({
  retries: z.number().int().nonnegative(),
  status: z.enum(['pending', 'failed']),
  lastError: z.string().optional(),
  lastAttempt: z.number(),
});

export type EmbeddingFailureRecord = z.infer<typeof EmbeddingFailureRecordSchema>;

// ============================================================================
// Embedding Queue Item
// ============================================================================

export const EmbeddingQueueItemSchema = z.object({
  id: z.string(),
  source: z.enum(['fact', 'document']),
  textToEmbed: z.string(),
  metadata: VectorMetadataSchema,
  enqueuedAt: z.number(),
});

export type EmbeddingQueueItem = z.infer<typeof EmbeddingQueueItemSchema>;

// ============================================================================
// Vector Index (persisted to disk)
// ============================================================================

export const VectorIndexSchema = z.object({
  version: z.number(),
  modelId: z.string(),
  dimensions: z.number(),
  entries: z.array(VectorEntrySchema),
  updatedAt: z.number(),
  failedEmbeddings: z.record(z.string(), EmbeddingFailureRecordSchema).default({}),
});

export type VectorIndex = z.infer<typeof VectorIndexSchema>;

// ============================================================================
// Search Result
// ============================================================================

export const SearchResultSchema = z.object({
  source: z.enum(['fact', 'document']),
  id: z.string(),
  score: z.number(),
  metadata: VectorMetadataSchema,
});

export type SearchResult = z.infer<typeof SearchResultSchema>;

// ============================================================================
// Search Filter Options
// ============================================================================

export interface SearchFilterOptions {
  source?: 'all' | 'documents' | 'facts';
  namespace?: string;
  tag?: string;
  topic?: string;
  content_type?: string;
  limit?: number;
  threshold?: number;
}

// ============================================================================
// Helper: Compose vector entry ID
// ============================================================================

export function makeFactVectorId(namespace: string, key: string): string {
  return `${FACT_ID_PREFIX}${namespace}\0${key}`;
}

export function makeDocVectorId(documentId: string): string {
  return `${DOC_ID_PREFIX}${documentId}`;
}
