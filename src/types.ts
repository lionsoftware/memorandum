import safeRegex from 'safe-regex2';
import { z } from 'zod';

// ============================================================================
// Constants
// ============================================================================

export const KEY_MAX_LEN = 256;
export const NAMESPACE_MAX_LEN = 64;
export const NAMESPACE_REGEX = /^[a-z0-9\-_.]+$/;
export const DEFAULT_NAMESPACE = 'default';
export const DEFAULT_LIST_LIMIT = 50;
export const DEFAULT_SEARCH_LIMIT = 20;

// ============================================================================
// Entity Types
// ============================================================================

export interface FactEntry {
  value: unknown;
  namespace: string;
  createdAt: number;
  updatedAt: number;
}

export interface FactMetadata {
  key: string;
  namespace: string;
  value: unknown;
  createdAt: number;
  updatedAt: number;
  ttl: number | null;
  remainingTtl: number | null;
}

export interface NamespaceInfo {
  namespace: string;
  count: number;
}

export interface MemoryStats {
  totalFacts: number;
  maxEntries: number;
  namespaceCount: number;
  lastSavedAt: number | null;
}

// ============================================================================
// Display Schemas (for LLM tool description - all fields optional)
// ============================================================================

export const MemoryWriteDisplaySchema = z.object({
  key: z.string().optional().describe('Unique identifier for the fact (1-256 characters)'),
  value: z.unknown().optional().describe('The data to store (any JSON-serializable value)'),
  namespace: z.string().optional().describe(
    `Namespace to organize facts (default: "${DEFAULT_NAMESPACE}", 1-64 chars, lowercase alphanumeric with -_.)`
  ),
  ttl_seconds: z.number().optional().describe('Optional time-to-live in seconds (positive integer)'),
});

export const MemoryReadDisplaySchema = z.object({
  key: z.string().optional().describe('The key of the fact to retrieve'),
  namespace: z.string().optional().describe(`Namespace to read from (default: "${DEFAULT_NAMESPACE}")`),
});

export const MemoryManageDisplaySchema = z.object({
  action: z.string().optional().describe(
    'Action to perform: "delete", "delete_namespace", "list", "search", "namespaces", "export", "import", "sync"'
  ),
  key: z.string().optional().describe('Key to delete (required for "delete" action)'),
  namespace: z.string().optional().describe('Namespace to operate on (various actions)'),
  pattern: z.string().optional().describe('Glob pattern to filter keys, e.g. "server-*" (for "list" action)'),
  limit: z.number().optional().describe(
    `Maximum number of results (default: ${DEFAULT_LIST_LIMIT} for list, ${DEFAULT_SEARCH_LIMIT} for search)`
  ),
  include_values: z.boolean().optional().describe('Include fact values in list results (default: false)'),
  include_stats: z.boolean().optional().describe('Include memory statistics in list results (default: false)'),
  query: z.string().optional().describe('Search query string (required for "search" action)'),
  data: z.string().optional().describe('JSON string to import (required for "import" action)'),
  merge: z.boolean().optional().describe('Merge imported data with existing facts (default: true, for "import" action)'),
});

// ============================================================================
// Input Schemas (for runtime validation - strict)
// ============================================================================

const namespaceField = z
  .string()
  .min(1, 'Namespace must not be empty')
  .max(NAMESPACE_MAX_LEN, `Namespace must not exceed ${NAMESPACE_MAX_LEN} characters`)
  .regex(NAMESPACE_REGEX, 'Namespace must contain only lowercase letters, numbers, hyphens, underscores, and dots');

export const MemoryWriteInputSchema = z.strictObject({
  key: z.string().min(1, 'Key must not be empty').max(KEY_MAX_LEN, `Key must not exceed ${KEY_MAX_LEN} characters`),
  value: z.unknown(),
  namespace: namespaceField.default(DEFAULT_NAMESPACE),
  ttl_seconds: z.number().int('TTL must be an integer').positive('TTL must be positive').optional(),
});

export const MemoryReadInputSchema = z.strictObject({
  key: z.string().min(1, 'Key must not be empty').max(KEY_MAX_LEN, `Key must not exceed ${KEY_MAX_LEN} characters`),
  namespace: namespaceField.default(DEFAULT_NAMESPACE),
});

export const MemoryManageInputSchema = z.discriminatedUnion('action', [
  z.strictObject({
    action: z.literal('delete'),
    key: z.string().min(1).max(KEY_MAX_LEN),
    namespace: namespaceField.default(DEFAULT_NAMESPACE),
  }),
  z.strictObject({
    action: z.literal('delete_namespace'),
    namespace: namespaceField,
  }),
  z.strictObject({
    action: z.literal('list'),
    namespace: namespaceField.optional(),
    pattern: z.string().optional(),
    limit: z.number().int().positive().default(DEFAULT_LIST_LIMIT),
    include_values: z.boolean().default(false),
    include_stats: z.boolean().default(false),
  }),
  z.strictObject({
    action: z.literal('search'),
    query: z.string().min(1, 'Search query must not be empty').refine(
      (val) => safeRegex(val),
      'Search query contains unsafe regex pattern (potential ReDoS)',
    ),
    namespace: namespaceField.optional(),
    limit: z.number().int().positive().default(DEFAULT_SEARCH_LIMIT),
  }),
  z.strictObject({ action: z.literal('namespaces') }),
  z.strictObject({ action: z.literal('export') }),
  z.strictObject({
    action: z.literal('import'),
    data: z.string().min(1, 'Import data must not be empty'),
    merge: z.boolean().default(true),
  }),
  z.strictObject({ action: z.literal('sync') }),
]);

// ============================================================================
// Inferred Types
// ============================================================================

export type MemoryWriteInput = z.infer<typeof MemoryWriteInputSchema>;
export type MemoryReadInput = z.infer<typeof MemoryReadInputSchema>;
export type MemoryManageInput = z.infer<typeof MemoryManageInputSchema>;
