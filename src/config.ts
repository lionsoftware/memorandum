/**
 * Configuration schema and loader for Memorandum MCP server.
 *
 * Config file: {storage_dir}/config.yaml
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { z } from 'zod';
import { parse as yamlParse } from 'yaml';

/** Zod schema defining all configuration options with defaults. */
export const ConfigSchema = z.object({
  /** Maximum number of facts in memory (LRU eviction threshold). */
  max_entries: z.number().int().positive().default(1000),
  /** Autosave interval in seconds (0 to disable). */
  autosave_interval_seconds: z.number().int().nonnegative().default(300),
  /** Base storage directory. */
  storage_dir: z.string().default('.memorandum'),
  /** Maximum document body size in bytes (default: 16 MiB). */
  max_document_size: z.number().int().positive().default(16 * 1024 * 1024),
  /** Embedding model ID for semantic search. */
  semantic_model: z.string().default('Xenova/multilingual-e5-small'),
  /** Model quantization dtype (e.g. q8, fp32). */
  semantic_model_dtype: z.string().default('q8'),
  /** Enable or disable semantic search. */
  semantic_enabled: z.boolean().default(true),
  /** Debounce delay in seconds before batch embedding (0 = immediate). */
  semantic_debounce_seconds: z.number().int().nonnegative().default(10),
  /** Maximum pending items in embedding queue before forced batch. */
  semantic_max_queue_size: z.number().int().positive().default(200),
  /** Maximum embedding retry attempts per item before marking as failed. */
  semantic_max_retries: z.number().int().positive().default(3),
});

/** Inferred configuration type from {@link ConfigSchema}. */
export type Config = z.infer<typeof ConfigSchema>;

/** Derived paths from config. */
export interface ResolvedPaths {
  storageDir: string;
  factsPath: string;
  documentsPath: string;
  cachePath: string;
  configPath: string;
}

/** Resolve all storage paths from the base storage_dir. */
export function resolvePaths(storageDir: string): ResolvedPaths {
  return {
    storageDir,
    factsPath: join(storageDir, 'facts', 'facts.json'),
    documentsPath: join(storageDir, 'documents'),
    cachePath: join(storageDir, 'cache'),
    configPath: join(storageDir, 'config.yaml'),
  };
}

/** Load config from YAML file, falling back to defaults. */
export function loadConfig(storageDir?: string): { config: Config; paths: ResolvedPaths } {
  const baseDir = storageDir ?? '.memorandum';
  const configPath = join(baseDir, 'config.yaml');

  let rawConfig: Record<string, unknown> = {};

  if (existsSync(configPath)) {
    const content = readFileSync(configPath, 'utf-8');
    const parsed = yamlParse(content);
    if (typeof parsed === 'object' && parsed !== null) {
      rawConfig = parsed as Record<string, unknown>;
    }
  }

  // Ensure storage_dir is set
  if (!rawConfig.storage_dir) {
    rawConfig.storage_dir = baseDir;
  }

  const config = ConfigSchema.parse(rawConfig);
  const paths = resolvePaths(config.storage_dir);

  return { config, paths };
}
