#!/usr/bin/env node

/**
 * Memorandum — MCP server with persistent memory.
 *
 * Two memory types:
 * - Facts: short key-value records with LRU expiration
 * - Documents: structured storage with metadata and semantic search
 */

import { createRequire } from 'node:module';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { loadConfig } from './config.js';
import { createLogger } from './logger.js';
import { MemoryStore } from './store.js';
import { DocumentStore } from './document-store.js';
import { VectorStore } from './vector-store.js';
import { Embedder } from './embedder.js';
import { SemanticIndex } from './semantic-index.js';
import { EmbeddingQueue } from './embedding-queue.js';
import { registerMemoryTools, type SyncFn } from './tools.js';
import { registerDocumentTools } from './document-tools.js';
import { registerSemanticTools } from './semantic-tools.js';

const _require = createRequire(import.meta.url);
const { version } = _require('../package.json') as { version: string };

/**
 * Ensure .memorandum/.gitignore exists with cache/ exclusion.
 */
function ensureGitignore(storageDir: string): void {
  const gitignorePath = join(storageDir, '.gitignore');
  const cacheEntry = 'cache/';

  if (!existsSync(storageDir)) {
    mkdirSync(storageDir, { recursive: true });
  }

  if (!existsSync(gitignorePath)) {
    writeFileSync(gitignorePath, `${cacheEntry}\n`, 'utf-8');
    return;
  }

  const content = readFileSync(gitignorePath, 'utf-8');
  if (!content.includes(cacheEntry)) {
    writeFileSync(gitignorePath, `${content.trimEnd()}\n${cacheEntry}\n`, 'utf-8');
  }
}

async function main(): Promise<void> {
  const storageDir = process.env.MEMORANDUM_STORAGE_DIR || '.memorandum';
  const logLevel = process.env.MEMORANDUM_LOG_LEVEL || 'info';
  const logger = createLogger(logLevel);

  const { config, paths } = loadConfig(storageDir);

  logger.info({ version, storageDir: paths.storageDir }, 'Memorandum starting');

  // Ensure storage directories
  if (!existsSync(paths.storageDir)) mkdirSync(paths.storageDir, { recursive: true });
  const factsDir = dirname(paths.factsPath);
  if (!existsSync(factsDir)) mkdirSync(factsDir, { recursive: true });

  // Initialize MCP server
  const server = new McpServer({
    name: 'memorandum',
    version,
  });

  // Initialize stores
  const store = new MemoryStore(config, paths, logger);
  await store.load();

  const documentStore = new DocumentStore(config, paths, logger);
  documentStore.loadIndex();

  // Component references for teardown
  let vectorStore: VectorStore | undefined;
  let semanticIndex: SemanticIndex | undefined;
  let embeddingQueue: EmbeddingQueue | undefined;

  // Sync callback
  const syncFn: SyncFn = async () => {
    const savedStores: string[] = [];
    let flushedEmbeddings = 0;

    if (embeddingQueue) {
      const result = await embeddingQueue.flush();
      flushedEmbeddings = result.succeeded;
    }
    if (vectorStore) {
      const written = await vectorStore.save();
      if (written) savedStores.push('vectors');
    }
    const docsWritten = documentStore.saveIndex();
    if (docsWritten) savedStores.push('documents');
    const factsWritten = await store.save();
    if (factsWritten) savedStores.push('facts');

    return { flushed_embeddings: flushedEmbeddings, saved_stores: savedStores };
  };

  // Register fact tools
  registerMemoryTools(server, store, logger, syncFn);

  // Register document tools
  registerDocumentTools(server, documentStore, logger);

  // Semantic search initialization
  if (config.semantic_enabled) {
    const embedder = new Embedder(config.semantic_model, config.semantic_model_dtype, logger);
    const indexPath = join(paths.cachePath, 'vector-index.json');
    const dimensions = 384; // e5-small default
    vectorStore = new VectorStore(indexPath, config.semantic_model, dimensions, logger);

    semanticIndex = new SemanticIndex(embedder, vectorStore, logger);
    await semanticIndex.initialize();
    semanticIndex.setMemoryStore(store);
    semanticIndex.setDocumentStore(documentStore);
    await semanticIndex.pruneOrphans();

    embeddingQueue = new EmbeddingQueue(embedder, vectorStore, logger, {
      debounceMs: config.semantic_debounce_seconds * 1000,
      maxQueueSize: config.semantic_max_queue_size,
      maxRetries: config.semantic_max_retries,
    });
    semanticIndex.setEmbeddingQueue(embeddingQueue);
    semanticIndex.enqueueMissing();

    documentStore.setSemanticIndex(semanticIndex);
    store.setSemanticIndex(semanticIndex);

    registerSemanticTools(server, semanticIndex, logger);

    ensureGitignore(paths.storageDir);
  }

  // Start autosave
  const intervalMs = config.autosave_interval_seconds * 1000;
  store.startAutosave(intervalMs);

  // Graceful shutdown
  const teardown = async () => {
    logger.info('Shutting down...');
    if (embeddingQueue) {
      await embeddingQueue.flush();
      embeddingQueue.dispose();
    }
    if (vectorStore) await vectorStore.save();
    documentStore.saveIndex();
    store.stopAutosave();
    await store.save();
    logger.info('Shutdown complete');
  };

  process.on('SIGINT', () => { teardown().then(() => process.exit(0)); });
  process.on('SIGTERM', () => { teardown().then(() => process.exit(0)); });

  // Start MCP transport
  const transport = new StdioServerTransport();
  await server.connect(transport);

  logger.info({ tools: 10, semantic: config.semantic_enabled }, 'Memorandum ready');
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
