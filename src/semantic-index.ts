import type { Logger } from 'pino';
import type { Embedder } from './embedder.js';
import type { VectorStore } from './vector-store.js';
import type { MemoryStore } from './store.js';
import type { DocumentStore } from './document-store.js';
import type { EmbeddingQueue } from './embedding-queue.js';
import type {
  VectorEntry, SearchResult, SearchFilterOptions, IndexStatus,
  FactVectorMetadata, DocumentVectorMetadata, EmbeddingQueueItem,
} from './semantic-types.js';
import { INDEX_STATUS_HINTS, FACT_ID_PREFIX, DOC_ID_PREFIX, makeFactVectorId, makeDocVectorId } from './semantic-types.js';
import { isInlineContentType } from './document-types.js';

/**
 * High-level semantic search index that coordinates the Embedder, VectorStore,
 * and EmbeddingQueue to index and search facts and documents.
 */
export class SemanticIndex {
  private readonly embedder: Embedder;
  private readonly vectorStore: VectorStore;
  private readonly logger: Logger;
  private memoryStore: MemoryStore | null = null;
  private documentStore: DocumentStore | null = null;
  private embeddingQueue: EmbeddingQueue | null = null;

  constructor(embedder: Embedder, vectorStore: VectorStore, logger: Logger) {
    this.embedder = embedder;
    this.vectorStore = vectorStore;
    this.logger = logger;
  }

  /** Assigns the embedding queue used for batched, asynchronous indexing. */
  setEmbeddingQueue(queue: EmbeddingQueue): void { this.embeddingQueue = queue; }
  /** The identifier of the embedding model currently in use. */
  get modelId(): string { return this.vectorStore.currentModelId; }
  /** Assigns the memory store used for fact lookups during reindex/prune. */
  setMemoryStore(store: MemoryStore): void { this.memoryStore = store; }
  /** Assigns the document store used for document lookups during reindex/prune. */
  setDocumentStore(store: DocumentStore): void { this.documentStore = store; }

  /**
   * Detects embedding dimensions by running a test embedding,
   * then loads the vector index from disk.
   */
  async initialize(): Promise<void> {
    try {
      const vector = await this.embedder.embedPassage('init');
      this.vectorStore.setDimensions(vector.length);
      this.logger.info({ dimensions: vector.length, model: this.vectorStore.currentModelId }, 'Embedding dimensions detected');
    } catch (err) {
      this.logger.warn({ error: err instanceof Error ? err.message : String(err) }, 'Failed to detect dimensions — using default');
    }
    this.vectorStore.load();
  }

  // ==========================================================================
  // Document Indexing
  // ==========================================================================

  /**
   * Enqueues a document for asynchronous embedding via the EmbeddingQueue.
   * @param id - The document identifier.
   * @param metadata - Document metadata (title, tags, content type, etc.).
   * @param body - Optional document body used for inline content types.
   */
  enqueueDocument(
    id: string,
    metadata: { title: string; topic?: string; tags: string[]; contentType: string; description?: string },
    body?: string,
  ): void {
    if (!this.embeddingQueue) return;

    const textToEmbed = isInlineContentType(metadata.contentType) && body
      ? body
      : [metadata.title, metadata.description, ...(metadata.tags || [])].filter(Boolean).join(' ');

    const docMetadata: DocumentVectorMetadata = {
      documentId: id, title: metadata.title,
      topic: metadata.topic, tags: metadata.tags, contentType: metadata.contentType,
    };

    this.embeddingQueue.enqueue({
      id: makeDocVectorId(id), source: 'document',
      textToEmbed, metadata: docMetadata, enqueuedAt: Date.now(),
    });
  }

  /**
   * Indexes a document immediately, or delegates to the queue if available.
   * @param id - The document identifier.
   * @param metadata - Document metadata (title, tags, content type, etc.).
   * @param body - Optional document body used for inline content types.
   */
  async indexDocument(
    id: string,
    metadata: { title: string; topic?: string; tags: string[]; contentType: string; description?: string },
    body?: string,
  ): Promise<void> {
    if (this.embeddingQueue) { this.enqueueDocument(id, metadata, body); return; }

    try {
      const textToEmbed = isInlineContentType(metadata.contentType) && body
        ? body
        : [metadata.title, metadata.description, ...(metadata.tags || [])].filter(Boolean).join(' ');

      const vector = await this.embedder.embedPassage(textToEmbed);
      const docMetadata: DocumentVectorMetadata = {
        documentId: id, title: metadata.title,
        topic: metadata.topic, tags: metadata.tags, contentType: metadata.contentType,
      };

      this.vectorStore.upsert({
        id: makeDocVectorId(id), vector, source: 'document',
        metadata: docMetadata, indexedAt: Date.now(),
      });
    } catch (err) {
      this.logger.warn({ documentId: id, error: err instanceof Error ? err.message : String(err) }, 'Failed to index document');
    }
  }

  /**
   * Removes a document's vector entry and dequeues any pending embedding.
   * @param id - The document identifier to remove.
   */
  removeDocument(id: string): void {
    const vectorId = makeDocVectorId(id);
    if (this.embeddingQueue) this.embeddingQueue.dequeue(vectorId);
    const removed = this.vectorStore.remove(vectorId);
    if (removed) {
      this.vectorStore.save().catch((err: unknown) => {
        this.logger.warn({ error: err instanceof Error ? err.message : String(err) }, 'Failed to save after document removal');
      });
    }
  }

  // ==========================================================================
  // Fact Indexing
  // ==========================================================================

  /**
   * Enqueues a fact for asynchronous embedding via the EmbeddingQueue.
   * @param key - The fact key.
   * @param namespace - The fact namespace.
   * @param value - The fact value to embed.
   */
  enqueueFact(key: string, namespace: string, value: unknown): void {
    if (!this.embeddingQueue) return;

    const valueStr = typeof value === 'string' ? value : JSON.stringify(value);
    const textToEmbed = `${namespace}: ${key} — ${valueStr}`;
    const preview = valueStr.length > 200 ? valueStr.slice(0, 200) : valueStr;
    const factMetadata: FactVectorMetadata = { key, namespace, preview };

    this.embeddingQueue.enqueue({
      id: makeFactVectorId(namespace, key), source: 'fact',
      textToEmbed, metadata: factMetadata, enqueuedAt: Date.now(),
    });
  }

  /**
   * Indexes a fact immediately, or delegates to the queue if available.
   * @param key - The fact key.
   * @param namespace - The fact namespace.
   * @param value - The fact value to embed.
   */
  async indexFact(key: string, namespace: string, value: unknown): Promise<void> {
    if (this.embeddingQueue) { this.enqueueFact(key, namespace, value); return; }

    try {
      const valueStr = typeof value === 'string' ? value : JSON.stringify(value);
      const textToEmbed = `${namespace}: ${key} — ${valueStr}`;
      const vector = await this.embedder.embedPassage(textToEmbed);
      const preview = valueStr.length > 200 ? valueStr.slice(0, 200) : valueStr;

      this.vectorStore.upsert({
        id: makeFactVectorId(namespace, key), vector, source: 'fact',
        metadata: { key, namespace, preview } as FactVectorMetadata, indexedAt: Date.now(),
      });
    } catch (err) {
      this.logger.warn({ key, namespace, error: err instanceof Error ? err.message : String(err) }, 'Failed to index fact');
    }
  }

  /**
   * Removes a fact's vector entry and dequeues any pending embedding.
   * @param key - The fact key.
   * @param namespace - The fact namespace.
   */
  removeFact(key: string, namespace: string): void {
    const vectorId = makeFactVectorId(namespace, key);
    if (this.embeddingQueue) this.embeddingQueue.dequeue(vectorId);
    const removed = this.vectorStore.remove(vectorId);
    if (removed) {
      this.vectorStore.save().catch((err: unknown) => {
        this.logger.warn({ error: err instanceof Error ? err.message : String(err) }, 'Failed to save after fact removal');
      });
    }
  }

  // ==========================================================================
  // Orphan Cleanup
  // ==========================================================================

  /**
   * Removes vector entries whose corresponding fact or document no longer
   * exists in the memory or document store.
   */
  async pruneOrphans(): Promise<void> {
    const entries = this.vectorStore.allEntries();
    const orphanIds: string[] = [];

    for (const entry of entries) {
      if (entry.id.startsWith(FACT_ID_PREFIX)) {
        const rest = entry.id.slice(FACT_ID_PREFIX.length);
        const sepIdx = rest.indexOf('\0');
        if (sepIdx === -1) { orphanIds.push(entry.id); continue; }
        const namespace = rest.slice(0, sepIdx);
        const key = rest.slice(sepIdx + 1);
        if (this.memoryStore && !this.memoryStore.read(key, namespace)) {
          orphanIds.push(entry.id);
        }
      } else if (entry.id.startsWith(DOC_ID_PREFIX)) {
        const docId = entry.id.slice(DOC_ID_PREFIX.length);
        if (this.documentStore) {
          if (!this.documentStore.getIndex().documents.some((d) => d.id === docId)) {
            orphanIds.push(entry.id);
          }
        }
      }
    }

    if (orphanIds.length > 0) {
      for (const id of orphanIds) this.vectorStore.remove(id);
      await this.vectorStore.save();
      this.logger.info({ pruned: orphanIds.length }, 'Pruned orphaned vector entries');
    }
  }

  // ==========================================================================
  // Startup Recovery
  // ==========================================================================

  /**
   * Scans for facts and documents that are missing from the vector index
   * (or have pending retry records) and enqueues them for embedding.
   * @returns The number of items enqueued.
   */
  enqueueMissing(): number {
    if (!this.embeddingQueue) return 0;

    let enqueued = 0;
    const existingIds = new Set(this.vectorStore.allEntries().map((e) => e.id));

    const failures = this.vectorStore.getFailedEmbeddings();
    for (const [id, record] of Object.entries(failures)) {
      if (record.status !== 'pending') continue;
      if (id.startsWith(FACT_ID_PREFIX) && this.memoryStore) {
        const rest = id.slice(FACT_ID_PREFIX.length);
        const sepIdx = rest.indexOf('\0');
        if (sepIdx === -1) continue;
        const namespace = rest.slice(0, sepIdx);
        const key = rest.slice(sepIdx + 1);
        const fact = this.memoryStore.read(key, namespace);
        if (fact) { this.enqueueFact(key, namespace, fact.value); enqueued++; }
      } else if (id.startsWith(DOC_ID_PREFIX) && this.documentStore) {
        const docId = id.slice(DOC_ID_PREFIX.length);
        const docMeta = this.documentStore.getIndex().documents.find((d) => d.id === docId);
        if (docMeta) {
          const fullDoc = this.documentStore.read({ id: docId, include_body: true });
          this.enqueueDocument(docId, {
            title: docMeta.title, topic: docMeta.topic,
            tags: docMeta.tags ?? [], contentType: docMeta.content_type,
            description: docMeta.description,
          }, fullDoc.body as string | undefined);
          enqueued++;
        }
      }
    }

    if (this.memoryStore) {
      const factsList = this.memoryStore.list({
        limit: this.memoryStore.maxEntries, includeValues: true, includeStats: false,
      });
      for (const fact of factsList.facts) {
        const key = fact.key as string;
        const namespace = fact.namespace as string;
        if (!existingIds.has(makeFactVectorId(namespace, key))) {
          this.enqueueFact(key, namespace, fact.value);
          enqueued++;
        }
      }
    }

    if (this.documentStore) {
      for (const docMeta of this.documentStore.getIndex().documents) {
        if (!existingIds.has(makeDocVectorId(docMeta.id))) {
          const fullDoc = this.documentStore.read({ id: docMeta.id, include_body: true });
          this.enqueueDocument(docMeta.id, {
            title: docMeta.title, topic: docMeta.topic,
            tags: docMeta.tags ?? [], contentType: docMeta.content_type,
            description: docMeta.description,
          }, fullDoc.body as string | undefined);
          enqueued++;
        }
      }
    }

    if (enqueued > 0) this.logger.info({ enqueued }, 'Enqueued missing embeddings for recovery');
    return enqueued;
  }

  // ==========================================================================
  // Search
  // ==========================================================================

  /**
   * Performs a semantic search against the vector index.
   * @param query - Natural-language search query.
   * @param options - Optional filters (source, namespace, tag, topic, limit, threshold).
   * @returns The index status, an optional hint, matching results, and total match count.
   */
  async search(query: string, options?: SearchFilterOptions): Promise<{
    status: IndexStatus; hint?: string; results: SearchResult[]; total: number;
  }> {
    const status = this.getStatus();
    const hint = INDEX_STATUS_HINTS[status];

    if (status === 'model_unavailable' || status === 'model_loading') {
      const reason = this.embedder.unavailableReason;
      return { status, hint: reason ? `${hint} Reason: ${reason}` : hint, results: [], total: 0 };
    }
    if (status === 'empty') return { status, hint, results: [], total: 0 };

    try {
      const queryVector = await this.embedder.embedQuery(query);
      const allResults = this.vectorStore.query(queryVector, { ...options, limit: Number.MAX_SAFE_INTEGER });
      const total = allResults.length;
      const limit = options?.limit ?? 10;
      return { status, hint, results: allResults.slice(0, limit), total };
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      this.logger.warn({ query, error: errorMsg }, 'Semantic search failed');
      return { status: 'model_unavailable', hint: `${INDEX_STATUS_HINTS.model_unavailable} Reason: ${errorMsg}`, results: [], total: 0 };
    }
  }

  // ==========================================================================
  // Reindex
  // ==========================================================================

  /**
   * Re-embeds and rebuilds the vector index for the specified source(s).
   * Repairs the model if needed before starting.
   * @param source - Which entries to reindex: "all", "documents", or "facts".
   * @returns Counts of indexed facts, documents, skipped items, and duration.
   */
  async reindex(source: 'all' | 'documents' | 'facts' = 'all'): Promise<{
    facts: number; documents: number; total: number; skipped: number; duration_ms: number;
  }> {
    await this.embedder.repairAndLoad();

    const start = Date.now();
    let factsIndexed = 0, docsIndexed = 0, skipped = 0;
    const newEntries: VectorEntry[] = [];

    if (source !== 'documents' && this.memoryStore) {
      const factsList = this.memoryStore.list({
        limit: this.memoryStore.maxEntries, includeValues: true, includeStats: false,
      });
      for (const fact of factsList.facts) {
        try {
          const key = fact.key as string;
          const namespace = fact.namespace as string;
          const valueStr = typeof fact.value === 'string' ? fact.value : JSON.stringify(fact.value);
          const vector = await this.embedder.embedPassage(`${namespace}: ${key} — ${valueStr}`);
          const preview = valueStr.length > 200 ? valueStr.slice(0, 200) : valueStr;
          newEntries.push({
            id: makeFactVectorId(namespace, key), vector, source: 'fact',
            metadata: { key, namespace, preview } as FactVectorMetadata, indexedAt: Date.now(),
          });
          factsIndexed++;
        } catch { skipped++; }
      }
    }

    if (source !== 'facts' && this.documentStore) {
      for (const docRecord of this.documentStore.list({}).documents) {
        try {
          const id = docRecord.id as string;
          const title = docRecord.title as string;
          const contentType = docRecord.content_type as string;
          const tags = (docRecord.tags as string[]) ?? [];
          const topic = docRecord.topic as string | undefined;
          const description = docRecord.description as string | undefined;

          let textToEmbed: string;
          if (isInlineContentType(contentType)) {
            const fullDoc = this.documentStore.read({ id, include_body: true });
            textToEmbed = (fullDoc.body as string) || [title, description, ...tags].filter(Boolean).join(' ');
          } else {
            textToEmbed = [title, description, ...tags].filter(Boolean).join(' ');
          }

          const vector = await this.embedder.embedPassage(textToEmbed);
          newEntries.push({
            id: makeDocVectorId(id), vector, source: 'document',
            metadata: { documentId: id, title, topic, tags, contentType } as DocumentVectorMetadata,
            indexedAt: Date.now(),
          });
          docsIndexed++;
        } catch { skipped++; }
      }
    }

    if (source === 'all') {
      this.vectorStore.rebuild(newEntries);
      this.vectorStore.setFailedEmbeddings({});
    } else {
      for (const entry of newEntries) this.vectorStore.upsert(entry);
    }

    await this.vectorStore.save();
    const duration_ms = Date.now() - start;
    this.logger.info({ factsIndexed, docsIndexed, skipped, duration_ms }, 'Reindex completed');

    return { facts: factsIndexed, documents: docsIndexed, total: factsIndexed + docsIndexed, skipped, duration_ms };
  }

  // ==========================================================================
  // Status
  // ==========================================================================

  /**
   * Returns the current status of the semantic index (e.g. "ready",
   * "empty", "model_loading", "model_unavailable", "stale_model", "partial").
   */
  getStatus(): IndexStatus {
    if (this.embedder.isUnavailable) return 'model_unavailable';
    if (this.embedder.isLoading) return 'model_loading';
    if (this.vectorStore.entryCount === 0) return 'empty';

    const indexModel = this.vectorStore.indexModelId;
    if (indexModel && indexModel !== this.vectorStore.currentModelId) return 'stale_model';

    const totalItems = this.getTotalItemCount();
    if (totalItems > 0 && this.vectorStore.entryCount < totalItems) return 'partial';

    return 'ready';
  }

  private getTotalItemCount(): number {
    let total = 0;
    if (this.memoryStore) total += this.memoryStore.size;
    if (this.documentStore) total += this.documentStore.getIndex().documents.length;
    return total;
  }
}
