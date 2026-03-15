import type { Logger } from 'pino';
import type { Embedder } from './embedder.js';
import type { VectorStore } from './vector-store.js';
import type { EmbeddingQueueItem, VectorEntry } from './semantic-types.js';

/** Configuration options for the embedding queue. */
export interface EmbeddingQueueConfig {
  /** Delay in milliseconds before processing a batch after the last enqueue. */
  debounceMs: number;
  /** Maximum number of pending items before a batch is triggered immediately. */
  maxQueueSize: number;
  /** Maximum retry attempts before marking an embedding as permanently failed. */
  maxRetries: number;
}

/** Summary of a completed embedding batch. */
export interface BatchResult {
  /** Number of items successfully embedded and stored. */
  succeeded: number;
  /** Number of items that failed during embedding. */
  failed: number;
  /** Number of items skipped (discarded or permanently failed). */
  skipped: number;
  /** Wall-clock time of the batch in milliseconds. */
  durationMs: number;
}

/**
 * Debounced queue that batches embedding requests and processes them
 * sequentially through the Embedder, persisting results to the VectorStore.
 */
export class EmbeddingQueue {
  private readonly embedder: Embedder;
  private readonly vectorStore: VectorStore;
  private readonly logger: Logger;
  private readonly config: EmbeddingQueueConfig;

  private pending = new Map<string, EmbeddingQueueItem>();
  private processingIds = new Set<string>();
  private discardedIds = new Set<string>();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private processing = false;
  private processingPromise: Promise<void> | null = null;

  constructor(embedder: Embedder, vectorStore: VectorStore, logger: Logger, config: EmbeddingQueueConfig) {
    this.embedder = embedder;
    this.vectorStore = vectorStore;
    this.logger = logger;
    this.config = config;
  }

  /**
   * Adds an item to the queue. Triggers an immediate batch if the queue
   * reaches maxQueueSize, otherwise resets the debounce timer.
   * @param item - The item to enqueue for embedding.
   */
  enqueue(item: EmbeddingQueueItem): void {
    this.pending.set(item.id, item);
    this.vectorStore.clearFailure(item.id);

    if (this.pending.size >= this.config.maxQueueSize) {
      this.cancelTimer();
      this.triggerBatch();
      return;
    }
    this.resetTimer();
  }

  /**
   * Removes an item from the pending queue. If the item is currently being
   * processed, marks it as discarded so its result will be ignored.
   * @param id - The identifier of the item to remove.
   */
  dequeue(id: string): void {
    this.pending.delete(id);
    if (this.processingIds.has(id)) this.discardedIds.add(id);
    this.vectorStore.clearFailure(id);
  }

  /**
   * Processes all pending items immediately, waiting for any in-flight
   * batch to complete first.
   * @returns The result of the flushed batch.
   */
  async flush(): Promise<BatchResult> {
    this.cancelTimer();
    if (this.pending.size === 0 && !this.processing) {
      return { succeeded: 0, failed: 0, skipped: 0, durationMs: 0 };
    }
    if (this.processing) await this.waitForProcessing();
    if (this.pending.size === 0) {
      return { succeeded: 0, failed: 0, skipped: 0, durationMs: 0 };
    }
    return this.processBatch();
  }

  /** Cancels any pending timer and clears the queue. */
  dispose(): void {
    this.cancelTimer();
    this.pending.clear();
  }

  private resetTimer(): void {
    this.cancelTimer();
    const delay = this.config.debounceMs === 0 ? 0 : this.config.debounceMs;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.triggerBatch();
    }, delay);
    this.timer.unref();
  }

  private cancelTimer(): void {
    if (this.timer !== null) { clearTimeout(this.timer); this.timer = null; }
  }

  private triggerBatch(): void {
    if (this.processing || this.pending.size === 0) return;
    this.processBatch().catch((err: unknown) => {
      this.logger.error({ error: err instanceof Error ? err.message : String(err) }, 'Batch processing failed');
    });
  }

  private async waitForProcessing(): Promise<void> {
    if (this.processingPromise) await this.processingPromise;
  }

  private async processBatch(): Promise<BatchResult> {
    const start = Date.now();
    this.processing = true;

    const batch = new Map(this.pending);
    this.pending.clear();
    this.processingIds = new Set(batch.keys());
    this.discardedIds.clear();

    let succeeded = 0, failed = 0, skipped = 0;

    const processPromise = (async () => {
      for (const [id, item] of batch) {
        const failureRecord = this.vectorStore.getFailedEmbeddings()[id];
        if (failureRecord && failureRecord.status === 'failed') {
          skipped++;
          continue;
        }

        try {
          const vector = await this.embedder.embedPassage(item.textToEmbed);
          if (this.discardedIds.has(id)) { skipped++; continue; }

          const entry: VectorEntry = {
            id, vector, source: item.source,
            metadata: item.metadata, indexedAt: Date.now(),
          };
          this.vectorStore.upsert(entry);
          this.vectorStore.clearFailure(id);
          succeeded++;
        } catch (err) {
          const errorMsg = err instanceof Error ? err.message : String(err);
          this.vectorStore.recordFailure(id, errorMsg, this.config.maxRetries);
          failed++;
          this.logger.warn({ id, error: errorMsg }, 'Embedding failed for item');
        }
      }

      if (succeeded > 0 || failed > 0) await this.vectorStore.save();
    })();

    this.processingPromise = processPromise;

    try {
      await processPromise;
    } finally {
      this.processing = false;
      this.processingIds.clear();
      this.discardedIds.clear();
      this.processingPromise = null;
    }

    const durationMs = Date.now() - start;
    this.logger.info({ succeeded, failed, skipped, durationMs, remaining: this.pending.size }, 'Embedding batch completed');

    if (this.pending.size > 0) this.resetTimer();

    return { succeeded, failed, skipped, durationMs };
  }
}
