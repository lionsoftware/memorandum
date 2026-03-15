import { existsSync, readFileSync } from 'node:fs';
import { writeFile, rename, mkdir, access } from 'node:fs/promises';
import { dirname } from 'node:path';
import type { Logger } from 'pino';
import type {
  VectorEntry, VectorIndex, SearchResult, SearchFilterOptions,
  FactVectorMetadata, DocumentVectorMetadata, EmbeddingFailureRecord,
} from './semantic-types.js';
import { VECTOR_INDEX_VERSION, DEFAULT_SEARCH_LIMIT, DEFAULT_SEARCH_THRESHOLD } from './semantic-types.js';

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Persisted in-memory vector store that supports cosine-similarity search,
 * atomic saves, and failure tracking for embedding retries.
 */
export class VectorStore {
  private readonly indexPath: string;
  private readonly modelId: string;
  private dimensions: number;
  private readonly logger: Logger;
  private entries: VectorEntry[] = [];
  private storedModelId: string | null = null;
  private failedEmbeddings: Record<string, EmbeddingFailureRecord> = {};
  private dirty = false;

  /**
   * @param indexPath - File path for the persisted JSON vector index.
   * @param modelId - Identifier of the embedding model producing the vectors.
   * @param dimensions - Expected dimensionality of embedding vectors.
   * @param logger - Pino logger instance.
   */
  constructor(indexPath: string, modelId: string, dimensions: number, logger: Logger) {
    this.indexPath = indexPath;
    this.modelId = modelId;
    this.dimensions = dimensions;
    this.logger = logger;
  }

  /** Loads the vector index from disk, or starts with an empty index if the file is missing or corrupt. */
  load(): void {
    if (!existsSync(this.indexPath)) {
      this.entries = [];
      this.storedModelId = null;
      return;
    }

    try {
      const raw = readFileSync(this.indexPath, 'utf-8');
      const parsed = JSON.parse(raw) as VectorIndex;
      if (typeof parsed !== 'object' || parsed === null || !Array.isArray(parsed.entries)) {
        throw new Error('Invalid vector index structure');
      }
      this.entries = parsed.entries;
      this.storedModelId = parsed.modelId ?? null;
      const rawFailed = (parsed as Record<string, unknown>).failedEmbeddings;
      this.failedEmbeddings = (rawFailed != null && typeof rawFailed === 'object' && !Array.isArray(rawFailed))
        ? rawFailed as Record<string, EmbeddingFailureRecord> : {};
      this.dirty = false;
      this.logger.debug({ entries: this.entries.length, modelId: this.storedModelId }, 'Vector index loaded');
    } catch (err) {
      this.logger.warn({ error: err instanceof Error ? err.message : String(err) }, 'Failed to parse vector index, starting empty');
      this.entries = [];
      this.storedModelId = null;
    }
  }

  /**
   * Atomically persists the vector index to disk via a temp-file rename.
   * Skips the write if nothing has changed, and refuses to overwrite a
   * non-empty file on disk with an empty in-memory index.
   * @returns True if the file was written, false otherwise.
   */
  async save(): Promise<boolean> {
    if (!this.dirty) return false;

    if (this.entries.length === 0 && existsSync(this.indexPath)) {
      try {
        const diskRaw = readFileSync(this.indexPath, 'utf-8');
        const diskIndex = JSON.parse(diskRaw) as VectorIndex;
        if (diskIndex && Array.isArray(diskIndex.entries) && diskIndex.entries.length > 0) {
          this.logger.warn('Skipping vector index save: disk has data but in-memory is empty');
          return false;
        }
      } catch { /* proceed */ }
    }

    const index: VectorIndex = {
      version: VECTOR_INDEX_VERSION,
      modelId: this.modelId,
      dimensions: this.dimensions,
      entries: this.entries,
      updatedAt: Date.now(),
      failedEmbeddings: this.failedEmbeddings,
    };

    try {
      const dir = dirname(this.indexPath);
      try { await access(dir); } catch { await mkdir(dir, { recursive: true }); }
      const data = JSON.stringify(index);
      const tmpPath = `${this.indexPath}.${process.pid}-${Date.now()}.tmp`;
      await writeFile(tmpPath, data, 'utf-8');
      await rename(tmpPath, this.indexPath);
      this.dirty = false;
      this.storedModelId = this.modelId;
      this.logger.debug({ entries: this.entries.length }, 'Vector index saved');
      return true;
    } catch (err) {
      this.logger.error({ error: err instanceof Error ? err.message : String(err) }, 'Failed to save vector index');
      return false;
    }
  }

  /**
   * Inserts or updates a vector entry by id.
   * @param entry - The vector entry to upsert.
   */
  upsert(entry: VectorEntry): void {
    const idx = this.entries.findIndex((e) => e.id === entry.id);
    if (idx !== -1) { this.entries[idx] = entry; } else { this.entries.push(entry); }
    this.dirty = true;
  }

  /**
   * Removes a vector entry by id.
   * @param id - The entry identifier.
   * @returns True if the entry was found and removed.
   */
  remove(id: string): boolean {
    const idx = this.entries.findIndex((e) => e.id === id);
    if (idx === -1) return false;
    this.entries.splice(idx, 1);
    this.dirty = true;
    return true;
  }

  /**
   * Replaces all entries in the store with the provided array.
   * @param entries - The new set of vector entries.
   */
  rebuild(entries: VectorEntry[]): void {
    this.entries = entries;
    this.dirty = true;
  }

  /**
   * Searches for the most similar entries using cosine similarity.
   * @param queryVector - The query embedding vector.
   * @param options - Optional filters and limits for the search.
   * @returns Matching results sorted by descending similarity score.
   */
  query(queryVector: number[], options?: SearchFilterOptions): SearchResult[] {
    const limit = options?.limit ?? DEFAULT_SEARCH_LIMIT;
    const threshold = options?.threshold ?? DEFAULT_SEARCH_THRESHOLD;

    let candidates = this.entries;

    if (options?.source === 'facts') candidates = candidates.filter((e) => e.source === 'fact');
    else if (options?.source === 'documents') candidates = candidates.filter((e) => e.source === 'document');

    if (options?.namespace !== undefined) {
      candidates = candidates.filter(
        (e) => e.source === 'fact' && (e.metadata as FactVectorMetadata).namespace === options.namespace,
      );
    }
    if (options?.tag !== undefined) {
      candidates = candidates.filter(
        (e) => e.source === 'document' && (e.metadata as DocumentVectorMetadata).tags.includes(options.tag!),
      );
    }
    if (options?.topic !== undefined) {
      candidates = candidates.filter(
        (e) => e.source === 'document' && (e.metadata as DocumentVectorMetadata).topic === options.topic,
      );
    }
    if (options?.content_type !== undefined) {
      candidates = candidates.filter(
        (e) => e.source === 'document' && (e.metadata as DocumentVectorMetadata).contentType === options.content_type,
      );
    }

    const scored: SearchResult[] = [];
    for (const entry of candidates) {
      if (queryVector.length !== entry.vector.length) continue;
      const score = cosineSimilarity(queryVector, entry.vector);
      if (score >= threshold) {
        scored.push({ source: entry.source, id: entry.id, score, metadata: entry.metadata });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }

  /** The number of vector entries currently stored. */
  get entryCount(): number { return this.entries.length; }

  /** Returns a read-only view of all stored vector entries. */
  allEntries(): ReadonlyArray<VectorEntry> { return this.entries; }

  /** Updates the expected embedding dimensions (e.g. after auto-detection). */
  setDimensions(n: number): void { this.dimensions = n; }

  /** The embedding model id configured for this store. */
  get currentModelId(): string { return this.modelId; }

  /** The model id stored in the persisted index, or null if not yet saved. */
  get indexModelId(): string | null { return this.storedModelId; }

  /** Returns the current map of failed embedding records. */
  getFailedEmbeddings(): Record<string, EmbeddingFailureRecord> { return this.failedEmbeddings; }

  /**
   * Replaces the entire failed-embeddings map (e.g. during reindex reset).
   * @param records - The new failure records to store.
   */
  setFailedEmbeddings(records: Record<string, EmbeddingFailureRecord>): void {
    this.failedEmbeddings = records;
    this.dirty = true;
  }

  /**
   * Removes a failure record for the given id, if one exists.
   * @param id - The entry identifier whose failure record should be cleared.
   */
  clearFailure(id: string): void {
    if (id in this.failedEmbeddings) { delete this.failedEmbeddings[id]; this.dirty = true; }
  }

  /**
   * Records or increments a failure for a given entry. Marks it as permanently
   * "failed" once the retry count reaches maxRetries.
   * @param id - The entry identifier that failed.
   * @param error - The error message from the failed attempt.
   * @param maxRetries - Maximum retries before marking as permanently failed.
   */
  recordFailure(id: string, error: string, maxRetries: number): void {
    const existing = this.failedEmbeddings[id];
    const retries = (existing?.retries ?? 0) + 1;
    this.failedEmbeddings[id] = {
      retries,
      status: retries >= maxRetries ? 'failed' : 'pending',
      lastError: error,
      lastAttempt: Date.now(),
    };
    this.dirty = true;
  }
}
