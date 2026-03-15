import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Logger } from 'pino';
import type { EmbeddingQueueItem, VectorEntry } from '../src/semantic-types.js';
import { EmbeddingQueue } from '../src/embedding-queue.js';

function mockLogger(): Logger {
  return {
    info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(),
    child: vi.fn().mockReturnThis(), fatal: vi.fn(), trace: vi.fn(),
    silent: vi.fn(), level: 'debug',
  } as unknown as Logger;
}

function makeItem(id: string, text = 'test text'): EmbeddingQueueItem {
  return {
    id, source: 'fact', textToEmbed: text,
    metadata: { key: 'k', namespace: 'ns', preview: text },
    enqueuedAt: Date.now(),
  };
}

function mockEmbedder() {
  return {
    embedPassage: vi.fn().mockResolvedValue([1, 0, 0]),
    embedQuery: vi.fn(),
  };
}

function mockVectorStore() {
  return {
    upsert: vi.fn(),
    save: vi.fn().mockResolvedValue(true),
    clearFailure: vi.fn(),
    recordFailure: vi.fn(),
    getFailedEmbeddings: vi.fn().mockReturnValue({}),
  };
}

describe('EmbeddingQueue - basic operations', () => {
  let embedder: ReturnType<typeof mockEmbedder>;
  let vectorStore: ReturnType<typeof mockVectorStore>;
  let queue: EmbeddingQueue;

  beforeEach(() => {
    embedder = mockEmbedder();
    vectorStore = mockVectorStore();
    queue = new EmbeddingQueue(embedder as any, vectorStore as any, mockLogger(), {
      debounceMs: 50, maxQueueSize: 100, maxRetries: 3,
    });
  });

  it('enqueue clears failure on vectorStore', () => {
    queue.enqueue(makeItem('item-1'));
    expect(vectorStore.clearFailure).toHaveBeenCalledWith('item-1');
  });

  it('flush processes all pending items', async () => {
    queue.enqueue(makeItem('item-1'));
    queue.enqueue(makeItem('item-2'));
    const result = await queue.flush();
    expect(result.succeeded).toBe(2);
    expect(result.failed).toBe(0);
    expect(vectorStore.upsert).toHaveBeenCalledTimes(2);
  });

  it('flush returns zeros when queue is empty', async () => {
    const result = await queue.flush();
    expect(result.succeeded).toBe(0);
    expect(result.failed).toBe(0);
  });

  it('dequeue removes pending item', async () => {
    queue.enqueue(makeItem('item-1'));
    queue.dequeue('item-1');
    const result = await queue.flush();
    expect(result.succeeded).toBe(0);
  });

  it('dispose clears pending queue', () => {
    queue.enqueue(makeItem('item-1'));
    queue.dispose();
    // After dispose, flush should have nothing to process
  });

  it('debounce triggers batch after delay', async () => {
    queue.enqueue(makeItem('item-1'));
    await new Promise(r => setTimeout(r, 100));
    expect(vectorStore.upsert).toHaveBeenCalled();
    queue.dispose();
  });

  it('maxQueueSize triggers immediate batch', async () => {
    const smallQueue = new EmbeddingQueue(embedder as any, vectorStore as any, mockLogger(), {
      debounceMs: 10000, maxQueueSize: 2, maxRetries: 3,
    });
    smallQueue.enqueue(makeItem('item-1'));
    smallQueue.enqueue(makeItem('item-2'));
    await new Promise(r => setTimeout(r, 50));
    expect(vectorStore.upsert).toHaveBeenCalledTimes(2);
    smallQueue.dispose();
  });

  it('handles embedding failures', async () => {
    embedder.embedPassage.mockRejectedValueOnce(new Error('embed failed'));
    queue.enqueue(makeItem('item-fail'));
    const result = await queue.flush();
    expect(result.failed).toBe(1);
    expect(vectorStore.recordFailure).toHaveBeenCalledWith('item-fail', 'embed failed', 3);
  });

  it('skips items with status=failed', async () => {
    vectorStore.getFailedEmbeddings.mockReturnValue({
      'item-skip': { retries: 3, status: 'failed', lastAttempt: Date.now() },
    });
    queue.enqueue(makeItem('item-skip'));
    const result = await queue.flush();
    expect(result.skipped).toBe(1);
    expect(result.succeeded).toBe(0);
  });

  it('deduplicates items by id', async () => {
    queue.enqueue(makeItem('item-1', 'text-v1'));
    queue.enqueue(makeItem('item-1', 'text-v2'));
    const result = await queue.flush();
    expect(result.succeeded).toBe(1);
    // The last enqueued value should be used
    expect(embedder.embedPassage).toHaveBeenCalledWith('text-v2');
  });

  it('saves vectorStore after batch', async () => {
    queue.enqueue(makeItem('item-1'));
    await queue.flush();
    expect(vectorStore.save).toHaveBeenCalled();
  });
});
