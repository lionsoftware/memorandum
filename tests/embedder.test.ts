import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Logger } from 'pino';

function mockLogger(): Logger {
  return {
    info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(),
    child: vi.fn().mockReturnThis(), fatal: vi.fn(), trace: vi.fn(),
    silent: vi.fn(), level: 'debug',
  } as unknown as Logger;
}

const mockPipelineFn = vi.fn();
const mockOutput = { tolist: () => [[0.1, 0.2, 0.3, 0.4, 0.5]] };
mockPipelineFn.mockResolvedValue(mockOutput);
const mockExtractor = Object.assign(mockPipelineFn, {}) as unknown;
const mockPipelineFactory = vi.fn().mockResolvedValue(mockExtractor);

vi.mock('@huggingface/transformers', () => ({
  pipeline: mockPipelineFactory,
  env: { allowRemoteModels: true, cacheDir: '/tmp/test-hf-cache' },
}));

const { Embedder } = await import('../src/embedder.js');

function makeEmbedder(modelId = 'multilingual-e5-small', dtype = 'q8') {
  const logger = mockLogger();
  return { embedder: new Embedder(modelId, dtype, logger), logger };
}

describe('Embedder - Lazy initialization', () => {
  beforeEach(() => {
    mockPipelineFn.mockClear();
    mockPipelineFactory.mockClear();
    mockPipelineFactory.mockResolvedValue(mockExtractor);
  });

  it('isLoaded is false before any embed call', () => {
    expect(makeEmbedder().embedder.isLoaded).toBe(false);
  });

  it('isLoaded becomes true after embedQuery', async () => {
    const { embedder } = makeEmbedder();
    await embedder.embedQuery('hello');
    expect(embedder.isLoaded).toBe(true);
  });

  it('pipeline factory called once even after multiple embed calls', async () => {
    const { embedder } = makeEmbedder();
    await embedder.embedQuery('first');
    await embedder.embedPassage('second');
    expect(mockPipelineFactory).toHaveBeenCalledTimes(1);
  });

  it('pipeline factory receives correct model id and options', async () => {
    const { embedder } = makeEmbedder('multilingual-e5-small', 'q8');
    await embedder.embedQuery('test');
    expect(mockPipelineFactory).toHaveBeenCalledWith(
      'feature-extraction', 'multilingual-e5-small',
      expect.objectContaining({ dtype: 'q8', device: 'cpu' }),
    );
  });
});

describe('Embedder - E5 prefix injection', () => {
  beforeEach(() => {
    mockPipelineFn.mockClear();
    mockPipelineFactory.mockClear();
    mockPipelineFactory.mockResolvedValue(mockExtractor);
  });

  it('embedQuery prepends "query: " for e5 model', async () => {
    const { embedder } = makeEmbedder('multilingual-e5-small');
    await embedder.embedQuery('some text');
    expect(mockPipelineFn).toHaveBeenCalledWith('query: some text', expect.anything());
  });

  it('embedPassage prepends "passage: " for e5 model', async () => {
    const { embedder } = makeEmbedder('multilingual-e5-small');
    await embedder.embedPassage('some passage');
    expect(mockPipelineFn).toHaveBeenCalledWith('passage: some passage', expect.anything());
  });

  it('no prefix for non-e5 model', async () => {
    const { embedder } = makeEmbedder('some-other-model');
    await embedder.embedQuery('raw text');
    expect(mockPipelineFn).toHaveBeenCalledWith('raw text', expect.anything());
  });

  it('e5 detection is case-insensitive', async () => {
    const { embedder } = makeEmbedder('Multilingual-E5-Small');
    await embedder.embedQuery('text');
    expect(mockPipelineFn).toHaveBeenCalledWith('query: text', expect.anything());
  });
});

describe('Embedder - Error handling', () => {
  beforeEach(() => {
    mockPipelineFn.mockClear();
    mockPipelineFactory.mockClear();
    mockPipelineFactory.mockResolvedValue(mockExtractor);
  });

  it('isUnavailable becomes true when pipeline factory throws', async () => {
    mockPipelineFactory.mockRejectedValueOnce(new Error('Network error'));
    const { embedder } = makeEmbedder();
    await expect(embedder.embedQuery('text')).rejects.toThrow('Network error');
    expect(embedder.isUnavailable).toBe(true);
  });

  it('subsequent calls throw immediately without retrying', async () => {
    mockPipelineFactory.mockRejectedValueOnce(new Error('Initial failure'));
    const { embedder } = makeEmbedder();
    await expect(embedder.embedQuery('first')).rejects.toThrow('Initial failure');
    await expect(embedder.embedQuery('second')).rejects.toThrow('Embedding model is unavailable');
    expect(mockPipelineFactory).toHaveBeenCalledTimes(1);
  });
});

describe('Embedder - Text truncation', () => {
  beforeEach(() => {
    mockPipelineFn.mockClear();
    mockPipelineFactory.mockClear();
    mockPipelineFactory.mockResolvedValue(mockExtractor);
  });

  it('short text passes through unchanged', async () => {
    const { embedder } = makeEmbedder('some-other-model');
    const shortText = 'a'.repeat(100);
    await embedder.embedQuery(shortText);
    expect(mockPipelineFn).toHaveBeenCalledWith(shortText, expect.anything());
  });

  it('text longer than 2000 chars is truncated', async () => {
    const { embedder } = makeEmbedder('some-other-model');
    await embedder.embedQuery('c'.repeat(3000));
    const calledWith = mockPipelineFn.mock.calls[0][0] as string;
    expect(calledWith.length).toBe(2000);
  });
});

describe('Embedder - Return value', () => {
  beforeEach(() => {
    mockPipelineFn.mockClear();
    mockPipelineFactory.mockClear();
    mockPipelineFactory.mockResolvedValue(mockExtractor);
  });

  it('embedQuery returns number[]', async () => {
    const { embedder } = makeEmbedder();
    expect(await embedder.embedQuery('hello')).toEqual([0.1, 0.2, 0.3, 0.4, 0.5]);
  });

  it('pipeline called with pooling=mean and normalize=true', async () => {
    const { embedder } = makeEmbedder();
    await embedder.embedQuery('check');
    expect(mockPipelineFn).toHaveBeenCalledWith(expect.any(String), { pooling: 'mean', normalize: true });
  });
});

describe('Embedder - repairAndLoad()', () => {
  beforeEach(() => {
    mockPipelineFn.mockClear();
    mockPipelineFactory.mockClear();
    mockPipelineFactory.mockResolvedValue(mockExtractor);
  });

  it('resets unavailable state and loads model', async () => {
    mockPipelineFactory.mockRejectedValueOnce(new Error('Network error'));
    const { embedder } = makeEmbedder();
    await expect(embedder.embedQuery('x')).rejects.toThrow();
    expect(embedder.isUnavailable).toBe(true);

    mockPipelineFactory.mockResolvedValueOnce(mockExtractor);
    await embedder.repairAndLoad();
    expect(embedder.isUnavailable).toBe(false);
    expect(embedder.isLoaded).toBe(true);
  });

  it('on corrupted cache error, retries once', async () => {
    const { embedder } = makeEmbedder();
    mockPipelineFactory
      .mockRejectedValueOnce(new Error('Protobuf parsing failed'))
      .mockResolvedValueOnce(mockExtractor);
    await embedder.repairAndLoad();
    expect(embedder.isLoaded).toBe(true);
    expect(mockPipelineFactory).toHaveBeenCalledTimes(2);
  });

  it('propagates non-cache errors without retrying', async () => {
    const { embedder } = makeEmbedder();
    mockPipelineFactory.mockRejectedValueOnce(new Error('Out of memory'));
    await expect(embedder.repairAndLoad()).rejects.toThrow('Out of memory');
    expect(mockPipelineFactory).toHaveBeenCalledTimes(1);
  });
});
