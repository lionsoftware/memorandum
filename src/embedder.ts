import { existsSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import type { Logger } from 'pino';
import type { FeatureExtractionPipeline } from '@huggingface/transformers';

/**
 * Manages loading and running a HuggingFace feature-extraction model
 * to produce vector embeddings from text.
 */
export class Embedder {
  private static readonly MAX_INPUT_CHARS = 2000;

  private readonly modelId: string;
  private readonly dtype: string;
  private readonly logger: Logger;

  private pipeline: FeatureExtractionPipeline | null = null;
  private loading: Promise<FeatureExtractionPipeline> | null = null;
  private unavailable = false;
  private failureReason: string | null = null;

  /**
   * @param modelId - HuggingFace model identifier (e.g. "intfloat/multilingual-e5-small").
   * @param dtype - Data type for the model weights (e.g. "fp32", "q8").
   * @param logger - Pino logger instance.
   */
  constructor(modelId: string, dtype: string, logger: Logger) {
    this.modelId = modelId;
    this.dtype = dtype;
    this.logger = logger;
  }

  /** Whether the embedding pipeline has been successfully loaded. */
  get isLoaded(): boolean { return this.pipeline != null; }
  /** Whether the embedding pipeline is currently being loaded. */
  get isLoading(): boolean { return this.loading != null; }
  /** Whether the model failed to load and is permanently unavailable. */
  get isUnavailable(): boolean { return this.unavailable; }
  /** The error message explaining why the model is unavailable, or null. */
  get unavailableReason(): string | null { return this.failureReason; }

  /**
   * Resets internal state, loads the pipeline, and retries after clearing
   * the model cache if a corrupted-cache error is detected.
   */
  async repairAndLoad(): Promise<void> {
    this.pipeline = null;
    this.loading = null;
    this.unavailable = false;
    this.failureReason = null;

    try {
      await this.getPipeline();
    } catch (err) {
      if (!this.isCorruptedCacheError(err)) throw err;

      this.logger.warn(
        { model: this.modelId, error: err instanceof Error ? err.message : String(err) },
        'Corrupted model cache detected, deleting cache and retrying',
      );
      await this.deleteModelCache();

      this.pipeline = null;
      this.loading = null;
      this.unavailable = false;
      this.failureReason = null;

      await this.getPipeline();
    }
  }

  /**
   * Embeds a search query, applying E5 "query:" prefix when applicable.
   * @param text - The query text to embed.
   * @returns The embedding vector.
   */
  async embedQuery(text: string): Promise<number[]> {
    return this.embed(this.formatQuery(text));
  }

  /**
   * Embeds a passage of text, applying E5 "passage:" prefix when applicable.
   * @param text - The passage text to embed.
   * @returns The embedding vector.
   */
  async embedPassage(text: string): Promise<number[]> {
    return this.embed(this.formatPassage(text));
  }

  private get isE5Model(): boolean {
    return this.modelId.toLowerCase().includes('e5');
  }

  private formatQuery(text: string): string {
    return this.isE5Model ? `query: ${text}` : text;
  }

  private formatPassage(text: string): string {
    return this.isE5Model ? `passage: ${text}` : text;
  }

  private truncateText(text: string): string {
    if (text.length <= Embedder.MAX_INPUT_CHARS) return text;
    return text.slice(0, Embedder.MAX_INPUT_CHARS);
  }

  private async embed(text: string): Promise<number[]> {
    const pipe = await this.getPipeline();
    const truncated = this.truncateText(text);
    const output = await pipe(truncated, { pooling: 'mean', normalize: true });
    return output.tolist()[0] as number[];
  }

  private async getPipeline(): Promise<FeatureExtractionPipeline> {
    if (this.pipeline) return this.pipeline;
    if (this.unavailable) throw new Error('Embedding model is unavailable');
    if (this.loading) return this.loading;

    this.loading = this.initPipeline();
    try {
      this.pipeline = await this.loading;
      return this.pipeline;
    } catch (err) {
      this.failureReason = err instanceof Error ? err.message : String(err);
      this.unavailable = true;
      this.logger.error({ model: this.modelId, error: this.failureReason }, 'Embedding model failed to load');
      throw err;
    } finally {
      this.loading = null;
    }
  }

  private isCorruptedCacheError(err: unknown): boolean {
    const msg = err instanceof Error ? err.message : String(err);
    return msg.includes('Protobuf parsing failed') || msg.includes('invalid model');
  }

  private async deleteModelCache(): Promise<void> {
    try {
      const { env } = await import('@huggingface/transformers');
      const cacheBase = String(env.cacheDir || '');
      if (!cacheBase) return;

      const modelCacheDir = join(cacheBase, this.modelId);
      if (existsSync(modelCacheDir)) {
        rmSync(modelCacheDir, { recursive: true, force: true });
        this.logger.info({ path: modelCacheDir }, 'Deleted corrupted model cache');
      }
    } catch (deleteErr) {
      this.logger.warn(
        { error: deleteErr instanceof Error ? deleteErr.message : String(deleteErr) },
        'Failed to delete model cache',
      );
    }
  }

  private async initPipeline(): Promise<FeatureExtractionPipeline> {
    const { pipeline, env } = await import('@huggingface/transformers');

    env.allowRemoteModels = true;

    const extractor = await pipeline('feature-extraction', this.modelId, {
      dtype: this.dtype as 'auto' | 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'bnb4' | 'q4f16',
      device: 'cpu',
      progress_callback: (progress: { status: string; file?: string; progress?: number }) => {
        if (progress.status === 'downloading' && progress.file) {
          this.logger.info(
            { file: progress.file, progress: progress.progress?.toFixed(1) },
            'Downloading embedding model',
          );
        }
      },
    });

    this.logger.info({ model: this.modelId, dtype: this.dtype }, 'Embedding model loaded');
    return extractor as FeatureExtractionPipeline;
  }
}
