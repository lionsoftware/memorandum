import { existsSync, readFileSync } from 'node:fs';
import { writeFile, rename, mkdir, access } from 'node:fs/promises';
import { dirname } from 'node:path';
import { LRUCache } from 'lru-cache';
import type { Logger } from 'pino';
import safeRegex from 'safe-regex2';
import type { Config, ResolvedPaths } from './config.js';
import type { FactEntry, FactMetadata, NamespaceInfo, MemoryStats } from './types.js';
import type { SemanticIndex } from './semantic-index.js';

/**
 * LRU-backed in-memory fact store with namespacing, TTL support,
 * optional semantic indexing, and JSON file persistence.
 */
export class MemoryStore {
  private readonly cache: LRUCache<string, FactEntry>;
  private readonly config: Config;
  private readonly storagePath: string;
  private readonly logger: Logger;
  private semanticIndex: SemanticIndex | null = null;
  private dirty = false;
  lastSavedAt: number | null = null;

  constructor(config: Config, paths: ResolvedPaths, logger: Logger) {
    this.config = config;
    this.storagePath = paths.factsPath;
    this.logger = logger;
    this.cache = new LRUCache<string, FactEntry>({
      max: config.max_entries,
      noDisposeOnSet: true,
      dispose: (_value: FactEntry, key: string) => {
        const parsed = MemoryStore.parseKey(key);
        this.logger.warn(
          { key: parsed.key, namespace: parsed.namespace },
          'Fact evicted from memory (LRU)',
        );
        if (this.semanticIndex) {
          this.semanticIndex.removeFact(parsed.key, parsed.namespace);
        }
      },
    });
  }

  /** Attaches a semantic index for embedding-based fact retrieval. */
  setSemanticIndex(index: SemanticIndex): void {
    this.semanticIndex = index;
  }

  /**
   * Creates a composite cache key by joining namespace and key with a null byte.
   * @param namespace - The namespace portion of the key.
   * @param key - The fact key within the namespace.
   * @returns The composite key string.
   */
  static makeKey(namespace: string, key: string): string {
    return `${namespace}\0${key}`;
  }

  /**
   * Splits a composite cache key back into its namespace and key parts.
   * Falls back to the "default" namespace when no separator is found.
   * @param compositeKey - The composite key to parse.
   * @returns An object containing the namespace and key.
   */
  static parseKey(compositeKey: string): { namespace: string; key: string } {
    const idx = compositeKey.indexOf('\0');
    if (idx === -1) return { namespace: 'default', key: compositeKey };
    return { namespace: compositeKey.slice(0, idx), key: compositeKey.slice(idx + 1) };
  }

  private getRemainingTtlSeconds(compositeKey: string): number | null {
    const ttlMs = this.cache.getRemainingTTL(compositeKey);
    const hasTtl = ttlMs > 0 && ttlMs !== Infinity;
    return hasTtl ? Math.ceil(ttlMs / 1000) : null;
  }

  /**
   * Writes or updates a fact in the store.
   * @param key - Fact identifier.
   * @param value - The fact payload.
   * @param namespace - Namespace to store the fact under.
   * @param ttlSeconds - Optional time-to-live in seconds.
   * @returns An object indicating whether a new fact was created.
   */
  write(key: string, value: unknown, namespace: string, ttlSeconds?: number): { created: boolean } {
    const compositeKey = MemoryStore.makeKey(namespace, key);
    const existing = this.cache.get(compositeKey);
    const now = Date.now();

    const entry: FactEntry = {
      value,
      namespace,
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    };

    const options: LRUCache.SetOptions<string, FactEntry, unknown> | undefined =
      ttlSeconds != null ? { ttl: ttlSeconds * 1000 } : undefined;

    this.cache.set(compositeKey, entry, options);
    this.dirty = true;

    const created = existing == null;
    this.logger.debug({ key, namespace, created }, created ? 'Fact created' : 'Fact updated');

    if (this.semanticIndex) {
      this.semanticIndex.enqueueFact(key, namespace, value);
    }

    return { created };
  }

  /**
   * Reads a single fact by key and namespace.
   * @param key - Fact identifier.
   * @param namespace - Namespace the fact belongs to.
   * @returns The fact metadata including TTL info, or null if not found.
   */
  read(key: string, namespace: string): FactMetadata | null {
    const compositeKey = MemoryStore.makeKey(namespace, key);
    const entry = this.cache.get(compositeKey);
    if (entry == null) return null;

    const ttlSeconds = this.getRemainingTtlSeconds(compositeKey);
    return {
      key, namespace,
      value: entry.value,
      createdAt: entry.createdAt,
      updatedAt: entry.updatedAt,
      ttl: ttlSeconds,
      remainingTtl: ttlSeconds,
    };
  }

  /**
   * Lists facts with optional namespace filtering, glob pattern matching, and pagination.
   * Can optionally include values and store-level statistics.
   */
  list(options: {
    namespace?: string;
    pattern?: string;
    limit: number;
    includeValues: boolean;
    includeStats: boolean;
  }): { facts: Array<Record<string, unknown>>; stats?: MemoryStats } {
    const results: Array<Record<string, unknown>> = [];
    let patternRegex: RegExp | null = null;

    if (options.pattern) {
      const escaped = options.pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&');
      const regexStr = escaped.replace(/\*/g, '.*').replace(/\?/g, '.');
      if (!safeRegex(`^${regexStr}$`)) {
        throw new Error(`Unsafe pattern (potential ReDoS): ${options.pattern}`);
      }
      patternRegex = new RegExp(`^${regexStr}$`);
    }

    for (const [compositeKey, entry] of this.cache.entries()) {
      if (results.length >= options.limit) break;
      const parsed = MemoryStore.parseKey(compositeKey);
      if (options.namespace && parsed.namespace !== options.namespace) continue;
      if (patternRegex && !patternRegex.test(parsed.key)) continue;

      const ttlSeconds = this.getRemainingTtlSeconds(compositeKey);
      const fact: Record<string, unknown> = {
        key: parsed.key, namespace: parsed.namespace,
        createdAt: entry.createdAt, updatedAt: entry.updatedAt,
        ttl: ttlSeconds, remainingTtl: ttlSeconds,
      };
      if (options.includeValues) fact.value = entry.value;
      results.push(fact);
    }

    const response: { facts: Array<Record<string, unknown>>; stats?: MemoryStats } = { facts: results };
    if (options.includeStats) {
      response.stats = {
        totalFacts: this.cache.size,
        maxEntries: this.config.max_entries,
        namespaceCount: this.getNamespaces().length,
        lastSavedAt: this.lastSavedAt,
      };
    }
    return response;
  }

  /** Returns all namespaces with their respective fact counts. */
  getNamespaces(): NamespaceInfo[] {
    const counts = new Map<string, number>();
    for (const compositeKey of this.cache.keys()) {
      const { namespace } = MemoryStore.parseKey(compositeKey);
      counts.set(namespace, (counts.get(namespace) ?? 0) + 1);
    }
    return Array.from(counts.entries()).map(([namespace, count]) => ({ namespace, count }));
  }

  /**
   * Searches facts by regex pattern, matching against both keys and values.
   * Validates the regex for safety against ReDoS before executing.
   */
  search(options: {
    query: string;
    namespace?: string;
    limit: number;
  }): { results: Array<Record<string, unknown>>; total_matches: number; limit: number } {
    if (!safeRegex(options.query)) {
      throw new Error(`Unsafe regexp (potential ReDoS): ${options.query}`);
    }
    let regex: RegExp;
    try {
      regex = new RegExp(options.query, 'i');
    } catch {
      throw new Error(`Invalid regexp: ${options.query}`);
    }

    const results: Array<Record<string, unknown>> = [];
    let totalMatches = 0;

    for (const [compositeKey, entry] of this.cache.entries()) {
      const parsed = MemoryStore.parseKey(compositeKey);
      if (options.namespace && parsed.namespace !== options.namespace) continue;

      const keyMatch = regex.test(parsed.key);
      const valueStr = JSON.stringify(entry.value);
      const valueMatch = regex.test(valueStr);
      if (!keyMatch && !valueMatch) continue;

      totalMatches++;
      if (results.length < options.limit) {
        const ttlSeconds = this.getRemainingTtlSeconds(compositeKey);
        results.push({
          key: parsed.key, namespace: parsed.namespace,
          value: entry.value,
          createdAt: entry.createdAt, updatedAt: entry.updatedAt,
          ttl: ttlSeconds, remainingTtl: ttlSeconds,
          match_in: keyMatch && valueMatch ? 'both' : keyMatch ? 'key' : 'value',
        });
      }
    }

    return { results, total_matches: totalMatches, limit: options.limit };
  }

  get size(): number { return this.cache.size; }
  get maxEntries(): number { return this.config.max_entries; }

  /**
   * Deletes a single fact by key and namespace.
   * @param key - Fact identifier.
   * @param namespace - Namespace the fact belongs to.
   * @returns True if the fact existed and was deleted.
   */
  delete(key: string, namespace: string): boolean {
    const compositeKey = MemoryStore.makeKey(namespace, key);
    const existed = this.cache.has(compositeKey);
    if (existed) {
      this.cache.delete(compositeKey);
      this.dirty = true;
    }
    return existed;
  }

  /**
   * Deletes all facts within the given namespace.
   * @param namespace - The namespace to clear.
   * @returns The number of facts deleted.
   */
  deleteNamespace(namespace: string): number {
    const keysToDelete: string[] = [];
    for (const compositeKey of this.cache.keys()) {
      if (MemoryStore.parseKey(compositeKey).namespace === namespace) {
        keysToDelete.push(compositeKey);
      }
    }
    for (const key of keysToDelete) this.cache.delete(key);
    if (keysToDelete.length > 0) this.dirty = true;
    return keysToDelete.length;
  }

  /**
   * Persists the cache to disk using atomic write (temp file + rename).
   * Skips saving when no changes have been made or when the cache is empty
   * but the file on disk still contains data (safety guard).
   * @returns True if the save was performed successfully.
   */
  async save(): Promise<boolean> {
    if (!this.dirty) return false;

    const filePath = this.storagePath;

    if (this.cache.size === 0 && existsSync(filePath)) {
      try {
        const diskRaw = readFileSync(filePath, 'utf-8');
        const diskData = JSON.parse(diskRaw) as unknown[];
        if (Array.isArray(diskData) && diskData.length > 0) {
          this.logger.warn({ diskEntries: diskData.length }, 'Skipping save: disk has data but cache is empty');
          return false;
        }
      } catch { /* proceed */ }
    }

    try {
      const dir = dirname(filePath);
      try { await access(dir); } catch { await mkdir(dir, { recursive: true }); }
      const data = JSON.stringify(this.cache.dump());
      const tmpPath = `${filePath}.${process.pid}-${Date.now()}.tmp`;
      await writeFile(tmpPath, data, 'utf-8');
      await rename(tmpPath, filePath);
      this.dirty = false;
      this.lastSavedAt = Date.now();
      this.logger.debug({ path: filePath, facts: this.cache.size }, 'Memory saved to disk');
      return true;
    } catch (err) {
      this.logger.error({ path: filePath, error: err instanceof Error ? err.message : String(err) }, 'Failed to save memory');
      return false;
    }
  }

  /** Loads facts from the on-disk JSON file into the cache. Starts empty if the file is missing or corrupt. */
  async load(): Promise<void> {
    const filePath = this.storagePath;
    try {
      if (!existsSync(filePath)) {
        this.logger.info({ path: filePath }, 'No memory file found, starting empty');
        return;
      }
      const raw = readFileSync(filePath, 'utf-8');
      const data = JSON.parse(raw) as Parameters<LRUCache<string, FactEntry>['load']>[0];
      this.cache.load(data);
      this.dirty = false;
      this.logger.debug({ path: filePath, facts: this.cache.size }, 'Memory loaded from disk');
    } catch (err) {
      this.logger.warn(
        { path: filePath, error: err instanceof Error ? err.message : String(err) },
        'Failed to load memory, starting empty',
      );
    }
  }

  private autosaveTimer: ReturnType<typeof setInterval> | null = null;

  /**
   * Starts a recurring autosave timer that persists dirty state to disk.
   * @param intervalMs - Interval between saves in milliseconds. Values <= 0 are ignored.
   */
  startAutosave(intervalMs: number): void {
    this.stopAutosave();
    if (intervalMs <= 0) return;
    this.autosaveTimer = setInterval(() => {
      this.save().catch((err: unknown) => {
        this.logger.warn({ error: err instanceof Error ? err.message : String(err) }, 'Autosave failed');
      });
    }, intervalMs);
    this.autosaveTimer.unref();
  }

  /** Stops the autosave timer if one is running. */
  stopAutosave(): void {
    if (this.autosaveTimer) {
      clearInterval(this.autosaveTimer);
      this.autosaveTimer = null;
    }
  }

  /**
   * Exports all facts as a versioned JSON-serializable object.
   * Includes TTL remaining at the time of export.
   * @returns An export payload with version, timestamp, count, and fact data.
   */
  export(): Record<string, unknown> {
    const data: Array<Record<string, unknown>> = [];
    for (const [compositeKey, entry] of this.cache.entries()) {
      const parsed = MemoryStore.parseKey(compositeKey);
      data.push({
        key: parsed.key, namespace: parsed.namespace,
        value: entry.value,
        createdAt: entry.createdAt, updatedAt: entry.updatedAt,
        ttl_seconds: this.getRemainingTtlSeconds(compositeKey),
      });
    }
    return { version: 1, exported_at: Date.now(), facts_count: data.length, data };
  }

  /**
   * Imports facts from a JSON string previously produced by {@link export}.
   * @param dataStr - JSON string containing the export payload.
   * @param merge - When true, merges into existing data; when false, replaces all facts.
   * @returns Summary with imported and overwritten counts.
   */
  import(dataStr: string, merge: boolean): Record<string, unknown> {
    let parsed: { version?: number; data?: Array<Record<string, unknown>> };
    try {
      parsed = JSON.parse(dataStr) as typeof parsed;
    } catch {
      throw new Error('Invalid import data: not valid JSON');
    }
    if (parsed.version !== 1) throw new Error(`Unsupported import format version: ${String(parsed.version)}`);
    if (!Array.isArray(parsed.data)) throw new Error('Invalid import data: missing data array');

    if (!merge) this.cache.clear();
    this.dirty = true;
    let importedCount = 0;
    let overwrittenCount = 0;

    for (const item of parsed.data) {
      const key = item.key as string;
      const namespace = item.namespace as string;
      if (!key || !namespace) continue;

      const compositeKey = MemoryStore.makeKey(namespace, key);
      if (this.cache.has(compositeKey)) overwrittenCount++;

      const now = Date.now();
      const entry: FactEntry = {
        value: item.value,
        namespace,
        createdAt: (item.createdAt as number) ?? now,
        updatedAt: (item.updatedAt as number) ?? (item.createdAt as number) ?? now,
      };

      const ttlSeconds = item.ttl_seconds as number | null | undefined;
      const options: LRUCache.SetOptions<string, FactEntry, unknown> | undefined =
        ttlSeconds != null && ttlSeconds > 0 ? { ttl: ttlSeconds * 1000 } : undefined;

      this.cache.set(compositeKey, entry, options);
      importedCount++;
    }

    return { success: true, imported_count: importedCount, overwritten_count: overwrittenCount };
  }
}
