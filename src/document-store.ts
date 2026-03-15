import {
  existsSync, mkdirSync, readFileSync, writeFileSync,
  renameSync, unlinkSync, readdirSync, statSync, accessSync,
  constants as fsConstants,
} from 'node:fs';
import { join, dirname, resolve, basename, parse as pathParse } from 'node:path';
import { createHash } from 'node:crypto';
import { stringify as yamlStringify, parse as yamlParse } from 'yaml';
import type { Logger } from 'pino';
import type { Config, ResolvedPaths } from './config.js';
import type { IndexFile, DocumentMetadata, FileValidationResult } from './document-types.js';
import { isInlineContentType, RESERVED_FIELDS, DEFAULT_LIST_LIMIT, mimeTypeFromExtension } from './document-types.js';
import { MemorandumError } from './errors.js';
import type { SemanticIndex } from './semantic-index.js';

function emptyIndex(): IndexFile {
  return { version: 1, next_id: 1, documents: [] };
}

/**
 * Manages document storage, indexing, and CRUD operations.
 * Supports inline (markdown with frontmatter) and binary (blob with YAML sidecar) documents.
 */
export class DocumentStore {
  private readonly config: Config;
  private readonly logger: Logger;
  private readonly documentsPath: string;
  private readonly blobsPath: string;
  private readonly indexPath: string;
  private index: IndexFile;
  private semanticIndex: SemanticIndex | null = null;
  private dirty = false;

  constructor(config: Config, paths: ResolvedPaths, logger: Logger) {
    this.config = config;
    this.logger = logger;
    this.documentsPath = paths.documentsPath;
    this.blobsPath = join(paths.documentsPath, 'blobs');
    this.indexPath = join(paths.documentsPath, '_index.yaml');
    this.index = emptyIndex();
  }

  /** Assigns the semantic index used for embedding-based search. */
  setSemanticIndex(index: SemanticIndex): void {
    this.semanticIndex = index;
  }

  // ============================================================================
  // Directory Management
  // ============================================================================

  /** Creates the documents and blobs directories if they do not already exist. */
  public ensureDirectories(): void {
    if (!existsSync(this.documentsPath)) {
      mkdirSync(this.documentsPath, { recursive: true });
    }
    if (!existsSync(this.blobsPath)) {
      mkdirSync(this.blobsPath, { recursive: true });
    }
  }

  // ============================================================================
  // Index Load / Save / Rebuild
  // ============================================================================

  /** Loads the document index from disk, rebuilding it if the file is missing or corrupt. */
  public loadIndex(): void {
    this.ensureDirectories();

    if (!existsSync(this.indexPath)) {
      this.logger.info({ path: this.indexPath }, 'Index file not found, rebuilding');
      this.rebuildIndex();
      return;
    }

    try {
      const raw = readFileSync(this.indexPath, 'utf-8');
      const parsed = yamlParse(raw) as IndexFile;
      if (
        typeof parsed !== 'object' || parsed === null ||
        typeof parsed.version !== 'number' ||
        typeof parsed.next_id !== 'number' ||
        !Array.isArray(parsed.documents)
      ) {
        throw new Error('Invalid index structure');
      }
      this.index = parsed;
      this.dirty = false;
      this.logger.debug({ path: this.indexPath, documents: this.index.documents.length }, 'Index loaded');
    } catch (err) {
      this.logger.warn(
        { path: this.indexPath, error: err instanceof Error ? err.message : String(err) },
        'Failed to parse index file, rebuilding',
      );
      this.rebuildIndex();
    }
  }

  /**
   * Persists the in-memory index to disk if it has been modified.
   * @returns `true` if the index was written, `false` if no save was needed.
   */
  public saveIndex(): boolean {
    if (!this.dirty) return false;

    if (this.index.documents.length === 0 && this.index.next_id <= 1 && existsSync(this.indexPath)) {
      try {
        const diskRaw = readFileSync(this.indexPath, 'utf-8');
        const diskIndex = yamlParse(diskRaw) as IndexFile;
        if (diskIndex && typeof diskIndex === 'object' && diskIndex.next_id > 1) {
          this.logger.warn('Skipping index save: disk has newer data');
          return false;
        }
      } catch { /* proceed */ }
    }

    const yaml = yamlStringify(this.index);
    this.atomicWriteFile(this.indexPath, yaml);
    this.dirty = false;
    return true;
  }

  /** Reconstructs the index by scanning all document files on disk. */
  public rebuildIndex(): void {
    this.logger.warn({ path: this.documentsPath }, 'Rebuilding document index from files');
    const newIndex: IndexFile = { version: 1, next_id: 1, documents: [] };

    if (!existsSync(this.documentsPath)) {
      this.index = newIndex;
      return;
    }

    const files = readdirSync(this.documentsPath);
    let maxId = 0;
    const SKIP_ENTRIES = new Set(['_index.yaml', 'blobs']);

    for (const file of files) {
      if (SKIP_ENTRIES.has(file) || file.endsWith('.tmp')) continue;
      const filePath = join(this.documentsPath, file);
      try {
        if (statSync(filePath).isDirectory()) continue;
      } catch { continue; }

      try {
        if (file.endsWith('.md')) {
          const content = readFileSync(filePath, 'utf-8');
          const { metadata } = this.parseFrontmatter(content);
          if (metadata.id && typeof metadata.id === 'string') {
            const doc: DocumentMetadata = {
              id: metadata.id as string,
              title: (metadata.title as string) ?? file,
              content_type: (metadata.content_type as string) ?? 'text/plain',
              tags: Array.isArray(metadata.tags) ? (metadata.tags as string[]) : [],
              created_at: (metadata.created_at as string) ?? new Date().toISOString(),
              updated_at: (metadata.updated_at as string) ?? new Date().toISOString(),
            };
            for (const [key, value] of Object.entries(metadata)) {
              if (!(key in doc)) doc[key] = value;
            }
            newIndex.documents.push(doc);
            const numPart = parseInt(doc.id.replace('doc-', ''), 10);
            if (!isNaN(numPart) && numPart > maxId) maxId = numPart;
          }
        } else if (file.endsWith('.yaml') && file !== '_index.yaml') {
          const content = readFileSync(filePath, 'utf-8');
          const metadata = yamlParse(content) as Record<string, unknown>;
          if (metadata && typeof metadata === 'object' && metadata.id && typeof metadata.id === 'string') {
            const doc: DocumentMetadata = {
              id: metadata.id as string,
              title: (metadata.title as string) ?? file,
              content_type: (metadata.content_type as string) ?? 'application/octet-stream',
              tags: Array.isArray(metadata.tags) ? (metadata.tags as string[]) : [],
              created_at: (metadata.created_at as string) ?? new Date().toISOString(),
              updated_at: (metadata.updated_at as string) ?? new Date().toISOString(),
            };
            for (const [key, value] of Object.entries(metadata)) {
              if (!(key in doc)) doc[key] = value;
            }
            newIndex.documents.push(doc);
            const numPart = parseInt(doc.id.replace('doc-', ''), 10);
            if (!isNaN(numPart) && numPart > maxId) maxId = numPart;
          }
        }
      } catch (err) {
        this.logger.warn(
          { file, error: err instanceof Error ? err.message : String(err) },
          'Failed to parse document during rebuild, skipping',
        );
      }
    }

    newIndex.next_id = maxId + 1;
    this.index = newIndex;
    this.dirty = true;
    this.saveIndex();
    this.logger.info({ documents: newIndex.documents.length, next_id: newIndex.next_id }, 'Index rebuilt');
  }

  // ============================================================================
  // Frontmatter Helpers
  // ============================================================================

  /**
   * Extracts YAML frontmatter and body from a markdown string.
   * @param content - Raw file content with optional `---` delimited frontmatter.
   * @returns Parsed metadata object and the remaining body text.
   */
  public parseFrontmatter(content: string): { metadata: Record<string, unknown>; body: string } {
    if (!content.startsWith('---\n')) return { metadata: {}, body: content };
    const closingIndex = content.indexOf('\n---\n', 4);
    if (closingIndex === -1) return { metadata: {}, body: content };

    const yamlBlock = content.slice(4, closingIndex);
    const body = content.slice(closingIndex + 5);

    try {
      const parsed = yamlParse(yamlBlock) as Record<string, unknown>;
      return { metadata: typeof parsed === 'object' && parsed !== null ? parsed : {}, body };
    } catch {
      return { metadata: {}, body: content };
    }
  }

  /**
   * Combines metadata and body into a markdown string with YAML frontmatter.
   * @param metadata - Key-value pairs to serialize as frontmatter.
   * @param body - The document body text.
   * @returns Formatted string with `---` delimited frontmatter followed by the body.
   */
  public serializeFrontmatter(metadata: Record<string, unknown>, body: string): string {
    const yaml = yamlStringify(metadata);
    return `---\n${yaml}---\n${body}`;
  }

  // ============================================================================
  // Atomic File Write
  // ============================================================================

  /**
   * Writes a file atomically by writing to a temporary file first, then renaming.
   * @param filePath - Destination file path.
   * @param content - String or Buffer content to write.
   */
  public atomicWriteFile(filePath: string, content: string | Buffer): void {
    const dir = dirname(filePath);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    const tmpPath = `${filePath}.${process.pid}-${Date.now()}.tmp`;
    if (typeof content === 'string') {
      writeFileSync(tmpPath, content, 'utf-8');
    } else {
      writeFileSync(tmpPath, content);
    }
    renameSync(tmpPath, filePath);
  }

  // ============================================================================
  // ID Generation
  // ============================================================================

  /**
   * Generates and returns the next unique document ID (e.g., `doc-001`).
   * Increments the internal counter and marks the index as dirty.
   */
  public getNextId(): string {
    const id = this.index.next_id;
    this.index.next_id += 1;
    this.dirty = true;
    return `doc-${String(id).padStart(3, '0')}`;
  }

  // ============================================================================
  // Index Access
  // ============================================================================

  /** Returns the current in-memory index. */
  public getIndex(): IndexFile { return this.index; }

  /** Replaces the in-memory index with the provided one. */
  public setIndex(index: IndexFile): void { this.index = index; }

  /** Absolute path to the documents directory. */
  get documentsDir(): string { return this.documentsPath; }

  /** Absolute path to the blobs subdirectory. */
  get blobsDir(): string { return this.blobsPath; }

  /** Maximum allowed document size in bytes. */
  get maxDocumentSize(): number { return this.config.max_document_size; }

  // ============================================================================
  // Validation Helpers
  // ============================================================================

  private validateDocumentId(id: string): void {
    if (!/^doc-\d{3,}$/.test(id)) {
      throw new MemorandumError('validation_error', `Invalid document ID format: '${id}'. Expected: doc-NNN`, { id });
    }
  }

  private validateBodySize(body: string): void {
    const bodySize = Buffer.byteLength(body, 'utf-8');
    if (bodySize > this.config.max_document_size) {
      throw new MemorandumError(
        'document_too_large',
        `Document body size (${bodySize} bytes) exceeds limit of ${this.config.max_document_size} bytes.`,
      );
    }
  }

  private validateMetadataKeys(metadata: Record<string, unknown>): void {
    const conflicting = Object.keys(metadata).filter((k) =>
      RESERVED_FIELDS.includes(k as (typeof RESERVED_FIELDS)[number]),
    );
    if (conflicting.length > 0) {
      throw new MemorandumError(
        'validation_error',
        `Custom metadata keys conflict with reserved fields: ${conflicting.join(', ')}`,
        { conflicting },
      );
    }
  }

  // ============================================================================
  // File Import Helpers
  // ============================================================================

  private resolveFilePath(filePath: string, explicitContentType?: string): FileValidationResult {
    const resolvedPath = resolve(filePath);

    let stat;
    try { stat = statSync(resolvedPath); } catch {
      throw new MemorandumError('file_not_found', `File not found: '${resolvedPath}'`);
    }
    if (stat.isDirectory()) {
      throw new MemorandumError('file_is_directory', `Path is a directory: '${resolvedPath}'`);
    }
    try { accessSync(resolvedPath, fsConstants.R_OK); } catch {
      throw new MemorandumError('file_access_denied', `No read permission: '${resolvedPath}'`);
    }
    if (stat.size > this.config.max_document_size) {
      throw new MemorandumError('document_too_large', `File size (${stat.size} bytes) exceeds limit`);
    }

    const fileBasename = basename(resolvedPath);
    const parsed = pathParse(resolvedPath);
    return {
      resolvedPath,
      size: stat.size,
      contentType: explicitContentType ?? mimeTypeFromExtension(parsed.ext),
      basename: fileBasename,
      nameWithoutExt: parsed.name,
    };
  }

  private injectFileContent(
    input: { content_type?: string; body?: string; file_path?: string; metadata?: Record<string, unknown> },
    logMessage: string,
  ): FileValidationResult | undefined {
    if (input.file_path === undefined) return undefined;

    const fileInfo = this.resolveFilePath(input.file_path, input.content_type);
    input.content_type = fileInfo.contentType;

    if (isInlineContentType(fileInfo.contentType)) {
      input.body = readFileSync(fileInfo.resolvedPath, 'utf-8');
    } else {
      input.body = readFileSync(fileInfo.resolvedPath).toString('base64');
    }

    input.metadata = { ...input.metadata, source_file: fileInfo.basename };
    this.logger.debug(
      { path: fileInfo.resolvedPath, contentType: fileInfo.contentType, size: fileInfo.size },
      logMessage,
    );
    return fileInfo;
  }

  // ============================================================================
  // CRUD Methods
  // ============================================================================

  /**
   * Creates or updates a document based on the `_mode` field.
   * Handles both inline (text) and binary content, including file imports.
   * @returns The document ID and whether it was newly created.
   */
  public write(input: {
    _mode: 'create' | 'update';
    id?: string;
    title?: string;
    content_type?: string;
    topic?: string;
    description?: string;
    tags?: string[];
    body?: string;
    file_path?: string;
    metadata?: Record<string, unknown>;
  }): { id: string; created: boolean } {
    return input._mode === 'update' ? this.writeUpdate(input) : this.writeCreate(input);
  }

  private writeCreate(input: {
    title?: string;
    content_type?: string;
    topic?: string;
    description?: string;
    tags?: string[];
    body?: string;
    file_path?: string;
    metadata?: Record<string, unknown>;
  }): { id: string; created: boolean } {
    const fileInfo = this.injectFileContent(input, 'File imported for document creation');
    if (fileInfo && !input.title) input.title = fileInfo.nameWithoutExt;
    if (!input.title) throw new MemorandumError('validation_error', 'title is required for document creation');
    if (input.body !== undefined) this.validateBodySize(input.body);
    if (input.metadata) this.validateMetadataKeys(input.metadata);

    const id = this.getNextId();
    const normalizedTags = (input.tags ?? []).map((t) => t.trim().toLowerCase());
    const now = new Date().toISOString();

    const metadata: DocumentMetadata = {
      id, title: input.title,
      content_type: input.content_type ?? 'text/plain',
      tags: normalizedTags, created_at: now, updated_at: now,
    };
    if (input.topic) metadata.topic = input.topic;
    if (input.description) metadata.description = input.description;
    if (input.metadata) {
      for (const [key, value] of Object.entries(input.metadata)) {
        if (!RESERVED_FIELDS.includes(key as (typeof RESERVED_FIELDS)[number])) metadata[key] = value;
      }
    }

    if (isInlineContentType(metadata.content_type)) {
      const filePath = join(this.documentsPath, `${id}.md`);
      this.atomicWriteFile(filePath, this.serializeFrontmatter(metadata, input.body ?? ''));
    } else {
      const bodyData = input.body ? Buffer.from(input.body, 'base64') : Buffer.alloc(0);
      const ext = metadata.content_type.split('/').pop() ?? 'bin';
      const blobFileName = `${id}.${ext}`;
      this.atomicWriteFile(join(this.blobsDir, blobFileName), bodyData);

      const sha256 = createHash('sha256').update(bodyData).digest('hex');
      metadata.blob_path = `blobs/${blobFileName}`;
      metadata.blob_size = bodyData.length;
      metadata.blob_sha256 = sha256;
      this.atomicWriteFile(join(this.documentsDir, `${id}.yaml`), yamlStringify(metadata));
    }

    this.index.documents.push(metadata);
    this.dirty = true;
    this.saveIndex();

    if (this.semanticIndex) {
      this.semanticIndex.enqueueDocument(id, {
        title: metadata.title, topic: metadata.topic,
        tags: metadata.tags, contentType: metadata.content_type,
        description: typeof metadata.description === 'string' ? metadata.description : undefined,
      }, input.body);
    }

    return { id, created: true };
  }

  private writeUpdate(input: {
    id?: string;
    title?: string;
    content_type?: string;
    topic?: string;
    description?: string;
    tags?: string[];
    body?: string;
    file_path?: string;
    metadata?: Record<string, unknown>;
  }): { id: string; created: boolean } {
    this.injectFileContent(input, 'File imported for document update');
    this.validateDocumentId(input.id!);

    const existingIdx = this.index.documents.findIndex((d) => d.id === input.id);
    if (existingIdx === -1) {
      throw new MemorandumError('document_not_found', `Document '${input.id}' not found`, { id: input.id });
    }
    const existing = this.index.documents[existingIdx]!;

    if (input.body !== undefined) this.validateBodySize(input.body);
    if (input.metadata) this.validateMetadataKeys(input.metadata);

    const wasInline = isInlineContentType(existing.content_type);

    if (input.title !== undefined) existing.title = input.title;
    if (input.content_type !== undefined) existing.content_type = input.content_type;
    if (input.topic !== undefined) existing.topic = input.topic;
    if (input.description !== undefined) existing.description = input.description;
    if (input.tags !== undefined) existing.tags = input.tags.map((t) => t.trim().toLowerCase());
    existing.updated_at = new Date().toISOString();

    if (input.metadata) {
      for (const [key, value] of Object.entries(input.metadata)) {
        if (!RESERVED_FIELDS.includes(key as (typeof RESERVED_FIELDS)[number])) existing[key] = value;
      }
    }

    const isInline = isInlineContentType(existing.content_type);
    if (wasInline !== isInline) {
      if (wasInline) {
        const oldMdPath = join(this.documentsPath, `${existing.id}.md`);
        if (existsSync(oldMdPath)) unlinkSync(oldMdPath);
      } else {
        const oldSidecarPath = join(this.documentsPath, `${existing.id}.yaml`);
        if (existsSync(oldSidecarPath)) unlinkSync(oldSidecarPath);
        if (existing.blob_path) {
          const oldBlobPath = join(this.documentsPath, existing.blob_path);
          if (existsSync(oldBlobPath)) unlinkSync(oldBlobPath);
        }
        delete existing.blob_path;
        delete existing.blob_size;
        delete existing.blob_sha256;
      }
    }

    if (isInline) {
      let bodyContent = input.body;
      if (bodyContent === undefined) {
        const filePath = join(this.documentsPath, `${existing.id}.md`);
        if (existsSync(filePath)) {
          bodyContent = this.parseFrontmatter(readFileSync(filePath, 'utf-8')).body;
        } else {
          bodyContent = '';
        }
      }
      this.atomicWriteFile(
        join(this.documentsPath, `${existing.id}.md`),
        this.serializeFrontmatter(existing, bodyContent ?? ''),
      );
    } else {
      if (input.body !== undefined) {
        if (existing.blob_path) {
          const oldBlobPath = join(this.documentsPath, existing.blob_path);
          const newExt = existing.content_type.split('/').pop() ?? 'bin';
          const newBlobFileName = `${existing.id}.${newExt}`;
          if (existing.blob_path !== `blobs/${newBlobFileName}` && existsSync(oldBlobPath)) {
            unlinkSync(oldBlobPath);
          }
        }
        const bodyData = Buffer.from(input.body, 'base64');
        const ext = existing.content_type.split('/').pop() ?? 'bin';
        const blobFileName = `${existing.id}.${ext}`;
        this.atomicWriteFile(join(this.blobsDir, blobFileName), bodyData);

        const sha256 = createHash('sha256').update(bodyData).digest('hex');
        existing.blob_path = `blobs/${blobFileName}`;
        existing.blob_size = bodyData.length;
        existing.blob_sha256 = sha256;
      }
      this.atomicWriteFile(join(this.documentsDir, `${existing.id}.yaml`), yamlStringify(existing));
    }

    this.index.documents[existingIdx] = existing;
    this.dirty = true;
    this.saveIndex();

    if (this.semanticIndex) {
      let bodyForIndex: string | undefined;
      if (isInlineContentType(existing.content_type)) {
        const filePath = join(this.documentsPath, `${existing.id}.md`);
        if (existsSync(filePath)) {
          bodyForIndex = this.parseFrontmatter(readFileSync(filePath, 'utf-8')).body;
        }
      }
      this.semanticIndex.enqueueDocument(existing.id, {
        title: existing.title, topic: existing.topic,
        tags: existing.tags, contentType: existing.content_type,
        description: typeof existing.description === 'string' ? existing.description : undefined,
      }, bodyForIndex);
    }

    return { id: input.id!, created: false };
  }

  /**
   * Reads a document by ID, returning its metadata and optionally its body content.
   * @param input.id - Document ID (e.g., `doc-001`).
   * @param input.include_body - Whether to include the body content (defaults to `true`).
   * @returns Document metadata and body. Binary bodies are base64-encoded.
   */
  public read(input: { id: string; include_body?: boolean }): Record<string, unknown> {
    this.validateDocumentId(input.id);
    const doc = this.index.documents.find((d) => d.id === input.id);
    if (!doc) throw new MemorandumError('document_not_found', `Document '${input.id}' not found`, { id: input.id });

    const result: Record<string, unknown> = { ...doc };

    if (input.include_body !== false) {
      if (isInlineContentType(doc.content_type)) {
        const filePath = join(this.documentsPath, `${doc.id}.md`);
        try {
          const { body } = this.parseFrontmatter(readFileSync(filePath, 'utf-8'));
          result['body'] = body;
        } catch (err) {
          if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
            this.logger.warn({ id: doc.id }, 'Document file missing on disk');
          } else { throw err; }
        }
      } else if (doc.blob_path) {
        const blobPath = join(this.documentsPath, doc.blob_path);
        if (existsSync(blobPath)) {
          const blobData = readFileSync(blobPath);
          result['body'] = blobData.toString('base64');
          const sha256 = createHash('sha256').update(blobData).digest('hex');
          result['integrity_ok'] = sha256 === doc.blob_sha256;
        }
      }
    }

    return result;
  }

  /**
   * Lists documents matching the given filters.
   * Supports filtering by tag, topic, content type, text search, and custom metadata.
   * @returns Matching documents (up to `limit`) and the total count.
   */
  public list(input: {
    tag?: string;
    topic?: string;
    content_type?: string;
    search?: string;
    metadata?: Record<string, unknown>;
    limit?: number;
  }): { documents: Record<string, unknown>[]; total: number } {
    const limit = input.limit ?? DEFAULT_LIST_LIMIT;
    let filtered: DocumentMetadata[] = [...this.index.documents];

    if (input.tag !== undefined) filtered = filtered.filter((d) => d.tags.includes(input.tag!));
    if (input.topic !== undefined) filtered = filtered.filter((d) => d.topic === input.topic);
    if (input.content_type !== undefined) filtered = filtered.filter((d) => d.content_type === input.content_type);
    if (input.search !== undefined) {
      const needle = input.search.toLowerCase();
      filtered = filtered.filter((d) =>
        d.title.toLowerCase().includes(needle) ||
        (typeof d.description === 'string' && d.description.toLowerCase().includes(needle)),
      );
    }
    if (input.metadata !== undefined) {
      for (const [key, value] of Object.entries(input.metadata)) {
        filtered = filtered.filter((d) => {
          const docValue = d[key];
          if (typeof value !== 'object' || value === null) return docValue === value;
          return JSON.stringify(docValue) === JSON.stringify(value);
        });
      }
    }

    const total = filtered.length;
    return { documents: filtered.slice(0, limit).map((d) => ({ ...d })), total };
  }

  /**
   * Deletes a document by ID, removing its files from disk and the index entry.
   * @param input.id - Document ID to delete.
   * @returns `{ deleted: true }` if found and removed, `{ deleted: false }` if not found.
   */
  public delete(input: { id: string }): { deleted: boolean } {
    this.validateDocumentId(input.id);
    const idx = this.index.documents.findIndex((d) => d.id === input.id);
    if (idx === -1) return { deleted: false };

    const doc = this.index.documents[idx]!;

    if (isInlineContentType(doc.content_type)) {
      const filePath = join(this.documentsPath, `${doc.id}.md`);
      if (existsSync(filePath)) unlinkSync(filePath);
    } else {
      const sidecarPath = join(this.documentsPath, `${doc.id}.yaml`);
      if (existsSync(sidecarPath)) unlinkSync(sidecarPath);
      if (doc.blob_path) {
        const blobPath = join(this.documentsPath, doc.blob_path);
        if (existsSync(blobPath)) unlinkSync(blobPath);
      }
    }

    this.index.documents.splice(idx, 1);
    this.dirty = true;
    this.saveIndex();

    if (this.semanticIndex) this.semanticIndex.removeDocument(doc.id);
    return { deleted: true };
  }
}
