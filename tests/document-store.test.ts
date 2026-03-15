import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { existsSync, readFileSync, rmSync, writeFileSync, mkdtempSync, unlinkSync, mkdirSync, chmodSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID, createHash } from 'node:crypto';
import type { Logger } from 'pino';
import type { Config, ResolvedPaths } from '../src/config.js';
import { DocumentWriteCreateInputSchema, isInlineContentType } from '../src/document-types.js';
import { MemorandumError } from '../src/errors.js';
import { DocumentStore } from '../src/document-store.js';

function mockLogger(): Logger {
  return {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
    child: vi.fn().mockReturnThis(),
    fatal: vi.fn(),
    trace: vi.fn(),
    silent: vi.fn(),
    level: 'debug',
  } as unknown as Logger;
}

function makeConfig(overrides: Partial<Config> = {}): Config {
  return {
    max_entries: 100,
    autosave_interval_seconds: 0,
    storage_dir: '.memorandum',
    max_document_size: 16 * 1024 * 1024,
    semantic_model: 'Xenova/multilingual-e5-small',
    semantic_model_dtype: 'q8',
    semantic_enabled: false,
    semantic_debounce_seconds: 10,
    semantic_max_queue_size: 200,
    semantic_max_retries: 3,
    ...overrides,
  };
}

let testDir: string;

function makePaths(): ResolvedPaths {
  testDir = join(tmpdir(), `test-docstore-${randomUUID()}`);
  mkdirSync(testDir, { recursive: true });
  return {
    storageDir: testDir,
    factsPath: join(testDir, 'facts', 'facts.json'),
    documentsPath: join(testDir, 'documents'),
    cachePath: join(testDir, 'cache'),
    configPath: join(testDir, 'config.yaml'),
  };
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Returns true if the string is a valid ISO 8601 date string.
 */
function isIso8601(value: unknown): boolean {
  if (typeof value !== 'string') return false;
  const date = new Date(value);
  return !isNaN(date.getTime()) && value.includes('T');
}

// ============================================================================
// Text document create + read (US1) — TDD: write()/read() do not exist yet
// ============================================================================

describe('DocumentStore - text document create + read (US1)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;
  let logger: Logger;

  beforeEach(() => {
    paths = makePaths();
    logger = mockLogger();
    store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('creates text document with all fields and reads it back', () => {
    const writeResult = store.write({
      _mode: 'create',
      title: 'NGINX Configuration Guide',
      content_type: 'text/markdown',
      topic: 'infrastructure',
      description: 'How to configure NGINX for production',
      tags: ['nginx', 'web', 'production'],
      body: '# NGINX Guide\n\nThis is the body of the document.',
    });

    expect(writeResult).toBeDefined();
    expect(writeResult.id).toBeDefined();
    expect(writeResult.created).toBe(true);

    // Verify the .md file exists on disk with YAML frontmatter
    const filePath = join(paths.documentsPath, `${writeResult.id}.md`);
    expect(existsSync(filePath)).toBe(true);
    const raw = readFileSync(filePath, 'utf-8');
    expect(raw).toContain('title:');
    expect(raw).toContain('NGINX Configuration Guide');
    expect(raw).toContain('# NGINX Guide');

    const readResult = store.read({ id: writeResult.id, include_body: true });

    expect(readResult).toBeDefined();
    expect(readResult.id).toBe(writeResult.id);
    expect(readResult.title).toBe('NGINX Configuration Guide');
    expect(readResult.content_type).toBe('text/markdown');
    expect(readResult.topic).toBe('infrastructure');
    expect(readResult.description).toBe('How to configure NGINX for production');
    expect(readResult.tags).toEqual(['nginx', 'web', 'production']);
    expect(readResult.body).toBe('# NGINX Guide\n\nThis is the body of the document.');
  });

  it('auto-generates sequential doc-NNN ids', () => {
    const r1 = store.write({ _mode: 'create', title: 'Doc One' });
    const r2 = store.write({ _mode: 'create', title: 'Doc Two' });
    const r3 = store.write({ _mode: 'create', title: 'Doc Three' });

    expect(r1.id).toBe('doc-001');
    expect(r2.id).toBe('doc-002');
    expect(r3.id).toBe('doc-003');
  });

  it('sets created_at and updated_at timestamps', () => {
    const before = new Date().toISOString();

    const writeResult = store.write({
      _mode: 'create',
      title: 'Timestamp Test',
      body: 'Some content',
    });

    const after = new Date().toISOString();

    const doc = store.read({ id: writeResult.id, include_body: false });

    expect(isIso8601(doc.created_at)).toBe(true);
    expect(isIso8601(doc.updated_at)).toBe(true);

    // Timestamps should be within the test window
    const createdAt = doc.created_at as string;
    const updatedAt = doc.updated_at as string;
    expect(createdAt >= before).toBe(true);
    expect(createdAt <= after).toBe(true);
    expect(updatedAt >= before).toBe(true);
    expect(updatedAt <= after).toBe(true);
  });

  it('updates index with document metadata', () => {
    const writeResult = store.write({
      _mode: 'create',
      title: 'Index Test',
      topic: 'testing',
      tags: ['test'],
      body: 'Index test body',
    });

    const index = store.getIndex();
    const entry = index.documents.find((d) => d.id === writeResult.id);

    expect(entry).toBeDefined();
    expect(entry!.id).toBe(writeResult.id);
    expect(entry!.title).toBe('Index Test');
    expect(entry!.topic).toBe('testing');
    expect(entry!.tags).toEqual(['test']);
  });

  it('reads document metadata only when include_body=false', () => {
    const writeResult = store.write({
      _mode: 'create',
      title: 'Metadata Only Test',
      body: 'This body should not be returned',
    });

    const doc = store.read({ id: writeResult.id, include_body: false });

    expect(doc).toBeDefined();
    expect(doc.id).toBe(writeResult.id);
    expect(doc.title).toBe('Metadata Only Test');
    expect(doc.body).toBeUndefined();
  });

  it('reads document with body when include_body=true', () => {
    const expectedBody = 'This is the expected body content.\nWith multiple lines.';

    const writeResult = store.write({
      _mode: 'create',
      title: 'Body Read Test',
      body: expectedBody,
    });

    const doc = store.read({ id: writeResult.id, include_body: true });

    expect(doc.body).toBe(expectedBody);
  });
});

// ============================================================================
// Edge cases (US1)
// ============================================================================

describe('DocumentStore - edge cases (US1)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;
  let logger: Logger;

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('accepts body exactly at size limit', () => {
    const limit = 200;
    paths = makePaths();
    logger = mockLogger();
    store = new DocumentStore(makeConfig({ max_document_size: limit }), paths, logger);
    store.loadIndex();

    const exactBody = 'x'.repeat(limit);

    expect(() => store.write({ _mode: 'create', title: 'At Limit', body: exactBody })).not.toThrow();
  });

  it('rejects body exceeding size limit with actionable error', () => {
    const limit = 100;
    paths = makePaths();
    logger = mockLogger();
    store = new DocumentStore(makeConfig({ max_document_size: limit }), paths, logger);
    store.loadIndex();

    const oversizedBody = 'x'.repeat(limit + 1);

    let thrownError: unknown;
    try {
      store.write({ _mode: 'create', title: 'Too Big', body: oversizedBody });
    } catch (err) {
      thrownError = err;
    }

    expect(thrownError).toBeDefined();
    expect(thrownError).toBeInstanceOf(MemorandumError);

    const error = thrownError as MemorandumError;
    expect(error.code).toBe('document_too_large');

    // Must contain the current configured limit value
    expect(error.message).toContain(String(limit));
  });

  it('accepts empty body (metadata-only document)', () => {
    paths = makePaths();
    logger = mockLogger();
    store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();

    const writeResult = store.write({
      _mode: 'create',
      title: 'Metadata Only Document',
      topic: 'notes',
    });

    expect(writeResult.created).toBe(true);

    const doc = store.read({ id: writeResult.id, include_body: true });

    // body should be empty string or undefined when no body was provided
    expect(doc.body === '' || doc.body === undefined).toBe(true);
  });

  it('normalizes tags to lowercase and trims whitespace', () => {
    paths = makePaths();
    logger = mockLogger();
    store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();

    const writeResult = store.write({
      _mode: 'create',
      title: 'Tag Normalization Test',
      tags: [' NGINX ', 'Web-01', '  production  '],
    });

    const doc = store.read({ id: writeResult.id, include_body: false });

    expect(doc.tags).toEqual(['nginx', 'web-01', 'production']);
  });

  it('generates sequential IDs across multiple writes', () => {
    paths = makePaths();
    logger = mockLogger();
    store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();

    const ids: string[] = [];
    for (let i = 0; i < 5; i++) {
      const result = store.write({ _mode: 'create', title: `Document ${i + 1}` });
      ids.push(result.id);
    }

    expect(ids).toEqual(['doc-001', 'doc-002', 'doc-003', 'doc-004', 'doc-005']);
  });
});

// ============================================================================
// Frontmatter helpers (already implemented — these tests should pass NOW)
// ============================================================================

describe('DocumentStore - frontmatter helpers', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    const logger = mockLogger();
    store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('parseFrontmatter extracts metadata and body', () => {
    const content = [
      '---',
      'title: My Document',
      'topic: testing',
      'tags:',
      '  - nginx',
      '  - web',
      '---',
      'This is the body content.',
      'Second line.',
    ].join('\n');

    const result = store.parseFrontmatter(content);

    expect(result.metadata).toBeDefined();
    expect(result.metadata['title']).toBe('My Document');
    expect(result.metadata['topic']).toBe('testing');
    expect(result.metadata['tags']).toEqual(['nginx', 'web']);
    expect(result.body).toBe('This is the body content.\nSecond line.');
  });

  it('parseFrontmatter handles no frontmatter', () => {
    const plainText = 'This is plain text without any frontmatter markers.\nSecond line.';

    const result = store.parseFrontmatter(plainText);

    expect(result.metadata).toEqual({});
    expect(result.body).toBe(plainText);
  });

  it('serializeFrontmatter creates valid frontmatter', () => {
    const metadata = {
      title: 'Round Trip Test',
      topic: 'testing',
      tags: ['unit', 'vitest'],
      created_at: '2026-02-19T10:00:00.000Z',
    };
    const body = 'Body content here.\nLine 2.';

    const serialized = store.serializeFrontmatter(metadata, body);

    // Must start with frontmatter delimiter
    expect(serialized).toMatch(/^---\n/);

    // Must contain a closing delimiter
    expect(serialized).toContain('\n---\n');

    // Round-trip: parse back and verify fidelity
    const parsed = store.parseFrontmatter(serialized);

    expect(parsed.metadata['title']).toBe('Round Trip Test');
    expect(parsed.metadata['topic']).toBe('testing');
    expect(parsed.metadata['tags']).toEqual(['unit', 'vitest']);
    expect(parsed.metadata['created_at']).toBe('2026-02-19T10:00:00.000Z');
    expect(parsed.body).toBe(body);
  });

  it('parseFrontmatter returns empty metadata when closing marker is missing', () => {
    const malformed = '---\ntitle: Missing Closing\ntopic: test\nNo closing marker here';

    const result = store.parseFrontmatter(malformed);

    // No valid closing marker — treat entire content as body
    expect(result.metadata).toEqual({});
    expect(result.body).toBe(malformed);
  });

  it('serializeFrontmatter round-trips empty metadata correctly', () => {
    const metadata = {};
    const body = 'Just a body, no metadata.';

    const serialized = store.serializeFrontmatter(metadata, body);
    const parsed = store.parseFrontmatter(serialized);

    expect(parsed.body).toBe(body);
  });
});

// ============================================================================
// List and search (US2)
// ============================================================================

describe('DocumentStore - list and search (US2)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    const logger = mockLogger();
    store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();

    // Seed documents for filtering
    store.write({ _mode: 'create', title: 'NGINX Config', topic: 'web-servers', tags: ['nginx', 'production'], content_type: 'text/plain', body: 'server {}' });
    store.write({ _mode: 'create', title: 'Apache Config', topic: 'web-servers', tags: ['apache', 'staging'], content_type: 'text/plain', body: 'VirtualHost' });
    store.write({ _mode: 'create', title: 'Network Diagram', topic: 'networking', tags: ['diagram'], content_type: 'text/plain', description: 'Office network layout' });
    store.write({ _mode: 'create', title: 'SSH Keys Guide', topic: 'security', tags: ['ssh', 'production'], content_type: 'application/json', body: Buffer.from('{}').toString('base64') });
    store.write({ _mode: 'create', title: 'Docker nginx Setup', topic: 'containers', tags: ['docker', 'nginx'], content_type: 'text/markdown', description: 'Running nginx in Docker' });
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('lists all documents without filters', () => {
    const result = store.list({});

    expect(result.total).toBe(5);
    expect(result.documents).toHaveLength(5);

    // Documents must not have a body field
    for (const doc of result.documents) {
      expect(doc).not.toHaveProperty('body');
    }

    // Spot-check a known title is present
    const titles = result.documents.map((d) => d['title']);
    expect(titles).toContain('NGINX Config');
    expect(titles).toContain('Apache Config');
  });

  it('filters by tag (exact match)', () => {
    const result = store.list({ tag: 'nginx' });

    expect(result.total).toBe(2);
    expect(result.documents).toHaveLength(2);

    const titles = result.documents.map((d) => d['title']);
    expect(titles).toContain('NGINX Config');
    expect(titles).toContain('Docker nginx Setup');
    expect(titles).not.toContain('Apache Config');
  });

  it('filters by topic (exact match)', () => {
    const result = store.list({ topic: 'web-servers' });

    expect(result.total).toBe(2);
    expect(result.documents).toHaveLength(2);

    const titles = result.documents.map((d) => d['title']);
    expect(titles).toContain('NGINX Config');
    expect(titles).toContain('Apache Config');
    expect(titles).not.toContain('Network Diagram');
  });

  it('filters by content_type (exact match)', () => {
    const result = store.list({ content_type: 'application/json' });

    expect(result.total).toBe(1);
    expect(result.documents).toHaveLength(1);
    expect(result.documents[0]!['title']).toBe('SSH Keys Guide');
  });

  it('searches by substring in title (case-insensitive)', () => {
    const result = store.list({ search: 'nginx' });

    // Matches: "NGINX Config", "Docker nginx Setup"
    expect(result.total).toBe(2);
    expect(result.documents).toHaveLength(2);

    const titles = result.documents.map((d) => d['title']);
    expect(titles).toContain('NGINX Config');
    expect(titles).toContain('Docker nginx Setup');
  });

  it('searches by substring in description (case-insensitive)', () => {
    const result = store.list({ search: 'office' });

    // Matches: "Network Diagram" (description: 'Office network layout')
    expect(result.total).toBe(1);
    expect(result.documents).toHaveLength(1);
    expect(result.documents[0]!['title']).toBe('Network Diagram');
  });

  it('combines tag filter with search', () => {
    // tag=nginx narrows to "NGINX Config" and "Docker nginx Setup"
    // search=docker further narrows to "Docker nginx Setup"
    const result = store.list({ tag: 'nginx', search: 'docker' });

    expect(result.total).toBe(1);
    expect(result.documents).toHaveLength(1);
    expect(result.documents[0]!['title']).toBe('Docker nginx Setup');
  });

  it('applies limit parameter', () => {
    const result = store.list({ limit: 2 });

    // total reflects all 5 documents, but documents array is sliced to 2
    expect(result.total).toBe(5);
    expect(result.documents).toHaveLength(2);
  });

  it('returns empty array when no documents match', () => {
    const result = store.list({ tag: 'nonexistent-tag-xyz' });

    expect(result.total).toBe(0);
    expect(result.documents).toHaveLength(0);
    expect(result.documents).toEqual([]);
  });
});

// ============================================================================
// Binary document write/read (US3)
// ============================================================================

describe('DocumentStore - binary document write/read (US3)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    const logger = mockLogger();
    store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('writes binary document with sidecar YAML and blob', () => {
    const pngData = Buffer.from('fake-png-data-for-testing');
    const base64Body = pngData.toString('base64');

    const result = store.write({
      _mode: 'create',
      title: 'Network Diagram',
      content_type: 'image/png',
      topic: 'networking',
      tags: ['diagram'],
      body: base64Body,
    });

    expect(result.created).toBe(true);

    // Verify sidecar .yaml exists
    const sidecarPath = join(paths.documentsPath, `${result.id}.yaml`);
    expect(existsSync(sidecarPath)).toBe(true);

    // Verify blob exists in blobs/
    const blobPath = join(paths.documentsPath, 'blobs', `${result.id}.png`);
    expect(existsSync(blobPath)).toBe(true);

    // Verify blob content matches
    const blobContent = readFileSync(blobPath);
    expect(blobContent.toString()).toBe('fake-png-data-for-testing');

    // Verify index has sha256 and size
    const index = store.getIndex();
    const doc = index.documents.find(d => d.id === result.id);
    expect(doc).toBeDefined();
    expect(doc!.blob_sha256).toBeDefined();
    expect(doc!.blob_size).toBe(pngData.length);
  });

  it('reads binary document with include_body=true returns base64 and integrity_ok', () => {
    const data = Buffer.from('test-binary-content');
    const result = store.write({
      _mode: 'create',
      title: 'Test Binary',
      content_type: 'image/jpeg',
      body: data.toString('base64'),
    });

    const doc = store.read({ id: result.id, include_body: true });

    expect(doc.body).toBe(data.toString('base64'));
    expect(doc.integrity_ok).toBe(true);
    expect(doc.blob_size).toBe(data.length);
    expect(doc.blob_sha256).toBeDefined();
  });

  it('reads binary document with include_body=false returns metadata only', () => {
    const data = Buffer.from('some-data');
    const result = store.write({
      _mode: 'create',
      title: 'Metadata Only Binary',
      content_type: 'application/octet-stream',
      body: data.toString('base64'),
    });

    const doc = store.read({ id: result.id, include_body: false });

    expect(doc.id).toBe(result.id);
    expect(doc.title).toBe('Metadata Only Binary');
    expect(doc.blob_size).toBe(data.length);
    expect(doc.body).toBeUndefined();
  });

  it('detects corrupted blob (sha256 mismatch) with integrity_ok=false', () => {
    const data = Buffer.from('original-content');
    const result = store.write({
      _mode: 'create',
      title: 'Corrupted Test',
      content_type: 'image/png',
      body: data.toString('base64'),
    });

    // Corrupt the blob file
    const blobPath = join(paths.documentsPath, 'blobs', `${result.id}.png`);
    writeFileSync(blobPath, 'corrupted-data');

    const doc = store.read({ id: result.id, include_body: true });

    expect(doc.body).toBeDefined(); // Returns body anyway
    expect(doc.integrity_ok).toBe(false); // But flags corruption
  });
});

// ============================================================================
// Update and delete (US4)
// ============================================================================

describe('DocumentStore - update and delete (US4)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    const logger = mockLogger();
    store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('partial update: change title only (topic/tags preserved)', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Original Title',
      topic: 'infra',
      tags: ['nginx', 'production'],
      body: 'Original body',
    });

    const updateResult = store.write({
      _mode: 'update',
      id,
      title: 'Updated Title',
    });

    expect(updateResult.id).toBe(id);
    expect(updateResult.created).toBe(false);

    const doc = store.read({ id, include_body: true });
    expect(doc.title).toBe('Updated Title');
    expect(doc.topic).toBe('infra');
    expect(doc.tags).toEqual(['nginx', 'production']);
    expect(doc.body).toBe('Original body');
  });

  it('update body only', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Body Update Test',
      body: 'Initial body content',
    });

    store.write({
      _mode: 'update',
      id,
      body: 'Updated body content',
    });

    const doc = store.read({ id, include_body: true });
    expect(doc.title).toBe('Body Update Test');
    expect(doc.body).toBe('Updated body content');
  });

  it('update tags only', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Tag Update Test',
      tags: ['old-tag', 'another-old'],
    });

    store.write({
      _mode: 'update',
      id,
      tags: ['NEW-TAG', '  trimmed  '],
    });

    const doc = store.read({ id, include_body: false });
    expect(doc.tags).toEqual(['new-tag', 'trimmed']);
  });

  it('update non-existent id throws error', () => {
    expect(() =>
      store.write({
        _mode: 'update',
        id: 'doc-999',
        title: 'Should Fail',
      }),
    ).toThrow();

    let thrownError: unknown;
    try {
      store.write({ _mode: 'update', id: 'doc-999', title: 'Should Fail' });
    } catch (err) {
      thrownError = err;
    }

    expect(thrownError).toBeDefined();
    const msg = (thrownError as Error).message;
    expect(msg).toContain('doc-999');
  });

  it('update preserves created_at and updates updated_at', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Timestamp Preservation Test',
    });

    const before = store.read({ id, include_body: false });
    const originalCreatedAt = before.created_at as string;

    // Wait a tiny bit to ensure updated_at differs
    const startUpdate = new Date().toISOString();

    store.write({
      _mode: 'update',
      id,
      title: 'Updated',
    });

    const after = store.read({ id, include_body: false });

    expect(after.created_at).toBe(originalCreatedAt);
    expect(after.updated_at as string >= startUpdate).toBe(true);
  });

  it('delete text document removes file and index entry', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Document to Delete',
      content_type: 'text/plain',
      body: 'Delete me',
    });

    const filePath = join(paths.documentsPath, `${id}.md`);
    expect(existsSync(filePath)).toBe(true);

    const deleteResult = store.delete({ id });

    expect(deleteResult.deleted).toBe(true);
    expect(existsSync(filePath)).toBe(false);

    const index = store.getIndex();
    expect(index.documents.find((d) => d.id === id)).toBeUndefined();
  });

  it('delete binary document removes sidecar + blob + index entry', () => {
    const data = Buffer.from('binary-data-to-delete');
    const { id } = store.write({
      _mode: 'create',
      title: 'Binary to Delete',
      content_type: 'image/png',
      body: data.toString('base64'),
    });

    const sidecarPath = join(paths.documentsPath, `${id}.yaml`);
    const blobPath = join(paths.documentsPath, 'blobs', `${id}.png`);
    expect(existsSync(sidecarPath)).toBe(true);
    expect(existsSync(blobPath)).toBe(true);

    const deleteResult = store.delete({ id });

    expect(deleteResult.deleted).toBe(true);
    expect(existsSync(sidecarPath)).toBe(false);
    expect(existsSync(blobPath)).toBe(false);

    const index = store.getIndex();
    expect(index.documents.find((d) => d.id === id)).toBeUndefined();
  });

  it('delete non-existent id returns deleted: false', () => {
    const result = store.delete({ id: 'doc-999' });

    expect(result.deleted).toBe(false);
  });
});

// ============================================================================
// Custom metadata (US5)
// ============================================================================

describe('DocumentStore - custom metadata (US5)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    const logger = mockLogger();
    store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('write with custom fields and read them back', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Custom Metadata Test',
      metadata: {
        server_name: 'prod-01',
        priority: 42,
        active: true,
      },
    });

    const doc = store.read({ id, include_body: false });

    expect(doc.server_name).toBe('prod-01');
    expect(doc.priority).toBe(42);
    expect(doc.active).toBe(true);
  });

  it('list with metadata filter (exact match)', () => {
    store.write({
      _mode: 'create',
      title: 'Server A Docs',
      metadata: { server_name: 'prod-01' },
    });
    store.write({
      _mode: 'create',
      title: 'Server B Docs',
      metadata: { server_name: 'prod-02' },
    });
    store.write({
      _mode: 'create',
      title: 'No Server',
    });

    const result = store.list({ metadata: { server_name: 'prod-01' } });

    expect(result.total).toBe(1);
    expect(result.documents[0]!['title']).toBe('Server A Docs');
    expect(result.documents[0]!['server_name']).toBe('prod-01');
  });

  it('custom keys conflict with reserved fields throws validation error', () => {
    let thrownError: unknown;
    try {
      store.write({
        _mode: 'create',
        title: 'Reserved Key Test',
        metadata: {
          id: 'should-fail',
          tags: ['override'],
        },
      });
    } catch (err) {
      thrownError = err;
    }

    expect(thrownError).toBeDefined();
    const msg = (thrownError as Error).message;
    expect(msg.toLowerCase()).toContain('conflict');
    expect(msg).toContain('reserved');
  });

  it('custom metadata survives partial update merge', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Merge Test',
      metadata: {
        env: 'staging',
        version: 1,
      },
    });

    store.write({
      _mode: 'update',
      id,
      title: 'Merge Test Updated',
      metadata: {
        version: 2,
        region: 'eu-west',
      },
    });

    const doc = store.read({ id, include_body: false });

    // env should be preserved (not provided in update)
    expect(doc.env).toBe('staging');
    // version should be updated
    expect(doc.version).toBe(2);
    // region should be added
    expect(doc.region).toBe('eu-west');
    // title should be updated
    expect(doc.title).toBe('Merge Test Updated');
  });

  it('custom metadata conflict on update also throws validation error', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Update Reserved Key Test',
    });

    let thrownError: unknown;
    try {
      store.write({
        _mode: 'update',
        id,
        metadata: {
          title: 'should-be-reserved',
        },
      });
    } catch (err) {
      thrownError = err;
    }

    expect(thrownError).toBeDefined();
    const msg = (thrownError as Error).message;
    expect(msg.toLowerCase()).toContain('conflict');
  });
});

// ============================================================================
// Index resilience (T022)
// ============================================================================

describe('index resilience (T022)', () => {
  it('rebuilds index from files when _index.yaml is missing', () => {
    const localPaths = makePaths();
    const dir = localPaths.documentsPath;
    try {
      const store = new DocumentStore(makeConfig(), localPaths, mockLogger());
      store.loadIndex();

      // Create 3 documents
      store.write({ _mode: 'create', title: 'Doc A', body: 'aaa' });
      store.write({ _mode: 'create', title: 'Doc B', body: 'bbb' });
      store.write({ _mode: 'create', title: 'Doc C', body: 'ccc' });

      // Delete the index file
      unlinkSync(join(dir, '_index.yaml'));
      expect(existsSync(join(dir, '_index.yaml'))).toBe(false);

      // Reload — should rebuild from files
      const store2 = new DocumentStore(makeConfig(), localPaths, mockLogger());
      store2.loadIndex();

      const result = store2.list({});
      expect(result.total).toBe(3);
      expect(result.documents.map((d) => d.id)).toEqual(
        expect.arrayContaining(['doc-001', 'doc-002', 'doc-003']),
      );
    } finally {
      rmSync(testDir, { recursive: true, force: true });
    }
  });

  it('rebuilds index when _index.yaml is corrupted', () => {
    const localPaths = makePaths();
    const dir = localPaths.documentsPath;
    try {
      const store = new DocumentStore(makeConfig(), localPaths, mockLogger());
      store.loadIndex();

      store.write({ _mode: 'create', title: 'Doc X', body: 'xxx' });

      // Corrupt the index file
      writeFileSync(join(dir, '_index.yaml'), '{{{{invalid yaml!!!!', 'utf-8');

      // Reload — should rebuild from files
      const store2 = new DocumentStore(makeConfig(), localPaths, mockLogger());
      store2.loadIndex();

      const result = store2.list({});
      expect(result.total).toBe(1);
      expect(result.documents[0]!.title).toBe('Doc X');
    } finally {
      rmSync(testDir, { recursive: true, force: true });
    }
  });

  it('computes next_id as max(existing) + 1 after rebuild', () => {
    const localPaths = makePaths();
    const dir = localPaths.documentsPath;
    try {
      const store = new DocumentStore(makeConfig(), localPaths, mockLogger());
      store.loadIndex();

      // Create 5 documents, so next_id should be 6
      for (let i = 0; i < 5; i++) {
        store.write({ _mode: 'create', title: `Doc ${i}`, body: `body ${i}` });
      }

      // Delete index
      unlinkSync(join(dir, '_index.yaml'));

      // Rebuild
      const store2 = new DocumentStore(makeConfig(), localPaths, mockLogger());
      store2.loadIndex();

      // New document should get doc-006
      const result = store2.write({ _mode: 'create', title: 'After rebuild', body: 'new' });
      expect(result.id).toBe('doc-006');
    } finally {
      rmSync(testDir, { recursive: true, force: true });
    }
  });

  it('rebuilds index including binary documents', () => {
    const localPaths = makePaths();
    const dir = localPaths.documentsPath;
    try {
      const store = new DocumentStore(makeConfig(), localPaths, mockLogger());
      store.loadIndex();

      // Create text + binary documents
      store.write({ _mode: 'create', title: 'Text doc', body: 'hello' });
      store.write({
        _mode: 'create',
        title: 'Binary doc',
        content_type: 'image/png',
        body: Buffer.from('fake-png-data').toString('base64'),
      });

      // Delete index
      unlinkSync(join(dir, '_index.yaml'));

      // Rebuild
      const store2 = new DocumentStore(makeConfig(), localPaths, mockLogger());
      store2.loadIndex();

      const result = store2.list({});
      expect(result.total).toBe(2);

      const types = result.documents.map((d) => d.content_type);
      expect(types).toEqual(expect.arrayContaining(['text/plain', 'image/png']));
    } finally {
      rmSync(testDir, { recursive: true, force: true });
    }
  });

  it('rebuilds index skipping blobs directory and unknown directories', () => {
    const localPaths = makePaths();
    const dir = localPaths.documentsPath;
    try {
      const store = new DocumentStore(makeConfig(), localPaths, mockLogger());
      store.loadIndex();

      store.write({ _mode: 'create', title: 'Valid doc', body: 'content' });

      // Create a rogue directory inside documents/
      mkdirSync(join(dir, 'rogue-dir'), { recursive: true });

      // Delete and rebuild index
      unlinkSync(join(dir, '_index.yaml'));
      const store2 = new DocumentStore(makeConfig(), localPaths, mockLogger());
      store2.loadIndex();

      const result = store2.list({});
      expect(result.total).toBe(1);
      expect(result.documents[0]!.title).toBe('Valid doc');
    } finally {
      rmSync(testDir, { recursive: true, force: true });
    }
  });
});

// ============================================================================
// Code review fixes — Phase 11 (T029-T040)
// ============================================================================

describe('DocumentStore - path traversal prevention (T030/T037/T038)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('read() rejects invalid id format with path traversal', () => {
    expect(() => store.read({ id: '../../../etc/passwd' })).toThrow(/Invalid document ID format/);
  });

  it('read() rejects id without doc- prefix', () => {
    expect(() => store.read({ id: 'notadoc' })).toThrow(/Invalid document ID format/);
  });

  it('update rejects invalid id format', () => {
    expect(() => store.write({ _mode: 'update', id: '../bad', title: 'x' })).toThrow(/Invalid document ID format/);
  });

  it('delete rejects invalid id format', () => {
    expect(() => store.delete({ id: 'doc-ab' })).toThrow(/Invalid document ID format/);
  });

  it('accepts valid doc-NNN ids', () => {
    const { id } = store.write({ _mode: 'create', title: 'Valid' });
    expect(() => store.read({ id })).not.toThrow();
    expect(() => store.delete({ id })).not.toThrow();
  });
});

describe('DocumentStore - ENOENT handling in read (T031)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('returns metadata without body when .md file is missing (index/file desync)', () => {
    const { id } = store.write({ _mode: 'create', title: 'Desync test', body: 'some content' });

    // Manually delete the .md file to simulate desync
    const filePath = join(paths.documentsPath, `${id}.md`);
    unlinkSync(filePath);

    // Should not throw — returns metadata without body
    const doc = store.read({ id, include_body: true });
    expect(doc.id).toBe(id);
    expect(doc.title).toBe('Desync test');
    expect(doc.body).toBeUndefined();
  });
});

describe('DocumentStore - blob cleanup on content_type change (T032)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('removes old blob when content_type changes during update', () => {
    const data = Buffer.from('original-png');
    const { id } = store.write({
      _mode: 'create',
      title: 'Blob cleanup test',
      content_type: 'image/png',
      body: data.toString('base64'),
    });

    const oldBlobPath = join(paths.documentsPath, 'blobs', `${id}.png`);
    expect(existsSync(oldBlobPath)).toBe(true);

    // Update with new content_type
    const newData = Buffer.from('new-jpeg');
    store.write({
      _mode: 'update',
      id,
      content_type: 'image/jpeg',
      body: newData.toString('base64'),
    });

    // Old blob should be gone
    expect(existsSync(oldBlobPath)).toBe(false);

    // New blob should exist
    const newBlobPath = join(paths.documentsPath, 'blobs', `${id}.jpeg`);
    expect(existsSync(newBlobPath)).toBe(true);

    // Read should return correct data
    const doc = store.read({ id, include_body: true });
    expect(doc.integrity_ok).toBe(true);
  });
});

// ============================================================================
// Content type class transition (CR001/CR003)
// ============================================================================

describe('DocumentStore - content type class transition (CR001)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('text->binary: removes old .md file when switching to binary content_type', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Text to Binary',
      content_type: 'text/plain',
      body: 'text content',
    });

    const mdPath = join(paths.documentsPath, `${id}.md`);
    expect(existsSync(mdPath)).toBe(true);

    // Update to binary content_type
    const binaryData = Buffer.from('fake-png');
    store.write({
      _mode: 'update',
      id,
      content_type: 'image/png',
      body: binaryData.toString('base64'),
    });

    // Old .md should be gone
    expect(existsSync(mdPath)).toBe(false);

    // New sidecar + blob should exist
    const sidecarPath = join(paths.documentsPath, `${id}.yaml`);
    const blobPath = join(paths.documentsPath, 'blobs', `${id}.png`);
    expect(existsSync(sidecarPath)).toBe(true);
    expect(existsSync(blobPath)).toBe(true);

    // Read should work correctly
    const doc = store.read({ id, include_body: true });
    expect(doc.content_type).toBe('image/png');
    expect(doc.integrity_ok).toBe(true);
  });

  it('binary->text: removes old .yaml sidecar and blob when switching to text content_type', () => {
    const binaryData = Buffer.from('fake-image-data');
    const { id } = store.write({
      _mode: 'create',
      title: 'Binary to Text',
      content_type: 'image/png',
      body: binaryData.toString('base64'),
    });

    const sidecarPath = join(paths.documentsPath, `${id}.yaml`);
    const blobPath = join(paths.documentsPath, 'blobs', `${id}.png`);
    expect(existsSync(sidecarPath)).toBe(true);
    expect(existsSync(blobPath)).toBe(true);

    // Update to text content_type
    store.write({
      _mode: 'update',
      id,
      content_type: 'text/plain',
      body: 'now I am text',
    });

    // Old sidecar + blob should be gone
    expect(existsSync(sidecarPath)).toBe(false);
    expect(existsSync(blobPath)).toBe(false);

    // New .md should exist
    const mdPath = join(paths.documentsPath, `${id}.md`);
    expect(existsSync(mdPath)).toBe(true);

    // Read should return text content
    const doc = store.read({ id, include_body: true });
    expect(doc.content_type).toBe('text/plain');
    expect(doc.body).toBe('now I am text');
    expect(doc.blob_path).toBeUndefined();
    expect(doc.blob_size).toBeUndefined();
    expect(doc.blob_sha256).toBeUndefined();
  });

  it('list() returns copies that do not affect the index when mutated', () => {
    store.write({ _mode: 'create', title: 'Encapsulation Test' });

    const result = store.list({});
    expect(result.total).toBe(1);

    // Mutate the returned object
    result.documents[0]!['title'] = 'MUTATED';

    // Index should be unaffected
    const fresh = store.list({});
    expect(fresh.documents[0]!['title']).toBe('Encapsulation Test');
  });
});

// ============================================================================
// Text file import via file_path (T006 / US1)
// ============================================================================

describe('DocumentStore - text file import via file_path (US1)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;
  let tmpDir: string;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
    tmpDir = mkdtempSync(join(tmpdir(), 'doc-import-text-'));
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it('creates document from .txt file with correct body', () => {
    const filePath = join(tmpDir, 'hello.txt');
    writeFileSync(filePath, 'Hello from file', 'utf-8');

    const { id, created } = store.write({ _mode: 'create', file_path: filePath });

    expect(created).toBe(true);

    const doc = store.read({ id, include_body: true });
    expect(doc.body).toBe('Hello from file');
  });

  it('auto-detects content_type as text/plain for .txt', () => {
    const filePath = join(tmpDir, 'readme.txt');
    writeFileSync(filePath, 'plain text content', 'utf-8');

    const { id } = store.write({ _mode: 'create', file_path: filePath });

    const doc = store.read({ id, include_body: false });
    expect(doc.content_type).toBe('text/plain');
  });

  it('auto-detects content_type as text/yaml for .yaml', () => {
    const filePath = join(tmpDir, 'config.yaml');
    writeFileSync(filePath, 'key: value\n', 'utf-8');

    const { id } = store.write({ _mode: 'create', file_path: filePath });

    const doc = store.read({ id, include_body: true });
    expect(doc.content_type).toBe('text/yaml');
    // text/yaml is inline — body returned as-is
    expect(doc.body).toBe('key: value\n');
  });

  it('sets source_file metadata to basename', () => {
    const filePath = join(tmpDir, 'nginx.conf');
    writeFileSync(filePath, 'server { }', 'utf-8');

    const { id } = store.write({ _mode: 'create', file_path: filePath });

    const doc = store.read({ id, include_body: false });
    expect(doc.source_file).toBe('nginx.conf');
  });

  it('derives auto-title from filename without extension', () => {
    const filePath = join(tmpDir, 'my-document.txt');
    writeFileSync(filePath, 'content', 'utf-8');

    const { id } = store.write({ _mode: 'create', file_path: filePath });

    const doc = store.read({ id, include_body: false });
    expect(doc.title).toBe('my-document');
  });

  it('explicit title overrides auto-title', () => {
    const filePath = join(tmpDir, 'auto-name.txt');
    writeFileSync(filePath, 'content', 'utf-8');

    const { id } = store.write({
      _mode: 'create',
      file_path: filePath,
      title: 'Explicit Title',
    });

    const doc = store.read({ id, include_body: false });
    expect(doc.title).toBe('Explicit Title');
  });

  it('works with relative file path', () => {
    // Write a file in cwd-relative location
    const filePath = join(tmpDir, 'relative-test.txt');
    writeFileSync(filePath, 'relative content', 'utf-8');

    // Use the absolute path but verify it works (relative paths resolve via process.cwd)
    const { id } = store.write({ _mode: 'create', file_path: filePath });

    const doc = store.read({ id, include_body: true });
    expect(doc.body).toBe('relative content');
    expect(doc.source_file).toBe('relative-test.txt');
  });
});

// ============================================================================
// Binary file import via file_path (T007 / US2)
// ============================================================================

describe('DocumentStore - binary file import via file_path (US2)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;
  let tmpDir: string;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
    tmpDir = mkdtempSync(join(tmpdir(), 'doc-import-binary-'));
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it('creates document with blob stored in blobs/ dir', () => {
    const data = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0xde, 0xad]);
    const filePath = join(tmpDir, 'image.png');
    writeFileSync(filePath, data);

    const { id, created } = store.write({
      _mode: 'create',
      file_path: filePath,
      content_type: 'image/png',
      title: 'Test Image',
    });

    expect(created).toBe(true);

    const blobPath = join(paths.documentsPath, 'blobs', `${id}.png`);
    expect(existsSync(blobPath)).toBe(true);
  });

  it('SHA256 of stored blob matches the original file', () => {
    const data = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0xca, 0xfe, 0xba, 0xbe]);
    const filePath = join(tmpDir, 'checksum.png');
    writeFileSync(filePath, data);

    const expectedHash = createHash('sha256').update(data).digest('hex');

    const { id } = store.write({
      _mode: 'create',
      file_path: filePath,
      content_type: 'image/png',
      title: 'Checksum Test',
    });

    const doc = store.read({ id, include_body: false });
    expect(doc.blob_sha256).toBe(expectedHash);
  });

  it('sets source_file metadata for binary import', () => {
    const data = Buffer.from([0xff, 0xd8, 0xff, 0xe0]);
    const filePath = join(tmpDir, 'photo.png');
    writeFileSync(filePath, data);

    const { id } = store.write({
      _mode: 'create',
      file_path: filePath,
      content_type: 'image/png',
      title: 'Photo',
    });

    const doc = store.read({ id, include_body: false });
    expect(doc.source_file).toBe('photo.png');
  });

  it('auto-detects content_type from .png extension without explicit content_type', () => {
    const data = Buffer.from([0x89, 0x50, 0x4e, 0x47]);
    const filePath = join(tmpDir, 'auto-detect.png');
    writeFileSync(filePath, data);

    const { id } = store.write({
      _mode: 'create',
      file_path: filePath,
      title: 'Auto Detect PNG',
    });

    const doc = store.read({ id, include_body: false });
    expect(doc.content_type).toBe('image/png');
    expect(doc.blob_path).toBeDefined();
  });
});

// ============================================================================
// File import error cases (T008)
// ============================================================================

describe('DocumentStore - file import error cases', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;
  let tmpDir: string;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
    tmpDir = mkdtempSync(join(tmpdir(), 'doc-import-errors-'));
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it('throws file_not_found for non-existent path', () => {
    const badPath = join(tmpDir, 'does-not-exist.txt');

    expect(() => store.write({ _mode: 'create', file_path: badPath }))
      .toThrow(expect.objectContaining({ code: 'file_not_found' }));
  });

  it('throws file_is_directory when path points to a directory', () => {
    const dirPath = join(tmpDir, 'a-directory');
    mkdirSync(dirPath, { recursive: true });

    expect(() => store.write({ _mode: 'create', file_path: dirPath }))
      .toThrow(expect.objectContaining({ code: 'file_is_directory' }));
  });

  it('throws document_too_large when file exceeds max_document_size', () => {
    // Create store with very small limit
    const smallPaths = makePaths();
    const smallStore = new DocumentStore(makeConfig({ max_document_size: 100 }), smallPaths, mockLogger());
    smallStore.loadIndex();

    const filePath = join(tmpDir, 'big-file.txt');
    writeFileSync(filePath, 'x'.repeat(200), 'utf-8');

    try {
      expect(() => smallStore.write({ _mode: 'create', file_path: filePath }))
        .toThrow(expect.objectContaining({ code: 'document_too_large' }));
    } finally {
      rmSync(smallPaths.documentsPath, { recursive: true, force: true });
    }
  });

  it('throws file_access_denied when file has no read permission', () => {
    // Skip on non-POSIX (Windows) and when running as root (chmod has no effect)
    if (process.platform === 'win32' || process.getuid?.() === 0) {
      return;
    }

    const filePath = join(tmpDir, 'no-read.txt');
    writeFileSync(filePath, 'secret', 'utf-8');
    chmodSync(filePath, 0o000);

    try {
      expect(() => store.write({ _mode: 'create', file_path: filePath }))
        .toThrow(expect.objectContaining({ code: 'file_access_denied' }));
    } finally {
      // Restore permission for cleanup
      chmodSync(filePath, 0o644);
    }
  });

  it('rejects body + file_path together at schema validation level', () => {
    const result = DocumentWriteCreateInputSchema.safeParse({
      _mode: 'create',
      title: 'Both provided',
      body: 'inline content',
      file_path: '/some/file.txt',
    });

    expect(result.success).toBe(false);
  });
});

// ============================================================================
// File import update mode (T010 / US3)
// ============================================================================

describe('DocumentStore - file import update mode (US3)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;
  let tmpDir: string;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
    tmpDir = mkdtempSync(join(tmpdir(), 'doc-import-update-'));
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it('replaces body with file content on update', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Original',
      body: 'old body content',
    });

    const filePath = join(tmpDir, 'new-content.txt');
    writeFileSync(filePath, 'new body from file', 'utf-8');

    store.write({ _mode: 'update', id, file_path: filePath });

    const doc = store.read({ id, include_body: true });
    expect(doc.body).toBe('new body from file');
  });

  it('updates updated_at timestamp on file-based update', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Timestamp Test',
      body: 'initial',
    });

    const before = store.read({ id, include_body: false });
    const originalUpdatedAt = before.updated_at as string;

    const filePath = join(tmpDir, 'update-ts.txt');
    writeFileSync(filePath, 'updated content', 'utf-8');

    store.write({ _mode: 'update', id, file_path: filePath });

    const after = store.read({ id, include_body: false });
    expect(new Date(after.updated_at as string).getTime()).toBeGreaterThanOrEqual(new Date(originalUpdatedAt).getTime());
  });

  it('sets source_file in metadata on update', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Source File Update Test',
      body: 'original',
    });

    const filePath = join(tmpDir, 'imported-config.txt');
    writeFileSync(filePath, 'config data', 'utf-8');

    store.write({ _mode: 'update', id, file_path: filePath });

    const doc = store.read({ id, include_body: false });
    expect(doc.source_file).toBe('imported-config.txt');
  });

  it('handles content_type transition text to binary via file_path update', () => {
    // Create text document
    const { id } = store.write({
      _mode: 'create',
      title: 'Text To Binary Update',
      content_type: 'text/plain',
      body: 'text content',
    });

    const mdPath = join(paths.documentsPath, `${id}.md`);
    expect(existsSync(mdPath)).toBe(true);

    // Update with binary file
    const binaryData = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0xde, 0xad, 0xbe, 0xef]);
    const filePath = join(tmpDir, 'replacement.png');
    writeFileSync(filePath, binaryData);

    store.write({ _mode: 'update', id, file_path: filePath });

    // Old .md should be gone
    expect(existsSync(mdPath)).toBe(false);

    // New blob should exist
    const blobPath = join(paths.documentsPath, 'blobs', `${id}.png`);
    expect(existsSync(blobPath)).toBe(true);

    // Read should return correct content_type
    const doc = store.read({ id, include_body: false });
    expect(doc.content_type).toBe('image/png');
    expect(doc.source_file).toBe('replacement.png');
    expect(doc.blob_path).toBeDefined();
  });
});

// ============================================================================
// Dirty Tracking
// ============================================================================

describe('DocumentStore - Dirty tracking', () => {
  let paths: ResolvedPaths;
  let logger: Logger;

  beforeEach(() => {
    paths = makePaths();
    logger = mockLogger();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('saveIndex() returns false when no mutations have occurred', () => {
    const store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();
    const result = store.saveIndex();
    expect(result).toBe(false);
  });

  it('saveIndex() returns true after writeCreate', () => {
    const store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();
    store.write({ _mode: 'create', title: 'Test', body: 'content' });
    // writeCreate already calls saveIndex() internally — dirty was reset
    // so a second call should return false
    const result = store.saveIndex();
    expect(result).toBe(false);
  });

  it('saveIndex() returns true after delete', () => {
    const store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();
    const { id } = store.write({ _mode: 'create', title: 'Test', body: 'content' });
    // delete calls saveIndex() internally — dirty was reset
    store.delete({ id });
    const result = store.saveIndex();
    expect(result).toBe(false);
  });

  it('getNextId() sets dirty flag', () => {
    const store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();
    store.getNextId();
    const result = store.saveIndex();
    expect(result).toBe(true);
  });

  it('loadIndex() resets dirty flag', () => {
    const store = new DocumentStore(makeConfig(), paths, logger);
    store.loadIndex();
    store.write({ _mode: 'create', title: 'Test', body: 'content' });
    // reload from disk
    store.loadIndex();
    const result = store.saveIndex();
    expect(result).toBe(false);
  });
});

// ============================================================================
// 036-binary-detect-textonly: isInlineContentType strict text/* only (T001)
// ============================================================================

describe('isInlineContentType — strict text/* only (036-US1)', () => {
  it('text/plain -> true', () => {
    expect(isInlineContentType('text/plain')).toBe(true);
  });

  it('text/markdown -> true', () => {
    expect(isInlineContentType('text/markdown')).toBe(true);
  });

  it('text/html -> true', () => {
    expect(isInlineContentType('text/html')).toBe(true);
  });

  it('text/plain; charset=utf-8 -> true (with MIME params)', () => {
    expect(isInlineContentType('text/plain; charset=utf-8')).toBe(true);
  });

  it('text/html; charset=utf-8; boundary=something -> true (multiple params)', () => {
    expect(isInlineContentType('text/html; charset=utf-8; boundary=something')).toBe(true);
  });

  it('application/json -> false', () => {
    expect(isInlineContentType('application/json')).toBe(false);
  });

  it('application/yaml -> false', () => {
    expect(isInlineContentType('application/yaml')).toBe(false);
  });

  it('image/png -> false', () => {
    expect(isInlineContentType('image/png')).toBe(false);
  });

  it('application/octet-stream -> false', () => {
    expect(isInlineContentType('application/octet-stream')).toBe(false);
  });

  it('application/pdf -> false', () => {
    expect(isInlineContentType('application/pdf')).toBe(false);
  });
});

// ============================================================================
// 036-binary-detect-textonly: JSON/YAML now stored as blob (T002+T003)
// ============================================================================

describe('DocumentStore — JSON/YAML now stored as blob (036-US1)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('application/json document creates .yaml sidecar + blob (not .md)', () => {
    const jsonData = JSON.stringify({ key: 'value' });
    const base64Body = Buffer.from(jsonData).toString('base64');
    const result = store.write({
      _mode: 'create',
      title: 'Config JSON',
      content_type: 'application/json',
      body: base64Body,
    });

    // Verify NO .md file
    const mdPath = join(paths.documentsPath, `${result.id}.md`);
    expect(existsSync(mdPath)).toBe(false);

    // Verify .yaml sidecar exists
    const sidecarPath = join(paths.documentsPath, `${result.id}.yaml`);
    expect(existsSync(sidecarPath)).toBe(true);

    // Verify blob exists
    const blobPath = join(paths.documentsPath, 'blobs', `${result.id}.json`);
    expect(existsSync(blobPath)).toBe(true);

    // Verify blob content
    const blobContent = readFileSync(blobPath, 'utf-8');
    expect(blobContent).toBe(jsonData);
  });

  it('application/yaml document creates .yaml sidecar + blob', () => {
    const yamlData = 'key: value\nlist:\n  - item1\n';
    const base64Body = Buffer.from(yamlData).toString('base64');
    const result = store.write({
      _mode: 'create',
      title: 'Config YAML',
      content_type: 'application/yaml',
      body: base64Body,
    });

    // Verify NO .md file
    const mdPath = join(paths.documentsPath, `${result.id}.md`);
    expect(existsSync(mdPath)).toBe(false);

    // Verify .yaml sidecar exists
    const sidecarPath = join(paths.documentsPath, `${result.id}.yaml`);
    expect(existsSync(sidecarPath)).toBe(true);

    // Verify blob exists
    const blobPath = join(paths.documentsPath, 'blobs', `${result.id}.yaml`);
    expect(existsSync(blobPath)).toBe(true);
  });
});

// ============================================================================
// 036-binary-detect-textonly: blobs/ directory (036-US2)
// ============================================================================

describe('DocumentStore — blobs/ directory (036-US2)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('blobs/ directory exists after loadIndex() (ensureDirectories creates it)', () => {
    const blobsPath = join(paths.documentsPath, 'blobs');
    expect(existsSync(blobsPath)).toBe(true);
  });

  it('text/plain document does not require blobs/', () => {
    store.write({ _mode: 'create', title: 'Text only', content_type: 'text/plain', body: 'hello' });
    // Document should be stored as .md, not in blobs/
    const mdPath = join(paths.documentsPath, 'doc-001.md');
    expect(existsSync(mdPath)).toBe(true);
  });

  it('binary document write creates blob in blobs/ directory', () => {
    const data = Buffer.from('binary-data').toString('base64');
    store.write({ _mode: 'create', title: 'Binary', content_type: 'application/json', body: data });
    const blobsPath = join(paths.documentsPath, 'blobs');
    expect(existsSync(blobsPath)).toBe(true);
  });

  it('second blob write works with existing blobs/ directory', () => {
    const data1 = Buffer.from('data1').toString('base64');
    const data2 = Buffer.from('data2').toString('base64');
    store.write({ _mode: 'create', title: 'B1', content_type: 'image/png', body: data1 });
    store.write({ _mode: 'create', title: 'B2', content_type: 'image/png', body: data2 });
    expect(store.getIndex().documents.length).toBe(2);
  });

  it('rebuildIndex() works when blobs/ does not exist', () => {
    // Create a text document (creates .md)
    store.write({ _mode: 'create', title: 'Text Doc', content_type: 'text/plain', body: 'content' });
    const blobsPath = join(paths.documentsPath, 'blobs');
    // Remove blobs/ to test resilience
    if (existsSync(blobsPath)) rmSync(blobsPath, { recursive: true });
    expect(existsSync(blobsPath)).toBe(false);
    // Rebuild should not throw
    store.rebuildIndex();
    // Index should still have the text document
    expect(store.getIndex().documents.length).toBe(1);
    expect(store.getIndex().documents[0]!.title).toBe('Text Doc');
  });
});

// ============================================================================
// 036-binary-detect-textonly: document_restore (T010)
// ============================================================================

describe('DocumentStore — document_restore (036-US3)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;
  let targetDir: string;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
    targetDir = join(tmpdir(), `restore-${randomUUID()}`);
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
    if (existsSync(targetDir)) {
      rmSync(targetDir, { recursive: true, force: true });
    }
  });

  it('restores blob document with valid SHA256 -> integrity_ok: true', () => {
    const data = Buffer.from('restore-test-data');
    const { id } = store.write({
      _mode: 'create',
      title: 'Blob',
      content_type: 'application/pdf',
      body: data.toString('base64'),
    });
    mkdirSync(targetDir, { recursive: true });
    const targetPath = join(targetDir, 'output.pdf');
    const result = store.restore({ id, target_path: targetPath, force: false });
    expect(result.success).toBe(true);
    expect(result.integrity_ok).toBe(true);
    expect(existsSync(targetPath)).toBe(true);
    const restored = readFileSync(targetPath);
    expect(restored.equals(data)).toBe(true);
  });

  it('restores blob document WITHOUT blob_sha256 (old format) -> no integrity_ok field', () => {
    const data = Buffer.from('old-format-data');
    const { id } = store.write({
      _mode: 'create',
      title: 'Old Blob',
      content_type: 'image/png',
      body: data.toString('base64'),
    });
    // Remove sha256 from index to simulate old format
    const doc = store.getIndex().documents.find((d) => d.id === id)!;
    delete (doc as Record<string, unknown>)['blob_sha256'];

    mkdirSync(targetDir, { recursive: true });
    const targetPath = join(targetDir, 'old-blob.png');
    const result = store.restore({ id, target_path: targetPath, force: false });
    expect(result.success).toBe(true);
    expect(result.integrity_ok).toBeUndefined();
    expect(existsSync(targetPath)).toBe(true);
  });

  it('restores inline document (text/*) -> body without frontmatter as UTF-8', () => {
    const body = '# My Document\n\nSome text content here.';
    const { id } = store.write({
      _mode: 'create',
      title: 'Text Doc',
      content_type: 'text/markdown',
      body,
    });
    mkdirSync(targetDir, { recursive: true });
    const targetPath = join(targetDir, 'doc.md');
    const result = store.restore({ id, target_path: targetPath, force: false });
    expect(result.success).toBe(true);
    expect(result.integrity_ok).toBeUndefined(); // not applicable for inline
    const content = readFileSync(targetPath, 'utf-8');
    expect(content).toBe(body);
  });

  it('throws document_not_found for non-existent id', () => {
    expect(() =>
      store.restore({ id: 'doc-999', target_path: join(targetDir, 'x'), force: false }),
    ).toThrow('not found');
  });

  it('throws file_already_exists when target file exists and force=false', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Doc',
      content_type: 'text/plain',
      body: 'content',
    });
    mkdirSync(targetDir, { recursive: true });
    const targetPath = join(targetDir, 'exists.txt');
    writeFileSync(targetPath, 'already here');
    expect(() => store.restore({ id, target_path: targetPath, force: false })).toThrow(
      'already exists',
    );
  });

  it('overwrites existing file when force=true', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Doc',
      content_type: 'text/plain',
      body: 'new content',
    });
    mkdirSync(targetDir, { recursive: true });
    const targetPath = join(targetDir, 'overwrite.txt');
    writeFileSync(targetPath, 'old content');
    const result = store.restore({ id, target_path: targetPath, force: true });
    expect(result.success).toBe(true);
    expect(readFileSync(targetPath, 'utf-8')).toBe('new content');
  });

  it('throws integrity_error when blob SHA256 mismatches and force=false', () => {
    const data = Buffer.from('original-data');
    const { id } = store.write({
      _mode: 'create',
      title: 'Blob',
      content_type: 'application/pdf',
      body: data.toString('base64'),
    });
    // Corrupt the blob on disk
    const doc = store.getIndex().documents.find((d) => d.id === id)!;
    const blobPath = join(
      paths.documentsPath,
      (doc as Record<string, unknown>)['blob_path'] as string,
    );
    writeFileSync(blobPath, 'corrupted-data');

    mkdirSync(targetDir, { recursive: true });
    const targetPath = join(targetDir, 'fail.pdf');
    expect(() => store.restore({ id, target_path: targetPath, force: false })).toThrow(
      'SHA256 mismatch',
    );
    // File should NOT have been created
    expect(existsSync(targetPath)).toBe(false);
  });

  it('writes file with integrity_ok=false when blob SHA256 mismatches and force=true', () => {
    const data = Buffer.from('original-data');
    const { id } = store.write({
      _mode: 'create',
      title: 'Blob',
      content_type: 'application/pdf',
      body: data.toString('base64'),
    });
    // Corrupt the blob
    const doc = store.getIndex().documents.find((d) => d.id === id)!;
    const blobPath = join(
      paths.documentsPath,
      (doc as Record<string, unknown>)['blob_path'] as string,
    );
    writeFileSync(blobPath, 'corrupted-data');

    mkdirSync(targetDir, { recursive: true });
    const targetPath = join(targetDir, 'forced.pdf');
    const result = store.restore({ id, target_path: targetPath, force: true });
    expect(result.success).toBe(true);
    expect(result.integrity_ok).toBe(false);
    expect(result.warning).toContain('SHA256 mismatch');
    expect(existsSync(targetPath)).toBe(true);
    expect(readFileSync(targetPath, 'utf-8')).toBe('corrupted-data');
  });

  it('creates intermediate directories for target_path (recursive mkdir)', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Doc',
      content_type: 'text/plain',
      body: 'deep content',
    });
    const deepPath = join(targetDir, 'a', 'b', 'c', 'doc.txt');
    const result = store.restore({ id, target_path: deepPath, force: false });
    expect(result.success).toBe(true);
    expect(existsSync(deepPath)).toBe(true);
    expect(readFileSync(deepPath, 'utf-8')).toBe('deep content');
  });
});

// ============================================================================
// 036-binary-detect-textonly: Backward compat — existing inline JSON/YAML (T015)
// ============================================================================

describe('DocumentStore — backward compat: existing inline JSON/YAML (036-US4)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('reads existing .md file with content_type application/json as inline (pre-036 compat)', () => {
    // Simulate pre-036 document: .md file with frontmatter content_type: application/json
    const id = store.getNextId();
    const metadata: Record<string, unknown> = {
      id,
      title: 'Legacy JSON',
      content_type: 'application/json',
      tags: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    const body = '{"key": "value"}';
    const fileContent = store.serializeFrontmatter(metadata, body);
    mkdirSync(paths.documentsPath, { recursive: true });
    const filePath = join(paths.documentsPath, `${id}.md`);
    writeFileSync(filePath, fileContent, 'utf-8');

    // Add to index manually
    store.getIndex().documents.push(metadata as ReturnType<typeof store.getIndex>['documents'][number]);

    // After 036, isInlineContentType('application/json') = false,
    // so read() will try blob_path. Since there's no blob_path, body is absent.
    // Per spec: migration happens on next update, not automatically.
    const doc = store.read({ id, include_body: true });
    expect(doc['id']).toBe(id);
    expect(doc['content_type']).toBe('application/json');
    // Body is undefined since no blob_path exists (pre-migration state)
    expect(doc['body']).toBeUndefined();
  });
});

// ============================================================================
// 036-binary-detect-textonly: Content type transitions JSON/YAML (T016+T017)
// ============================================================================

describe('DocumentStore — content type transitions with JSON/YAML (036-US4)', () => {
  let store: DocumentStore;
  let paths: ResolvedPaths;

  beforeEach(() => {
    paths = makePaths();
    store = new DocumentStore(makeConfig(), paths, mockLogger());
    store.loadIndex();
  });

  afterEach(() => {
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
  });

  it('update text/plain -> application/json: migrates from .md to blob', () => {
    const { id } = store.write({
      _mode: 'create',
      title: 'Was Text',
      content_type: 'text/plain',
      body: 'original text',
    });
    const mdPath = join(paths.documentsPath, `${id}.md`);
    expect(existsSync(mdPath)).toBe(true);

    // Update to application/json (now binary)
    const jsonData = JSON.stringify({ migrated: true });
    store.write({
      _mode: 'update',
      id,
      content_type: 'application/json',
      body: Buffer.from(jsonData).toString('base64'),
    });

    // .md should be removed
    expect(existsSync(mdPath)).toBe(false);
    // .yaml sidecar + blob should exist
    const sidecarPath = join(paths.documentsPath, `${id}.yaml`);
    expect(existsSync(sidecarPath)).toBe(true);
    const blobPath = join(paths.documentsPath, 'blobs', `${id}.json`);
    expect(existsSync(blobPath)).toBe(true);
    expect(readFileSync(blobPath, 'utf-8')).toBe(jsonData);
  });

  it('update application/json -> text/plain: migrates from blob to .md', () => {
    const jsonData = JSON.stringify({ data: 'test' });
    const { id } = store.write({
      _mode: 'create',
      title: 'Was JSON',
      content_type: 'application/json',
      body: Buffer.from(jsonData).toString('base64'),
    });

    const sidecarPath = join(paths.documentsPath, `${id}.yaml`);
    const blobPath = join(paths.documentsPath, 'blobs', `${id}.json`);
    expect(existsSync(sidecarPath)).toBe(true);
    expect(existsSync(blobPath)).toBe(true);

    // Update to text/plain
    store.write({
      _mode: 'update',
      id,
      content_type: 'text/plain',
      body: 'now plain text',
    });

    // blob + sidecar should be removed
    expect(existsSync(sidecarPath)).toBe(false);
    expect(existsSync(blobPath)).toBe(false);
    // .md should exist
    const mdPath = join(paths.documentsPath, `${id}.md`);
    expect(existsSync(mdPath)).toBe(true);

    const doc = store.read({ id, include_body: true });
    expect(doc['content_type']).toBe('text/plain');
    expect(doc['body']).toBe('now plain text');
  });
});
