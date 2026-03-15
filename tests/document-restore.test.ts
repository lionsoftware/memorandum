import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { existsSync, readFileSync, rmSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { Logger } from 'pino';
import type { Config, ResolvedPaths } from '../src/config.js';
import { DocumentStore } from '../src/document-store.js';

function mockLogger(): Logger {
  return {
    info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(),
    child: vi.fn().mockReturnThis(), fatal: vi.fn(), trace: vi.fn(),
    silent: vi.fn(), level: 'debug',
  } as unknown as Logger;
}

function makeConfig(): Config {
  return {
    max_entries: 100, autosave_interval_seconds: 0, storage_dir: '.memorandum',
    max_document_size: 16 * 1024 * 1024, semantic_model: 'Xenova/multilingual-e5-small',
    semantic_model_dtype: 'q8', semantic_enabled: false, semantic_debounce_seconds: 10,
    semantic_max_queue_size: 200, semantic_max_retries: 3,
  };
}

let testDir: string;

function makePaths(): ResolvedPaths {
  testDir = join(tmpdir(), `test-restore-${randomUUID()}`);
  mkdirSync(testDir, { recursive: true });
  return {
    storageDir: testDir,
    factsPath: join(testDir, 'facts', 'facts.json'),
    documentsPath: join(testDir, 'documents'),
    cachePath: join(testDir, 'cache'),
    configPath: join(testDir, 'config.yaml'),
  };
}

describe('DocumentStore.restore - inline documents', () => {
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

  it('restores inline document body to target file', () => {
    const body = '# Hello\n\nThis is a test document.';
    store.write({ _mode: 'create', title: 'Test Doc', body, content_type: 'text/markdown' });

    const targetPath = join(testDir, 'restored.md');
    const result = store.restore({ id: 'doc-001', target_path: targetPath, force: false });

    expect(result.success).toBe(true);
    expect(result.id).toBe('doc-001');
    expect(existsSync(targetPath)).toBe(true);
    expect(readFileSync(targetPath, 'utf-8')).toBe(body);
  });

  it('strips frontmatter and writes only body', () => {
    store.write({ _mode: 'create', title: 'With Meta', body: 'Just the body.', content_type: 'text/plain' });

    const targetPath = join(testDir, 'body-only.txt');
    store.restore({ id: 'doc-001', target_path: targetPath, force: false });

    const content = readFileSync(targetPath, 'utf-8');
    expect(content).toBe('Just the body.');
    expect(content).not.toContain('---');
  });

  it('creates parent directories if they do not exist', () => {
    store.write({ _mode: 'create', title: 'Nested', body: 'content', content_type: 'text/plain' });

    const targetPath = join(testDir, 'deep', 'nested', 'dir', 'file.txt');
    store.restore({ id: 'doc-001', target_path: targetPath, force: false });

    expect(existsSync(targetPath)).toBe(true);
  });

  it('throws if target file already exists and force=false', () => {
    store.write({ _mode: 'create', title: 'Doc', body: 'content', content_type: 'text/plain' });

    const targetPath = join(testDir, 'existing.txt');
    writeFileSync(targetPath, 'old content', 'utf-8');

    expect(() => store.restore({ id: 'doc-001', target_path: targetPath, force: false }))
      .toThrow('File already exists');
  });

  it('overwrites existing file when force=true', () => {
    store.write({ _mode: 'create', title: 'Doc', body: 'new content', content_type: 'text/plain' });

    const targetPath = join(testDir, 'existing.txt');
    writeFileSync(targetPath, 'old content', 'utf-8');

    store.restore({ id: 'doc-001', target_path: targetPath, force: true });
    expect(readFileSync(targetPath, 'utf-8')).toBe('new content');
  });

  it('throws for non-existent document ID', () => {
    expect(() => store.restore({ id: 'doc-999', target_path: join(testDir, 'out.txt'), force: false }))
      .toThrow("Document 'doc-999' not found");
  });

  it('throws for invalid document ID format', () => {
    expect(() => store.restore({ id: 'bad-id', target_path: join(testDir, 'out.txt'), force: false }))
      .toThrow('Invalid document ID');
  });
});

describe('DocumentStore.restore - binary documents', () => {
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

  it('restores binary document to target file with integrity check', () => {
    const binaryData = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]); // PNG header
    const base64 = binaryData.toString('base64');
    store.write({ _mode: 'create', title: 'Test PNG', body: base64, content_type: 'image/png' });

    const targetPath = join(testDir, 'restored.png');
    const result = store.restore({ id: 'doc-001', target_path: targetPath, force: false });

    expect(result.success).toBe(true);
    expect(result.integrity_ok).toBe(true);
    expect(existsSync(targetPath)).toBe(true);

    const restored = readFileSync(targetPath);
    expect(Buffer.compare(restored, binaryData)).toBe(0);
  });

  it('throws on SHA256 mismatch when force=false', () => {
    const binaryData = Buffer.from([0x01, 0x02, 0x03]);
    store.write({ _mode: 'create', title: 'Blob', body: binaryData.toString('base64'), content_type: 'application/octet-stream' });

    // Corrupt the blob on disk
    const doc = store.getIndex().documents[0]!;
    const blobPath = join(paths.documentsPath, doc.blob_path!);
    writeFileSync(blobPath, Buffer.from([0xFF, 0xFE, 0xFD]));

    const targetPath = join(testDir, 'corrupted.bin');
    expect(() => store.restore({ id: 'doc-001', target_path: targetPath, force: false }))
      .toThrow('SHA256 mismatch');
  });

  it('writes with warning on SHA256 mismatch when force=true', () => {
    const binaryData = Buffer.from([0x01, 0x02, 0x03]);
    store.write({ _mode: 'create', title: 'Blob', body: binaryData.toString('base64'), content_type: 'application/octet-stream' });

    // Corrupt the blob on disk
    const doc = store.getIndex().documents[0]!;
    const blobPath = join(paths.documentsPath, doc.blob_path!);
    writeFileSync(blobPath, Buffer.from([0xFF, 0xFE, 0xFD]));

    const targetPath = join(testDir, 'forced.bin');
    const result = store.restore({ id: 'doc-001', target_path: targetPath, force: true });

    expect(result.success).toBe(true);
    expect(result.integrity_ok).toBe(false);
    expect(result.warning).toContain('SHA256 mismatch');
    expect(existsSync(targetPath)).toBe(true);
  });
});

describe('isInlineContentType - MIME params handling', () => {
  it('handles content type with charset param', async () => {
    const { isInlineContentType } = await import('../src/document-types.js');
    expect(isInlineContentType('text/plain; charset=utf-8')).toBe(true);
    expect(isInlineContentType('text/html; charset=iso-8859-1')).toBe(true);
  });

  it('rejects non-text with params', async () => {
    const { isInlineContentType } = await import('../src/document-types.js');
    expect(isInlineContentType('application/pdf; version=1.7')).toBe(false);
    expect(isInlineContentType('image/png; quality=90')).toBe(false);
  });

  it('handles base types correctly', async () => {
    const { isInlineContentType } = await import('../src/document-types.js');
    expect(isInlineContentType('text/plain')).toBe(true);
    expect(isInlineContentType('text/markdown')).toBe(true);
    expect(isInlineContentType('application/octet-stream')).toBe(false);
    expect(isInlineContentType('image/jpeg')).toBe(false);
  });
});
