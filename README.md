# memorandum-mcp

[README on Russian](README.ru.md)

MCP server with persistent memory for AI agents. Two memory types:

- **Facts** — short key-value records with LRU eviction and optional TTL
- **Documents** — structured storage with metadata, tags, topics, and semantic search

## Features

- LRU cache with configurable size and per-entry TTL
- Namespace isolation for facts
- Document storage with YAML index and Markdown frontmatter
- Binary document support (base64 encoding, SHA-256 integrity)
- File import with automatic MIME type detection
- Semantic search via local HuggingFace embedding model (multilingual-e5-small)
- Debounced batch embedding queue with failure tracking and retry
- Atomic file writes (temp file + rename) for crash safety
- Autosave with configurable interval
- Export/import for facts backup and migration

## Installation

```bash
npm install memorandum-mcp
```

Or run directly with npx:

```bash
npx memorandum-mcp
```

## Configuration

Create `.memorandum/config.yaml` in your working directory:

```yaml
max_entries: 2048
autosave_interval_seconds: 60
storage_dir: .memorandum/
```

### All options

| Option | Default | Description |
|--------|---------|-------------|
| `max_entries` | `1000` | Maximum facts in LRU cache |
| `autosave_interval_seconds` | `300` | Autosave interval (0 to disable) |
| `storage_dir` | `.memorandum` | Base storage directory |
| `max_document_size` | `16777216` | Max document body size in bytes (16 MiB) |
| `semantic_enabled` | `true` | Enable semantic search |
| `semantic_model` | `Xenova/multilingual-e5-small` | HuggingFace embedding model |
| `semantic_model_dtype` | `q8` | Model quantization type |
| `semantic_debounce_seconds` | `10` | Delay before batch embedding |
| `semantic_max_queue_size` | `200` | Queue size before forced batch |
| `semantic_max_retries` | `3` | Max retries per failed embedding |

### Environment variables

| Variable | Description |
|----------|-------------|
| `MEMORANDUM_STORAGE_DIR` | Override storage directory path |
| `MEMORANDUM_LOG_LEVEL` | Log level: `debug`, `info`, `warn`, `error` |

## Storage layout

```
.memorandum/
  config.yaml          # Configuration
  .gitignore           # Auto-generated, excludes cache/
  facts/
    facts.json         # LRU cache dump
  documents/
    _index.yaml        # Document index
    doc-001.md         # Inline document (with YAML frontmatter)
    doc-002.yaml       # Binary document sidecar
    blobs/
      doc-002.png      # Binary blob
  cache/
    vector-index.json  # Semantic search index
```

## MCP Tools

### Facts (3 tools)

#### `memory_write`

Write or update a fact.

```json
{
  "key": "server-web-01",
  "value": { "ip": "192.168.1.10", "role": "webserver" },
  "namespace": "servers",
  "ttl_seconds": 3600
}
```

#### `memory_read`

Read a fact by key and namespace.

```json
{ "key": "server-web-01", "namespace": "servers" }
```

#### `memory_manage`

Manage facts with actions: `list`, `search`, `namespaces`, `delete`, `delete_namespace`, `export`, `import`, `sync`.

```json
{ "action": "list", "namespace": "servers", "pattern": "web-*", "include_values": true }
```

```json
{ "action": "search", "query": "webserver", "limit": 5 }
```

```json
{ "action": "sync" }
```

### Documents (4 tools)

#### `document_write`

Create or update a document.

```json
{
  "title": "Server inventory",
  "body": "# Servers\n\n- web-01: 192.168.1.10\n- db-01: 192.168.1.20",
  "topic": "infrastructure",
  "tags": ["servers", "inventory"],
  "content_type": "text/markdown"
}
```

Update by providing `id`:

```json
{ "id": "doc-001", "tags": ["servers", "inventory", "updated"] }
```

Import a local file:

```json
{ "file_path": "/path/to/document.md" }
```

#### `document_read`

```json
{ "id": "doc-001", "include_body": true }
```

#### `document_list`

```json
{ "tag": "servers", "search": "inventory", "limit": 10 }
```

#### `document_delete`

```json
{ "id": "doc-001" }
```

### Semantic Search (2 tools)

#### `semantic_search`

Search by meaning across facts and documents.

```json
{
  "query": "web server configuration",
  "source": "all",
  "limit": 10,
  "threshold": 0.3
}
```

#### `semantic_reindex`

Rebuild the vector index.

```json
{ "source": "all" }
```

## Usage with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memorandum": {
      "command": "npx",
      "args": ["memorandum-mcp"]
    }
  }
}
```

## Usage with Claude Code

Add to your `.claude/settings.json` or project settings:

```json
{
  "mcpServers": {
    "memorandum": {
      "command": "npx",
      "args": ["memorandum-mcp"]
    }
  }
}
```

## Development

```bash
git clone https://github.com/lionsoftware/memorandum.git
cd memorandum
npm install
npm run build
npm test
```

## License

[GPL-3.0](LICENSE)
