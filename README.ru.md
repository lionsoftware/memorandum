# memorandum-mcp

[README in English](README.md)

MCP-сервер с персистентной памятью для AI-агентов. Два типа памяти:

- **Факты** — короткие key-value записи с LRU-вытеснением и опциональным TTL
- **Документы** — структурированное хранилище с метаданными, тегами, топиками и семантическим поиском

## Возможности

- LRU-кэш с настраиваемым размером и TTL для каждой записи
- Изоляция фактов по пространствам имён (namespaces)
- Хранение документов с YAML-индексом и Markdown-фронтматтером
- Поддержка бинарных документов (base64, проверка целостности SHA-256)
- Импорт файлов с автоматическим определением MIME-типа
- Семантический поиск через локальную модель HuggingFace (multilingual-e5-small)
- Очередь пакетного эмбеддинга с debounce, отслеживанием ошибок и повторными попытками
- Атомарная запись файлов (временный файл + rename) для защиты от сбоев
- Автосохранение с настраиваемым интервалом
- Экспорт/импорт фактов для резервного копирования и миграции

## Установка

```bash
npm install memorandum-mcp
```

Или запуск напрямую через npx:

```bash
npx memorandum-mcp
```

## Конфигурация

Создайте `.memorandum/config.yaml` в рабочей директории:

```yaml
max_entries: 2048
autosave_interval_seconds: 60
storage_dir: .memorandum/
```

### Все параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `max_entries` | `1000` | Максимум фактов в LRU-кэше |
| `autosave_interval_seconds` | `300` | Интервал автосохранения (0 — отключить) |
| `storage_dir` | `.memorandum` | Базовая директория хранения |
| `max_document_size` | `16777216` | Максимальный размер документа в байтах (16 МиБ) |
| `semantic_enabled` | `true` | Включить семантический поиск |
| `semantic_model` | `Xenova/multilingual-e5-small` | Модель эмбеддингов HuggingFace |
| `semantic_model_dtype` | `q8` | Тип квантизации модели |
| `semantic_debounce_seconds` | `10` | Задержка перед пакетным эмбеддингом |
| `semantic_max_queue_size` | `200` | Размер очереди для принудительного батча |
| `semantic_max_retries` | `3` | Максимум повторных попыток на запись |

### Переменные окружения

| Переменная | Описание |
|------------|----------|
| `MEMORANDUM_STORAGE_DIR` | Переопределить путь к директории хранения |
| `MEMORANDUM_LOG_LEVEL` | Уровень логирования: `debug`, `info`, `warn`, `error` |

## Структура хранилища

```
.memorandum/
  config.yaml          # Конфигурация
  .gitignore           # Автоматически создаётся, исключает cache/
  facts/
    facts.json         # Дамп LRU-кэша
  documents/
    _index.yaml        # Индекс документов
    doc-001.md         # Текстовый документ (с YAML-фронтматтером)
    doc-002.yaml       # Сайдкар бинарного документа
    blobs/
      doc-002.png      # Бинарный блоб
  cache/
    vector-index.json  # Индекс семантического поиска
```

## MCP-инструменты

### Факты (3 инструмента)

#### `memory_write`

Записать или обновить факт.

```json
{
  "key": "server-web-01",
  "value": { "ip": "192.168.1.10", "role": "webserver" },
  "namespace": "servers",
  "ttl_seconds": 3600
}
```

#### `memory_read`

Прочитать факт по ключу и пространству имён.

```json
{ "key": "server-web-01", "namespace": "servers" }
```

#### `memory_manage`

Управление фактами. Действия: `list`, `search`, `namespaces`, `delete`, `delete_namespace`, `export`, `import`, `sync`.

```json
{ "action": "list", "namespace": "servers", "pattern": "web-*", "include_values": true }
```

```json
{ "action": "search", "query": "webserver", "limit": 5 }
```

```json
{ "action": "sync" }
```

### Документы (4 инструмента)

#### `document_write`

Создать или обновить документ.

```json
{
  "title": "Инвентарь серверов",
  "body": "# Серверы\n\n- web-01: 192.168.1.10\n- db-01: 192.168.1.20",
  "topic": "infrastructure",
  "tags": ["servers", "inventory"],
  "content_type": "text/markdown"
}
```

Обновление по `id`:

```json
{ "id": "doc-001", "tags": ["servers", "inventory", "updated"] }
```

Импорт локального файла:

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

### Семантический поиск (2 инструмента)

#### `semantic_search`

Поиск по смыслу среди фактов и документов.

```json
{
  "query": "конфигурация веб-сервера",
  "source": "all",
  "limit": 10,
  "threshold": 0.3
}
```

#### `semantic_reindex`

Перестроить векторный индекс.

```json
{ "source": "all" }
```

## Использование с Claude Desktop

Добавьте в `claude_desktop_config.json`:

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

## Использование с Claude Code

Добавьте в `.claude/settings.json` или настройки проекта:

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

## Разработка

```bash
git clone https://github.com/lionsoftware/memorandum.git
cd memorandum
npm install
npm run build
npm test
```

## Лицензия

[GPL-3.0](LICENSE)
