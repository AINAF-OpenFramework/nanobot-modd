# Triune Memory System

The Triune Memory Translator is a universal `.md` → `.yaml` translation system that enables nanobot to read token-efficient `.yaml` files while preserving human-readable `.md` files for editing and inspection.

## Overview

The Triune Memory concept maintains three layers:

1. **Human Layer** (`.md` files) - Human-readable, editable markdown files
2. **AI Layer** (`.yaml` files) - Machine-efficient, token-optimized structured data
3. **Sync Layer** (Translator) - Bidirectional synchronization between the two

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        WORKSPACE                                 │
├─────────────────────────────────────────────────────────────────┤
│  Human Layer (.md)          │          AI Layer (.yaml)         │
│  ─────────────────          │          ───────────────          │
│  AGENTS.md      ←───────────┼──────→   AGENTS.yaml              │
│  SOUL.md        ←───────────┼──────→   SOUL.yaml                │
│  USER.md        ←───────────┼──────→   USER.yaml                │
│  TOOLS.md       ←───────────┼──────→   TOOLS.yaml               │
│  IDENTITY.md    ←───────────┼──────→   IDENTITY.yaml            │
│                             │                                    │
│  memory/MEMORY.md ←─────────┼──────→   memory/MEMORY.yaml       │
│  memory/HISTORY.md ←────────┼──────→   memory/HISTORY.yaml      │
└─────────────────────────────┴───────────────────────────────────┘
                              │
                      Translator Module
                              │
                    nanobot/utils/translator.py
```

## Benefits

### Token Efficiency
- YAML format removes redundant markdown syntax
- Structured data is more compact
- Estimated 20-40% reduction in context token usage

### Maintainability
- Edit `.md` files with any text editor
- Human-readable documentation preserved
- Version control friendly

### Flexibility
- Fallback to `.md` when `.yaml` doesn't exist
- Bidirectional sync supported
- Incremental sync based on modification time

## Usage

### CLI Command

```bash
# Sync markdown to YAML (default)
nanobot translate

# Sync YAML back to markdown
nanobot translate -d yaml-to-md

# Bidirectional sync
nanobot translate -d bidirectional

# Preview changes without modifying files
nanobot translate --dry-run
```

### Automatic Sync

By default, nanobot syncs `.md` files to `.yaml` on startup. Configure this in `config.json`:

```json
{
  "translator": {
    "enabled": true,
    "autoSyncOnStartup": true,
    "syncDirection": "md_to_yaml",
    "excludedFiles": ["README.md", "CHANGELOG.md"]
  }
}
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `true` | Enable/disable translator |
| `autoSyncOnStartup` | `true` | Auto-sync when nanobot starts |
| `syncDirection` | `"md_to_yaml"` | Direction: `md_to_yaml`, `yaml_to_md`, `bidirectional` |
| `excludedFiles` | `["README.md", ...]` | Files to skip during sync |

## File Format

### Markdown Input

```markdown
---
version: 1.0
---

# Agent Instructions

You are a helpful AI assistant.

## Guidelines

- Be concise and accurate
- Use tools when needed

```python
# Example code block
print("Hello")
```
```

### YAML Output

```yaml
_meta:
  source: AGENTS.md
  format_version: '1.0'
frontmatter:
  version: 1.0
title: Agent Instructions
sections:
  - title: Agent Instructions
    content: You are a helpful AI assistant.
    subsections:
      - title: Guidelines
        lists:
          - type: bullet
            items:
              - Be concise and accurate
              - Use tools when needed
        code_blocks:
          - language: python
            content: |
              # Example code block
              print("Hello")
```

## API Reference

### Translator Module (`nanobot/utils/translator.py`)

#### Functions

##### `parse_md_to_yaml(md_path: Path, yaml_path: Path) -> bool`
Convert markdown file to structured YAML.

##### `export_yaml_to_md(yaml_path: Path, md_path: Path) -> bool`
Export YAML back to readable markdown.

##### `sync_directory(directory: Path, direction: str = "md_to_yaml") -> SyncStats`
Sync all files in a directory.

##### `sync_all(workspace: Path, direction: str = "md_to_yaml") -> dict`
Full project sync - workspace and memory directories.

##### `needs_sync(md_path: Path, yaml_path: Path) -> bool`
Check if sync is needed based on modification time.

##### `load_yaml_or_md(base_path: Path, base_name: str) -> tuple`
Load content preferring YAML, falling back to MD.

#### Classes

##### `TranslatorConfig`
Configuration dataclass with fields:
- `enabled: bool = True`
- `auto_sync_on_startup: bool = True`
- `sync_direction: str = "bidirectional"`
- `excluded_files: list[str]`

### Markdown Parser (`nanobot/utils/md_parser.py`)

##### `parse_markdown(content: str) -> ParsedMarkdown`
Parse markdown into structured dict.

##### `extract_frontmatter(content: str) -> tuple[dict | None, str]`
Extract YAML frontmatter from markdown.

##### `extract_sections(content: str) -> list[Section]`
Extract sections by header levels.

##### `extract_code_blocks(content: str) -> list[CodeBlock]`
Extract fenced code blocks.

##### `extract_lists(content: str) -> list[dict]`
Extract bullet/numbered/checkbox lists.

### YAML Writer (`nanobot/utils/yaml_writer.py`)

##### `write_yaml(data: dict, path: Path) -> bool`
Write dict to YAML with formatting.

##### `yaml_to_markdown(data: dict) -> str`
Convert YAML structure back to markdown.

## Migration Guide

### Existing Workspaces

1. Run sync command:
   ```bash
   nanobot translate
   ```

2. Verify YAML files created:
   ```bash
   ls ~/.nanobot/workspace/*.yaml
   ```

3. Nanobot will automatically prefer YAML files on next run.

### Manual Testing

```python
from nanobot.utils.translator import sync_all
from pathlib import Path

workspace = Path("~/.nanobot/workspace").expanduser()
result = sync_all(workspace, direction="md_to_yaml")
print(f"Synced: {result['synced']}, Skipped: {result['skipped']}")
```

## Troubleshooting

### YAML Not Loading

1. Check file exists: `ls workspace/*.yaml`
2. Verify YAML is valid: `python -c "import yaml; yaml.safe_load(open('FILE.yaml'))"`
3. Check for parse errors in logs

### Sync Not Working

1. Check modification times: `ls -la workspace/*.md workspace/*.yaml`
2. Force resync: Edit `.md` file and run translate again
3. Try dry-run first: `nanobot translate --dry-run`

### Content Missing After Sync

- Code blocks and lists are extracted into structured fields
- Content is restructured, not lost
- Use `export_yaml_to_md` to verify round-trip

## Best Practices

1. **Edit .md files** - Let translator handle YAML generation
2. **Use frontmatter** - Metadata in YAML frontmatter is preserved
3. **Commit both formats** - Version control both .md and .yaml
4. **Review dry-run** - Use `--dry-run` before bulk sync
5. **Exclude documentation** - Add non-agent files to `excludedFiles`

## Technical Details

### Supported Markdown Elements

- YAML frontmatter (delimited by `---`)
- Headers (levels 1-6)
- Bullet lists (`-`, `*`, `+`)
- Numbered lists (`1.`, `2.`, etc.)
- Checkbox lists (`- [ ]`, `- [x]`)
- Fenced code blocks (with language tags)
- Paragraphs and text content

### Sync Algorithm

1. Scan directory for `.md` files
2. Check if corresponding `.yaml` exists
3. Compare modification times
4. If `.md` is newer or `.yaml` missing, parse and convert
5. Update `.yaml` file
6. Log operation

### Performance

- Sync time: < 100ms for typical workspace
- Incremental sync based on mtime (no full re-parse)
- Minimal memory footprint
