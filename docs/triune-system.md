# Triune Memory System Architecture

## Overview

The Triune Memory System is a dual-format memory architecture that maintains both human-readable Markdown (`.md`) and token-efficient YAML (`.yaml`) representations of configuration and documentation files. This system ensures optimal human comprehension while minimizing LLM token consumption during runtime.

## Core Principles

### 1. **Dual Representation**
- **Markdown (`.md`)** - Source of truth for human editing
  - Rich formatting, explanations, examples
  - Easy to read and modify
  - Version control friendly
- **YAML (`.yaml`)** - Runtime format for LLM consumption
  - Structured, token-efficient
  - Fast parsing
  - Reduced context window usage

### 2. **Synchronization**
- Bidirectional translation between MD ↔ YAML
- Automated sync on file changes
- Integrity verification with checksums
- Drift detection and auto-repair

### 3. **Runtime Preference**
- Runtime systems prefer YAML when available
- Graceful fallback to MD if YAML missing
- Warnings logged for sync drift

## System Components

### Translation Infrastructure

Located in `nanobot/utils/`:

#### `translator.py`
Core translation engine providing:
- `parse_md_to_yaml()` - Convert MD to YAML
- `export_yaml_to_md()` - Convert YAML to MD
- `sync_directory()` - Sync files in a directory
- `sync_all()` - Full project sync
- `load_yaml_or_md()` - Runtime loader with YAML preference

#### `md_parser.py`
Markdown parsing utilities:
- Extract frontmatter (YAML headers)
- Parse sections and subsections
- Extract code blocks and lists
- Preserve document structure

#### `yaml_writer.py`
YAML serialization:
- `write_yaml()` - Write structured YAML
- `yaml_to_markdown()` - Reverse conversion
- Formatting and style preservation

### Integrity Verification System

Located in `nanobot/triune/`:

#### `checksums.py`
Checksum management:
- **ChecksumManager** - Thread-safe checksum tracking
- MD5/SHA256 hash computation
- Persistent storage in `.triune/checksums.json`
- Verification of MD-YAML pair integrity

#### `verifier.py`
Sync verification:
- **TriuneVerifier** - Full repository verification
- Detect missing YAML files
- Identify drifted files (MD newer than YAML)
- Find orphaned YAML files
- Validate YAML parseability
- Auto-repair with `--fix` flag

### YAML Loaders

Thread-safe singleton loaders for different modules:

#### `nanobot/soul/loader.py` ✅
Soul configuration (personality, goals, strategies)
- Thread-safe singleton pattern
- Cached loading
- Default config generation

#### `nanobot/governance/loader.py`
Governance policies and rules
- Policies, rules, constraints
- Runtime validation

#### `nanobot/memory/loader.py`
Memory configurations and schemas
- Memory schemas
- Templates
- Configuration settings

#### `nanobot/latent/loader.py`
Latent reasoning patterns
- Reasoning patterns
- Heuristics
- Configuration

#### `nanobot/game/loader.py`
Game learning configurations
- Game configurations
- Reward models
- Learning parameters

### CLI Tools

Located in `nanobot/cli/`:

#### `triune_commands.py`
Verification command interface:

```bash
# Check sync status
nanobot triune verify

# Auto-regenerate missing/drifted YAML
nanobot triune verify --fix

# Detailed report
nanobot triune verify --report

# Verify specific path
nanobot triune verify --path /path/to/verify
```

#### Status Output
```
Triune Sync Status:
✓ 45 MD files synced
✗ 3 MD files missing YAML
⚠ 2 YAML files drifted from MD
○ 1 orphaned YAML file

Details:
- Missing YAML: governance/policy.md, memory/cache.md
- Drifted: latent/reasoning.yaml (MD modified 5min ago)
- Orphaned: old_config.yaml
```

### Health Monitoring

#### `/health` Endpoint Enhancement
The health endpoint (`nanobot/gateway.py`) includes Triune status:

```json
{
  "status": "ok",
  "triune": {
    "sync_status": "partial_sync",
    "yaml_validity": "valid",
    "loader_status": {
      "soul": "loaded",
      "governance": "not_configured",
      "memory": "loaded",
      "latent": "not_configured",
      "game": "not_configured"
    },
    "stats": {
      "total_md_files": 48,
      "synced_yaml_files": 45,
      "drifted_files": 2,
      "missing_yaml": 3,
      "orphaned_yaml": 1
    },
    "last_verification": "2026-02-15T10:30:00Z"
  }
}
```

## Workflow

### Initial Setup
1. Create `.md` files with documentation/configuration
2. Run `nanobot triune verify --fix` to generate YAML
3. Checksums stored in `.triune/checksums.json`

### Development Workflow
1. Edit `.md` files (human-readable format)
2. Run `nanobot triune verify --fix` to sync YAML
3. Runtime uses YAML for efficiency
4. Verification detects any manual YAML edits

### Runtime Operation
1. Loaders attempt to load `.yaml` first
2. Fall back to `.md` if YAML missing
3. Log warnings for drift detection
4. Periodic verification checks integrity

### CI/CD Integration
```bash
# In CI pipeline
nanobot triune verify --report

# Fail build if sync issues
if [ $? -ne 0 ]; then
  echo "Triune sync issues detected"
  exit 1
fi
```

## File Structure

```
nanobot/
├── triune/                      # Verification system
│   ├── __init__.py
│   ├── checksums.py            # Checksum management
│   └── verifier.py             # Sync verification
├── governance/                  # Governance module
│   ├── __init__.py
│   └── loader.py               # YAML loader
├── memory/                      # Memory module
│   ├── __init__.py
│   └── loader.py               # YAML loader
├── latent/                      # Latent reasoning
│   ├── __init__.py
│   └── loader.py               # YAML loader
├── game/                        # Game learning
│   └── loader.py               # YAML loader
├── soul/                        # Soul configuration
│   ├── loader.py               # Existing YAML loader
│   └── schema.py               # Data models
├── utils/                       # Translation utilities
│   ├── translator.py           # Core translator
│   ├── md_parser.py            # MD parsing
│   └── yaml_writer.py          # YAML writing
└── cli/
    ├── commands.py             # Main CLI
    └── triune_commands.py      # Triune CLI

.triune/
└── checksums.json              # Checksum storage

workspace/
├── *.md                        # Source files
└── *.yaml                      # Generated mirrors
```

## Design Patterns

### Singleton Loaders
All loaders follow the Soul loader pattern:
- Thread-safe singleton per config path
- Cached configuration after first load
- `force_reload` flag for updates
- Graceful handling of missing files

### Checksum Strategy
- MD5 for fast verification (default)
- SHA256 for high-security needs
- Checksums stored with metadata
- Last verification timestamp

### Error Handling
- Missing files: Return empty dict, log warning
- Invalid YAML: Return empty dict, log error
- Sync drift: Log warning, continue execution
- Runtime failures: Graceful degradation

## Performance Considerations

### Token Efficiency
- YAML format ~30-50% fewer tokens than Markdown
- Reduced context window usage
- Faster parsing and serialization

### Caching
- Loaders cache after first load
- Checksums cached in memory
- `force_reload` available when needed

### Lazy Loading
- Loaders instantiate on first use
- Configuration loaded on first access
- Minimal startup overhead

## Security

### Checksum Integrity
- Detect unauthorized modifications
- Verify sync before critical operations
- Audit trail in checksums.json

### Configuration Validation
- YAML schema validation
- Type checking via Pydantic (Soul)
- Runtime validation in loaders

## Best Practices

### For Developers
1. **Always edit .md files**, not .yaml
2. Run `nanobot triune verify --fix` after changes
3. Commit both .md and .yaml files
4. Review checksums.json for audit trail

### For Operators
1. Monitor `/health` endpoint for Triune status
2. Set up alerts for sync drift
3. Run verification in CI/CD pipeline
4. Backup checksums.json with config files

### For Contributors
1. Follow existing loader patterns
2. Add unit tests for new loaders
3. Update documentation
4. Test both .md and .yaml paths

## Troubleshooting

### YAML Files Out of Sync
```bash
nanobot triune verify --report
nanobot triune verify --fix
```

### Orphaned YAML Files
1. Identify with `nanobot triune verify --report`
2. Check if corresponding .md was deleted
3. Remove orphaned .yaml manually

### Invalid YAML Syntax
1. Verify reports invalid files
2. Regenerate with `--fix` flag
3. If persists, check .md source

### Loader Errors
1. Check health endpoint for loader status
2. Verify .yaml file exists and is valid
3. Check file permissions
4. Review logs for detailed errors

## Future Enhancements

### Planned Features
- [ ] Watch mode for auto-sync on file changes
- [ ] Compression for large YAML files
- [ ] Encryption for sensitive configs
- [ ] Remote sync to object storage
- [ ] Web UI for verification status

### Optimization Opportunities
- [ ] Parallel checksum computation
- [ ] Incremental sync (only changed files)
- [ ] Background verification task
- [ ] Cache warming on startup

## References

- [Triune Memory Concept](../TRIUNE_MEMORY.md)
- [Soul Loader Implementation](../nanobot/soul/loader.py)
- [Translation Tests](../tests/test_translator.py)
- [Phase 2 Tests](../tests/test_triune_phase2.py)
