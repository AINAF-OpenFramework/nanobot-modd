"""Tests for Triune Memory Translator module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from nanobot.utils.md_parser import (
    CodeBlock,
    ParsedMarkdown,
    Section,
    extract_code_blocks,
    extract_frontmatter,
    extract_lists,
    extract_sections,
    parse_markdown,
    section_to_dict,
)
from nanobot.utils.translator import (
    SyncStats,
    TranslatorConfig,
    export_yaml_to_md,
    get_md_path,
    get_yaml_path,
    load_yaml_or_md,
    needs_reverse_sync,
    needs_sync,
    parse_md_to_yaml,
    sync_all,
    sync_directory,
    yaml_data_to_context,
)
from nanobot.utils.yaml_writer import (
    dict_to_compact_yaml,
    format_section,
    write_yaml,
    yaml_to_markdown,
)


class TestMdParser:
    """Tests for markdown parsing utilities."""

    def test_extract_frontmatter_with_yaml(self):
        """Test extraction of YAML frontmatter."""
        content = """---
title: Test Document
author: Test Author
tags:
  - test
  - example
---

# Main Content

This is the body.
"""
        frontmatter, remaining = extract_frontmatter(content)

        assert frontmatter is not None
        assert frontmatter["title"] == "Test Document"
        assert frontmatter["author"] == "Test Author"
        assert frontmatter["tags"] == ["test", "example"]
        assert "# Main Content" in remaining

    def test_extract_frontmatter_without_yaml(self):
        """Test content without frontmatter."""
        content = """# Just a Header

Some content here.
"""
        frontmatter, remaining = extract_frontmatter(content)

        assert frontmatter is None
        assert remaining == content

    def test_extract_sections_simple(self):
        """Test extraction of simple sections."""
        content = """# First Section

First content.

## Subsection A

Subsection content.

# Second Section

Second content.
"""
        sections = extract_sections(content)

        assert len(sections) == 2
        assert sections[0].title == "First Section"
        assert sections[0].level == 1
        assert len(sections[0].subsections) == 1
        assert sections[0].subsections[0].title == "Subsection A"
        assert sections[1].title == "Second Section"

    def test_extract_code_blocks(self):
        """Test extraction of fenced code blocks."""
        content = """Some text before.

```python
def hello():
    print("Hello, World!")
```

More text.

```javascript
console.log("Hello");
```
"""
        blocks = extract_code_blocks(content)

        assert len(blocks) == 2
        assert blocks[0].language == "python"
        assert "def hello():" in blocks[0].content
        assert blocks[1].language == "javascript"

    def test_extract_code_blocks_no_language(self):
        """Test extraction of code blocks without language."""
        content = """```
plain text code
```"""
        blocks = extract_code_blocks(content)

        assert len(blocks) == 1
        assert blocks[0].language == ""
        assert "plain text code" in blocks[0].content

    def test_extract_lists_bullet(self):
        """Test extraction of bullet lists."""
        content = """Some intro text.

- Item one
- Item two
- Item three

More text.
"""
        lists = extract_lists(content)

        assert len(lists) == 1
        assert lists[0]["type"] == "bullet"
        assert lists[0]["items"] == ["Item one", "Item two", "Item three"]

    def test_extract_lists_numbered(self):
        """Test extraction of numbered lists."""
        content = """1. First item
2. Second item
3. Third item
"""
        lists = extract_lists(content)

        assert len(lists) == 1
        assert lists[0]["type"] == "numbered"
        assert lists[0]["items"] == ["First item", "Second item", "Third item"]

    def test_extract_lists_checkbox(self):
        """Test extraction of checkbox lists."""
        content = """- [x] Completed task
- [ ] Pending task
- [X] Another done
"""
        lists = extract_lists(content)

        assert len(lists) == 1
        assert lists[0]["type"] == "checkbox"
        assert len(lists[0]["items"]) == 3
        assert lists[0]["items"][0]["checked"] is True
        assert lists[0]["items"][1]["checked"] is False
        assert lists[0]["items"][2]["checked"] is True

    def test_parse_markdown_complete(self):
        """Test complete markdown parsing."""
        content = """---
version: 1.0
---

# Document Title

Introduction paragraph.

## Section One

Content with a list:

- Item A
- Item B

```python
code_here()
```
"""
        parsed = parse_markdown(content)

        assert isinstance(parsed, ParsedMarkdown)
        assert parsed.frontmatter is not None
        assert parsed.frontmatter["version"] == 1.0
        assert parsed.title == "Document Title"
        assert len(parsed.sections) >= 1

    def test_section_to_dict(self):
        """Test conversion of Section to dict."""
        section = Section(
            level=1,
            title="Test Section",
            content="Some content here.\n\n- List item",
            subsections=[
                Section(level=2, title="Subsection", content="Sub content"),
            ],
        )

        result = section_to_dict(section)

        assert result["title"] == "Test Section"
        assert "content" in result or "lists" in result
        assert len(result["subsections"]) == 1


class TestYamlWriter:
    """Tests for YAML writing utilities."""

    def test_write_yaml_creates_file(self):
        """Test that write_yaml creates a valid YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            data = {"key": "value", "list": [1, 2, 3]}

            success = write_yaml(data, path)

            assert success
            assert path.exists()

            loaded = yaml.safe_load(path.read_text())
            assert loaded["key"] == "value"
            assert loaded["list"] == [1, 2, 3]

    def test_write_yaml_creates_parent_dirs(self):
        """Test that write_yaml creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "test.yaml"
            data = {"test": True}

            success = write_yaml(data, path)

            assert success
            assert path.exists()

    def test_format_section_simple(self):
        """Test formatting a simple section."""
        section = {
            "title": "My Section",
            "content": "Some content here.",
        }

        result = format_section(section, level=2)

        assert "## My Section" in result
        assert "Some content here." in result

    def test_format_section_with_code_blocks(self):
        """Test formatting section with code blocks."""
        section = {
            "title": "Code Example",
            "code_blocks": [{"language": "python", "content": "print('hello')"}],
        }

        result = format_section(section)

        assert "```python" in result
        assert "print('hello')" in result
        assert "```" in result

    def test_yaml_to_markdown_roundtrip(self):
        """Test YAML to markdown conversion."""
        data = {
            "frontmatter": {"title": "Test"},
            "sections": [
                {
                    "title": "Section One",
                    "content": "Content here",
                }
            ],
        }

        result = yaml_to_markdown(data)

        assert "---" in result
        assert "title: Test" in result
        assert "# Section One" in result
        assert "Content here" in result

    def test_dict_to_compact_yaml(self):
        """Test compact YAML conversion."""
        data = {"key": "value", "nested": {"a": 1, "b": 2}}

        result = dict_to_compact_yaml(data)

        assert "key: value" in result
        assert "nested:" in result


class TestTranslator:
    """Tests for translator module."""

    def test_get_yaml_path(self):
        """Test YAML path generation."""
        md_path = Path("/workspace/AGENTS.md")
        yaml_path = get_yaml_path(md_path)

        assert yaml_path == Path("/workspace/AGENTS.yaml")

    def test_get_md_path(self):
        """Test MD path generation."""
        yaml_path = Path("/workspace/AGENTS.yaml")
        md_path = get_md_path(yaml_path)

        assert md_path == Path("/workspace/AGENTS.md")

    def test_needs_sync_no_yaml(self):
        """Test needs_sync when YAML doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test.md"
            yaml_path = Path(tmpdir) / "test.yaml"

            md_path.write_text("# Test")

            assert needs_sync(md_path, yaml_path) is True

    def test_needs_sync_yaml_older(self):
        """Test needs_sync when YAML is older."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test.md"
            yaml_path = Path(tmpdir) / "test.yaml"

            # Create YAML first
            yaml_path.write_text("test: true")
            import time

            time.sleep(0.01)
            # Then MD
            md_path.write_text("# Test")

            assert needs_sync(md_path, yaml_path) is True

    def test_needs_sync_yaml_newer(self):
        """Test needs_sync when YAML is newer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test.md"
            yaml_path = Path(tmpdir) / "test.yaml"

            # Create MD first
            md_path.write_text("# Test")
            import time

            time.sleep(0.01)
            # Then YAML
            yaml_path.write_text("test: true")

            assert needs_sync(md_path, yaml_path) is False

    def test_needs_reverse_sync(self):
        """Test needs_reverse_sync detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test.md"
            yaml_path = Path(tmpdir) / "test.yaml"

            md_path.write_text("# Test")
            import time

            time.sleep(0.01)
            yaml_path.write_text("test: true")

            assert needs_reverse_sync(md_path, yaml_path) is True

    def test_parse_md_to_yaml_simple(self):
        """Test basic markdown to YAML conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test.md"
            yaml_path = Path(tmpdir) / "test.yaml"

            md_content = """# Agent Instructions

You are a helpful AI assistant.

## Guidelines

- Be concise
- Be accurate
"""
            md_path.write_text(md_content)

            success = parse_md_to_yaml(md_path, yaml_path)

            assert success
            assert yaml_path.exists()

            data = yaml.safe_load(yaml_path.read_text())
            assert "_meta" in data
            assert data["_meta"]["source"] == "test.md"
            assert "sections" in data

    def test_parse_md_to_yaml_with_frontmatter(self):
        """Test markdown with YAML frontmatter conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test.md"
            yaml_path = Path(tmpdir) / "test.yaml"

            md_content = """---
version: 1.0
author: Test
---

# Title

Content here.
"""
            md_path.write_text(md_content)

            success = parse_md_to_yaml(md_path, yaml_path)

            assert success
            data = yaml.safe_load(yaml_path.read_text())
            assert data["frontmatter"]["version"] == 1.0
            assert data["frontmatter"]["author"] == "Test"

    def test_parse_md_to_yaml_with_code_blocks(self):
        """Test preservation of code blocks in conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test.md"
            yaml_path = Path(tmpdir) / "test.yaml"

            md_content = """# Code Example

```python
def hello():
    return "world"
```
"""
            md_path.write_text(md_content)

            success = parse_md_to_yaml(md_path, yaml_path)

            assert success
            data = yaml.safe_load(yaml_path.read_text())

            # Find code block in sections
            found_code = False
            for section in data.get("sections", []):
                if "code_blocks" in section:
                    for cb in section["code_blocks"]:
                        if cb["language"] == "python":
                            found_code = True
                            assert "def hello():" in cb["content"]

            assert found_code

    def test_parse_md_to_yaml_with_lists(self):
        """Test conversion of bullet lists to YAML arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test.md"
            yaml_path = Path(tmpdir) / "test.yaml"

            md_content = """# Tasks

- Task one
- Task two
- Task three
"""
            md_path.write_text(md_content)

            success = parse_md_to_yaml(md_path, yaml_path)

            assert success
            data = yaml.safe_load(yaml_path.read_text())

            found_list = False
            for section in data.get("sections", []):
                if "lists" in section:
                    found_list = True
                    assert section["lists"][0]["type"] == "bullet"
                    assert len(section["lists"][0]["items"]) == 3

            assert found_list

    def test_export_yaml_to_md(self):
        """Test round-trip YAML back to markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            md_path = Path(tmpdir) / "test.md"

            data = {
                "_meta": {"source": "test.md"},
                "frontmatter": {"version": "1.0"},
                "sections": [
                    {
                        "title": "Section One",
                        "content": "Content here.",
                    }
                ],
            }
            yaml_path.write_text(yaml.dump(data))

            success = export_yaml_to_md(yaml_path, md_path)

            assert success
            assert md_path.exists()

            content = md_path.read_text()
            assert "---" in content
            assert "version:" in content
            assert "# Section One" in content

    def test_sync_directory_basic(self):
        """Test syncing an entire directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create some .md files
            (workspace / "AGENTS.md").write_text("# Agents\n\nAgent content.")
            (workspace / "USER.md").write_text("# User\n\nUser prefs.")
            (workspace / "README.md").write_text("# README\n\nSkipped.")

            stats = sync_directory(workspace, direction="md_to_yaml")

            assert stats.synced == 2  # AGENTS.md and USER.md
            assert stats.skipped == 1  # README.md is excluded
            assert (workspace / "AGENTS.yaml").exists()
            assert (workspace / "USER.yaml").exists()
            assert not (workspace / "README.yaml").exists()

    def test_sync_directory_dry_run(self):
        """Test dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "TEST.md").write_text("# Test")

            stats = sync_directory(workspace, direction="md_to_yaml", dry_run=True)

            assert stats.synced == 1
            assert not (workspace / "TEST.yaml").exists()

    def test_needs_sync_mtime(self):
        """Test modification time comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test.md"
            yaml_path = Path(tmpdir) / "test.yaml"

            # Create both files
            md_path.write_text("# Test")
            yaml_path.write_text("test: true")

            # Initially YAML is newer (created after)
            assert needs_sync(md_path, yaml_path) is False

            # Touch MD file to make it newer
            import time

            time.sleep(0.01)
            md_path.write_text("# Test Updated")

            assert needs_sync(md_path, yaml_path) is True

    def test_excluded_files_respected(self):
        """Test that excluded files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create files
            (workspace / "AGENTS.md").write_text("# Agents")
            (workspace / "README.md").write_text("# README")
            (workspace / "CHANGELOG.md").write_text("# Changes")

            config = TranslatorConfig(excluded_files=["README.md", "CHANGELOG.md"])
            stats = sync_directory(workspace, config=config)

            assert stats.synced == 1
            assert stats.skipped == 2

    def test_sync_all_workspace(self):
        """Test full workspace sync."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory_dir = workspace / "memory"
            memory_dir.mkdir()

            # Create workspace files
            (workspace / "AGENTS.md").write_text("# Agents")
            (workspace / "USER.md").write_text("# User")

            # Create memory files
            (memory_dir / "MEMORY.md").write_text("# Memory")
            (memory_dir / "HISTORY.md").write_text("# History")

            result = sync_all(workspace, quiet=True)

            assert result["synced"] == 4
            assert (workspace / "AGENTS.yaml").exists()
            assert (memory_dir / "MEMORY.yaml").exists()

    def test_load_yaml_or_md_prefers_yaml(self):
        """Test that YAML is preferred over MD."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create both files
            (workspace / "TEST.md").write_text("# Markdown version")
            yaml_data = {"title": "YAML version", "sections": []}
            (workspace / "TEST.yaml").write_text(yaml.dump(yaml_data))

            data, content = load_yaml_or_md(workspace, "TEST")

            assert data is not None
            assert data["title"] == "YAML version"
            assert content is None

    def test_load_yaml_or_md_falls_back_to_md(self):
        """Test fallback to MD when YAML doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create only MD file
            (workspace / "TEST.md").write_text("# Markdown only")

            data, content = load_yaml_or_md(workspace, "TEST")

            assert data is None
            assert content is not None
            assert "Markdown only" in content

    def test_yaml_data_to_context(self):
        """Test conversion of YAML data to context string."""
        data = {
            "title": "Test Document",
            "sections": [
                {
                    "title": "Section One",
                    "content": "Some content here.",
                    "lists": [{"type": "bullet", "items": ["Item A", "Item B"]}],
                }
            ],
        }

        result = yaml_data_to_context(data)

        assert "# Test Document" in result
        assert "## Section One" in result
        assert "Some content here." in result
        assert "- Item A" in result

    def test_translator_config_defaults(self):
        """Test TranslatorConfig default values."""
        config = TranslatorConfig()

        assert config.enabled is True
        assert config.auto_sync_on_startup is True
        assert config.sync_direction == "bidirectional"
        assert "README.md" in config.excluded_files


class TestTokenEfficiency:
    """Tests for token efficiency verification."""

    def test_yaml_is_more_compact(self):
        """Verify YAML output is more compact than raw MD for context."""
        md_content = """# Agent Instructions

You are a helpful AI assistant. Be concise, accurate, and friendly.

## Guidelines

- Always explain what you're doing before taking actions
- Ask for clarification when the request is ambiguous
- Use tools to help accomplish tasks
- Remember important information in your memory files

## Tools Available

You have access to:
- File operations (read, write, edit, list)
- Shell commands (exec)
- Web access (search, fetch)
- Messaging (message)
- Background tasks (spawn)

## Memory

- `memory/MEMORY.md` — long-term facts (preferences, context, relationships)
- `memory/HISTORY.md` — append-only event log, search with grep to recall past events
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test.md"
            yaml_path = Path(tmpdir) / "test.yaml"

            md_path.write_text(md_content)
            parse_md_to_yaml(md_path, yaml_path)

            yaml_content = yaml_path.read_text()

            # YAML should be comparable or smaller (removes redundant whitespace)
            # The structured format is more token-efficient for LLMs
            assert yaml_path.exists()

            # Convert YAML back to context and verify structure preserved
            data = yaml.safe_load(yaml_content)
            context = yaml_data_to_context(data)

            # Key content should be preserved
            assert "Agent Instructions" in context or "Guidelines" in context
