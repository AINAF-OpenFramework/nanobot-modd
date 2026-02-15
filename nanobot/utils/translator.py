"""Core translator module for Triune Memory Translator.

Provides bidirectional translation between human-readable .md files
and token-efficient .yaml files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nanobot.utils.md_parser import (
    parse_markdown,
    section_to_dict,
)
from nanobot.utils.yaml_writer import write_yaml, yaml_to_markdown

logger = logging.getLogger(__name__)


@dataclass
class TranslatorConfig:
    """Configuration for the translator module."""

    enabled: bool = True
    auto_sync_on_startup: bool = True
    watch_for_changes: bool = False
    sync_direction: str = "bidirectional"  # md_to_yaml, yaml_to_md, bidirectional
    excluded_files: list[str] = field(
        default_factory=lambda: ["README.md", "CHANGELOG.md", "IMPLEMENTATION_SUMMARY.md"]
    )
    log_level: str = "INFO"


@dataclass
class SyncStats:
    """Statistics from a sync operation."""

    synced: int = 0
    skipped: int = 0
    errors: int = 0
    details: list[str] = field(default_factory=list)


def get_yaml_path(md_path: Path) -> Path:
    """
    Get corresponding .yaml path for .md file.

    Args:
        md_path: Path to .md file

    Returns:
        Path to corresponding .yaml file
    """
    return md_path.with_suffix(".yaml")


def get_md_path(yaml_path: Path) -> Path:
    """
    Get corresponding .md path for .yaml file.

    Args:
        yaml_path: Path to .yaml file

    Returns:
        Path to corresponding .md file
    """
    return yaml_path.with_suffix(".md")


def needs_sync(md_path: Path, yaml_path: Path) -> bool:
    """
    Check if sync is needed based on mtime.

    Args:
        md_path: Path to .md file
        yaml_path: Path to .yaml file

    Returns:
        True if sync needed (md is newer or yaml doesn't exist)
    """
    if not md_path.exists():
        return False

    if not yaml_path.exists():
        return True

    md_mtime = md_path.stat().st_mtime
    yaml_mtime = yaml_path.stat().st_mtime

    return md_mtime > yaml_mtime


def needs_reverse_sync(md_path: Path, yaml_path: Path) -> bool:
    """
    Check if reverse sync is needed (yaml to md).

    Args:
        md_path: Path to .md file
        yaml_path: Path to .yaml file

    Returns:
        True if yaml is newer than md
    """
    if not yaml_path.exists():
        return False

    if not md_path.exists():
        return True

    md_mtime = md_path.stat().st_mtime
    yaml_mtime = yaml_path.stat().st_mtime

    return yaml_mtime > md_mtime


def parse_md_to_yaml(md_path: Path, yaml_path: Path) -> bool:
    """
    Convert markdown file to structured YAML.

    Args:
        md_path: Path to source .md file
        yaml_path: Path to destination .yaml file

    Returns:
        True if successful, False otherwise
    """
    try:
        content = md_path.read_text(encoding="utf-8")
        parsed = parse_markdown(content)

        # Build YAML structure
        data: dict[str, Any] = {
            "_meta": {
                "source": md_path.name,
                "format_version": "1.0",
            }
        }

        if parsed.frontmatter:
            data["frontmatter"] = parsed.frontmatter

        if parsed.title:
            data["title"] = parsed.title

        # Convert sections to dict format
        if parsed.sections:
            data["sections"] = [section_to_dict(s) for s in parsed.sections]

        # Write YAML
        success = write_yaml(data, yaml_path)

        if success:
            logger.info(f"Translator: Synced {md_path.name} → {yaml_path.name}")
        else:
            logger.warning(f"Translator: Failed to write {yaml_path.name}")

        return success

    except Exception as e:
        logger.error(f"Translator: Error converting {md_path}: {e}")
        return False


def export_yaml_to_md(yaml_path: Path, md_path: Path) -> bool:
    """
    Export YAML back to readable markdown.

    Args:
        yaml_path: Path to source .yaml file
        md_path: Path to destination .md file

    Returns:
        True if successful, False otherwise
    """
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            logger.warning(f"Translator: Invalid YAML structure in {yaml_path}")
            return False

        markdown = yaml_to_markdown(data)

        md_path.write_text(markdown, encoding="utf-8")
        logger.info(f"Translator: Exported {yaml_path.name} → {md_path.name}")
        return True

    except Exception as e:
        logger.error(f"Translator: Error exporting {yaml_path}: {e}")
        return False


def sync_directory(
    directory: Path,
    direction: str = "md_to_yaml",
    config: TranslatorConfig | None = None,
    dry_run: bool = False,
) -> SyncStats:
    """
    Sync all files in a directory.

    Args:
        directory: Directory to sync
        direction: Sync direction (md_to_yaml, yaml_to_md, bidirectional)
        config: Translator configuration
        dry_run: If True, don't make changes

    Returns:
        SyncStats with counts of synced/skipped/errors
    """
    config = config or TranslatorConfig()
    stats = SyncStats()

    if not directory.exists():
        return stats

    # Find all .md files in directory (non-recursive)
    md_files = list(directory.glob("*.md"))

    for md_path in md_files:
        # Skip excluded files
        if md_path.name in config.excluded_files:
            stats.skipped += 1
            stats.details.append(f"Skipped (excluded): {md_path.name}")
            continue

        yaml_path = get_yaml_path(md_path)

        # Determine sync direction
        if direction in ("md_to_yaml", "bidirectional"):
            if needs_sync(md_path, yaml_path):
                if dry_run:
                    stats.synced += 1
                    stats.details.append(f"Would sync: {md_path.name} → {yaml_path.name}")
                else:
                    if parse_md_to_yaml(md_path, yaml_path):
                        stats.synced += 1
                        stats.details.append(f"Synced: {md_path.name} → {yaml_path.name}")
                    else:
                        stats.errors += 1
                        stats.details.append(f"Error: {md_path.name}")
            else:
                stats.skipped += 1
                stats.details.append(f"Skipped (up to date): {md_path.name}")

        if direction in ("yaml_to_md", "bidirectional"):
            if needs_reverse_sync(md_path, yaml_path):
                if dry_run:
                    stats.synced += 1
                    stats.details.append(f"Would sync: {yaml_path.name} → {md_path.name}")
                else:
                    if export_yaml_to_md(yaml_path, md_path):
                        stats.synced += 1
                        stats.details.append(f"Synced: {yaml_path.name} → {md_path.name}")
                    else:
                        stats.errors += 1
                        stats.details.append(f"Error: {yaml_path.name}")

    return stats


def sync_all(
    workspace: Path,
    direction: str = "md_to_yaml",
    config: TranslatorConfig | None = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> dict[str, int]:
    """
    Full project sync - syncs workspace and memory directories.

    Args:
        workspace: Workspace root path
        direction: Sync direction (md_to_yaml, yaml_to_md, bidirectional)
        config: Translator configuration
        dry_run: If True, don't make changes
        quiet: If True, suppress info logging

    Returns:
        Dict with synced/skipped/errors counts
    """
    config = config or TranslatorConfig()
    total_stats = SyncStats()

    # Directories to sync
    directories = [
        workspace,  # Main workspace (AGENTS.md, SOUL.md, etc.)
        workspace / "memory",  # Memory files
    ]

    # Also check for skill directories
    skills_dir = workspace / "skills"
    if skills_dir.exists():
        for skill_path in skills_dir.iterdir():
            if skill_path.is_dir():
                directories.append(skill_path)

    for directory in directories:
        stats = sync_directory(directory, direction, config, dry_run)
        total_stats.synced += stats.synced
        total_stats.skipped += stats.skipped
        total_stats.errors += stats.errors
        total_stats.details.extend(stats.details)

    if not quiet:
        logger.info(
            f"Translator: Sync complete - "
            f"synced={total_stats.synced}, skipped={total_stats.skipped}, errors={total_stats.errors}"
        )

    return {
        "synced": total_stats.synced,
        "skipped": total_stats.skipped,
        "errors": total_stats.errors,
    }


def load_yaml_or_md(base_path: Path, base_name: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Load content preferring YAML, falling back to MD.

    Args:
        base_path: Directory containing files
        base_name: Base name without extension (e.g., "AGENTS")

    Returns:
        Tuple of (data dict or None, content string or None)
        Returns (yaml_data, None) if YAML loaded
        Returns (None, md_content) if MD loaded
        Returns (None, None) if neither exists
    """
    yaml_path = base_path / f"{base_name}.yaml"
    md_path = base_path / f"{base_name}.md"

    # Prefer YAML
    if yaml_path.exists():
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            return data, None
        except Exception as e:
            logger.warning(f"Failed to load {yaml_path}: {e}")

    # Fall back to MD
    if md_path.exists():
        try:
            content = md_path.read_text(encoding="utf-8")
            return None, content
        except Exception as e:
            logger.warning(f"Failed to load {md_path}: {e}")

    return None, None


def yaml_data_to_context(data: dict[str, Any]) -> str:
    """
    Convert YAML data to context string for LLM.

    Args:
        data: YAML data dictionary

    Returns:
        Formatted context string
    """
    parts: list[str] = []

    # Handle title
    if "title" in data:
        parts.append(f"# {data['title']}")

    # Handle sections
    for section in data.get("sections", []):
        parts.append(_format_section_for_context(section))

    return "\n\n".join(parts)


def _format_section_for_context(section: dict[str, Any], level: int = 1) -> str:
    """Format a section for context display."""
    parts: list[str] = []

    title = section.get("title", "")
    if title:
        header = "#" * (level + 1)
        parts.append(f"{header} {title}")

    content = section.get("content", "")
    if content:
        parts.append(content)

    # Include code blocks
    for cb in section.get("code_blocks", []):
        lang = cb.get("language", "")
        code = cb.get("content", "")
        parts.append(f"```{lang}\n{code}\n```")

    # Include lists
    for lst in section.get("lists", []):
        list_type = lst.get("type", "bullet")
        items = lst.get("items", [])
        for i, item in enumerate(items):
            if list_type == "checkbox" and isinstance(item, dict):
                checked = "x" if item.get("checked") else " "
                parts.append(f"- [{checked}] {item.get('text', '')}")
            elif list_type == "numbered":
                parts.append(f"{i + 1}. {item}")
            else:
                parts.append(f"- {item}")

    # Handle subsections
    for sub in section.get("subsections", []):
        parts.append(_format_section_for_context(sub, level + 1))

    return "\n".join(parts)
