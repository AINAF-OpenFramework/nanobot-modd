"""YAML output utilities for Triune Memory Translator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_yaml(data: dict[str, Any], path: Path) -> bool:
    """
    Write dict to YAML with formatting.

    Args:
        data: Dictionary to serialize
        path: Path to write to

    Returns:
        True if successful, False otherwise
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        yaml_content = yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
        )

        path.write_text(yaml_content, encoding="utf-8")
        return True
    except Exception:
        return False


def yaml_to_markdown(data: dict[str, Any]) -> str:
    """
    Convert YAML structure back to markdown.

    Args:
        data: YAML data dictionary

    Returns:
        Markdown formatted string
    """
    parts: list[str] = []

    # Handle frontmatter
    if "frontmatter" in data and data["frontmatter"]:
        frontmatter_yaml = yaml.dump(
            data["frontmatter"],
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        parts.append(f"---\n{frontmatter_yaml}---\n")

    # Handle sections
    if "sections" in data:
        for section in data["sections"]:
            parts.append(format_section(section, level=1))

    return "\n".join(parts)


def format_section(section: dict[str, Any], level: int = 1) -> str:
    """
    Format a section dict as markdown.

    Args:
        section: Section dictionary
        level: Header level (1-6)

    Returns:
        Markdown formatted section
    """
    parts: list[str] = []

    # Add header if title exists
    title = section.get("title", "")
    if title:
        header = "#" * level
        parts.append(f"{header} {title}\n")

    # Add content
    content = section.get("content", "")
    if content:
        parts.append(f"{content}\n")

    # Add code blocks
    code_blocks = section.get("code_blocks", [])
    for cb in code_blocks:
        lang = cb.get("language", "")
        code = cb.get("content", "")
        parts.append(f"```{lang}\n{code}\n```\n")

    # Add lists
    lists = section.get("lists", [])
    for lst in lists:
        list_type = lst.get("type", "bullet")
        items = lst.get("items", [])

        for i, item in enumerate(items):
            if list_type == "checkbox":
                if isinstance(item, dict):
                    checked = "x" if item.get("checked", False) else " "
                    text = item.get("text", "")
                    parts.append(f"- [{checked}] {text}")
                else:
                    parts.append(f"- [ ] {item}")
            elif list_type == "numbered":
                parts.append(f"{i + 1}. {item}")
            else:
                parts.append(f"- {item}")

        parts.append("")  # Blank line after list

    # Add subsections
    subsections = section.get("subsections", [])
    for sub in subsections:
        parts.append(format_section(sub, level=level + 1))

    return "\n".join(parts)


def dict_to_compact_yaml(data: dict[str, Any]) -> str:
    """
    Convert dict to compact YAML string (for context injection).

    Args:
        data: Dictionary to convert

    Returns:
        Compact YAML string
    """
    return yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        indent=2,
        width=120,
    )
