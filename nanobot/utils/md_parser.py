"""Markdown parsing utilities for Triune Memory Translator."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Section:
    """Represents a markdown section."""

    level: int
    title: str
    content: str
    subsections: list[Section] = field(default_factory=list)


@dataclass
class CodeBlock:
    """Represents a fenced code block."""

    language: str
    content: str


@dataclass
class ListItem:
    """Represents a list item."""

    text: str
    checked: bool | None = None  # None for non-checkbox items


@dataclass
class ParsedMarkdown:
    """Result of parsing a markdown document."""

    frontmatter: dict[str, Any] | None
    title: str | None
    sections: list[Section]
    raw_content: str


def parse_markdown(content: str) -> ParsedMarkdown:
    """
    Parse markdown into structured dict.

    Args:
        content: Raw markdown content

    Returns:
        ParsedMarkdown with frontmatter, title, sections, and raw content
    """
    frontmatter, body = extract_frontmatter(content)
    sections = extract_sections(body)
    title = sections[0].title if sections else None

    return ParsedMarkdown(
        frontmatter=frontmatter,
        title=title,
        sections=sections,
        raw_content=content,
    )


def extract_frontmatter(content: str) -> tuple[dict[str, Any] | None, str]:
    """
    Extract YAML frontmatter from markdown.

    Args:
        content: Raw markdown content

    Returns:
        Tuple of (frontmatter dict or None, remaining content)
    """
    # Match YAML frontmatter between --- delimiters at the start
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return None, content

    try:
        import yaml

        frontmatter_text = match.group(1)
        frontmatter = yaml.safe_load(frontmatter_text)
        remaining = content[match.end() :]
        return frontmatter, remaining
    except Exception:
        return None, content


def extract_sections(content: str) -> list[Section]:
    """
    Extract sections by header levels.

    Args:
        content: Markdown content (without frontmatter)

    Returns:
        List of Section objects with nested subsections
    """
    # Split content by headers
    header_pattern = r"^(#{1,6})\s+(.+)$"
    lines = content.split("\n")

    sections: list[Section] = []
    current_section: Section | None = None
    current_content_lines: list[str] = []

    for line in lines:
        match = re.match(header_pattern, line)

        if match:
            # Save previous section content
            if current_section is not None:
                current_section.content = "\n".join(current_content_lines).strip()
                sections.append(current_section)
            elif current_content_lines:
                # Content before first header
                sections.append(
                    Section(level=0, title="", content="\n".join(current_content_lines).strip())
                )

            # Start new section
            level = len(match.group(1))
            title = match.group(2).strip()
            current_section = Section(level=level, title=title, content="")
            current_content_lines = []
        else:
            current_content_lines.append(line)

    # Don't forget the last section
    if current_section is not None:
        current_section.content = "\n".join(current_content_lines).strip()
        sections.append(current_section)
    elif current_content_lines:
        sections.append(
            Section(level=0, title="", content="\n".join(current_content_lines).strip())
        )

    # Build hierarchy (nest subsections)
    return _build_section_hierarchy(sections)


def _build_section_hierarchy(flat_sections: list[Section]) -> list[Section]:
    """Build hierarchical section structure from flat list."""
    if not flat_sections:
        return []

    result: list[Section] = []
    stack: list[Section] = []

    for section in flat_sections:
        # Pop sections from stack that are same level or higher (lower number = higher level)
        while stack and stack[-1].level >= section.level and section.level > 0:
            stack.pop()

        if stack and section.level > 0:
            # Add as subsection of the last item on stack
            stack[-1].subsections.append(section)
        else:
            # Top-level section
            result.append(section)

        if section.level > 0:
            stack.append(section)

    return result


def extract_code_blocks(content: str) -> list[CodeBlock]:
    """
    Extract fenced code blocks.

    Args:
        content: Markdown content

    Returns:
        List of CodeBlock objects
    """
    # Match fenced code blocks with optional language
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)

    return [CodeBlock(language=lang or "", content=code.strip()) for lang, code in matches]


def extract_lists(content: str) -> list[dict[str, Any]]:
    """
    Extract bullet/numbered lists.

    Args:
        content: Markdown content

    Returns:
        List of dicts with type and items
    """
    lists: list[dict[str, Any]] = []

    # Split into blocks
    blocks = re.split(r"\n\n+", content)

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        # Check first line for list type
        first_line = lines[0].strip()

        # Checkbox list
        if re.match(r"^-\s*\[[ xX]\]", first_line):
            items = []
            for line in lines:
                match = re.match(r"^-\s*\[([xX ])\]\s*(.+)$", line.strip())
                if match:
                    checked = match.group(1).lower() == "x"
                    items.append({"text": match.group(2), "checked": checked})
            if items:
                lists.append({"type": "checkbox", "items": items})

        # Bullet list
        elif re.match(r"^[-*+]\s+", first_line):
            items = []
            for line in lines:
                match = re.match(r"^[-*+]\s+(.+)$", line.strip())
                if match:
                    items.append(match.group(1))
            if items:
                lists.append({"type": "bullet", "items": items})

        # Numbered list
        elif re.match(r"^\d+\.\s+", first_line):
            items = []
            for line in lines:
                match = re.match(r"^\d+\.\s+(.+)$", line.strip())
                if match:
                    items.append(match.group(1))
            if items:
                lists.append({"type": "numbered", "items": items})

    return lists


def section_to_dict(section: Section) -> dict[str, Any]:
    """
    Convert a Section to a dictionary for YAML serialization.

    Args:
        section: Section object to convert

    Returns:
        Dictionary representation
    """
    result: dict[str, Any] = {}

    if section.title:
        result["title"] = section.title

    if section.content:
        # Extract any code blocks and lists from content
        code_blocks = extract_code_blocks(section.content)
        lists = extract_lists(section.content)

        # Clean content (remove code blocks for separate storage)
        clean_content = re.sub(r"```\w*\n.*?```", "", section.content, flags=re.DOTALL).strip()

        if clean_content:
            result["content"] = clean_content

        if code_blocks:
            result["code_blocks"] = [
                {"language": cb.language, "content": cb.content} for cb in code_blocks
            ]

        if lists:
            result["lists"] = lists

    if section.subsections:
        result["subsections"] = [section_to_dict(sub) for sub in section.subsections]

    return result
