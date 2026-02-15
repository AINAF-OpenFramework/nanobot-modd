"""Sync verification engine for Triune Memory System.

Detects drift between MD and YAML files, validates sync integrity,
and provides detailed status reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nanobot.triune.checksums import ChecksumManager
from nanobot.utils.translator import (
    get_yaml_path,
    needs_sync,
    parse_md_to_yaml,
)

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verification scan."""

    total_md_files: int = 0
    synced_yaml_files: int = 0
    missing_yaml: list[str] = field(default_factory=list)
    drifted_files: list[tuple[str, str]] = field(default_factory=list)
    orphaned_yaml: list[str] = field(default_factory=list)
    invalid_yaml: list[str] = field(default_factory=list)
    last_verification: str | None = None

    @property
    def sync_status(self) -> str:
        """Get overall sync status."""
        if self.total_md_files == 0:
            return "no_files"
        if self.synced_yaml_files == self.total_md_files:
            if not self.drifted_files:
                return "fully_synced"
            return "synced_with_drift"
        if self.synced_yaml_files == 0:
            return "no_sync"
        return "partial_sync"


class TriuneVerifier:
    """Verifies Triune Memory sync integrity."""

    def __init__(
        self,
        root_path: Path,
        checksums_file: Path,
        excluded_files: list[str] | None = None,
    ):
        """
        Initialize verifier.

        Args:
            root_path: Root directory to scan
            checksums_file: Path to checksums.json
            excluded_files: List of files to exclude
        """
        self._root_path = root_path
        self._checksum_mgr = ChecksumManager(checksums_file)
        self._excluded_files = excluded_files or [
            "README.md",
            "CHANGELOG.md",
            "IMPLEMENTATION_SUMMARY.md",
        ]

    def verify_all(self, fix: bool = False) -> VerificationResult:
        """
        Verify all MD-YAML pairs in repository.

        Args:
            fix: If True, auto-regenerate drifted YAML files

        Returns:
            VerificationResult with status and details
        """
        result = VerificationResult()
        result.last_verification = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Find all .md files recursively
        md_files = list(self._root_path.rglob("*.md"))
        result.total_md_files = len(md_files)

        # Track all YAML files to detect orphans
        yaml_files = set(self._root_path.rglob("*.yaml"))
        matched_yaml = set()

        for md_path in md_files:
            # Skip excluded files
            if md_path.name in self._excluded_files:
                continue

            yaml_path = get_yaml_path(md_path)
            matched_yaml.add(yaml_path)

            # Check if YAML exists
            if not yaml_path.exists():
                if fix:
                    if self._fix_missing_yaml(md_path, yaml_path):
                        result.synced_yaml_files += 1
                    else:
                        result.missing_yaml.append(str(md_path.relative_to(self._root_path)))
                else:
                    result.missing_yaml.append(str(md_path.relative_to(self._root_path)))
                continue

            result.synced_yaml_files += 1

            # Validate YAML can be parsed
            if not self._validate_yaml(yaml_path):
                result.invalid_yaml.append(str(yaml_path.relative_to(self._root_path)))
                continue

            # Check for drift
            is_drifted, reason = self._check_drift(md_path, yaml_path)
            if is_drifted:
                result.drifted_files.append(
                    (
                        str(md_path.relative_to(self._root_path)),
                        reason,
                    )
                )
                if fix:
                    self._fix_drifted_yaml(md_path, yaml_path)

        # Find orphaned YAML files
        for yaml_path in yaml_files:
            if yaml_path not in matched_yaml:
                result.orphaned_yaml.append(str(yaml_path.relative_to(self._root_path)))

        return result

    def _validate_yaml(self, yaml_path: Path) -> bool:
        """
        Validate that YAML file is parseable.

        Args:
            yaml_path: Path to YAML file

        Returns:
            True if valid
        """
        try:
            yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            return True
        except Exception as e:
            logger.warning(f"Invalid YAML {yaml_path}: {e}")
            return False

    def _check_drift(self, md_path: Path, yaml_path: Path) -> tuple[bool, str]:
        """
        Check if MD-YAML pair has drifted.

        Args:
            md_path: Path to .md file
            yaml_path: Path to .yaml file

        Returns:
            Tuple of (is_drifted, reason)
        """
        # First check timestamps
        if needs_sync(md_path, yaml_path):
            md_time = datetime.fromtimestamp(md_path.stat().st_mtime)
            yaml_time = datetime.fromtimestamp(yaml_path.stat().st_mtime)
            age = (md_time - yaml_time).total_seconds()
            return True, f"md_newer_by_{int(age)}s"

        # Then check checksums
        is_valid, reason = self._checksum_mgr.verify_pair(md_path, yaml_path)
        if not is_valid:
            if reason == "no_checksum_record":
                # No checksum yet, create one
                self._checksum_mgr.update_checksums(md_path, yaml_path)
                self._checksum_mgr.save_checksums()
                return False, "ok"
            return True, reason

        return False, "ok"

    def _fix_missing_yaml(self, md_path: Path, yaml_path: Path) -> bool:
        """
        Generate missing YAML file.

        Args:
            md_path: Path to .md file
            yaml_path: Path to .yaml file

        Returns:
            True if successful
        """
        logger.info(f"Fixing missing YAML: {md_path} → {yaml_path}")
        success = parse_md_to_yaml(md_path, yaml_path)
        if success:
            self._checksum_mgr.update_checksums(md_path, yaml_path)
            self._checksum_mgr.save_checksums()
        return success

    def _fix_drifted_yaml(self, md_path: Path, yaml_path: Path) -> bool:
        """
        Regenerate drifted YAML file.

        Args:
            md_path: Path to .md file
            yaml_path: Path to .yaml file

        Returns:
            True if successful
        """
        logger.info(f"Fixing drifted YAML: {md_path} → {yaml_path}")
        success = parse_md_to_yaml(md_path, yaml_path)
        if success:
            self._checksum_mgr.update_checksums(md_path, yaml_path)
            self._checksum_mgr.save_checksums()
        return success

    def get_detailed_report(self, result: VerificationResult) -> str:
        """
        Generate detailed text report.

        Args:
            result: Verification result

        Returns:
            Formatted report string
        """
        lines = [
            "Triune Sync Status:",
            f"✓ {result.synced_yaml_files} MD files synced",
        ]

        if result.missing_yaml:
            lines.append(f"✗ {len(result.missing_yaml)} MD files missing YAML")
        if result.drifted_files:
            lines.append(f"⚠ {len(result.drifted_files)} YAML files drifted from MD")
        if result.orphaned_yaml:
            lines.append(f"○ {len(result.orphaned_yaml)} orphaned YAML files")
        if result.invalid_yaml:
            lines.append(f"✗ {len(result.invalid_yaml)} invalid YAML files")

        if result.missing_yaml or result.drifted_files or result.orphaned_yaml:
            lines.append("\nDetails:")

        if result.missing_yaml:
            lines.append("- Missing YAML:")
            for path in result.missing_yaml[:10]:  # Limit to first 10
                lines.append(f"  • {path}")
            if len(result.missing_yaml) > 10:
                lines.append(f"  ... and {len(result.missing_yaml) - 10} more")

        if result.drifted_files:
            lines.append("- Drifted:")
            for path, reason in result.drifted_files[:10]:
                lines.append(f"  • {path} ({reason})")
            if len(result.drifted_files) > 10:
                lines.append(f"  ... and {len(result.drifted_files) - 10} more")

        if result.orphaned_yaml:
            lines.append("- Orphaned:")
            for path in result.orphaned_yaml[:10]:
                lines.append(f"  • {path}")
            if len(result.orphaned_yaml) > 10:
                lines.append(f"  ... and {len(result.orphaned_yaml) - 10} more")

        return "\n".join(lines)
