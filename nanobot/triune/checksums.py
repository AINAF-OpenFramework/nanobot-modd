"""Checksum management for Triune Memory System.

Provides MD5/SHA256 checksums for tracking MD ↔ YAML sync integrity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FileChecksum:
    """Checksum metadata for a file pair."""

    md_path: str
    yaml_path: str
    md_checksum: str
    yaml_checksum: str
    algorithm: str = "md5"
    last_verified: str | None = None


class ChecksumManager:
    """Manages checksums for MD-YAML file pairs."""

    def __init__(self, checksums_file: Path):
        """
        Initialize checksum manager.

        Args:
            checksums_file: Path to checksums.json storage
        """
        self._checksums_file = checksums_file
        self._checksums: dict[str, FileChecksum] = {}
        self._load_checksums()

    def _load_checksums(self) -> None:
        """Load checksums from storage."""
        if not self._checksums_file.exists():
            return

        try:
            data = json.loads(self._checksums_file.read_text(encoding="utf-8"))
            for key, item in data.items():
                self._checksums[key] = FileChecksum(**item)
            logger.info(f"Loaded {len(self._checksums)} checksums")
        except Exception as e:
            logger.error(f"Failed to load checksums: {e}")

    def save_checksums(self) -> bool:
        """
        Save checksums to storage.

        Returns:
            True if successful
        """
        try:
            self._checksums_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                key: {
                    "md_path": cs.md_path,
                    "yaml_path": cs.yaml_path,
                    "md_checksum": cs.md_checksum,
                    "yaml_checksum": cs.yaml_checksum,
                    "algorithm": cs.algorithm,
                    "last_verified": cs.last_verified,
                }
                for key, cs in self._checksums.items()
            }
            self._checksums_file.write_text(
                json.dumps(data, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save checksums: {e}")
            return False

    def compute_checksum(self, file_path: Path, algorithm: str = "md5") -> str | None:
        """
        Compute checksum for a file.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5 or sha256)

        Returns:
            Hex digest string or None if error
        """
        if not file_path.exists():
            return None

        try:
            if algorithm == "md5":
                hasher = hashlib.md5()
            elif algorithm == "sha256":
                hasher = hashlib.sha256()
            else:
                logger.error(f"Unsupported algorithm: {algorithm}")
                return None

            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)

            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute checksum for {file_path}: {e}")
            return None

    def update_checksums(
        self,
        md_path: Path,
        yaml_path: Path,
        algorithm: str = "md5",
    ) -> bool:
        """
        Update checksums for a MD-YAML pair.

        Args:
            md_path: Path to .md file
            yaml_path: Path to .yaml file
            algorithm: Hash algorithm

        Returns:
            True if successful
        """
        md_checksum = self.compute_checksum(md_path, algorithm)
        yaml_checksum = self.compute_checksum(yaml_path, algorithm)

        if md_checksum is None or yaml_checksum is None:
            return False

        key = str(md_path)
        self._checksums[key] = FileChecksum(
            md_path=str(md_path),
            yaml_path=str(yaml_path),
            md_checksum=md_checksum,
            yaml_checksum=yaml_checksum,
            algorithm=algorithm,
            last_verified=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        return True

    def verify_pair(self, md_path: Path, yaml_path: Path) -> tuple[bool, str]:
        """
        Verify if MD-YAML pair checksums match stored values.

        Args:
            md_path: Path to .md file
            yaml_path: Path to .yaml file

        Returns:
            Tuple of (is_valid, reason)
        """
        key = str(md_path)
        stored = self._checksums.get(key)

        if stored is None:
            return False, "no_checksum_record"

        if not md_path.exists():
            return False, "md_missing"

        if not yaml_path.exists():
            return False, "yaml_missing"

        # Compute current checksums
        md_checksum = self.compute_checksum(md_path, stored.algorithm)
        yaml_checksum = self.compute_checksum(yaml_path, stored.algorithm)

        if md_checksum is None or yaml_checksum is None:
            return False, "checksum_computation_failed"

        if md_checksum != stored.md_checksum:
            return False, "md_modified"

        if yaml_checksum != stored.yaml_checksum:
            return False, "yaml_modified"

        return True, "valid"

    def get_all_checksums(self) -> dict[str, FileChecksum]:
        """Get all stored checksums."""
        return self._checksums.copy()

    def remove_checksum(self, md_path: Path) -> bool:
        """
        Remove checksum record for a file pair.

        Args:
            md_path: Path to .md file

        Returns:
            True if removed
        """
        key = str(md_path)
        if key in self._checksums:
            del self._checksums[key]
            return True
        return False
