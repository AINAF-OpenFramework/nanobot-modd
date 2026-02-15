"""Governance policies and rules loader.

Thread-safe singleton loader for governance.yaml configuration.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class GovernanceLoader:
    """Thread-safe singleton loader for governance.yaml."""

    _instances: dict[str, "GovernanceLoader"] = {}
    _lock = threading.Lock()

    def __init__(self, config_path: Path):
        """
        Initialize the GovernanceLoader.

        Args:
            config_path: Path to governance.yaml file
        """
        self._config_path = config_path
        self._config: dict[str, Any] | None = None
        self._config_lock = threading.Lock()

    @classmethod
    def get_instance(cls, config_path: Path) -> "GovernanceLoader":
        """
        Get singleton instance for a given config path.

        Args:
            config_path: Path to governance.yaml

        Returns:
            Singleton GovernanceLoader instance
        """
        config_key = str(config_path.resolve())
        with cls._lock:
            if config_key not in cls._instances:
                cls._instances[config_key] = cls(config_path)
                logger.info(f"governance.loader.get_instance path={config_key}")
            return cls._instances[config_key]

    @classmethod
    def reset_instance(cls, config_path: Path | None = None) -> None:
        """
        Reset singleton instance (for testing).

        Args:
            config_path: Optional specific path to reset. If None, resets all.
        """
        with cls._lock:
            if config_path is None:
                cls._instances.clear()
            else:
                config_key = str(config_path.resolve())
                cls._instances.pop(config_key, None)

    def load(self, force_reload: bool = False) -> dict[str, Any]:
        """
        Load governance configuration from YAML.

        Caches configuration after first load unless force_reload is True.
        Returns empty dict if file doesn't exist.

        Args:
            force_reload: Force re-reading file even if cached

        Returns:
            Governance configuration dict
        """
        with self._config_lock:
            if self._config is not None and not force_reload:
                return self._config

            if not self._config_path.exists():
                logger.warning(
                    f"governance.loader.load status=missing path={self._config_path}"
                )
                self._config = {}
                return self._config

            try:
                raw_data = yaml.safe_load(
                    self._config_path.read_text(encoding="utf-8")
                )
                self._config = raw_data if isinstance(raw_data, dict) else {}
                logger.info(
                    f"governance.loader.load status=loaded path={self._config_path} "
                    f"policies={len(self._config.get('policies', []))}"
                )
                return self._config
            except Exception as e:
                logger.error(
                    f"governance.loader.load error={e} path={self._config_path}"
                )
                self._config = {}
                return self._config

    def get_policies(self) -> list[dict[str, Any]]:
        """
        Get all governance policies.

        Returns:
            List of policy dictionaries
        """
        config = self.load()
        return config.get("policies", [])

    def get_rules(self) -> list[dict[str, Any]]:
        """
        Get all governance rules.

        Returns:
            List of rule dictionaries
        """
        config = self.load()
        return config.get("rules", [])

    def get_constraints(self) -> dict[str, Any]:
        """
        Get governance constraints.

        Returns:
            Constraints dictionary
        """
        config = self.load()
        return config.get("constraints", {})
