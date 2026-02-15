"""Game learning configurations and reward models loader.

Thread-safe singleton loader for game.yaml configuration.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class GameLoader:
    """Thread-safe singleton loader for game.yaml."""

    _instances: dict[str, "GameLoader"] = {}
    _lock = threading.Lock()

    def __init__(self, config_path: Path):
        """
        Initialize the GameLoader.

        Args:
            config_path: Path to game.yaml file
        """
        self._config_path = config_path
        self._config: dict[str, Any] | None = None
        self._config_lock = threading.Lock()

    @classmethod
    def get_instance(cls, config_path: Path) -> "GameLoader":
        """
        Get singleton instance for a given config path.

        Args:
            config_path: Path to game.yaml

        Returns:
            Singleton GameLoader instance
        """
        config_key = str(config_path.resolve())
        with cls._lock:
            if config_key not in cls._instances:
                cls._instances[config_key] = cls(config_path)
                logger.info(f"game.loader.get_instance path={config_key}")
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
        Load game learning configuration from YAML.

        Caches configuration after first load unless force_reload is True.
        Returns empty dict if file doesn't exist.

        Args:
            force_reload: Force re-reading file even if cached

        Returns:
            Game configuration dict
        """
        with self._config_lock:
            if self._config is not None and not force_reload:
                return self._config

            if not self._config_path.exists():
                logger.warning(
                    f"game.loader.load status=missing path={self._config_path}"
                )
                self._config = {}
                return self._config

            try:
                raw_data = yaml.safe_load(
                    self._config_path.read_text(encoding="utf-8")
                )
                self._config = raw_data if isinstance(raw_data, dict) else {}
                logger.info(
                    f"game.loader.load status=loaded path={self._config_path} "
                    f"configs={len(self._config.get('configurations', []))}"
                )
                return self._config
            except Exception as e:
                logger.error(
                    f"game.loader.load error={e} path={self._config_path}"
                )
                self._config = {}
                return self._config

    def get_configurations(self) -> list[dict[str, Any]]:
        """
        Get all game configurations.

        Returns:
            List of configuration dictionaries
        """
        config = self.load()
        return config.get("configurations", [])

    def get_reward_models(self) -> list[dict[str, Any]]:
        """
        Get all reward models.

        Returns:
            List of reward model dictionaries
        """
        config = self.load()
        return config.get("reward_models", [])

    def get_learning_config(self) -> dict[str, Any]:
        """
        Get learning configuration.

        Returns:
            Learning configuration dictionary
        """
        config = self.load()
        return config.get("learning", {})
