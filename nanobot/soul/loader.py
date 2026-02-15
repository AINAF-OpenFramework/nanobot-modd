"""Soul configuration loader with thread-safe singleton pattern."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from nanobot.soul.schema import (
    GameConfig,
    Goal,
    PersonalityTrait,
    SoulConfig,
    Strategy,
)


class SoulLoader:
    """
    Thread-safe singleton loader for soul.yaml configuration.

    Provides methods to load and access the soul configuration including
    personality traits, goals, and strategies. The configuration is cached
    after first load for efficiency.

    CRITICAL: This loader reads ONLY soul.yaml, never soul.md.
    """

    _instances: dict[str, "SoulLoader"] = {}
    _lock = threading.Lock()

    def __init__(self, workspace: Path):
        """
        Initialize the SoulLoader.

        Args:
            workspace: Path to the workspace directory containing soul.yaml
        """
        self._workspace = workspace
        self._soul_file = workspace / "soul.yaml"
        self._config: SoulConfig | None = None
        self._config_lock = threading.Lock()

    @classmethod
    def get_instance(cls, workspace: Path) -> "SoulLoader":
        """
        Get the singleton instance for a given workspace.

        Thread-safe factory method that ensures only one SoulLoader
        exists per workspace path.

        Args:
            workspace: Path to the workspace directory

        Returns:
            The singleton SoulLoader instance for this workspace
        """
        workspace_key = str(workspace.resolve())
        with cls._lock:
            if workspace_key not in cls._instances:
                cls._instances[workspace_key] = cls(workspace)
                logger.info(f"soul.loader.get_instance workspace={workspace_key}")
            return cls._instances[workspace_key]

    @classmethod
    def reset_instance(cls, workspace: Path | None = None) -> None:
        """
        Reset the singleton instance for testing purposes.

        Args:
            workspace: Optional specific workspace to reset. If None, resets all.
        """
        with cls._lock:
            if workspace is None:
                cls._instances.clear()
            else:
                workspace_key = str(workspace.resolve())
                cls._instances.pop(workspace_key, None)

    def load(self, force_reload: bool = False) -> SoulConfig:
        """
        Load the soul configuration from soul.yaml.

        Caches the configuration after first load unless force_reload is True.
        Creates a default configuration if soul.yaml doesn't exist.

        Args:
            force_reload: Force re-reading the file even if cached

        Returns:
            The loaded SoulConfig
        """
        with self._config_lock:
            if self._config is not None and not force_reload:
                return self._config

            if not self._soul_file.exists():
                logger.info(
                    f"soul.loader.load status=creating_default path={self._soul_file}"
                )
                self._config = self._create_default_config()
                self._save_config(self._config)
                return self._config

            try:
                raw_data = yaml.safe_load(self._soul_file.read_text(encoding="utf-8"))
                self._config = self._parse_config(raw_data)
                logger.info(
                    f"soul.loader.load status=loaded path={self._soul_file} "
                    f"traits={len(self._config.traits)} goals={len(self._config.goals)}"
                )
                return self._config
            except Exception as e:
                logger.error(f"soul.loader.load error={e} path={self._soul_file}")
                self._config = self._create_default_config()
                return self._config

    def get_trait_weights(self) -> dict[str, float]:
        """
        Get a dictionary mapping trait names to their weights.

        Returns:
            Dictionary of trait name to weight mappings
        """
        config = self.load()
        return {trait.name: trait.weight for trait in config.traits}

    def get_active_goals(self) -> list[str]:
        """
        Get list of active goal names sorted by priority.

        Returns:
            List of goal names sorted by priority (highest first)
        """
        config = self.load()
        sorted_goals = sorted(config.goals, key=lambda g: g.priority, reverse=True)
        return [goal.name for goal in sorted_goals]

    def get_trait(self, name: str) -> PersonalityTrait | None:
        """
        Get a specific trait by name.

        Args:
            name: The trait name to look up

        Returns:
            The PersonalityTrait or None if not found
        """
        config = self.load()
        for trait in config.traits:
            if trait.name == name:
                return trait
        return None

    def get_goal(self, name: str) -> Goal | None:
        """
        Get a specific goal by name.

        Args:
            name: The goal name to look up

        Returns:
            The Goal or None if not found
        """
        config = self.load()
        for goal in config.goals:
            if goal.name == name:
                return goal
        return None

    def get_strategy(self, condition: str) -> Strategy | None:
        """
        Get a strategy by its condition.

        Args:
            condition: The condition to match

        Returns:
            The first matching Strategy or None
        """
        config = self.load()
        for strategy in config.strategies:
            if strategy.condition == condition:
                return strategy
        return None

    def _parse_config(self, raw_data: dict[str, Any]) -> SoulConfig:
        """Parse raw YAML data into SoulConfig."""
        traits = [
            PersonalityTrait(**t) for t in raw_data.get("traits", [])
        ]
        goals = [
            Goal(**g) for g in raw_data.get("goals", [])
        ]
        strategies = [
            Strategy(**s) for s in raw_data.get("strategies", [])
        ]

        game_data = raw_data.get("game", {})
        game_config = GameConfig(
            default_reasoning_depth=game_data.get("default_reasoning_depth", 2),
            monte_carlo_samples=game_data.get("monte_carlo_samples", 3),
            beam_width=game_data.get("beam_width", 4),
            risk_tolerance=game_data.get("risk_tolerance", 0.5),
        )

        return SoulConfig(
            version=raw_data.get("version", "1.0"),
            name=raw_data.get("name", "nanobot"),
            traits=traits,
            goals=goals,
            strategies=strategies,
            game=game_config,
            game_strategies=raw_data.get("game_strategies", {}),
        )

    def _create_default_config(self) -> SoulConfig:
        """Create a default SoulConfig."""
        return SoulConfig(
            version="1.0",
            name="nanobot",
            traits=[
                PersonalityTrait(
                    name="analytical",
                    weight=1.3,
                    description="Thorough analysis before action",
                    affects=["reasoning_depth"],
                ),
                PersonalityTrait(
                    name="cautious",
                    weight=1.1,
                    description="Careful risk assessment",
                    affects=["risk_tolerance"],
                ),
                PersonalityTrait(
                    name="adaptive",
                    weight=1.2,
                    description="Flexible strategy adjustment",
                    affects=[],
                ),
                PersonalityTrait(
                    name="efficient",
                    weight=1.4,
                    description="Resource-conscious decisions",
                    affects=[],
                ),
            ],
            goals=[
                Goal(
                    name="win_game",
                    priority=10,
                    description="Win the game",
                    actions=["attack", "win", "checkmate"],
                ),
                Goal(
                    name="avoid_loss",
                    priority=9,
                    description="Avoid losing",
                    actions=["defend", "block", "protect"],
                ),
                Goal(
                    name="conserve_tokens",
                    priority=7,
                    description="Conserve computational resources",
                    actions=["reuse_memory", "cache"],
                ),
            ],
            strategies=[
                Strategy(
                    name="aggressive_opening",
                    condition="early_game",
                    approach="aggressive",
                    traits_boost={"aggressive": 0.3},
                ),
                Strategy(
                    name="defensive_recovery",
                    condition="losing",
                    approach="defensive",
                    traits_boost={"cautious": 0.4},
                ),
                Strategy(
                    name="calculated_endgame",
                    condition="endgame",
                    approach="calculated",
                    traits_boost={"analytical": 0.5},
                ),
            ],
            game=GameConfig(
                default_reasoning_depth=2,
                monte_carlo_samples=3,
                beam_width=4,
                risk_tolerance=0.4,
            ),
        )

    def _save_config(self, config: SoulConfig) -> None:
        """Save configuration to soul.yaml."""
        data = {
            "version": config.version,
            "name": config.name,
            "traits": [
                {
                    "name": t.name,
                    "weight": t.weight,
                    "description": t.description,
                    "affects": t.affects,
                }
                for t in config.traits
            ],
            "goals": [
                {
                    "name": g.name,
                    "priority": g.priority,
                    "description": g.description,
                    "actions": g.actions,
                }
                for g in config.goals
            ],
            "strategies": [
                {
                    "name": s.name,
                    "condition": s.condition,
                    "approach": s.approach,
                    "traits_boost": s.traits_boost,
                }
                for s in config.strategies
            ],
            "game": {
                "default_reasoning_depth": config.game.default_reasoning_depth,
                "monte_carlo_samples": config.game.monte_carlo_samples,
                "beam_width": config.game.beam_width,
                "risk_tolerance": config.game.risk_tolerance,
            },
            "game_strategies": config.game_strategies,
        }

        self._soul_file.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        logger.info(f"soul.loader.save path={self._soul_file}")
