"""Tests for SoulLoader module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from nanobot.soul.loader import SoulLoader
from nanobot.soul.schema import SoulConfig


class TestSoulLoader:
    """Tests for SoulLoader class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before and after each test."""
        SoulLoader.reset_instance()
        yield
        SoulLoader.reset_instance()

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def workspace_with_soul_yaml(self, temp_workspace):
        """Create a workspace with a soul.yaml file."""
        soul_data = {
            "version": "1.0",
            "name": "test-bot",
            "traits": [
                {
                    "name": "analytical",
                    "weight": 1.5,
                    "description": "Test trait",
                    "affects": ["reasoning_depth"],
                },
                {
                    "name": "cautious",
                    "weight": 1.2,
                    "description": "Cautious trait",
                    "affects": ["risk_tolerance"],
                },
            ],
            "goals": [
                {
                    "name": "win_game",
                    "priority": 10,
                    "description": "Win the game",
                    "actions": ["attack", "win"],
                },
                {
                    "name": "survive",
                    "priority": 5,
                    "description": "Stay alive",
                    "actions": ["defend"],
                },
            ],
            "strategies": [
                {
                    "name": "aggressive_opening",
                    "condition": "early_game",
                    "approach": "aggressive",
                    "traits_boost": {"aggressive": 0.3},
                },
            ],
            "game": {
                "default_reasoning_depth": 3,
                "monte_carlo_samples": 5,
                "beam_width": 6,
                "risk_tolerance": 0.3,
            },
        }
        soul_file = temp_workspace / "soul.yaml"
        soul_file.write_text(yaml.dump(soul_data), encoding="utf-8")
        return temp_workspace

    def test_load_creates_default_when_missing(self, temp_workspace):
        """Test that load creates a default config when soul.yaml is missing."""
        loader = SoulLoader(temp_workspace)
        config = loader.load()

        assert config is not None
        assert isinstance(config, SoulConfig)
        assert config.version == "1.0"
        assert len(config.traits) > 0
        assert len(config.goals) > 0

        # Check that file was created
        assert (temp_workspace / "soul.yaml").exists()

    def test_load_parses_yaml_correctly(self, workspace_with_soul_yaml):
        """Test that load correctly parses an existing soul.yaml."""
        loader = SoulLoader(workspace_with_soul_yaml)
        config = loader.load()

        assert config.version == "1.0"
        assert config.name == "test-bot"
        assert len(config.traits) == 2
        assert len(config.goals) == 2
        assert len(config.strategies) == 1

        # Check trait parsing
        analytical = next(t for t in config.traits if t.name == "analytical")
        assert analytical.weight == 1.5
        assert "reasoning_depth" in analytical.affects

        # Check goal parsing
        win_goal = next(g for g in config.goals if g.name == "win_game")
        assert win_goal.priority == 10
        assert "attack" in win_goal.actions

        # Check game config
        assert config.game.default_reasoning_depth == 3
        assert config.game.monte_carlo_samples == 5

    def test_load_caches_config(self, workspace_with_soul_yaml):
        """Test that load caches the config after first load."""
        loader = SoulLoader(workspace_with_soul_yaml)

        config1 = loader.load()
        config2 = loader.load()

        # Same object should be returned (cached)
        assert config1 is config2

    def test_load_force_reload(self, workspace_with_soul_yaml):
        """Test that force_reload=True re-reads the file."""
        loader = SoulLoader(workspace_with_soul_yaml)

        config1 = loader.load()

        # Modify the file
        soul_file = workspace_with_soul_yaml / "soul.yaml"
        data = yaml.safe_load(soul_file.read_text())
        data["name"] = "modified-bot"
        soul_file.write_text(yaml.dump(data))

        # Force reload
        config2 = loader.load(force_reload=True)

        assert config2.name == "modified-bot"

    def test_get_trait_weights(self, workspace_with_soul_yaml):
        """Test get_trait_weights returns correct mapping."""
        loader = SoulLoader(workspace_with_soul_yaml)
        weights = loader.get_trait_weights()

        assert isinstance(weights, dict)
        assert "analytical" in weights
        assert weights["analytical"] == 1.5
        assert "cautious" in weights
        assert weights["cautious"] == 1.2

    def test_get_active_goals(self, workspace_with_soul_yaml):
        """Test get_active_goals returns sorted goals."""
        loader = SoulLoader(workspace_with_soul_yaml)
        goals = loader.get_active_goals()

        assert isinstance(goals, list)
        assert len(goals) == 2
        # Should be sorted by priority (highest first)
        assert goals[0] == "win_game"  # priority 10
        assert goals[1] == "survive"  # priority 5

    def test_singleton_pattern(self, temp_workspace):
        """Test that get_instance returns the same instance."""
        instance1 = SoulLoader.get_instance(temp_workspace)
        instance2 = SoulLoader.get_instance(temp_workspace)

        assert instance1 is instance2

    def test_singleton_different_workspaces(self, temp_workspace):
        """Test that different workspaces get different instances."""
        with tempfile.TemporaryDirectory() as other_dir:
            other_workspace = Path(other_dir)

            instance1 = SoulLoader.get_instance(temp_workspace)
            instance2 = SoulLoader.get_instance(other_workspace)

            assert instance1 is not instance2

    def test_get_trait(self, workspace_with_soul_yaml):
        """Test get_trait returns correct trait."""
        loader = SoulLoader(workspace_with_soul_yaml)

        trait = loader.get_trait("analytical")
        assert trait is not None
        assert trait.name == "analytical"
        assert trait.weight == 1.5

        # Non-existent trait
        assert loader.get_trait("nonexistent") is None

    def test_get_goal(self, workspace_with_soul_yaml):
        """Test get_goal returns correct goal."""
        loader = SoulLoader(workspace_with_soul_yaml)

        goal = loader.get_goal("win_game")
        assert goal is not None
        assert goal.name == "win_game"
        assert goal.priority == 10

        # Non-existent goal
        assert loader.get_goal("nonexistent") is None

    def test_get_strategy(self, workspace_with_soul_yaml):
        """Test get_strategy returns correct strategy."""
        loader = SoulLoader(workspace_with_soul_yaml)

        strategy = loader.get_strategy("early_game")
        assert strategy is not None
        assert strategy.name == "aggressive_opening"
        assert strategy.condition == "early_game"

        # Non-existent condition
        assert loader.get_strategy("nonexistent") is None

    def test_reset_instance_specific(self, temp_workspace):
        """Test reset_instance with specific workspace."""
        with tempfile.TemporaryDirectory() as other_dir:
            other_workspace = Path(other_dir)

            instance1 = SoulLoader.get_instance(temp_workspace)
            instance2 = SoulLoader.get_instance(other_workspace)

            # Reset only temp_workspace
            SoulLoader.reset_instance(temp_workspace)

            # New instance for temp_workspace
            instance1_new = SoulLoader.get_instance(temp_workspace)
            instance2_same = SoulLoader.get_instance(other_workspace)

            assert instance1 is not instance1_new
            assert instance2 is instance2_same

    def test_thread_safety(self, temp_workspace):
        """Test that singleton is thread-safe."""
        import threading

        instances = []
        errors = []

        def get_loader():
            try:
                instance = SoulLoader.get_instance(temp_workspace)
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_loader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(instances) == 10
        # All should be the same instance
        assert all(i is instances[0] for i in instances)
