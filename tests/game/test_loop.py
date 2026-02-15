"""Tests for autonomous game loop module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.game.loop import AutonomousGameLoop, GameLoopConfig, play_game
from nanobot.game.state_engine import GameRules


class MockRules(GameRules):
    """Mock game rules for testing."""

    def __init__(self, max_turns: int = 10):
        self._turn = 0
        self._max_turns = max_turns
        self._game_over = False

    def get_legal_moves(self, state: dict[str, Any]) -> list[str]:
        if self._game_over:
            return []
        return ["0", "1", "2", "3", "4"]

    def apply_move(self, state: dict[str, Any], move: str) -> dict[str, Any]:
        new_state = state.copy()
        new_state["last_move"] = move
        new_state["turn"] = state.get("turn", 0) + 1
        self._turn = new_state["turn"]
        if self._turn >= self._max_turns:
            self._game_over = True
        return new_state

    def check_win_conditions(self, state: dict[str, Any]) -> dict[str, Any]:
        turn = state.get("turn", 0)
        if turn >= self._max_turns or self._game_over:
            return {"game_over": True, "winner": "X", "status": "win"}
        return {"game_over": False, "winner": None, "status": "in_progress"}

    def get_next_player(self, state: dict[str, Any]) -> str:
        return "O" if state.get("current_player") == "X" else "X"


class TestGameLoopConfig:
    """Tests for GameLoopConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GameLoopConfig()

        assert config.max_turns == 100
        assert config.turn_timeout == 30
        assert config.auto_save_strategy is True
        assert config.log_all_moves is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GameLoopConfig(
            max_turns=50,
            turn_timeout=10,
            auto_save_strategy=False,
            log_all_moves=False,
        )

        assert config.max_turns == 50
        assert config.turn_timeout == 10
        assert config.auto_save_strategy is False
        assert config.log_all_moves is False


class TestAutonomousGameLoop:
    """Tests for AutonomousGameLoop class."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            # Create soul.yaml
            soul_data = {
                "version": "1.0",
                "name": "test-bot",
                "traits": [
                    {"name": "analytical", "weight": 1.3, "affects": ["reasoning_depth"]},
                ],
                "goals": [
                    {"name": "win_game", "priority": 10, "actions": ["attack", "win"]},
                ],
                "strategies": [],
                "game": {
                    "default_reasoning_depth": 2,
                    "monte_carlo_samples": 3,
                    "beam_width": 4,
                    "risk_tolerance": 0.4,
                },
            }
            (workspace / "soul.yaml").write_text(yaml.dump(soul_data))
            yield workspace

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock()
        provider.chat = AsyncMock()
        return provider

    @pytest.fixture
    def mock_rules(self):
        """Create mock game rules."""
        return MockRules(max_turns=5)

    @pytest.fixture
    def game_loop(self, mock_provider, mock_rules, temp_workspace):
        """Create a game loop for testing."""
        from nanobot.soul.loader import SoulLoader
        SoulLoader.reset_instance()

        loop = AutonomousGameLoop(
            provider=mock_provider,
            model="test-model",
            environment=mock_rules,
            workspace=temp_workspace,
            config=GameLoopConfig(max_turns=5),
            game_type="test_game",
            player_id="test_player",
        )
        return loop

    def test_init_creates_components(self, game_loop):
        """Test that init creates all required components."""
        assert game_loop._soul_loader is not None
        assert game_loop._trait_scorer is not None
        assert game_loop._memory_store is not None
        assert game_loop._strategy_memory is not None
        assert game_loop._game_memory is not None
        assert game_loop._state_engine is not None
        assert game_loop._reasoning_engine is not None
        assert game_loop._game_logger is not None

    def test_init_sets_properties(self, game_loop):
        """Test that init sets properties correctly."""
        assert game_loop.game_id is not None
        assert game_loop.turn_number == 0
        assert game_loop.is_running is False

    @pytest.mark.asyncio
    async def test_run_completes_game(self, game_loop):
        """Test that run completes a game."""
        # Mock the reasoning engine to return valid moves
        mock_state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="play 0", confidence=0.9, reasoning="best move"),
            ],
            entropy=0.3,
            strategic_direction="Play move 0",
        )
        game_loop._reasoning_engine.latent_reasoner.reason = AsyncMock(
            return_value=mock_state
        )

        result = await game_loop.run()

        assert "game_id" in result
        assert "outcome" in result
        assert "turns" in result
        assert "duration_seconds" in result
        assert "session_summary" in result

    @pytest.mark.asyncio
    async def test_run_respects_max_turns(self, game_loop):
        """Test that run respects max_turns config."""
        # Mock the reasoning engine
        mock_state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="play 0", confidence=0.9, reasoning="test"),
            ],
            entropy=0.3,
            strategic_direction="test",
        )
        game_loop._reasoning_engine.latent_reasoner.reason = AsyncMock(
            return_value=mock_state
        )

        result = await game_loop.run()

        # Should stop at or before max_turns (5)
        assert result["turns"] <= 5

    @pytest.mark.asyncio
    async def test_stop_halts_loop(self, game_loop):
        """Test that stop() halts the game loop."""
        # Mock the reasoning engine
        mock_state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="play 0", confidence=0.9, reasoning="test"),
            ],
            entropy=0.3,
            strategic_direction="test",
        )
        game_loop._reasoning_engine.latent_reasoner.reason = AsyncMock(
            return_value=mock_state
        )

        # Start the game in background and stop it
        import asyncio

        async def stop_after_delay():
            await asyncio.sleep(0.1)
            game_loop.stop()

        results = await asyncio.gather(
            game_loop.run(),
            stop_after_delay(),
            return_exceptions=True,
        )

        # Should have stopped
        assert game_loop.is_running is False

    def test_get_state(self, game_loop):
        """Test get_state returns current state."""
        state = game_loop.get_state()
        assert isinstance(state, dict)

    def test_get_history(self, game_loop):
        """Test get_history returns move history."""
        history = game_loop.get_history()
        assert isinstance(history, list)
        assert len(history) == 0  # No moves yet

    @pytest.mark.asyncio
    async def test_play_turn(self, game_loop):
        """Test _play_turn plays a single turn."""
        # Mock the reasoning engine
        mock_state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="play 0", confidence=0.9, reasoning="test"),
            ],
            entropy=0.3,
            strategic_direction="test",
        )
        game_loop._reasoning_engine.latent_reasoner.reason = AsyncMock(
            return_value=mock_state
        )

        game_state = game_loop.get_state()
        result = await game_loop._play_turn(game_state)

        assert result is not None
        assert "turn" in result
        assert "move" in result
        assert "player" in result

    def test_build_context_summary(self, game_loop):
        """Test _build_context_summary builds context."""
        game_state = {"turn": 5}
        memory_nodes = []

        context = game_loop._build_context_summary(game_state, memory_nodes)

        assert isinstance(context, str)
        assert "Turn: 0" in context or "Turn:" in context
        assert "Game Type: test_game" in context


class TestPlayGame:
    """Tests for play_game function."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            soul_data = {
                "version": "1.0",
                "name": "test-bot",
                "traits": [],
                "goals": [],
                "strategies": [],
                "game": {
                    "default_reasoning_depth": 2,
                    "monte_carlo_samples": 3,
                    "beam_width": 4,
                    "risk_tolerance": 0.4,
                },
            }
            (workspace / "soul.yaml").write_text(yaml.dump(soul_data))
            yield workspace

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock()
        provider.chat = AsyncMock()
        return provider

    @pytest.mark.asyncio
    async def test_play_game_completes(self, mock_provider, temp_workspace):
        """Test play_game convenience function completes."""
        from nanobot.soul.loader import SoulLoader
        SoulLoader.reset_instance()

        rules = MockRules(max_turns=3)

        # Mock reasoning
        with patch(
            "nanobot.game.reasoning_engine.GameReasoningEngine.select_best_move"
        ) as mock_select:
            mock_select.return_value = "0"

            result = await play_game(
                provider=mock_provider,
                model="test-model",
                environment=rules,
                workspace=temp_workspace,
                game_type="test",
                player_id="test_player",
                config=GameLoopConfig(max_turns=3),
            )

        assert "game_id" in result
        assert "outcome" in result

    @pytest.mark.asyncio
    async def test_play_game_with_initial_state(self, mock_provider, temp_workspace):
        """Test play_game with initial state."""
        from nanobot.soul.loader import SoulLoader
        SoulLoader.reset_instance()

        rules = MockRules(max_turns=3)

        with patch(
            "nanobot.game.reasoning_engine.GameReasoningEngine.select_best_move"
        ) as mock_select:
            mock_select.return_value = "0"

            result = await play_game(
                provider=mock_provider,
                model="test-model",
                environment=rules,
                workspace=temp_workspace,
                initial_state={"custom": "state", "turn": 0},
                config=GameLoopConfig(max_turns=3),
            )

        assert result is not None
