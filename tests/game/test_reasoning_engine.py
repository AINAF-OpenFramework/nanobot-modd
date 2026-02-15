"""Tests for game reasoning engine module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.game.reasoning_engine import GameReasoningEngine
from nanobot.game.state_engine import GameRules, GameStateEngine


class MockRules(GameRules):
    """Mock rules for testing."""

    def get_legal_moves(self, state: dict[str, Any]) -> list[str]:
        return state.get("legal_moves", ["0", "1", "2"])

    def apply_move(self, state: dict[str, Any], move: str) -> dict[str, Any]:
        new_state = state.copy()
        new_state["last_move"] = move
        return new_state

    def check_win_conditions(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"game_over": False, "winner": None, "status": "In progress"}

    def get_next_player(self, state: dict[str, Any]) -> str:
        return "O" if state.get("current_player") == "X" else "X"


class TestGameReasoningEngine:
    """Tests for GameReasoningEngine class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock()
        provider.chat = AsyncMock()
        return provider

    @pytest.fixture
    def mock_state_engine(self):
        """Create a mock state engine."""
        rules = MockRules()
        engine = GameStateEngine(
            game_id="test-game",
            rules=rules,
            initial_state={"board": [""] * 9, "current_player": "X", "legal_moves": ["0", "1", "2"]},
        )
        return engine

    @pytest.fixture
    def reasoning_engine(self, mock_provider, mock_state_engine):
        """Create a reasoning engine for testing."""
        return GameReasoningEngine(
            provider=mock_provider,
            model="test-model",
            state_engine=mock_state_engine,
            timeout_seconds=5,
            memory_config={"clarify_entropy_threshold": 0.5},
        )

    def test_init_creates_latent_reasoner(self, mock_provider, mock_state_engine):
        """Test that init creates a LatentReasoner."""
        engine = GameReasoningEngine(
            provider=mock_provider,
            model="gpt-4",
            state_engine=mock_state_engine,
        )

        assert engine.latent_reasoner is not None
        assert engine.provider == mock_provider
        assert engine.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_select_best_move_returns_move(self, reasoning_engine, mock_provider):
        """Test select_best_move returns a valid move."""
        # Mock the latent reasoner's reason method
        mock_state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="move 1", confidence=0.9, reasoning="best move"),
                Hypothesis(intent="move 0", confidence=0.5, reasoning="okay move"),
            ],
            entropy=0.5,
            strategic_direction="Play move 1",
        )
        reasoning_engine.latent_reasoner.reason = AsyncMock(return_value=mock_state)

        result = await reasoning_engine.select_best_move()

        assert result in ["0", "1", "2"]
        reasoning_engine.latent_reasoner.reason.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_select_best_move_no_legal_moves(self, reasoning_engine, mock_state_engine):
        """Test select_best_move returns None when no legal moves."""
        # Update state to have no legal moves
        mock_state_engine.update({"legal_moves": []})

        result = await reasoning_engine.select_best_move()

        assert result is None

    @pytest.mark.asyncio
    async def test_select_best_move_calls_latent_reasoner(self, reasoning_engine):
        """Test that select_best_move calls latent_reasoner.reason()."""
        mock_state = SuperpositionalState(
            hypotheses=[Hypothesis(intent="0", confidence=0.9, reasoning="test")],
            entropy=0.3,
            strategic_direction="Play 0",
        )
        reasoning_engine.latent_reasoner.reason = AsyncMock(return_value=mock_state)

        await reasoning_engine.select_best_move(context_summary="test context")

        reasoning_engine.latent_reasoner.reason.assert_awaited_once()
        call_args = reasoning_engine.latent_reasoner.reason.call_args
        assert "test context" in call_args[0][1] or any(
            "test context" in str(v) for v in call_args[1].values()
        )

    @pytest.mark.asyncio
    async def test_evaluate_moves_returns_scores(self, reasoning_engine):
        """Test evaluate_moves returns confidence scores."""
        mock_state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="0", confidence=0.8, reasoning="good"),
                Hypothesis(intent="1", confidence=0.6, reasoning="okay"),
            ],
            entropy=0.4,
            strategic_direction="Evaluate moves",
        )
        reasoning_engine.latent_reasoner.reason = AsyncMock(return_value=mock_state)

        result = await reasoning_engine.evaluate_moves()

        assert isinstance(result, dict)
        assert "0" in result
        assert "1" in result
        assert "2" in result  # All legal moves should be present
        assert all(0.0 <= score <= 1.0 for score in result.values())

    @pytest.mark.asyncio
    async def test_evaluate_moves_empty_when_no_moves(self, reasoning_engine, mock_state_engine):
        """Test evaluate_moves returns empty dict when no legal moves."""
        mock_state_engine.update({"legal_moves": []})

        result = await reasoning_engine.evaluate_moves()

        assert result == {}

    @pytest.mark.asyncio
    async def test_evaluate_specific_moves(self, reasoning_engine):
        """Test evaluate_moves with specific moves list."""
        mock_state = SuperpositionalState(
            hypotheses=[Hypothesis(intent="0", confidence=0.7, reasoning="test")],
            entropy=0.3,
            strategic_direction="Test",
        )
        reasoning_engine.latent_reasoner.reason = AsyncMock(return_value=mock_state)

        result = await reasoning_engine.evaluate_moves(moves=["0", "1"])

        assert "0" in result
        assert "1" in result
        assert "2" not in result  # Not in the specified list

    @pytest.mark.asyncio
    async def test_simulate_future_states(self, reasoning_engine):
        """Test simulate_future_states returns simulations."""
        mock_state = SuperpositionalState(
            hypotheses=[Hypothesis(intent="response", confidence=0.7, reasoning="prediction")],
            entropy=0.4,
            strategic_direction="Opponent likely responds with...",
        )
        reasoning_engine.latent_reasoner.reason = AsyncMock(return_value=mock_state)

        result = await reasoning_engine.simulate_future_states(depth=2)

        assert isinstance(result, list)
        assert len(result) <= 3  # Limited to top 3 moves
        for sim in result:
            assert "initial_move" in sim
            assert "simulated_state" in sim
            assert "reasoning" in sim
            assert "entropy" in sim

    @pytest.mark.asyncio
    async def test_simulate_future_states_empty_when_no_moves(
        self, reasoning_engine, mock_state_engine
    ):
        """Test simulate returns empty list when no legal moves."""
        mock_state_engine.update({"legal_moves": []})

        result = await reasoning_engine.simulate_future_states()

        assert result == []

    def test_extract_move_from_reasoning(self, reasoning_engine):
        """Test _extract_move_from_reasoning finds matching move."""
        state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="play position 1", confidence=0.9, reasoning="test"),
            ],
            entropy=0.3,
            strategic_direction="test",
        )

        result = reasoning_engine._extract_move_from_reasoning(state, ["0", "1", "2"])

        assert result == "1"

    def test_extract_move_fallback_to_first(self, reasoning_engine):
        """Test _extract_move_from_reasoning falls back to first move."""
        state = SuperpositionalState(
            hypotheses=[],
            entropy=0.0,
            strategic_direction="test",
        )

        result = reasoning_engine._extract_move_from_reasoning(state, ["a", "b", "c"])

        assert result == "a"

    def test_extract_move_evaluations(self, reasoning_engine):
        """Test _extract_move_evaluations extracts scores."""
        state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="move 0 is good", confidence=0.9, reasoning="test"),
                Hypothesis(intent="move 2", confidence=0.7, reasoning="test"),
            ],
            entropy=0.4,
            strategic_direction="test",
        )

        result = reasoning_engine._extract_move_evaluations(state, ["0", "1", "2"])

        assert result["0"] == 0.9
        assert result["1"] == 0.5  # Default
        assert result["2"] == 0.7
