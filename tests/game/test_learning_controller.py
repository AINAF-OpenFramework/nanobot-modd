"""Tests for learning controller module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.game.environment import ActionResult, GameState
from nanobot.game.learning_controller import (
    GameLearningController,
    LearningConfig,
    LearningState,
)
from nanobot.game.state_engine import GameRules


class MockRules(GameRules):
    """Mock game rules for testing."""

    def get_legal_moves(self, state: dict[str, Any]) -> list[str]:
        return state.get("legal_moves", ["0", "1", "2"])

    def apply_move(self, state: dict[str, Any], move: str) -> dict[str, Any]:
        new_state = state.copy()
        new_state["last_move"] = move
        new_state["turn_number"] = new_state.get("turn_number", 0) + 1
        return new_state

    def check_win_conditions(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("winner"):
            return {"game_over": True, "winner": state["winner"], "status": "Won"}
        return {"game_over": False, "winner": None, "status": "In progress"}

    def get_next_player(self, state: dict[str, Any]) -> str:
        current = state.get("current_player", "X")
        return "O" if current == "X" else "X"


class TestLearningConfig:
    """Tests for LearningConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LearningConfig()
        assert config.visual_encoder_type == "auto"
        assert config.visual_embedding_dim == 256
        assert config.reasoning_timeout == 10
        assert config.memory_top_k == 5
        assert config.store_all_moves is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LearningConfig(
            visual_encoder_type="grid",
            visual_embedding_dim=128,
            reasoning_timeout=20,
            fusion_weights=(0.5, 0.25, 0.25),
        )
        assert config.visual_encoder_type == "grid"
        assert config.visual_embedding_dim == 128
        assert config.reasoning_timeout == 20
        assert config.fusion_weights == (0.5, 0.25, 0.25)


class TestLearningState:
    """Tests for LearningState dataclass."""

    def test_init(self):
        """Test initial state."""
        state = LearningState(game_id="test-game")
        assert state.game_id == "test-game"
        assert state.episode == 0
        assert state.total_moves == 0
        assert state.wins == 0
        assert state.losses == 0
        assert state.draws == 0

    def test_record_move(self):
        """Test recording a move."""
        state = LearningState(game_id="test")
        state.record_move(
            move="4",
            reward=0.1,
            state_before={"board": []},
            state_after={"board": ["X"]},
        )

        assert state.total_moves == 1
        assert state.total_reward == 0.1
        assert len(state.move_history) == 1
        assert state.move_history[0]["move"] == "4"

    def test_record_multiple_moves(self):
        """Test recording multiple moves."""
        state = LearningState(game_id="test")
        state.record_move("1", 0.1)
        state.record_move("2", 0.2)
        state.record_move("3", 0.3)

        assert state.total_moves == 3
        assert abs(state.total_reward - 0.6) < 0.01
        assert len(state.move_history) == 3

    def test_record_game_end_win(self):
        """Test recording a win."""
        state = LearningState(game_id="test")
        state.record_move("1", 0.0)
        state.record_game_end("win")

        assert state.wins == 1
        assert state.episode == 1
        assert len(state.move_history) == 0  # Reset for next episode

    def test_record_game_end_loss(self):
        """Test recording a loss."""
        state = LearningState(game_id="test")
        state.record_game_end("loss")

        assert state.losses == 1
        assert state.episode == 1

    def test_record_game_end_draw(self):
        """Test recording a draw."""
        state = LearningState(game_id="test")
        state.record_game_end("draw")

        assert state.draws == 1
        assert state.episode == 1

    def test_get_stats(self):
        """Test getting statistics."""
        state = LearningState(game_id="test")
        state.record_move("1", 0.5)
        state.record_move("2", 0.5)
        state.record_game_end("win")
        state.record_move("3", -0.5)
        state.record_game_end("loss")

        stats = state.get_stats()

        assert stats["game_id"] == "test"
        assert stats["episode"] == 2
        assert stats["total_moves"] == 3
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 0.5

    def test_get_stats_no_games(self):
        """Test getting stats with no completed games."""
        state = LearningState(game_id="test")
        stats = state.get_stats()

        assert stats["win_rate"] == 0.0
        assert stats["avg_moves_per_game"] == 0


class TestGameLearningController:
    """Tests for GameLearningController class."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock()
        provider.chat = AsyncMock()
        return provider

    @pytest.fixture
    def mock_environment(self):
        """Create a mock game environment."""
        env = MagicMock()
        env.game_id = "test-game"
        env.game_type = "tictactoe"
        env.get_state.return_value = GameState(
            board=[["", "", ""], ["", "", ""], ["", "", ""]],
            current_player="X",
            turn_number=0,
            legal_moves=["0,0", "0,1", "0,2", "1,0", "1,1", "1,2", "2,0", "2,1", "2,2"],
            game_over=False,
            winner=None,
        )
        env.get_screenshot.return_value = None
        env.execute_action.return_value = ActionResult(
            success=True,
            new_state=GameState(
                board=[["X", "", ""], ["", "", ""], ["", "", ""]],
                current_player="O",
                turn_number=1,
                legal_moves=["0,1", "0,2", "1,0", "1,1", "1,2", "2,0", "2,1", "2,2"],
                game_over=False,
            ),
            reward=0.0,
            message="Move executed",
        )
        return env

    @pytest.fixture
    def controller(self, mock_provider, mock_environment, temp_workspace):
        """Create a learning controller for testing."""
        config = LearningConfig(
            visual_encoder_type="grid",  # Use grid encoder (no PyTorch needed)
            store_all_moves=True,
        )
        return GameLearningController(
            provider=mock_provider,
            model="test-model",
            environment=mock_environment,
            workspace=temp_workspace,
            config=config,
            rules=MockRules(),
        )

    def test_init(self, controller):
        """Test controller initialization."""
        assert controller.game_id == "test-game"
        assert controller.game_type == "tictactoe"
        assert controller.model == "test-model"
        assert controller.memory_store is not None
        assert controller.strategy_memory is not None
        assert controller.visual_encoder is not None
        assert controller.fusion_layer is not None
        assert controller.state_engine is not None
        assert controller.reasoning_engine is not None

    @pytest.mark.asyncio
    async def test_perceive(self, controller, mock_environment):
        """Test perception."""
        game_state, visual = await controller.perceive()

        assert isinstance(game_state, GameState)
        assert game_state.current_player == "X"
        mock_environment.get_state.assert_called()

    @pytest.mark.asyncio
    async def test_select_action_game_over(self, controller, mock_environment):
        """Test select_action returns None when game is over."""
        mock_environment.get_state.return_value = GameState(
            board=[["X", "X", "X"], ["O", "O", ""], ["", "", ""]],
            current_player="O",
            turn_number=5,
            legal_moves=[],
            game_over=True,
            winner="X",
        )

        game_state = mock_environment.get_state()
        result = await controller.select_action(game_state)

        assert result is None

    @pytest.mark.asyncio
    async def test_select_action_uses_reasoning_engine(self, controller, mock_environment):
        """Test that select_action uses reasoning_engine.select_best_move()."""
        game_state = mock_environment.get_state()

        # Mock the reasoning engine
        mock_state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="move 1,1", confidence=0.9, reasoning="center is best"),
            ],
            entropy=0.3,
        )
        controller.reasoning_engine.latent_reasoner.reason = AsyncMock(return_value=mock_state)

        result = await controller.select_action(game_state)

        # Should have called reason
        controller.reasoning_engine.latent_reasoner.reason.assert_awaited()

    @pytest.mark.asyncio
    async def test_execute_action(self, controller, mock_environment):
        """Test action execution."""
        result = await controller.execute_action("1,1")

        assert result.success is True
        mock_environment.execute_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_observe_and_reflect_stores_strategy(
        self, controller, mock_environment
    ):
        """Test that observe_and_reflect stores strategy."""
        prev_state = GameState(
            board=[["", "", ""], ["", "", ""], ["", "", ""]],
            current_player="X",
            turn_number=0,
            legal_moves=["0,0", "1,1"],
            game_over=False,
        )
        new_state = GameState(
            board=[["X", "", ""], ["", "", ""], ["", "", ""]],
            current_player="O",
            turn_number=1,
            legal_moves=["0,1", "1,1"],
            game_over=False,
        )
        result = ActionResult(success=True, new_state=new_state, reward=0.1)

        await controller.observe_and_reflect(prev_state, "0,0", result, new_state)

        # Check that move was recorded
        assert controller.learning_state.total_moves == 1

    @pytest.mark.asyncio
    async def test_observe_and_reflect_updates_als_on_game_end(
        self, controller, mock_environment
    ):
        """Test that observe_and_reflect updates ALS when game ends."""
        prev_state = GameState(
            board=[["X", "X", ""], ["O", "O", ""], ["", "", ""]],
            current_player="X",
            turn_number=4,
            legal_moves=["0,2"],
            game_over=False,
        )
        new_state = GameState(
            board=[["X", "X", "X"], ["O", "O", ""], ["", "", ""]],
            current_player="O",
            turn_number=5,
            legal_moves=[],
            game_over=True,
            winner="X",
        )
        result = ActionResult(success=True, new_state=new_state, reward=1.0)

        await controller.observe_and_reflect(prev_state, "0,2", result, new_state)

        # Check that game end was recorded
        assert controller.learning_state.wins == 1

    def test_get_health_status(self, controller):
        """Test getting health status."""
        status = controller.get_health_status()

        assert status["status"] == "active"
        assert status["game_id"] == "test-game"
        assert status["game_type"] == "tictactoe"
        assert "win_rate" in status
        assert "visual_encoder" in status
