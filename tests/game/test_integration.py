"""Integration tests for game learning components."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.game.environment import ActionResult, APIGameAdapter, GameState
from nanobot.game.health import (
    clear_all_controllers,
    extend_health_payload,
    get_game_health_status,
    register_game_controller,
    unregister_game_controller,
)
from nanobot.game.learning_controller import (
    GameLearningController,
    LearningConfig,
)
from nanobot.game.state_engine import GameRules


class MockTicTacToeGame:
    """Mock TicTacToe game for integration testing."""

    def __init__(self):
        self.board = [""] * 9
        self.current_player = "X"
        self.turn = 0
        self.game_over = False
        self.winner: str | None = None

    def get_state(self) -> dict[str, Any]:
        """Get current game state as dict."""
        legal_moves = [str(i) for i in range(9) if self.board[i] == ""]
        return {
            "board": self.board.copy(),
            "current_player": self.current_player,
            "turn_number": self.turn,
            "legal_moves": legal_moves,
            "game_over": self.game_over,
            "winner": self.winner,
            "score": 0.0,
        }

    def execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute a move action."""
        move = action.get("move")
        if move is None:
            return {"success": False, "error": "No move provided"}

        try:
            pos = int(move)
        except ValueError:
            return {"success": False, "error": f"Invalid move: {move}"}

        if pos < 0 or pos > 8 or self.board[pos] != "":
            return {"success": False, "error": f"Invalid position: {pos}"}

        # Make the move
        self.board[pos] = self.current_player
        self.turn += 1

        # Check for win
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Cols
            [0, 4, 8], [2, 4, 6],  # Diagonals
        ]
        for pattern in win_patterns:
            if all(self.board[i] == self.current_player for i in pattern):
                self.game_over = True
                self.winner = self.current_player
                break

        # Check for draw
        if not self.game_over and all(cell != "" for cell in self.board):
            self.game_over = True

        # Switch player
        if not self.game_over:
            self.current_player = "O" if self.current_player == "X" else "X"

        return {
            "success": True,
            "new_state": self.get_state(),
            "reward": 1.0 if self.winner else 0.0,
            "message": "Move executed",
        }

    def reset(self) -> None:
        """Reset the game."""
        self.board = [""] * 9
        self.current_player = "X"
        self.turn = 0
        self.game_over = False
        self.winner = None


class MockTicTacToeRules(GameRules):
    """Game rules for TicTacToe."""

    def get_legal_moves(self, state: dict[str, Any]) -> list[str]:
        board = state.get("board", [])
        return [str(i) for i in range(9) if i < len(board) and board[i] == ""]

    def apply_move(self, state: dict[str, Any], move: str) -> dict[str, Any]:
        new_state = state.copy()
        board = new_state.get("board", [""] * 9).copy()
        pos = int(move)
        board[pos] = new_state.get("current_player", "X")
        new_state["board"] = board
        new_state["turn_number"] = new_state.get("turn_number", 0) + 1
        # Switch player
        curr = new_state.get("current_player", "X")
        new_state["current_player"] = "O" if curr == "X" else "X"
        return new_state

    def check_win_conditions(self, state: dict[str, Any]) -> dict[str, Any]:
        board = state.get("board", [""] * 9)
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6],
        ]
        for pattern in win_patterns:
            cells = [board[i] for i in pattern if i < len(board)]
            if len(cells) == 3 and cells[0] != "" and cells[0] == cells[1] == cells[2]:
                return {"game_over": True, "winner": cells[0], "status": "Won"}
        if all(cell != "" for cell in board):
            return {"game_over": True, "winner": None, "status": "Draw"}
        return {"game_over": False, "winner": None, "status": "In progress"}

    def get_next_player(self, state: dict[str, Any]) -> str:
        return "O" if state.get("current_player") == "X" else "X"


class TestHealthEndpoint:
    """Tests for health endpoint functionality."""

    def setup_method(self):
        """Clear controllers before each test."""
        clear_all_controllers()

    def teardown_method(self):
        """Clear controllers after each test."""
        clear_all_controllers()

    def test_register_game_controller(self):
        """Test registering a game controller."""
        mock_controller = MagicMock()
        mock_controller.get_health_status.return_value = {"status": "active"}

        register_game_controller("game-1", mock_controller)
        status = get_game_health_status()

        assert status["active_games"] == 1
        assert "game-1" in status["games"]
        assert status["games"]["game-1"]["status"] == "active"

    def test_unregister_game_controller(self):
        """Test unregistering a game controller."""
        mock_controller = MagicMock()
        register_game_controller("game-1", mock_controller)
        unregister_game_controller("game-1")

        status = get_game_health_status()
        assert status["active_games"] == 0
        assert "game-1" not in status["games"]

    def test_unregister_nonexistent_controller(self):
        """Test unregistering a controller that doesn't exist."""
        # Should not raise an error
        unregister_game_controller("nonexistent")

    def test_get_game_health_status_multiple_games(self):
        """Test health status with multiple games."""
        controller1 = MagicMock()
        controller1.get_health_status.return_value = {"status": "active", "turns": 5}

        controller2 = MagicMock()
        controller2.get_health_status.return_value = {"status": "paused", "turns": 10}

        register_game_controller("game-1", controller1)
        register_game_controller("game-2", controller2)

        status = get_game_health_status()

        assert status["active_games"] == 2
        assert status["games"]["game-1"]["turns"] == 5
        assert status["games"]["game-2"]["status"] == "paused"

    def test_extend_health_payload(self):
        """Test extending health payload with game status."""
        mock_controller = MagicMock()
        mock_controller.get_health_status.return_value = {"status": "active"}
        register_game_controller("test-game", mock_controller)

        payload = {"service": "nanobot", "version": "1.0"}
        extended = extend_health_payload(payload)

        assert "games" in extended
        assert extended["games"]["active_games"] == 1
        assert extended["service"] == "nanobot"

    def test_controller_without_health_method(self):
        """Test controller without get_health_status method."""
        controller = MagicMock(spec=[])  # No methods
        controller.game_type = "chess"

        register_game_controller("game-1", controller)
        status = get_game_health_status()

        assert "game-1" in status["games"]
        assert status["games"]["game-1"]["status"] == "active"
        assert status["games"]["game-1"]["game_type"] == "chess"


class TestGameLearningIntegration:
    """Integration tests for game learning components."""

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
    def tictactoe_game(self):
        """Create a TicTacToe game instance."""
        return MockTicTacToeGame()

    @pytest.fixture
    def api_adapter(self, tictactoe_game):
        """Create an API adapter for TicTacToe."""
        return APIGameAdapter(
            game_id="tictactoe-1",
            game_type="tictactoe",
            state_getter=tictactoe_game.get_state,
            action_executor=tictactoe_game.execute_action,
        )

    def test_api_adapter_get_state(self, api_adapter, tictactoe_game):
        """Test APIGameAdapter.get_state()."""
        state = api_adapter.get_state()

        assert isinstance(state, GameState)
        assert state.current_player == "X"
        assert state.turn_number == 0
        assert len(state.legal_moves) == 9

    def test_api_adapter_execute_action(self, api_adapter, tictactoe_game):
        """Test APIGameAdapter.execute_action()."""
        result = api_adapter.execute_action({"type": "move", "move": "4"})

        assert result.success is True
        assert result.new_state is not None
        assert result.new_state.turn_number == 1

    def test_full_game_flow(self, api_adapter, tictactoe_game):
        """Test playing through a full game via the adapter."""
        # Play moves: X wins with 0, 1, 2
        moves = ["0", "3", "1", "4", "2"]

        for move in moves:
            state = api_adapter.get_state()
            if state.game_over:
                break

            result = api_adapter.execute_action({"type": "move", "move": move})
            assert result.success is True

        final_state = api_adapter.get_state()
        assert final_state.game_over is True
        assert final_state.winner == "X"

    @pytest.mark.asyncio
    async def test_learning_controller_with_mock_game(
        self, mock_provider, api_adapter, temp_workspace, tictactoe_game
    ):
        """Test GameLearningController with mock TicTacToe game."""
        config = LearningConfig(
            visual_encoder_type="grid",
            store_all_moves=True,
        )

        controller = GameLearningController(
            provider=mock_provider,
            model="test-model",
            environment=api_adapter,
            workspace=temp_workspace,
            config=config,
            rules=MockTicTacToeRules(),
        )

        # Mock the reasoning engine to return a valid move
        mock_state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="move 4", confidence=0.9, reasoning="center"),
            ],
            entropy=0.3,
        )
        controller.reasoning_engine.latent_reasoner.reason = AsyncMock(
            return_value=mock_state
        )

        # Play one turn
        move, result = await controller.play_turn()

        assert move in ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
        assert result is not None
        assert result.success is True

        # Verify learning state was updated
        assert controller.learning_state.total_moves == 1

    @pytest.mark.asyncio
    async def test_strategy_storage_on_game_end(
        self, mock_provider, api_adapter, temp_workspace, tictactoe_game
    ):
        """Test that strategies are stored when game ends."""
        config = LearningConfig(
            visual_encoder_type="grid",
            store_all_moves=True,
        )

        controller = GameLearningController(
            provider=mock_provider,
            model="test-model",
            environment=api_adapter,
            workspace=temp_workspace,
            config=config,
            rules=MockTicTacToeRules(),
        )

        # Set up game near end
        tictactoe_game.board = ["X", "X", "", "O", "O", "", "", "", ""]
        tictactoe_game.turn = 4
        tictactoe_game.current_player = "X"

        # Mock reasoning to return winning move
        mock_state = SuperpositionalState(
            hypotheses=[
                Hypothesis(intent="move 2", confidence=0.95, reasoning="winning"),
            ],
            entropy=0.1,
        )
        controller.reasoning_engine.latent_reasoner.reason = AsyncMock(
            return_value=mock_state
        )

        # Play winning turn
        move, result = await controller.play_turn()

        # Verify game ended with win
        final_state = api_adapter.get_state()
        assert final_state.game_over is True
        assert final_state.winner == "X"

        # Verify learning stats
        assert controller.learning_state.wins == 1
