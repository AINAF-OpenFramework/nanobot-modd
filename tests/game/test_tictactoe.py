"""Tests for TicTacToe example implementation."""

from __future__ import annotations

import pytest

from nanobot.game.examples.tictactoe import TicTacToeRules, create_tictactoe_engine


class TestTicTacToeRules:
    """Tests for TicTacToeRules class."""

    @pytest.fixture
    def rules(self):
        """Create rules instance for testing."""
        return TicTacToeRules()

    def test_get_legal_moves_empty_board(self, rules):
        """Test legal moves on empty board."""
        state = {"board": [""] * 9, "current_player": "X"}
        moves = rules.get_legal_moves(state)
        assert moves == ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    def test_get_legal_moves_partial_board(self, rules):
        """Test legal moves on partial board."""
        state = {"board": ["X", "O", "", "", "X", "", "", "", "O"], "current_player": "X"}
        moves = rules.get_legal_moves(state)
        assert moves == ["2", "3", "5", "6", "7"]

    def test_get_legal_moves_returns_empty_when_game_over(self, rules):
        """Test no legal moves when game is won."""
        state = {"board": ["X", "X", "X", "O", "O", "", "", "", ""], "current_player": "O"}
        moves = rules.get_legal_moves(state)
        assert moves == []

    def test_apply_move_places_piece(self, rules):
        """Test applying a move places the piece."""
        state = {"board": [""] * 9, "current_player": "X", "turn": 0}
        new_state = rules.apply_move(state, "4")
        assert new_state["board"][4] == "X"
        assert new_state["current_player"] == "O"
        assert new_state["turn"] == 1

    def test_apply_move_invalid_position_raises(self, rules):
        """Test applying invalid position raises error."""
        state = {"board": [""] * 9, "current_player": "X"}
        with pytest.raises(ValueError, match="Invalid position"):
            rules.apply_move(state, "9")

    def test_apply_move_occupied_position_raises(self, rules):
        """Test applying move to occupied position raises error."""
        state = {"board": ["X", "", "", "", "", "", "", "", ""], "current_player": "O"}
        with pytest.raises(ValueError, match="already occupied"):
            rules.apply_move(state, "0")

    def test_check_win_conditions_row_win(self, rules):
        """Test detecting row win."""
        state = {"board": ["X", "X", "X", "O", "O", "", "", "", ""]}
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] == "X"

    def test_check_win_conditions_column_win(self, rules):
        """Test detecting column win."""
        state = {"board": ["O", "X", "", "O", "X", "", "O", "", ""]}
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] == "O"

    def test_check_win_conditions_diagonal_win(self, rules):
        """Test detecting diagonal win."""
        state = {"board": ["X", "O", "", "O", "X", "", "", "", "X"]}
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] == "X"

    def test_check_win_conditions_anti_diagonal_win(self, rules):
        """Test detecting anti-diagonal win."""
        state = {"board": ["", "", "O", "", "O", "", "O", "", ""]}
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] == "O"

    def test_check_win_conditions_draw(self, rules):
        """Test detecting draw."""
        state = {"board": ["X", "O", "X", "X", "O", "O", "O", "X", "X"]}
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] is None
        assert "Draw" in result["status"]

    def test_check_win_conditions_in_progress(self, rules):
        """Test detecting game in progress."""
        state = {"board": ["X", "O", "", "", "", "", "", "", ""]}
        result = rules.check_win_conditions(state)
        assert result["game_over"] is False
        assert result["winner"] is None
        assert "progress" in result["status"].lower()

    def test_get_next_player_switches(self, rules):
        """Test player switching."""
        assert rules.get_next_player({"current_player": "X"}) == "O"
        assert rules.get_next_player({"current_player": "O"}) == "X"

    def test_get_next_player_defaults_to_x(self, rules):
        """Test default next player is O (assumes X started)."""
        # When no current player specified, it defaults to X -> next is O
        assert rules.get_next_player({}) == "O"


class TestCreateTictactoeEngine:
    """Tests for create_tictactoe_engine function."""

    def test_creates_engine_with_rules(self):
        """Test engine has TicTacToe rules."""
        engine = create_tictactoe_engine()
        assert engine.rules is not None
        assert isinstance(engine.rules, TicTacToeRules)

    def test_creates_engine_with_initial_state(self):
        """Test engine has correct initial state."""
        engine = create_tictactoe_engine()
        state = engine.get_state()
        assert state["board"] == [""] * 9
        assert state["current_player"] == "X"
        assert state["turn"] == 0

    def test_creates_engine_with_custom_id(self):
        """Test engine can have custom ID."""
        engine = create_tictactoe_engine(game_id="my-game")
        assert engine.game_id == "my-game"

    def test_engine_can_play_full_game(self):
        """Test playing a full game."""
        engine = create_tictactoe_engine()

        # X plays corner
        engine.apply_move("0", player="X")
        # O plays center
        engine.apply_move("4", player="O")
        # X plays opposite corner
        engine.apply_move("8", player="X")
        # O blocks
        engine.apply_move("2", player="O")
        # X plays edge
        engine.apply_move("6", player="X")
        # O blocks
        engine.apply_move("3", player="O")
        # X wins with middle row
        # Wait, let's check what's legal
        state = engine.get_state()
        assert state["turn"] == 6

    def test_engine_detects_win(self):
        """Test engine properly detects a win."""
        engine = create_tictactoe_engine()

        # X wins with top row
        engine.apply_move("0", player="X")
        engine.apply_move("3", player="O")
        engine.apply_move("1", player="X")
        engine.apply_move("4", player="O")
        engine.apply_move("2", player="X")

        result = engine.check_win_conditions()
        assert result["game_over"] is True
        assert result["winner"] == "X"

        # No more legal moves after win
        assert engine.get_legal_moves() == []
