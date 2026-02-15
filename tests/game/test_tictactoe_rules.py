"""Tests for TicTacToe game rules."""

from __future__ import annotations

import pytest

from nanobot.game.rules.tictactoe import TicTacToeRules


class TestTicTacToeRules:
    """Tests for TicTacToeRules class."""

    @pytest.fixture
    def rules(self):
        """Create TicTacToeRules instance."""
        return TicTacToeRules()

    @pytest.fixture
    def initial_state(self, rules):
        """Create initial game state."""
        return rules.create_initial_state()

    def test_init(self):
        """Test rules initialization."""
        rules = TicTacToeRules()
        assert rules.board_size == 3
        assert rules.players == ["X", "O"]

    def test_create_initial_state(self, rules):
        """Test initial state creation."""
        state = rules.create_initial_state()
        assert state["current_player"] == "X"
        assert state["move_count"] == 0
        assert len(state["board"]) == 3
        assert all(len(row) == 3 for row in state["board"])
        assert all(cell == "" for row in state["board"] for cell in row)

    def test_create_initial_state_custom_player(self, rules):
        """Test initial state with custom starting player."""
        state = rules.create_initial_state(starting_player="O")
        assert state["current_player"] == "O"

    def test_get_legal_moves_empty_board(self, rules, initial_state):
        """Test get_legal_moves on empty board."""
        moves = rules.get_legal_moves(initial_state)
        assert len(moves) == 9
        assert "r0c0" in moves
        assert "r2c2" in moves

    def test_get_legal_moves_partial_board(self, rules):
        """Test get_legal_moves on partially filled board."""
        state = {
            "board": [
                ["X", "O", ""],
                ["", "X", ""],
                ["", "", "O"],
            ],
            "current_player": "X",
            "move_count": 5,
        }
        moves = rules.get_legal_moves(state)
        assert len(moves) == 5  # r0c2, r1c0, r1c2, r2c0, r2c1
        assert "r0c2" in moves
        assert "r1c0" in moves
        assert "r1c2" in moves
        assert "r2c0" in moves
        assert "r2c1" in moves

    def test_apply_move_valid(self, rules, initial_state):
        """Test applying a valid move."""
        new_state = rules.apply_move(initial_state, "r0c0")
        assert new_state["board"][0][0] == "X"
        assert new_state["current_player"] == "O"
        assert new_state["move_count"] == 1

    def test_apply_move_switches_player(self, rules, initial_state):
        """Test that apply_move switches player."""
        state1 = rules.apply_move(initial_state, "r0c0")
        assert state1["current_player"] == "O"

        state2 = rules.apply_move(state1, "r0c1")
        assert state2["current_player"] == "X"

    def test_apply_move_invalid_format(self, rules, initial_state):
        """Test applying move with invalid format."""
        with pytest.raises(ValueError, match="Invalid move format"):
            rules.apply_move(initial_state, "invalid")

    def test_apply_move_out_of_bounds(self, rules, initial_state):
        """Test applying move out of bounds."""
        with pytest.raises(ValueError, match="out of bounds"):
            rules.apply_move(initial_state, "r5c5")

    def test_apply_move_occupied_cell(self, rules):
        """Test applying move to occupied cell."""
        state = {
            "board": [
                ["X", "", ""],
                ["", "", ""],
                ["", "", ""],
            ],
            "current_player": "O",
            "move_count": 1,
        }
        with pytest.raises(ValueError, match="already occupied"):
            rules.apply_move(state, "r0c0")

    def test_check_win_row(self, rules):
        """Test win condition - horizontal row."""
        state = {
            "board": [
                ["X", "X", "X"],
                ["O", "O", ""],
                ["", "", ""],
            ],
            "current_player": "O",
            "move_count": 5,
        }
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] == "X"
        assert "row" in result["status"]

    def test_check_win_column(self, rules):
        """Test win condition - vertical column."""
        state = {
            "board": [
                ["O", "X", "X"],
                ["O", "", ""],
                ["O", "", ""],
            ],
            "current_player": "X",
            "move_count": 5,
        }
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] == "O"
        assert "column" in result["status"]

    def test_check_win_main_diagonal(self, rules):
        """Test win condition - main diagonal."""
        state = {
            "board": [
                ["X", "O", "O"],
                ["", "X", ""],
                ["", "", "X"],
            ],
            "current_player": "O",
            "move_count": 5,
        }
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] == "X"
        assert "diagonal" in result["status"].lower()

    def test_check_win_anti_diagonal(self, rules):
        """Test win condition - anti-diagonal."""
        state = {
            "board": [
                ["X", "X", "O"],
                ["", "O", "X"],
                ["O", "", "X"],
            ],
            "current_player": "X",
            "move_count": 6,
        }
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] == "O"
        assert "diagonal" in result["status"].lower()

    def test_check_draw(self, rules):
        """Test draw condition."""
        state = {
            "board": [
                ["X", "O", "X"],
                ["O", "X", "O"],
                ["O", "X", "O"],
            ],
            "current_player": "X",
            "move_count": 9,
        }
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] is None
        assert "draw" in result["status"].lower()

    def test_check_in_progress(self, rules):
        """Test game in progress."""
        state = {
            "board": [
                ["X", "O", ""],
                ["", "X", ""],
                ["", "", ""],
            ],
            "current_player": "O",
            "move_count": 3,
        }
        result = rules.check_win_conditions(state)
        assert result["game_over"] is False
        assert result["winner"] is None
        assert "progress" in result["status"].lower()

    def test_get_next_player(self, rules):
        """Test get_next_player."""
        state_x = {"current_player": "X"}
        assert rules.get_next_player(state_x) == "O"

        state_o = {"current_player": "O"}
        assert rules.get_next_player(state_o) == "X"

    def test_get_board_string(self, rules):
        """Test board string representation."""
        state = {
            "board": [
                ["X", "O", "X"],
                ["", "X", ""],
                ["O", "", "O"],
            ],
            "current_player": "O",
            "move_count": 6,
        }
        board_str = rules.get_board_string(state)
        assert "X" in board_str
        assert "O" in board_str
        assert "|" in board_str

    def test_full_game_sequence(self, rules):
        """Test a full game sequence."""
        state = rules.create_initial_state()

        # X plays top-left
        state = rules.apply_move(state, "r0c0")
        assert state["board"][0][0] == "X"

        # O plays center
        state = rules.apply_move(state, "r1c1")
        assert state["board"][1][1] == "O"

        # X plays top-middle
        state = rules.apply_move(state, "r0c1")
        assert state["board"][0][1] == "X"

        # O plays bottom-left
        state = rules.apply_move(state, "r2c0")
        assert state["board"][2][0] == "O"

        # X plays top-right to win (row 0)
        state = rules.apply_move(state, "r0c2")
        assert state["board"][0][2] == "X"

        # Check win
        result = rules.check_win_conditions(state)
        assert result["game_over"] is True
        assert result["winner"] == "X"
