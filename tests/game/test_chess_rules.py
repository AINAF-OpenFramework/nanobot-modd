"""Tests for Chess game rules scaffold."""

from __future__ import annotations

import pytest

from nanobot.game.rules.chess import ChessRules


class TestChessRules:
    """Tests for ChessRules scaffold."""

    @pytest.fixture
    def rules(self):
        """Create ChessRules instance."""
        return ChessRules()

    @pytest.fixture
    def initial_state(self, rules):
        """Create initial game state."""
        return rules.create_initial_state()

    def test_init(self):
        """Test rules initialization."""
        rules = ChessRules()
        assert rules.board_size == 8
        assert rules.players == ["white", "black"]

    def test_create_initial_state(self, rules):
        """Test initial state creation."""
        state = rules.create_initial_state()
        assert state["current_player"] == "white"
        assert state["move_count"] == 0
        assert len(state["board"]) == 8
        assert all(len(row) == 8 for row in state["board"])

        # Check white pieces on bottom
        assert state["board"][7][0] == "wR"  # White rook
        assert state["board"][7][4] == "wK"  # White king
        assert state["board"][6][0] == "wP"  # White pawn

        # Check black pieces on top
        assert state["board"][0][0] == "bR"  # Black rook
        assert state["board"][0][4] == "bK"  # Black king
        assert state["board"][1][0] == "bP"  # Black pawn

        # Check middle is empty
        assert state["board"][3][3] == ""

        # Check castling rights
        assert state["castling_rights"]["white_kingside"] is True
        assert state["castling_rights"]["black_queenside"] is True

    def test_create_initial_state_custom_player(self, rules):
        """Test initial state with custom starting player."""
        state = rules.create_initial_state(starting_player="black")
        assert state["current_player"] == "black"

    def test_get_legal_moves_scaffold(self, rules, initial_state):
        """Test get_legal_moves returns empty (scaffold)."""
        moves = rules.get_legal_moves(initial_state)
        # Scaffold returns empty list
        assert moves == []

    def test_apply_move_scaffold_raises(self, rules, initial_state):
        """Test apply_move raises NotImplementedError (scaffold)."""
        with pytest.raises(NotImplementedError, match="scaffold"):
            rules.apply_move(initial_state, "e2e4")

    def test_check_win_conditions_scaffold(self, rules, initial_state):
        """Test check_win_conditions returns in progress (scaffold)."""
        result = rules.check_win_conditions(initial_state)
        assert result["game_over"] is False
        assert result["winner"] is None
        assert "progress" in result["status"].lower()

    def test_get_next_player(self, rules):
        """Test get_next_player."""
        state_white = {"current_player": "white"}
        assert rules.get_next_player(state_white) == "black"

        state_black = {"current_player": "black"}
        assert rules.get_next_player(state_black) == "white"

    def test_get_board_string(self, rules, initial_state):
        """Test board string representation."""
        board_str = rules.get_board_string(initial_state)
        assert "a  b  c  d  e  f  g  h" in board_str
        assert "wK" in board_str or "wK " in board_str
        assert "bP" in board_str or "bP " in board_str
        # Check row numbers
        assert "8" in board_str
        assert "1" in board_str

    def test_generate_test_positions(self, rules):
        """Test test position generation."""
        positions = rules.generate_test_positions()
        assert len(positions) >= 3

        # Check initial position
        assert positions[0]["move_count"] == 0
        assert positions[0]["board"][7][4] == "wK"

        # Check that positions are different
        assert positions[0] != positions[1]
        assert positions[1] != positions[2]

    def test_score_position_initial(self, rules, initial_state):
        """Test position scoring on initial state."""
        score = rules.score_position(initial_state)
        # Initial position should be balanced (score near 0)
        assert abs(score) < 0.1

    def test_score_position_material_advantage(self, rules):
        """Test position scoring with material advantage."""
        # White up a queen
        state = {
            "board": [
                ["", "", "", "", "bK", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "wQ", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "wK", "", "", ""],
            ],
            "current_player": "white",
            "move_count": 50,
        }
        score = rules.score_position(state)
        # White should have positive score (queen = 9)
        assert score > 8.0

    def test_score_position_black_advantage(self, rules):
        """Test position scoring with black material advantage."""
        # Black up two rooks
        state = {
            "board": [
                ["bR", "", "", "", "bK", "", "", "bR"],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "wK", "", "", ""],
            ],
            "current_player": "white",
            "move_count": 50,
        }
        score = rules.score_position(state)
        # Black should have negative score (2 rooks = -10)
        assert score < -9.0

    def test_castling_rights_in_state(self, rules, initial_state):
        """Test that castling rights are tracked in state."""
        assert "castling_rights" in initial_state
        assert initial_state["castling_rights"]["white_kingside"] is True
        assert initial_state["castling_rights"]["white_queenside"] is True
        assert initial_state["castling_rights"]["black_kingside"] is True
        assert initial_state["castling_rights"]["black_queenside"] is True

    def test_en_passant_tracking(self, rules, initial_state):
        """Test that en passant target is tracked in state."""
        assert "en_passant_target" in initial_state
        assert initial_state["en_passant_target"] is None

    def test_halfmove_clock(self, rules, initial_state):
        """Test halfmove clock initialization."""
        assert "halfmove_clock" in initial_state
        assert initial_state["halfmove_clock"] == 0

    def test_fullmove_number(self, rules, initial_state):
        """Test fullmove number initialization."""
        assert "fullmove_number" in initial_state
        assert initial_state["fullmove_number"] == 1

    def test_board_representation_consistency(self, rules):
        """Test that board representation is consistent."""
        state = rules.create_initial_state()

        # All white pieces should start with 'w'
        for row_idx in [6, 7]:
            for piece in state["board"][row_idx]:
                if piece:
                    assert piece[0] == "w"

        # All black pieces should start with 'b'
        for row_idx in [0, 1]:
            for piece in state["board"][row_idx]:
                if piece:
                    assert piece[0] == "b"

        # Middle rows should be empty
        for row_idx in [2, 3, 4, 5]:
            for piece in state["board"][row_idx]:
                assert piece == ""
