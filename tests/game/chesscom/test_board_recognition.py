"""Tests for Chess.com board recognition module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class TestBoardRecognizer:
    """Tests for BoardRecognizer class."""

    @pytest.fixture
    def mock_cv2(self):
        """Mock cv2 module."""
        with patch("nanobot.game.chesscom.board_recognition.cv2", True):
            yield

    @pytest.fixture
    def recognizer(self, mock_cv2):
        """Create BoardRecognizer instance."""
        from nanobot.game.chesscom.board_recognition import BoardRecognizer

        return BoardRecognizer(piece_theme="default")

    @pytest.fixture
    def sample_board_image(self):
        """Create sample board image."""
        return np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)

    def test_init(self, mock_cv2):
        """Test initialization."""
        from nanobot.game.chesscom.board_recognition import BoardRecognizer

        recognizer = BoardRecognizer(piece_theme="neo")
        assert recognizer.piece_theme == "neo"

    def test_recognize_pieces_returns_8x8_board(self, recognizer, sample_board_image):
        """Test piece recognition returns 8x8 board."""
        board_state = recognizer.recognize_pieces(sample_board_image)

        assert isinstance(board_state, list)
        assert len(board_state) == 8
        assert all(len(row) == 8 for row in board_state)

    def test_recognize_pieces_starting_position(self, recognizer, sample_board_image):
        """Test that placeholder returns starting position."""
        board_state = recognizer.recognize_pieces(sample_board_image)

        # Check black pieces on top
        assert board_state[0][0] == "r"
        assert board_state[0][4] == "k"
        assert board_state[1][0] == "p"

        # Check white pieces on bottom
        assert board_state[7][0] == "R"
        assert board_state[7][4] == "K"
        assert board_state[6][0] == "P"

        # Check empty middle
        assert board_state[3][3] == ""

    def test_to_fen_starting_position(self, recognizer):
        """Test FEN conversion for starting position."""
        board_state = [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", "p", "p", "p", "p"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["P", "P", "P", "P", "P", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"],
        ]

        fen = recognizer.to_fen(board_state, turn="w")

        # Should match standard starting position
        assert fen.startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w")

    def test_to_fen_with_black_turn(self, recognizer):
        """Test FEN conversion with black to move."""
        board_state = [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", "p", "p", "p", "p"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["P", "P", "P", "P", "P", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"],
        ]

        fen = recognizer.to_fen(board_state, turn="b")

        assert " b " in fen  # Black to move

    def test_to_fen_partial_board(self, recognizer):
        """Test FEN conversion for partial board."""
        board_state = [
            ["r", "", "", "", "k", "", "", "r"],
            ["p", "p", "", "", "", "p", "p", "p"],
            ["", "", "n", "", "", "", "", ""],
            ["", "", "", "p", "p", "", "", ""],
            ["", "", "", "P", "P", "", "", ""],
            ["", "", "N", "", "", "", "", ""],
            ["P", "P", "", "", "", "P", "P", "P"],
            ["R", "", "", "", "K", "", "", "R"],
        ]

        fen = recognizer.to_fen(board_state, turn="w")

        # Check FEN format
        assert fen.count("/") == 7  # 7 slashes separate 8 ranks
        parts = fen.split(" ")
        assert len(parts) >= 2  # At least pieces and turn

    def test_detect_orientation(self, recognizer, sample_board_image):
        """Test orientation detection."""
        orientation = recognizer.detect_orientation(sample_board_image)

        assert isinstance(orientation, str)
        assert orientation in ["white", "black"]

    def test_detect_last_move(self, recognizer, sample_board_image):
        """Test last move detection."""
        last_move = recognizer.detect_last_move(sample_board_image)

        # Placeholder returns None
        assert last_move is None

    def test_piece_symbols_constant(self):
        """Test that piece symbols are correctly defined."""
        from nanobot.game.chesscom.board_recognition import BoardRecognizer

        symbols = BoardRecognizer.PIECE_SYMBOLS

        # Check all pieces exist
        assert symbols["white_pawn"] == "P"
        assert symbols["white_king"] == "K"
        assert symbols["black_pawn"] == "p"
        assert symbols["black_king"] == "k"

        # Check all 12 pieces defined
        assert len(symbols) == 12

    def test_import_error_without_cv2(self):
        """Test that ImportError is raised when opencv is not available."""
        from unittest.mock import patch

        with patch("nanobot.game.chesscom.board_recognition.cv2", None):
            from nanobot.game.chesscom.board_recognition import BoardRecognizer

            with pytest.raises(ImportError, match="opencv-python is required"):
                BoardRecognizer()
