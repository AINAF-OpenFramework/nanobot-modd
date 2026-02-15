"""Tests for Chess.com GUI automation module."""

from __future__ import annotations

import pytest


class TestChessComAutomation:
    """Tests for ChessComAutomation class."""

    @pytest.fixture
    def automation(self):
        """Create ChessComAutomation instance."""
        from unittest.mock import patch
        
        # Mock pyautogui to avoid actual mouse movements
        with patch("nanobot.game.chesscom.gui_automation.pyautogui"):
            from nanobot.game.chesscom.gui_automation import ChessComAutomation
            
            board_region = (100, 100, 600, 600)
            return ChessComAutomation(
                board_region=board_region,
                min_move_delay=0.1,
                max_move_delay=0.2,
                human_like=False,
            )

    def test_init(self):
        """Test initialization."""
        from unittest.mock import patch
        
        with patch("nanobot.game.chesscom.gui_automation.pyautogui"):
            from nanobot.game.chesscom.gui_automation import ChessComAutomation
            
            board_region = (100, 100, 600, 600)
            automation = ChessComAutomation(
                board_region=board_region,
                min_move_delay=0.5,
                max_move_delay=2.0,
                human_like=True,
            )
            
            assert automation.board_region == board_region
            assert automation.min_move_delay == 0.5
            assert automation.max_move_delay == 2.0
            assert automation.human_like is True

    def test_square_to_screen_coords_white(self, automation):
        """Test square to screen coordinate conversion (white orientation)."""
        # e4 square (file e = index 4, rank 4 = index 3)
        coords = automation.square_to_screen_coords("e4", "white")
        
        assert isinstance(coords, tuple)
        assert len(coords) == 2
        
        x, y = coords
        # Center of e4 square for white orientation
        # x = 100 + (4 + 0.5) * 75 = 437.5
        # y = 100 + (7 - 3 + 0.5) * 75 = 437.5
        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_square_to_screen_coords_black(self, automation):
        """Test square to screen coordinate conversion (black orientation)."""
        coords = automation.square_to_screen_coords("e4", "black")
        
        assert isinstance(coords, tuple)
        assert len(coords) == 2
        
        # Should be flipped compared to white orientation
        x, y = coords
        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_square_to_screen_coords_corners(self, automation):
        """Test corner squares."""
        # a1 - bottom left for white
        a1 = automation.square_to_screen_coords("a1", "white")
        assert isinstance(a1, tuple)
        
        # h8 - top right for white
        h8 = automation.square_to_screen_coords("h8", "white")
        assert isinstance(h8, tuple)
        
        # Coordinates should be different
        assert a1 != h8

    def test_square_to_screen_coords_invalid_square(self, automation):
        """Test invalid square notation."""
        with pytest.raises(ValueError, match="Invalid square"):
            automation.square_to_screen_coords("i9", "white")
        
        with pytest.raises(ValueError, match="Invalid square format"):
            automation.square_to_screen_coords("e", "white")

    def test_files_and_ranks_constants(self):
        """Test FILES and RANKS constants."""
        from nanobot.game.chesscom.gui_automation import ChessComAutomation
        
        assert ChessComAutomation.FILES == "abcdefgh"
        assert ChessComAutomation.RANKS == "12345678"

    def test_execute_move_simple(self, automation):
        """Test executing a simple move (mocked)."""
        from unittest.mock import patch
        
        with patch("nanobot.game.chesscom.gui_automation.pyautogui") as mock_gui:
            result = automation.execute_move("e2e4", "white")
            
            # Should succeed (mocked)
            assert result is True

    def test_execute_move_with_promotion(self, automation):
        """Test executing a move with promotion (mocked)."""
        from unittest.mock import patch
        
        with patch("nanobot.game.chesscom.gui_automation.pyautogui") as mock_gui:
            result = automation.execute_move("e7e8q", "white")
            
            # Should succeed (mocked)
            assert result is True

    def test_execute_move_invalid_format(self, automation):
        """Test executing move with invalid format."""
        from unittest.mock import patch
        
        with patch("nanobot.game.chesscom.gui_automation.pyautogui") as mock_gui:
            result = automation.execute_move("e2", "white")
            
            # Should fail
            assert result is False

    def test_random_delay(self, automation):
        """Test random delay."""
        import time
        
        start = time.time()
        automation.random_delay()
        elapsed = time.time() - start
        
        # Should be within min/max delay range
        assert automation.min_move_delay <= elapsed <= automation.max_move_delay * 1.5

    def test_handle_promotion(self, automation):
        """Test promotion handler (placeholder)."""
        # Should not raise any exceptions
        automation.handle_promotion("q")

    def test_import_error_without_pyautogui(self):
        """Test that ImportError is raised when pyautogui is not available."""
        from unittest.mock import patch
        
        with patch("nanobot.game.chesscom.gui_automation.pyautogui", None):
            from nanobot.game.chesscom.gui_automation import ChessComAutomation
            
            with pytest.raises(ImportError, match="pyautogui is required"):
                ChessComAutomation((100, 100, 600, 600))
