"""Tests for Chess.com screen capture module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestChessComScreenCapture:
    """Tests for ChessComScreenCapture class."""

    @pytest.fixture
    def mock_mss(self):
        """Mock mss module."""
        with patch("nanobot.game.chesscom.screen_capture.mss") as mock:
            mock_instance = MagicMock()
            mock.mss.return_value = mock_instance

            # Mock monitor info
            mock_instance.monitors = [
                {"width": 1920, "height": 1080},  # All monitors
                {"width": 1920, "height": 1080, "left": 0, "top": 0},  # Primary
            ]

            # Mock screenshot - return a proper numpy array directly
            test_image = np.random.randint(0, 255, (600, 600, 4), dtype=np.uint8)
            mock_screenshot = MagicMock()
            # Make np.array(mock_screenshot) work properly
            mock_screenshot.__array__ = lambda dtype=None, copy=True: test_image
            mock_instance.grab.return_value = mock_screenshot

            yield mock

    @pytest.fixture
    def screen_capture(self, mock_mss):
        """Create ChessComScreenCapture instance."""
        from nanobot.game.chesscom.screen_capture import ChessComScreenCapture

        return ChessComScreenCapture(region=(100, 100, 600, 600))

    def test_init_with_region(self, mock_mss):
        """Test initialization with explicit region."""
        from nanobot.game.chesscom.screen_capture import ChessComScreenCapture

        region = (100, 100, 600, 600)
        capture = ChessComScreenCapture(region=region)

        assert capture._region == region
        assert capture._board_region is None

    def test_init_without_region(self, mock_mss):
        """Test initialization without region (auto-detect)."""
        from nanobot.game.chesscom.screen_capture import ChessComScreenCapture

        capture = ChessComScreenCapture()

        assert capture._region is None
        assert capture._board_region is None

    def test_detect_board_region(self, mock_mss):
        """Test board region detection."""
        from nanobot.game.chesscom.screen_capture import ChessComScreenCapture

        capture = ChessComScreenCapture()
        region = capture.detect_board_region()

        assert isinstance(region, tuple)
        assert len(region) == 4
        assert all(isinstance(x, int) for x in region)

        x, y, width, height = region
        assert width > 0
        assert height > 0

    def test_capture_board_with_region(self, screen_capture, mock_mss):
        """Test board capture with explicit region."""
        img = screen_capture.capture_board()

        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3  # Height x Width x Channels
        assert img.shape[2] == 3  # RGB

    def test_capture_board_auto_detect(self, mock_mss):
        """Test board capture with auto-detection."""
        from nanobot.game.chesscom.screen_capture import ChessComScreenCapture

        capture = ChessComScreenCapture()
        img = capture.capture_board()

        assert isinstance(img, np.ndarray)
        assert capture._board_region is not None

    def test_is_my_turn(self, screen_capture):
        """Test turn detection."""
        # Placeholder implementation always returns True
        result = screen_capture.is_my_turn()
        assert isinstance(result, bool)

    def test_get_game_status(self, screen_capture):
        """Test game status detection."""
        status = screen_capture.get_game_status()
        assert isinstance(status, str)
        assert status in ["playing", "ended", "waiting"]

    def test_close(self, screen_capture, mock_mss):
        """Test cleanup."""
        screen_capture.close()
        # Should not raise any exceptions

    def test_import_error_without_mss(self):
        """Test that ImportError is raised when mss is not available."""
        with patch("nanobot.game.chesscom.screen_capture.mss", None):
            from nanobot.game.chesscom.screen_capture import ChessComScreenCapture

            with pytest.raises(ImportError, match="mss is required"):
                ChessComScreenCapture()
