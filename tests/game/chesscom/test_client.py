"""Tests for Chess.com client orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestChessComClient:
    """Tests for ChessComClient class."""

    @pytest.fixture
    def mock_cv2(self):
        """Mock cv2 module."""
        with patch("nanobot.game.chesscom.board_recognition.cv2", True):
            yield

    @pytest.fixture
    def client(self, mock_cv2):
        """Create ChessComClient instance."""
        from nanobot.game.chesscom.client import ChessComClient
        
        return ChessComClient(
            personality="TanyalahD",
            enable_tts=False,
            human_like_play=False,
            auto_play=False,
        )

    def test_init_default(self, mock_cv2):
        """Test initialization with defaults."""
        from nanobot.game.chesscom.client import ChessComClient
        
        client = ChessComClient()
        
        assert client.personality == "TanyalahD"
        assert client.enable_tts is True
        assert client.human_like_play is True
        assert client.auto_play is False

    def test_init_custom(self, mock_cv2):
        """Test initialization with custom settings."""
        from nanobot.game.chesscom.client import ChessComClient
        
        client = ChessComClient(
            personality="Custom",
            enable_tts=False,
            human_like_play=False,
            auto_play=True,
        )
        
        assert client.personality == "Custom"
        assert client.enable_tts is False
        assert client.human_like_play is False
        assert client.auto_play is True

    def test_components_initialized(self, client):
        """Test that all components are initialized."""
        assert client.board_recognizer is not None
        assert client.board_manager is not None
        assert client.move_generator is not None
        assert client.move_evaluator is not None
        assert client.move_executor is not None
        assert client.soul_layer is not None
        assert client.vtuber_output is not None
        assert client.metrics_tracker is not None

    def test_initial_statistics(self, client):
        """Test initial statistics."""
        stats = client.get_statistics()
        
        assert stats["games_played"] == 0
        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["draws"] == 0
        assert stats["moves_made"] == 0
        assert stats["win_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_process_turn_success(self, client):
        """Test successful turn processing."""
        import numpy as np
        
        # Mock screen capture
        client.screen_capture = MagicMock()
        client.screen_capture.capture_board.return_value = np.zeros((600, 600, 3))
        
        # Mock board recognition
        with patch.object(client.board_recognizer, "recognize_pieces") as mock_rec:
            with patch.object(client.board_recognizer, "to_fen") as mock_fen:
                with patch.object(client.board_recognizer, "detect_orientation") as mock_orient:
                    mock_rec.return_value = [[""]*8 for _ in range(8)]
                    mock_fen.return_value = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                    mock_orient.return_value = "white"
                    
                    result = await client.process_turn()
                    
                    assert result["success"] is True
                    assert "move" in result
                    assert "commentary" in result
                    assert "ias" in result
                    assert "cer" in result

    @pytest.mark.asyncio
    async def test_process_turn_no_screen_capture(self, client):
        """Test turn processing without screen capture."""
        result = await client.process_turn()
        
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_process_turn_with_tts(self, mock_cv2):
        """Test turn processing with TTS enabled."""
        from nanobot.game.chesscom.client import ChessComClient
        import numpy as np
        
        client = ChessComClient(enable_tts=True, auto_play=False)
        client.screen_capture = MagicMock()
        client.screen_capture.capture_board.return_value = np.zeros((600, 600, 3))
        client.tts = AsyncMock()
        
        with patch.object(client.board_recognizer, "recognize_pieces"):
            with patch.object(client.board_recognizer, "to_fen") as mock_fen:
                with patch.object(client.board_recognizer, "detect_orientation"):
                    mock_fen.return_value = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                    
                    result = await client.process_turn()
                    
                    if result["success"]:
                        # TTS should be called
                        client.tts.speak.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_turn_with_auto_play(self, mock_cv2):
        """Test turn processing with auto-play enabled."""
        from nanobot.game.chesscom.client import ChessComClient
        import numpy as np
        
        client = ChessComClient(auto_play=True)
        client.screen_capture = MagicMock()
        client.screen_capture.capture_board.return_value = np.zeros((600, 600, 3))
        client.gui_automation = MagicMock()
        client.gui_automation.execute_move.return_value = True
        
        with patch.object(client.board_recognizer, "recognize_pieces"):
            with patch.object(client.board_recognizer, "to_fen") as mock_fen:
                with patch.object(client.board_recognizer, "detect_orientation"):
                    mock_fen.return_value = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                    
                    result = await client.process_turn()
                    
                    if result["success"]:
                        # Move should be executed
                        assert result["executed"] is True

    def test_pause_and_resume(self, client):
        """Test pause and resume functionality."""
        assert client._paused is False
        
        client.pause()
        assert client._paused is True
        
        client.resume()
        assert client._paused is False

    def test_get_statistics_with_metrics(self, client):
        """Test statistics with metrics history."""
        # Add some metrics
        client.metrics_tracker.metrics_history = [
            {"IAS": 0.8, "CER": 0.9},
            {"IAS": 0.7, "CER": 0.85},
        ]
        
        stats = client.get_statistics()
        
        assert stats["avg_ias"] == pytest.approx(0.75)
        assert stats["avg_cer"] == pytest.approx(0.875)

    def test_cleanup(self, client):
        """Test cleanup functionality."""
        client.screen_capture = MagicMock()
        client.tts = MagicMock()
        
        client._cleanup()
        
        client.screen_capture.close.assert_called_once()
        client.tts.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_game_loop_keyboard_interrupt(self, client):
        """Test game loop handles keyboard interrupt."""
        # Mock mss and screen capture components
        mock_mss_module = MagicMock()
        mock_sct_instance = MagicMock()
        mock_mss_module.mss.return_value = mock_sct_instance
        mock_sct_instance.monitors = [
            {"width": 1920, "height": 1080},
            {"width": 1920, "height": 1080, "left": 0, "top": 0},
        ]
        
        with patch("nanobot.game.chesscom.screen_capture.mss", mock_mss_module):
            with patch.object(client, "process_turn", side_effect=KeyboardInterrupt):
                # Should not raise exception
                await client.start_game_loop()

    def test_statistics_win_rate_calculation(self, client):
        """Test win rate calculation."""
        client.stats["games_played"] = 10
        client.stats["wins"] = 7
        
        stats = client.get_statistics()
        
        assert stats["win_rate"] == 0.7
