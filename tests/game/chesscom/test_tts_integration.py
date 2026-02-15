"""Tests for Chess.com TTS integration module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTTSIntegration:
    """Tests for TTSIntegration class."""

    @pytest.fixture
    def tts(self, tmp_path):
        """Create TTSIntegration instance with temp cache."""
        from nanobot.game.chesscom.tts_integration import TTSIntegration
        
        with patch.object(Path, "home", return_value=tmp_path):
            return TTSIntegration(
                provider="local",
                cache_enabled=True,
            )

    def test_init_default(self, tmp_path):
        """Test initialization with defaults."""
        from nanobot.game.chesscom.tts_integration import TTSIntegration
        
        with patch.object(Path, "home", return_value=tmp_path):
            tts = TTSIntegration()
            
            assert tts.provider == "elevenlabs"
            assert tts.cache_enabled is True

    def test_init_with_api_key(self, tmp_path):
        """Test initialization with explicit API key."""
        from nanobot.game.chesscom.tts_integration import TTSIntegration
        
        with patch.object(Path, "home", return_value=tmp_path):
            tts = TTSIntegration(
                provider="elevenlabs",
                voice_id="test_voice",
                api_key="test_key",
            )
            
            assert tts.api_key == "test_key"
            assert tts.voice_id == "test_voice"

    def test_init_api_key_from_env(self, tmp_path, monkeypatch):
        """Test API key loaded from environment."""
        from nanobot.game.chesscom.tts_integration import TTSIntegration
        
        monkeypatch.setenv("ELEVENLABS_API_KEY", "env_key")
        
        with patch.object(Path, "home", return_value=tmp_path):
            tts = TTSIntegration(provider="elevenlabs")
            
            assert tts.api_key == "env_key"

    @pytest.mark.asyncio
    async def test_speak_non_blocking(self, tts):
        """Test non-blocking speech."""
        with patch.object(tts, "generate_audio", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = b"audio_data"
            
            with patch.object(tts, "play_audio"):
                await tts.speak("Test message", block=False)
                
                mock_gen.assert_called_once_with("Test message")

    @pytest.mark.asyncio
    async def test_speak_blocking(self, tts):
        """Test blocking speech."""
        with patch.object(tts, "generate_audio", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = b"audio_data"
            
            with patch.object(tts, "play_audio") as mock_play:
                await tts.speak("Test message", block=True)
                
                mock_gen.assert_called_once_with("Test message")
                mock_play.assert_called_once_with(b"audio_data")

    @pytest.mark.asyncio
    async def test_generate_audio_local(self, tts):
        """Test local audio generation."""
        audio_data = await tts.generate_audio("Test message")
        
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        
        # Should be valid WAV file
        assert audio_data.startswith(b"RIFF")

    @pytest.mark.asyncio
    async def test_generate_audio_caching(self, tts):
        """Test audio caching."""
        # Generate audio first time
        audio1 = await tts.generate_audio("Test message")
        
        # Generate same text again
        audio2 = await tts.generate_audio("Test message")
        
        # Should return cached result
        assert audio1 == audio2

    @pytest.mark.asyncio
    async def test_generate_audio_elevenlabs_fallback(self, tmp_path):
        """Test ElevenLabs falls back to local."""
        from nanobot.game.chesscom.tts_integration import TTSIntegration
        
        with patch.object(Path, "home", return_value=tmp_path):
            tts = TTSIntegration(provider="elevenlabs", api_key="test")
            
            audio_data = await tts.generate_audio("Test")
            
            # Should fall back to local implementation
            assert isinstance(audio_data, bytes)

    def test_play_audio_no_sounddevice(self, tts):
        """Test play_audio when sounddevice is not available."""
        with patch("nanobot.game.chesscom.tts_integration.sd", None):
            # Should not raise, just log warning
            tts.play_audio(b"audio_data")

    def test_play_audio_with_sounddevice(self, tts):
        """Test play_audio with sounddevice mocked."""
        with patch("nanobot.game.chesscom.tts_integration.sd") as mock_sd:
            with patch("nanobot.game.chesscom.tts_integration.sf") as mock_sf:
                mock_sf.read.return_value = ([0.0] * 100, 16000)
                
                tts.play_audio(b"audio_data")
                
                mock_sd.play.assert_called_once()

    def test_stop(self, tts):
        """Test stop functionality."""
        with patch("nanobot.game.chesscom.tts_integration.sd") as mock_sd:
            tts.stop()
            mock_sd.stop.assert_called_once()

    def test_get_cache_key(self, tts):
        """Test cache key generation."""
        key1 = tts._get_cache_key("Test message")
        key2 = tts._get_cache_key("Test message")
        key3 = tts._get_cache_key("Different message")
        
        # Same text should generate same key
        assert key1 == key2
        
        # Different text should generate different key
        assert key1 != key3
        
        # Should be valid MD5 hash
        assert len(key1) == 32
        assert all(c in "0123456789abcdef" for c in key1)

    def test_cache_directory_creation(self, tmp_path):
        """Test cache directory is created."""
        from nanobot.game.chesscom.tts_integration import TTSIntegration
        
        with patch.object(Path, "home", return_value=tmp_path):
            tts = TTSIntegration(cache_enabled=True)
            
            assert tts.cache_dir.exists()
            assert tts.cache_dir.is_dir()
