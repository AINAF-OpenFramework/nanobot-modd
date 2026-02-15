"""TTS integration module for VTuber commentary."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None


class TTSIntegration:
    """
    Integrates with TTS APIs for VTuber commentary.
    
    Supports multiple providers (ElevenLabs, Google, local TTS) with
    async audio generation, playback, and caching.
    """

    def __init__(
        self,
        provider: str = "elevenlabs",
        voice_id: str | None = None,
        api_key: str | None = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize TTS integration.
        
        Args:
            provider: TTS provider ("elevenlabs", "google", "local")
            voice_id: Voice ID for the selected provider
            api_key: API key (from environment if not provided)
            cache_enabled: Enable caching of generated audio
        """
        self.provider = provider
        self.voice_id = voice_id
        self.cache_enabled = cache_enabled
        
        # Get API key from environment if not provided
        if api_key is None:
            if provider == "elevenlabs":
                api_key = os.environ.get("ELEVENLABS_API_KEY")
            elif provider == "google":
                api_key = os.environ.get("GOOGLE_TTS_API_KEY")
        
        self.api_key = api_key
        
        # Setup cache directory
        self.cache_dir = Path.home() / ".cache" / "nanobot" / "tts"
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio playback state
        self._current_stream: Any = None
        
        logger.debug(
            f"TTSIntegration initialized: provider={provider}, "
            f"voice_id={voice_id}, cache={cache_enabled}"
        )

    async def speak(self, text: str, block: bool = False) -> None:
        """
        Generate and play speech from text.
        
        Args:
            text: Text to speak
            block: If True, wait for audio to finish before returning
            
        Example:
            >>> tts = TTSIntegration(provider="local")
            >>> await tts.speak("Hello, chess fans!")
        """
        try:
            # Generate audio
            audio_data = await self.generate_audio(text)
            
            # Play audio
            if block:
                self.play_audio(audio_data)
            else:
                # Non-blocking: play in background
                asyncio.create_task(self._play_audio_async(audio_data))
            
            logger.info(f"Speaking: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to speak text: {e}")

    async def generate_audio(self, text: str) -> bytes:
        """
        Generate audio bytes from text.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(text)
            cache_path = self.cache_dir / f"{cache_key}.wav"
            
            if cache_path.exists():
                logger.debug(f"Using cached audio for: {text[:30]}...")
                return cache_path.read_bytes()
        
        # Generate new audio
        if self.provider == "elevenlabs":
            audio_data = await self._generate_elevenlabs(text)
        elif self.provider == "google":
            audio_data = await self._generate_google(text)
        else:  # local
            audio_data = await self._generate_local(text)
        
        # Cache if enabled
        if self.cache_enabled:
            cache_path.write_bytes(audio_data)
            logger.debug(f"Cached audio for: {text[:30]}...")
        
        return audio_data

    async def _generate_elevenlabs(self, text: str) -> bytes:
        """
        Generate audio using ElevenLabs API.
        
        Note: This is a placeholder implementation. In production, this would
        make actual API calls to ElevenLabs.
        """
        if httpx is None:
            raise ImportError("httpx is required for API-based TTS")
        
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")
        
        # Placeholder: Would make actual API call
        logger.warning("ElevenLabs TTS is not fully implemented (placeholder)")
        return await self._generate_local(text)

    async def _generate_google(self, text: str) -> bytes:
        """
        Generate audio using Google Cloud TTS.
        
        Note: This is a placeholder implementation.
        """
        if httpx is None:
            raise ImportError("httpx is required for API-based TTS")
        
        # Placeholder: Would make actual API call
        logger.warning("Google TTS is not fully implemented (placeholder)")
        return await self._generate_local(text)

    async def _generate_local(self, text: str) -> bytes:
        """
        Generate audio using local TTS.
        
        Note: This is a minimal placeholder that returns silence.
        In production, this would use a local TTS engine like pyttsx3 or espeak.
        """
        # Placeholder: Generate 1 second of silence as WAV
        import struct
        import wave
        from io import BytesIO
        
        sample_rate = 16000
        duration = 1.0  # seconds
        num_samples = int(sample_rate * duration)
        
        # Create WAV file in memory
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Write silence
            for _ in range(num_samples):
                wav_file.writeframes(struct.pack("<h", 0))
        
        logger.debug(f"Generated local TTS (placeholder): {text[:30]}...")
        return buffer.getvalue()

    def play_audio(self, audio_data: bytes) -> None:
        """
        Play audio data.
        
        Args:
            audio_data: Audio data to play (WAV format)
        """
        if sd is None or sf is None:
            logger.warning("sounddevice/soundfile not available, skipping audio playback")
            return
        
        try:
            from io import BytesIO
            
            # Read audio data
            buffer = BytesIO(audio_data)
            data, sample_rate = sf.read(buffer)
            
            # Play audio
            sd.play(data, sample_rate)
            sd.wait()
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")

    async def _play_audio_async(self, audio_data: bytes) -> None:
        """Play audio asynchronously."""
        await asyncio.to_thread(self.play_audio, audio_data)

    def stop(self) -> None:
        """Stop current audio playback."""
        if sd is not None:
            sd.stop()
        logger.debug("Stopped audio playback")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Include provider and voice_id in hash
        cache_string = f"{self.provider}:{self.voice_id}:{text}"
        return hashlib.md5(cache_string.encode()).hexdigest()
