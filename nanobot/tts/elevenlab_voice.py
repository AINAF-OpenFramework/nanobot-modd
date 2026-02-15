"""ElevenLabs-compatible TTS client with non-blocking helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class TTSResult:
    """Text-to-speech output payload."""

    audio: bytes
    text: str
    voice_id: str
    provider: str = "elevenlabs"


class ElevenLabsVoice:
    """Minimal ElevenLabs API wrapper with deterministic fallback mode."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = "default",
        model_id: str = "eleven_turbo_v2",
        enabled: bool = True,
        base_url: str = "https://api.elevenlabs.io/v1",
        timeout_seconds: float = 8.0,
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize text to speech bytes.

        If API credentials are not configured, this returns lightweight simulated
        audio bytes so integration can run in local and test environments.
        """
        if not self.enabled:
            return TTSResult(audio=b"", text=text, voice_id=self.voice_id)

        if not self.api_key:
            payload = f"simulated-audio:{self.voice_id}:{text}".encode("utf-8")
            return TTSResult(audio=payload, text=text, voice_id=self.voice_id)

        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        request = {
            "text": text,
            "model_id": self.model_id,
        }
        headers = {
            "xi-api-key": self.api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, json=request, headers=headers)
            response.raise_for_status()
            return TTSResult(audio=response.content, text=text, voice_id=self.voice_id)

    async def stream(self, text: str, chunk_size: int = 1024):
        """Yield audio chunks to allow non-blocking playback pipelines."""
        result = await self.synthesize(text)
        if not result.audio:
            return

        for i in range(0, len(result.audio), chunk_size):
            yield result.audio[i : i + chunk_size]
            await asyncio.sleep(0)
