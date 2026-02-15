import pytest

from nanobot.tts.elevenlab_voice import ElevenLabsVoice


@pytest.mark.asyncio
async def test_elevenlabs_fallback_synthesizes_without_api_key():
    tts = ElevenLabsVoice(api_key=None, voice_id="tanyalahd")

    result = await tts.synthesize("Brilliant fork on f7!")

    assert result.voice_id == "tanyalahd"
    assert b"Brilliant fork" in result.audio


@pytest.mark.asyncio
async def test_tts_stream_returns_chunked_audio():
    tts = ElevenLabsVoice(api_key=None)

    chunks = [chunk async for chunk in tts.stream("hello", chunk_size=4)]

    assert chunks
    assert all(len(chunk) <= 4 for chunk in chunks)
