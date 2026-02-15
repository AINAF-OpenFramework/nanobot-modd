# Voice Module

`nanobot.tts.elevenlab_voice.ElevenLabsVoice` provides async voice synthesis for stream flows.

- Uses ElevenLabs API when `api_key` is configured
- Falls back to deterministic simulated audio bytes when API key is absent (for local/dev tests)
- Supports chunked async streaming via `stream()` for non-blocking playback pipelines

This is integrated by default through `nanobot.interaction.StreamController`.
