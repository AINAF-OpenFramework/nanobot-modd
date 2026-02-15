"""Orchestrates Nanobot text, voice synthesis, and avatar reactions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import monotonic
from typing import Any

from nanobot.tts.elevenlab_voice import ElevenLabsVoice, TTSResult
from nanobot.vtuber.avatar import AvatarController
from nanobot.vtuber.body_tracker import BodyTracker
from nanobot.vtuber.face_tracker import FaceTracker


@dataclass(slots=True)
class StreamEvent:
    """Result of a synchronized stream turn."""

    text: str
    persona_text: str
    expression: str
    gesture: str
    audio: bytes
    latency_ms: float


class StreamController:
    """High-level controller that synchronizes persona, avatar, and voice."""

    def __init__(
        self,
        avatar: AvatarController | None = None,
        face_tracker: FaceTracker | None = None,
        body_tracker: BodyTracker | None = None,
        tts: ElevenLabsVoice | None = None,
        enable_avatar: bool = True,
        enable_tts: bool = True,
    ):
        self.avatar = avatar or AvatarController(enable_streaming=enable_avatar)
        self.face_tracker = face_tracker or FaceTracker(enabled=False)
        self.body_tracker = body_tracker or BodyTracker(enabled=False)
        self.tts = tts or ElevenLabsVoice(enabled=enable_tts)
        self.enable_avatar = enable_avatar
        self.enable_tts = enable_tts

    async def process_turn(
        self,
        text: str,
        persona_cues: dict[str, str | float],
        game_state: dict[str, Any] | None = None,
        memory_context: dict[str, str] | None = None,
    ) -> StreamEvent:
        """Process one turn: text -> persona -> avatar -> speech."""
        started = monotonic()
        persona_text = self._compose_persona_text(text, memory_context)

        expression, intensity = self.face_tracker.map_persona_to_expression(persona_cues)
        phase = self._infer_phase(persona_cues, game_state)
        gesture = self.body_tracker.map_context_to_gesture({"phase": phase})

        tasks = []
        if self.enable_avatar:
            tasks.append(self._apply_avatar_updates(expression, intensity, gesture))
        if self.enable_tts:
            tasks.append(self.tts.synthesize(persona_text))

        results = await asyncio.gather(*tasks) if tasks else []

        tts_result = TTSResult(audio=b"", text=persona_text, voice_id=self.tts.voice_id)
        for item in results:
            if isinstance(item, TTSResult):
                tts_result = item

        latency_ms = (monotonic() - started) * 1000.0
        return StreamEvent(
            text=text,
            persona_text=persona_text,
            expression=expression,
            gesture=gesture,
            audio=tts_result.audio,
            latency_ms=latency_ms,
        )

    async def _apply_avatar_updates(self, expression: str, intensity: float, gesture: str) -> None:
        self.avatar.apply_expression(expression, intensity)
        self.avatar.apply_gesture(gesture)

    def _compose_persona_text(self, text: str, memory_context: dict[str, str] | None) -> str:
        """Preserve tone using available Triune/Fractal context hints."""
        if not memory_context:
            return text

        tone = memory_context.get("triune_tone")
        style = memory_context.get("fractal_style")
        prefix_parts = [part for part in [tone, style] if part]
        if not prefix_parts:
            return text
        return f"[{'; '.join(prefix_parts)}] {text}"

    def _infer_phase(
        self,
        persona_cues: dict[str, str | float],
        game_state: dict[str, Any] | None,
    ) -> str:
        if "phase" in persona_cues:
            return str(persona_cues["phase"])
        if game_state and game_state.get("is_checkmate"):
            return "checkmate"
        if game_state and game_state.get("is_check"):
            return "check"
        return "neutral"
