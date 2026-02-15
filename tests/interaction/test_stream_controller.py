import chess
import pytest

from nanobot.game.vtuber.soul_layer import SoulLayerIntegration
from nanobot.interaction.stream_controller import StreamController


@pytest.mark.asyncio
async def test_stream_controller_syncs_persona_voice_and_avatar():
    soul = SoulLayerIntegration()
    board = chess.Board()

    comment = soul.generate_comment("e2e4", board=board, score=0.6, previous_score=0.0)
    controller = StreamController()

    event = await controller.process_turn(
        text=comment,
        persona_cues={"mood": "thinking", "confidence": 0.8},
        game_state={"is_check": False},
        memory_context={
            "triune_tone": "sassy",
            "fractal_style": "analytical folklore",
        },
    )

    assert event.expression == "thinking"
    assert event.gesture == "idle"
    assert event.audio
    assert "sassy" in event.persona_text
    assert event.latency_ms >= 0
