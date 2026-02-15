#!/usr/bin/env python
"""VTuber + voice chess demo using modular stream controller."""

from __future__ import annotations

import argparse
import asyncio

import chess
from loguru import logger

from nanobot.game.vtuber.soul_layer import SoulLayerIntegration
from nanobot.interaction.stream_controller import StreamController


async def run_demo(moves: int = 4) -> None:
    board = chess.Board()
    soul = SoulLayerIntegration(personality="TanyalahD")
    controller = StreamController()

    logger.info("Starting VTuber chess demo")
    for index in range(1, moves + 1):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = legal_moves[0]
        move_uci = move.uci()
        comment = soul.generate_comment(move_uci, board=board, score=0.2 * index, previous_score=0.0)

        event = await controller.process_turn(
            text=comment,
            persona_cues={"mood": "thinking", "confidence": 0.75},
            game_state={"ply": board.ply()},
            memory_context={
                "triune_tone": "sassy",
                "fractal_style": "analytical folklore",
            },
        )

        board.push(move)
        logger.info(
            "Move {} {} | expr={} gesture={} audio_bytes={} latency_ms={:.1f}",
            index,
            move_uci,
            event.expression,
            event.gesture,
            len(event.audio),
            event.latency_ms,
        )

    logger.info("Demo complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="VTuber chess demo")
    parser.add_argument("--moves", type=int, default=4, help="Number of legal moves to simulate")
    args = parser.parse_args()

    asyncio.run(run_demo(args.moves))


if __name__ == "__main__":
    main()
