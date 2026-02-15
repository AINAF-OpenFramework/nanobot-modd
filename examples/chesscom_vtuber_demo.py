#!/usr/bin/env python
"""
Chess.com VTuber Demo - TanyalahD plays Chess.com autonomously.

This demo showcases the full integration of TanyalahD VTuber personality
with Chess.com for autonomous chess play with real-time commentary.

Usage:
    python examples/chesscom_vtuber_demo.py [--no-tts] [--manual]

Options:
    --no-tts    Disable text-to-speech commentary
    --manual    Manual mode: suggest moves but don't execute them

Requirements:
    - Chess.com open in browser with visible board
    - Screen region configured (auto-detects by default)
    - Optional: ElevenLabs API key for high-quality TTS

Controls:
    - Ctrl+C: Pause/stop the game loop
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from nanobot.game.chesscom import ChessComClient


async def main(enable_tts: bool = True, auto_play: bool = True) -> None:
    """
    Run the Chess.com VTuber demo.

    Args:
        enable_tts: Enable text-to-speech commentary
        auto_play: Automatically execute moves (False for manual mode)
    """
    # Initialize the client
    client = ChessComClient(
        personality="TanyalahD",
        enable_tts=enable_tts,
        human_like_play=True,
        auto_play=auto_play,
    )

    # Display startup info
    print("=" * 60)
    print("🎮 TanyalahD Chess VTuber - Chess.com Integration")
    print("=" * 60)
    print()
    print("📺 Make sure Chess.com is visible on your screen")
    print("🎯 The chess board will be auto-detected")
    print("⏸️  Press Ctrl+C to pause or stop")
    print()

    if auto_play:
        print("🤖 AUTO-PLAY MODE: TanyalahD will make moves automatically")
    else:
        print("💭 MANUAL MODE: TanyalahD will suggest moves only")

    if enable_tts:
        print("🔊 TTS ENABLED: Commentary will be spoken")
    else:
        print("🔇 TTS DISABLED: Text commentary only")

    print()
    print("Starting in 3 seconds...")
    print("=" * 60)

    await asyncio.sleep(3)

    # Start the game loop
    try:
        await client.start_game_loop()
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    finally:
        # Display statistics
        stats = client.get_statistics()
        print()
        print("=" * 60)
        print("📊 Session Statistics")
        print("=" * 60)
        print(f"Games Played: {stats['games_played']}")
        print(f"Wins: {stats['wins']}")
        print(f"Losses: {stats['losses']}")
        print(f"Draws: {stats['draws']}")
        print(f"Moves Made: {stats['moves_made']}")
        print(f"Win Rate: {stats['win_rate']:.1%}")
        print(f"Avg IAS: {stats['avg_ias']:.3f}")
        print(f"Avg CER: {stats['avg_cer']:.3f}")
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chess.com VTuber Demo - TanyalahD plays chess autonomously",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech commentary",
    )

    parser.add_argument(
        "--manual",
        action="store_true",
        help="Manual mode: suggest moves but don't execute them",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        main(
            enable_tts=not args.no_tts,
            auto_play=not args.manual,
        )
    )
