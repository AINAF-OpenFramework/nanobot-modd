#!/usr/bin/env python
"""
Chess Board Demo - Demonstrates Chess Scaffold and Visual Perception

This demo showcases:
1. Chess board state representation (scaffold)
2. Visual perception on chess boards
3. Strategy scoring hooks
4. Test position generation

Note: This is a scaffold. Full chess move generation and validation
require integration with a chess engine (e.g., python-chess).

Usage:
    python examples/chess_board_demo.py [--positions NUM]
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.game.fusion import FusionConfig, MultimodalFusionLayer
from nanobot.game.rules.chess import ChessRules
from nanobot.game.strategy_memory import StrategyMemory
from nanobot.game.visual_perception import SimpleGridEncoder

# Try to import PIL for visualization
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image saving disabled")


def create_chess_board_image(state: dict) -> np.ndarray:
    """Create a visual representation of a chess board."""
    size = 400
    cell_size = size // 8

    # Create checkerboard pattern
    img = np.zeros((size, size, 3), dtype=np.uint8)

    board = state.get("board", [])

    for row in range(8):
        for col in range(8):
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size

            # Checkerboard pattern
            if (row + col) % 2 == 0:
                img[y1:y2, x1:x2] = [240, 217, 181]  # Light square
            else:
                img[y1:y2, x1:x2] = [181, 136, 99]  # Dark square

            # Add piece indicator
            if row < len(board) and col < len(board[row]):
                piece = board[row][col]
                if piece:
                    color = piece[0]  # 'w' or 'b'
                    y_center = (y1 + y2) // 2
                    x_center = (x1 + x2) // 2
                    piece_size = cell_size // 3

                    if color == "w":
                        img[
                            y_center - piece_size : y_center + piece_size,
                            x_center - piece_size : x_center + piece_size,
                        ] = [255, 255, 255]
                    else:
                        img[
                            y_center - piece_size : y_center + piece_size,
                            x_center - piece_size : x_center + piece_size,
                        ] = [50, 50, 50]

    return img


def analyze_position(
    state: dict,
    rules: ChessRules,
    strategy_memory: StrategyMemory,
    fusion_layer: MultimodalFusionLayer,
    encoder: SimpleGridEncoder,
    position_name: str = "Position",
) -> dict:
    """
    Analyze a chess position using the game learning layer.

    Args:
        state: Chess game state
        rules: Chess rules
        strategy_memory: Strategy memory
        fusion_layer: Multimodal fusion layer
        encoder: Visual encoder
        position_name: Name for logging

    Returns:
        Analysis dictionary
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Analyzing: {position_name}")
    logger.info(f"{'=' * 60}")

    # Display board
    logger.info(f"\n{rules.get_board_string(state)}")

    # Visual encoding
    board_image = create_chess_board_image(state)
    visual_embedding = encoder.encode(board_image)

    logger.info(f"Visual embedding: {visual_embedding.dimensions}D, confidence={visual_embedding.confidence:.2f}")

    # Multimodal fusion
    fused = fusion_layer.fuse(
        game_state=state,
        visual_embedding=visual_embedding,
        query=f"chess position {state['current_player']}",
    )

    logger.info(f"Fused embedding: {fused.dimensions}D")
    logger.info(f"  - State component: {fused.components['state']:.3f}")
    logger.info(f"  - Visual component: {fused.components['visual']:.3f}")
    logger.info(f"  - Memory component: {fused.components['memory']:.3f}")

    # Position scoring (material count)
    score = rules.score_position(state)
    logger.info(f"Position score (material): {score:+.1f}")

    # Store as strategy (for future retrieval)
    strategy_memory.store_strategy(
        state=state,
        move="analysis",
        outcome={"score": score, "analysis": position_name},
        game_type="chess",
        tags=["position_analysis"],
    )

    return {
        "visual_dim": visual_embedding.dimensions,
        "fused_dim": fused.dimensions,
        "score": score,
    }


def run_demo(num_positions: int = 3):
    """Run the chess board demo."""
    logger.info("=" * 60)
    logger.info("Chess Board Demo - Visual Perception & Strategy Hooks")
    logger.info("=" * 60)
    logger.info("\nNote: This is a scaffold. Full chess engine integration")
    logger.info("is not yet implemented. This demo focuses on:")
    logger.info("  - Board representation")
    logger.info("  - Visual perception")
    logger.info("  - Strategy scoring hooks")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize components
        memory_store = MemoryStore(Path(tmpdir))
        strategy_memory = StrategyMemory(memory_store)

        fusion_config = FusionConfig(
            total_dim=512,
            state_dim=128,
            visual_dim=256,
            memory_dim=128,
        )
        fusion_layer = MultimodalFusionLayer(config=fusion_config, memory_store=memory_store)

        encoder = SimpleGridEncoder(embedding_dim=256, grid_size=(8, 8))
        rules = ChessRules()

        # Generate test positions
        logger.info(f"\nGenerating {num_positions} test positions...")
        positions = rules.generate_test_positions()

        if num_positions > len(positions):
            logger.warning(f"Requested {num_positions} but only {len(positions)} available")
            num_positions = len(positions)

        # Analyze each position
        analyses = []
        position_names = [
            "Initial Position",
            "After 1. e4",
            "Endgame: K+Q vs K",
        ]

        for i in range(num_positions):
            position_name = position_names[i] if i < len(position_names) else f"Position {i+1}"
            analysis = analyze_position(
                state=positions[i],
                rules=rules,
                strategy_memory=strategy_memory,
                fusion_layer=fusion_layer,
                encoder=encoder,
                position_name=position_name,
            )
            analyses.append(analysis)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Demo Complete - Summary")
        logger.info("=" * 60)
        logger.info(f"Positions analyzed: {num_positions}")
        logger.info(f"Strategy memory integration: ✅ Working")

        # Compare positions
        logger.info("\nPosition Scores (material count):")
        for i, (name, analysis) in enumerate(zip(position_names[:num_positions], analyses)):
            logger.info(f"  {name}: {analysis['score']:+.1f}")

        # Test strategy retrieval
        logger.info("\nTesting strategy retrieval...")
        test_state = positions[0]
        retrieved = strategy_memory.retrieve_relevant_strategies(
            state=test_state,
            k=3,
            game_type="chess",
        )
        logger.info(f"Retrieved {len(retrieved)} relevant strategies")

        logger.info("\n✅ Chess Board Demo completed successfully!")
        logger.info("\nNext steps for full chess support:")
        logger.info("  1. Integrate python-chess or similar engine")
        logger.info("  2. Implement legal move generation")
        logger.info("  3. Add position evaluation (beyond material)")
        logger.info("  4. Enable move application and validation")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chess Board Demo")
    parser.add_argument(
        "--positions",
        type=int,
        default=3,
        help="Number of positions to analyze (default: 3)",
    )

    args = parser.parse_args()

    run_demo(num_positions=args.positions)


if __name__ == "__main__":
    main()
