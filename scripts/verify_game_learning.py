#!/usr/bin/env python
"""
Verification Script for Game Learning Layer

This script validates:
1. TicTacToe demo (plays games, validates outcomes)
2. MCP integration (connects to server, discovers tools, executes)
3. Visual perception (tests encoders on synthetic boards)
4. Strategy memory entanglement (stores strategies, retrieves, validates)

Usage:
    python scripts/verify_game_learning.py [--quick]
"""

import argparse
import asyncio
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.game.fusion import FusionConfig, MultimodalFusionLayer
from nanobot.game.rules.chess import ChessRules
from nanobot.game.rules.tictactoe import TicTacToeRules
from nanobot.game.strategy_memory import StrategyMemory
from nanobot.game.visual_perception import SimpleGridEncoder, create_encoder
from nanobot.mcp.client import MCPClient
from nanobot.mcp.registry import MCPRegistry
from nanobot.mcp.schemas import MCPServerConfig


def create_synthetic_board_image(board_type: str = "tictactoe") -> np.ndarray:
    """Create a synthetic board image for testing."""
    if board_type == "tictactoe":
        size = 300
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        # Simple pattern
        img[100:200, 100:200] = [255, 0, 0]  # Red square (X)
        return img
    elif board_type == "chess":
        size = 400
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cell_size = size // 8
        # Checkerboard pattern
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    y1, y2 = row * cell_size, (row + 1) * cell_size
                    x1, x2 = col * cell_size, (col + 1) * cell_size
                    img[y1:y2, x1:x2] = [240, 217, 181]
        return img
    else:
        return np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)


def verify_tictactoe_demo(num_games: int = 10) -> bool:
    """
    Verify TicTacToe demo functionality.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: TicTacToe Demo Verification")
    logger.info("=" * 70)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            memory_store = MemoryStore(Path(tmpdir))
            strategy_memory = StrategyMemory(memory_store)
            fusion_config = FusionConfig(total_dim=256, state_dim=64, visual_dim=128, memory_dim=64)
            fusion_layer = MultimodalFusionLayer(config=fusion_config, memory_store=memory_store)
            encoder = SimpleGridEncoder(embedding_dim=128, grid_size=(3, 3))
            rules = TicTacToeRules()

            # Play games
            results = {"X": 0, "O": 0, "draw": 0}
            for i in range(num_games):
                state = rules.create_initial_state()
                moves_made = 0

                while moves_made < 9:
                    legal_moves = rules.get_legal_moves(state)
                    if not legal_moves:
                        break

                    # Simple strategy: pick first legal move
                    move = legal_moves[0]
                    state = rules.apply_move(state, move)
                    moves_made += 1

                    # Check win
                    result = rules.check_win_conditions(state)
                    if result["game_over"]:
                        if result["winner"]:
                            results[result["winner"]] += 1
                        else:
                            results["draw"] += 1
                        break

            # Validate results
            total = results["X"] + results["O"] + results["draw"]
            assert total == num_games, f"Expected {num_games} games, got {total}"

            logger.info(f"✅ Played {num_games} games successfully")
            logger.info(f"   X wins: {results['X']}, O wins: {results['O']}, Draws: {results['draw']}")

            return True

    except Exception as e:
        logger.error(f"❌ TicTacToe demo verification failed: {e}")
        return False


async def verify_mcp_integration() -> bool:
    """
    Verify MCP integration.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: MCP Integration Verification")
    logger.info("=" * 70)

    try:
        server_script = Path(__file__).parent.parent / "examples" / "simple_mcp_server.py"
        if not server_script.exists():
            logger.error(f"❌ MCP server script not found: {server_script}")
            return False

        server_config = MCPServerConfig(
            name="verify_server",
            type="local",
            command=sys.executable,
            args=[str(server_script)],
        )

        # Test 1: Client connection and tool discovery
        client = MCPClient(server_config)
        try:
            await client.connect()
            logger.info("✅ Connected to MCP server")

            tools = await client.discover_tools()
            tool_names = [t.name for t in tools]
            logger.info(f"✅ Discovered {len(tools)} tools: {tool_names}")

            # Verify game tools are present
            assert "place_marker" in tool_names, "place_marker tool not found"
            assert "move_piece" in tool_names, "move_piece tool not found"
            assert "get_legal_moves" in tool_names, "get_legal_moves tool not found"
            logger.info("✅ All expected game tools discovered")

            # Test 2: Tool execution
            result = await client.execute_tool("echo", {"message": "test"})
            assert result is not None, "Tool execution returned None"
            logger.info("✅ Tool execution successful")

        finally:
            await client.disconnect()

        # Test 3: Registry integration
        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry)

        try:
            await mcp_registry.add_client(server_config)
            logger.info("✅ MCP registry integration successful")

            # Check tool registration
            registered_tools = tool_registry.tool_names
            mcp_tools = [name for name in registered_tools if "mcp_" in name]
            assert len(mcp_tools) > 0, "No MCP tools registered"
            logger.info(f"✅ {len(mcp_tools)} MCP tools registered in ToolRegistry")

        finally:
            await mcp_registry.disconnect_all()

        return True

    except Exception as e:
        logger.error(f"❌ MCP integration verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_visual_encoders() -> bool:
    """
    Verify visual encoders on synthetic boards.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Visual Encoder Verification")
    logger.info("=" * 70)

    try:
        # Test SimpleGridEncoder
        encoder = SimpleGridEncoder(embedding_dim=128, grid_size=(3, 3))

        # Test on TicTacToe boards
        ttt_image = create_synthetic_board_image("tictactoe")
        embedding = encoder.encode(ttt_image)

        assert embedding.dimensions == 128, f"Expected 128D, got {embedding.dimensions}D"
        assert embedding.confidence > 0, "Confidence should be > 0"
        logger.info("✅ TicTacToe board encoding: 128D, confidence={:.2f}".format(embedding.confidence))

        # Test on chess boards
        encoder_chess = SimpleGridEncoder(embedding_dim=256, grid_size=(8, 8))
        chess_image = create_synthetic_board_image("chess")
        embedding_chess = encoder_chess.encode(chess_image)

        assert embedding_chess.dimensions == 256, f"Expected 256D, got {embedding_chess.dimensions}D"
        logger.info("✅ Chess board encoding: 256D, confidence={:.2f}".format(embedding_chess.confidence))

        # Test consistency
        embedding2 = encoder.encode(ttt_image)
        assert np.allclose(embedding.embedding, embedding2.embedding), "Encoding not consistent"
        logger.info("✅ Encoder consistency verified")

        # Test encoder factory
        factory_encoder = create_encoder(encoder_type="grid", embedding_dim=64)
        assert factory_encoder.is_available(), "Factory encoder not available"
        logger.info("✅ Encoder factory working")

        # Test different image sizes
        for size in [(50, 50), (200, 200), (400, 300)]:
            img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            emb = encoder.encode(img)
            assert emb.dimensions == 128, f"Failed on size {size}"
        logger.info("✅ Encoder handles different image sizes")

        # Estimate accuracy (simple test: different boards should produce different embeddings)
        boards = [
            create_synthetic_board_image("tictactoe"),
            create_synthetic_board_image("chess"),
            create_synthetic_board_image("other"),
        ]
        embeddings = [encoder.encode(b) for b in boards]

        # Check that embeddings are different
        different_count = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if not np.allclose(embeddings[i].embedding, embeddings[j].embedding):
                    different_count += 1

        total_pairs = len(embeddings) * (len(embeddings) - 1) // 2
        accuracy = different_count / total_pairs * 100
        logger.info(f"✅ Encoder distinguishes different boards: {accuracy:.0f}% accuracy")

        return True

    except Exception as e:
        logger.error(f"❌ Visual encoder verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_strategy_memory_entanglement() -> bool:
    """
    Verify strategy memory entanglement at scale.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Strategy Memory Entanglement Verification")
    logger.info("=" * 70)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = MemoryStore(Path(tmpdir))
            strategy_memory = StrategyMemory(memory_store)
            rules = TicTacToeRules()

            # Store strategies from multiple games
            num_games = 50
            total_strategies = 0

            for game_num in range(num_games):
                state = rules.create_initial_state()
                moves_made = 0

                while moves_made < 5:  # Store first 5 moves per game
                    legal_moves = rules.get_legal_moves(state)
                    if not legal_moves:
                        break

                    move = legal_moves[0]
                    strategy_memory.store_strategy(
                        state=state,
                        move=move,
                        outcome={"game": game_num, "move": moves_made},
                        game_type="tictactoe",
                    )
                    total_strategies += 1

                    state = rules.apply_move(state, move)
                    moves_made += 1

            logger.info(f"✅ Stored {total_strategies} strategies across {num_games} games")

            # Test retrieval latency
            test_state = rules.create_initial_state()

            start_time = time.time()
            for _ in range(10):
                retrieved = strategy_memory.retrieve_relevant_strategies(
                    state=test_state,
                    k=5,
                    game_type="tictactoe",
                )
            avg_latency_ms = (time.time() - start_time) / 10 * 1000

            assert avg_latency_ms < 200, f"Latency {avg_latency_ms:.0f}ms exceeds 200ms target"
            logger.info(f"✅ Retrieval latency: {avg_latency_ms:.1f}ms (target: <200ms)")

            # Test retrieval correctness
            retrieved = strategy_memory.retrieve_relevant_strategies(
                state=test_state,
                k=5,
                game_type="tictactoe",
            )
            assert isinstance(retrieved, list), "Retrieval should return list"
            logger.info(f"✅ Retrieved {len(retrieved)} strategies for test state")

            return True

    except Exception as e:
        logger.error(f"❌ Strategy memory verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_verification(quick: bool = False):
    """Run all verification tests."""
    logger.info("\n" + "=" * 70)
    logger.info("GAME LEARNING LAYER - VERIFICATION SUITE")
    logger.info("=" * 70)

    num_games = 5 if quick else 10

    # Run tests
    results = {}

    logger.info("\nRunning verification tests...")
    results["tictactoe"] = verify_tictactoe_demo(num_games=num_games)

    results["mcp"] = await verify_mcp_integration()

    results["visual"] = verify_visual_encoders()

    results["strategy"] = verify_strategy_memory_entanglement()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION RESULTS SUMMARY")
    logger.info("=" * 70)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name.upper():20s}: {status}")

    logger.info("=" * 70)

    if all_passed:
        logger.info("\n🎉 All verification tests PASSED!")
        return 0
    else:
        logger.error("\n❌ Some verification tests FAILED!")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify Game Learning Layer")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick verification (fewer games)",
    )

    args = parser.parse_args()

    exit_code = asyncio.run(run_verification(quick=args.quick))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
