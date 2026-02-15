#!/usr/bin/env python
"""
Chess Module Verification Script

Tests all chess module components to ensure they work correctly:
- BoardStateManager: FEN parsing, board vectors, move application
- MoveGenerator: Legal move generation, move encoding
- MoveEvaluator: Move scoring, IAS ranking
- MoveExecutor: Move selection and execution
- StrategyMemory: Move recording and retrieval
- SoulLayerIntegration: Commentary generation
- VTuberOutput: Output formatting
- MetricsTracker: IAS/CER computation and logging

Usage:
    python scripts/verify_chess_module.py [--verbose]
"""

import argparse
import sys
import tempfile
from pathlib import Path

import chess
from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.game.engines.chess_board import BoardStateManager
from nanobot.game.engines.chess_evaluator import MoveEvaluator
from nanobot.game.engines.chess_executor import MoveExecutor
from nanobot.game.engines.chess_moves import MoveGenerator
from nanobot.game.metrics.game_metrics import MetricsTracker
from nanobot.game.strategy_memory import StrategyMemory
from nanobot.game.vtuber.soul_layer import SoulLayerIntegration
from nanobot.game.vtuber.vtuber_bridge import VTuberOutput


def test_board_state_manager() -> bool:
    """Test BoardStateManager functionality."""
    logger.info("\n=== Testing BoardStateManager ===")
    
    try:
        manager = BoardStateManager()
        
        # Test load_fen
        manager.load_fen("startpos")
        assert manager.get_current_turn() == "white"
        logger.info("✓ load_fen('startpos') works")
        
        # Test FEN string
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        manager.load_fen(fen)
        logger.info("✓ load_fen(FEN string) works")
        
        # Test get_board_vector
        vector = manager.get_board_vector()
        assert len(vector) == 768  # 64 squares × 12 piece types
        logger.info(f"✓ get_board_vector() returns {len(vector)} elements")
        
        # Test update_board
        success = manager.update_board("e2e4")
        assert success
        assert manager.get_current_turn() == "black"
        logger.info("✓ update_board('e2e4') works")
        
        # Test get_state_dict
        state = manager.get_state_dict()
        assert "board" in state
        assert "current_player" in state
        logger.info("✓ get_state_dict() works")
        
        logger.info("✅ BoardStateManager: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ BoardStateManager test failed: {e}")
        return False


def test_move_generator() -> bool:
    """Test MoveGenerator functionality."""
    logger.info("\n=== Testing MoveGenerator ===")
    
    try:
        generator = MoveGenerator()
        board = chess.Board()
        
        # Test generate_legal_moves_from_board
        moves = generator.generate_legal_moves_from_board(board)
        assert len(moves) == 20  # Starting position has 20 legal moves
        logger.info(f"✓ generate_legal_moves_from_board() returns {len(moves)} moves")
        
        # Test encode_moves_as_vectors
        vectors = generator.encode_moves_as_vectors(moves)
        assert len(vectors) == len(moves)
        assert len(vectors[0]) == 133  # 64 + 64 + 5
        logger.info(f"✓ encode_moves_as_vectors() returns {len(vectors)} vectors")
        
        # Test decode_move_vector
        decoded = generator.decode_move_vector(vectors[0])
        assert len(decoded) > 0
        logger.info(f"✓ decode_move_vector() works: {decoded}")
        
        logger.info("✅ MoveGenerator: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ MoveGenerator test failed: {e}")
        return False


def test_move_evaluator() -> bool:
    """Test MoveEvaluator functionality."""
    logger.info("\n=== Testing MoveEvaluator ===")
    
    try:
        evaluator = MoveEvaluator()
        board = chess.Board()
        
        generator = MoveGenerator()
        moves = generator.generate_legal_moves_from_board(board)
        move_vectors = generator.encode_moves_as_vectors(moves)
        
        manager = BoardStateManager()
        board_vector = manager.get_board_vector()
        
        # Test score_moves
        scores = evaluator.score_moves(
            board_vector,
            move_vectors,
            board=board,
            moves=moves,
        )
        assert len(scores) == len(moves)
        logger.info(f"✓ score_moves() returns {len(scores)} scores")
        logger.info(f"  Score range: [{min(scores):.2f}, {max(scores):.2f}]")
        
        # Test rank_moves_by_IAS
        ranked = evaluator.rank_moves_by_IAS(scores)
        assert len(ranked) == len(scores)
        assert ranked[0] == scores.index(max(scores))
        logger.info(f"✓ rank_moves_by_IAS() works, top move index: {ranked[0]}")
        
        logger.info("✅ MoveEvaluator: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ MoveEvaluator test failed: {e}")
        return False


def test_move_executor() -> bool:
    """Test MoveExecutor functionality."""
    logger.info("\n=== Testing MoveExecutor ===")
    
    try:
        executor = MoveExecutor(temperature=0.1)
        
        # Test select_best_move
        scores = [0.5, 0.8, 0.3, 0.9]
        moves = ["e2e4", "d2d4", "g1f3", "c2c4"]
        best_idx = executor.select_best_move(scores, moves, use_temperature=False)
        assert best_idx == 3  # Highest score
        logger.info(f"✓ select_best_move() works, selected index: {best_idx}")
        
        # Test execute_move
        board = chess.Board()
        manager = BoardStateManager()
        vector = manager.get_board_vector()
        
        new_vector, success = executor.execute_move(vector, "e2e4", board)
        assert success
        assert new_vector is not None
        assert len(new_vector) == 768
        logger.info("✓ execute_move() works")
        
        logger.info("✅ MoveExecutor: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ MoveExecutor test failed: {e}")
        return False


def test_strategy_memory() -> bool:
    """Test StrategyMemory functionality."""
    logger.info("\n=== Testing StrategyMemory ===")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = MemoryStore(Path(tmpdir))
            strategy_memory = StrategyMemory(memory_store)
            
            # Test record_move
            success = strategy_memory.record_move(
                game_id="test_game",
                move="e2e4",
                state={"board": [], "current_player": "white"}
            )
            assert success
            logger.info("✓ record_move() works")
            
            # Test retrieve_memory
            nodes = strategy_memory.retrieve_memory(
                state={"board": [], "current_player": "white"},
                game_type="chess",
                k=5,
            )
            assert isinstance(nodes, list)
            logger.info(f"✓ retrieve_memory() works, found {len(nodes)} nodes")
        
        logger.info("✅ StrategyMemory: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ StrategyMemory test failed: {e}")
        return False


def test_soul_layer() -> bool:
    """Test SoulLayerIntegration functionality."""
    logger.info("\n=== Testing SoulLayerIntegration ===")
    
    try:
        soul = SoulLayerIntegration(personality="TanyalahD")
        
        # Test generate_comment
        comment = soul.generate_comment(
            move="e2e4",
            score=0.5,
        )
        assert len(comment) > 0
        logger.info(f"✓ generate_comment() works: '{comment[:50]}...'")
        
        # Test get_game_summary
        summary = soul.get_game_summary(
            result="1-0",
            move_count=25,
            highlights=["Brilliant sacrifice!", "Amazing endgame"],
        )
        assert len(summary) > 0
        logger.info("✓ get_game_summary() works")
        
        logger.info("✅ SoulLayerIntegration: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ SoulLayerIntegration test failed: {e}")
        return False


def test_vtuber_bridge() -> bool:
    """Test VTuberOutput functionality."""
    logger.info("\n=== Testing VTuberOutput ===")
    
    try:
        vtuber = VTuberOutput(output_format="json", enable_tts=True)
        
        # Test send_move_and_comment
        output = vtuber.send_move_and_comment(
            move="e2e4",
            comment="Good opening!",
            score=0.5,
            emotion="happy",
        )
        assert "move" in output
        assert "comment" in output
        assert "tts_text" in output
        logger.info("✓ send_move_and_comment() works")
        
        # Test send_game_event
        event = vtuber.send_game_event(
            event_type="checkmate",
            message="Checkmate!",
            emotion="excited",
        )
        assert "type" in event
        assert "message" in event
        logger.info("✓ send_game_event() works")
        
        logger.info("✅ VTuberOutput: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ VTuberOutput test failed: {e}")
        return False


def test_metrics_tracker() -> bool:
    """Test MetricsTracker functionality."""
    logger.info("\n=== Testing MetricsTracker ===")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "metrics.log"
            tracker = MetricsTracker(log_file=str(log_file))
            
            # Test compute_IAS_CER
            scores = [0.5, 0.8, 0.3, 0.9]
            ias, cer = tracker.compute_IAS_CER(
                move="e2e4",
                scores=scores,
                selected_idx=3,
            )
            assert 0.0 <= ias <= 1.0
            assert 0.0 <= cer <= 1.0
            logger.info(f"✓ compute_IAS_CER() works: IAS={ias:.3f}, CER={cer:.3f}")
            
            # Test log
            tracker.log(IAS=ias, CER=cer, move="e2e4", game_id="test")
            assert len(tracker.metrics_history) == 1
            logger.info("✓ log() works")
            
            # Test get_statistics
            stats = tracker.get_statistics()
            assert stats["count"] == 1
            assert "IAS" in stats
            assert "CER" in stats
            logger.info(f"✓ get_statistics() works: {stats}")
            
            # Verify log file was created
            assert log_file.exists()
            logger.info("✓ Log file created")
        
        logger.info("✅ MetricsTracker: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ MetricsTracker test failed: {e}")
        return False


def run_all_tests(verbose: bool = False) -> int:
    """
    Run all verification tests.
    
    Args:
        verbose: Enable verbose logging
        
    Returns:
        0 if all tests passed, 1 otherwise
    """
    if not verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    
    logger.info("=" * 70)
    logger.info("Chess Module Verification")
    logger.info("=" * 70)
    
    tests = [
        ("BoardStateManager", test_board_state_manager),
        ("MoveGenerator", test_move_generator),
        ("MoveEvaluator", test_move_evaluator),
        ("MoveExecutor", test_move_executor),
        ("StrategyMemory", test_strategy_memory),
        ("SoulLayerIntegration", test_soul_layer),
        ("VTuberOutput", test_vtuber_bridge),
        ("MetricsTracker", test_metrics_tracker),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"❌ {name} test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Verification Summary")
    logger.info("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        logger.info("\n🎉 All chess module components verified successfully!")
        return 0
    else:
        logger.error(f"\n❌ {total_count - passed_count} test(s) failed")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify chess module implementation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    exit_code = run_all_tests(verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
