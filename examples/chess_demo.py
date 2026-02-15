#!/usr/bin/env python
"""
Chess Game Demo - Complete chess gameplay with VTuber AI integration.

Demonstrates the full chess module including:
- Board state management
- Move generation and evaluation
- Q·K attention-style scoring
- VTuber personality commentary
- IAS/CER metrics tracking

Usage:
    python examples/chess_demo.py [--moves NUM] [--output FORMAT]
"""

import argparse
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


def play_chess_game(
    max_moves: int = 10,
    output_format: str = "text",
) -> dict:
    """
    Play a sample chess game with full AI integration.
    
    Args:
        max_moves: Maximum number of moves to play
        output_format: Output format for VTuber bridge
        
    Returns:
        Game statistics dictionary
    """
    logger.info("=" * 60)
    logger.info("Chess Game Demo - TanyalahD AI Chess Module")
    logger.info("=" * 60)
    
    # Initialize components
    board_manager = BoardStateManager()
    move_generator = MoveGenerator()
    move_executor = MoveExecutor(temperature=0.2)
    
    # Initialize strategy memory
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_store = MemoryStore(Path(tmpdir))
        strategy_memory = StrategyMemory(memory_store)
        
        move_evaluator = MoveEvaluator(strategy_memory=strategy_memory)
        soul_layer = SoulLayerIntegration(personality="TanyalahD")
        vtuber = VTuberOutput(output_format=output_format)
        metrics = MetricsTracker()
        
        # Start game
        board_manager.load_fen("startpos")
        board = chess.Board()
        
        game_id = "demo_game_1"
        move_count = 0
        previous_score = 0.0
        
        logger.info("\nStarting position:")
        logger.info(board)
        
        vtuber.send_game_event(
            event_type="start",
            message="Let's play some chess! I'm ready~ ♟️",
            emotion="excited"
        )
        
        # Game loop
        while move_count < max_moves and not board.is_game_over():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Move {move_count + 1} - {board_manager.get_current_turn().title()} to move")
            logger.info(f"{'=' * 60}")
            
            # Generate legal moves
            legal_moves = move_generator.generate_legal_moves_from_board(board)
            logger.info(f"Legal moves: {len(legal_moves)}")
            
            if not legal_moves:
                break
            
            # Encode moves
            move_vectors = move_generator.encode_moves_as_vectors(legal_moves)
            board_vector = board_manager.get_board_vector()
            
            # Evaluate moves
            scores = move_evaluator.score_moves(
                board_vector,
                move_vectors,
                board=board,
                moves=legal_moves,
            )
            
            # Rank moves
            ranked_indices = move_evaluator.rank_moves_by_IAS(scores)
            
            # Select best move
            best_idx = move_executor.select_best_move(
                scores,
                moves=legal_moves,
                use_temperature=True,
            )
            
            selected_move = legal_moves[best_idx]
            selected_score = scores[best_idx]
            
            # Compute metrics
            ias, cer = metrics.compute_IAS_CER(
                selected_move,
                scores,
                selected_idx=best_idx,
            )
            
            # Generate commentary
            comment = soul_layer.generate_comment(
                selected_move,
                board_vector=board_vector,
                board=board,
                score=selected_score,
                previous_score=previous_score,
            )
            
            # Output move with commentary
            logger.info(f"\nSelected move: {selected_move}")
            logger.info(f"Score: {selected_score:.2f}")
            logger.info(f"IAS: {ias:.3f} | CER: {cer:.3f}")
            
            vtuber.send_move_and_comment(
                move=selected_move,
                comment=comment,
                score=selected_score,
                emotion=soul_layer._classify_move(
                    selected_move,
                    board,
                    selected_score,
                    previous_score,
                ),
            )
            
            # Record in strategy memory
            strategy_memory.record_move(
                game_id=game_id,
                move=selected_move,
                state=board_manager.get_state_dict(),
            )
            
            # Log metrics
            metrics.log(
                IAS=ias,
                CER=cer,
                move=selected_move,
                game_id=game_id,
            )
            
            # Execute move
            board_manager.update_board(selected_move)
            board.push(chess.Move.from_uci(selected_move))
            
            previous_score = selected_score
            move_count += 1
            
            # Display board
            logger.info("\n" + str(board))
            
            # Check for game end
            if board.is_checkmate():
                winner = "Black" if board.turn == chess.WHITE else "White"
                vtuber.send_game_event(
                    event_type="checkmate",
                    message=f"Checkmate! {winner} wins! 🎉",
                    emotion="excited",
                    metadata={"winner": winner, "moves": move_count},
                )
                break
            elif board.is_stalemate():
                vtuber.send_game_event(
                    event_type="draw",
                    message="Stalemate! It's a draw~ 🤝",
                    emotion="neutral",
                )
                break
            elif board.is_check():
                vtuber.send_game_event(
                    event_type="check",
                    message="Check! The king is in danger! ⚡",
                    emotion="excited",
                )
        
        # Game summary
        logger.info("\n" + "=" * 60)
        logger.info("Game Complete")
        logger.info("=" * 60)
        
        result = board.result()
        summary = soul_layer.get_game_summary(result, move_count)
        
        logger.info(f"\nResult: {result}")
        logger.info(f"Moves played: {move_count}")
        logger.info(f"\n{summary}")
        
        # Metrics summary
        stats = metrics.get_statistics()
        logger.info("\nMetrics Summary:")
        logger.info(f"  Total moves tracked: {stats['count']}")
        logger.info(f"  IAS: mean={stats['IAS']['mean']:.3f}, "
                   f"std={stats['IAS']['std']:.3f}")
        logger.info(f"  CER: mean={stats['CER']['mean']:.3f}, "
                   f"std={stats['CER']['std']:.3f}")
        
        return {
            "moves": move_count,
            "result": result,
            "metrics": stats,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Chess Game Demo with VTuber AI"
    )
    parser.add_argument(
        "--moves",
        type=int,
        default=10,
        help="Maximum number of moves to play (default: 10)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "rich"],
        default="text",
        help="Output format (default: text)",
    )
    
    args = parser.parse_args()
    
    try:
        stats = play_chess_game(
            max_moves=args.moves,
            output_format=args.output,
        )
        
        logger.info("\n✅ Chess demo completed successfully!")
        logger.info(f"Final stats: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Chess demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
