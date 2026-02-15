"""Move executor for selecting and executing chess moves."""

from __future__ import annotations

import random

import chess
from loguru import logger


class MoveExecutor:
    """
    Selects and executes chess moves based on evaluation scores.
    
    Provides options for deterministic (best move) or stochastic (temperature-based)
    move selection for variety in gameplay.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Initialize the move executor.
        
        Args:
            temperature: Randomness in move selection (0.0 = deterministic, higher = more random)
        """
        self.temperature = temperature
        logger.debug(f"chess_executor.MoveExecutor initialized with temperature={temperature}")

    def select_best_move(
        self,
        scores: list[float],
        moves: list[str] | None = None,
        use_temperature: bool = True,
    ) -> int:
        """
        Select the best move index based on scores.
        
        Args:
            scores: List of move scores
            moves: Optional list of move strings for logging
            use_temperature: If True, use temperature-based sampling for variety
            
        Returns:
            Index of selected move
            
        Example:
            >>> executor = MoveExecutor(temperature=0.1)
            >>> scores = [0.5, 0.8, 0.3, 0.9]
            >>> moves = ["e2e4", "d2d4", "g1f3", "e2e3"]
            >>> best_idx = executor.select_best_move(scores, moves)
            >>> best_idx == 3  # Highest score (0.9)
            True
        """
        if not scores:
            logger.warning("chess_executor.select_best_move: no scores provided")
            return -1
        
        if use_temperature and self.temperature > 0.0:
            # Temperature-based sampling for variety
            selected_idx = self._sample_with_temperature(scores)
        else:
            # Deterministic: select highest score
            selected_idx = scores.index(max(scores))
        
        if moves and 0 <= selected_idx < len(moves):
            logger.debug(
                f"chess_executor.select_best_move selected move {moves[selected_idx]} "
                f"(score={scores[selected_idx]:.3f})"
            )
        else:
            logger.debug(
                f"chess_executor.select_best_move selected index {selected_idx} "
                f"(score={scores[selected_idx]:.3f})"
            )
        
        return selected_idx

    def execute_move(
        self,
        board_vector: list[float],
        move: str,
        board: chess.Board | None = None,
    ) -> tuple[list[float] | None, bool]:
        """
        Apply the move and return the updated board state.
        
        Args:
            board_vector: Current board vector (768 elements)
            move: Move in UCI notation
            board: Optional chess.Board object to update
            
        Returns:
            Tuple of (updated_board_vector or None, success)
            
        Example:
            >>> executor = MoveExecutor()
            >>> import chess
            >>> board = chess.Board()
            >>> from nanobot.game.engines.chess_board import BoardStateManager
            >>> manager = BoardStateManager()
            >>> vector = manager.get_board_vector()
            >>> new_vector, success = executor.execute_move(vector, "e2e4", board)
            >>> success
            True
        """
        if not board:
            logger.warning(
                "chess_executor.execute_move: no board provided, "
                "cannot update board state"
            )
            return None, False
        
        try:
            chess_move = chess.Move.from_uci(move)
            
            if chess_move in board.legal_moves:
                board.push(chess_move)
                
                # Generate new board vector
                new_vector = self._board_to_vector(board)
                
                logger.debug(f"chess_executor.execute_move successfully applied {move}")
                return new_vector, True
            else:
                logger.warning(f"chess_executor.execute_move illegal move: {move}")
                return None, False
                
        except (ValueError, chess.InvalidMoveError) as e:
            logger.error(f"chess_executor.execute_move error: {e}")
            return None, False

    def _sample_with_temperature(self, scores: list[float]) -> int:
        """
        Sample move index using temperature-based softmax.
        
        Args:
            scores: List of move scores
            
        Returns:
            Sampled move index
        """
        if self.temperature == 0.0:
            return scores.index(max(scores))
        
        # Apply temperature scaling
        import math
        
        # Avoid overflow by subtracting max score
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / self.temperature) for s in scores]
        total = sum(exp_scores)
        
        if total == 0:
            # Fallback to uniform sampling
            return random.randint(0, len(scores) - 1)
        
        # Compute probabilities
        probabilities = [exp_s / total for exp_s in exp_scores]
        
        # Sample
        rand_val = random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probabilities):
            cumulative += prob
            if rand_val <= cumulative:
                return idx
        
        # Fallback (shouldn't reach here)
        return len(scores) - 1

    def _board_to_vector(self, board: chess.Board) -> list[float]:
        """
        Convert chess.Board to 768-element vector.
        
        Args:
            board: chess.Board object
            
        Returns:
            768-element board vector
        """
        vector = []
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            square_encoding = [0.0] * 12
            
            if piece:
                piece_idx = piece_map[piece.piece_type]
                idx = piece_idx if piece.color == chess.WHITE else piece_idx + 6
                square_encoding[idx] = 1.0
            
            vector.extend(square_encoding)
        
        return vector
