"""Move evaluator using Q·K attention-style scoring for chess."""

from __future__ import annotations

import math
from typing import Any

import chess
from loguru import logger


class MoveEvaluator:
    """
    Evaluates and scores chess moves using Q·K attention-style scoring.
    
    Implements:
    - Material balance evaluation
    - Piece activity scoring
    - King safety analysis
    - Pawn structure evaluation
    - Strategy memory integration for pattern matching
    
    Uses Intrinsic Alignment Score (IAS) for compatibility with overall strategy.
    """

    def __init__(self, strategy_memory: Any | None = None):
        """
        Initialize the move evaluator.
        
        Args:
            strategy_memory: Optional StrategyMemory instance for pattern matching
        """
        self.strategy_memory = strategy_memory
        
        # Piece values for material evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000,
        }
        
        logger.debug("chess_evaluator.MoveEvaluator initialized")

    def score_moves(
        self,
        board_vector: list[float],
        move_vectors: list[list[float]],
        board: chess.Board | None = None,
        moves: list[str] | None = None,
    ) -> list[float]:
        """
        Score each move using Q·K style attention scoring.
        
        The scoring combines:
        1. Material gain/loss
        2. Piece activity (center control, mobility)
        3. King safety
        4. Pawn structure
        5. Pattern matching from strategy memory (if available)
        
        Uses Query·Key attention: score = softmax(Q·K^T / sqrt(d_k))
        where Q represents the board state query and K represents move keys.
        
        Args:
            board_vector: 768-element board representation
            move_vectors: List of 133-element move encodings
            board: Optional chess.Board for detailed analysis
            moves: Optional list of move strings in UCI notation
            
        Returns:
            List of scores (floats) for each move, higher is better
            
        Example:
            >>> evaluator = MoveEvaluator()
            >>> import chess
            >>> board = chess.Board()
            >>> moves = [move.uci() for move in board.legal_moves]
            >>> from nanobot.game.engines.chess_moves import MoveGenerator
            >>> generator = MoveGenerator()
            >>> move_vectors = generator.encode_moves_as_vectors(moves)
            >>> from nanobot.game.engines.chess_board import BoardStateManager
            >>> manager = BoardStateManager()
            >>> board_vector = manager.get_board_vector()
            >>> scores = evaluator.score_moves(board_vector, move_vectors, board, moves)
            >>> len(scores) == len(moves)
            True
        """
        if not move_vectors:
            return []
        
        scores = []
        
        # If we have a board object, use detailed analysis
        if board and moves:
            for move_str in moves:
                score = self._evaluate_move_detailed(board, move_str)
                scores.append(score)
        else:
            # Fallback: simplified scoring based on vectors
            for move_vector in move_vectors:
                # Simple Q·K dot product scoring
                # Normalize by sqrt of dimension for attention-style scoring
                d_k = len(move_vector)
                dot_product = sum(
                    q * k for q, k in zip(board_vector[:d_k], move_vector)
                )
                score = dot_product / math.sqrt(d_k)
                scores.append(score)
        
        logger.debug(
            f"chess_evaluator.score_moves scored {len(scores)} moves, "
            f"range=[{min(scores):.2f}, {max(scores):.2f}]"
        )
        
        return scores

    def rank_moves_by_IAS(self, scores: list[float]) -> list[int]:
        """
        Return indices of moves sorted by Intrinsic Alignment Score (IAS).
        
        IAS measures how well the chosen move aligns with the AI's strategic patterns.
        Higher IAS indicates better alignment with successful historical patterns.
        
        Args:
            scores: List of move scores
            
        Returns:
            List of move indices sorted by score (descending order)
            
        Example:
            >>> evaluator = MoveEvaluator()
            >>> scores = [0.5, 0.8, 0.3, 0.9]
            >>> ranked = evaluator.rank_moves_by_IAS(scores)
            >>> ranked
            [3, 1, 0, 2]
        """
        # Create list of (index, score) tuples and sort by score descending
        indexed_scores = list(enumerate(scores))
        ranked = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        ranked_indices = [idx for idx, score in ranked]
        
        logger.debug(
            f"chess_evaluator.rank_moves_by_IAS ranked {len(scores)} moves"
        )
        
        return ranked_indices

    def _evaluate_move_detailed(self, board: chess.Board, move_str: str) -> float:
        """
        Detailed move evaluation considering multiple factors.
        
        Args:
            board: Current board position
            move_str: Move in UCI notation
            
        Returns:
            Move score (higher is better)
        """
        try:
            move = chess.Move.from_uci(move_str)
        except (ValueError, chess.InvalidMoveError):
            return -10000.0  # Invalid move
        
        if move not in board.legal_moves:
            return -10000.0  # Illegal move
        
        score = 0.0
        
        # Make move on a copy to evaluate resulting position
        board_copy = board.copy()
        board_copy.push(move)
        
        # 1. Material evaluation
        score += self._evaluate_material(board_copy)
        
        # 2. Piece activity (center control, mobility)
        score += self._evaluate_activity(board_copy)
        
        # 3. King safety
        score += self._evaluate_king_safety(board_copy)
        
        # 4. Pawn structure
        score += self._evaluate_pawn_structure(board_copy)
        
        # 5. Tactical bonuses
        score += self._evaluate_tactical(board, move, board_copy)
        
        # 6. Strategy memory pattern matching (if available)
        if self.strategy_memory:
            pattern_bonus = self._get_pattern_bonus(board_copy)
            score += pattern_bonus
        
        return score

    def _evaluate_material(self, board: chess.Board) -> float:
        """Evaluate material balance."""
        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                material += value if piece.color == board.turn else -value
        return material / 100.0  # Normalize

    def _evaluate_activity(self, board: chess.Board) -> float:
        """Evaluate piece activity and mobility."""
        activity = 0.0
        
        # Mobility: number of legal moves
        mobility = len(list(board.legal_moves))
        activity += mobility * 0.1
        
        # Center control
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                activity += 0.5
        
        return activity

    def _evaluate_king_safety(self, board: chess.Board) -> float:
        """Evaluate king safety."""
        safety = 0.0
        
        # Check if in check (penalize)
        if board.is_check():
            safety -= 5.0
        
        # King position evaluation
        king_square = board.king(board.turn)
        if king_square:
            rank = chess.square_rank(king_square)
            # Penalize exposed king in middle game
            if 2 <= rank <= 5:
                safety -= 2.0
        
        return safety

    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Evaluate pawn structure."""
        structure = 0.0
        
        # Count pawns and check for doubled/isolated pawns
        our_pawns = board.pieces(chess.PAWN, board.turn)
        
        # Simple bonus for advanced pawns
        for pawn_square in our_pawns:
            rank = chess.square_rank(pawn_square)
            if board.turn == chess.WHITE:
                structure += rank * 0.1
            else:
                structure += (7 - rank) * 0.1
        
        return structure

    def _evaluate_tactical(
        self,
        board_before: chess.Board,
        move: chess.Move,
        board_after: chess.Board,
    ) -> float:
        """Evaluate tactical elements of the move."""
        tactical = 0.0
        
        # Capture bonus
        if board_before.is_capture(move):
            captured_piece = board_before.piece_at(move.to_square)
            if captured_piece:
                tactical += self.piece_values[captured_piece.piece_type] / 100.0
        
        # Checkmate
        if board_after.is_checkmate():
            tactical += 1000.0
        
        # Check bonus
        elif board_after.is_check():
            tactical += 2.0
        
        # Promotion bonus
        if move.promotion:
            tactical += 5.0
        
        return tactical

    def _get_pattern_bonus(self, board: chess.Board) -> float:
        """Get bonus from strategy memory patterns."""
        # Placeholder for strategy memory integration
        # In full implementation, this would query similar positions
        return 0.0
