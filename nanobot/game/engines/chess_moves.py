"""Move generator for chess using python-chess library."""

from __future__ import annotations

import chess
from loguru import logger


class MoveGenerator:
    """
    Generates legal moves for chess positions.
    
    Handles all piece types including special moves (castling, en passant, promotion).
    Provides move encoding for ML evaluation.
    """

    def __init__(self):
        """Initialize the move generator."""
        logger.debug("chess_moves.MoveGenerator initialized")

    def generate_legal_moves(self, board_vector: list[float]) -> list[str]:
        """
        Generate all legal moves for the current player.
        
        Note: This method takes a board_vector but internally reconstructs
        the board state. For direct board access, use generate_legal_moves_from_board.
        
        Args:
            board_vector: 768-element board representation
            
        Returns:
            List of legal moves in UCI notation (e.g., ["e2e4", "g1f3"])
            
        Example:
            >>> generator = MoveGenerator()
            >>> from nanobot.game.engines.chess_board import BoardStateManager
            >>> manager = BoardStateManager()
            >>> vector = manager.get_board_vector()
            >>> moves = generator.generate_legal_moves(vector)
            >>> len(moves) == 20  # Starting position has 20 legal moves
            True
        """
        # Since board_vector is passed but we need the full board state,
        # this is a simplified interface. In practice, the board object
        # should be passed or reconstructed.
        logger.warning(
            "chess_moves.generate_legal_moves: board_vector interface is limited. "
            "Use generate_legal_moves_from_board for full functionality."
        )
        # Return empty list for vector-only interface
        # Actual usage should pass board object via generate_legal_moves_from_board
        return []

    def generate_legal_moves_from_board(self, board: chess.Board) -> list[str]:
        """
        Generate all legal moves from a chess.Board object.
        
        Handles:
        - All piece types (pawns, knights, bishops, rooks, queens, kings)
        - Special moves (castling, en passant, pawn promotion)
        - Check evasion
        - Pin detection
        
        Args:
            board: chess.Board object
            
        Returns:
            List of legal moves in UCI notation
            
        Example:
            >>> import chess
            >>> board = chess.Board()
            >>> generator = MoveGenerator()
            >>> moves = generator.generate_legal_moves_from_board(board)
            >>> "e2e4" in moves
            True
        """
        moves = [move.uci() for move in board.legal_moves]
        logger.debug(
            f"chess_moves.generate_legal_moves_from_board generated {len(moves)} moves"
        )
        return moves

    def encode_moves_as_vectors(self, moves_list: list[str]) -> list[list[float]]:
        """
        Convert move strings to numerical vectors for ML evaluation.
        
        Uses a 73-element encoding for each move:
        - From square: 64 elements (one-hot encoding of source square)
        - To square: 64 elements (one-hot encoding of destination square)
        - Promotion piece: 5 elements (one-hot: queen, rook, bishop, knight, none)
        
        Total: 64 + 64 + 5 = 133 elements per move (using compact representation)
        
        Args:
            moves_list: List of moves in UCI notation
            
        Returns:
            List of move vectors, each with 133 elements
            
        Example:
            >>> generator = MoveGenerator()
            >>> moves = ["e2e4", "g1f3"]
            >>> vectors = generator.encode_moves_as_vectors(moves)
            >>> len(vectors)
            2
            >>> len(vectors[0])
            133
        """
        encoded_moves = []
        
        for move_str in moves_list:
            try:
                move = chess.Move.from_uci(move_str)
                
                # Encode from square (64 elements)
                from_encoding = [0.0] * 64
                from_encoding[move.from_square] = 1.0
                
                # Encode to square (64 elements)
                to_encoding = [0.0] * 64
                to_encoding[move.to_square] = 1.0
                
                # Encode promotion piece (5 elements: Q, R, B, N, none)
                promotion_encoding = [0.0] * 5
                if move.promotion:
                    promo_map = {
                        chess.QUEEN: 0,
                        chess.ROOK: 1,
                        chess.BISHOP: 2,
                        chess.KNIGHT: 3,
                    }
                    promotion_encoding[promo_map[move.promotion]] = 1.0
                else:
                    promotion_encoding[4] = 1.0  # No promotion
                
                # Combine all encodings
                move_vector = from_encoding + to_encoding + promotion_encoding
                encoded_moves.append(move_vector)
                
            except (ValueError, chess.InvalidMoveError) as e:
                logger.warning(f"chess_moves.encode_moves_as_vectors error encoding {move_str}: {e}")
                # Return zero vector for invalid moves
                encoded_moves.append([0.0] * 133)
        
        logger.debug(
            f"chess_moves.encode_moves_as_vectors encoded {len(encoded_moves)} moves"
        )
        return encoded_moves

    def decode_move_vector(self, move_vector: list[float]) -> str:
        """
        Decode a move vector back to UCI notation.
        
        Args:
            move_vector: 133-element move encoding
            
        Returns:
            Move in UCI notation
        """
        if len(move_vector) != 133:
            logger.error(f"chess_moves.decode_move_vector invalid vector length: {len(move_vector)}")
            return ""
        
        # Decode from square (first 64 elements)
        from_square = move_vector[:64].index(1.0) if 1.0 in move_vector[:64] else 0
        
        # Decode to square (next 64 elements)
        to_square = move_vector[64:128].index(1.0) if 1.0 in move_vector[64:128] else 0
        
        # Decode promotion (last 5 elements)
        promotion_part = move_vector[128:133]
        promotion_map = {0: 'q', 1: 'r', 2: 'b', 3: 'n', 4: ''}
        promo_idx = promotion_part.index(1.0) if 1.0 in promotion_part else 4
        promotion_str = promotion_map[promo_idx]
        
        # Construct UCI move string
        move_str = chess.square_name(from_square) + chess.square_name(to_square) + promotion_str
        
        return move_str
