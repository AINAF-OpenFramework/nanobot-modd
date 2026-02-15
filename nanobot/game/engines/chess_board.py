"""Board state manager for chess using python-chess library."""

from __future__ import annotations

from typing import Any

import chess
from loguru import logger


class BoardStateManager:
    """
    Manages chess board state using python-chess library.
    
    Provides FEN parsing, board vector representation, and move application.
    Integrates with the TanyalahD VTuber AI system for chess gameplay.
    """

    def __init__(self):
        """Initialize the board state manager with starting position."""
        self.board = chess.Board()
        logger.debug("chess_board.BoardStateManager initialized")

    def load_fen(self, fen_str: str) -> None:
        """
        Parse FEN string into internal board representation.
        
        Handles "startpos" as the standard starting position.
        
        Args:
            fen_str: FEN notation string or "startpos" for starting position
            
        Raises:
            ValueError: If FEN string is invalid
            
        Example:
            >>> manager = BoardStateManager()
            >>> manager.load_fen("startpos")
            >>> manager.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        """
        if fen_str == "startpos":
            self.board = chess.Board()
            logger.debug("chess_board.load_fen loaded starting position")
        else:
            try:
                self.board = chess.Board(fen_str)
                logger.debug(f"chess_board.load_fen loaded FEN: {fen_str[:50]}...")
            except ValueError as e:
                logger.error(f"chess_board.load_fen invalid FEN: {fen_str}")
                raise ValueError(f"Invalid FEN string: {e}")

    def get_board_vector(self) -> list[float]:
        """
        Return standardized board vector for ML evaluation.
        
        Returns a 768-element vector representing:
        - 64 squares × 12 piece types (6 pieces × 2 colors)
        
        Each square has a 12-element one-hot encoding:
        [wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK]
        
        Returns:
            768-element list representing the board state
            
        Example:
            >>> manager = BoardStateManager()
            >>> vector = manager.get_board_vector()
            >>> len(vector)
            768
        """
        vector = []
        
        # Piece type mapping to indices
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # Iterate through all squares (a1 to h8)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            square_encoding = [0.0] * 12
            
            if piece:
                piece_idx = piece_map[piece.piece_type]
                # White pieces: indices 0-5, Black pieces: indices 6-11
                idx = piece_idx if piece.color == chess.WHITE else piece_idx + 6
                square_encoding[idx] = 1.0
            
            vector.extend(square_encoding)
        
        logger.debug(f"chess_board.get_board_vector generated {len(vector)}-element vector")
        return vector

    def update_board(self, move: str) -> bool:
        """
        Apply a move in algebraic notation and update board state.
        
        Supports UCI format (e.g., "e2e4", "e7e8q" for promotion).
        
        Args:
            move: Move in UCI algebraic notation
            
        Returns:
            True if move was applied successfully, False otherwise
            
        Example:
            >>> manager = BoardStateManager()
            >>> manager.update_board("e2e4")
            True
            >>> manager.update_board("invalid")
            False
        """
        try:
            chess_move = chess.Move.from_uci(move)
            if chess_move in self.board.legal_moves:
                self.board.push(chess_move)
                logger.debug(f"chess_board.update_board applied move: {move}")
                return True
            else:
                logger.warning(f"chess_board.update_board illegal move: {move}")
                return False
        except (ValueError, chess.InvalidMoveError) as e:
            logger.error(f"chess_board.update_board error parsing move {move}: {e}")
            return False

    def get_current_turn(self) -> str:
        """
        Get the current player to move.
        
        Returns:
            "white" or "black"
        """
        return "white" if self.board.turn == chess.WHITE else "black"

    def get_fen(self) -> str:
        """
        Get the current board state as FEN string.
        
        Returns:
            FEN notation of current position
        """
        return self.board.fen()

    def get_state_dict(self) -> dict[str, Any]:
        """
        Get board state as dictionary compatible with ChessRules format.
        
        Returns:
            Dictionary with board, current_player, and game metadata
        """
        # Convert to 8x8 board representation
        board_2d = []
        for rank in range(7, -1, -1):  # Start from rank 8 down to rank 1
            row = []
            for file in range(8):  # Files a-h (0-7)
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                if piece:
                    color = 'w' if piece.color == chess.WHITE else 'b'
                    piece_symbol = piece.symbol().upper()
                    row.append(f"{color}{piece_symbol}")
                else:
                    row.append("")
            board_2d.append(row)
        
        return {
            "board": board_2d,
            "current_player": self.get_current_turn(),
            "move_count": self.board.fullmove_number,
            "fen": self.get_fen(),
            "halfmove_clock": self.board.halfmove_clock,
            "fullmove_number": self.board.fullmove_number,
        }

    def is_game_over(self) -> bool:
        """
        Check if the game is over (checkmate, stalemate, or draw).
        
        Returns:
            True if game is over, False otherwise
        """
        return self.board.is_game_over()

    def get_result(self) -> str:
        """
        Get the game result.
        
        Returns:
            "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw), or "*" (ongoing)
        """
        if not self.board.is_game_over():
            return "*"
        return self.board.result()
