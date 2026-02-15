"""TicTacToe game rules implementation."""

from __future__ import annotations

import copy
from typing import Any

from loguru import logger

from nanobot.game.rules.base import BoardPosition


class TicTacToeRules:
    """
    TicTacToe game rules implementation.

    Implements the GameRules protocol for a standard 3x3 TicTacToe game.
    Players alternate placing 'X' and 'O' markers on the board.
    First player to get 3 in a row (horizontal, vertical, or diagonal) wins.

    State format:
        {
            "board": [["X", "O", ""], ["", "X", ""], ["", "", "O"]],
            "current_player": "X",
            "move_count": 5
        }

    Move format: "r0c0", "r1c2", etc. (row and column indices)
    """

    def __init__(self) -> None:
        """Initialize TicTacToe rules."""
        self.board_size = 3
        self.players = ["X", "O"]
        logger.debug("game.rules.tictactoe.init")

    def get_legal_moves(self, state: dict[str, Any]) -> list[str]:
        """
        Get all legal moves (empty cells) for the current state.

        Args:
            state: Current game state

        Returns:
            List of legal move strings (e.g., ["r0c0", "r1c2"])
        """
        board = state.get("board", [])
        legal_moves = []

        for row in range(self.board_size):
            for col in range(self.board_size):
                if row < len(board) and col < len(board[row]):
                    if board[row][col] == "":
                        legal_moves.append(f"r{row}c{col}")

        logger.debug(
            f"game.rules.tictactoe.get_legal_moves "
            f"count={len(legal_moves)} moves={legal_moves}"
        )
        return legal_moves

    def apply_move(self, state: dict[str, Any], move: str) -> dict[str, Any]:
        """
        Apply a move to the state and return the new state.

        Args:
            state: Current game state
            move: Move string (e.g., "r0c1")

        Returns:
            New game state after applying the move

        Raises:
            ValueError: If the move is invalid
        """
        # Deep copy the state
        new_state = copy.deepcopy(state)

        # Parse move
        try:
            pos = BoardPosition.from_string(move)
        except ValueError as e:
            logger.error(f"game.rules.tictactoe.apply_move invalid_move={move} error={e}")
            raise ValueError(f"Invalid move format: {move}") from e

        # Validate position is in bounds
        if not (0 <= pos.row < self.board_size and 0 <= pos.col < self.board_size):
            raise ValueError(f"Move out of bounds: {move}")

        # Get current board
        board = new_state.get("board", [])
        if not board:
            raise ValueError("Board not found in state")

        # Validate cell is empty
        if board[pos.row][pos.col] != "":
            raise ValueError(f"Cell already occupied: {move}")

        # Apply move
        current_player = new_state.get("current_player", "X")
        board[pos.row][pos.col] = current_player

        # Update state
        new_state["board"] = board
        new_state["move_count"] = new_state.get("move_count", 0) + 1

        # Switch player
        next_player = self.get_next_player(new_state)
        new_state["current_player"] = next_player

        logger.debug(
            f"game.rules.tictactoe.apply_move "
            f"move={move} player={current_player} move_count={new_state['move_count']}"
        )

        return new_state

    def _is_board_full(self, board: list[list[str]]) -> bool:
        """Check if the board is completely filled."""
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row][col] == "":
                    return False
        return True

    def check_win_conditions(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Check if the game has ended and determine the winner.

        Args:
            state: Current game state

        Returns:
            Dictionary with:
                - game_over: bool
                - winner: str or None ("X", "O", or None for draw)
                - status: str describing the result
        """
        board = state.get("board", [])

        # Check rows
        for row in range(self.board_size):
            if self._check_line([board[row][col] for col in range(self.board_size)]):
                winner = board[row][0]
                logger.info(
                    f"game.rules.tictactoe.check_win_conditions "
                    f"winner={winner} condition=row{row}"
                )
                return {
                    "game_over": True,
                    "winner": winner,
                    "status": f"{winner} wins (row {row})",
                }

        # Check columns
        for col in range(self.board_size):
            if self._check_line([board[row][col] for row in range(self.board_size)]):
                winner = board[0][col]
                logger.info(
                    f"game.rules.tictactoe.check_win_conditions "
                    f"winner={winner} condition=col{col}"
                )
                return {
                    "game_over": True,
                    "winner": winner,
                    "status": f"{winner} wins (column {col})",
                }

        # Check diagonals
        # Top-left to bottom-right
        if self._check_line([board[i][i] for i in range(self.board_size)]):
            winner = board[0][0]
            logger.info(
                f"game.rules.tictactoe.check_win_conditions "
                f"winner={winner} condition=diagonal_main"
            )
            return {
                "game_over": True,
                "winner": winner,
                "status": f"{winner} wins (main diagonal)",
            }

        # Top-right to bottom-left
        if self._check_line(
            [board[i][self.board_size - 1 - i] for i in range(self.board_size)]
        ):
            winner = board[0][self.board_size - 1]
            logger.info(
                f"game.rules.tictactoe.check_win_conditions "
                f"winner={winner} condition=diagonal_anti"
            )
            return {
                "game_over": True,
                "winner": winner,
                "status": f"{winner} wins (anti-diagonal)",
            }

        # Check for draw (board full)
        if self._is_board_full(board):
            logger.info("game.rules.tictactoe.check_win_conditions draw")
            return {"game_over": True, "winner": None, "status": "Draw"}

        # Game continues
        return {"game_over": False, "winner": None, "status": "In progress"}

    def get_next_player(self, state: dict[str, Any]) -> str:
        """
        Get the next player to move.

        Args:
            state: Current game state

        Returns:
            Next player identifier ("X" or "O")
        """
        current = state.get("current_player", "X")
        next_player = "O" if current == "X" else "X"
        return next_player

    def _check_line(self, line: list[str]) -> bool:
        """
        Check if a line (row, column, or diagonal) has all the same non-empty marker.

        Args:
            line: List of cell values

        Returns:
            True if all cells are the same and non-empty
        """
        if not line or line[0] == "":
            return False
        return all(cell == line[0] for cell in line)

    def create_initial_state(self, starting_player: str = "X") -> dict[str, Any]:
        """
        Create an initial game state.

        Args:
            starting_player: Player to move first ("X" or "O")

        Returns:
            Initial game state dictionary
        """
        return {
            "board": [["" for _ in range(self.board_size)] for _ in range(self.board_size)],
            "current_player": starting_player,
            "move_count": 0,
        }

    def get_board_string(self, state: dict[str, Any]) -> str:
        """
        Get a human-readable string representation of the board.

        Args:
            state: Current game state

        Returns:
            String representation of the board
        """
        board = state.get("board", [])
        lines = []
        for i, row in enumerate(board):
            # Format each cell, using space for empty
            formatted_row = [cell if cell else " " for cell in row]
            lines.append(" | ".join(formatted_row))
            if i < len(board) - 1:
                lines.append("-" * (len(formatted_row) * 4 - 1))
        return "\n".join(lines)
