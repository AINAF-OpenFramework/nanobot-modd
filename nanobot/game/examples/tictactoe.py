"""Tic Tac Toe game rules implementation."""

from __future__ import annotations

import copy
from typing import Any

from loguru import logger

from nanobot.game.state_engine import GameRules, GameStateEngine


class TicTacToeRules(GameRules):
    """
    Implementation of Tic Tac Toe game rules.

    The board is represented as a list of 9 positions (0-8):
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8

    Board state values:
    - None or "" = empty
    - "X" = player X
    - "O" = player O
    """

    WINNING_COMBINATIONS = [
        # Rows
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        # Columns
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        # Diagonals
        [0, 4, 8],
        [2, 4, 6],
    ]

    def get_legal_moves(self, state: dict[str, Any]) -> list[str]:
        """
        Get all legal moves (empty positions) for the current state.

        Args:
            state: Current game state with 'board' key

        Returns:
            List of legal move identifiers (position numbers as strings)
        """
        board = state.get("board", [""] * 9)

        # Check if game is already over
        win_conditions = self.check_win_conditions(state)
        if win_conditions.get("game_over"):
            return []

        legal_moves = []
        for i, cell in enumerate(board):
            if not cell:  # Empty cell
                legal_moves.append(str(i))

        return legal_moves

    def apply_move(self, state: dict[str, Any], move: str) -> dict[str, Any]:
        """
        Apply a move to the current state.

        Args:
            state: Current game state
            move: Position to place the piece (0-8 as string)

        Returns:
            New state after applying the move
        """
        new_state = copy.deepcopy(state)
        board = new_state.get("board", [""] * 9)
        current_player = new_state.get("current_player", "X")

        position = int(move)
        if position < 0 or position > 8:
            raise ValueError(f"Invalid position: {move}")
        if board[position]:
            raise ValueError(f"Position {move} is already occupied")

        board[position] = current_player
        new_state["board"] = board
        new_state["current_player"] = self.get_next_player(state)
        new_state["turn"] = state.get("turn", 0) + 1

        logger.debug(
            f"game.tictactoe.apply_move position={move} "
            f"player={current_player} turn={new_state['turn']}"
        )
        return new_state

    def check_win_conditions(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Check if the game has ended (win/draw).

        Args:
            state: Current game state

        Returns:
            Dictionary with:
            - game_over: bool
            - winner: str or None
            - status: str describing the state
        """
        board = state.get("board", [""] * 9)

        # Check for winner
        for combo in self.WINNING_COMBINATIONS:
            cells = [board[i] for i in combo]
            if cells[0] and cells[0] == cells[1] == cells[2]:
                return {
                    "game_over": True,
                    "winner": cells[0],
                    "status": f"Player {cells[0]} wins!",
                }

        # Check for draw (all cells filled)
        if all(cell for cell in board):
            return {
                "game_over": True,
                "winner": None,
                "status": "Draw - no winner",
            }

        # Game continues
        return {
            "game_over": False,
            "winner": None,
            "status": "Game in progress",
        }

    def get_next_player(self, state: dict[str, Any]) -> str:
        """
        Get the next player to move.

        Args:
            state: Current game state

        Returns:
            "X" or "O"
        """
        current = state.get("current_player", "X")
        return "O" if current == "X" else "X"


def create_tictactoe_engine(game_id: str | None = None) -> GameStateEngine:
    """
    Create a new Tic Tac Toe game engine.

    Args:
        game_id: Optional game ID (auto-generated if not provided)

    Returns:
        Configured GameStateEngine for Tic Tac Toe
    """
    initial_state = {
        "board": [""] * 9,
        "current_player": "X",
        "turn": 0,
    }

    engine = GameStateEngine(
        game_id=game_id,
        rules=TicTacToeRules(),
        initial_state=initial_state,
    )

    logger.info(f"game.tictactoe.create_engine game_id={engine.game_id}")
    return engine
