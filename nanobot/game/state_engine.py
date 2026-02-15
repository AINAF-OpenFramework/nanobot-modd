"""Game state engine for managing game state and rules."""

from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from loguru import logger


@runtime_checkable
class GameRules(Protocol):
    """
    Protocol defining the interface for game rules implementations.

    All game-specific rule implementations must conform to this protocol
    to be compatible with the GameStateEngine.
    """

    def get_legal_moves(self, state: dict[str, Any]) -> list[str]:
        """
        Get all legal moves for the current game state.

        Args:
            state: Current game state dictionary

        Returns:
            List of legal move identifiers
        """
        ...

    def apply_move(self, state: dict[str, Any], move: str) -> dict[str, Any]:
        """
        Apply a move to the current state and return the new state.

        Args:
            state: Current game state dictionary
            move: The move to apply

        Returns:
            New game state after applying the move
        """
        ...

    def check_win_conditions(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Check the current state for win conditions.

        Args:
            state: Current game state dictionary

        Returns:
            Dictionary containing win condition information:
            - game_over: bool indicating if game has ended
            - winner: str or None indicating the winner
            - status: str describing the game status
        """
        ...

    def get_next_player(self, state: dict[str, Any]) -> str:
        """
        Get the identifier of the next player to move.

        Args:
            state: Current game state dictionary

        Returns:
            Identifier of the next player
        """
        ...


@dataclass
class GameHistoryEntry:
    """
    Represents a single entry in the game history.

    Args:
        state: Game state at this point
        move: Move that was applied (None for initial state)
        player: Player who made the move
        timestamp: When the move was made
        metadata: Additional metadata about this entry
    """

    state: dict[str, Any]
    move: str | None = None
    player: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class GameStateEngine:
    """
    Thread-safe game state engine for managing game state.

    This engine maintains the current game state, history, and provides
    methods for state manipulation including move application, simulation,
    and rollback.

    Args:
        game_id: Unique identifier for this game
        rules: GameRules implementation for this game type
        initial_state: Optional initial state dictionary
    """

    def __init__(
        self,
        game_id: str | None = None,
        rules: GameRules | None = None,
        initial_state: dict[str, Any] | None = None,
    ):
        self._lock = threading.RLock()
        self.game_id = game_id or str(uuid.uuid4())
        self.rules = rules
        self._state: dict[str, Any] = initial_state or {}
        self._history: list[GameHistoryEntry] = []

        if initial_state:
            self._history.append(
                GameHistoryEntry(
                    state=copy.deepcopy(initial_state),
                    move=None,
                    player=None,
                    metadata={"type": "initial"},
                )
            )

        logger.info(f"game.state_engine.init game_id={self.game_id}")

    def update(self, new_state: dict[str, Any]) -> None:
        """
        Update the current game state.

        Args:
            new_state: New state dictionary to set
        """
        with self._lock:
            self._state = copy.deepcopy(new_state)
            self._history.append(
                GameHistoryEntry(
                    state=copy.deepcopy(new_state),
                    move=None,
                    player=None,
                    metadata={"type": "update"},
                )
            )
            logger.debug(f"game.state_engine.update game_id={self.game_id}")

    def simulate(self, move: str) -> dict[str, Any]:
        """
        Simulate a move without modifying the actual game state.

        Args:
            move: The move to simulate

        Returns:
            The resulting state after the simulated move

        Raises:
            ValueError: If no rules are configured or move is invalid
        """
        with self._lock:
            if not self.rules:
                logger.error(f"game.state_engine.simulate error=no_rules game_id={self.game_id}")
                raise ValueError("No game rules configured for simulation")

            legal_moves = self.rules.get_legal_moves(self._state)
            if move not in legal_moves:
                logger.error(
                    f"game.state_engine.simulate error=illegal_move "
                    f"game_id={self.game_id} move={move}"
                )
                raise ValueError(f"Illegal move: {move}. Legal moves: {legal_moves}")

            simulated_state = self.rules.apply_move(copy.deepcopy(self._state), move)
            logger.debug(
                f"game.state_engine.simulate game_id={self.game_id} move={move}"
            )
            return simulated_state

    def apply_move(self, move: str, player: str | None = None) -> dict[str, Any]:
        """
        Apply a move to the current state and update history.

        Args:
            move: The move to apply
            player: Optional player identifier who made the move

        Returns:
            The new game state after applying the move

        Raises:
            ValueError: If no rules are configured or move is invalid
        """
        with self._lock:
            if not self.rules:
                logger.error(f"game.state_engine.apply_move error=no_rules game_id={self.game_id}")
                raise ValueError("No game rules configured")

            legal_moves = self.rules.get_legal_moves(self._state)
            if move not in legal_moves:
                logger.error(
                    f"game.state_engine.apply_move error=illegal_move "
                    f"game_id={self.game_id} move={move}"
                )
                raise ValueError(f"Illegal move: {move}. Legal moves: {legal_moves}")

            new_state = self.rules.apply_move(copy.deepcopy(self._state), move)
            self._state = new_state

            self._history.append(
                GameHistoryEntry(
                    state=copy.deepcopy(new_state),
                    move=move,
                    player=player,
                    metadata={"type": "move"},
                )
            )

            logger.info(
                f"game.state_engine.apply_move game_id={self.game_id} "
                f"move={move} player={player}"
            )
            return copy.deepcopy(new_state)

    def get_legal_moves(self) -> list[str]:
        """
        Get all legal moves for the current state.

        Returns:
            List of legal move identifiers

        Raises:
            ValueError: If no rules are configured
        """
        with self._lock:
            if not self.rules:
                logger.error(
                    f"game.state_engine.get_legal_moves error=no_rules game_id={self.game_id}"
                )
                raise ValueError("No game rules configured")

            return self.rules.get_legal_moves(self._state)

    def get_state(self) -> dict[str, Any]:
        """
        Get a copy of the current game state.

        Returns:
            Deep copy of the current state dictionary
        """
        with self._lock:
            return copy.deepcopy(self._state)

    def rollback(self, steps: int = 1) -> dict[str, Any]:
        """
        Rollback the game state by a number of steps.

        Args:
            steps: Number of steps to rollback

        Returns:
            The state after rollback

        Raises:
            ValueError: If cannot rollback requested number of steps
        """
        with self._lock:
            if steps < 1:
                raise ValueError("Steps must be at least 1")

            if len(self._history) <= steps:
                logger.error(
                    f"game.state_engine.rollback error=insufficient_history "
                    f"game_id={self.game_id} steps={steps} history_len={len(self._history)}"
                )
                raise ValueError(
                    f"Cannot rollback {steps} steps. "
                    f"History only has {len(self._history)} entries."
                )

            # Remove the last 'steps' entries
            for _ in range(steps):
                self._history.pop()

            # Set state to the last remaining entry
            self._state = copy.deepcopy(self._history[-1].state)

            logger.info(
                f"game.state_engine.rollback game_id={self.game_id} "
                f"steps={steps} new_history_len={len(self._history)}"
            )
            return copy.deepcopy(self._state)

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """
        Reset the game to initial state.

        Args:
            initial_state: Optional new initial state. If not provided,
                          resets to the first state in history or empty dict.
        """
        with self._lock:
            if initial_state is not None:
                self._state = copy.deepcopy(initial_state)
            elif self._history:
                self._state = copy.deepcopy(self._history[0].state)
            else:
                self._state = {}

            self._history = [
                GameHistoryEntry(
                    state=copy.deepcopy(self._state),
                    move=None,
                    player=None,
                    metadata={"type": "reset"},
                )
            ]

            logger.info(f"game.state_engine.reset game_id={self.game_id}")

    def get_history(self) -> list[GameHistoryEntry]:
        """
        Get a copy of the game history.

        Returns:
            List of GameHistoryEntry objects
        """
        with self._lock:
            return [
                GameHistoryEntry(
                    state=copy.deepcopy(entry.state),
                    move=entry.move,
                    player=entry.player,
                    timestamp=entry.timestamp,
                    metadata=copy.deepcopy(entry.metadata),
                )
                for entry in self._history
            ]

    def check_win_conditions(self) -> dict[str, Any]:
        """
        Check win conditions for the current state.

        Returns:
            Dictionary with win condition information

        Raises:
            ValueError: If no rules are configured
        """
        with self._lock:
            if not self.rules:
                logger.error(
                    f"game.state_engine.check_win_conditions error=no_rules "
                    f"game_id={self.game_id}"
                )
                raise ValueError("No game rules configured")

            return self.rules.check_win_conditions(self._state)

    def get_current_player(self) -> str:
        """
        Get the current player to move.

        Returns:
            Player identifier

        Raises:
            ValueError: If no rules are configured
        """
        with self._lock:
            if not self.rules:
                raise ValueError("No game rules configured")

            return self.rules.get_next_player(self._state)
