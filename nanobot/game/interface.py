"""Game observation interface for parsing game state and injecting into context."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from nanobot.agent.memory_types import ContextBlock


class GameObservation(BaseModel):
    """
    Model representing the current state of a game observation.

    This is the canonical format for game state that can be parsed from
    various game backends and injected into the agent's context.

    Args:
        game_id: Unique identifier for this game instance
        game_type: Type of game (e.g., "tictactoe", "chess")
        turn_number: Current turn number (0-indexed)
        current_player: Identifier for the player whose turn it is
        board_state: Current state of the game board (format varies by game)
        legal_moves: List of legal moves available to the current player
        win_conditions: Description of win conditions or current game status
    """

    game_id: str
    game_type: str
    turn_number: int = 0
    current_player: str
    board_state: dict[str, Any] = Field(default_factory=dict)
    legal_moves: list[str] = Field(default_factory=list)
    win_conditions: dict[str, Any] = Field(default_factory=dict)


def parse_observation(raw_input: dict[str, Any] | str) -> GameObservation:
    """
    Parse a raw game observation into a GameObservation model.

    Args:
        raw_input: Either a dictionary or JSON string representing game state

    Returns:
        A validated GameObservation instance

    Raises:
        ValueError: If the input cannot be parsed or is missing required fields
    """
    try:
        if isinstance(raw_input, str):
            data = json.loads(raw_input)
        else:
            data = raw_input

        if not isinstance(data, dict):
            logger.error(f"game.interface.parse_observation error=invalid_type type={type(data)}")
            raise ValueError("Input must be a dictionary or JSON string representing a dictionary")

        observation = GameObservation.model_validate(data)
        logger.debug(
            f"game.interface.parse_observation game_id={observation.game_id} "
            f"game_type={observation.game_type} turn={observation.turn_number}"
        )
        return observation

    except json.JSONDecodeError as e:
        logger.error(f"game.interface.parse_observation error=json_decode error_msg={e}")
        raise ValueError(f"Failed to parse JSON input: {e}") from e
    except Exception as e:
        logger.error(f"game.interface.parse_observation error=validation error_msg={e}")
        raise ValueError(f"Failed to validate game observation: {e}") from e


def inject_into_context(
    observation: GameObservation,
    context_builder: Any,
) -> ContextBlock:
    """
    Inject a game observation into the context builder.

    This creates a ContextBlock containing formatted game state information
    that can be added to the agent's context for decision-making.

    Args:
        observation: The parsed game observation
        context_builder: The ContextBuilder instance (not used directly but
                         provided for future extensibility)

    Returns:
        A ContextBlock containing the formatted game context
    """
    content = build_game_context_prompt(observation)
    block = ContextBlock(
        name="game_state",
        content=content,
        metadata={
            "game_id": observation.game_id,
            "game_type": observation.game_type,
            "turn_number": observation.turn_number,
        },
    )
    logger.info(
        f"game.interface.inject_into_context game_id={observation.game_id} "
        f"game_type={observation.game_type}"
    )
    return block


def build_game_context_prompt(observation: GameObservation) -> str:
    """
    Build a formatted prompt string from a game observation.

    This creates a human-readable and LLM-parseable representation of
    the current game state that can be included in the agent's context.

    Args:
        observation: The parsed game observation

    Returns:
        A formatted string representing the game state
    """
    lines = [
        "# Current Game State",
        "",
        f"**Game Type:** {observation.game_type}",
        f"**Game ID:** {observation.game_id}",
        f"**Turn:** {observation.turn_number}",
        f"**Current Player:** {observation.current_player}",
        "",
        "## Board State",
        _format_board_state(observation.board_state),
        "",
        "## Legal Moves",
        _format_legal_moves(observation.legal_moves),
        "",
        "## Win Conditions",
        _format_win_conditions(observation.win_conditions),
    ]
    return "\n".join(lines)


def _format_board_state(board_state: dict[str, Any]) -> str:
    """Format board state for display."""
    if not board_state:
        return "(empty)"
    return json.dumps(board_state, indent=2)


def _format_legal_moves(legal_moves: list[str]) -> str:
    """Format legal moves for display."""
    if not legal_moves:
        return "(no legal moves)"
    return ", ".join(legal_moves)


def _format_win_conditions(win_conditions: dict[str, Any]) -> str:
    """Format win conditions for display."""
    if not win_conditions:
        return "(standard rules)"

    parts = []
    if "winner" in win_conditions:
        parts.append(f"Winner: {win_conditions['winner']}")
    if "game_over" in win_conditions:
        parts.append(f"Game Over: {win_conditions['game_over']}")
    if "status" in win_conditions:
        parts.append(f"Status: {win_conditions['status']}")

    return "\n".join(parts) if parts else json.dumps(win_conditions, indent=2)
