"""Game move tool for executing game moves."""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class GameMoveTool(Tool):
    """
    Tool for executing game moves.

    This tool validates moves against legal moves and executes them
    through a configured game action handler.

    Args:
        legal_moves_provider: Callable that returns current legal moves
        move_handler: Callable that handles move execution
    """

    def __init__(
        self,
        legal_moves_provider: Any | None = None,
        move_handler: Any | None = None,
    ):
        self._legal_moves_provider = legal_moves_provider
        self._move_handler = move_handler
        self._legal_moves: list[str] = []

    @property
    def name(self) -> str:
        return "game_move"

    @property
    def description(self) -> str:
        return (
            "Execute a game move. The move must be one of the legal moves "
            "available in the current game state."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": "The move to execute (must be a legal move)",
                },
                "game_id": {
                    "type": "string",
                    "description": "The ID of the game to execute the move in",
                },
                "player": {
                    "type": "string",
                    "description": "Optional player identifier making the move",
                },
            },
            "required": ["move", "game_id"],
        }

    async def execute(
        self,
        move: str,
        game_id: str,
        player: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute a game move.

        Validates the move against legal moves and executes it through
        the configured move handler.

        Args:
            move: The move to execute
            game_id: ID of the game
            player: Optional player identifier
            **kwargs: Additional parameters

        Returns:
            String result describing the execution outcome
        """
        logger.debug(
            f"game.tool.game_move.execute game_id={game_id} move={move} player={player}"
        )

        # Get legal moves
        legal_moves = self._get_legal_moves()

        # Validate move
        if legal_moves and move not in legal_moves:
            error_msg = (
                f"Illegal move: {move}. "
                f"Legal moves are: {', '.join(legal_moves)}"
            )
            logger.warning(
                f"game.tool.game_move.execute error=illegal_move "
                f"game_id={game_id} move={move}"
            )
            return f"Error: {error_msg}"

        # Execute move through handler if available
        if self._move_handler:
            try:
                result = await self._call_handler(move, game_id, player)
                logger.info(
                    f"game.tool.game_move.execute success=true "
                    f"game_id={game_id} move={move}"
                )
                return result
            except Exception as e:
                logger.error(
                    f"game.tool.game_move.execute error={e} "
                    f"game_id={game_id} move={move}"
                )
                return f"Error: Move execution failed: {str(e)}"

        # Default success response if no handler
        logger.info(
            f"game.tool.game_move.execute success=true "
            f"game_id={game_id} move={move} handler=none"
        )
        return f"Move '{move}' executed successfully for game {game_id}"

    def set_legal_moves(self, moves: list[str]) -> None:
        """
        Set the current legal moves for validation.

        Args:
            moves: List of legal move identifiers
        """
        self._legal_moves = moves.copy()
        logger.debug(f"game.tool.game_move.set_legal_moves count={len(moves)}")

    def set_move_handler(self, handler: Any) -> None:
        """
        Set the move execution handler.

        Args:
            handler: Callable that handles move execution
        """
        self._move_handler = handler

    def set_legal_moves_provider(self, provider: Any) -> None:
        """
        Set the legal moves provider.

        Args:
            provider: Callable that returns current legal moves
        """
        self._legal_moves_provider = provider

    def _get_legal_moves(self) -> list[str]:
        """Get legal moves from provider or cached list."""
        if self._legal_moves_provider:
            try:
                return self._legal_moves_provider()
            except Exception as e:
                logger.warning(
                    f"game.tool.game_move._get_legal_moves "
                    f"error=provider_failed error_msg={e}"
                )
        return self._legal_moves

    async def _call_handler(
        self,
        move: str,
        game_id: str,
        player: str | None,
    ) -> str:
        """Call the move handler, handling both sync and async handlers."""
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(self._move_handler):
            return await self._move_handler(move, game_id, player)
        else:
            # Wrap sync handler in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._move_handler, move, game_id, player
            )
