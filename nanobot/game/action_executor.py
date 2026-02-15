"""Game action executor using the ToolRegistry."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.telemetry.metrics import tool_execution_count


class GameActionExecutor:
    """
    Executes game actions using the ToolRegistry.

    This executor provides move execution with timeout handling,
    retry logic, and rate limiting for reliable game action execution.

    Args:
        registry: ToolRegistry instance for tool execution
        timeout_seconds: Default timeout for action execution
        max_retries: Maximum number of retry attempts on failure
        rate_limit_delay: Minimum delay between executions in seconds
    """

    def __init__(
        self,
        registry: ToolRegistry,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        rate_limit_delay: float = 0.1,
    ):
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self._last_execution_time: float = 0.0
        logger.info(
            f"game.action_executor.init timeout={timeout_seconds} "
            f"max_retries={max_retries} rate_limit={rate_limit_delay}"
        )

    async def execute_move(
        self,
        move: str,
        game_id: str,
        additional_params: dict[str, Any] | None = None,
        tool_name: str = "game_move",
    ) -> dict[str, Any]:
        """
        Execute a game move with timeout and retry handling.

        Uses asyncio.wait_for() for timeout enforcement and implements
        retry logic with exponential backoff on failures.

        Args:
            move: The move to execute
            game_id: ID of the game
            additional_params: Additional parameters for the tool
            tool_name: Name of the tool to use (default: "game_move")

        Returns:
            Dictionary with execution result:
            - success: bool indicating if execution succeeded
            - result: The tool execution result (if successful)
            - error: Error message (if failed)
            - attempts: Number of attempts made
        """
        params = {
            "move": move,
            "game_id": game_id,
            **(additional_params or {}),
        }

        # Apply rate limiting
        await self._apply_rate_limit()

        last_error: str = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"game.action_executor.execute_move attempt={attempt} "
                    f"game_id={game_id} move={move}"
                )

                result = await asyncio.wait_for(
                    self.registry.execute(tool_name, params),
                    timeout=self.timeout_seconds,
                )

                # Check if result indicates an error
                if result.startswith("Error:"):
                    last_error = result
                    tool_execution_count.labels(
                        tool_name=tool_name, status="error"
                    ).inc()
                    logger.warning(
                        f"game.action_executor.execute_move error={result} "
                        f"attempt={attempt} game_id={game_id}"
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(self._get_backoff_delay(attempt))
                    continue

                # Success
                tool_execution_count.labels(
                    tool_name=tool_name, status="success"
                ).inc()
                logger.info(
                    f"game.action_executor.execute_move success=true "
                    f"game_id={game_id} move={move} attempts={attempt}"
                )
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt,
                }

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout_seconds} seconds"
                tool_execution_count.labels(
                    tool_name=tool_name, status="timeout"
                ).inc()
                logger.error(
                    f"game.action_executor.execute_move error=timeout "
                    f"attempt={attempt} game_id={game_id}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self._get_backoff_delay(attempt))

            except Exception as e:
                last_error = str(e)
                tool_execution_count.labels(
                    tool_name=tool_name, status="exception"
                ).inc()
                logger.error(
                    f"game.action_executor.execute_move error={e} "
                    f"attempt={attempt} game_id={game_id}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self._get_backoff_delay(attempt))

        # All retries exhausted
        return self.handle_failure(
            move=move,
            game_id=game_id,
            error=last_error,
            attempts=self.max_retries,
        )

    async def validate_move(
        self,
        move: str,
        legal_moves: list[str],
    ) -> dict[str, Any]:
        """
        Validate a move against the list of legal moves.

        Args:
            move: The move to validate
            legal_moves: List of legal moves

        Returns:
            Dictionary with validation result:
            - valid: bool indicating if move is legal
            - error: Error message (if invalid)
        """
        if not legal_moves:
            logger.warning("game.action_executor.validate_move error=no_legal_moves")
            return {
                "valid": False,
                "error": "No legal moves available",
            }

        if move not in legal_moves:
            logger.warning(
                f"game.action_executor.validate_move error=illegal_move "
                f"move={move} legal_moves={legal_moves}"
            )
            return {
                "valid": False,
                "error": f"Illegal move: {move}. Legal moves: {', '.join(legal_moves)}",
            }

        logger.debug(f"game.action_executor.validate_move move={move} valid=true")
        return {"valid": True}

    def handle_failure(
        self,
        move: str,
        game_id: str,
        error: str,
        attempts: int,
    ) -> dict[str, Any]:
        """
        Handle execution failure after all retries exhausted.

        Args:
            move: The move that failed
            game_id: ID of the game
            error: The last error message
            attempts: Number of attempts made

        Returns:
            Dictionary with failure information
        """
        logger.error(
            f"game.action_executor.handle_failure game_id={game_id} "
            f"move={move} error={error} attempts={attempts}"
        )
        return {
            "success": False,
            "error": error,
            "attempts": attempts,
            "move": move,
            "game_id": game_id,
        }

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between executions."""
        import time

        current_time = time.time()
        time_since_last = current_time - self._last_execution_time

        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        self._last_execution_time = time.time()

    def _get_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return min(2 ** (attempt - 1), 10.0)  # Cap at 10 seconds
