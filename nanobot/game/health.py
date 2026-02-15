"""Game health status module for monitoring game controllers."""

from __future__ import annotations

from typing import Any

from loguru import logger

# Module-level registry for game controllers
_game_controllers: dict[str, Any] = {}


def register_game_controller(game_id: str, controller: Any) -> None:
    """
    Register a game controller for health monitoring.

    Args:
        game_id: Unique identifier for the game
        controller: GameLearningController or similar object
    """
    _game_controllers[game_id] = controller
    logger.info(f"game.health.register_game_controller game_id={game_id}")


def unregister_game_controller(game_id: str) -> None:
    """
    Unregister a game controller from health monitoring.

    Args:
        game_id: Unique identifier for the game
    """
    if game_id in _game_controllers:
        del _game_controllers[game_id]
        logger.info(f"game.health.unregister_game_controller game_id={game_id}")
    else:
        logger.warning(
            f"game.health.unregister_game_controller game_id={game_id} not_found"
        )


def get_game_health_status() -> dict[str, Any]:
    """
    Get health status for all registered game controllers.

    Returns:
        Dictionary containing health information for each game:
        - active_games: Number of active games
        - games: Dict mapping game_id to individual game status
    """
    games_status: dict[str, Any] = {}

    for game_id, controller in _game_controllers.items():
        try:
            # Check if controller has get_health_status method
            if hasattr(controller, "get_health_status"):
                games_status[game_id] = controller.get_health_status()
            else:
                # Basic status for controllers without health method
                games_status[game_id] = {
                    "status": "active",
                    "game_type": getattr(controller, "game_type", "unknown"),
                }
        except Exception as e:
            logger.error(
                f"game.health.get_game_health_status error={e} game_id={game_id}"
            )
            games_status[game_id] = {
                "status": "error",
                "error": str(e),
            }

    status = {
        "active_games": len(_game_controllers),
        "games": games_status,
    }

    logger.debug(
        f"game.health.get_game_health_status active_games={len(_game_controllers)}"
    )
    return status


def extend_health_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Extend an existing health payload with game status information.

    This function is designed to be called by the main health endpoint
    to include game-specific information in the overall health response.

    Args:
        payload: Existing health payload dictionary

    Returns:
        Extended payload with game health information
    """
    game_health = get_game_health_status()

    # Add game status to payload
    payload["games"] = game_health

    logger.debug(
        f"game.health.extend_health_payload "
        f"active_games={game_health['active_games']}"
    )
    return payload


def get_registered_game_ids() -> list[str]:
    """
    Get list of all registered game IDs.

    Returns:
        List of game IDs currently registered
    """
    return list(_game_controllers.keys())


def get_game_controller(game_id: str) -> Any | None:
    """
    Get a registered game controller by ID.

    Args:
        game_id: Unique identifier for the game

    Returns:
        The game controller or None if not found
    """
    return _game_controllers.get(game_id)


def clear_all_controllers() -> None:
    """
    Clear all registered game controllers.

    This is primarily for testing purposes.
    """
    _game_controllers.clear()
    logger.info("game.health.clear_all_controllers")
