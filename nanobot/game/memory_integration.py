"""Game memory integration using the existing MemoryStore."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import ContentType, FractalNode


class GameMemoryManager:
    """
    Manager for game-specific memory operations.

    Wraps the existing MemoryStore to provide game-specific
    functionality like session storage and strategy retrieval.

    CRITICAL: MUST use memory.save_fractal_node() and get_entangled_context()
    for all storage and retrieval operations.
    """

    def __init__(self, memory_store: MemoryStore):
        """
        Initialize the GameMemoryManager.

        Args:
            memory_store: The MemoryStore instance to wrap
        """
        self._memory = memory_store
        logger.info("game.memory_integration.init")

    @property
    def memory(self) -> MemoryStore:
        """Get the underlying memory store."""
        return self._memory

    def store_game_session(
        self,
        game_id: str,
        game_type: str,
        moves: list[dict[str, Any]],
        outcome: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> FractalNode:
        """
        Store a complete game session as a fractal node.

        Uses self._memory.save_fractal_node() to persist the session.

        Args:
            game_id: Unique identifier for the game
            game_type: Type of game (e.g., "tictactoe", "chess")
            moves: List of moves with their metadata
            outcome: Final outcome of the game
            metadata: Additional session metadata

        Returns:
            The created FractalNode containing the session
        """
        session_content = json.dumps(
            {
                "game_id": game_id,
                "game_type": game_type,
                "moves": moves,
                "outcome": outcome,
                "move_count": len(moves),
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
        )

        # Generate tags based on game data
        tags = [
            "game_session",
            f"game:{game_type}",
            f"game_id:{game_id}",
        ]

        # Add outcome-based tags
        if outcome.get("winner"):
            tags.append(f"winner:{outcome['winner']}")
        if outcome.get("result"):
            tags.append(f"result:{outcome['result']}")

        # Generate summary
        result_str = outcome.get("result", "completed")
        winner_str = f" - winner: {outcome['winner']}" if outcome.get("winner") else ""
        summary = f"{game_type} session ({len(moves)} moves) - {result_str}{winner_str}"

        node = self._memory.save_fractal_node(
            content=session_content,
            tags=tags,
            summary=summary,
            content_type=ContentType.TEXT,
        )

        logger.info(
            f"game.memory_integration.store_game_session game_id={game_id} "
            f"game_type={game_type} moves={len(moves)} node_id={node.id}"
        )

        return node

    def apply_importance_decay(
        self,
        decay_rate: float = 0.01,
        min_importance: float = 0.0,
    ) -> int:
        """
        Apply importance decay to all game-related memory nodes.

        Reduces the importance of older game memories to prioritize
        recent experiences.

        Args:
            decay_rate: Rate of decay per hour since last update
            min_importance: Minimum importance value (floor)

        Returns:
            Number of nodes that were updated
        """
        # Get all game session nodes
        query = "game_session game"
        nodes = self._memory.get_entangled_context(query=query, top_k=100)

        updated_count = 0
        current_time = time.time()

        for node in nodes:
            if "game_session" not in node.tags:
                continue

            # Calculate time-based decay
            node_timestamp = node.timestamp.timestamp()
            hours_elapsed = (current_time - node_timestamp) / 3600.0
            decay_factor = (1.0 - decay_rate) ** hours_elapsed

            # Apply decay
            old_importance = node.importance
            new_importance = max(min_importance, node.importance * decay_factor)

            if abs(new_importance - old_importance) > 0.001:
                node.importance = new_importance
                self._memory._update_node(node)
                updated_count += 1

        logger.info(
            f"game.memory_integration.apply_importance_decay "
            f"decay_rate={decay_rate} updated={updated_count}"
        )

        return updated_count

    def retrieve_winning_strategies(
        self,
        game_type: str,
        k: int = 10,
    ) -> list[FractalNode]:
        """
        Retrieve winning strategies for a specific game type.

        Uses self._memory.get_entangled_context() with a query
        optimized for finding winning patterns.

        Args:
            game_type: Type of game to search for
            k: Maximum number of strategies to retrieve

        Returns:
            List of FractalNodes containing winning strategies
        """
        query = f"strategy game:{game_type} win winning"
        nodes = self._memory.get_entangled_context(query=query, top_k=k * 2)

        # Filter to winning strategies
        winning_nodes = []
        for node in nodes:
            is_winning = any(
                "result:win" in tag or "winner:" in tag
                for tag in node.tags
            )
            if is_winning and "strategy" in node.tags:
                winning_nodes.append(node)
                if len(winning_nodes) >= k:
                    break

        logger.debug(
            f"game.memory_integration.retrieve_winning_strategies "
            f"game_type={game_type} found={len(winning_nodes)}"
        )

        return winning_nodes

    def retrieve_similar_game_states(
        self,
        state: dict[str, Any],
        game_type: str,
        k: int = 5,
    ) -> list[FractalNode]:
        """
        Retrieve memory nodes similar to a given game state.

        Args:
            state: Current game state to match
            game_type: Type of game
            k: Maximum number of nodes to retrieve

        Returns:
            List of similar FractalNodes
        """
        # Build query from state characteristics
        query_parts = ["game", f"game:{game_type}"]

        if "board" in state:
            query_parts.append("board")
        if "turn" in state:
            query_parts.append(f"turn:{state['turn']}")
        if "current_player" in state:
            query_parts.append(f"player:{state['current_player']}")

        query = " ".join(query_parts)
        nodes = self._memory.get_entangled_context(query=query, top_k=k)

        logger.debug(
            f"game.memory_integration.retrieve_similar_game_states "
            f"game_type={game_type} found={len(nodes)}"
        )

        return nodes

    def link_related_sessions(
        self,
        node_id_1: str,
        node_id_2: str,
        strength: float = 0.5,
    ) -> bool:
        """
        Create an entanglement link between two game sessions.

        Args:
            node_id_1: First session node ID
            node_id_2: Second session node ID
            strength: Entanglement strength (0.0 to 1.0)

        Returns:
            True if linking was successful
        """
        node_1 = self._memory.get_node_by_id(node_id_1)
        node_2 = self._memory.get_node_by_id(node_id_2)

        if not node_1 or not node_2:
            logger.warning(
                f"game.memory_integration.link_related_sessions "
                f"error=node_not_found node_id_1={node_id_1} node_id_2={node_id_2}"
            )
            return False

        # Create bidirectional entanglement
        strength = max(0.0, min(1.0, strength))
        node_1.entangled_ids[node_id_2] = strength
        node_2.entangled_ids[node_id_1] = strength

        self._memory._update_node(node_1)
        self._memory._update_node(node_2)

        logger.info(
            f"game.memory_integration.link_related_sessions "
            f"node_id_1={node_id_1} node_id_2={node_id_2} strength={strength:.3f}"
        )

        return True

    def get_session_history(
        self,
        game_type: str | None = None,
        limit: int = 20,
    ) -> list[FractalNode]:
        """
        Get recent game session history.

        Args:
            game_type: Optional filter by game type
            limit: Maximum number of sessions to return

        Returns:
            List of session FractalNodes
        """
        query = "game_session"
        if game_type:
            query += f" game:{game_type}"

        nodes = self._memory.get_entangled_context(query=query, top_k=limit * 2)

        # Filter to session nodes only
        sessions = [
            node for node in nodes
            if "game_session" in node.tags
        ][:limit]

        return sessions
