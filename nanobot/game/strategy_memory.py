"""Strategy memory that wraps the existing MemoryStore."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import ContentType, FractalNode


class StrategyMemory:
    """
    Strategy memory that wraps the existing MemoryStore.

    This provides game-specific memory operations for storing and retrieving
    strategies, outcomes, and game patterns using the underlying MemoryStore.

    CRITICAL: This class MUST use self.memory.save_fractal_node(),
    self.memory.get_entangled_context(), and self.memory._update_node()
    for all memory operations.

    Args:
        memory_store: The MemoryStore instance to wrap
    """

    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store
        logger.info("game.strategy_memory.init")

    def store_strategy(
        self,
        state: dict[str, Any],
        move: str,
        outcome: dict[str, Any],
        game_type: str = "unknown",
        tags: list[str] | None = None,
    ) -> FractalNode:
        """
        Store a strategy (state + move + outcome) in memory.

        Uses self.memory.save_fractal_node() to persist the strategy
        as a fractal node with appropriate tags and summary.

        Args:
            state: The game state when the move was made
            move: The move that was played
            outcome: The outcome of the move (win/loss/draw, score, etc.)
            game_type: Type of game for categorization
            tags: Additional tags for the strategy

        Returns:
            The created FractalNode containing the strategy
        """
        strategy_content = json.dumps(
            {
                "state": state,
                "move": move,
                "outcome": outcome,
                "game_type": game_type,
            },
            indent=2,
        )

        strategy_tags = ["strategy", f"game:{game_type}", f"move:{move}"]
        if tags:
            strategy_tags.extend(tags)

        # Generate outcome-based tags
        if outcome.get("winner"):
            strategy_tags.append(f"outcome:{outcome['winner']}")
        if outcome.get("result"):
            strategy_tags.append(f"result:{outcome['result']}")

        # Create summary for quick retrieval
        summary = self._generate_strategy_summary(state, move, outcome, game_type)

        node = self.memory.save_fractal_node(
            content=strategy_content,
            tags=strategy_tags,
            summary=summary,
            content_type=ContentType.TEXT,
        )

        logger.info(
            f"game.strategy_memory.store_strategy node_id={node.id} "
            f"game_type={game_type} move={move}"
        )
        return node

    def retrieve_relevant_strategies(
        self,
        state: dict[str, Any],
        k: int = 5,
        game_type: str | None = None,
    ) -> list[FractalNode]:
        """
        Retrieve strategies relevant to the current game state.

        Uses self.memory.get_entangled_context() to find similar strategies
        based on state similarity and entanglement relationships.

        Args:
            state: The current game state to match against
            k: Maximum number of strategies to retrieve
            game_type: Optional game type filter

        Returns:
            List of FractalNodes containing relevant strategies
        """
        # Build query from state characteristics
        query = self._build_state_query(state, game_type)

        # Use the entangled context retrieval for hybrid scoring
        nodes = self.memory.get_entangled_context(query=query, top_k=k)

        # Filter to only strategy nodes
        strategy_nodes = [
            node for node in nodes
            if "strategy" in node.tags
        ]

        logger.info(
            f"game.strategy_memory.retrieve_relevant_strategies "
            f"query_len={len(query)} found={len(strategy_nodes)} k={k}"
        )
        return strategy_nodes

    def update_strategy_weight(
        self,
        node_id: str,
        outcome: dict[str, Any],
        adjustment: float = 0.1,
    ) -> bool:
        """
        Update the weight/importance of a strategy based on new outcome.

        Uses self.memory._update_node() to persist the weight change.

        Args:
            node_id: ID of the strategy node to update
            outcome: The outcome that triggered the update
            adjustment: Base adjustment value (positive or negative)

        Returns:
            True if update was successful, False otherwise
        """
        node = self.memory.get_node_by_id(node_id)
        if not node:
            logger.warning(
                f"game.strategy_memory.update_strategy_weight "
                f"error=node_not_found node_id={node_id}"
            )
            return False

        # Calculate adjustment based on outcome
        if outcome.get("result") == "win" or outcome.get("winner"):
            importance_delta = abs(adjustment)
        elif outcome.get("result") == "loss":
            importance_delta = -abs(adjustment)
        else:
            importance_delta = 0.0

        # Update importance with bounds
        old_importance = node.importance
        new_importance = max(0.0, min(1.0, node.importance + importance_delta))
        node.importance = new_importance

        # Update the node
        self.memory._update_node(node)

        logger.info(
            f"game.strategy_memory.update_strategy_weight node_id={node_id} "
            f"old_importance={old_importance:.3f} new_importance={new_importance:.3f}"
        )
        return True

    def get_winning_strategies(
        self,
        game_type: str,
        k: int = 10,
    ) -> list[FractalNode]:
        """
        Retrieve strategies that led to wins for a specific game type.

        Args:
            game_type: The game type to filter by
            k: Maximum number of strategies to retrieve

        Returns:
            List of winning strategy nodes
        """
        query = f"strategy game:{game_type} win winning"
        nodes = self.memory.get_entangled_context(query=query, top_k=k)

        # Filter to winning strategies
        winning_nodes = [
            node for node in nodes
            if "strategy" in node.tags
            and any("outcome:win" in tag or "result:win" in tag for tag in node.tags)
        ]

        logger.debug(
            f"game.strategy_memory.get_winning_strategies "
            f"game_type={game_type} found={len(winning_nodes)}"
        )
        return winning_nodes

    def link_strategies(
        self,
        node_id_1: str,
        node_id_2: str,
        strength: float = 0.5,
    ) -> bool:
        """
        Create an entanglement link between two strategy nodes.

        Args:
            node_id_1: First node ID
            node_id_2: Second node ID
            strength: Entanglement strength (0.0 to 1.0)

        Returns:
            True if linking was successful, False otherwise
        """
        node_1 = self.memory.get_node_by_id(node_id_1)
        node_2 = self.memory.get_node_by_id(node_id_2)

        if not node_1 or not node_2:
            logger.warning(
                f"game.strategy_memory.link_strategies "
                f"error=node_not_found node_id_1={node_id_1} node_id_2={node_id_2}"
            )
            return False

        # Create bidirectional entanglement
        node_1.entangled_ids[node_id_2] = max(0.0, min(1.0, strength))
        node_2.entangled_ids[node_id_1] = max(0.0, min(1.0, strength))

        self.memory._update_node(node_1)
        self.memory._update_node(node_2)

        logger.info(
            f"game.strategy_memory.link_strategies "
            f"node_id_1={node_id_1} node_id_2={node_id_2} strength={strength:.3f}"
        )
        return True

    def _generate_strategy_summary(
        self,
        state: dict[str, Any],
        move: str,
        outcome: dict[str, Any],
        game_type: str,
    ) -> str:
        """Generate a summary string for a strategy."""
        outcome_str = outcome.get("result", outcome.get("winner", "unknown"))
        state_summary = self._summarize_state(state)
        return f"{game_type} strategy: {move} in {state_summary} -> {outcome_str}"

    def _summarize_state(self, state: dict[str, Any]) -> str:
        """Create a brief summary of a game state."""
        if "board" in state:
            return "board position"
        if "turn" in state:
            return f"turn {state['turn']}"
        return "game position"

    def _build_state_query(
        self,
        state: dict[str, Any],
        game_type: str | None = None,
    ) -> str:
        """Build a query string from a game state."""
        query_parts = ["strategy"]

        if game_type:
            query_parts.append(f"game:{game_type}")

        # Extract key state characteristics
        if "board" in state:
            query_parts.append("board")
        if "current_player" in state:
            query_parts.append(f"player:{state['current_player']}")
        if "turn" in state:
            query_parts.append(f"turn:{state['turn']}")

        return " ".join(query_parts)
