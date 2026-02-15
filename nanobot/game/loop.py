"""Autonomous game loop integrating soul, reasoning, and memory."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import FractalNode
from nanobot.game.logger import GameLogger
from nanobot.game.memory_integration import GameMemoryManager
from nanobot.game.reasoning_engine import GameReasoningEngine
from nanobot.game.state_engine import GameRules, GameStateEngine
from nanobot.game.strategy_memory import StrategyMemory
from nanobot.providers.base import LLMProvider
from nanobot.soul.loader import SoulLoader
from nanobot.soul.traits import TraitScorer


class GameLoopConfig(BaseModel):
    """
    Configuration for the autonomous game loop.

    Attributes:
        max_turns: Maximum number of turns before stopping
        turn_timeout: Timeout in seconds for each turn
        auto_save_strategy: Whether to save strategies after each game
        log_all_moves: Whether to log all moves with full metrics
    """

    max_turns: int = Field(default=100, ge=1)
    turn_timeout: int = Field(default=30, ge=1)
    auto_save_strategy: bool = True
    log_all_moves: bool = True


class AutonomousGameLoop:
    """
    Autonomous game loop that integrates soul, reasoning, and memory.

    Provides a complete game-playing agent that uses personality traits
    for decision making, stores successful strategies, and logs all moves.

    CRITICAL RULES:
    - MUST call reasoning_engine.select_best_move() for move selection
    - MUST call strategy_memory.store_strategy() to save strategies
    - MUST read soul.yaml only, never soul.md
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        environment: GameRules,
        workspace: Path,
        rules: GameRules | None = None,
        config: GameLoopConfig | None = None,
        game_type: str = "unknown",
        player_id: str = "nanobot",
    ):
        """
        Initialize the autonomous game loop.

        Args:
            provider: LLM provider for reasoning
            model: Model identifier to use
            environment: Game environment (implements GameRules)
            workspace: Path to workspace directory
            rules: Optional separate rules (uses environment if not provided)
            config: Loop configuration
            game_type: Type of game being played
            player_id: Identifier for this agent
        """
        self._config = config or GameLoopConfig()
        self._game_type = game_type
        self._player_id = player_id
        self._running = False
        self._game_id = str(uuid.uuid4())

        # Initialize components
        self._workspace = workspace
        self._rules = rules or environment

        # Soul layer
        self._soul_loader = SoulLoader.get_instance(workspace)
        self._trait_scorer = TraitScorer(self._soul_loader)

        # Memory components
        self._memory_store = MemoryStore(workspace)
        self._strategy_memory = StrategyMemory(self._memory_store)
        self._game_memory = GameMemoryManager(self._memory_store)

        # Game state
        self._state_engine = GameStateEngine(
            game_id=self._game_id,
            rules=self._rules,
        )

        # Get soul-based reasoning parameters
        reasoning_depth = self._trait_scorer.get_reasoning_depth({})
        monte_carlo_samples = self._trait_scorer.get_monte_carlo_samples({})

        # Reasoning engine
        self._reasoning_engine = GameReasoningEngine(
            provider=provider,
            model=model,
            state_engine=self._state_engine,
            timeout_seconds=self._config.turn_timeout,
            memory_config={
                "reasoning_depth": reasoning_depth,
                "monte_carlo_samples": monte_carlo_samples,
            },
        )

        # Logger
        self._game_logger = GameLogger(
            game_id=self._game_id,
            log_dir=workspace / "logs" if workspace else None,
        )

        # Turn tracking
        self._turn_number = 0
        self._moves_log: list[dict[str, Any]] = []

        logger.info(
            f"game.loop.init game_id={self._game_id} game_type={game_type} "
            f"player_id={player_id} reasoning_depth={reasoning_depth}"
        )

    @property
    def game_id(self) -> str:
        """Get the game ID."""
        return self._game_id

    @property
    def turn_number(self) -> int:
        """Get the current turn number."""
        return self._turn_number

    @property
    def is_running(self) -> bool:
        """Check if the game loop is running."""
        return self._running

    async def run(self) -> dict[str, Any]:
        """
        Run the autonomous game loop until completion.

        Returns:
            Dictionary containing game results and statistics
        """
        self._running = True
        self._turn_number = 0
        start_time = time.time()

        logger.info(f"game.loop.run starting game_id={self._game_id}")

        try:
            while self._running and self._turn_number < self._config.max_turns:
                # Get current game state
                game_state = self._state_engine.get_state()

                # Check win conditions
                try:
                    win_status = self._state_engine.check_win_conditions()
                    if win_status.get("game_over"):
                        logger.info(
                            f"game.loop.run game_over game_id={self._game_id} "
                            f"status={win_status}"
                        )
                        break
                except ValueError:
                    pass  # No rules configured for win check

                # Play a turn
                turn_result = await self._play_turn(game_state)
                if turn_result is None:
                    # No valid move available
                    logger.warning(
                        f"game.loop.run no_valid_move game_id={self._game_id} "
                        f"turn={self._turn_number}"
                    )
                    break

                self._turn_number += 1

        except asyncio.CancelledError:
            logger.info(f"game.loop.run cancelled game_id={self._game_id}")
        except Exception as e:
            logger.error(f"game.loop.run error={e} game_id={self._game_id}")
        finally:
            self._running = False

        # Get final outcome
        try:
            final_status = self._state_engine.check_win_conditions()
        except ValueError:
            final_status = {"game_over": True, "status": "completed"}

        # Calculate duration
        duration_seconds = time.time() - start_time

        # Store session if configured
        if self._config.auto_save_strategy:
            self._game_memory.store_game_session(
                game_id=self._game_id,
                game_type=self._game_type,
                moves=self._moves_log,
                outcome=final_status,
                metadata={
                    "player_id": self._player_id,
                    "duration_seconds": duration_seconds,
                    "turns": self._turn_number,
                },
            )

        # Get session summary
        summary = self._game_logger.get_session_summary()

        result = {
            "game_id": self._game_id,
            "game_type": self._game_type,
            "outcome": final_status,
            "turns": self._turn_number,
            "duration_seconds": duration_seconds,
            "session_summary": summary,
        }

        logger.info(
            f"game.loop.run completed game_id={self._game_id} "
            f"turns={self._turn_number} duration={duration_seconds:.1f}s"
        )

        return result

    async def _play_turn(
        self,
        game_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Play a single turn of the game.

        Args:
            game_state: Current game state

        Returns:
            Turn result dictionary, or None if no valid move
        """
        turn_start = time.time()

        # Retrieve relevant memory context
        memory_nodes = self._strategy_memory.retrieve_relevant_strategies(
            state=game_state,
            k=5,
            game_type=self._game_type,
        )

        # Build context summary
        context_summary = self._build_context_summary(game_state, memory_nodes)

        # Get trait-based parameters
        soul_config = self._soul_loader.load()
        trait_weights = self._soul_loader.get_trait_weights()
        reasoning_depth = self._trait_scorer.get_reasoning_depth(game_state)

        # Select best move using reasoning engine
        # CRITICAL: Must use reasoning_engine.select_best_move()
        selected_move = await self._reasoning_engine.select_best_move(context_summary)

        if not selected_move:
            return None

        # Apply the move
        try:
            new_state = self._state_engine.apply_move(selected_move, self._player_id)
        except ValueError as e:
            logger.warning(f"game.loop._play_turn invalid_move={e}")
            return None

        # Calculate reasoning time
        reasoning_time_ms = int((time.time() - turn_start) * 1000)

        # Detect active strategy
        active_strategy = self._trait_scorer._detect_active_strategy(
            game_state, soul_config.strategies
        )

        # Determine goal alignment
        goal_alignment = []
        for goal in soul_config.goals:
            if any(action.lower() in selected_move.lower() for action in goal.actions):
                goal_alignment.append(goal.name)

        # Log the move if configured
        if self._config.log_all_moves:
            self._game_logger.log_move(
                turn_number=self._turn_number,
                move=selected_move,
                player=self._player_id,
                reasoning_depth=reasoning_depth,
                entropy=0.0,  # Would come from reasoning state
                monte_carlo_samples=self._trait_scorer.get_monte_carlo_samples(game_state),
                traits_applied=trait_weights,
                strategy_used=active_strategy.name if active_strategy else None,
                goal_alignment=goal_alignment,
                confidence=0.8,  # Would come from hypothesis confidence
                reasoning_time_ms=reasoning_time_ms,
            )

        # Store strategy
        # CRITICAL: Must use strategy_memory.store_strategy()
        try:
            win_status = self._state_engine.check_win_conditions()
        except ValueError:
            win_status = {}

        self._strategy_memory.store_strategy(
            state=game_state,
            move=selected_move,
            outcome=win_status,
            game_type=self._game_type,
            tags=[
                f"turn:{self._turn_number}",
                f"player:{self._player_id}",
            ],
        )

        # Record move for session storage
        move_record = {
            "turn": self._turn_number,
            "move": selected_move,
            "player": self._player_id,
            "state_before": game_state,
            "state_after": new_state,
            "reasoning_time_ms": reasoning_time_ms,
        }
        self._moves_log.append(move_record)

        logger.info(
            f"game.loop._play_turn game_id={self._game_id} turn={self._turn_number} "
            f"move={selected_move} time_ms={reasoning_time_ms}"
        )

        return move_record

    def _build_context_summary(
        self,
        game_state: dict[str, Any],
        memory_nodes: list[FractalNode],
    ) -> str:
        """
        Build a context summary for reasoning.

        Args:
            game_state: Current game state
            memory_nodes: Relevant memory nodes

        Returns:
            Context summary string
        """
        parts = []

        # Add soul context
        self._soul_loader.load()  # Ensure config is loaded
        active_goals = self._soul_loader.get_active_goals()
        parts.append(f"Active Goals: {', '.join(active_goals[:3])}")

        # Add trait info
        trait_weights = self._soul_loader.get_trait_weights()
        top_traits = sorted(trait_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        parts.append(f"Primary Traits: {', '.join(t[0] for t in top_traits)}")

        # Add game state summary
        parts.append(f"Turn: {self._turn_number}")
        parts.append(f"Game Type: {self._game_type}")

        # Add relevant strategy hints from memory
        if memory_nodes:
            parts.append("Relevant strategies from memory:")
            for node in memory_nodes[:3]:
                parts.append(f"  - {node.context_summary}")

        return "\n".join(parts)

    def stop(self) -> None:
        """Stop the game loop."""
        self._running = False
        logger.info(f"game.loop.stop game_id={self._game_id}")

    def get_state(self) -> dict[str, Any]:
        """Get the current game state."""
        return self._state_engine.get_state()

    def get_history(self) -> list[dict[str, Any]]:
        """Get the move history."""
        return list(self._moves_log)


async def play_game(
    provider: LLMProvider,
    model: str,
    environment: GameRules,
    workspace: Path,
    game_type: str = "unknown",
    player_id: str = "nanobot",
    initial_state: dict[str, Any] | None = None,
    config: GameLoopConfig | None = None,
) -> dict[str, Any]:
    """
    Play a complete game using the autonomous game loop.

    Convenience function that creates and runs an AutonomousGameLoop.

    Args:
        provider: LLM provider for reasoning
        model: Model identifier to use
        environment: Game environment (implements GameRules)
        workspace: Path to workspace directory
        game_type: Type of game being played
        player_id: Identifier for this agent
        initial_state: Optional initial game state
        config: Optional loop configuration

    Returns:
        Game results dictionary
    """
    loop = AutonomousGameLoop(
        provider=provider,
        model=model,
        environment=environment,
        workspace=workspace,
        config=config,
        game_type=game_type,
        player_id=player_id,
    )

    # Set initial state if provided
    if initial_state:
        loop._state_engine.update(initial_state)

    return await loop.run()
