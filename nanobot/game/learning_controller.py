"""Game learning controller that orchestrates the learning loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.game.environment import ActionResult, GameEnvironmentAdapter, GameState
from nanobot.game.fusion import FusionConfig, FusedEmbedding, MultimodalFusionLayer
from nanobot.game.health import register_game_controller, unregister_game_controller
from nanobot.game.reasoning_engine import GameReasoningEngine
from nanobot.game.state_engine import GameRules, GameStateEngine
from nanobot.game.strategy_memory import StrategyMemory
from nanobot.game.visual_perception import VisualEmbedding, create_encoder
from nanobot.providers.base import LLMProvider


@dataclass
class LearningConfig:
    """
    Configuration for the game learning controller.

    Args:
        visual_encoder_type: Type of visual encoder ('auto', 'mobilenet', 'efficientnet', 'grid')
        visual_embedding_dim: Dimension for visual embeddings
        visual_device: PyTorch device for visual encoder
        fusion_weights: Tuple of (state_weight, visual_weight, memory_weight)
        reasoning_timeout: Timeout in seconds for reasoning operations
        monte_carlo_samples: Number of samples for Monte Carlo tree search (if used)
        beam_width: Beam width for beam search reasoning
        memory_top_k: Number of memory nodes to retrieve
        store_all_moves: Whether to store all moves or only significant ones
        log_screenshots: Whether to log screenshots to memory
    """

    visual_encoder_type: str = "auto"
    visual_embedding_dim: int = 256
    visual_device: str = "cpu"
    fusion_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    reasoning_timeout: int = 10
    monte_carlo_samples: int = 100
    beam_width: int = 3
    memory_top_k: int = 5
    store_all_moves: bool = True
    log_screenshots: bool = False


@dataclass
class LearningState:
    """
    Tracks the learning progress for a game session.

    Args:
        game_id: Unique identifier for the game
        episode: Current episode number
        total_moves: Total moves made across all episodes
        total_reward: Cumulative reward
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        move_history: History of moves in current episode
    """

    game_id: str
    episode: int = 0
    total_moves: int = 0
    total_reward: float = 0.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    move_history: list[dict[str, Any]] = field(default_factory=list)

    def record_move(
        self,
        move: str,
        reward: float = 0.0,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a move in the history.

        Args:
            move: The move that was made
            reward: Reward received for the move
            state_before: Game state before the move
            state_after: Game state after the move
        """
        self.move_history.append(
            {
                "move": move,
                "reward": reward,
                "state_before": state_before,
                "state_after": state_after,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.total_moves += 1
        self.total_reward += reward

    def record_game_end(self, result: str) -> None:
        """
        Record the end of a game.

        Args:
            result: Result of the game ('win', 'loss', 'draw')
        """
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        elif result == "draw":
            self.draws += 1

        self.episode += 1
        self.move_history = []  # Reset for next episode

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the learning progress.

        Returns:
            Dictionary containing learning statistics
        """
        total_games = self.wins + self.losses + self.draws
        win_rate = self.wins / total_games if total_games > 0 else 0.0

        return {
            "game_id": self.game_id,
            "episode": self.episode,
            "total_moves": self.total_moves,
            "total_reward": self.total_reward,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": win_rate,
            "avg_moves_per_game": (
                self.total_moves / total_games if total_games > 0 else 0
            ),
        }


class GameLearningController:
    """
    Main controller for game learning that orchestrates the learning loop.

    The controller integrates:
    - GameEnvironmentAdapter: For game interaction
    - VisualEncoder: For screenshot processing
    - MultimodalFusionLayer: For combining modalities
    - GameStateEngine: For state management
    - GameReasoningEngine: For move selection (uses LatentReasoner)
    - StrategyMemory: For strategy storage (uses MemoryStore)

    The learning loop follows: perceive -> select -> execute -> observe

    Args:
        provider: LLM provider for reasoning
        model: Model identifier for reasoning
        environment: GameEnvironmentAdapter for game interaction
        workspace: Path to workspace directory
        config: LearningConfig with settings
        rules: Optional GameRules for state engine
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        environment: GameEnvironmentAdapter,
        workspace: Path | str,
        config: LearningConfig | None = None,
        rules: GameRules | None = None,
    ):
        self.provider = provider
        self.model = model
        self.environment = environment
        self.workspace = Path(workspace)
        self.config = config or LearningConfig()
        self.game_id = environment.game_id
        self.game_type = environment.game_type

        # Initialize components
        self.memory_store = MemoryStore(self.workspace)
        self.strategy_memory = StrategyMemory(self.memory_store)

        # Visual encoder
        self.visual_encoder = create_encoder(
            encoder_type=self.config.visual_encoder_type,
            embedding_dim=self.config.visual_embedding_dim,
            device=self.config.visual_device,
        )

        # Fusion layer
        fusion_config = FusionConfig(
            state_weight=self.config.fusion_weights[0],
            visual_weight=self.config.fusion_weights[1],
            memory_weight=self.config.fusion_weights[2],
            memory_top_k=self.config.memory_top_k,
        )
        self.fusion_layer = MultimodalFusionLayer(
            config=fusion_config,
            memory_store=self.memory_store,
        )

        # State engine
        self.state_engine = GameStateEngine(
            game_id=self.game_id,
            rules=rules,
        )

        # Reasoning engine
        self.reasoning_engine = GameReasoningEngine(
            provider=provider,
            model=model,
            state_engine=self.state_engine,
            timeout_seconds=self.config.reasoning_timeout,
        )

        # Learning state
        self.learning_state = LearningState(game_id=self.game_id)

        # Register with health monitoring
        register_game_controller(self.game_id, self)

        logger.info(
            f"game.learning_controller.init game_id={self.game_id} "
            f"game_type={self.game_type} model={model}"
        )

    async def perceive(self) -> tuple[GameState, VisualEmbedding | None]:
        """
        Perceive the current game state and visual information.

        Returns:
            Tuple of (GameState, VisualEmbedding or None)
        """
        # Get game state from environment
        game_state = self.environment.get_state()

        # Update state engine
        self.state_engine.update(game_state.to_dict())

        # Get screenshot and encode
        visual_embedding = None
        screenshot = self.environment.get_screenshot()
        if screenshot is not None:
            try:
                visual_embedding = self.visual_encoder.encode(screenshot)
                logger.debug(
                    f"game.learning_controller.perceive "
                    f"visual_embedding_dim={visual_embedding.dimensions}"
                )
            except Exception as e:
                logger.warning(
                    f"game.learning_controller.perceive visual_encode_error={e}"
                )

        logger.debug(
            f"game.learning_controller.perceive game_id={self.game_id} "
            f"turn={game_state.turn_number} has_visual={visual_embedding is not None}"
        )

        return game_state, visual_embedding

    async def select_action(
        self,
        game_state: GameState,
        visual_embedding: VisualEmbedding | None = None,
    ) -> str | None:
        """
        Select the best action based on current perception.

        MUST use self.reasoning_engine.select_best_move() for move selection.

        Args:
            game_state: Current game state
            visual_embedding: Optional visual embedding

        Returns:
            Selected move string or None if no valid move
        """
        if game_state.game_over:
            logger.debug(
                f"game.learning_controller.select_action game_over=True "
                f"game_id={self.game_id}"
            )
            return None

        if not game_state.legal_moves:
            logger.warning(
                f"game.learning_controller.select_action no_legal_moves "
                f"game_id={self.game_id}"
            )
            return None

        # Build query for memory retrieval
        query = f"game:{self.game_type} turn:{game_state.turn_number} "
        query += f"player:{game_state.current_player}"

        # Fuse modalities
        fused = self.fusion_layer.fuse(
            game_state=game_state.to_dict(),
            visual_embedding=visual_embedding,
            query=query,
        )

        # Retrieve memory nodes for context
        memory_nodes = self.memory_store.get_entangled_context(
            query=query, top_k=self.config.memory_top_k
        )

        # Build context summary for reasoning
        context_summary = self.fusion_layer.build_context_summary(
            game_state=game_state.to_dict(),
            fused=fused,
            memory_nodes=memory_nodes,
        )

        # MUST use reasoning_engine.select_best_move()
        selected_move = await self.reasoning_engine.select_best_move(
            context_summary=context_summary
        )

        logger.info(
            f"game.learning_controller.select_action game_id={self.game_id} "
            f"selected_move={selected_move}"
        )

        return selected_move

    async def execute_action(self, move: str) -> ActionResult:
        """
        Execute the selected action in the game environment.

        Args:
            move: The move to execute

        Returns:
            ActionResult containing success status and new state
        """
        action = {"type": "move", "move": move}
        result = self.environment.execute_action(action)

        logger.info(
            f"game.learning_controller.execute_action game_id={self.game_id} "
            f"move={move} success={result.success}"
        )

        return result

    async def observe_and_reflect(
        self,
        prev_state: GameState,
        move: str,
        result: ActionResult,
        new_state: GameState,
    ) -> None:
        """
        Observe the result and reflect on the action.

        MUST use self.strategy_memory.store_strategy() for storing strategies.

        Actions:
        1. Record move in learning_state
        2. Store strategy with tags
        3. Update entanglements via _update_node()
        4. On game end: update_als() with reflection

        Args:
            prev_state: State before the action
            move: The action that was taken
            result: Result of the action
            new_state: State after the action
        """
        # Calculate reward
        reward = result.reward
        if new_state.game_over:
            if new_state.winner == prev_state.current_player:
                reward += 1.0
            elif new_state.winner is not None:
                reward -= 1.0

        # Record move in learning state
        self.learning_state.record_move(
            move=move,
            reward=reward,
            state_before=prev_state.to_dict(),
            state_after=new_state.to_dict(),
        )

        # Build outcome dictionary
        outcome = {
            "success": result.success,
            "reward": reward,
            "game_over": new_state.game_over,
            "winner": new_state.winner,
        }

        # Determine result for tags
        result_tag = "in_progress"
        if new_state.game_over:
            if new_state.winner == prev_state.current_player:
                result_tag = "win"
            elif new_state.winner is not None:
                result_tag = "loss"
            else:
                result_tag = "draw"
            outcome["result"] = result_tag

        # Build tags
        tags = [
            f"turn:{prev_state.turn_number}",
            f"player:{prev_state.current_player}",
        ]

        # MUST use strategy_memory.store_strategy()
        if self.config.store_all_moves or new_state.game_over:
            node = self.strategy_memory.store_strategy(
                state=prev_state.to_dict(),
                move=move,
                outcome=outcome,
                game_type=self.game_type,
                tags=tags,
            )

            # Update entanglements with recent nodes
            if len(self.learning_state.move_history) > 1:
                # Link to previous move's strategy
                prev_move_data = self.learning_state.move_history[-2]
                if prev_move_data.get("strategy_node_id"):
                    self.strategy_memory.link_strategies(
                        node_id_1=node.id,
                        node_id_2=prev_move_data["strategy_node_id"],
                        strength=0.5,
                    )

            # Store node ID for entanglement tracking
            self.learning_state.move_history[-1]["strategy_node_id"] = node.id

            logger.debug(
                f"game.learning_controller.observe_and_reflect "
                f"stored_strategy node_id={node.id}"
            )

        # On game end, update ALS with reflection
        if new_state.game_over:
            # Capture move count before reset
            moves_this_game = len(self.learning_state.move_history)

            # Record game end in learning state (this clears move_history)
            self.learning_state.record_game_end(result_tag)

            # Generate reflection
            stats = self.learning_state.get_stats()
            reflection = (
                f"Game {self.game_type} ended: {result_tag}. "
                f"Total moves this game: {moves_this_game}. "
                f"Win rate: {stats['win_rate']:.2%}. "
                f"Episode: {stats['episode']}."
            )

            # Update ALS with focus and reflection
            self.memory_store.update_als(
                focus=f"Learning {self.game_type}",
                reflection=reflection,
            )

            logger.info(
                f"game.learning_controller.observe_and_reflect game_ended "
                f"game_id={self.game_id} result={result_tag} "
                f"win_rate={stats['win_rate']:.2%}"
            )
        else:
            logger.debug(
                f"game.learning_controller.observe_and_reflect "
                f"game_id={self.game_id} move={move} reward={reward:.3f}"
            )

    async def play_turn(self) -> tuple[str | None, ActionResult | None]:
        """
        Play a single turn: perceive -> select -> execute -> observe.

        Returns:
            Tuple of (move, ActionResult) or (None, None) if game over
        """
        # Perceive
        game_state, visual_embedding = await self.perceive()

        if game_state.game_over:
            logger.info(
                f"game.learning_controller.play_turn game_over=True "
                f"game_id={self.game_id}"
            )
            return None, None

        # Select action
        move = await self.select_action(game_state, visual_embedding)
        if move is None:
            logger.warning(
                f"game.learning_controller.play_turn no_move_selected "
                f"game_id={self.game_id}"
            )
            return None, None

        # Execute action
        result = await self.execute_action(move)

        # Get new state after action
        new_game_state, _ = await self.perceive()

        # Observe and reflect
        await self.observe_and_reflect(game_state, move, result, new_game_state)

        return move, result

    async def play_game(self, max_turns: int = 100) -> dict[str, Any]:
        """
        Play a complete game.

        Args:
            max_turns: Maximum number of turns before stopping

        Returns:
            Dictionary with game results and statistics
        """
        logger.info(
            f"game.learning_controller.play_game start "
            f"game_id={self.game_id} max_turns={max_turns}"
        )

        turns_played = 0
        last_result: ActionResult | None = None

        for turn in range(max_turns):
            move, result = await self.play_turn()

            if move is None or result is None:
                break

            last_result = result
            turns_played += 1

            if result.new_state and result.new_state.game_over:
                break

        # Get final state
        final_state = self.environment.get_state()

        # Compile results
        stats = self.learning_state.get_stats()
        game_result = {
            "game_id": self.game_id,
            "game_type": self.game_type,
            "turns_played": turns_played,
            "game_over": final_state.game_over,
            "winner": final_state.winner,
            "final_score": final_state.score,
            "learning_stats": stats,
        }

        logger.info(
            f"game.learning_controller.play_game complete "
            f"game_id={self.game_id} turns={turns_played} "
            f"winner={final_state.winner}"
        )

        return game_result

    def get_health_status(self) -> dict[str, Any]:
        """
        Get health status for this game controller.

        Returns:
            Dictionary with health information
        """
        stats = self.learning_state.get_stats()

        try:
            current_state = self.environment.get_state()
            game_over = current_state.game_over
            turn = current_state.turn_number
        except Exception:
            game_over = False
            turn = 0

        return {
            "status": "active",
            "game_id": self.game_id,
            "game_type": self.game_type,
            "game_over": game_over,
            "current_turn": turn,
            "episode": stats["episode"],
            "total_moves": stats["total_moves"],
            "win_rate": stats["win_rate"],
            "visual_encoder": type(self.visual_encoder).__name__,
            "model": self.model,
        }

    def __del__(self) -> None:
        """Cleanup when controller is destroyed."""
        try:
            unregister_game_controller(self.game_id)
        except Exception:
            pass  # Ignore errors during cleanup
