"""Game reasoning engine that wraps the existing LatentReasoner."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from nanobot.agent.latent import LatentReasoner
from nanobot.agent.memory_types import SuperpositionalState
from nanobot.game.state_engine import GameStateEngine
from nanobot.providers.base import LLMProvider


class GameReasoningEngine:
    """
    Game reasoning engine that wraps the existing LatentReasoner.

    This engine provides game-specific reasoning capabilities by leveraging
    the underlying LatentReasoner for hypothesis generation and evaluation.

    CRITICAL: This class MUST use self.latent_reasoner.reason() for all
    reasoning operations - no separate reasoning logic should be implemented.

    Args:
        provider: LLM provider for reasoning operations
        model: Model identifier to use
        state_engine: GameStateEngine instance for game state access
        timeout_seconds: Timeout for reasoning operations
        memory_config: Configuration for the latent reasoner
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        state_engine: GameStateEngine,
        timeout_seconds: int = 10,
        memory_config: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.model = model
        self.state_engine = state_engine
        self.latent_reasoner = LatentReasoner(
            provider=provider,
            model=model,
            timeout_seconds=timeout_seconds,
            memory_config=memory_config or {},
        )
        logger.info(
            f"game.reasoning_engine.init game_id={state_engine.game_id} model={model}"
        )

    async def select_best_move(self, context_summary: str = "") -> str | None:
        """
        Select the best move based on the current game state.

        Uses the LatentReasoner to analyze the game state and select
        the optimal move from available legal moves.

        Args:
            context_summary: Additional context for the reasoning

        Returns:
            The selected best move, or None if no valid move found
        """
        state = self.state_engine.get_state()
        legal_moves = self.state_engine.get_legal_moves()

        if not legal_moves:
            logger.warning(
                f"game.reasoning_engine.select_best_move no_legal_moves "
                f"game_id={self.state_engine.game_id}"
            )
            return None

        user_message = self._build_move_selection_prompt(state, legal_moves)
        full_context = self._build_game_context(state, context_summary)

        reasoning_state = await self.latent_reasoner.reason(user_message, full_context)

        selected_move = self._extract_move_from_reasoning(reasoning_state, legal_moves)

        logger.info(
            f"game.reasoning_engine.select_best_move game_id={self.state_engine.game_id} "
            f"selected_move={selected_move} entropy={reasoning_state.entropy:.3f}"
        )
        return selected_move

    async def evaluate_moves(
        self,
        moves: list[str] | None = None,
        context_summary: str = "",
    ) -> dict[str, float]:
        """
        Evaluate all legal moves and return confidence scores.

        Uses the LatentReasoner to generate hypotheses about each move
        and returns confidence scores based on the reasoning.

        Args:
            moves: Optional list of moves to evaluate. If None, uses all legal moves.
            context_summary: Additional context for the evaluation

        Returns:
            Dictionary mapping moves to confidence scores (0.0 to 1.0)
        """
        state = self.state_engine.get_state()
        legal_moves = moves or self.state_engine.get_legal_moves()

        if not legal_moves:
            return {}

        user_message = self._build_evaluation_prompt(state, legal_moves)
        full_context = self._build_game_context(state, context_summary)

        reasoning_state = await self.latent_reasoner.reason(user_message, full_context)

        evaluations = self._extract_move_evaluations(reasoning_state, legal_moves)

        logger.info(
            f"game.reasoning_engine.evaluate_moves game_id={self.state_engine.game_id} "
            f"num_moves={len(evaluations)} entropy={reasoning_state.entropy:.3f}"
        )
        return evaluations

    async def simulate_future_states(
        self,
        depth: int = 2,
        context_summary: str = "",
    ) -> list[dict[str, Any]]:
        """
        Simulate future game states using reasoning to predict opponent moves.

        Uses the LatentReasoner to analyze potential future game states
        and returns a list of predicted state progressions.

        Args:
            depth: Number of moves to simulate ahead
            context_summary: Additional context for the simulation

        Returns:
            List of dictionaries containing simulated state progressions
        """
        state = self.state_engine.get_state()
        legal_moves = self.state_engine.get_legal_moves()

        if not legal_moves:
            return []

        simulations: list[dict[str, Any]] = []

        for move in legal_moves[:3]:  # Limit to top 3 moves for efficiency
            try:
                simulated_state = self.state_engine.simulate(move)

                user_message = self._build_simulation_prompt(
                    state, move, simulated_state, depth
                )
                full_context = self._build_game_context(state, context_summary)

                reasoning_state = await self.latent_reasoner.reason(
                    user_message, full_context
                )

                simulation = {
                    "initial_move": move,
                    "simulated_state": simulated_state,
                    "reasoning": reasoning_state.strategic_direction,
                    "entropy": reasoning_state.entropy,
                    "hypotheses": [
                        {
                            "intent": h.intent,
                            "confidence": h.confidence,
                            "reasoning": h.reasoning,
                        }
                        for h in reasoning_state.hypotheses
                    ],
                }
                simulations.append(simulation)

            except ValueError as e:
                logger.warning(
                    f"game.reasoning_engine.simulate_future_states "
                    f"error=simulation_failed move={move} error_msg={e}"
                )
                continue

        logger.info(
            f"game.reasoning_engine.simulate_future_states "
            f"game_id={self.state_engine.game_id} depth={depth} "
            f"num_simulations={len(simulations)}"
        )
        return simulations

    def _build_move_selection_prompt(
        self, state: dict[str, Any], legal_moves: list[str]
    ) -> str:
        """Build prompt for move selection."""
        return (
            f"You are playing a game. The current state is:\n{json.dumps(state, indent=2)}\n\n"
            f"Legal moves available: {', '.join(legal_moves)}\n\n"
            f"Analyze the position and select the best move. "
            f"Your intent should be the move you want to make."
        )

    def _build_evaluation_prompt(
        self, state: dict[str, Any], legal_moves: list[str]
    ) -> str:
        """Build prompt for move evaluation."""
        return (
            f"Evaluate the following moves for the current game state:\n"
            f"State: {json.dumps(state, indent=2)}\n\n"
            f"Moves to evaluate: {', '.join(legal_moves)}\n\n"
            f"For each move, analyze its strategic value. "
            f"Each hypothesis should represent a move and its confidence."
        )

    def _build_simulation_prompt(
        self,
        current_state: dict[str, Any],
        move: str,
        simulated_state: dict[str, Any],
        depth: int,
    ) -> str:
        """Build prompt for future state simulation."""
        return (
            f"Simulate the game {depth} moves ahead.\n"
            f"Current state: {json.dumps(current_state, indent=2)}\n"
            f"Move played: {move}\n"
            f"Resulting state: {json.dumps(simulated_state, indent=2)}\n\n"
            f"Predict the opponent's response and analyze likely outcomes."
        )

    def _build_game_context(
        self, state: dict[str, Any], additional_context: str
    ) -> str:
        """Build the full context summary for reasoning."""
        context_parts = [
            f"Game ID: {self.state_engine.game_id}",
            f"Current State: {json.dumps(state, indent=2)}",
        ]

        try:
            win_conditions = self.state_engine.check_win_conditions()
            context_parts.append(f"Win Conditions: {json.dumps(win_conditions)}")
        except ValueError:
            pass  # No rules configured

        if additional_context:
            context_parts.append(f"Additional Context: {additional_context}")

        return "\n".join(context_parts)

    def _extract_move_from_reasoning(
        self, reasoning_state: SuperpositionalState, legal_moves: list[str]
    ) -> str | None:
        """Extract the best move from reasoning hypotheses."""
        if not reasoning_state.hypotheses:
            # Fall back to first legal move if no hypotheses
            return legal_moves[0] if legal_moves else None

        # Look for a hypothesis that matches a legal move
        best_hypothesis = max(reasoning_state.hypotheses, key=lambda h: h.confidence)

        # Try to match the intent to a legal move
        intent = best_hypothesis.intent.strip().lower()

        for move in legal_moves:
            if move.lower() in intent or intent in move.lower():
                return move

        # If no match found, return the first legal move
        return legal_moves[0] if legal_moves else None

    def _extract_move_evaluations(
        self, reasoning_state: SuperpositionalState, legal_moves: list[str]
    ) -> dict[str, float]:
        """Extract move evaluations from reasoning hypotheses."""
        evaluations: dict[str, float] = {}

        # Initialize all moves with base confidence
        for move in legal_moves:
            evaluations[move] = 0.5

        # Update based on hypotheses
        for hypothesis in reasoning_state.hypotheses:
            intent = hypothesis.intent.strip().lower()
            for move in legal_moves:
                if move.lower() in intent or intent in move.lower():
                    evaluations[move] = max(evaluations[move], hypothesis.confidence)
                    break

        return evaluations
