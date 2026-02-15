"""Trait-based scoring for hypothesis evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.soul.loader import SoulLoader
from nanobot.soul.schema import Strategy


class TraitScorer:
    """
    Scores hypotheses based on soul personality traits and goals.

    This class AUGMENTS the existing hypothesis scoring from LatentReasoner,
    it does not replace it. The score modifications are applied as multipliers
    to the base score.

    CRITICAL: Does not replace existing reasoning logic. Only augments scores.
    """

    def __init__(self, soul_loader: SoulLoader):
        """
        Initialize the TraitScorer.

        Args:
            soul_loader: SoulLoader instance for accessing soul configuration
        """
        self._soul_loader = soul_loader

    @classmethod
    def from_workspace(cls, workspace: Path) -> "TraitScorer":
        """
        Create a TraitScorer from a workspace path.

        Args:
            workspace: Path to the workspace directory

        Returns:
            A new TraitScorer instance
        """
        soul_loader = SoulLoader.get_instance(workspace)
        return cls(soul_loader)

    def score_hypothesis(
        self,
        hypothesis: str,
        base_score: float,
        game_state: dict[str, Any] | None = None,
    ) -> float:
        """
        Score a hypothesis based on soul traits, goals, and active strategy.

        AUGMENTS the base_score with trait and goal modifiers. Does not replace
        the original scoring logic.

        Args:
            hypothesis: The hypothesis text (intent description)
            base_score: The base confidence score from LatentReasoner
            game_state: Optional game state for strategy detection

        Returns:
            Modified score bounded to [0.0, 1.0]
        """
        config = self._soul_loader.load()
        game_state = game_state or {}

        # Start with base score
        modified_score = base_score

        # Apply trait weights
        trait_modifier = self._calculate_trait_modifier(hypothesis, config.traits)
        modified_score *= trait_modifier

        # Apply goal alignment boost
        goal_boost = self._calculate_goal_boost(hypothesis, config.goals)
        modified_score += goal_boost

        # Apply active strategy modifiers
        active_strategy = self._detect_active_strategy(game_state, config.strategies)
        if active_strategy:
            strategy_modifier = self._calculate_strategy_modifier(
                hypothesis, active_strategy, config.traits
            )
            modified_score *= strategy_modifier

        # Bound the score
        final_score = max(0.0, min(1.0, modified_score))

        logger.debug(
            f"soul.traits.score_hypothesis base={base_score:.3f} "
            f"trait_mod={trait_modifier:.3f} goal_boost={goal_boost:.3f} "
            f"final={final_score:.3f}"
        )

        return final_score

    def get_reasoning_depth(self, game_state: dict[str, Any] | None = None) -> int:
        """
        Get the reasoning depth based on traits and game state.

        Uses the 'analytical' trait to modify the base reasoning depth.

        Args:
            game_state: Optional game state for context

        Returns:
            The recommended reasoning depth
        """
        config = self._soul_loader.load()
        base_depth = config.game.default_reasoning_depth

        # Check for analytical trait
        analytical_weight = 1.0
        for trait in config.traits:
            if trait.name == "analytical" and "reasoning_depth" in trait.affects:
                analytical_weight = trait.weight
                break

        # Apply analytical modifier
        modified_depth = int(base_depth * analytical_weight)

        # Check for active strategy that might affect depth
        if game_state:
            active_strategy = self._detect_active_strategy(game_state, config.strategies)
            if active_strategy and active_strategy.traits_boost.get("analytical"):
                boost = active_strategy.traits_boost["analytical"]
                modified_depth = int(modified_depth * (1 + boost))

        # Bound between 1 and 10
        final_depth = max(1, min(10, modified_depth))

        logger.debug(
            f"soul.traits.get_reasoning_depth base={base_depth} final={final_depth}"
        )

        return final_depth

    def get_monte_carlo_samples(self, game_state: dict[str, Any] | None = None) -> int:
        """
        Get the number of Monte Carlo samples based on configuration.

        Args:
            game_state: Optional game state for context

        Returns:
            The recommended number of Monte Carlo samples
        """
        config = self._soul_loader.load()
        base_samples = config.game.monte_carlo_samples

        # Check for efficient trait (reduces samples to save resources)
        efficient_modifier = 1.0
        for trait in config.traits:
            if trait.name == "efficient":
                # Efficient trait reduces samples but maintains quality
                efficient_modifier = max(0.7, 1.0 / trait.weight)
                break

        modified_samples = int(base_samples * efficient_modifier)

        # Bound between 1 and 20
        final_samples = max(1, min(20, modified_samples))

        logger.debug(
            f"soul.traits.get_monte_carlo_samples base={base_samples} final={final_samples}"
        )

        return final_samples

    def get_risk_tolerance(self, game_state: dict[str, Any] | None = None) -> float:
        """
        Get the risk tolerance based on traits and game state.

        Args:
            game_state: Optional game state for context

        Returns:
            Risk tolerance value between 0.0 and 1.0
        """
        config = self._soul_loader.load()
        base_tolerance = config.game.risk_tolerance

        # Check for cautious trait
        for trait in config.traits:
            if trait.name == "cautious" and "risk_tolerance" in trait.affects:
                # Cautious trait reduces risk tolerance
                base_tolerance *= (1.0 / trait.weight)
                break

        # Check for active strategy
        if game_state:
            active_strategy = self._detect_active_strategy(game_state, config.strategies)
            if active_strategy:
                if active_strategy.approach == "aggressive":
                    base_tolerance *= 1.3
                elif active_strategy.approach == "defensive":
                    base_tolerance *= 0.7

        return max(0.0, min(1.0, base_tolerance))

    def _calculate_trait_modifier(
        self,
        hypothesis: str,
        traits: list,
    ) -> float:
        """Calculate trait-based score modifier."""
        hypothesis_lower = hypothesis.lower()
        modifier = 1.0

        for trait in traits:
            # Check if hypothesis aligns with trait
            trait_name_lower = trait.name.lower()
            if trait_name_lower in hypothesis_lower:
                # Apply trait weight as modifier
                modifier *= trait.weight

        return modifier

    def _calculate_goal_boost(
        self,
        hypothesis: str,
        goals: list,
    ) -> float:
        """Calculate goal-based score boost."""
        hypothesis_lower = hypothesis.lower()
        boost = 0.0

        for goal in goals:
            # Check if hypothesis mentions goal-aligned actions
            for action in goal.actions:
                if action.lower() in hypothesis_lower:
                    # Boost based on goal priority (normalized to 0.0-0.1 range)
                    boost += (goal.priority / 100.0)
                    break  # Only count each goal once

        return boost

    def _detect_active_strategy(
        self,
        game_state: dict[str, Any],
        strategies: list[Strategy],
    ) -> Strategy | None:
        """Detect which strategy should be active based on game state."""
        # Check for common game state indicators
        turn = game_state.get("turn", 0)
        score_diff = game_state.get("score_diff", 0)
        game_phase = game_state.get("phase", "")

        for strategy in strategies:
            condition = strategy.condition.lower()

            # Early game detection
            if condition == "early_game":
                if turn <= 3 or game_phase == "opening":
                    return strategy

            # Losing detection
            if condition == "losing":
                if score_diff < 0 or game_phase == "losing":
                    return strategy

            # Endgame detection
            if condition == "endgame":
                if game_phase == "endgame" or turn >= 20:
                    return strategy

            # Direct phase match
            if condition == game_phase:
                return strategy

        return None

    def _calculate_strategy_modifier(
        self,
        hypothesis: str,
        strategy: Strategy,
        traits: list,
    ) -> float:
        """Calculate strategy-based score modifier."""
        modifier = 1.0

        # Apply trait boosts from strategy
        for trait_name, boost in strategy.traits_boost.items():
            # Find the trait
            for trait in traits:
                if trait.name == trait_name:
                    # Apply boosted weight
                    modifier *= (trait.weight + boost)
                    break

        # Apply approach-based modifier
        hypothesis_lower = hypothesis.lower()
        if strategy.approach == "aggressive":
            if any(word in hypothesis_lower for word in ["attack", "aggressive", "win", "capture"]):
                modifier *= 1.2
        elif strategy.approach == "defensive":
            if any(word in hypothesis_lower for word in ["defend", "block", "protect", "safe"]):
                modifier *= 1.2

        return modifier
