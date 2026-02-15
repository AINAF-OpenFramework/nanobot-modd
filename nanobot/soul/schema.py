"""Schema definitions for the Soul configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PersonalityTrait(BaseModel):
    """
    A personality trait that influences agent behavior and reasoning.

    Attributes:
        name: Unique identifier for the trait
        weight: Influence weight (0.0 to 2.0), defaults to 1.0
        description: Human-readable description of the trait
        affects: List of reasoning aspects this trait influences
    """

    name: str
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    description: str = ""
    affects: list[str] = Field(default_factory=list)


class Goal(BaseModel):
    """
    A goal that the agent strives to achieve during gameplay.

    Attributes:
        name: Unique identifier for the goal
        priority: Priority level (1-10), higher means more important
        description: Human-readable description of the goal
        actions: List of action types that align with this goal
    """

    name: str
    priority: int = Field(default=5, ge=1, le=10)
    description: str = ""
    actions: list[str] = Field(default_factory=list)


class Strategy(BaseModel):
    """
    A conditional strategy that modifies behavior based on game state.

    Attributes:
        name: Unique identifier for the strategy
        condition: Condition that activates this strategy (e.g., "early_game", "losing")
        approach: General approach type (e.g., "aggressive", "defensive")
        traits_boost: Trait name to boost value mapping when strategy is active
    """

    name: str
    condition: str
    approach: str = ""
    traits_boost: dict[str, float] = Field(default_factory=dict)


class GameConfig(BaseModel):
    """
    Configuration parameters for game reasoning.

    Attributes:
        default_reasoning_depth: Base depth for reasoning iterations
        monte_carlo_samples: Number of Monte Carlo samples for move evaluation
        beam_width: Width of beam search for move exploration
        risk_tolerance: Base risk tolerance (0.0 to 1.0)
    """

    default_reasoning_depth: int = Field(default=2, ge=1, le=10)
    monte_carlo_samples: int = Field(default=3, ge=1, le=20)
    beam_width: int = Field(default=4, ge=1, le=20)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)


class SoulConfig(BaseModel):
    """
    Complete soul configuration for the agent.

    Attributes:
        version: Configuration version string
        name: Name of the agent
        traits: List of personality traits
        goals: List of goals the agent strives for
        strategies: List of conditional strategies
        game: Game-specific configuration parameters
        game_strategies: Game type to strategy mapping for game-specific overrides
    """

    version: str = "1.0"
    name: str = "nanobot"
    traits: list[PersonalityTrait] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)
    strategies: list[Strategy] = Field(default_factory=list)
    game: GameConfig = Field(default_factory=GameConfig)
    game_strategies: dict[str, list[str]] = Field(default_factory=dict)
