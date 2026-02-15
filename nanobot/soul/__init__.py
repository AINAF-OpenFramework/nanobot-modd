"""Soul configuration module for Nanobot personality and behavior."""

from nanobot.soul.loader import SoulLoader
from nanobot.soul.schema import (
    GameConfig,
    Goal,
    PersonalityTrait,
    SoulConfig,
    Strategy,
)
from nanobot.soul.traits import TraitScorer

__all__ = [
    "SoulLoader",
    "SoulConfig",
    "PersonalityTrait",
    "Goal",
    "Strategy",
    "GameConfig",
    "TraitScorer",
]
