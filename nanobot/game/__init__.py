"""Game cognition stack for Nanobot."""

from nanobot.game.action_executor import GameActionExecutor
from nanobot.game.environment import (
    ActionResult,
    APIGameAdapter,
    GameEnvironmentAdapter,
    GameState,
    GUIGameAdapter,
)
from nanobot.game.fusion import FusedEmbedding, FusionConfig, MultimodalFusionLayer
from nanobot.game.health import (
    extend_health_payload,
    get_game_health_status,
    register_game_controller,
    unregister_game_controller,
)
from nanobot.game.interface import GameObservation, inject_into_context, parse_observation
from nanobot.game.learning_controller import (
    GameLearningController,
    LearningConfig,
    LearningState,
)
from nanobot.game.reasoning_engine import GameReasoningEngine
from nanobot.game.state_engine import GameRules, GameStateEngine
from nanobot.game.strategy_memory import StrategyMemory
from nanobot.game.visual_perception import (
    EfficientNetEncoder,
    LightweightCNNEncoder,
    SimpleGridEncoder,
    VisualEmbedding,
    VisualEncoder,
    create_encoder,
)

__all__ = [
    # Interface
    "GameObservation",
    "parse_observation",
    "inject_into_context",
    # State engine
    "GameStateEngine",
    "GameRules",
    # Reasoning
    "GameReasoningEngine",
    # Memory
    "StrategyMemory",
    # Action
    "GameActionExecutor",
    # Environment (new)
    "GameState",
    "ActionResult",
    "GameEnvironmentAdapter",
    "APIGameAdapter",
    "GUIGameAdapter",
    # Visual perception (new)
    "VisualEmbedding",
    "VisualEncoder",
    "LightweightCNNEncoder",
    "EfficientNetEncoder",
    "SimpleGridEncoder",
    "create_encoder",
    # Fusion (new)
    "FusedEmbedding",
    "FusionConfig",
    "MultimodalFusionLayer",
    # Learning controller (new)
    "LearningConfig",
    "LearningState",
    "GameLearningController",
    # Health (new)
    "register_game_controller",
    "unregister_game_controller",
    "get_game_health_status",
    "extend_health_payload",
]
