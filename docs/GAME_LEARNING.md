# Game Learning Layer

The Game Learning Layer provides a comprehensive framework for building AI agents that can learn to play games through visual perception, multimodal fusion, and strategy memory.

## Overview: 6-Step Learning Loop

The learning process follows a continuous loop:

1. **Perceive** - Capture game state and screenshot
2. **Encode** - Transform visual input into embeddings
3. **Fuse** - Combine state, visual, and memory representations
4. **Reason** - Select best action using LLM reasoning
5. **Execute** - Perform action in game environment
6. **Reflect** - Store strategy and update learning state

```
┌────────────────────────────────────────────────────────────────────┐
│                    GAME LEARNING LOOP                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ PERCEIVE │──▶│  ENCODE  │──▶│   FUSE   │──▶│  REASON  │        │
│  │          │   │ (Visual) │   │(Multimod)│   │  (LLM)   │        │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘        │
│       ▲                                            │               │
│       │                                            ▼               │
│  ┌────┴─────┐                               ┌──────────┐          │
│  │ REFLECT  │◀──────────────────────────────│ EXECUTE  │          │
│  │(Strategy)│                               │ (Action) │          │
│  └──────────┘                               └──────────┘          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   GameLearningController                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────┐              │
│  │  GameEnvironment     │    │   VisualEncoder      │              │
│  │  Adapter             │    │   (MobileNet/Grid)   │              │
│  │  - APIGameAdapter    │    └──────────────────────┘              │
│  │  - GUIGameAdapter    │              │                           │
│  └──────────────────────┘              ▼                           │
│            │                 ┌──────────────────────┐              │
│            ▼                 │  MultimodalFusion    │              │
│  ┌──────────────────────┐    │  Layer               │              │
│  │  GameStateEngine     │───▶│  - State encoding    │              │
│  │  (Rules + History)   │    │  - Visual embedding  │              │
│  └──────────────────────┘    │  - Memory context    │              │
│                              └──────────────────────┘              │
│                                        │                           │
│                                        ▼                           │
│  ┌──────────────────────┐    ┌──────────────────────┐              │
│  │  StrategyMemory      │◀───│  GameReasoning       │              │
│  │  (MemoryStore)       │    │  Engine              │              │
│  │  - Store strategies  │    │  (LatentReasoner)    │              │
│  │  - Retrieve context  │    └──────────────────────┘              │
│  └──────────────────────┘                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### Environment Adapters

#### APIGameAdapter
For games with programmatic APIs:

```python
from nanobot.game import APIGameAdapter, GameState

def get_game_state() -> dict:
    return {
        "board": game.get_board(),
        "current_player": game.current_player,
        "legal_moves": game.get_legal_moves(),
    }

def execute_move(action: dict) -> dict:
    result = game.make_move(action["move"])
    return {
        "success": True,
        "new_state": get_game_state(),
        "reward": result.reward,
    }

adapter = APIGameAdapter(
    game_id="game-001",
    game_type="chess",
    state_getter=get_game_state,
    action_executor=execute_move,
)
```

#### GUIGameAdapter
For games requiring screen capture:

```python
from nanobot.game import GUIGameAdapter

def parse_screenshot(image) -> dict:
    # Computer vision logic to extract game state
    return {"board": [...], "current_player": "X", ...}

adapter = GUIGameAdapter(
    game_id="game-001",
    game_type="solitaire",
    state_parser=parse_screenshot,
    game_window_region=(100, 100, 800, 600),
)
```

### Visual Perception

```python
from nanobot.game import create_encoder, SimpleGridEncoder

# Auto-select best available encoder
encoder = create_encoder(encoder_type="auto", embedding_dim=256)

# Or explicitly use grid encoder (no PyTorch required)
encoder = SimpleGridEncoder(embedding_dim=128, grid_size=(8, 8))

# Encode a screenshot
screenshot = game.get_screenshot()
embedding = encoder.encode(screenshot)
print(f"Embedding shape: {embedding.embedding.shape}")
print(f"Confidence: {embedding.confidence}")
```

### Multimodal Fusion

```python
from nanobot.game import FusionConfig, MultimodalFusionLayer
from nanobot.agent.memory import MemoryStore

# Configure fusion weights
config = FusionConfig(
    total_dim=512,
    state_dim=128,
    visual_dim=256,
    memory_dim=128,
    state_weight=0.4,
    visual_weight=0.3,
    memory_weight=0.3,
)

# Create fusion layer
memory_store = MemoryStore(workspace_path)
fusion = MultimodalFusionLayer(config=config, memory_store=memory_store)

# Fuse modalities
fused = fusion.fuse(
    game_state={"board": [...], "current_player": "X"},
    visual_embedding=visual_embedding,
    query="strategy game:tictactoe",
)

# Build context for LLM
context = fusion.build_context_summary(game_state, fused, memory_nodes)
```

### Learning Controller

```python
from pathlib import Path
from nanobot.game import GameLearningController, LearningConfig
from nanobot.providers import create_provider

# Configuration
config = LearningConfig(
    visual_encoder_type="auto",
    visual_embedding_dim=256,
    reasoning_timeout=10,
    memory_top_k=5,
    store_all_moves=True,
)

# Create controller
controller = GameLearningController(
    provider=create_provider("openai"),
    model="gpt-4",
    environment=adapter,
    workspace=Path("./workspace"),
    config=config,
    rules=TicTacToeRules(),
)

# Play a single turn
move, result = await controller.play_turn()

# Play a full game
game_result = await controller.play_game(max_turns=100)
print(f"Winner: {game_result['winner']}")
print(f"Win rate: {game_result['learning_stats']['win_rate']:.2%}")
```

## Configuration

```json
{
  "game_learning": {
    "visual_encoder_type": "auto",
    "visual_embedding_dim": 256,
    "visual_device": "cpu",
    "fusion_weights": [0.4, 0.3, 0.3],
    "reasoning_timeout": 10,
    "memory_top_k": 5,
    "store_all_moves": true,
    "log_screenshots": false
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `visual_encoder_type` | string | `"auto"` | Encoder type: `auto`, `mobilenet`, `efficientnet`, `grid` |
| `visual_embedding_dim` | int | `256` | Dimension of visual embeddings |
| `visual_device` | string | `"cpu"` | PyTorch device for neural network encoders |
| `fusion_weights` | tuple | `(0.4, 0.3, 0.3)` | Weights for (state, visual, memory) |
| `reasoning_timeout` | int | `10` | Timeout in seconds for LLM reasoning |
| `memory_top_k` | int | `5` | Number of memory nodes to retrieve |
| `store_all_moves` | bool | `true` | Store all moves or only game-ending ones |
| `log_screenshots` | bool | `false` | Save screenshots to memory |

## Health Endpoint

The game learning layer integrates with health monitoring:

```python
from nanobot.game import get_game_health_status, extend_health_payload

# Get game-specific health status
status = get_game_health_status()
# {
#     "active_games": 2,
#     "games": {
#         "game-001": {
#             "status": "active",
#             "game_type": "tictactoe",
#             "current_turn": 5,
#             "win_rate": 0.65
#         },
#         "game-002": {...}
#     }
# }

# Extend main health endpoint
health_payload = {"service": "nanobot", "status": "healthy"}
extended = extend_health_payload(health_payload)
```

### Health Response Schema

```json
{
  "service": "nanobot",
  "status": "healthy",
  "games": {
    "active_games": 1,
    "games": {
      "tictactoe-001": {
        "status": "active",
        "game_id": "tictactoe-001",
        "game_type": "tictactoe",
        "game_over": false,
        "current_turn": 5,
        "episode": 3,
        "total_moves": 45,
        "win_rate": 0.667,
        "visual_encoder": "SimpleGridEncoder",
        "model": "gpt-4"
      }
    }
  }
}
```

## Example: TicTacToe Bot

Complete example of a TicTacToe learning agent:

```python
import asyncio
from pathlib import Path
from typing import Any

from nanobot.game import (
    APIGameAdapter,
    GameLearningController,
    LearningConfig,
)
from nanobot.game.state_engine import GameRules
from nanobot.providers import create_provider


class TicTacToeRules(GameRules):
    """TicTacToe game rules implementation."""

    def get_legal_moves(self, state: dict[str, Any]) -> list[str]:
        board = state.get("board", [""] * 9)
        return [str(i) for i in range(9) if board[i] == ""]

    def apply_move(self, state: dict[str, Any], move: str) -> dict[str, Any]:
        new_state = state.copy()
        board = new_state.get("board", [""] * 9).copy()
        pos = int(move)
        board[pos] = new_state.get("current_player", "X")
        new_state["board"] = board
        new_state["turn_number"] = new_state.get("turn_number", 0) + 1
        # Switch player
        curr = new_state.get("current_player", "X")
        new_state["current_player"] = "O" if curr == "X" else "X"
        return new_state

    def check_win_conditions(self, state: dict[str, Any]) -> dict[str, Any]:
        board = state.get("board", [""] * 9)
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Cols
            [0, 4, 8], [2, 4, 6],  # Diagonals
        ]
        for pattern in win_patterns:
            cells = [board[i] for i in pattern]
            if cells[0] != "" and cells[0] == cells[1] == cells[2]:
                return {"game_over": True, "winner": cells[0]}
        if "" not in board:
            return {"game_over": True, "winner": None}  # Draw
        return {"game_over": False, "winner": None}

    def get_next_player(self, state: dict[str, Any]) -> str:
        return "O" if state.get("current_player") == "X" else "X"


class TicTacToeGame:
    """Simple TicTacToe game implementation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [""] * 9
        self.current_player = "X"
        self.turn = 0
        self.game_over = False
        self.winner = None

    def get_state(self) -> dict[str, Any]:
        legal_moves = [str(i) for i in range(9) if self.board[i] == ""]
        return {
            "board": self.board.copy(),
            "current_player": self.current_player,
            "turn_number": self.turn,
            "legal_moves": legal_moves,
            "game_over": self.game_over,
            "winner": self.winner,
        }

    def execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        move = int(action.get("move", -1))
        if move < 0 or move > 8 or self.board[move] != "":
            return {"success": False, "error": "Invalid move"}

        self.board[move] = self.current_player
        self.turn += 1

        # Check win
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6],
        ]
        for pattern in win_patterns:
            if all(self.board[i] == self.current_player for i in pattern):
                self.game_over = True
                self.winner = self.current_player
                break

        # Check draw
        if not self.game_over and "" not in self.board:
            self.game_over = True

        # Switch player
        if not self.game_over:
            self.current_player = "O" if self.current_player == "X" else "X"

        return {
            "success": True,
            "new_state": self.get_state(),
            "reward": 1.0 if self.winner else 0.0,
        }


async def main():
    # Initialize game
    game = TicTacToeGame()

    # Create environment adapter
    adapter = APIGameAdapter(
        game_id="tictactoe-001",
        game_type="tictactoe",
        state_getter=game.get_state,
        action_executor=game.execute_action,
    )

    # Create learning controller
    config = LearningConfig(
        visual_encoder_type="grid",
        reasoning_timeout=15,
        store_all_moves=True,
    )

    controller = GameLearningController(
        provider=create_provider("openai"),
        model="gpt-4",
        environment=adapter,
        workspace=Path("./workspace"),
        config=config,
        rules=TicTacToeRules(),
    )

    # Play multiple games
    for episode in range(10):
        game.reset()
        result = await controller.play_game(max_turns=9)
        print(f"Episode {episode + 1}: Winner = {result['winner']}")

    # Print final stats
    stats = controller.learning_state.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Episodes: {stats['episode']}")
    print(f"  Win Rate: {stats['win_rate']:.2%}")
    print(f"  Total Moves: {stats['total_moves']}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Dependencies

### Required
- `numpy` - Numerical computing
- `Pillow` - Image processing
- `loguru` - Logging
- `pydantic` - Data validation

### Optional
- `torch` + `torchvision` - Neural network visual encoders (MobileNet, EfficientNet)
- `pyautogui` - GUI automation for screen capture
- `selenium` - Web-based game automation

Install optional dependencies as needed:

```bash
# For neural network visual encoders
pip install torch torchvision

# For GUI automation
pip install pyautogui

# For web game automation
pip install selenium
```

## Key Integration Points

The game learning layer integrates with existing nanobot components:

| Component | Integration |
|-----------|-------------|
| `GameReasoningEngine.select_best_move()` | Used for move selection via LatentReasoner |
| `StrategyMemory.store_strategy()` | Used to store game strategies |
| `MemoryStore.get_entangled_context()` | Used for memory retrieval in fusion |
| `MemoryStore._update_node()` | Used to update strategy entanglements |
| `MemoryStore.update_als()` | Used to update Active Learning State on game end |
