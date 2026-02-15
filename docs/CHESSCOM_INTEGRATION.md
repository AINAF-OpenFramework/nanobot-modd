# Chess.com Integration Documentation

## Overview

The Chess.com integration enables TanyalahD VTuber AI to autonomously play chess on Chess.com with real-time commentary. This integration combines computer vision, move generation, VTuber personality, and GUI automation.

## Architecture

The system consists of five main modules:

1. **Screen Capture** (`screen_capture.py`) - Captures the Chess.com board from your screen
2. **Board Recognition** (`board_recognition.py`) - Detects pieces and converts to FEN notation
3. **GUI Automation** (`gui_automation.py`) - Executes moves with human-like mouse movements
4. **TTS Integration** (`tts_integration.py`) - Converts commentary to speech
5. **Client Orchestrator** (`client.py`) - Coordinates all components

## Installation

### Basic Installation

```bash
pip install nanobot-ai[chesscom]
```

This installs the required dependencies:
- `pyautogui` - Mouse and keyboard automation
- `opencv-python` - Computer vision for board recognition
- `mss` - Fast screen capture
- `httpx` - HTTP client for TTS APIs
- `sounddevice` & `soundfile` - Audio playback

### From Source

```bash
git clone https://github.com/HKUDS/nanobot.git
cd nanobot
pip install -e ".[chesscom]"
```

## Quick Start

### Manual Mode (Move Suggestions Only)

```python
import asyncio
from nanobot.game.chesscom import ChessComClient

async def main():
    client = ChessComClient(
        personality="TanyalahD",
        enable_tts=True,
        auto_play=False,  # Manual mode
    )
    
    await client.start_game_loop()

asyncio.run(main())
```

### Auto-Play Mode

```python
import asyncio
from nanobot.game.chesscom import ChessComClient

async def main():
    client = ChessComClient(
        personality="TanyalahD",
        enable_tts=True,
        human_like_play=True,
        auto_play=True,  # Automatically execute moves
    )
    
    await client.start_game_loop()

asyncio.run(main())
```

### Using the Demo Script

```bash
# Manual mode with TTS
python examples/chesscom_vtuber_demo.py

# Auto-play mode with TTS
python examples/chesscom_vtuber_demo.py

# Manual mode without TTS
python examples/chesscom_vtuber_demo.py --no-tts --manual
```

## Configuration

### Board Region

By default, the system auto-detects the chess board. For better performance, you can specify the exact region:

```python
from nanobot.game.chesscom import ChessComScreenCapture

# Specify board region (x, y, width, height)
capture = ChessComScreenCapture(region=(100, 100, 600, 600))
```

To find your board region:
1. Take a screenshot of Chess.com
2. Use an image editor to note the pixel coordinates
3. The board should be square (same width and height)

### TTS Configuration

#### Using Local TTS (Default)

```python
from nanobot.game.chesscom import TTSIntegration

tts = TTSIntegration(provider="local")
```

#### Using ElevenLabs

```python
import os
os.environ["ELEVENLABS_API_KEY"] = "your-api-key"

from nanobot.game.chesscom import TTSIntegration

tts = TTSIntegration(
    provider="elevenlabs",
    voice_id="your-voice-id"
)
```

#### Using Google Cloud TTS

```python
import os
os.environ["GOOGLE_TTS_API_KEY"] = "your-api-key"

from nanobot.game.chesscom import TTSIntegration

tts = TTSIntegration(provider="google")
```

### Move Timing

Adjust timing parameters to control how quickly moves are made:

```python
from nanobot.game.chesscom import ChessComAutomation

automation = ChessComAutomation(
    board_region=(100, 100, 600, 600),
    min_move_delay=1.0,  # Minimum 1 second delay
    max_move_delay=3.0,  # Maximum 3 seconds delay
    human_like=True,     # Enable human-like movements
)
```

## Features

### Screen Capture
- Auto-detection of chess board region
- Multi-monitor support
- Continuous capture mode
- Turn detection (placeholder - requires implementation)

### Board Recognition
- FEN string conversion
- Piece detection (placeholder - uses starting position)
- Orientation detection (white/black)
- Last move detection (placeholder)

**Note:** The current implementation uses a placeholder that returns the starting position. For production use, you'll need to implement actual piece detection using template matching or a CNN model.

### GUI Automation
- Human-like mouse movements using Bezier curves
- Randomized delays to simulate thinking
- Drag-and-drop move execution
- Pawn promotion handling (placeholder)
- Safety limits to avoid detection

### TTS Integration
- Multiple provider support (ElevenLabs, Google, local)
- Audio caching for common phrases
- Async audio generation
- Non-blocking playback

### VTuber Personality
- Integration with TanyalahD personality system
- Context-aware commentary
- Emotion-based expressions
- Move evaluation feedback

## Safety Considerations

### Rate Limiting

The system includes built-in safety measures:

- **Minimum move delay:** 0.5 seconds (configurable)
- **Human-like timing:** Randomized delays between moves
- **Bezier curve movements:** Non-linear mouse paths
- **Fail-safe:** Move mouse to corner to emergency stop (pyautogui feature)

### Detection Avoidance

To minimize the risk of detection:

1. **Enable human-like play:**
   ```python
   client = ChessComClient(human_like_play=True)
   ```

2. **Use randomized delays:**
   ```python
   automation = ChessComAutomation(
       min_move_delay=1.0,
       max_move_delay=5.0,
   )
   ```

3. **Session limits:** Take breaks between games

4. **Manual oversight:** Use manual mode to review suggested moves

### Terms of Service

**Important:** Using automation on Chess.com may violate their Terms of Service. This integration is provided for educational and research purposes only. Use at your own risk.

## Troubleshooting

### Board Not Detected

If auto-detection fails:

1. Ensure Chess.com is visible on screen
2. Use a standard board size (not too small)
3. Specify the region manually:
   ```python
   capture = ChessComScreenCapture(region=(x, y, width, height))
   ```

### Moves Not Executing

Check:

1. Board region is correct
2. Chess.com window is active and in focus
3. No dialogs or popups blocking the board
4. Sufficient delay between moves

### TTS Not Working

Verify:

1. Audio output is enabled
2. `sounddevice` and `soundfile` are installed
3. API keys are set (if using external providers)
4. Check volume settings

### Import Errors

Install the optional dependencies:

```bash
pip install 'nanobot-ai[chesscom]'
```

Or install individually:

```bash
pip install pyautogui opencv-python mss httpx sounddevice soundfile
```

## Advanced Usage

### Custom Piece Templates

To improve board recognition accuracy:

1. Capture piece images for your theme
2. Save them in `nanobot/game/chesscom/assets/piece_templates/`
3. Follow the naming convention in the README

### Integrating with Strategy Memory

```python
from nanobot.agent.memory import MemoryStore
from nanobot.game.strategy_memory import StrategyMemory
from nanobot.game.chesscom import ChessComClient

# Initialize memory
memory_store = MemoryStore()
strategy_memory = StrategyMemory(memory_store)

# Use with client
client = ChessComClient()
# Strategy memory is automatically used by the chess engines
```

### Custom Personalities

Extend the VTuber personality:

```python
from nanobot.game.vtuber.soul_layer import SoulLayerIntegration

class CustomPersonality(SoulLayerIntegration):
    def __init__(self):
        super().__init__(personality="Custom")
        # Add custom response templates
        self.responses["custom_opening"] = [
            "Let's try something creative!",
            "Time for an unorthodox opening!",
        ]
```

### Monitoring Metrics

Track performance with IAS/CER metrics:

```python
client = ChessComClient()
# ... play some games ...

stats = client.get_statistics()
print(f"Win Rate: {stats['win_rate']:.1%}")
print(f"Avg IAS: {stats['avg_ias']:.3f}")
print(f"Avg CER: {stats['avg_cer']:.3f}")
```

## API Reference

### ChessComClient

Main orchestrator for Chess.com integration.

```python
ChessComClient(
    personality: str = "TanyalahD",
    enable_tts: bool = True,
    human_like_play: bool = True,
    auto_play: bool = False,
)
```

**Methods:**
- `start_game_loop()` - Start the autonomous game loop
- `process_turn()` - Process a single turn
- `pause()` - Pause auto-play
- `resume()` - Resume auto-play
- `get_statistics()` - Get game statistics

### ChessComScreenCapture

Captures screenshots of the Chess.com board.

```python
ChessComScreenCapture(region: tuple[int, int, int, int] | None = None)
```

**Methods:**
- `capture_board()` - Capture current board state
- `detect_board_region()` - Auto-detect board location
- `is_my_turn()` - Check if it's player's turn (placeholder)
- `get_game_status()` - Get game status (placeholder)
- `close()` - Cleanup resources

### BoardRecognizer

Recognizes pieces and converts to FEN.

```python
BoardRecognizer(piece_theme: str = "default")
```

**Methods:**
- `recognize_pieces(board_image)` - Detect pieces from image
- `to_fen(board_state, turn)` - Convert to FEN string
- `detect_orientation(board_image)` - Detect player color
- `detect_last_move(board_image)` - Detect highlighted move

### ChessComAutomation

Executes moves via GUI automation.

```python
ChessComAutomation(
    board_region: tuple[int, int, int, int],
    min_move_delay: float = 0.5,
    max_move_delay: float = 2.0,
    human_like: bool = True,
)
```

**Methods:**
- `execute_move(move, board_orientation)` - Execute a move
- `square_to_screen_coords(square, orientation)` - Convert square to coordinates
- `human_like_mouse_move(start, end)` - Move mouse with bezier curve
- `random_delay()` - Add randomized delay
- `handle_promotion(piece)` - Handle pawn promotion

### TTSIntegration

Text-to-speech integration.

```python
TTSIntegration(
    provider: str = "elevenlabs",
    voice_id: str | None = None,
    api_key: str | None = None,
    cache_enabled: bool = True,
)
```

**Methods:**
- `speak(text, block)` - Generate and play speech
- `generate_audio(text)` - Generate audio from text
- `play_audio(audio_data)` - Play audio data
- `stop()` - Stop playback

## Contributing

To improve the Chess.com integration:

1. **Board Recognition:** Implement actual piece detection using template matching or CNNs
2. **Turn Detection:** Add logic to detect when it's the player's turn
3. **Game Status:** Implement detection of game end conditions
4. **Piece Templates:** Create template sets for popular Chess.com themes
5. **Testing:** Add integration tests with actual Chess.com boards

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Disclaimer

This software is provided for educational and research purposes only. Using automation on Chess.com may violate their Terms of Service. The authors are not responsible for any consequences of using this software.
