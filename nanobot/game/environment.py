"""Game environment adapter for unified game interaction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from loguru import logger

# Optional dependencies
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore

try:
    import pyautogui

    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    pyautogui = None  # type: ignore

try:
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.remote.webdriver import WebDriver

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    ActionChains = None  # type: ignore
    WebDriver = None  # type: ignore


@dataclass
class GameState:
    """
    Represents the current state of a game.

    Args:
        board: The game board representation (varies by game type)
        current_player: Identifier of the player to move
        turn_number: Current turn number
        legal_moves: List of valid moves in current state
        game_over: Whether the game has ended
        winner: The winner, if game is over
        score: Current score or evaluation
        metadata: Additional game-specific metadata
        timestamp: When this state was captured
    """

    board: Any
    current_player: str
    turn_number: int = 0
    legal_moves: list[str] = field(default_factory=list)
    game_over: bool = False
    winner: str | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the game state to a dictionary.

        Returns:
            Dictionary representation of the game state
        """
        return {
            "board": self.board,
            "current_player": self.current_player,
            "turn_number": self.turn_number,
            "legal_moves": self.legal_moves,
            "game_over": self.game_over,
            "winner": self.winner,
            "score": self.score,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ActionResult:
    """
    Result of executing an action in the game environment.

    Args:
        success: Whether the action was executed successfully
        new_state: The resulting game state, if successful
        reward: Reward signal from the action
        message: Human-readable message about the result
        error: Error message if action failed
    """

    success: bool
    new_state: GameState | None = None
    reward: float = 0.0
    message: str = ""
    error: str | None = None


class GameEnvironmentAdapter(ABC):
    """
    Abstract base class for game environment adapters.

    Provides a unified interface for interacting with different game types,
    whether through APIs, GUIs, or other interfaces.

    Args:
        game_id: Unique identifier for the game session
        game_type: Type of game (e.g., "tictactoe", "chess")
    """

    def __init__(self, game_id: str, game_type: str):
        self.game_id = game_id
        self.game_type = game_type
        logger.info(
            f"game.environment.adapter.init game_id={game_id} game_type={game_type}"
        )

    @abstractmethod
    def get_state(self) -> GameState:
        """
        Get the current game state.

        Returns:
            Current GameState
        """
        ...

    def get_screenshot(self) -> Any | None:
        """
        Get a screenshot of the current game display.

        Returns:
            PIL Image if available, else None
        """
        return None

    @abstractmethod
    def execute_action(self, action: dict[str, Any]) -> ActionResult:
        """
        Execute an action in the game.

        Args:
            action: Action to execute (format depends on game type)

        Returns:
            ActionResult containing success status and new state
        """
        ...


class APIGameAdapter(GameEnvironmentAdapter):
    """
    Game environment adapter for API-based games.

    Uses callable functions to interact with game APIs.

    Args:
        game_id: Unique identifier for the game session
        game_type: Type of game
        state_getter: Callable that returns current game state as dict
        action_executor: Callable that executes actions and returns result dict
        screenshot_getter: Optional callable to get screenshot
    """

    def __init__(
        self,
        game_id: str,
        game_type: str,
        state_getter: Callable[[], dict[str, Any]],
        action_executor: Callable[[dict[str, Any]], dict[str, Any]],
        screenshot_getter: Callable[[], Any] | None = None,
    ):
        super().__init__(game_id, game_type)
        self._state_getter = state_getter
        self._action_executor = action_executor
        self._screenshot_getter = screenshot_getter
        logger.debug(f"game.environment.api_adapter.init game_id={game_id}")

    def get_state(self) -> GameState:
        """
        Get the current game state from the API.

        Returns:
            Current GameState
        """
        try:
            state_dict = self._state_getter()
            game_state = GameState(
                board=state_dict.get("board"),
                current_player=state_dict.get("current_player", ""),
                turn_number=state_dict.get("turn_number", 0),
                legal_moves=state_dict.get("legal_moves", []),
                game_over=state_dict.get("game_over", False),
                winner=state_dict.get("winner"),
                score=state_dict.get("score", 0.0),
                metadata=state_dict.get("metadata", {}),
            )
            logger.debug(
                f"game.environment.api_adapter.get_state game_id={self.game_id} "
                f"turn={game_state.turn_number}"
            )
            return game_state
        except Exception as e:
            logger.error(
                f"game.environment.api_adapter.get_state error={e} "
                f"game_id={self.game_id}"
            )
            raise

    def get_screenshot(self) -> Any | None:
        """
        Get a screenshot from the API if available.

        Returns:
            Screenshot image or None
        """
        if self._screenshot_getter:
            try:
                return self._screenshot_getter()
            except Exception as e:
                logger.warning(
                    f"game.environment.api_adapter.get_screenshot error={e} "
                    f"game_id={self.game_id}"
                )
        return None

    def execute_action(self, action: dict[str, Any]) -> ActionResult:
        """
        Execute an action through the API.

        Args:
            action: Action to execute

        Returns:
            ActionResult with execution outcome
        """
        try:
            result_dict = self._action_executor(action)
            new_state = None

            if result_dict.get("success", False) and "new_state" in result_dict:
                new_state_dict = result_dict["new_state"]
                new_state = GameState(
                    board=new_state_dict.get("board"),
                    current_player=new_state_dict.get("current_player", ""),
                    turn_number=new_state_dict.get("turn_number", 0),
                    legal_moves=new_state_dict.get("legal_moves", []),
                    game_over=new_state_dict.get("game_over", False),
                    winner=new_state_dict.get("winner"),
                    score=new_state_dict.get("score", 0.0),
                    metadata=new_state_dict.get("metadata", {}),
                )

            result = ActionResult(
                success=result_dict.get("success", False),
                new_state=new_state,
                reward=result_dict.get("reward", 0.0),
                message=result_dict.get("message", ""),
                error=result_dict.get("error"),
            )

            logger.info(
                f"game.environment.api_adapter.execute_action game_id={self.game_id} "
                f"action={action} success={result.success}"
            )
            return result

        except Exception as e:
            logger.error(
                f"game.environment.api_adapter.execute_action error={e} "
                f"game_id={self.game_id} action={action}"
            )
            return ActionResult(
                success=False,
                error=str(e),
                message=f"Action execution failed: {e}",
            )


class GUIGameAdapter(GameEnvironmentAdapter):
    """
    Game environment adapter for GUI-based games.

    Uses PyAutoGUI or Selenium for GUI automation.

    Args:
        game_id: Unique identifier for the game session
        game_type: Type of game
        state_parser: Callable that parses game state from screenshot
        use_selenium: Whether to use Selenium (default: use PyAutoGUI)
        selenium_driver: Selenium WebDriver instance if using Selenium
        game_window_region: Screen region tuple (x, y, width, height) for game window
    """

    def __init__(
        self,
        game_id: str,
        game_type: str,
        state_parser: Callable[[Any], dict[str, Any]],
        use_selenium: bool = False,
        selenium_driver: Any | None = None,
        game_window_region: tuple[int, int, int, int] | None = None,
    ):
        super().__init__(game_id, game_type)
        self._state_parser = state_parser
        self._use_selenium = use_selenium
        self._driver = selenium_driver
        self._region = game_window_region
        self._last_screenshot: Any | None = None

        # Validate dependencies
        if use_selenium and not SELENIUM_AVAILABLE:
            logger.warning(
                f"game.environment.gui_adapter.init selenium_not_available "
                f"game_id={game_id}"
            )
        elif not use_selenium and not PYAUTOGUI_AVAILABLE:
            logger.warning(
                f"game.environment.gui_adapter.init pyautogui_not_available "
                f"game_id={game_id}"
            )

        logger.debug(
            f"game.environment.gui_adapter.init game_id={game_id} "
            f"use_selenium={use_selenium}"
        )

    def get_state(self) -> GameState:
        """
        Get the current game state by capturing and parsing the screen.

        Returns:
            Current GameState parsed from screenshot
        """
        try:
            screenshot = self._capture_screenshot()
            self._last_screenshot = screenshot

            state_dict = self._state_parser(screenshot)
            game_state = GameState(
                board=state_dict.get("board"),
                current_player=state_dict.get("current_player", ""),
                turn_number=state_dict.get("turn_number", 0),
                legal_moves=state_dict.get("legal_moves", []),
                game_over=state_dict.get("game_over", False),
                winner=state_dict.get("winner"),
                score=state_dict.get("score", 0.0),
                metadata=state_dict.get("metadata", {}),
            )

            logger.debug(
                f"game.environment.gui_adapter.get_state game_id={self.game_id} "
                f"turn={game_state.turn_number}"
            )
            return game_state

        except Exception as e:
            logger.error(
                f"game.environment.gui_adapter.get_state error={e} "
                f"game_id={self.game_id}"
            )
            raise

    def get_screenshot(self) -> Any | None:
        """
        Get the most recent screenshot.

        Returns:
            Last captured screenshot or capture new one
        """
        if self._last_screenshot is not None:
            return self._last_screenshot
        try:
            return self._capture_screenshot()
        except Exception as e:
            logger.warning(
                f"game.environment.gui_adapter.get_screenshot error={e} "
                f"game_id={self.game_id}"
            )
            return None

    def execute_action(self, action: dict[str, Any]) -> ActionResult:
        """
        Execute an action through GUI automation.

        Args:
            action: Action to execute. Supported formats:
                - {"type": "click", "x": int, "y": int}
                - {"type": "keypress", "key": str}
                - {"type": "type", "text": str}

        Returns:
            ActionResult with execution outcome
        """
        try:
            action_type = action.get("type", "")

            if self._use_selenium and SELENIUM_AVAILABLE:
                self._execute_selenium_action(action)
            elif PYAUTOGUI_AVAILABLE:
                self._execute_pyautogui_action(action)
            else:
                return ActionResult(
                    success=False,
                    error="No GUI automation library available",
                    message="Neither PyAutoGUI nor Selenium is available",
                )

            # Get new state after action
            new_state = self.get_state()

            result = ActionResult(
                success=True,
                new_state=new_state,
                message=f"Executed {action_type} action",
            )

            logger.info(
                f"game.environment.gui_adapter.execute_action game_id={self.game_id} "
                f"action_type={action_type} success=True"
            )
            return result

        except Exception as e:
            logger.error(
                f"game.environment.gui_adapter.execute_action error={e} "
                f"game_id={self.game_id} action={action}"
            )
            return ActionResult(
                success=False,
                error=str(e),
                message=f"GUI action failed: {e}",
            )

    def _capture_screenshot(self) -> Any:
        """Capture a screenshot of the game window."""
        if self._use_selenium and SELENIUM_AVAILABLE and self._driver:
            # Get screenshot from Selenium
            png_data = self._driver.get_screenshot_as_png()
            if PIL_AVAILABLE:
                import io

                return Image.open(io.BytesIO(png_data))
            return png_data

        elif PYAUTOGUI_AVAILABLE:
            # Use PyAutoGUI to capture screen region
            if self._region:
                return pyautogui.screenshot(region=self._region)
            return pyautogui.screenshot()

        raise RuntimeError("No screenshot method available")

    def _execute_pyautogui_action(self, action: dict[str, Any]) -> None:
        """
        Execute action using PyAutoGUI.

        Args:
            action: Action dictionary with type and parameters
        """
        if not PYAUTOGUI_AVAILABLE:
            raise RuntimeError("PyAutoGUI not available")

        action_type = action.get("type", "")

        if action_type == "click":
            x = action.get("x", 0)
            y = action.get("y", 0)
            button = action.get("button", "left")
            pyautogui.click(x=x, y=y, button=button)
            logger.debug(f"game.environment.gui.pyautogui click x={x} y={y}")

        elif action_type == "keypress":
            key = action.get("key", "")
            pyautogui.press(key)
            logger.debug(f"game.environment.gui.pyautogui keypress key={key}")

        elif action_type == "type":
            text = action.get("text", "")
            interval = action.get("interval", 0.05)
            pyautogui.typewrite(text, interval=interval)
            logger.debug(f"game.environment.gui.pyautogui type text_len={len(text)}")

        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def _execute_selenium_action(self, action: dict[str, Any]) -> None:
        """
        Execute action using Selenium ActionChains.

        Args:
            action: Action dictionary with type and parameters
        """
        if not SELENIUM_AVAILABLE or not self._driver:
            raise RuntimeError("Selenium not available or driver not set")

        action_type = action.get("type", "")
        action_chain = ActionChains(self._driver)

        if action_type == "click":
            x = action.get("x", 0)
            y = action.get("y", 0)
            # Use offset from current position
            action_chain.move_by_offset(x, y).click().perform()
            logger.debug(f"game.environment.gui.selenium click x={x} y={y}")

        elif action_type == "keypress":
            key = action.get("key", "")
            action_chain.send_keys(key).perform()
            logger.debug(f"game.environment.gui.selenium keypress key={key}")

        elif action_type == "type":
            text = action.get("text", "")
            action_chain.send_keys(text).perform()
            logger.debug(f"game.environment.gui.selenium type text_len={len(text)}")

        else:
            raise ValueError(f"Unknown action type: {action_type}")
