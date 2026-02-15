"""Screen capture module for Chess.com game area."""

from __future__ import annotations

import numpy as np
from loguru import logger

try:
    import mss
except ImportError:
    mss = None


class ChessComScreenCapture:
    """
    Captures screenshots of the Chess.com game area.

    Supports configurable screen regions, continuous capture mode,
    turn detection, and multi-monitor setups.
    """

    def __init__(self, region: tuple[int, int, int, int] | None = None):
        """
        Initialize screen capture for Chess.com.

        Args:
            region: Optional (x, y, width, height) tuple for capture area.
                   If None, auto-detect the chess board.
        """
        if mss is None:
            raise ImportError(
                "mss is required for screen capture. "
                "Install with: pip install 'nanobot-ai[chesscom]'"
            )

        self.sct = mss.mss()
        self._region = region
        self._board_region: tuple[int, int, int, int] | None = None

        logger.debug(f"ChessComScreenCapture initialized with region={region}")

    def capture_board(self) -> np.ndarray:
        """
        Capture the current chess board from screen.

        Returns:
            NumPy array containing the board image (RGB format)

        Raises:
            RuntimeError: If board region cannot be determined
        """
        if self._board_region is None and self._region is None:
            logger.info("Auto-detecting board region...")
            self._board_region = self.detect_board_region()

        region = self._board_region or self._region
        if region is None:
            raise RuntimeError("Unable to determine board region")

        # Capture screen region using mss
        x, y, width, height = region
        monitor = {"top": y, "left": x, "width": width, "height": height}

        screenshot = self.sct.grab(monitor)
        # Convert to numpy array (RGB)
        img = np.array(screenshot)

        # mss returns BGRA, convert to RGB
        if img.shape[-1] == 4:
            img = img[:, :, [2, 1, 0]]  # BGRA to RGB

        logger.debug(f"Captured board image: shape={img.shape}")
        return img

    def detect_board_region(self) -> tuple[int, int, int, int]:
        """
        Auto-detect the chess board region on screen.

        Returns:
            Tuple of (x, y, width, height) for the board region

        Note:
            This is a placeholder implementation. In production, this would
            use image processing to detect Chess.com's board layout.
            For now, it returns a default region for the primary monitor.
        """
        # Get primary monitor info
        monitors = self.sct.monitors
        if len(monitors) > 1:
            primary = monitors[1]  # Index 0 is all monitors combined
        else:
            raise RuntimeError("No monitors detected")

        # Default to center region (placeholder for actual detection)
        width = 600
        height = 600
        x = (primary["width"] - width) // 2
        y = (primary["height"] - height) // 2

        logger.info(f"Detected board region: x={x}, y={y}, w={width}, h={height}")
        return (x, y, width, height)

    def is_my_turn(self) -> bool:
        """
        Detect if it's currently the player's turn.

        Returns:
            True if it's the player's turn, False otherwise

        Note:
            This is a placeholder implementation. In production, this would
            analyze the captured image for turn indicators (e.g., highlighted
            board border, move timer, etc.)
        """
        # Placeholder: Would analyze captured image for turn indicators
        logger.debug("Checking if it's my turn (placeholder)")
        return True

    def get_game_status(self) -> str:
        """
        Get current game status.

        Returns:
            Game status: "playing", "ended", or "waiting"

        Note:
            This is a placeholder implementation. In production, this would
            detect game end screens, waiting lobbies, etc.
        """
        # Placeholder: Would analyze screen for game state
        logger.debug("Getting game status (placeholder)")
        return "playing"

    def close(self) -> None:
        """Close the screen capture session."""
        if hasattr(self, "sct"):
            self.sct.close()
            logger.debug("Screen capture session closed")
