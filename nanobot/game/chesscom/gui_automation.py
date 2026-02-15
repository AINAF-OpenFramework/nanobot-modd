"""GUI automation module for executing chess moves on Chess.com."""

from __future__ import annotations

import random
import time

from loguru import logger

try:
    import pyautogui
except ImportError:
    pyautogui = None


class ChessComAutomation:
    """
    Executes chess moves via mouse automation.

    Implements human-like timing, randomized delays, and bezier curve
    mouse movements to avoid detection.
    """

    # Chess board square mapping
    FILES = "abcdefgh"
    RANKS = "12345678"

    def __init__(
        self,
        board_region: tuple[int, int, int, int],
        min_move_delay: float = 0.5,
        max_move_delay: float = 2.0,
        human_like: bool = True,
    ):
        """
        Initialize GUI automation for Chess.com.

        Args:
            board_region: (x, y, width, height) of the chess board
            min_move_delay: Minimum delay before making a move (seconds)
            max_move_delay: Maximum delay before making a move (seconds)
            human_like: Enable human-like mouse movements and timing
        """
        if pyautogui is None:
            raise ImportError(
                "pyautogui is required for GUI automation. "
                "Install with: pip install 'nanobot-ai[chesscom]'"
            )

        self.board_region = board_region
        self.min_move_delay = min_move_delay
        self.max_move_delay = max_move_delay
        self.human_like = human_like

        # Safety: Set fail-safe to allow emergency stop
        pyautogui.FAILSAFE = True

        logger.debug(
            f"ChessComAutomation initialized: region={board_region}, "
            f"delays=({min_move_delay}, {max_move_delay}), human_like={human_like}"
        )

    def execute_move(
        self, move: str, board_orientation: str = "white"
    ) -> bool:
        """
        Execute a chess move on Chess.com.

        Args:
            move: Move in UCI notation (e.g., "e2e4", "e7e8q" for promotion)
            board_orientation: "white" or "black" (affects square coordinates)

        Returns:
            True if move was executed successfully

        Example:
            >>> automation = ChessComAutomation((100, 100, 600, 600))
            >>> automation.execute_move("e2e4", "white")
            True
        """
        try:
            # Parse move
            if len(move) < 4:
                logger.error(f"Invalid move format: {move}")
                return False

            from_square = move[:2]
            to_square = move[2:4]
            promotion = move[4:5] if len(move) > 4 else None

            # Random delay before move
            if self.human_like:
                self.random_delay()

            # Get screen coordinates
            from_coords = self.square_to_screen_coords(from_square, board_orientation)
            to_coords = self.square_to_screen_coords(to_square, board_orientation)

            # Execute move
            if self.human_like:
                self.human_like_mouse_move(from_coords, from_coords)
            else:
                pyautogui.moveTo(from_coords[0], from_coords[1])

            # Click and drag to target square
            pyautogui.mouseDown()
            time.sleep(0.05 + random.uniform(0, 0.1))

            if self.human_like:
                self.human_like_mouse_move(from_coords, to_coords)
            else:
                pyautogui.moveTo(to_coords[0], to_coords[1])

            pyautogui.mouseUp()

            # Handle promotion if needed
            if promotion:
                time.sleep(0.2)
                self.handle_promotion(promotion)

            logger.info(f"Executed move: {move} (orientation={board_orientation})")
            return True

        except Exception as e:
            logger.error(f"Failed to execute move {move}: {e}")
            return False

    def square_to_screen_coords(
        self, square: str, orientation: str
    ) -> tuple[int, int]:
        """
        Convert chess square (e.g., 'e4') to screen coordinates.

        Args:
            square: Chess square in algebraic notation (e.g., "e4")
            orientation: "white" or "black"

        Returns:
            Tuple of (x, y) screen coordinates
        """
        if len(square) != 2:
            raise ValueError(f"Invalid square format: {square}")

        file_char = square[0]
        rank_char = square[1]

        if file_char not in self.FILES or rank_char not in self.RANKS:
            raise ValueError(f"Invalid square: {square}")

        file_idx = self.FILES.index(file_char)
        rank_idx = self.RANKS.index(rank_char)

        # Flip coordinates if playing as black
        if orientation == "black":
            file_idx = 7 - file_idx
            rank_idx = 7 - rank_idx

        # Calculate screen coordinates
        x, y, width, height = self.board_region
        square_width = width / 8
        square_height = height / 8

        # Calculate center of square
        screen_x = int(x + (file_idx + 0.5) * square_width)
        screen_y = int(y + (7 - rank_idx + 0.5) * square_height)

        return (screen_x, screen_y)

    def human_like_mouse_move(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> None:
        """
        Move mouse with human-like bezier curve path.

        Args:
            start: Starting (x, y) coordinates
            end: Ending (x, y) coordinates
        """
        # Generate bezier curve points
        control1 = (
            start[0] + random.randint(-50, 50),
            start[1] + random.randint(-50, 50),
        )
        control2 = (
            end[0] + random.randint(-50, 50),
            end[1] + random.randint(-50, 50),
        )

        # Simplified bezier curve (would use more sophisticated path in production)
        steps = random.randint(20, 40)
        duration = random.uniform(0.3, 0.6)

        for i in range(steps + 1):
            t = i / steps
            # Cubic bezier formula
            x = (
                (1 - t) ** 3 * start[0]
                + 3 * (1 - t) ** 2 * t * control1[0]
                + 3 * (1 - t) * t**2 * control2[0]
                + t**3 * end[0]
            )
            y = (
                (1 - t) ** 3 * start[1]
                + 3 * (1 - t) ** 2 * t * control1[1]
                + 3 * (1 - t) * t**2 * control2[1]
                + t**3 * end[1]
            )

            pyautogui.moveTo(int(x), int(y), duration=duration / steps)
            time.sleep(0.001)

    def random_delay(self) -> None:
        """Add randomized delay to simulate human thinking time."""
        delay = random.uniform(self.min_move_delay, self.max_move_delay)
        logger.debug(f"Waiting {delay:.2f}s before move")
        time.sleep(delay)

    def handle_promotion(self, piece: str) -> None:
        """
        Handle pawn promotion UI selection.

        Args:
            piece: Piece to promote to ("q", "r", "b", "n")
        """
        # Placeholder: Would click on promotion UI
        # Chess.com typically shows a popup with piece choices
        logger.debug(f"Handling promotion to {piece} (placeholder)")

        # In production, this would:
        # 1. Wait for promotion UI to appear
        # 2. Detect piece positions in UI
        # 3. Click on the appropriate piece
        time.sleep(0.3)
