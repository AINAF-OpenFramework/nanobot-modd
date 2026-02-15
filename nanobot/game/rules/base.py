"""Base utilities and helpers for game rules implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BoardPosition:
    """Represents a position on a game board."""

    row: int
    col: int

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple representation."""
        return (self.row, self.col)

    def __str__(self) -> str:
        """String representation (e.g., 'r0c1')."""
        return f"r{self.row}c{self.col}"

    @classmethod
    def from_string(cls, s: str) -> BoardPosition:
        """Parse from string format (e.g., 'r0c1')."""
        if not s.startswith("r"):
            raise ValueError(f"Invalid position string: {s}")
        parts = s[1:].split("c")
        if len(parts) != 2:
            raise ValueError(f"Invalid position string: {s}")
        return cls(row=int(parts[0]), col=int(parts[1]))


@dataclass
class BoardState:
    """Generic board state representation."""

    board: list[list[str]]
    current_player: str
    move_count: int = 0

    @property
    def rows(self) -> int:
        """Number of rows in the board."""
        return len(self.board)

    @property
    def cols(self) -> int:
        """Number of columns in the board."""
        return len(self.board[0]) if self.board else 0

    def get_cell(self, row: int, col: int) -> str:
        """Get value at board position."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.board[row][col]
        raise ValueError(f"Position ({row}, {col}) out of bounds")

    def set_cell(self, row: int, col: int, value: str) -> None:
        """Set value at board position."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.board[row][col] = value
        else:
            raise ValueError(f"Position ({row}, {col}) out of bounds")

    def is_empty(self, row: int, col: int) -> bool:
        """Check if a position is empty."""
        return self.get_cell(row, col) == ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "board": self.board,
            "current_player": self.current_player,
            "move_count": self.move_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoardState:
        """Create from dictionary representation."""
        return cls(
            board=data["board"],
            current_player=data["current_player"],
            move_count=data.get("move_count", 0),
        )
