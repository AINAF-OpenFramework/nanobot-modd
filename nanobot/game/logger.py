"""Game logging for move tracking and observability."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class MoveLogEntry(BaseModel):
    """
    A structured log entry for a single game move.

    Contains comprehensive information about the move including reasoning
    metrics, trait applications, and prediction outcomes.
    """

    game_id: str
    turn_number: int
    timestamp: datetime = Field(default_factory=datetime.now)
    move: str
    player: str
    reasoning_depth: int = 0
    entropy: float = 0.0
    monte_carlo_samples: int = 0
    traits_applied: dict[str, float] = Field(default_factory=dict)
    strategy_used: str | None = None
    goal_alignment: list[str] = Field(default_factory=list)
    predicted_outcome: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    reasoning_time_ms: int = 0

    def to_json(self) -> str:
        """
        Serialize entry to JSON string.

        Returns:
            JSON string representation
        """
        return self.model_dump_json()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert entry to dictionary.

        Returns:
            Dictionary representation
        """
        data = self.model_dump()
        # Convert datetime to ISO format string
        data["timestamp"] = self.timestamp.isoformat()
        return data


class GameLogger:
    """
    Logger for game moves and outcomes in JSONL format.

    Provides structured logging of game moves with reasoning metrics
    and session summary capabilities.
    """

    def __init__(
        self,
        game_id: str | None = None,
        log_dir: Path | None = None,
    ):
        """
        Initialize the GameLogger.

        Args:
            game_id: Unique identifier for the game session
            log_dir: Directory for log files (optional)
        """
        self._game_id = game_id or str(uuid.uuid4())
        self._log_dir = log_dir
        self._log_file: Path | None = None
        self._entries: list[MoveLogEntry] = []
        self._outcomes: dict[int, dict[str, Any]] = {}
        self._session_start = datetime.now()

        if log_dir:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_dir / f"game_{self._game_id}.jsonl"

        logger.info(f"game.logger.init game_id={self._game_id}")

    @property
    def game_id(self) -> str:
        """Get the game ID."""
        return self._game_id

    def log_move(
        self,
        turn_number: int,
        move: str,
        player: str,
        reasoning_depth: int = 0,
        entropy: float = 0.0,
        monte_carlo_samples: int = 0,
        traits_applied: dict[str, float] | None = None,
        strategy_used: str | None = None,
        goal_alignment: list[str] | None = None,
        predicted_outcome: dict[str, Any] | None = None,
        confidence: float = 0.0,
        reasoning_time_ms: int = 0,
    ) -> MoveLogEntry:
        """
        Log a game move with full reasoning metrics.

        Args:
            turn_number: The turn number in the game
            move: The move that was played
            player: The player who made the move
            reasoning_depth: Depth of reasoning used
            entropy: Entropy of the hypothesis distribution
            monte_carlo_samples: Number of MC samples used
            traits_applied: Dictionary of trait names to applied weights
            strategy_used: Name of the active strategy (if any)
            goal_alignment: List of goals this move aligns with
            predicted_outcome: Predicted outcome dictionary
            confidence: Confidence in the move (0.0 to 1.0)
            reasoning_time_ms: Time spent reasoning in milliseconds

        Returns:
            The created MoveLogEntry
        """
        entry = MoveLogEntry(
            game_id=self._game_id,
            turn_number=turn_number,
            move=move,
            player=player,
            reasoning_depth=reasoning_depth,
            entropy=entropy,
            monte_carlo_samples=monte_carlo_samples,
            traits_applied=traits_applied or {},
            strategy_used=strategy_used,
            goal_alignment=goal_alignment or [],
            predicted_outcome=predicted_outcome or {},
            confidence=confidence,
            reasoning_time_ms=reasoning_time_ms,
        )

        self._entries.append(entry)

        # Write to file if configured
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")

        logger.info(
            f"game.logger.log_move game_id={self._game_id} turn={turn_number} "
            f"move={move} player={player} confidence={confidence:.3f}"
        )

        return entry

    def log_outcome(
        self,
        turn: int,
        actual_outcome: dict[str, Any],
    ) -> None:
        """
        Log the actual outcome for a turn.

        Used for comparing predictions to actual results.

        Args:
            turn: The turn number
            actual_outcome: The actual outcome dictionary
        """
        self._outcomes[turn] = actual_outcome

        # Write outcome to log file
        if self._log_file:
            outcome_entry = {
                "type": "outcome",
                "game_id": self._game_id,
                "turn": turn,
                "actual_outcome": actual_outcome,
                "timestamp": datetime.now().isoformat(),
            }
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(outcome_entry) + "\n")

        logger.debug(
            f"game.logger.log_outcome game_id={self._game_id} turn={turn} "
            f"outcome={actual_outcome}"
        )

    def get_session_summary(self) -> dict[str, Any]:
        """
        Get a summary of the game session.

        Returns:
            Dictionary containing session statistics
        """
        if not self._entries:
            return {
                "game_id": self._game_id,
                "total_moves": 0,
                "duration_seconds": 0,
                "session_start": self._session_start.isoformat(),
            }

        # Calculate statistics
        total_moves = len(self._entries)
        avg_confidence = sum(e.confidence for e in self._entries) / total_moves
        avg_entropy = sum(e.entropy for e in self._entries) / total_moves
        avg_reasoning_time = sum(e.reasoning_time_ms for e in self._entries) / total_moves
        total_reasoning_time = sum(e.reasoning_time_ms for e in self._entries)

        # Trait usage statistics
        trait_usage: dict[str, int] = {}
        for entry in self._entries:
            for trait_name in entry.traits_applied:
                trait_usage[trait_name] = trait_usage.get(trait_name, 0) + 1

        # Strategy usage statistics
        strategy_usage: dict[str, int] = {}
        for entry in self._entries:
            if entry.strategy_used:
                strategy_usage[entry.strategy_used] = (
                    strategy_usage.get(entry.strategy_used, 0) + 1
                )

        # Calculate duration
        session_end = datetime.now()
        duration = (session_end - self._session_start).total_seconds()

        # Count prediction accuracy if we have outcomes
        correct_predictions = 0
        total_predictions = 0
        for entry in self._entries:
            if entry.turn_number in self._outcomes:
                total_predictions += 1
                actual = self._outcomes[entry.turn_number]
                predicted = entry.predicted_outcome
                # Simple check - could be made more sophisticated
                if predicted.get("result") == actual.get("result"):
                    correct_predictions += 1

        summary = {
            "game_id": self._game_id,
            "total_moves": total_moves,
            "duration_seconds": duration,
            "session_start": self._session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "average_confidence": round(avg_confidence, 3),
            "average_entropy": round(avg_entropy, 3),
            "average_reasoning_time_ms": round(avg_reasoning_time, 1),
            "total_reasoning_time_ms": total_reasoning_time,
            "trait_usage": trait_usage,
            "strategy_usage": strategy_usage,
            "prediction_accuracy": (
                round(correct_predictions / total_predictions, 3)
                if total_predictions > 0
                else None
            ),
        }

        logger.info(
            f"game.logger.get_session_summary game_id={self._game_id} "
            f"total_moves={total_moves} duration={duration:.1f}s"
        )

        return summary

    def get_entries(self) -> list[MoveLogEntry]:
        """Get all logged entries."""
        return list(self._entries)

    def clear(self) -> None:
        """Clear all logged entries."""
        self._entries.clear()
        self._outcomes.clear()
