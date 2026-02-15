"""Tests for game logger module."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from nanobot.game.logger import GameLogger, MoveLogEntry


class TestMoveLogEntry:
    """Tests for MoveLogEntry class."""

    def test_create_entry(self):
        """Test creating a move log entry."""
        entry = MoveLogEntry(
            game_id="test-game",
            turn_number=5,
            move="e2e4",
            player="white",
            reasoning_depth=3,
            entropy=0.5,
            confidence=0.8,
        )

        assert entry.game_id == "test-game"
        assert entry.turn_number == 5
        assert entry.move == "e2e4"
        assert entry.player == "white"
        assert entry.reasoning_depth == 3
        assert entry.entropy == 0.5
        assert entry.confidence == 0.8
        assert isinstance(entry.timestamp, datetime)

    def test_to_json(self):
        """Test serializing entry to JSON."""
        entry = MoveLogEntry(
            game_id="test-game",
            turn_number=1,
            move="a1",
            player="X",
        )

        json_str = entry.to_json()
        data = json.loads(json_str)

        assert data["game_id"] == "test-game"
        assert data["turn_number"] == 1
        assert data["move"] == "a1"
        assert data["player"] == "X"

    def test_to_dict(self):
        """Test converting entry to dictionary."""
        entry = MoveLogEntry(
            game_id="test-game",
            turn_number=1,
            move="a1",
            player="X",
            traits_applied={"analytical": 1.5},
            goal_alignment=["win_game"],
        )

        data = entry.to_dict()

        assert isinstance(data, dict)
        assert data["game_id"] == "test-game"
        assert data["traits_applied"] == {"analytical": 1.5}
        assert data["goal_alignment"] == ["win_game"]
        assert isinstance(data["timestamp"], str)  # Should be ISO format

    def test_default_values(self):
        """Test default values for optional fields."""
        entry = MoveLogEntry(
            game_id="test",
            turn_number=0,
            move="m1",
            player="p1",
        )

        assert entry.reasoning_depth == 0
        assert entry.entropy == 0.0
        assert entry.monte_carlo_samples == 0
        assert entry.traits_applied == {}
        assert entry.strategy_used is None
        assert entry.goal_alignment == []
        assert entry.predicted_outcome == {}
        assert entry.confidence == 0.0
        assert entry.reasoning_time_ms == 0


class TestGameLogger:
    """Tests for GameLogger class."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def game_logger(self):
        """Create a game logger without file output."""
        return GameLogger(game_id="test-game")

    @pytest.fixture
    def file_logger(self, temp_log_dir):
        """Create a game logger with file output."""
        return GameLogger(game_id="test-game", log_dir=temp_log_dir)

    def test_init(self, game_logger):
        """Test logger initialization."""
        assert game_logger.game_id == "test-game"

    def test_init_generates_game_id(self):
        """Test logger generates game_id if not provided."""
        logger = GameLogger()
        assert logger.game_id is not None
        assert len(logger.game_id) > 0

    def test_log_move_creates_entry(self, game_logger):
        """Test log_move creates and returns entry."""
        entry = game_logger.log_move(
            turn_number=1,
            move="a1",
            player="X",
            reasoning_depth=2,
            entropy=0.3,
            confidence=0.9,
        )

        assert isinstance(entry, MoveLogEntry)
        assert entry.game_id == "test-game"
        assert entry.turn_number == 1
        assert entry.move == "a1"
        assert entry.player == "X"

    def test_log_move_stores_entry(self, game_logger):
        """Test log_move stores entry internally."""
        game_logger.log_move(turn_number=1, move="a1", player="X")
        game_logger.log_move(turn_number=2, move="b2", player="O")

        entries = game_logger.get_entries()
        assert len(entries) == 2

    def test_log_move_writes_to_file(self, file_logger, temp_log_dir):
        """Test log_move writes to JSONL file."""
        file_logger.log_move(turn_number=1, move="a1", player="X")
        file_logger.log_move(turn_number=2, move="b2", player="O")

        log_file = temp_log_dir / f"game_{file_logger.game_id}.jsonl"
        assert log_file.exists()

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

        # Verify content
        entry1 = json.loads(lines[0])
        assert entry1["turn_number"] == 1
        assert entry1["move"] == "a1"

    def test_log_move_with_all_fields(self, game_logger):
        """Test log_move with all optional fields."""
        entry = game_logger.log_move(
            turn_number=5,
            move="e2e4",
            player="white",
            reasoning_depth=3,
            entropy=0.5,
            monte_carlo_samples=10,
            traits_applied={"analytical": 1.5, "aggressive": 1.2},
            strategy_used="aggressive_opening",
            goal_alignment=["win_game", "attack"],
            predicted_outcome={"winner": "white", "confidence": 0.8},
            confidence=0.85,
            reasoning_time_ms=150,
        )

        assert entry.reasoning_depth == 3
        assert entry.entropy == 0.5
        assert entry.monte_carlo_samples == 10
        assert entry.traits_applied == {"analytical": 1.5, "aggressive": 1.2}
        assert entry.strategy_used == "aggressive_opening"
        assert entry.goal_alignment == ["win_game", "attack"]
        assert entry.predicted_outcome == {"winner": "white", "confidence": 0.8}
        assert entry.confidence == 0.85
        assert entry.reasoning_time_ms == 150

    def test_log_outcome(self, game_logger):
        """Test logging outcome for a turn."""
        game_logger.log_move(turn_number=1, move="a1", player="X")
        game_logger.log_outcome(turn=1, actual_outcome={"result": "win"})

        # Outcome should be stored
        summary = game_logger.get_session_summary()
        # Prediction accuracy should be calculated when we have outcomes
        assert summary is not None

    def test_log_outcome_writes_to_file(self, file_logger, temp_log_dir):
        """Test log_outcome writes to file."""
        file_logger.log_move(turn_number=1, move="a1", player="X")
        file_logger.log_outcome(turn=1, actual_outcome={"result": "win"})

        log_file = temp_log_dir / f"game_{file_logger.game_id}.jsonl"
        lines = log_file.read_text().strip().split("\n")

        # Should have move + outcome entries
        assert len(lines) == 2

        outcome_entry = json.loads(lines[1])
        assert outcome_entry["type"] == "outcome"
        assert outcome_entry["turn"] == 1
        assert outcome_entry["actual_outcome"]["result"] == "win"

    def test_get_session_summary_empty(self, game_logger):
        """Test session summary with no moves."""
        summary = game_logger.get_session_summary()

        assert summary["game_id"] == "test-game"
        assert summary["total_moves"] == 0
        assert summary["duration_seconds"] == 0

    def test_get_session_summary(self, game_logger):
        """Test session summary with moves."""
        game_logger.log_move(
            turn_number=1,
            move="a1",
            player="X",
            confidence=0.8,
            entropy=0.3,
            reasoning_time_ms=100,
            traits_applied={"analytical": 1.5},
            strategy_used="opening",
        )
        game_logger.log_move(
            turn_number=2,
            move="b2",
            player="O",
            confidence=0.6,
            entropy=0.5,
            reasoning_time_ms=200,
            traits_applied={"analytical": 1.5},
            strategy_used="opening",
        )

        summary = game_logger.get_session_summary()

        assert summary["total_moves"] == 2
        assert summary["average_confidence"] == 0.7  # (0.8 + 0.6) / 2
        assert summary["average_entropy"] == 0.4  # (0.3 + 0.5) / 2
        assert summary["average_reasoning_time_ms"] == 150.0
        assert summary["total_reasoning_time_ms"] == 300
        assert summary["trait_usage"]["analytical"] == 2
        assert summary["strategy_usage"]["opening"] == 2

    def test_get_entries(self, game_logger):
        """Test getting all logged entries."""
        game_logger.log_move(turn_number=1, move="a1", player="X")
        game_logger.log_move(turn_number=2, move="b2", player="O")

        entries = game_logger.get_entries()

        assert len(entries) == 2
        assert entries[0].turn_number == 1
        assert entries[1].turn_number == 2

    def test_clear(self, game_logger):
        """Test clearing logged entries."""
        game_logger.log_move(turn_number=1, move="a1", player="X")
        game_logger.log_outcome(turn=1, actual_outcome={})

        game_logger.clear()

        assert len(game_logger.get_entries()) == 0
        summary = game_logger.get_session_summary()
        assert summary["total_moves"] == 0

    def test_prediction_accuracy(self, game_logger):
        """Test prediction accuracy calculation."""
        # Log moves with predictions
        game_logger.log_move(
            turn_number=1,
            move="a1",
            player="X",
            predicted_outcome={"result": "continue"},
        )
        game_logger.log_move(
            turn_number=2,
            move="b2",
            player="O",
            predicted_outcome={"result": "win"},
        )

        # Log actual outcomes
        game_logger.log_outcome(turn=1, actual_outcome={"result": "continue"})
        game_logger.log_outcome(turn=2, actual_outcome={"result": "loss"})

        summary = game_logger.get_session_summary()

        # 1 correct (turn 1), 1 incorrect (turn 2) = 0.5 accuracy
        assert summary["prediction_accuracy"] == 0.5
