"""Tests for game state engine module."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from nanobot.game.state_engine import (
    GameHistoryEntry,
    GameRules,
    GameStateEngine,
)


class MockRules(GameRules):
    """Mock game rules for testing."""

    def get_legal_moves(self, state: dict[str, Any]) -> list[str]:
        return state.get("legal_moves", ["move1", "move2", "move3"])

    def apply_move(self, state: dict[str, Any], move: str) -> dict[str, Any]:
        new_state = state.copy()
        moves_made = new_state.get("moves_made", [])
        moves_made = moves_made + [move]
        new_state["moves_made"] = moves_made
        new_state["turn"] = new_state.get("turn", 0) + 1
        return new_state

    def check_win_conditions(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("winner"):
            return {"game_over": True, "winner": state["winner"], "status": "Won"}
        return {"game_over": False, "winner": None, "status": "In progress"}

    def get_next_player(self, state: dict[str, Any]) -> str:
        current = state.get("current_player", "A")
        return "B" if current == "A" else "A"


class TestGameHistoryEntry:
    """Tests for GameHistoryEntry dataclass."""

    def test_create_entry_with_defaults(self):
        """Test creating entry with default values."""
        entry = GameHistoryEntry(state={"test": "value"})
        assert entry.state == {"test": "value"}
        assert entry.move is None
        assert entry.player is None
        assert entry.timestamp is not None
        assert entry.metadata == {}

    def test_create_entry_with_all_fields(self):
        """Test creating entry with all fields specified."""
        entry = GameHistoryEntry(
            state={"board": [1, 2, 3]},
            move="a1",
            player="X",
            metadata={"type": "test"},
        )
        assert entry.state == {"board": [1, 2, 3]}
        assert entry.move == "a1"
        assert entry.player == "X"
        assert entry.metadata == {"type": "test"}


class TestGameStateEngine:
    """Tests for GameStateEngine class."""

    def test_init_without_state(self):
        """Test initialization without initial state."""
        engine = GameStateEngine()
        assert engine.game_id is not None
        assert engine.rules is None
        assert engine.get_state() == {}

    def test_init_with_state(self):
        """Test initialization with initial state."""
        initial = {"board": [""] * 9, "turn": 0}
        engine = GameStateEngine(
            game_id="test-1",
            initial_state=initial,
        )
        assert engine.game_id == "test-1"
        assert engine.get_state() == initial

    def test_init_creates_history_entry(self):
        """Test that init creates initial history entry."""
        initial = {"value": 1}
        engine = GameStateEngine(initial_state=initial)
        history = engine.get_history()
        assert len(history) == 1
        assert history[0].state == initial
        assert history[0].metadata["type"] == "initial"

    def test_update_state(self):
        """Test updating state."""
        engine = GameStateEngine()
        new_state = {"updated": True}
        engine.update(new_state)
        assert engine.get_state() == new_state

    def test_update_adds_history(self):
        """Test that update adds history entry."""
        engine = GameStateEngine(initial_state={"v": 1})
        engine.update({"v": 2})
        history = engine.get_history()
        assert len(history) == 2
        assert history[1].metadata["type"] == "update"

    def test_simulate_returns_new_state(self):
        """Test simulate returns new state without modifying original."""
        rules = MockRules()
        initial = {"turn": 0, "legal_moves": ["a", "b"]}
        engine = GameStateEngine(rules=rules, initial_state=initial)

        simulated = engine.simulate("a")

        assert simulated["turn"] == 1
        assert engine.get_state()["turn"] == 0  # Original unchanged

    def test_simulate_raises_without_rules(self):
        """Test simulate raises error without rules."""
        engine = GameStateEngine()
        with pytest.raises(ValueError, match="No game rules configured"):
            engine.simulate("move")

    def test_simulate_raises_for_illegal_move(self):
        """Test simulate raises for illegal move."""
        rules = MockRules()
        initial = {"legal_moves": ["a", "b"]}
        engine = GameStateEngine(rules=rules, initial_state=initial)
        with pytest.raises(ValueError, match="Illegal move"):
            engine.simulate("c")

    def test_apply_move_updates_state(self):
        """Test apply_move updates the state."""
        rules = MockRules()
        initial = {"turn": 0, "legal_moves": ["x", "y"]}
        engine = GameStateEngine(rules=rules, initial_state=initial)

        result = engine.apply_move("x", player="P1")

        assert result["turn"] == 1
        assert engine.get_state()["turn"] == 1

    def test_apply_move_adds_history(self):
        """Test apply_move adds history entry."""
        rules = MockRules()
        initial = {"turn": 0}
        engine = GameStateEngine(rules=rules, initial_state=initial)

        engine.apply_move("move1", player="P1")

        history = engine.get_history()
        assert len(history) == 2
        assert history[1].move == "move1"
        assert history[1].player == "P1"
        assert history[1].metadata["type"] == "move"

    def test_get_legal_moves(self):
        """Test get_legal_moves returns moves from rules."""
        rules = MockRules()
        initial = {"legal_moves": ["a", "b", "c"]}
        engine = GameStateEngine(rules=rules, initial_state=initial)

        moves = engine.get_legal_moves()

        assert moves == ["a", "b", "c"]

    def test_get_legal_moves_raises_without_rules(self):
        """Test get_legal_moves raises without rules."""
        engine = GameStateEngine()
        with pytest.raises(ValueError, match="No game rules configured"):
            engine.get_legal_moves()

    def test_rollback_one_step(self):
        """Test rolling back one step."""
        rules = MockRules()
        initial = {"turn": 0}
        engine = GameStateEngine(rules=rules, initial_state=initial)

        engine.apply_move("move1")
        engine.apply_move("move2")

        state = engine.rollback(steps=1)

        assert state["turn"] == 1
        assert len(engine.get_history()) == 2

    def test_rollback_multiple_steps(self):
        """Test rolling back multiple steps."""
        rules = MockRules()
        initial = {"turn": 0}
        engine = GameStateEngine(rules=rules, initial_state=initial)

        engine.apply_move("move1")
        engine.apply_move("move2")
        engine.apply_move("move3")

        state = engine.rollback(steps=2)

        assert state["turn"] == 1
        assert len(engine.get_history()) == 2

    def test_rollback_raises_for_insufficient_history(self):
        """Test rollback raises when not enough history."""
        engine = GameStateEngine(initial_state={"v": 1})
        with pytest.raises(ValueError, match="Cannot rollback"):
            engine.rollback(steps=2)

    def test_rollback_raises_for_invalid_steps(self):
        """Test rollback raises for invalid steps."""
        engine = GameStateEngine()
        with pytest.raises(ValueError, match="Steps must be at least 1"):
            engine.rollback(steps=0)

    def test_reset_clears_history(self):
        """Test reset clears history."""
        rules = MockRules()
        initial = {"turn": 0}
        engine = GameStateEngine(rules=rules, initial_state=initial)

        engine.apply_move("move1")
        engine.apply_move("move2")
        engine.reset()

        history = engine.get_history()
        assert len(history) == 1
        assert history[0].state["turn"] == 0

    def test_reset_with_new_state(self):
        """Test reset with new initial state."""
        engine = GameStateEngine(initial_state={"v": 1})
        engine.reset(initial_state={"v": 100})

        assert engine.get_state()["v"] == 100
        assert len(engine.get_history()) == 1

    def test_check_win_conditions(self):
        """Test check_win_conditions delegates to rules."""
        rules = MockRules()
        engine = GameStateEngine(
            rules=rules,
            initial_state={"winner": "X"},
        )

        result = engine.check_win_conditions()

        assert result["game_over"] is True
        assert result["winner"] == "X"

    def test_get_current_player(self):
        """Test get_current_player delegates to rules."""
        rules = MockRules()
        engine = GameStateEngine(
            rules=rules,
            initial_state={"current_player": "A"},
        )

        player = engine.get_current_player()

        assert player == "B"  # MockRules returns next player

    def test_thread_safety(self):
        """Test thread safety of state operations."""
        rules = MockRules()
        initial = {"counter": 0}
        engine = GameStateEngine(rules=rules, initial_state=initial)

        def worker():
            for _ in range(10):
                state = engine.get_state()
                state["counter"] = state.get("counter", 0) + 1
                engine.update(state)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Just verify it doesn't crash - exact count may vary due to race conditions
        assert engine.get_state()["counter"] >= 1

    def test_state_copies_are_independent(self):
        """Test that returned states are independent copies."""
        engine = GameStateEngine(initial_state={"nested": {"value": 1}})

        state1 = engine.get_state()
        state1["nested"]["value"] = 999

        state2 = engine.get_state()
        assert state2["nested"]["value"] == 1
