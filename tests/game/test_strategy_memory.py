"""Tests for strategy memory module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.game.strategy_memory import StrategyMemory


class TestStrategyMemory:
    """Tests for StrategyMemory class."""

    @pytest.fixture
    def memory_store(self):
        """Create a memory store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            yield store

    @pytest.fixture
    def strategy_memory(self, memory_store):
        """Create a strategy memory for testing."""
        return StrategyMemory(memory_store)

    def test_init(self, memory_store):
        """Test initialization."""
        sm = StrategyMemory(memory_store)
        assert sm.memory == memory_store

    def test_store_strategy_creates_node(self, strategy_memory):
        """Test store_strategy creates a fractal node."""
        state = {"board": ["X", "O", ""], "turn": 2}
        move = "2"
        outcome = {"result": "win", "winner": "X"}

        node = strategy_memory.store_strategy(
            state=state,
            move=move,
            outcome=outcome,
            game_type="tictactoe",
        )

        assert node is not None
        assert node.id is not None
        assert "strategy" in node.tags
        assert "game:tictactoe" in node.tags
        assert "move:2" in node.tags

    def test_store_strategy_includes_outcome_tags(self, strategy_memory):
        """Test store_strategy adds outcome-based tags."""
        node = strategy_memory.store_strategy(
            state={"board": []},
            move="a1",
            outcome={"winner": "X", "result": "win"},
            game_type="chess",
        )

        assert "outcome:X" in node.tags
        assert "result:win" in node.tags

    def test_store_strategy_content_is_json(self, strategy_memory):
        """Test stored content is valid JSON."""
        state = {"position": [1, 2, 3]}
        move = "e2e4"
        outcome = {"evaluation": 0.5}

        node = strategy_memory.store_strategy(
            state=state,
            move=move,
            outcome=outcome,
            game_type="chess",
        )

        content = json.loads(node.content)
        assert content["state"] == state
        assert content["move"] == move
        assert content["outcome"] == outcome
        assert content["game_type"] == "chess"

    def test_store_strategy_with_custom_tags(self, strategy_memory):
        """Test store_strategy accepts custom tags."""
        node = strategy_memory.store_strategy(
            state={},
            move="1",
            outcome={},
            game_type="tictactoe",
            tags=["opening", "aggressive"],
        )

        assert "opening" in node.tags
        assert "aggressive" in node.tags
        assert "strategy" in node.tags

    def test_retrieve_relevant_strategies_returns_list(self, strategy_memory):
        """Test retrieve_relevant_strategies returns node list."""
        # Store some strategies
        strategy_memory.store_strategy(
            state={"board": ["X"]},
            move="0",
            outcome={"result": "win"},
            game_type="tictactoe",
        )
        strategy_memory.store_strategy(
            state={"board": ["O"]},
            move="1",
            outcome={"result": "loss"},
            game_type="tictactoe",
        )

        result = strategy_memory.retrieve_relevant_strategies(
            state={"board": ["X"]},
            k=5,
            game_type="tictactoe",
        )

        assert isinstance(result, list)

    def test_retrieve_uses_memory_get_entangled_context(self, strategy_memory, memory_store):
        """Test retrieve calls memory.get_entangled_context."""
        # Store a strategy first
        strategy_memory.store_strategy(
            state={"board": []},
            move="test",
            outcome={},
            game_type="test",
        )

        # Retrieve
        result = strategy_memory.retrieve_relevant_strategies(
            state={"board": []},
            k=3,
        )

        # Should return strategies (may be empty if no match)
        assert isinstance(result, list)

    def test_update_strategy_weight_increases_importance(self, strategy_memory):
        """Test update_strategy_weight increases importance on win."""
        node = strategy_memory.store_strategy(
            state={},
            move="1",
            outcome={"result": "win"},
            game_type="test",
        )
        original_importance = node.importance

        success = strategy_memory.update_strategy_weight(
            node_id=node.id,
            outcome={"result": "win"},
            adjustment=0.2,
        )

        assert success is True
        updated_node = strategy_memory.memory.get_node_by_id(node.id)
        assert updated_node.importance > original_importance

    def test_update_strategy_weight_decreases_importance(self, strategy_memory):
        """Test update_strategy_weight decreases importance on loss."""
        node = strategy_memory.store_strategy(
            state={},
            move="1",
            outcome={},
            game_type="test",
        )
        # Set initial importance
        node.importance = 0.5
        strategy_memory.memory._update_node(node)

        success = strategy_memory.update_strategy_weight(
            node_id=node.id,
            outcome={"result": "loss"},
            adjustment=0.2,
        )

        assert success is True
        updated_node = strategy_memory.memory.get_node_by_id(node.id)
        assert updated_node.importance < 0.5

    def test_update_strategy_weight_bounds_importance(self, strategy_memory):
        """Test update_strategy_weight keeps importance in [0, 1]."""
        node = strategy_memory.store_strategy(
            state={},
            move="1",
            outcome={},
            game_type="test",
        )
        node.importance = 0.95
        strategy_memory.memory._update_node(node)

        # Try to increase beyond 1.0
        strategy_memory.update_strategy_weight(
            node_id=node.id,
            outcome={"result": "win"},
            adjustment=0.5,
        )

        updated_node = strategy_memory.memory.get_node_by_id(node.id)
        assert updated_node.importance <= 1.0

    def test_update_strategy_weight_returns_false_for_missing_node(self, strategy_memory):
        """Test update_strategy_weight returns False for missing node."""
        success = strategy_memory.update_strategy_weight(
            node_id="nonexistent-id",
            outcome={"result": "win"},
        )

        assert success is False

    def test_get_winning_strategies(self, strategy_memory):
        """Test get_winning_strategies filters by outcome."""
        strategy_memory.store_strategy(
            state={},
            move="1",
            outcome={"result": "win"},
            game_type="tictactoe",
        )
        strategy_memory.store_strategy(
            state={},
            move="2",
            outcome={"result": "loss"},
            game_type="tictactoe",
        )

        result = strategy_memory.get_winning_strategies(
            game_type="tictactoe",
            k=10,
        )

        assert isinstance(result, list)
        # All returned strategies should be winning
        # (tags 'outcome:win' or 'result:win' are generated by store_strategy)
        for node in result:
            has_win_tag = any(
                tag.startswith("outcome:") and "win" in tag.lower()
                or tag.startswith("result:") and "win" in tag.lower()
                for tag in node.tags
            )
            assert has_win_tag, f"Expected winning strategy tags, got: {node.tags}"

    def test_link_strategies_creates_bidirectional_link(self, strategy_memory):
        """Test link_strategies creates bidirectional entanglement."""
        node1 = strategy_memory.store_strategy(
            state={},
            move="1",
            outcome={},
            game_type="test",
        )
        node2 = strategy_memory.store_strategy(
            state={},
            move="2",
            outcome={},
            game_type="test",
        )

        success = strategy_memory.link_strategies(
            node_id_1=node1.id,
            node_id_2=node2.id,
            strength=0.8,
        )

        assert success is True

        updated_node1 = strategy_memory.memory.get_node_by_id(node1.id)
        updated_node2 = strategy_memory.memory.get_node_by_id(node2.id)

        assert node2.id in updated_node1.entangled_ids
        assert updated_node1.entangled_ids[node2.id] == 0.8
        assert node1.id in updated_node2.entangled_ids
        assert updated_node2.entangled_ids[node1.id] == 0.8

    def test_link_strategies_bounds_strength(self, strategy_memory):
        """Test link_strategies bounds strength to [0, 1]."""
        node1 = strategy_memory.store_strategy(
            state={},
            move="1",
            outcome={},
            game_type="test",
        )
        node2 = strategy_memory.store_strategy(
            state={},
            move="2",
            outcome={},
            game_type="test",
        )

        strategy_memory.link_strategies(
            node_id_1=node1.id,
            node_id_2=node2.id,
            strength=1.5,  # Over max
        )

        updated_node1 = strategy_memory.memory.get_node_by_id(node1.id)
        assert updated_node1.entangled_ids[node2.id] == 1.0

    def test_link_strategies_returns_false_for_missing_node(self, strategy_memory):
        """Test link_strategies returns False if node doesn't exist."""
        node1 = strategy_memory.store_strategy(
            state={},
            move="1",
            outcome={},
            game_type="test",
        )

        success = strategy_memory.link_strategies(
            node_id_1=node1.id,
            node_id_2="nonexistent",
            strength=0.5,
        )

        assert success is False

    def test_generate_strategy_summary(self, strategy_memory):
        """Test _generate_strategy_summary creates readable summary."""
        summary = strategy_memory._generate_strategy_summary(
            state={"board": []},
            move="e2e4",
            outcome={"result": "win"},
            game_type="chess",
        )

        assert "chess" in summary
        assert "e2e4" in summary
        assert "win" in summary

    def test_build_state_query(self, strategy_memory):
        """Test _build_state_query creates searchable query."""
        query = strategy_memory._build_state_query(
            state={"board": [], "current_player": "X", "turn": 5},
            game_type="tictactoe",
        )

        assert "strategy" in query
        assert "game:tictactoe" in query
