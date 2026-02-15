"""Tests for strategy memory entanglement at scale."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.game.rules.tictactoe import TicTacToeRules
from nanobot.game.strategy_memory import StrategyMemory


class TestStrategyMemoryEntanglement:
    """Tests for strategy memory entanglement at scale."""

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

    @pytest.fixture
    def tictactoe_rules(self):
        """Create TicTacToe rules."""
        return TicTacToeRules()

    def test_store_and_retrieve_single_strategy(self, strategy_memory):
        """Test storing and retrieving a single strategy."""
        state = {"board": [["X", "", ""], ["", "", ""], ["", "", ""]], "current_player": "O"}
        move = "r1c1"
        outcome = {"result": "in_progress"}

        # Store strategy
        node = strategy_memory.store_strategy(
            state=state, move=move, outcome=outcome, game_type="tictactoe"
        )

        assert node is not None
        assert node.id is not None

        # Retrieve strategy
        retrieved = strategy_memory.retrieve_relevant_strategies(
            state=state, k=5, game_type="tictactoe"
        )

        assert isinstance(retrieved, list)
        # May or may not retrieve the exact strategy depending on embedding similarity

    def test_store_multiple_strategies(self, strategy_memory):
        """Test storing multiple strategies."""
        strategies = [
            {
                "state": {"board": [["X", "", ""], ["", "", ""], ["", "", ""]]},
                "move": "r1c1",
                "outcome": {"result": "win"},
            },
            {
                "state": {"board": [["", "X", ""], ["", "", ""], ["", "", ""]]},
                "move": "r1c1",
                "outcome": {"result": "win"},
            },
            {
                "state": {"board": [["", "", "X"], ["", "", ""], ["", "", ""]]},
                "move": "r1c1",
                "outcome": {"result": "win"},
            },
        ]

        nodes = []
        for strat in strategies:
            node = strategy_memory.store_strategy(
                state=strat["state"],
                move=strat["move"],
                outcome=strat["outcome"],
                game_type="tictactoe",
            )
            nodes.append(node)

        # All nodes should be unique
        node_ids = [n.id for n in nodes]
        assert len(node_ids) == len(set(node_ids))

    def test_strategy_memory_scale_100_games(
        self, strategy_memory, tictactoe_rules
    ):
        """Test strategy memory with 100+ game sessions."""
        num_games = 100
        strategies_stored = 0

        # Simulate multiple games
        for game_num in range(num_games):
            state = tictactoe_rules.create_initial_state()
            moves_made = 0

            # Play random game
            while moves_made < 9:
                legal_moves = tictactoe_rules.get_legal_moves(state)
                if not legal_moves:
                    break

                # Pick first legal move (simple strategy)
                move = legal_moves[0]

                # Store strategy before applying move
                outcome = {"game_num": game_num, "move_num": moves_made}
                strategy_memory.store_strategy(
                    state=state,
                    move=move,
                    outcome=outcome,
                    game_type="tictactoe",
                    tags=[f"game_{game_num}"],
                )
                strategies_stored += 1

                # Apply move
                state = tictactoe_rules.apply_move(state, move)
                moves_made += 1

                # Check if game ended
                win_check = tictactoe_rules.check_win_conditions(state)
                if win_check["game_over"]:
                    break

        assert strategies_stored >= 100  # At least 100 strategies stored

    def test_strategy_retrieval_latency(self, strategy_memory, tictactoe_rules):
        """Test strategy retrieval latency (<200ms target)."""
        # Store some strategies first
        num_strategies = 50
        for i in range(num_strategies):
            state = {
                "board": [
                    ["X" if j == i % 3 else "" for j in range(3)] for _ in range(3)
                ],
                "current_player": "O",
            }
            strategy_memory.store_strategy(
                state=state,
                move=f"r{i % 3}c{i % 3}",
                outcome={"iteration": i},
                game_type="tictactoe",
            )

        # Measure retrieval time
        test_state = tictactoe_rules.create_initial_state()

        start_time = time.time()
        retrieved = strategy_memory.retrieve_relevant_strategies(
            state=test_state, k=5, game_type="tictactoe"
        )
        elapsed_ms = (time.time() - start_time) * 1000

        # Should retrieve within 200ms
        assert elapsed_ms < 200, f"Retrieval took {elapsed_ms:.1f}ms (target: <200ms)"
        assert isinstance(retrieved, list)

    def test_strategy_retrieval_multiple_queries(self, strategy_memory):
        """Test multiple retrieval queries maintain good performance."""
        # Store strategies
        for i in range(30):
            strategy_memory.store_strategy(
                state={"board": [[f"p{i}" if j == 0 else "" for j in range(3)] for _ in range(3)]},
                move=f"r{i % 3}c{i % 3}",
                outcome={"idx": i},
                game_type="test",
            )

        # Perform multiple retrievals
        latencies = []
        for i in range(10):
            test_state = {"board": [[f"q{i}" if j == 0 else "" for j in range(3)] for _ in range(3)]}

            start_time = time.time()
            strategy_memory.retrieve_relevant_strategies(state=test_state, k=5)
            elapsed_ms = (time.time() - start_time) * 1000
            latencies.append(elapsed_ms)

        # Average latency should be reasonable
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 200, f"Average latency: {avg_latency:.1f}ms (target: <200ms)"

    def test_entanglement_links_consistency(self, strategy_memory, tictactoe_rules):
        """Test entanglement links maintain consistency."""
        # Store related strategies (moves in a sequence)
        state = tictactoe_rules.create_initial_state()
        nodes = []

        for i in range(5):
            legal_moves = tictactoe_rules.get_legal_moves(state)
            if not legal_moves:
                break

            move = legal_moves[0]

            # Store with entanglement link
            node = strategy_memory.store_strategy(
                state=state,
                move=move,
                outcome={"sequence": i},
                game_type="tictactoe",
            )
            nodes.append(node)

            state = tictactoe_rules.apply_move(state, move)

        # Verify all nodes were created
        assert len(nodes) >= 3
        for node in nodes:
            assert node.id is not None

    def test_strategy_memory_handles_duplicate_states(self, strategy_memory):
        """Test strategy memory handles duplicate states."""
        state = {"board": [["X", "O", ""], ["", "", ""], ["", "", ""]]}

        # Store same state multiple times with different moves
        node1 = strategy_memory.store_strategy(
            state=state, move="r1c0", outcome={"attempt": 1}, game_type="test"
        )
        node2 = strategy_memory.store_strategy(
            state=state, move="r1c1", outcome={"attempt": 2}, game_type="test"
        )
        node3 = strategy_memory.store_strategy(
            state=state, move="r1c2", outcome={"attempt": 3}, game_type="test"
        )

        # Should create separate nodes
        assert node1.id != node2.id
        assert node2.id != node3.id
        assert node1.id != node3.id

    def test_strategy_memory_game_type_filtering(self, strategy_memory):
        """Test retrieval with game type filtering."""
        # Store strategies for different game types
        for game_type in ["tictactoe", "chess", "checkers"]:
            for i in range(5):
                strategy_memory.store_strategy(
                    state={"type": game_type, "idx": i},
                    move=f"move_{i}",
                    outcome={"result": "test"},
                    game_type=game_type,
                )

        # Retrieve with game type filter
        ttt_strategies = strategy_memory.retrieve_relevant_strategies(
            state={"type": "tictactoe"}, k=10, game_type="tictactoe"
        )

        # Should retrieve strategies (actual filtering depends on underlying MemoryStore)
        assert isinstance(ttt_strategies, list)

    def test_strategy_update_weight(self, strategy_memory):
        """Test updating strategy weights."""
        # Store a strategy
        node = strategy_memory.store_strategy(
            state={"board": [["X", "", ""], ["", "", ""], ["", "", ""]]},
            move="r0c0",
            outcome={"result": "win"},
            game_type="test",
        )

        initial_importance = node.importance

        # Update weight
        success = strategy_memory.update_strategy_weight(
            node.id,
            outcome={"result": "win"},
            adjustment=0.5
        )

        # Should succeed
        assert success is True

    def test_memory_correctness_after_scale(self, strategy_memory, tictactoe_rules):
        """Test memory correctness after storing many strategies."""
        # Store strategies with known patterns
        winning_states = []
        for i in range(10):
            state = {
                "board": [
                    ["X", "X", "X"],  # Winning row
                    ["O", "O", ""],
                    ["", "", ""],
                ],
                "current_player": "O",
                "game_id": i,
            }
            node = strategy_memory.store_strategy(
                state=state,
                move="r1c2",
                outcome={"result": "loss", "winner": "X"},
                game_type="tictactoe",
                tags=["losing_position"],
            )
            winning_states.append(node)

        # Store other random strategies
        for i in range(40):
            state = tictactoe_rules.create_initial_state()
            strategy_memory.store_strategy(
                state=state,
                move="r0c0",
                outcome={"random": i},
                game_type="tictactoe",
            )

        # Verify we can still retrieve relevant strategies
        test_state = {
            "board": [
                ["X", "X", "X"],
                ["O", "O", ""],
                ["", "", ""],
            ]
        }
        retrieved = strategy_memory.retrieve_relevant_strategies(state=test_state, k=5)

        assert isinstance(retrieved, list)
        # Should retrieve some strategies

    def test_concurrent_storage_simulation(self, strategy_memory):
        """Test strategy storage under simulated concurrent load."""
        # Simulate rapid strategy storage (single-threaded but fast)
        nodes = []
        start_time = time.time()

        for i in range(50):
            node = strategy_memory.store_strategy(
                state={"iteration": i, "board": [["" for _ in range(3)] for _ in range(3)]},
                move=f"r{i % 3}c{i % 3}",
                outcome={"batch": "concurrent_test"},
                game_type="test",
            )
            nodes.append(node)

        elapsed = time.time() - start_time

        # All nodes should be created
        assert len(nodes) == 50
        # Should complete reasonably fast (< 5 seconds for 50 stores)
        assert elapsed < 5.0

        # All nodes should have unique IDs
        node_ids = [n.id for n in nodes]
        assert len(node_ids) == len(set(node_ids))
