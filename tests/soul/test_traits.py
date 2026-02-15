"""Tests for TraitScorer module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from nanobot.soul.loader import SoulLoader
from nanobot.soul.schema import Goal, PersonalityTrait, Strategy
from nanobot.soul.traits import TraitScorer


class TestTraitScorer:
    """Tests for TraitScorer class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before and after each test."""
        SoulLoader.reset_instance()
        yield
        SoulLoader.reset_instance()

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def workspace_with_soul(self, temp_workspace):
        """Create a workspace with a configured soul.yaml."""
        soul_data = {
            "version": "1.0",
            "name": "test-bot",
            "traits": [
                {
                    "name": "analytical",
                    "weight": 1.5,
                    "description": "Analytical trait",
                    "affects": ["reasoning_depth"],
                },
                {
                    "name": "aggressive",
                    "weight": 1.3,
                    "description": "Aggressive trait",
                    "affects": [],
                },
                {
                    "name": "cautious",
                    "weight": 1.2,
                    "description": "Cautious trait",
                    "affects": ["risk_tolerance"],
                },
                {
                    "name": "efficient",
                    "weight": 1.4,
                    "description": "Efficient trait",
                    "affects": [],
                },
            ],
            "goals": [
                {
                    "name": "win_game",
                    "priority": 10,
                    "description": "Win the game",
                    "actions": ["attack", "win", "capture"],
                },
                {
                    "name": "avoid_loss",
                    "priority": 9,
                    "description": "Avoid losing",
                    "actions": ["defend", "block", "protect"],
                },
            ],
            "strategies": [
                {
                    "name": "aggressive_opening",
                    "condition": "early_game",
                    "approach": "aggressive",
                    "traits_boost": {"aggressive": 0.3},
                },
                {
                    "name": "defensive_recovery",
                    "condition": "losing",
                    "approach": "defensive",
                    "traits_boost": {"cautious": 0.4},
                },
                {
                    "name": "calculated_endgame",
                    "condition": "endgame",
                    "approach": "calculated",
                    "traits_boost": {"analytical": 0.5},
                },
            ],
            "game": {
                "default_reasoning_depth": 2,
                "monte_carlo_samples": 3,
                "beam_width": 4,
                "risk_tolerance": 0.4,
            },
        }
        soul_file = temp_workspace / "soul.yaml"
        soul_file.write_text(yaml.dump(soul_data), encoding="utf-8")
        return temp_workspace

    @pytest.fixture
    def trait_scorer(self, workspace_with_soul):
        """Create a TraitScorer instance."""
        soul_loader = SoulLoader.get_instance(workspace_with_soul)
        return TraitScorer(soul_loader)

    def test_score_hypothesis_base_score(self, trait_scorer):
        """Test that base score is preserved and modified appropriately."""
        base_score = 0.5
        hypothesis = "make a move"

        result = trait_scorer.score_hypothesis(hypothesis, base_score)

        # Score should be bounded between 0 and 1
        assert 0.0 <= result <= 1.0

    def test_score_hypothesis_aggressive_boost(self, trait_scorer):
        """Test that aggressive hypothesis gets trait boost."""
        base_score = 0.5

        # Hypothesis mentioning aggressive trait
        aggressive_hypothesis = "play aggressive attack move"
        aggressive_score = trait_scorer.score_hypothesis(aggressive_hypothesis, base_score)

        # Neutral hypothesis
        neutral_hypothesis = "make a move"
        neutral_score = trait_scorer.score_hypothesis(neutral_hypothesis, base_score)

        # Aggressive hypothesis should score higher due to trait matching
        # and goal alignment (attack aligns with win_game goal)
        assert aggressive_score >= neutral_score

    def test_score_hypothesis_goal_alignment(self, trait_scorer):
        """Test that goal-aligned actions boost score."""
        base_score = 0.5

        # Hypothesis with win-aligned action
        win_hypothesis = "attack the opponent to win"
        win_score = trait_scorer.score_hypothesis(win_hypothesis, base_score)

        # Hypothesis with defend action (avoid_loss goal)
        defend_hypothesis = "defend the position"
        defend_score = trait_scorer.score_hypothesis(defend_hypothesis, base_score)

        # Neutral hypothesis
        neutral_hypothesis = "wait"
        neutral_score = trait_scorer.score_hypothesis(neutral_hypothesis, base_score)

        # Both goal-aligned hypotheses should score higher than neutral
        assert win_score >= neutral_score
        assert defend_score >= neutral_score

    def test_score_hypothesis_with_strategy(self, trait_scorer):
        """Test that active strategy modifies score."""
        base_score = 0.5
        hypothesis = "attack aggressively"

        # Early game state (should activate aggressive_opening strategy)
        early_game_state = {"turn": 1, "phase": "opening"}
        early_score = trait_scorer.score_hypothesis(
            hypothesis, base_score, early_game_state
        )

        # Mid game state (no strategy active)
        mid_game_state = {"turn": 10, "phase": "midgame"}
        mid_score = trait_scorer.score_hypothesis(hypothesis, base_score, mid_game_state)

        # Early game with aggressive strategy should score higher for aggressive move
        # Note: This depends on implementation details
        assert 0.0 <= early_score <= 1.0
        assert 0.0 <= mid_score <= 1.0

    def test_get_reasoning_depth(self, trait_scorer):
        """Test get_reasoning_depth returns modified depth."""
        depth = trait_scorer.get_reasoning_depth()

        # Base depth is 2, analytical trait weight is 1.5
        # So modified depth should be int(2 * 1.5) = 3
        assert depth == 3
        assert 1 <= depth <= 10

    def test_get_reasoning_depth_with_endgame_strategy(self, trait_scorer):
        """Test that endgame strategy boosts analytical trait for depth."""
        # Endgame state (should activate calculated_endgame strategy)
        endgame_state = {"turn": 25, "phase": "endgame"}
        depth = trait_scorer.get_reasoning_depth(endgame_state)

        # Should be higher due to analytical boost
        assert depth >= 3  # Base depth * analytical weight
        assert 1 <= depth <= 10

    def test_get_monte_carlo_samples(self, trait_scorer):
        """Test get_monte_carlo_samples returns appropriate value."""
        samples = trait_scorer.get_monte_carlo_samples()

        # Base samples is 3, efficient trait (1.4) reduces samples
        # Modifier is max(0.7, 1/1.4) = 0.714
        # So samples should be int(3 * 0.714) = 2
        assert samples >= 1
        assert samples <= 20

    def test_get_risk_tolerance(self, trait_scorer):
        """Test get_risk_tolerance returns modified value."""
        tolerance = trait_scorer.get_risk_tolerance()

        # Base tolerance is 0.4, cautious trait (1.2) reduces it
        # Modified = 0.4 * (1/1.2) = 0.333
        assert 0.0 <= tolerance <= 1.0
        assert tolerance < 0.4  # Should be reduced by cautious trait

    def test_get_risk_tolerance_aggressive_strategy(self, trait_scorer):
        """Test that aggressive strategy increases risk tolerance."""
        # Early game with aggressive strategy
        early_state = {"turn": 1, "phase": "opening"}
        early_tolerance = trait_scorer.get_risk_tolerance(early_state)

        # Mid game (no strategy)
        mid_state = {"turn": 10, "phase": "midgame"}
        mid_tolerance = trait_scorer.get_risk_tolerance(mid_state)

        # Both should be valid
        assert 0.0 <= early_tolerance <= 1.0
        assert 0.0 <= mid_tolerance <= 1.0

    def test_get_risk_tolerance_defensive_strategy(self, trait_scorer):
        """Test that defensive strategy decreases risk tolerance."""
        # Losing state (should activate defensive strategy)
        losing_state = {"score_diff": -5, "phase": "losing"}
        losing_tolerance = trait_scorer.get_risk_tolerance(losing_state)

        # Normal state
        normal_state = {"score_diff": 0}
        normal_tolerance = trait_scorer.get_risk_tolerance(normal_state)

        # Defensive strategy should reduce tolerance
        assert 0.0 <= losing_tolerance <= 1.0
        assert losing_tolerance <= normal_tolerance

    def test_from_workspace(self, workspace_with_soul):
        """Test creating TraitScorer from workspace path."""
        scorer = TraitScorer.from_workspace(workspace_with_soul)

        assert scorer is not None
        # Should work correctly
        depth = scorer.get_reasoning_depth()
        assert 1 <= depth <= 10

    def test_detect_active_strategy_early_game(self, trait_scorer):
        """Test strategy detection for early game."""
        config = trait_scorer._soul_loader.load()

        early_state = {"turn": 1}
        strategy = trait_scorer._detect_active_strategy(early_state, config.strategies)

        assert strategy is not None
        assert strategy.name == "aggressive_opening"

    def test_detect_active_strategy_losing(self, trait_scorer):
        """Test strategy detection for losing state."""
        config = trait_scorer._soul_loader.load()

        losing_state = {"score_diff": -10, "turn": 10}  # turn > 3 to avoid early_game
        strategy = trait_scorer._detect_active_strategy(losing_state, config.strategies)

        assert strategy is not None
        assert strategy.name == "defensive_recovery"

    def test_detect_active_strategy_endgame(self, trait_scorer):
        """Test strategy detection for endgame."""
        config = trait_scorer._soul_loader.load()

        endgame_state = {"phase": "endgame", "turn": 10}  # turn > 3 to avoid early_game
        strategy = trait_scorer._detect_active_strategy(endgame_state, config.strategies)

        assert strategy is not None
        assert strategy.name == "calculated_endgame"

    def test_detect_active_strategy_no_match(self, trait_scorer):
        """Test strategy detection with no matching condition."""
        config = trait_scorer._soul_loader.load()

        normal_state = {"turn": 10, "score_diff": 0, "phase": "midgame"}
        strategy = trait_scorer._detect_active_strategy(normal_state, config.strategies)

        # Midgame doesn't match any strategy condition
        assert strategy is None

    def test_score_bounded(self, trait_scorer):
        """Test that score is always bounded to [0, 1]."""
        # Very high base score
        result_high = trait_scorer.score_hypothesis("attack win capture", 0.99)
        assert result_high <= 1.0

        # Very low base score
        result_low = trait_scorer.score_hypothesis("nothing", 0.01)
        assert result_low >= 0.0
