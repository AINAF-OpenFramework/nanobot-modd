"""Tests for game interface module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from nanobot.agent.memory_types import ContextBlock
from nanobot.game.interface import (
    GameObservation,
    build_game_context_prompt,
    inject_into_context,
    parse_observation,
)


class TestGameObservation:
    """Tests for GameObservation model."""

    def test_create_observation_with_required_fields(self):
        """Test creating observation with only required fields."""
        obs = GameObservation(
            game_id="test-game-1",
            game_type="tictactoe",
            current_player="X",
        )
        assert obs.game_id == "test-game-1"
        assert obs.game_type == "tictactoe"
        assert obs.current_player == "X"
        assert obs.turn_number == 0
        assert obs.board_state == {}
        assert obs.legal_moves == []
        assert obs.win_conditions == {}

    def test_create_observation_with_all_fields(self):
        """Test creating observation with all fields."""
        obs = GameObservation(
            game_id="test-game-2",
            game_type="chess",
            turn_number=5,
            current_player="white",
            board_state={"pieces": ["K", "Q"]},
            legal_moves=["e2e4", "d2d4"],
            win_conditions={"checkmate": False},
        )
        assert obs.game_id == "test-game-2"
        assert obs.game_type == "chess"
        assert obs.turn_number == 5
        assert obs.current_player == "white"
        assert obs.board_state == {"pieces": ["K", "Q"]}
        assert obs.legal_moves == ["e2e4", "d2d4"]
        assert obs.win_conditions == {"checkmate": False}


class TestParseObservation:
    """Tests for parse_observation function."""

    def test_parse_dict_input(self):
        """Test parsing dictionary input."""
        data = {
            "game_id": "test-1",
            "game_type": "tictactoe",
            "current_player": "X",
            "turn_number": 3,
            "board_state": {"board": ["X", "O", "", "", "X", "", "", "", "O"]},
            "legal_moves": ["2", "3", "5", "6", "7"],
        }
        obs = parse_observation(data)
        assert obs.game_id == "test-1"
        assert obs.turn_number == 3
        assert obs.legal_moves == ["2", "3", "5", "6", "7"]

    def test_parse_json_string_input(self):
        """Test parsing JSON string input."""
        data = {
            "game_id": "test-2",
            "game_type": "chess",
            "current_player": "white",
        }
        json_str = json.dumps(data)
        obs = parse_observation(json_str)
        assert obs.game_id == "test-2"
        assert obs.game_type == "chess"

    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_observation("not valid json {")

    def test_parse_missing_required_field_raises_error(self):
        """Test that missing required field raises ValueError."""
        data = {"game_id": "test"}  # Missing game_type and current_player
        with pytest.raises(ValueError, match="Failed to validate"):
            parse_observation(data)

    def test_parse_non_dict_raises_error(self):
        """Test that non-dict input raises ValueError."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            parse_observation(json.dumps([1, 2, 3]))


class TestInjectIntoContext:
    """Tests for inject_into_context function."""

    def test_inject_returns_context_block(self):
        """Test that inject returns a ContextBlock."""
        obs = GameObservation(
            game_id="inject-test",
            game_type="tictactoe",
            current_player="X",
        )
        mock_context_builder = MagicMock()

        result = inject_into_context(obs, mock_context_builder)

        assert isinstance(result, ContextBlock)
        assert result.name == "game_state"
        assert result.metadata["game_id"] == "inject-test"
        assert result.metadata["game_type"] == "tictactoe"
        assert result.metadata["turn_number"] == 0

    def test_inject_includes_game_content(self):
        """Test that inject includes formatted game content."""
        obs = GameObservation(
            game_id="content-test",
            game_type="chess",
            turn_number=10,
            current_player="black",
        )
        mock_context_builder = MagicMock()

        result = inject_into_context(obs, mock_context_builder)

        assert "Game Type:" in result.content
        assert "chess" in result.content
        assert "Turn:" in result.content
        assert "10" in result.content


class TestBuildGameContextPrompt:
    """Tests for build_game_context_prompt function."""

    def test_prompt_contains_all_sections(self):
        """Test that prompt contains all expected sections."""
        obs = GameObservation(
            game_id="prompt-test",
            game_type="tictactoe",
            current_player="O",
            turn_number=4,
            legal_moves=["0", "2", "6"],
        )

        prompt = build_game_context_prompt(obs)

        assert "# Current Game State" in prompt
        assert "## Board State" in prompt
        assert "## Legal Moves" in prompt
        assert "## Win Conditions" in prompt
        assert "prompt-test" in prompt
        assert "tictactoe" in prompt
        assert "0, 2, 6" in prompt

    def test_prompt_empty_legal_moves(self):
        """Test prompt with no legal moves."""
        obs = GameObservation(
            game_id="no-moves",
            game_type="tictactoe",
            current_player="X",
            legal_moves=[],
        )

        prompt = build_game_context_prompt(obs)
        assert "(no legal moves)" in prompt

    def test_prompt_with_win_conditions(self):
        """Test prompt with win conditions."""
        obs = GameObservation(
            game_id="win-test",
            game_type="tictactoe",
            current_player="X",
            win_conditions={"winner": "O", "game_over": True},
        )

        prompt = build_game_context_prompt(obs)
        assert "Winner: O" in prompt
        assert "Game Over: True" in prompt
