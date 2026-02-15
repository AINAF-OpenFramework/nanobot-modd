"""Tests for GameMoveTool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.game_move import GameMoveTool


class TestGameMoveTool:
    """Tests for GameMoveTool class."""

    @pytest.fixture
    def tool(self):
        """Create a tool instance for testing."""
        return GameMoveTool()

    def test_name(self, tool):
        """Test tool name."""
        assert tool.name == "game_move"

    def test_description(self, tool):
        """Test tool description."""
        assert "game move" in tool.description.lower()

    def test_parameters_schema(self, tool):
        """Test parameters schema structure."""
        params = tool.parameters
        assert params["type"] == "object"
        assert "move" in params["properties"]
        assert "game_id" in params["properties"]
        assert "move" in params["required"]
        assert "game_id" in params["required"]

    @pytest.mark.asyncio
    async def test_execute_without_handler(self, tool):
        """Test execute without move handler returns success."""
        result = await tool.execute(
            move="e2e4",
            game_id="test-game",
        )

        assert "executed successfully" in result.lower()
        assert "e2e4" in result
        assert "test-game" in result

    @pytest.mark.asyncio
    async def test_execute_validates_against_legal_moves(self, tool):
        """Test execute validates move against legal moves."""
        tool.set_legal_moves(["a", "b", "c"])

        result = await tool.execute(
            move="x",
            game_id="test",
        )

        assert "Error" in result
        assert "Illegal move" in result

    @pytest.mark.asyncio
    async def test_execute_allows_legal_move(self, tool):
        """Test execute allows legal move."""
        tool.set_legal_moves(["a", "b", "c"])

        result = await tool.execute(
            move="b",
            game_id="test",
        )

        assert "Error" not in result
        assert "executed successfully" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_with_async_handler(self, tool):
        """Test execute with async move handler."""
        async_handler = AsyncMock(return_value="Move applied!")
        tool.set_move_handler(async_handler)

        result = await tool.execute(
            move="1",
            game_id="game-1",
            player="X",
        )

        assert result == "Move applied!"
        async_handler.assert_awaited_once_with("1", "game-1", "X")

    @pytest.mark.asyncio
    async def test_execute_with_sync_handler(self, tool):
        """Test execute with sync move handler."""
        def sync_handler(move, game_id, player):
            return f"Sync: {move}"

        tool.set_move_handler(sync_handler)

        result = await tool.execute(
            move="2",
            game_id="game-2",
        )

        assert "Sync: 2" in result

    @pytest.mark.asyncio
    async def test_execute_handler_error(self, tool):
        """Test execute handles handler error."""
        async_handler = AsyncMock(side_effect=Exception("Handler failed"))
        tool.set_move_handler(async_handler)

        result = await tool.execute(
            move="1",
            game_id="test",
        )

        assert "Error" in result
        assert "failed" in result.lower()

    def test_set_legal_moves(self, tool):
        """Test set_legal_moves stores moves."""
        tool.set_legal_moves(["x", "y", "z"])
        assert tool._legal_moves == ["x", "y", "z"]

    def test_set_legal_moves_provider(self, tool):
        """Test set_legal_moves_provider stores provider."""
        provider = lambda: ["a", "b"]
        tool.set_legal_moves_provider(provider)
        assert tool._get_legal_moves() == ["a", "b"]

    def test_legal_moves_provider_takes_precedence(self, tool):
        """Test provider takes precedence over cached moves."""
        tool.set_legal_moves(["cached1", "cached2"])
        tool.set_legal_moves_provider(lambda: ["provider1", "provider2"])

        assert tool._get_legal_moves() == ["provider1", "provider2"]

    def test_legal_moves_provider_error_falls_back(self, tool):
        """Test fallback to cached moves on provider error."""
        tool.set_legal_moves(["fallback"])
        tool.set_legal_moves_provider(lambda: (_ for _ in ()).throw(Exception("fail")))

        # Should fallback to cached moves (not crash)
        moves = tool._get_legal_moves()
        assert moves == ["fallback"]

    @pytest.mark.asyncio
    async def test_execute_empty_legal_moves_allows_any(self, tool):
        """Test execute with empty legal moves list allows any move."""
        # When no legal moves are set/provided, any move is allowed
        result = await tool.execute(
            move="any_move",
            game_id="test",
        )

        assert "Error" not in result
        assert "executed successfully" in result.lower()

    def test_to_schema(self, tool):
        """Test to_schema returns proper OpenAI format."""
        schema = tool.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "game_move"
        assert "parameters" in schema["function"]
