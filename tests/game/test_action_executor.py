"""Tests for game action executor module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.game.action_executor import GameActionExecutor


class TestGameActionExecutor:
    """Tests for GameActionExecutor class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock tool registry."""
        registry = MagicMock(spec=ToolRegistry)
        registry.execute = AsyncMock(return_value="Move executed successfully")
        return registry

    @pytest.fixture
    def executor(self, mock_registry):
        """Create an executor for testing."""
        return GameActionExecutor(
            registry=mock_registry,
            timeout_seconds=5,
            max_retries=3,
            rate_limit_delay=0.01,  # Short delay for tests
        )

    def test_init_sets_parameters(self, mock_registry):
        """Test initialization sets parameters."""
        executor = GameActionExecutor(
            registry=mock_registry,
            timeout_seconds=10,
            max_retries=5,
            rate_limit_delay=0.5,
        )

        assert executor.timeout_seconds == 10
        assert executor.max_retries == 5
        assert executor.rate_limit_delay == 0.5

    @pytest.mark.asyncio
    async def test_execute_move_success(self, executor, mock_registry):
        """Test successful move execution."""
        mock_registry.execute.return_value = "Move applied"

        result = await executor.execute_move(
            move="e2e4",
            game_id="test-game",
        )

        assert result["success"] is True
        assert result["result"] == "Move applied"
        assert result["attempts"] == 1
        mock_registry.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_move_with_additional_params(self, executor, mock_registry):
        """Test execute_move passes additional parameters."""
        await executor.execute_move(
            move="a1",
            game_id="game-1",
            additional_params={"player": "X"},
        )

        call_args = mock_registry.execute.call_args
        assert call_args[0][1]["move"] == "a1"
        assert call_args[0][1]["game_id"] == "game-1"
        assert call_args[0][1]["player"] == "X"

    @pytest.mark.asyncio
    async def test_execute_move_retries_on_error(self, executor, mock_registry):
        """Test execute_move retries on error response."""
        mock_registry.execute.side_effect = [
            "Error: temporary failure",
            "Error: temporary failure",
            "Success",
        ]

        result = await executor.execute_move(
            move="1",
            game_id="test",
        )

        assert result["success"] is True
        assert result["attempts"] == 3

    @pytest.mark.asyncio
    async def test_execute_move_fails_after_max_retries(self, executor, mock_registry):
        """Test execute_move fails after max retries."""
        mock_registry.execute.return_value = "Error: permanent failure"

        result = await executor.execute_move(
            move="1",
            game_id="test",
        )

        assert result["success"] is False
        assert result["attempts"] == 3
        assert "permanent failure" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_move_handles_timeout(self, executor, mock_registry):
        """Test execute_move handles timeout."""
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(10)
            return "Never returned"

        mock_registry.execute.side_effect = slow_execute
        executor.timeout_seconds = 0.1

        result = await executor.execute_move(
            move="1",
            game_id="test",
        )

        assert result["success"] is False
        assert "Timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_move_handles_exception(self, executor, mock_registry):
        """Test execute_move handles exceptions."""
        mock_registry.execute.side_effect = Exception("Unexpected error")

        result = await executor.execute_move(
            move="1",
            game_id="test",
        )

        assert result["success"] is False
        assert "Unexpected error" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_move_valid(self, executor):
        """Test validate_move with valid move."""
        result = await executor.validate_move(
            move="1",
            legal_moves=["0", "1", "2"],
        )

        assert result["valid"] is True
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_validate_move_invalid(self, executor):
        """Test validate_move with invalid move."""
        result = await executor.validate_move(
            move="3",
            legal_moves=["0", "1", "2"],
        )

        assert result["valid"] is False
        assert "Illegal move" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_move_no_legal_moves(self, executor):
        """Test validate_move with no legal moves."""
        result = await executor.validate_move(
            move="1",
            legal_moves=[],
        )

        assert result["valid"] is False
        assert "No legal moves" in result["error"]

    def test_handle_failure_returns_dict(self, executor):
        """Test handle_failure returns proper dict."""
        result = executor.handle_failure(
            move="a1",
            game_id="game-123",
            error="Test error",
            attempts=3,
        )

        assert result["success"] is False
        assert result["error"] == "Test error"
        assert result["attempts"] == 3
        assert result["move"] == "a1"
        assert result["game_id"] == "game-123"

    @pytest.mark.asyncio
    async def test_rate_limiting(self, executor, mock_registry):
        """Test rate limiting between executions."""
        executor.rate_limit_delay = 0.1
        mock_registry.execute.return_value = "ok"

        import time
        start = time.time()

        await executor.execute_move(move="1", game_id="g1")
        await executor.execute_move(move="2", game_id="g2")

        elapsed = time.time() - start
        assert elapsed >= 0.1  # At least one rate limit delay

    def test_get_backoff_delay(self, executor):
        """Test exponential backoff delay calculation."""
        assert executor._get_backoff_delay(1) == 1.0  # 2^0
        assert executor._get_backoff_delay(2) == 2.0  # 2^1
        assert executor._get_backoff_delay(3) == 4.0  # 2^2
        assert executor._get_backoff_delay(10) == 10.0  # Capped at 10

    @pytest.mark.asyncio
    async def test_custom_tool_name(self, executor, mock_registry):
        """Test execute_move with custom tool name."""
        mock_registry.execute.return_value = "Success"

        await executor.execute_move(
            move="1",
            game_id="test",
            tool_name="custom_move_tool",
        )

        call_args = mock_registry.execute.call_args
        assert call_args[0][0] == "custom_move_tool"
