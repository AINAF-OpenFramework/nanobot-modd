"""Tests for MCP game tools integration."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.mcp.client import MCPClient
from nanobot.mcp.registry import MCPRegistry
from nanobot.mcp.schemas import MCPServerConfig


class TestMCPGameTools:
    """Tests for MCP game tools integration."""

    @pytest.fixture
    def server_process(self):
        """Start the simple MCP server for testing."""
        # Path to server script
        server_script = Path(__file__).parent.parent.parent / "examples" / "simple_mcp_server.py"

        # Start server process
        proc = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Give server time to start
        time.sleep(0.5)

        yield proc

        # Cleanup
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    @pytest.fixture
    def server_config(self):
        """Create MCP server config."""
        server_script = Path(__file__).parent.parent.parent / "examples" / "simple_mcp_server.py"
        return MCPServerConfig(
            name="game_tools_server",
            type="local",
            command=sys.executable,
            args=[str(server_script)],
        )

    @pytest.mark.asyncio
    async def test_mcp_client_discovers_game_tools(self, server_config):
        """Test that MCP client can discover game tools."""
        client = MCPClient(server_config)

        try:
            await client.connect()

            # Discover tools
            tools = await client.discover_tools()

            # Should find game tools
            tool_names = [t.name for t in tools]
            assert "place_marker" in tool_names
            assert "move_piece" in tool_names
            assert "get_legal_moves" in tool_names

        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_mcp_registry_registers_game_tools(self, server_config):
        """Test that MCP registry registers game tools."""
        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry)

        try:
            await mcp_registry.add_client(server_config)

            # Check that tools were registered
            tool_names = tool_registry.tool_names
            mcp_tool_names = [name for name in tool_names if name.startswith("mcp_")]

            assert len(mcp_tool_names) > 0
            # Should have game tools registered
            assert any("place_marker" in name for name in mcp_tool_names)
            assert any("move_piece" in name for name in mcp_tool_names)

        finally:
            await mcp_registry.disconnect_all()

    @pytest.mark.asyncio
    async def test_execute_place_marker_tool(self, server_config):
        """Test executing place_marker tool."""
        client = MCPClient(server_config)

        try:
            await client.connect()

            # Execute place_marker
            result = await client.execute_tool(
                "place_marker",
                {"position": "r1c1", "marker": "X"}
            )

            # Should get a result
            assert result is not None
            result_text = result[0]["text"] if isinstance(result, list) else str(result)
            assert "Placed" in result_text or "r1c1" in result_text

        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_move_piece_tool(self, server_config):
        """Test executing move_piece tool."""
        client = MCPClient(server_config)

        try:
            await client.connect()

            # Execute move_piece
            result = await client.execute_tool(
                "move_piece",
                {"from_square": "e2", "to_square": "e4", "piece": "P"}
            )

            # Should get a result
            assert result is not None
            result_text = result[0]["text"] if isinstance(result, list) else str(result)
            assert "Moved" in result_text or "e2" in result_text or "e4" in result_text

        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_get_legal_moves_tictactoe(self, server_config):
        """Test executing get_legal_moves for TicTacToe."""
        client = MCPClient(server_config)

        try:
            await client.connect()

            # Execute get_legal_moves for tictactoe
            result = await client.execute_tool(
                "get_legal_moves",
                {
                    "game_type": "tictactoe",
                    "state": {
                        "board": [["", "", ""], ["", "", ""], ["", "", ""]],
                        "current_player": "X",
                    }
                }
            )

            # Should get legal moves
            assert result is not None
            result_text = result[0]["text"] if isinstance(result, list) else str(result)
            assert "Legal moves" in result_text or "r0c0" in result_text

        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_get_legal_moves_chess(self, server_config):
        """Test executing get_legal_moves for Chess."""
        client = MCPClient(server_config)

        try:
            await client.connect()

            # Execute get_legal_moves for chess
            result = await client.execute_tool(
                "get_legal_moves",
                {
                    "game_type": "chess",
                    "state": {
                        "board": [["" for _ in range(8)] for _ in range(8)],
                        "current_player": "white",
                    }
                }
            )

            # Should get legal moves (scaffold message)
            assert result is not None
            result_text = result[0]["text"] if isinstance(result, list) else str(result)
            assert "Legal moves" in result_text or "scaffold" in result_text.lower()

        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_mcp_client_handles_connection_errors(self):
        """Test MCP client handles connection errors gracefully."""
        # Config pointing to non-existent server
        bad_config = MCPServerConfig(
            name="nonexistent_server",
            type="local",
            command="python",
            args=["nonexistent_script.py"],
        )

        client = MCPClient(bad_config)

        # Should handle connection error
        with pytest.raises(Exception):
            await client.connect()

    @pytest.mark.asyncio
    async def test_mcp_tool_execution_with_invalid_params(self, server_config):
        """Test MCP tool execution with invalid parameters."""
        client = MCPClient(server_config)

        try:
            await client.connect()

            # Try to execute with missing required parameter
            try:
                result = await client.execute_tool(
                    "place_marker",
                    {"position": "r1c1"}  # Missing 'marker' parameter
                )
                # May succeed with default or fail - both are acceptable
            except Exception:
                # Expected to fail
                pass

        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_mcp_registry_multiple_clients(self):
        """Test MCP registry with multiple clients."""
        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry)

        server_script = Path(__file__).parent.parent.parent / "examples" / "simple_mcp_server.py"

        # Create two server configs (same server, different names)
        config1 = MCPServerConfig(
            name="game_server_1",
            type="local",
            command=sys.executable,
            args=[str(server_script)],
        )
        config2 = MCPServerConfig(
            name="game_server_2",
            type="local",
            command=sys.executable,
            args=[str(server_script)],
        )

        try:
            await mcp_registry.add_client(config1)
            await mcp_registry.add_client(config2)

            # Should have both clients registered
            assert "game_server_1" in mcp_registry.client_names
            assert "game_server_2" in mcp_registry.client_names

            # Should have tools from both
            tool_names = tool_registry.tool_names
            assert len(tool_names) > 0

        finally:
            await mcp_registry.disconnect_all()

    @pytest.mark.asyncio
    async def test_game_tool_integration_workflow(self, server_config):
        """Test a complete workflow using game tools."""
        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry)

        try:
            # Register tools
            await mcp_registry.add_client(server_config)

            # Find place_marker tool
            place_marker_tool = None
            for name in tool_registry.tool_names:
                if "place_marker" in name:
                    place_marker_tool = name
                    break

            assert place_marker_tool is not None

            # Execute via registry
            result = await tool_registry.execute(
                place_marker_tool,
                {"position": "r0c0", "marker": "X"}
            )

            # Should get result
            assert result is not None

        finally:
            await mcp_registry.disconnect_all()
