"""MCP Server to expose Nanobot tools via MCP protocol."""

import asyncio
import json
import sys
from typing import Any

from loguru import logger

from nanobot.agent.tools.registry import ToolRegistry


class MCPServer:
    """
    MCP Server that exposes Nanobot tools to external MCP clients.

    Implements the MCP protocol for tool listing and execution,
    allowing external tools to use Nanobot's capabilities.
    """

    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize the MCP server.

        Args:
            tool_registry: ToolRegistry containing tools to expose
        """
        self._registry = tool_registry
        self._running = False

    async def start(self) -> None:
        """
        Start the MCP server (stdio mode).

        Reads JSON-RPC messages from stdin and writes responses to stdout.
        """
        self._running = True
        logger.info("MCP Server starting (stdio mode)")

        try:
            while self._running:
                # Read message from stdin
                line = await self._read_line()
                if not line:
                    break

                try:
                    message = json.loads(line)
                    response = await self._handle_message(message)

                    # Write response if this was a request (not a notification)
                    if response and "id" in message:
                        await self._write_message(response)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error",
                        },
                    }
                    await self._write_message(error_response)

        except Exception as e:
            logger.error(f"MCP Server error: {e}")
        finally:
            logger.info("MCP Server stopped")

    async def stop(self) -> None:
        """Stop the MCP server."""
        self._running = False

    async def _handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """
        Handle an incoming JSON-RPC message.

        Args:
            message: Parsed JSON-RPC message

        Returns:
            Response message or None for notifications
        """
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params", {})

        # Handle initialize
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": "nanobot-mcp-server",
                        "version": "0.1.0",
                    },
                },
            }

        # Handle initialized notification
        if method == "notifications/initialized":
            logger.info("Client initialized")
            return None

        # Handle tools/list
        if method == "tools/list":
            tools = self._list_tools()
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": tools,
                },
            }

        # Handle tools/call
        if method == "tools/call":
            result = await self._call_tool(params)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result,
            }

        # Unknown method
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }

    def _list_tools(self) -> list[dict[str, Any]]:
        """
        List all available tools in MCP format.

        Returns:
            List of tool definitions
        """
        tools = []
        for tool_name in self._registry.tool_names:
            tool = self._registry.get(tool_name)
            if tool:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.parameters,
                })

        return tools

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool and return the result.

        Args:
            params: Tool call parameters (name, arguments)

        Returns:
            Tool execution result in MCP format
        """
        try:
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if not tool_name:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Error: Missing tool name",
                        }
                    ],
                    "isError": True,
                }

            # Execute the tool
            result = await self._registry.execute(tool_name, arguments)

            # Check if result is an error
            is_error = result.startswith("Error:")

            return {
                "content": [
                    {
                        "type": "text",
                        "text": result,
                    }
                ],
                "isError": is_error,
            }

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {str(e)}",
                    }
                ],
                "isError": True,
            }

    async def _read_line(self) -> str:
        """
        Read a line from stdin asynchronously.

        Returns:
            Line string or empty string on EOF
        """
        loop = asyncio.get_event_loop()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        return line.strip()

    async def _write_message(self, message: dict[str, Any]) -> None:
        """
        Write a JSON-RPC message to stdout.

        Args:
            message: Message to write
        """
        line = json.dumps(message) + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()
