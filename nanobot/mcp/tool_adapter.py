"""Adapter to wrap MCP tools as Nanobot Tool interface."""

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.mcp.client import MCPClient
from nanobot.mcp.schemas import MCPToolSchema


class MCPToolAdapter(Tool):
    """
    Adapter that wraps an MCP tool as a Nanobot Tool.

    This allows MCP tools to be registered in the ToolRegistry
    and executed like any native Nanobot tool.
    """

    def __init__(self, mcp_client: MCPClient, tool_schema: MCPToolSchema):
        """
        Initialize the adapter.

        Args:
            mcp_client: Connected MCP client instance
            tool_schema: MCP tool schema from the server
        """
        self._client = mcp_client
        self._schema = tool_schema
        self._name = f"mcp_{mcp_client.config.name}_{tool_schema.name}"

    @property
    def name(self) -> str:
        """Get tool name (prefixed with mcp_<server>_)."""
        return self._name

    @property
    def description(self) -> str:
        """Get tool description from MCP schema."""
        base_desc = self._schema.description
        server_name = self._client.config.name
        return f"[MCP:{server_name}] {base_desc}"

    @property
    def parameters(self) -> dict[str, Any]:
        """
        Convert MCP input schema to Nanobot parameters format.

        MCP uses JSON Schema, which is compatible with Nanobot's
        parameter format, so we can return it directly.
        """
        # Ensure we have a proper object schema
        schema = self._schema.input_schema

        # If schema doesn't specify type, default to object
        if "type" not in schema:
            schema = {"type": "object", **schema}

        return schema

    async def execute(self, **kwargs: Any) -> str:
        """
        Execute the MCP tool via the client.

        Args:
            **kwargs: Tool parameters

        Returns:
            Tool execution result as string
        """
        try:
            # Check if client is still connected
            if not self._client.is_connected():
                return f"Error: MCP server '{self._client.config.name}' is not connected"

            # Execute the tool
            response = await self._client.execute_tool(
                self._schema.name,
                kwargs
            )

            # Convert response to string
            result = response.to_string()

            logger.debug(
                f"Executed MCP tool '{self._schema.name}' "
                f"on '{self._client.config.name}': {len(result)} chars"
            )

            return result

        except Exception as e:
            error_msg = f"Error executing MCP tool '{self._schema.name}': {str(e)}"
            logger.error(error_msg)
            return error_msg
