"""MCP (Model Context Protocol) integration for Nanobot."""

from nanobot.mcp.client import MCPClient
from nanobot.mcp.config_loader import MCPConfigLoader
from nanobot.mcp.registry import MCPRegistry
from nanobot.mcp.schemas import (
    MCPExecutionRequest,
    MCPExecutionResponse,
    MCPServerConfig,
    MCPToolSchema,
)
from nanobot.mcp.server import MCPServer
from nanobot.mcp.tool_adapter import MCPToolAdapter

__all__ = [
    "MCPClient",
    "MCPConfigLoader",
    "MCPRegistry",
    "MCPServer",
    "MCPToolAdapter",
    "MCPExecutionRequest",
    "MCPExecutionResponse",
    "MCPServerConfig",
    "MCPToolSchema",
]
