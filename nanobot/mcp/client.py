"""MCP client for connecting to and interacting with MCP servers."""

import asyncio
import json
import os
from typing import Any

from loguru import logger

from nanobot.mcp.schemas import (
    MCPExecutionRequest,
    MCPExecutionResponse,
    MCPServerConfig,
    MCPToolSchema,
)


class MCPClient:
    """
    Client for connecting to and interacting with MCP servers.
    
    Supports stdio-based MCP servers running as local processes.
    Handles tool discovery, execution, and lifecycle management.
    """
    
    def __init__(self, config: MCPServerConfig):
        """
        Initialize the MCP client.
        
        Args:
            config: Server configuration (command, args, env)
        """
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self._message_id = 0
        self._connected = False
        self._tools_cache: list[MCPToolSchema] = []
    
    async def connect(self) -> None:
        """
        Connect to the MCP server by spawning the process.
        
        Raises:
            RuntimeError: If connection fails
        """
        if self._connected:
            logger.warning(f"MCP client '{self.config.name}' already connected")
            return
        
        try:
            # Prepare environment
            env = os.environ.copy()
            if self.config.env:
                env.update(self.config.env)
            
            # Build command
            cmd_parts = self.config.command.split()
            if self.config.args:
                cmd_parts.extend(self.config.args)
            
            logger.info(f"Starting MCP server '{self.config.name}': {' '.join(cmd_parts)}")
            
            # Start process with stdio pipes
            self.process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            
            # Send initialization message
            init_msg = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "nanobot",
                        "version": "0.1.0",
                    },
                },
            }
            
            response = await self._send_request(init_msg)
            if "error" in response:
                raise RuntimeError(f"Initialization failed: {response['error']}")
            
            # Send initialized notification
            initialized_msg = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            await self._send_notification(initialized_msg)
            
            self._connected = True
            logger.info(f"MCP server '{self.config.name}' connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{self.config.name}': {e}")
            await self.disconnect()
            raise RuntimeError(f"Connection failed: {e}") from e
    
    async def discover_tools(self) -> list[MCPToolSchema]:
        """
        Discover available tools from the MCP server.
        
        Returns:
            List of tool schemas
        
        Raises:
            RuntimeError: If not connected or discovery fails
        """
        if not self._connected:
            raise RuntimeError("Not connected to MCP server")
        
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
            }
            
            response = await self._send_request(request)
            
            if "error" in response:
                raise RuntimeError(f"Tool discovery failed: {response['error']}")
            
            tools_data = response.get("result", {}).get("tools", [])
            
            # Parse tools into MCPToolSchema objects
            tools = []
            for tool_data in tools_data:
                try:
                    tool = MCPToolSchema(
                        name=tool_data["name"],
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                    )
                    tools.append(tool)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid tool schema: {e}")
            
            self._tools_cache = tools
            logger.info(f"Discovered {len(tools)} tools from '{self.config.name}'")
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to discover tools from '{self.config.name}': {e}")
            raise RuntimeError(f"Tool discovery failed: {e}") from e
    
    async def execute_tool(self, tool_name: str, parameters: dict[str, Any]) -> MCPExecutionResponse:
        """
        Execute a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
        
        Returns:
            Tool execution response
        
        Raises:
            RuntimeError: If not connected or execution fails
        """
        if not self._connected:
            raise RuntimeError("Not connected to MCP server")
        
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters,
                },
            }
            
            response = await self._send_request(request)
            
            if "error" in response:
                error_data = response["error"]
                return MCPExecutionResponse(
                    content=[{"text": str(error_data.get("message", "Unknown error"))}],
                    is_error=True,
                )
            
            result = response.get("result", {})
            content = result.get("content", [])
            
            return MCPExecutionResponse(content=content, is_error=False)
            
        except Exception as e:
            logger.error(f"Failed to execute tool '{tool_name}' on '{self.config.name}': {e}")
            return MCPExecutionResponse(
                content=[{"text": f"Execution failed: {str(e)}"}],
                is_error=True,
            )
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server and cleanup resources."""
        if not self._connected:
            return
        
        self._connected = False
        
        if self.process:
            try:
                # Try graceful shutdown
                if self.process.stdin:
                    self.process.stdin.close()
                
                # Wait briefly for process to exit
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    # Force kill if doesn't exit gracefully
                    self.process.kill()
                    await self.process.wait()
                
                logger.info(f"MCP server '{self.config.name}' disconnected")
                
            except Exception as e:
                logger.warning(f"Error during disconnect of '{self.config.name}': {e}")
            finally:
                self.process = None
    
    def is_connected(self) -> bool:
        """Check if the client is currently connected."""
        return self._connected and self.process is not None
    
    def _next_id(self) -> int:
        """Generate next message ID."""
        self._message_id += 1
        return self._message_id
    
    async def _send_request(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Send a JSON-RPC request and wait for response.
        
        Args:
            message: JSON-RPC message to send
        
        Returns:
            Response message
        """
        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not running")
        
        # Send request
        request_line = json.dumps(message) + "\n"
        self.process.stdin.write(request_line.encode("utf-8"))
        await self.process.stdin.drain()
        
        # Read response
        if not self.process.stdout:
            raise RuntimeError("Process stdout not available")
        
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("Connection closed by server")
        
        return json.loads(response_line.decode("utf-8"))
    
    async def _send_notification(self, message: dict[str, Any]) -> None:
        """
        Send a JSON-RPC notification (no response expected).
        
        Args:
            message: JSON-RPC notification to send
        """
        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not running")
        
        notification_line = json.dumps(message) + "\n"
        self.process.stdin.write(notification_line.encode("utf-8"))
        await self.process.stdin.drain()
