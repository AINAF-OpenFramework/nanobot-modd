"""MCP Registry for managing MCP clients and tool registration."""

import asyncio

from loguru import logger

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.mcp.client import MCPClient
from nanobot.mcp.schemas import MCPServerConfig
from nanobot.mcp.tool_adapter import MCPToolAdapter


class MCPRegistry:
    """
    Registry for managing MCP clients and integrating with ToolRegistry.

    Handles the lifecycle of MCP connections, tool discovery,
    and registration of MCP tools into Nanobot's ToolRegistry.
    """

    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize the MCP registry.

        Args:
            tool_registry: Nanobot's tool registry to register MCP tools into
        """
        self._tool_registry = tool_registry
        self._clients: dict[str, MCPClient] = {}
        self._registered_tools: dict[str, list[str]] = {}  # client_name -> tool names

    async def add_client(self, config: MCPServerConfig) -> None:
        """
        Add and connect a new MCP client.

        Args:
            config: Server configuration

        Raises:
            RuntimeError: If connection or tool discovery fails
        """
        client_name = config.name

        # Check if already registered
        if client_name in self._clients:
            logger.warning(f"MCP client '{client_name}' already registered")
            return

        try:
            # Create and connect client
            client = MCPClient(config)
            await client.connect()

            # Register tools
            await self.register_mcp_tools(client)

            # Store client
            self._clients[client_name] = client

            logger.info(f"MCP client '{client_name}' added successfully")

        except Exception as e:
            logger.error(f"Failed to add MCP client '{client_name}': {e}")
            raise

    async def register_mcp_tools(self, client: MCPClient) -> None:
        """
        Discover and register tools from an MCP client.

        Args:
            client: Connected MCP client
        """
        client_name = client.config.name

        try:
            # Discover tools
            tools = await client.discover_tools()

            if not tools:
                logger.warning(f"No tools discovered from MCP client '{client_name}'")
                return

            # Wrap and register each tool
            registered = []
            for tool_schema in tools:
                try:
                    adapter = MCPToolAdapter(client, tool_schema)
                    self._tool_registry.register(adapter)
                    registered.append(adapter.name)
                    logger.debug(f"Registered MCP tool: {adapter.name}")
                except Exception as e:
                    logger.warning(f"Failed to register tool '{tool_schema.name}': {e}")

            # Track registered tools
            self._registered_tools[client_name] = registered

            logger.info(
                f"Registered {len(registered)} tools from MCP client '{client_name}'"
            )

        except Exception as e:
            logger.error(f"Failed to register tools from '{client_name}': {e}")
            raise

    def unregister_tools(self, client_name: str) -> None:
        """
        Unregister all tools from a specific MCP client.

        Args:
            client_name: Name of the MCP client
        """
        if client_name not in self._registered_tools:
            logger.warning(f"No tools registered for client '{client_name}'")
            return

        tool_names = self._registered_tools[client_name]
        for tool_name in tool_names:
            self._tool_registry.unregister(tool_name)

        del self._registered_tools[client_name]
        logger.info(f"Unregistered {len(tool_names)} tools from '{client_name}'")

    async def refresh(self) -> None:
        """
        Re-discover and update tools from all connected clients.

        This can be used to pick up new tools or changes in tool schemas.
        """
        logger.info("Refreshing MCP tools from all clients")

        for client_name, client in self._clients.items():
            try:
                # Unregister old tools
                self.unregister_tools(client_name)

                # Re-discover and register
                await self.register_mcp_tools(client)

            except Exception as e:
                logger.error(f"Failed to refresh tools from '{client_name}': {e}")

    async def connect_all(self) -> None:
        """
        Connect all MCP clients (called from AgentLoop.run()).

        This is a no-op if clients are already connected during add_client.
        """
        # Clients are connected during add_client, so this is mainly
        # for reconnection logic if needed in the future
        pass

    async def disconnect_all(self) -> None:
        """
        Disconnect all MCP clients and cleanup.

        Should be called when shutting down the AgentLoop.
        """
        logger.info("Disconnecting all MCP clients")

        # Disconnect all clients
        disconnect_tasks = []
        for client_name, client in self._clients.items():
            self.unregister_tools(client_name)
            disconnect_tasks.append(client.disconnect())

        # Wait for all disconnections
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        self._clients.clear()
        logger.info("All MCP clients disconnected")

    @property
    def client_names(self) -> list[str]:
        """Get list of registered client names."""
        return list(self._clients.keys())

    def get_client(self, name: str) -> MCPClient | None:
        """
        Get an MCP client by name.

        Args:
            name: Client name

        Returns:
            MCPClient instance or None if not found
        """
        return self._clients.get(name)
