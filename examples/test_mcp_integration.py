#!/usr/bin/env python
"""
Test script to verify MCP integration works end-to-end.

This script tests:
1. Loading MCP configuration
2. Connecting to an MCP server
3. Discovering tools
4. Executing tools
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.mcp.client import MCPClient
from nanobot.mcp.registry import MCPRegistry
from nanobot.mcp.schemas import MCPServerConfig


async def test_mcp_integration():
    """Test MCP integration end-to-end."""
    print("=" * 60)
    print("Testing MCP Integration")
    print("=" * 60)

    # Step 1: Create a test MCP server config
    print("\n1. Creating MCP server configuration...")
    server_config = MCPServerConfig(
        name="test_echo",
        type="local",
        command="python",
        args=["examples/simple_mcp_server.py"],
    )
    print(f"   ✓ Config created: {server_config.name}")

    # Step 2: Connect to the MCP server
    print("\n2. Connecting to MCP server...")
    client = MCPClient(server_config)
    try:
        await client.connect()
        print(f"   ✓ Connected to {server_config.name}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False

    # Step 3: Discover tools
    print("\n3. Discovering tools...")
    try:
        tools = await client.discover_tools()
        print(f"   ✓ Discovered {len(tools)} tools:")
        for tool in tools:
            print(f"     - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"   ✗ Tool discovery failed: {e}")
        await client.disconnect()
        return False

    # Step 4: Test tool execution
    print("\n4. Testing tool execution...")
    try:
        # Test echo tool
        response = await client.execute_tool("echo", {"message": "Hello MCP!"})
        result = response.to_string()
        print(f"   ✓ Echo tool: {result}")

        # Test uppercase tool
        response = await client.execute_tool("uppercase", {"text": "hello world"})
        result = response.to_string()
        print(f"   ✓ Uppercase tool: {result}")
    except Exception as e:
        print(f"   ✗ Tool execution failed: {e}")
        await client.disconnect()
        return False

    # Step 5: Test ToolRegistry integration
    print("\n5. Testing ToolRegistry integration...")
    tool_registry = ToolRegistry()
    mcp_registry = MCPRegistry(tool_registry)

    try:
        # Register tools via MCP registry
        await mcp_registry.register_mcp_tools(client)
        print("   ✓ Registered MCP tools in ToolRegistry")

        # List registered tools
        tool_names = tool_registry.tool_names
        mcp_tools = [name for name in tool_names if name.startswith("mcp_")]
        print(f"   ✓ MCP tools in registry: {len(mcp_tools)}")
        for name in mcp_tools:
            print(f"     - {name}")

        # Execute via registry
        if mcp_tools:
            test_tool = mcp_tools[0]
            if "echo" in test_tool:
                result = await tool_registry.execute(test_tool, {"message": "Via registry!"})
                print(f"   ✓ Tool execution via registry: {result}")
    except Exception as e:
        print(f"   ✗ Registry integration failed: {e}")
        await client.disconnect()
        return False

    # Cleanup
    print("\n6. Cleaning up...")
    await mcp_registry.disconnect_all()
    print("   ✓ Disconnected from MCP server")

    print("\n" + "=" * 60)
    print("✅ All tests passed! MCP integration is working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_mcp_integration())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
