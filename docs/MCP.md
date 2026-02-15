# MCP (Model Context Protocol) Integration

## Overview

Nanobot now supports the Model Context Protocol (MCP), enabling seamless integration with external tools and services. MCP allows Nanobot to dynamically discover, connect to, and use tools from any MCP-compatible server.

## What is MCP?

The Model Context Protocol is an open protocol that standardizes how AI applications communicate with external tools and data sources. It enables:

- **Dynamic Tool Discovery**: Automatically discover available tools from MCP servers
- **Standardized Communication**: Consistent protocol for tool invocation
- **Extensibility**: Easy integration with new tools without code changes
- **Type Safety**: JSON Schema-based tool parameter validation

## Architecture

MCP integrates seamlessly with Nanobot's existing architecture:

```
Soul (.md intent) → Translator (.py) → YAML → AgentLoop → ToolRegistry → MCP Client → External MCP Servers
```

### Components

1. **MCPClient**: Manages connections to individual MCP servers
2. **MCPRegistry**: Coordinates multiple MCP clients and tool registration
3. **MCPToolAdapter**: Wraps MCP tools as native Nanobot tools
4. **MCPConfigLoader**: Loads server configurations from YAML
5. **MCPServer**: Exposes Nanobot's tools via MCP (optional)

## Configuration

MCP servers are configured in `nanobot/config/mcp.yaml`:

```yaml
mcp_servers:
  # Example: Filesystem server
  - name: filesystem
    type: local
    command: npx
    args:
      - "@modelcontextprotocol/server-filesystem"
      - "/workspace"
  
  # Example: Custom Python server
  - name: my_tools
    type: local
    command: python
    args:
      - "-m"
      - "my_mcp_server"
    env:
      API_KEY: "your-api-key-here"
```

### Configuration Fields

- **name** (required): Unique identifier for the server
- **type** (optional): Connection type - "local" or "stdio" (default: "local")
- **command** (required): Command to start the MCP server
- **args** (optional): List of command-line arguments
- **env** (optional): Environment variables for the server process

## Usage

### Automatic Initialization

MCP integration is automatically initialized when:
1. The `nanobot/config/mcp.yaml` file exists in your workspace
2. AgentLoop starts up

No code changes are required - just configure your MCP servers and they're ready to use!

### Tool Naming

MCP tools are automatically prefixed to avoid naming conflicts:
- Format: `mcp_<server_name>_<tool_name>`
- Example: `mcp_filesystem_read_file`

### Example: Using a Chess Engine

1. Create `nanobot/config/mcp.yaml`:
```yaml
mcp_servers:
  - name: stockfish
    type: local
    command: python
    args:
      - "-m"
      - "stockfish_mcp_server"
```

2. Start Nanobot - the chess engine tools are automatically available

3. Ask: "What's the best move in this position: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"

4. Nanobot uses the `mcp_stockfish_analyze_position` tool automatically

## Creating MCP Servers

You can create your own MCP servers to expose custom tools to Nanobot.

### Minimal Python MCP Server Example

```python
import asyncio
import json
import sys

async def handle_request(request):
    method = request.get("method")
    
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "my-server", "version": "1.0.0"},
        }
    
    elif method == "tools/list":
        return {
            "tools": [
                {
                    "name": "my_tool",
                    "description": "My custom tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"}
                        },
                        "required": ["input"]
                    }
                }
            ]
        }
    
    elif method == "tools/call":
        tool_name = request["params"]["name"]
        args = request["params"]["arguments"]
        
        # Implement your tool logic here
        result = f"Processed: {args.get('input', '')}"
        
        return {
            "content": [{"type": "text", "text": result}]
        }

async def main():
    while True:
        line = await asyncio.get_event_loop().run_in_executor(
            None, sys.stdin.readline
        )
        if not line:
            break
        
        request = json.loads(line)
        msg_id = request.get("id")
        
        result = await handle_request(request)
        
        if msg_id is not None:
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result
            }
            print(json.dumps(response), flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

## Exposing Nanobot Tools via MCP

You can expose Nanobot's tools to external MCP clients using `MCPServer`:

```python
from nanobot.mcp import MCPServer
from nanobot.agent.tools.registry import ToolRegistry

# Create registry with your tools
registry = ToolRegistry()
# ... register tools ...

# Start MCP server
server = MCPServer(registry)
await server.start()  # Runs in stdio mode
```

This allows other applications to use Nanobot's capabilities as MCP tools.

## Error Handling

MCP integration is designed to be resilient:

- **Connection Failures**: Logged as warnings, won't crash AgentLoop
- **Tool Discovery Errors**: Invalid tools are skipped with warnings
- **Execution Errors**: Returned as descriptive error strings
- **Disconnections**: Handled gracefully on shutdown

## Lifecycle Management

MCP clients are automatically managed:

1. **Startup**: Clients connect when `AgentLoop.run()` is called
2. **Runtime**: Tools are available throughout the agent's lifetime
3. **Shutdown**: Clients disconnect when the agent loop exits

## Advanced Features

### Dynamic Refresh

Refresh tools from all connected servers:

```python
if agent_loop.mcp_registry:
    await agent_loop.mcp_registry.refresh()
```

### Manual Client Management

```python
from nanobot.mcp import MCPRegistry, MCPServerConfig

registry = MCPRegistry(tool_registry)

# Add client
config = MCPServerConfig(
    name="my_server",
    type="local",
    command="python",
    args=["-m", "my_server"]
)
await registry.add_client(config)

# Remove tools
registry.unregister_tools("my_server")

# Disconnect
await registry.disconnect_all()
```

## Troubleshooting

### MCP Server Not Connecting

1. Check that the command in `mcp.yaml` is valid
2. Verify the server binary/script exists
3. Check logs for connection errors
4. Test the MCP server standalone

### Tools Not Appearing

1. Verify MCP server implements `tools/list` method
2. Check that tool schemas are valid JSON Schema
3. Look for warnings in logs about invalid tools
4. Ensure `mcp.yaml` is in the correct location

### Tool Execution Errors

1. Check parameter types match tool schema
2. Verify MCP server is still running
3. Review server logs for execution errors
4. Test tool execution directly with the MCP server

## Best Practices

1. **Validate Configuration**: Test MCP servers independently before integrating
2. **Use Descriptive Names**: Choose clear, unique names for servers and tools
3. **Handle Errors Gracefully**: MCP servers should return descriptive error messages
4. **Monitor Logs**: Check logs regularly for connection issues
5. **Version Management**: Keep MCP protocol version consistent across servers

## Security Considerations

1. **Command Validation**: Be cautious with commands in `mcp.yaml`
2. **Environment Variables**: Don't commit sensitive data in config files
3. **Tool Permissions**: Understand what capabilities each MCP tool has
4. **Network Access**: MCP servers may access networks/filesystems
5. **Process Isolation**: Each MCP server runs as a separate process

## References

- [MCP Specification](https://modelcontextprotocol.io/)
- [MCP SDK](https://github.com/modelcontextprotocol)
- [Example MCP Servers](https://github.com/modelcontextprotocol/servers)

## Support

For issues with MCP integration:
1. Check the troubleshooting section above
2. Review logs for error messages
3. Test MCP servers independently
4. Create an issue with server configuration and logs
