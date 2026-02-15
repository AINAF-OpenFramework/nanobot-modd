#!/usr/bin/env python
"""
Example MCP server for testing Nanobot's MCP integration.

This is a minimal MCP server that exposes a simple "echo" tool.
It can be used to test that Nanobot can discover and execute MCP tools.

Usage:
    python examples/simple_mcp_server.py

Then configure in nanobot/config/mcp.yaml:
    mcp_servers:
      - name: echo_server
        type: local
        command: python
        args:
          - "examples/simple_mcp_server.py"
"""

import asyncio
import json
import sys


async def handle_request(request: dict) -> dict:
    """Handle an MCP protocol request."""
    method = request.get("method")

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "simple-echo-server",
                "version": "1.0.0",
            },
        }

    elif method == "tools/list":
        return {
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo back the input message",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message to echo back",
                            }
                        },
                        "required": ["message"],
                    },
                },
                {
                    "name": "uppercase",
                    "description": "Convert text to uppercase",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to convert",
                            }
                        },
                        "required": ["text"],
                    },
                },
                {
                    "name": "place_marker",
                    "description": "Place a marker (X or O) on a TicTacToe board",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "position": {
                                "type": "string",
                                "description": "Position in format 'r0c0', 'r1c2', etc.",
                            },
                            "marker": {
                                "type": "string",
                                "description": "Marker to place: 'X' or 'O'",
                                "enum": ["X", "O"],
                            },
                        },
                        "required": ["position", "marker"],
                    },
                },
                {
                    "name": "move_piece",
                    "description": "Move a chess piece from one square to another",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "from_square": {
                                "type": "string",
                                "description": "Starting square (e.g., 'e2')",
                            },
                            "to_square": {
                                "type": "string",
                                "description": "Destination square (e.g., 'e4')",
                            },
                            "piece": {
                                "type": "string",
                                "description": "Piece being moved (e.g., 'P', 'N', 'B', 'R', 'Q', 'K')",
                            },
                        },
                        "required": ["from_square", "to_square"],
                    },
                },
                {
                    "name": "get_legal_moves",
                    "description": "Get legal moves for a game position",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "game_type": {
                                "type": "string",
                                "description": "Type of game: 'tictactoe' or 'chess'",
                                "enum": ["tictactoe", "chess"],
                            },
                            "state": {
                                "type": "object",
                                "description": "Current game state",
                            },
                        },
                        "required": ["game_type", "state"],
                    },
                },
            ]
        }

    elif method == "tools/call":
        tool_name = request["params"]["name"]
        args = request["params"]["arguments"]

        # Execute the appropriate tool
        if tool_name == "echo":
            result = f"Echo: {args.get('message', '')}"
        elif tool_name == "uppercase":
            result = args.get("text", "").upper()
        elif tool_name == "place_marker":
            position = args.get("position", "")
            marker = args.get("marker", "")
            result = f"Placed {marker} at {position}"
        elif tool_name == "move_piece":
            from_sq = args.get("from_square", "")
            to_sq = args.get("to_square", "")
            piece = args.get("piece", "piece")
            result = f"Moved {piece} from {from_sq} to {to_sq}"
        elif tool_name == "get_legal_moves":
            game_type = args.get("game_type", "")
            # Simplified: return mock legal moves
            if game_type == "tictactoe":
                result = "Legal moves: r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2"
            elif game_type == "chess":
                result = "Legal moves: e2e4, d2d4, Nf3, Nc3 (scaffold - full chess not implemented)"
            else:
                result = f"Unknown game type: {game_type}"
        else:
            return {
                "content": [
                    {"type": "text", "text": f"Unknown tool: {tool_name}"}
                ],
                "isError": True,
            }

        return {
            "content": [
                {"type": "text", "text": result}
            ],
            "isError": False,
        }

    # Unknown method
    return {}


async def main():
    """Run the MCP server in stdio mode."""
    # Notify stderr that we're starting (doesn't interfere with stdio protocol)
    sys.stderr.write("Simple MCP Echo Server started\n")
    sys.stderr.flush()

    while True:
        # Read request from stdin
        line = await asyncio.get_event_loop().run_in_executor(
            None, sys.stdin.readline
        )

        if not line:
            break

        try:
            request = json.loads(line.strip())
        except json.JSONDecodeError as e:
            sys.stderr.write(f"JSON decode error: {e}\n")
            continue

        msg_id = request.get("id")

        # Handle request
        try:
            result = await handle_request(request)

            # Send response (only if request had an ID)
            if msg_id is not None:
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": result,
                }
                print(json.dumps(response), flush=True)
        except Exception as e:
            sys.stderr.write(f"Error handling request: {e}\n")
            if msg_id is not None:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32603,
                        "message": str(e),
                    },
                }
                print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.stderr.write("\nServer stopped\n")
