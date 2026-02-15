"""MCP protocol schemas and types."""

from dataclasses import dataclass
from typing import Any


@dataclass
class MCPToolSchema:
    """
    Schema for a tool exposed by an MCP server.

    Represents the tool definition in MCP format, including
    its name, description, and input schema.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    def __post_init__(self):
        """Validate the schema after initialization."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not isinstance(self.input_schema, dict):
            raise ValueError("Input schema must be a dictionary")


@dataclass
class MCPExecutionRequest:
    """
    Request to execute a tool on an MCP server.

    Contains the tool name and arguments to pass to the tool.
    """

    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class MCPExecutionResponse:
    """
    Response from executing a tool on an MCP server.

    Contains the result content and any metadata.
    """

    content: list[dict[str, Any]]
    is_error: bool = False

    def to_string(self) -> str:
        """Convert response content to a string representation."""
        if self.is_error:
            # Extract error message
            if self.content:
                error_msg = self.content[0].get("text", str(self.content))
                return f"Error: {error_msg}"
            return "Error: Unknown error occurred"

        # Concatenate all text content
        texts = []
        for item in self.content:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            elif isinstance(item, dict):
                # Try to extract any text-like content
                texts.append(str(item.get("content", str(item))))
            else:
                texts.append(str(item))

        return "\n".join(texts) if texts else ""


@dataclass
class MCPServerConfig:
    """
    Configuration for an MCP server connection.

    Specifies how to connect to an MCP server (command, args, environment).
    """

    name: str
    type: str  # "local" or "stdio"
    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None

    def __post_init__(self):
        """Set default values."""
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}
