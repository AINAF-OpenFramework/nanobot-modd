"""Configuration loader for MCP servers from YAML."""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from nanobot.mcp.schemas import MCPServerConfig


class MCPConfigLoader:
    """Loads and validates MCP server configurations from YAML."""

    @staticmethod
    def load(config_path: Path) -> list[MCPServerConfig]:
        """
        Load MCP server configurations from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            List of validated MCPServerConfig instances

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"MCP config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                logger.warning(f"Empty MCP config file: {config_path}")
                return []

            # Validate and parse server configs
            servers_data = config_data.get("mcp_servers", [])
            if not isinstance(servers_data, list):
                raise ValueError("'mcp_servers' must be a list")

            servers = []
            for idx, server_data in enumerate(servers_data):
                try:
                    if not MCPConfigLoader.validate(server_data):
                        logger.warning(f"Skipping invalid server config at index {idx}")
                        continue

                    server = MCPServerConfig(
                        name=server_data["name"],
                        type=server_data.get("type", "local"),
                        command=server_data["command"],
                        args=server_data.get("args"),
                        env=server_data.get("env"),
                    )
                    servers.append(server)

                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid server config at index {idx}: {e}")

            logger.info(f"Loaded {len(servers)} MCP server configs from {config_path}")
            return servers

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load MCP config: {e}") from e

    @staticmethod
    def validate(config: dict[str, Any]) -> bool:
        """
        Validate a single MCP server configuration.

        Args:
            config: Server configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        # Required fields
        if not isinstance(config, dict):
            logger.error("Server config must be a dictionary")
            return False

        if "name" not in config:
            logger.error("Server config missing required field 'name'")
            return False

        if "command" not in config:
            logger.error("Server config missing required field 'command'")
            return False

        # Validate name
        name = config["name"]
        if not isinstance(name, str) or not name.strip():
            logger.error(f"Server 'name' must be a non-empty string: {name}")
            return False

        # Validate command
        command = config["command"]
        if not isinstance(command, str) or not command.strip():
            logger.error(f"Server 'command' must be a non-empty string: {command}")
            return False

        # Validate optional fields
        if "type" in config:
            server_type = config["type"]
            if server_type not in ("local", "stdio"):
                logger.error(f"Server 'type' must be 'local' or 'stdio': {server_type}")
                return False

        if "args" in config:
            args = config["args"]
            if not isinstance(args, list):
                logger.error(f"Server 'args' must be a list: {args}")
                return False
            if not all(isinstance(arg, str) for arg in args):
                logger.error("All items in 'args' must be strings")
                return False

        if "env" in config:
            env = config["env"]
            if not isinstance(env, dict):
                logger.error(f"Server 'env' must be a dictionary: {env}")
                return False
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
                logger.error("All keys and values in 'env' must be strings")
                return False

        return True
