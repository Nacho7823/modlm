"""Factory for creating MCP clients based on configuration."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .client import HTTPMCPClient, LocalMCPClient, MCPClientBase, RemoteMCPClient
from llmlib.models import MCPServerConfig, MCPConnectionError

logger = logging.getLogger(__name__)


def create_mcp_client(
    name: str,
    server_type: str,
    timeout: int = 30,
    command: Optional[list[str]] = None,
    url: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    env: Optional[dict[str, str]] = None,
) -> MCPClientBase:
    """Factory function to create MCP clients based on type.

    Args:
        name: Name identifier for the MCP server
        server_type: Type of MCP server ("local", "stdio", "http", or "remote")
        timeout: Connection timeout in seconds
        command: Command and arguments for local/stdio servers
        url: URL for http/remote servers
        headers: HTTP headers for http/remote servers
        env: Environment variables for local/stdio servers

    Returns:
        An MCP client instance

    Raises:
        MCPConnectionError: If server_type is invalid or required parameters are missing
    """
    if server_type in ("local", "stdio"):
        if not command:
            raise MCPConnectionError("command is required for local MCP servers")
        return LocalMCPClient(name=name, command=command, timeout=timeout, env=env)

    if server_type in ("http", "remote"):
        if not url:
            raise MCPConnectionError("url is required for HTTP MCP servers")
        return HTTPMCPClient(name=name, url=url, timeout=timeout, headers=headers)

    raise MCPConnectionError(f"Unknown MCP server type: {server_type}")


def create_mcp_client_from_config(config: dict[str, Any]) -> MCPClientBase:
    """Create an MCP client from a configuration dictionary.

    Args:
        config: Configuration dictionary with keys:
            - name: Server name
            - type: "local", "stdio", "http", or "remote"
            - command: List of command parts (for local/stdio)
            - url: URL (for http/remote)
            - headers: HTTP headers (for http/remote)
            - env: Environment variables (for local/stdio)
            - timeout: Connection timeout

    Returns:
        An MCP client instance
    """
    return create_mcp_client(
        name=config.get("name", "unnamed"),
        server_type=config.get("type", "local"),
        timeout=config.get("timeout", 30),
        command=config.get("command"),
        url=config.get("url"),
        headers=config.get("headers"),
        env=config.get("env"),
    )
