"""MCP Manager - central service for managing MCP server connections."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from llmlib import (
    HTTPMCPClient,
    MCPClientBase,
    LocalMCPClient,
    MCPToolResult,
    RemoteMCPClient,
    ToolSchema,
    create_mcp_client,
)

from ..core.config import settings
from ..core.exceptions import StorageException

logger = logging.getLogger(__name__)


class MCPManager:
    """Central manager for MCP server connections."""

    def __init__(self):
        self._clients: dict[str, MCPClientBase] = {}
        self._server_configs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def register_server(self, config: dict[str, Any]) -> None:
        """Register an MCP server configuration.

        Args:
            config: Dictionary with server configuration (name, type, command/url, etc.)
        """
        name = config.get("name")
        if not name:
            raise ValueError("Server name is required")

        self._server_configs[name] = config
        logger.info(f"Registered MCP server config: {name}")

    def unregister_server(self, name: str) -> None:
        """Unregister an MCP server.

        Args:
            name: Name of the server to unregister
        """
        if name in self._server_configs:
            del self._server_configs[name]

        if name in self._clients:
            asyncio.create_task(self._clients[name].disconnect())
            del self._clients[name]

        logger.info(f"Unregistered MCP server: {name}")

    async def connect(self, name: str, connect_timeout: int = 10) -> bool:
        """Connect to an MCP server.

        Args:
            name: Name of the server to connect to
            connect_timeout: Timeout in seconds for connection

        Returns:
            True if connection successful
        """
        config = self._server_configs.get(name)
        if not config:
            raise ValueError(f"MCP server '{name}' not registered")

        async with self._lock:
            if name in self._clients:
                await self._clients[name].disconnect()

            timeout = config.get("timeout", 30)
            client = create_mcp_client(
                name=name,
                server_type=config.get("type", "local"),
                timeout=timeout,
                command=config.get("command"),
                url=config.get("url"),
                headers=config.get("headers"),
                env=config.get("env"),
            )

            success = await client.connect(connect_timeout=connect_timeout)
            if success:
                self._clients[name] = client
                logger.info(f"Connected to MCP server: {name}")
            else:
                logger.error(f"Failed to connect to MCP server: {name}")

            return success

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server.

        Args:
            name: Name of the server to disconnect from
        """
        async with self._lock:
            if name in self._clients:
                await self._clients[name].disconnect()
                del self._clients[name]
                logger.info(f"Disconnected from MCP server: {name}")

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all enabled MCP servers.

        Returns:
            Dictionary mapping server names to connection status
        """
        results = {}

        for name, config in self._server_configs.items():
            if config.get("enabled", True):
                try:
                    results[name] = await self.connect(name)
                except Exception as e:
                    logger.error(f"Failed to connect to {name}: {e}")
                    results[name] = False

        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        async with self._lock:
            for name in list(self._clients.keys()):
                await self._clients[name].disconnect()
            self._clients.clear()

    def list_servers(self) -> list[dict[str, Any]]:
        """List all registered server configurations.

        Returns:
            List of server configuration dictionaries
        """
        return list(self._server_configs.values())

    def get_server(self, name: str) -> Optional[dict[str, Any]]:
        """Get a server configuration by name.

        Args:
            name: Name of the server

        Returns:
            Server configuration or None
        """
        return self._server_configs.get(name)

    def get_client(self, name: str) -> Optional[MCPClientBase]:
        """Get a connected MCP client.

        Args:
            name: Name of the server

        Returns:
            MCP client or None if not connected
        """
        return self._clients.get(name)

    def is_connected(self, name: str) -> bool:
        """Check if connected to a server.

        Args:
            name: Name of the server

        Returns:
            True if connected
        """
        client = self._clients.get(name)
        return client is not None and client.is_connected

    async def list_tools(self, name: str) -> list[ToolSchema]:
        """List tools available on an MCP server.

        Args:
            name: Name of the server

        Returns:
            List of available tools
        """
        client = self._clients.get(name)
        if not client or not client.is_connected:
            raise RuntimeError(f"Not connected to MCP server '{name}'")

        return await client.list_tools()

    async def call_tool(
        self, name: str, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        """Call a tool on an MCP server.

        Args:
            name: Name of the server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        client = self._clients.get(name)
        if not client or not client.is_connected:
            raise RuntimeError(f"Not connected to MCP server '{name}'")

        return await client.call_tool(tool_name, arguments)

    async def test_connection(self, name: str) -> dict[str, Any]:
        """Test connection to an MCP server.

        Args:
            name: Name of the server

        Returns:
            Dictionary with status, tools, and error (if any)
        """
        logger.info(f"Testing connection to MCP server: {name}")

        config = self._server_configs.get(name)
        if not config:
            logger.error(
                f"Config not found for '{name}'. Available: {list(self._server_configs.keys())}"
            )
            return {
                "status": "error",
                "error": f"Server '{name}' not found in config",
                "tools": [],
                "tool_count": 0,
            }

        try:
            if not self.is_connected(name):
                config_timeout = config.get("timeout", 30)
                connect_timeout = min(config_timeout, 10)
                success = await self.connect(name, connect_timeout=connect_timeout)
                if not success:
                    return {
                        "status": "error",
                        "error": "Connection failed - server did not respond",
                        "tools": [],
                        "tool_count": 0,
                    }

            tools = await self.list_tools(name)
            tool_names = [t.name for t in tools]
            logger.info(
                f"Successfully connected to {name}, found {len(tool_names)} tools"
            )
            return {
                "status": "connected",
                "tools": tool_names,
                "tool_count": len(tools),
            }
        except Exception as e:
            logger.error(f"Connection test failed for {name}: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "tools": [], "tool_count": 0}


mcp_manager = MCPManager()


def get_mcp_manager() -> MCPManager:
    """Return the MCP manager singleton."""
    return mcp_manager


__all__ = ["MCPManager", "mcp_manager", "get_mcp_manager"]
