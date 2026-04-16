"""MCP registry for runtime operations."""

from __future__ import annotations

from typing import Any

from llmlib.mcp import HTTPMCPClient


class MCPRegistry:
    """Tracks configured MCP servers and in-memory clients."""

    def __init__(self) -> None:
        self._servers: dict[str, str] = {}
        self._clients: dict[str, Any] = {}

    def load_servers(self, servers: dict[str, str]) -> list[str]:
        """Load server definitions and instantiate clients.

        Returns warning messages for failed client initialization.
        """
        self._servers = dict(servers)
        self._clients = {}
        warnings: list[str] = []
        for name, url in self._servers.items():
            try:
                self._clients[name] = HTTPMCPClient(name=name, url=url)
            except Exception as error:
                warnings.append(f"Failed to connect to MCP server '{name}': {error}")
        return warnings

    def list_servers(self) -> dict[str, str]:
        """Return configured servers."""
        return dict(self._servers)

    def list_clients(self) -> dict[str, Any]:
        """Return MCP client map."""
        return dict(self._clients)

    def add_server(self, name: str, url: str) -> str:
        """Add a server and instantiate a client."""
        self._servers[name] = url
        try:
            self._clients[name] = HTTPMCPClient(name=name, url=url)
            return f"MCP server '{name}' added."
        except Exception as error:
            return f"MCP server '{name}' added but failed to connect: {error}"

    def remove_server(self, name: str) -> bool:
        """Remove a server/client by name."""
        if name not in self._servers:
            return False
        del self._servers[name]
        self._clients.pop(name, None)
        return True
