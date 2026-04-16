"""MCP registry for runtime operations."""

from __future__ import annotations

from typing import Any

from llmlib.mcp import HTTPMCPClient


class MCPRegistry:
    """Tracks configured MCP servers and creates short-lived clients."""

    def __init__(self) -> None:
        self._servers: dict[str, str] = {}

    def load_servers(self, servers: dict[str, str]) -> list[str]:
        """Load server definitions and validate they can be instantiated."""
        self._servers = dict(servers)
        warnings: list[str] = []
        for name, url in self._servers.items():
            try:
                HTTPMCPClient(name=name, url=url)
            except Exception as error:
                warnings.append(f"Failed to initialize MCP server '{name}': {error}")
        return warnings

    def list_servers(self) -> dict[str, str]:
        """Return configured servers."""
        return dict(self._servers)

    def build_clients(self) -> dict[str, Any]:
        """Build fresh MCP clients for the current operation."""
        clients: dict[str, Any] = {}
        for name, url in self._servers.items():
            clients[name] = HTTPMCPClient(name=name, url=url)
        return clients

    def add_server(self, name: str, url: str) -> str:
        """Add a server and validate it can be instantiated."""
        self._servers[name] = url
        try:
            HTTPMCPClient(name=name, url=url)
            return f"MCP server '{name}' added."
        except Exception as error:
            return f"MCP server '{name}' added but failed to connect: {error}"

    def remove_server(self, name: str) -> bool:
        """Remove a server/client by name."""
        if name not in self._servers:
            return False
        del self._servers[name]
        return True
