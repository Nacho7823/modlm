"""MCP service for llmapp."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmlib.mcp import HTTPMCPClient


class MCPService:
    """Handles MCP server lifecycle management."""

    def __init__(self, app: "Any") -> None:
        self._app = app

    def list(self) -> dict[str, str]:
        """List all configured MCP servers."""
        return self._app.config_manager.get_mcp_servers()

    def add(self, name: str, url: str) -> str:
        """Add and connect an MCP server."""
        from llmlib.mcp import HTTPMCPClient

        self._app.config_manager.add_mcp_server(name, url)
        try:
            client = HTTPMCPClient(name=name, url=url)
            self._app.mcp_clients[name] = client
            return f"MCP server '{name}' added."
        except Exception as e:
            return f"MCP server '{name}' added but failed to connect: {e}"

    def remove(self, name: str) -> str:
        """Remove an MCP server."""
        if self._app.config_manager.remove_mcp_server(name):
            if name in self._app.mcp_clients:
                del self._app.mcp_clients[name]
            return f"MCP server '{name}' removed."
        return f"MCP server '{name}' not found."
