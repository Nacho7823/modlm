"""Config service for llmapp."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmlib.llm import OpenAI
    from llmlib.mcp import HTTPMCPClient


class ConfigService:
    """Handles configuration loading and display."""

    def __init__(self, app: "Any") -> None:
        self._app = app

    def init_clients(self) -> None:
        """Initialize LLM clients from config."""
        from llmlib.llm import OpenAI

        api_url, api_key, model = self._app.config_manager.get_api_config()
        self._app.client = OpenAI(base_url=api_url, api_key=api_key, model=model)
        self._app.client_async = OpenAI(base_url=api_url, api_key=api_key, model=model)
        self._init_mcp_clients()

    def _create_client(self, api_url: str, api_key: str, model: str) -> "OpenAI":
        from llmlib.llm import OpenAI

        return OpenAI(base_url=api_url, api_key=api_key, model=model)

    def _init_mcp_clients(self) -> None:
        """Initialize MCP clients for each configured server."""
        from llmlib.mcp import HTTPMCPClient

        for name, url in self._app.config_manager.get_mcp_servers().items():
            try:
                client = HTTPMCPClient(name=name, url=url)
                self._app.mcp_clients[name] = client
            except Exception as e:
                self._app._add_system_message(
                    f"Failed to connect to MCP server '{name}': {e}"
                )

    def show(self) -> None:
        """Display current configuration."""
        api_url, api_key, model = self._app.config_manager.get_api_config()
        mcp_servers = self._app.config_manager.get_mcp_servers()
        lines = [
            "[bold]Current Configuration[/bold]",
            f"API URL: {api_url}",
            f"API Key: {'(set)' if api_key else '(not set)'}",
            f"Model: {model}",
            f"Streaming: {'on' if self._app.config_manager.get('streaming', True) else 'off'}",
        ]
        if mcp_servers:
            lines.append("MCP Servers:")
            for name, url in mcp_servers.items():
                lines.append(f"  - {name}: {url}")
        self._app._add_system_message("\n".join(lines))
