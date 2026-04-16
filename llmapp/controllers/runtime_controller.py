"""Runtime and configuration actions for ChatApp."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmlib.runtime import LLMSettings

if TYPE_CHECKING:
    from llmapp.app import ChatApp


class RuntimeController:
    """Coordinates app config persistence and llmlib runtime wiring."""

    def __init__(self, app: "ChatApp") -> None:
        self._app = app

    def configure_runtime(self) -> None:
        api_url, api_key, model = self._app.config_manager.get_api_config()
        settings = LLMSettings(api_url=api_url, api_key=api_key, model=model)
        warnings = self._app.runtime.configure(
            settings,
            self._app.config_manager.get_mcp_servers(),
        )
        for warning in warnings:
            self._app._add_message("system", warning)

    def show_config(self) -> None:
        api_url, api_key, model = self._app.config_manager.get_api_config()
        mcp_servers = self._app.runtime.list_mcp_servers()
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
        self._app._add_message("system", "\n".join(lines))

    def list_mcp_servers(self) -> dict[str, str]:
        return self._app.runtime.list_mcp_servers()

    def add_mcp_server(self, name: str, url: str) -> str:
        self._app.config_manager.add_mcp_server(name, url)
        return self._app.runtime.add_mcp_server(name, url)

    def remove_mcp_server(self, name: str) -> str:
        if self._app.config_manager.remove_mcp_server(name):
            self._app.runtime.remove_mcp_server(name)
            return f"MCP server '{name}' removed."
        return f"MCP server '{name}' not found."

    def set_streaming(self, enabled: bool) -> None:
        self._app.config_manager.set("streaming", enabled)
        self._app.config_manager.save()
