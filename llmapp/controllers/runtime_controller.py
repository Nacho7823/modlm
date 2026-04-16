"""Runtime and configuration actions for ChatApp."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmlib.models import LLMSettings, MCPServerConfig

if TYPE_CHECKING:
    from llmapp.app import ChatApp


class RuntimeController:
    """Coordinates app config persistence and llmlib wiring."""

    def __init__(self, app: "ChatApp") -> None:
        self._app = app

    async def shutdown(self) -> None:
        """Cleanup all runtime resources."""
        await self._app.runtime.shutdown()

    async def configure_runtime(self) -> None:
        api_url, api_key, model = self._app.config_manager.get_api_config()
        settings = LLMSettings(api_url=api_url, api_key=api_key, model=model)
        warnings = await self._app.runtime.configure(
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
            for name, cfg in mcp_servers.items():
                target = cfg.url if cfg.server_type == "remote" else " ".join(cfg.command)
                lines.append(f"  - {name} ({cfg.server_type}): {target}")
        self._app._add_message("system", "\n".join(lines))

    def list_mcp_servers(self) -> dict[str, MCPServerConfig]:
        return self._app.runtime.list_mcp_servers()

    async def add_mcp_server(self, config: MCPServerConfig) -> str:
        self._app.config_manager.add_mcp_server(config)
        return await self._app.runtime.add_mcp_server(config)

    async def remove_mcp_server(self, name: str) -> str:
        if self._app.config_manager.remove_mcp_server(name):
            await self._app.runtime.remove_mcp_server(name)
            return f"MCP server '{name}' removed."
        return f"MCP server '{name}' not found."

    def set_streaming(self, enabled: bool) -> None:
        self._app.config_manager.set("streaming", enabled)
        self._app.config_manager.save()
