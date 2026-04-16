"""MCP registry for runtime operations."""

from __future__ import annotations

import asyncio
from llmlib.mcp import HTTPMCPClient, LocalMCPClient
from llmlib.models import MCPServerConfig, MCPConnectionError
from llmlib.mcp import create_mcp_client


class MCPRegistry:
    """Tracks configured MCP servers and creates short-lived clients."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, Any] = {}

    async def load_servers(self, servers: dict[str, MCPServerConfig]) -> list[str]:
        """Load server configs and return validation warnings."""
        self._servers = dict(servers)
        if not servers:
            return []
            
        # Run all validations concurrently
        results = await asyncio.gather(
            *(self._validate(cfg) for cfg in servers.values()),
            return_exceptions=True
        )
        
        warnings = []
        for res in results:
            if isinstance(res, list):
                warnings.extend(res)
            elif isinstance(res, Exception):
                warnings.append(str(res))
        return warnings

    def list_servers(self) -> dict[str, MCPServerConfig]:
        """Return configured servers."""
        return dict(self._servers)

    def get_clients(self) -> dict[str, Any]:
        """Get or build clients for all configured servers."""
        for name, cfg in self._servers.items():
            if name not in self._clients:
                self._clients[name] = self._build_client(cfg)
        return dict(self._clients)

    async def shutdown_all(self) -> None:
        """Gracefully disconnect all clients."""
        seen_ids = set()
        tasks = []
        for client in self._clients.values():
            client_id = id(client)
            if client_id in seen_ids:
                continue
            seen_ids.add(client_id)
            
            if hasattr(client, "disconnect"):
                tasks.append(client.disconnect())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._clients.clear()

    async def add_server(self, config: MCPServerConfig) -> str:
        """Add a server and validate it can be connected to."""
        self._servers[config.name] = config
        warnings = await self._validate(config)
        if warnings:
            return f"MCP server '{config.name}' added but failed to connect: {warnings[0]}"
        return f"MCP server '{config.name}' added."

    async def remove_server(self, name: str) -> bool:
        """Remove a server by name and disconnect its client."""
        if name not in self._servers:
            return False
            
        client = self._clients.pop(name, None)
        if client and hasattr(client, "disconnect"):
            try:
                await client.disconnect()
            except Exception:
                pass
                
        del self._servers[name]
        return True

    def _build_client(self, cfg: MCPServerConfig) -> object:
        if cfg.server_type == "local":
            return LocalMCPClient(name=cfg.name, command=cfg.command)
        return HTTPMCPClient(name=cfg.name, url=cfg.url, headers=cfg.headers)

    async def _validate(self, cfg: MCPServerConfig) -> list[str]:
        try:
            client = self._build_client(cfg)
            # Use a shorter timeout for validation to avoid blocking for too long
            status = await client.test_connection()
            if status["status"] == "error":
                return [status["error"]]
            return []
        except Exception as error:
            return [str(error)]
