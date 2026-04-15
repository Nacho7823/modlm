import json
import logging
from pathlib import Path
from typing import Any, Optional

from ..core.config import settings
from ..core.exceptions import StorageException

logger = logging.getLogger(__name__)

DEFAULT_MCP_SERVERS: list[dict[str, Any]] = []


class ConfigStorage:
    def __init__(self):
        self._data_dir = Path(settings.data_dir)
        self._config_file = self._data_dir / settings.config_file
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self._config_file.exists():
            return {"providers": {}, "active_provider": "openai"}
        try:
            with open(self._config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted config file, resetting: {e}")
            return {"providers": {}, "active_provider": "openai"}
        except Exception as e:
            raise StorageException(
                operation="load_config",
                reason=str(e),
            )

    def save(self, data: dict[str, Any]) -> None:
        try:
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Config saved to file")
        except Exception as e:
            raise StorageException(
                operation="save_config",
                reason=str(e),
            )

    def get_provider_config(self, provider: str) -> Optional[dict[str, Any]]:
        data = self.load()
        return data.get("providers", {}).get(provider)

    def set_provider_config(self, provider: str, config: dict[str, Any]) -> None:
        data = self.load()
        if "providers" not in data:
            data["providers"] = {}
        data["providers"][provider] = config
        self.save(data)

    def set_active_provider(self, provider: str) -> None:
        data = self.load()
        data["active_provider"] = provider
        self.save(data)

    def get_active_provider(self) -> str:
        data = self.load()
        return data.get("active_provider", "openai")

    def get_mcp_servers(self) -> list[dict[str, Any]]:
        """Get all MCP server configurations."""
        data = self.load()
        return data.get("mcp_servers", DEFAULT_MCP_SERVERS)

    def set_mcp_servers(self, servers: list[dict[str, Any]]) -> None:
        """Save MCP server configurations."""
        data = self.load()
        data["mcp_servers"] = servers
        self.save(data)

    def get_mcp_server(self, name: str) -> Optional[dict[str, Any]]:
        """Get a specific MCP server configuration."""
        servers = self.get_mcp_servers()
        for server in servers:
            if server.get("name") == name:
                return server
        return None

    def set_mcp_server(self, name: str, config: dict[str, Any]) -> None:
        """Add or update an MCP server configuration."""
        servers = self.get_mcp_servers()
        updated = False

        for i, server in enumerate(servers):
            if server.get("name") == name:
                servers[i] = config
                updated = True
                break

        if not updated:
            servers.append(config)

        self.set_mcp_servers(servers)

    def delete_mcp_server(self, name: str) -> bool:
        """Delete an MCP server configuration.

        Returns:
            True if server was deleted
        """
        servers = self.get_mcp_servers()
        original_count = len(servers)
        servers = [s for s in servers if s.get("name") != name]

        if len(servers) < original_count:
            self.set_mcp_servers(servers)
            return True

        return False


config_storage = ConfigStorage()
