"""Configuration management for llmapp."""

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class ConfigManager:
    """Manages application configuration from file and environment."""

    DEFAULT_CONFIG = {
        "api_url": "http://127.0.0.1:1234/v1",
        "api_key": "",
        "model": "granite-4.0-h-micro",
        "mcp_servers": {},
    }

    def __init__(self, config_dir: Path | None = None) -> None:
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = config_dir
        self.config_file = config_dir / "config.json"
        self._config: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        env_path = self.config_dir.parent / ".env"
        load_dotenv(env_path)

        env_url = load_dotenv(env_path, verbose=True)
        self._config = dict(self.DEFAULT_CONFIG)

        self._config["api_url"] = self._load_env("LLM_API_URL", self._config["api_url"])
        self._config["api_key"] = self._load_env("LLM_API_KEY", "")
        self._config["model"] = self._load_env("LLM_MODEL", self._config["model"])

        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                file_config = json.load(f)
                for key, value in file_config.items():
                    if key == "mcp_servers":
                        self._config["mcp_servers"] = value
                    elif key not in ("api_url", "api_key", "model"):
                        self._config[key] = value
                if "api_url" in file_config:
                    self._config["api_url"] = file_config["api_url"]
                if "api_key" in file_config:
                    self._config["api_key"] = file_config["api_key"]
                if "model" in file_config:
                    self._config["model"] = file_config["model"]

        return self._config

    def _load_env(self, key: str, default: str) -> str:
        import os

        return os.environ.get(key, default)

    def save(self) -> None:
        with open(self.config_file, "w") as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._config[key] = value

    def get_api_config(self) -> tuple[str, str, str]:
        return (
            self._config.get("api_url", ""),
            self._config.get("api_key", ""),
            self._config.get("model", ""),
        )

    def get_mcp_servers(self) -> dict[str, str]:
        return self._config.get("mcp_servers", {})

    def add_mcp_server(self, name: str, url: str) -> None:
        mcp_servers = self._config.get("mcp_servers", {})
        mcp_servers[name] = url
        self._config["mcp_servers"] = mcp_servers
        self.save()

    def remove_mcp_server(self, name: str) -> bool:
        mcp_servers = self._config.get("mcp_servers", {})
        if name in mcp_servers:
            del mcp_servers[name]
            self._config["mcp_servers"] = mcp_servers
            self.save()
            return True
        return False
