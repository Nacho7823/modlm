"""Configuration management for llmapp."""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _get_default_config() -> dict[str, Any]:
    """Load default config from environment variables."""
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    return {
        "api_url": os.getenv("LLM_API_URL", "http://127.0.0.1:1234/v1"),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "model": os.getenv("LLM_MODEL", "qwen3.5-4b"),
        "streaming": os.getenv("LLM_STREAMING", "true").lower() == "true",
        "mcp_servers": {},
    }


class ConfigManager:
    """Manages application configuration from file and environment."""

    DEFAULT_CONFIG = _get_default_config()

    def __init__(self, config_dir: Path | None = None) -> None:
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = config_dir
        self.config_file = config_dir / "config.json"
        self._config: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        env_path = self.config_dir.parent / ".env"
        load_dotenv(env_path, verbose=True)

        self._config = dict(self.DEFAULT_CONFIG)

        self._load_from_env()
        self._load_from_file()

        return self._config

    def _load_from_env(self) -> None:
        self._config["api_url"] = self._load_env("LLM_API_URL", self._config["api_url"])
        self._config["api_key"] = self._load_env("LLM_API_KEY", "")
        self._config["model"] = self._load_env("LLM_MODEL", self._config["model"])
        self._config["streaming"] = self._load_env_bool(
            "LLM_STREAMING", self._config["streaming"]
        )

    def _load_from_file(self) -> None:
        if not self.config_file.exists():
            return

        with open(self.config_file, "r", encoding="utf-8") as file_handle:
            file_config = json.load(file_handle)

        self._merge_non_api_keys(file_config)
        self._override_api_keys(file_config)

    def _merge_non_api_keys(self, file_config: dict[str, Any]) -> None:
        for key, value in file_config.items():
            if key == "mcp_servers":
                self._config["mcp_servers"] = value
            elif key not in ("api_url", "api_key", "model"):
                self._config[key] = value

    def _override_api_keys(self, file_config: dict[str, Any]) -> None:
        if "api_url" in file_config:
            self._config["api_url"] = file_config["api_url"]
        if "api_key" in file_config:
            self._config["api_key"] = file_config["api_key"]
        if "model" in file_config:
            self._config["model"] = file_config["model"]
        if "streaming" in file_config:
            self._config["streaming"] = bool(file_config["streaming"])

    def _load_env(self, key: str, default: str) -> str:
        return os.environ.get(key, default)

    def _load_env_bool(self, key: str, default: bool) -> bool:
        raw = os.environ.get(key)
        if raw is None:
            return default
        return raw.lower() in ("1", "true", "yes", "on")

    def save(self) -> None:
        with open(self.config_file, "w", encoding="utf-8") as file_handle:
            json.dump(self._config, file_handle, indent=2)

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
