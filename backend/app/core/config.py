from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "modllm"
    app_version: str = "1.0.0"
    debug: bool = False

    # API
    api_prefix: str = "/api"

    # CORS
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://192.168.100.48:5173",
        "http://localhost:3000",
    ]

    # Storage
    data_dir: str = "data"
    chat_history_file: str = "chat_history.json"
    config_file: str = "config.json"

    # LLM defaults
    default_temperature: float = 0.7
    default_max_tokens: int = 2048
    default_model: str = "gpt-3.5-turbo"

    # OpenAI compatible defaults
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
