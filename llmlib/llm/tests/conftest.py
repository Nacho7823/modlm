"""Test fixtures for LLM module - Real API tests."""

from __future__ import annotations

import os
import pytest
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from llmlib.llm import OpenAI


API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")


def is_api_available() -> bool:
    """Check if the API is available."""
    try:
        client = OpenAI(base_url=API_URL, api_key=API_KEY)
        client._client.get(f"{API_URL.rstrip('/')}/api/tags")
        client.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def api_config():
    """API configuration from environment variables."""
    return {
        "url": API_URL,
        "api_key": API_KEY,
        "model": MODEL,
    }


@pytest.fixture
def real_client(api_config):
    """Real OpenAI client connected to the configured API."""
    client = OpenAI(
        base_url=api_config["url"],
        api_key=api_config["api_key"],
        model=api_config["model"],
    )
    yield client
    client.close()


@pytest.fixture
def model(api_config):
    """Model name for tests."""
    return api_config["model"]


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "api: mark test as requiring API")
