"""Test fixtures for LLM module - Real API tests."""

from __future__ import annotations

import os
import pytest
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from llmlib.llm import OpenAI

DEFAULT_API_URL = "http://localhost:11434/v1"

API_URL = os.getenv("LLM_API_URL", DEFAULT_API_URL)
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")


def is_api_available() -> bool:
    """Check if the API is available (returns 200)."""
    try:
        with OpenAI(base_url=API_URL, api_key=API_KEY) as client:
            response = client._client.post(
                f"{API_URL.rstrip('/')}/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
            )
        return response.status_code == 200
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
    if not is_api_available():
        pytest.skip("API not available or returns empty content")
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


@pytest.fixture
def default_api_url():
    """Default API URL for tests that don't need real API."""
    return DEFAULT_API_URL


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "api: mark test as requiring API")
