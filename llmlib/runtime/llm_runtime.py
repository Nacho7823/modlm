"""Runtime holder for LLM client instances."""

from __future__ import annotations

from llmlib.llm import OpenAI

from .types import LLMSettings


class LLMRuntime:
    """Creates and stores sync/async OpenAI-compatible clients."""

    def __init__(self) -> None:
        self._client: OpenAI | None = None
        self._client_async: OpenAI | None = None
        self._settings: LLMSettings | None = None

    def configure(self, settings: LLMSettings) -> None:
        """Build clients from settings."""
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.api_url,
            api_key=settings.api_key,
            model=settings.model,
        )
        self._client_async = OpenAI(
            base_url=settings.api_url,
            api_key=settings.api_key,
            model=settings.model,
        )

    @property
    def settings(self) -> LLMSettings | None:
        return self._settings

    @property
    def client(self) -> OpenAI | None:
        return self._client

    @property
    def client_async(self) -> OpenAI | None:
        return self._client_async
