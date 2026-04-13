"""Lightweight OpenAI-compatible client without API key requirement."""

from __future__ import annotations

import httpx
from typing import Any, Sequence


class ChatCompletion:
    """Chat completion response object."""

    def __init__(self, choices: list[dict[str, Any]], model: str | None = None):
        self.choices = choices
        self.model = model

    @property
    def id(self) -> str:
        return "chatcmpl-local"

    @property
    def usage(self) -> dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class Message:
    """Message object."""

    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role


class Choice:
    """Choice object."""

    def __init__(self, message: Message, index: int = 0):
        self.message = message
        self.index = index


class OpenAI:
    """OpenAI-compatible client without API key requirement."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(60.0),
        )

    def close(self) -> None:
        self._client.close()

    @property
    def chat(self) -> Chat:
        return Chat(self._client, self.base_url)


class Chat:
    """Chat endpoint."""

    def __init__(self, client: httpx.Client, base_url: str):
        self._client = client
        self._base_url = base_url

    @property
    def completions(self) -> Completions:
        return Completions(self._client, self._base_url)


class Completions:
    """Chat completions endpoint."""

    def __init__(self, client: httpx.Client, base_url: str):
        self._client = client
        self._base_url = base_url

    def create(
        self,
        model: str,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        response = self._client.post(
            "/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        choices = [
            Choice(
                Message(
                    content=choice.get("message", {}).get("content", ""),
                    role=choice.get("message", {}).get("role", "assistant"),
                ),
                index=choice.get("index", i),
            )
            for i, choice in enumerate(data.get("choices", []))
        ]

        return ChatCompletion(choices=choices, model=model)
