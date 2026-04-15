"""Streaming support for LLM responses."""

import asyncio
from typing import Any, AsyncIterator

from llmlib.llm import OpenAI


class StreamResponder:
    """Handles LLM streaming, yielding chunks to caller."""

    def __init__(self, client_async: OpenAI, messages: list[dict[str, Any]]):
        self._client = client_async
        self._messages = messages

    async def stream(self, prompt: str) -> AsyncIterator[tuple[str, str]]:
        """Yield (content, reasoning_content) chunks from the LLM."""
        if not self._client:
            raise ValueError("LLM client not initialized")

        messages = [
            {"role": m["role"], "content": m["content"]} for m in self._messages
        ]
        messages.append({"role": "user", "content": prompt})

        async for chunk in await self._client.chat_async.completions.create(
            messages=messages, stream=True
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")
            yield (content, reasoning)
