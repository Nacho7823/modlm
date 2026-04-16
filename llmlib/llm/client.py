"""OpenAI-compatible LLM client."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Sequence

import httpx

from llmlib.models import ChatCompletion, Choice, Message, Tool

if TYPE_CHECKING:
    import asyncio


CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"
SSE_DATA_PREFIX = "data: "
SSE_DONE_MARKER = "[DONE]"


class OpenAI:
    """OpenAI-compatible client without API key requirement."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )
        self._async_client: httpx.AsyncClient | None = None

    def close(self) -> None:
        self._client.close()
        if self._async_client:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_client.aclose())
            except RuntimeError:
                pass

    def __enter__(self) -> "OpenAI":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._client.close()
        self._client = None

    @property
    def chat(self) -> "Chat":
        return Chat(self._client, self.base_url, self.model)

    @property
    def chat_async(self) -> "AsyncChat":
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return AsyncChat(self._async_client, self.base_url, self.model)

    def fetch_models(self) -> list[str]:
        """Fetch available models from the API."""
        base = self.base_url.rstrip("/")

        if "ollama" in base.lower():
            return self._fetch_ollama_models(base)
        elif "lmstudio" in base.lower() or "lm studio" in base.lower():
            return self._fetch_lmstudio_models(base)
        else:
            return self._fetch_openai_models(base)

    def _fetch_ollama_models(self, base: str) -> list[str]:
        """Fetch models from Ollama API."""
        try:
            resp = self._client.get(f"{base}/api/tags")
            if resp.is_success:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def _fetch_lmstudio_models(self, base: str) -> list[str]:
        """Fetch models from LM Studio API."""
        try:
            resp = self._client.get(f"{base}/v1/models")
            if resp.is_success:
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return []

    def _fetch_openai_models(self, base: str) -> list[str]:
        """Fetch models from OpenAI-compatible API."""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            resp = self._client.get(f"{base}/models", headers=headers)
            if resp.is_success:
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return []


class Chat:
    """Chat endpoint."""

    def __init__(self, client: httpx.Client, base_url: str, model: str | None = None):
        self._client = client
        self._model = model

    @property
    def completions(self) -> "Completions":
        return Completions(self._client, self._model)

    @property
    def stream_completions(self) -> "StreamCompletions":
        return StreamCompletions(self._client, self._model)


class AsyncChat:
    """Async chat endpoint."""

    def __init__(
        self, client: httpx.AsyncClient, base_url: str, model: str | None = None
    ):
        self._client = client
        self._model = model

    @property
    def completions(self) -> "AsyncCompletions":
        return AsyncCompletions(self._client, self._model)



class Completions:
    """Chat completions endpoint."""

    def __init__(self, client: httpx.Client, model: str | None = None):
        self._client = client
        self._model = model

    def create(
        self,
        model: str | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        payload = _build_payload(model or self._model, messages, tools, **kwargs)
        response = self._client.post(CHAT_COMPLETIONS_ENDPOINT, json=payload)
        response.raise_for_status()
        return ChatCompletion.from_raw(response.json())


class StreamCompletions:
    """Streaming chat completions endpoint."""

    def __init__(self, client: httpx.Client, model: str | None = None):
        self._client = client
        self._model = model

    def create(
        self,
        model: str | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        payload = _build_payload(model or self._model, messages, tools, stream=True, **kwargs)
        with self._client.stream("POST", CHAT_COMPLETIONS_ENDPOINT, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                done, chunk = _parse_sse_data_line(line)
                if done: break
                if chunk: yield chunk


class AsyncCompletions:
    """Async chat completions endpoint."""

    def __init__(self, client: httpx.AsyncClient, model: str | None = None):
        self._client = client
        self._model = model

    async def create(
        self,
        model: str | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        tools: list[Tool] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[dict[str, Any]]:
        payload = _build_payload(model or self._model, messages, tools, stream=stream, **kwargs)
        if stream:
            return self._stream_response(payload)
        
        response = await self._client.post(CHAT_COMPLETIONS_ENDPOINT, json=payload)
        response.raise_for_status()
        return ChatCompletion.from_raw(response.json())

    async def _stream_response(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        async with self._client.stream("POST", CHAT_COMPLETIONS_ENDPOINT, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                done, chunk = _parse_sse_data_line(line)
                if done: break
                if chunk:
                    yield chunk
                    await asyncio.sleep(0)  # Yield to event loop for smoother TUI updates



def _build_payload(
    model: str | None,
    messages: Sequence[dict[str, Any]] | None,
    tools: list[Tool] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if not model: raise ValueError("model is required")
    payload = {"model": model, "messages": list(messages or [])}
    if tools:
        payload["tools"] = [t.to_openai_schema() for t in tools]
    payload.update(kwargs)
    return payload


def _parse_sse_data_line(raw_line: str) -> tuple[bool, dict[str, Any] | None]:
    prefix = "data:"
    line = raw_line.strip()
    if not line.startswith(prefix):
        return False, None
    
    data = line[len(prefix) :].strip()
    if data == "[DONE]":
        return True, None
    if not data:
        return False, None
        
    try:
        return False, json.loads(data)
    except json.JSONDecodeError:
        return False, None




