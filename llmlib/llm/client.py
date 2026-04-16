"""OpenAI-compatible LLM client."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Sequence

import httpx

from .models import ChatCompletion, Choice, Message
from .tools import Tool

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
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        final_model, payload = _prepare_completion_request(
            default_model=self._model,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )

        response = self._client.post(
            CHAT_COMPLETIONS_ENDPOINT,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return _parse_completion(data, final_model)


class StreamCompletions:
    """Streaming chat completions endpoint."""

    def __init__(self, client: httpx.Client, model: str | None = None):
        self._client = client
        self._model = model

    def create(
        self,
        model: str | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        _, payload = _prepare_completion_request(
            default_model=self._model,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )
        payload["stream"] = True

        with self._client.stream(
            "POST",
            CHAT_COMPLETIONS_ENDPOINT,
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                done, chunk = _parse_sse_data_line(line)
                if done:
                    break
                if chunk is not None:
                    yield chunk


class AsyncCompletions:
    """Async chat completions endpoint."""

    def __init__(self, client: httpx.AsyncClient, model: str | None = None):
        self._client = client
        self._model = model

    async def create(
        self,
        model: str | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[dict[str, Any]]:
        final_model, payload = _prepare_completion_request(
            default_model=self._model,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )

        if stream:
            return self._stream_response(payload)
        else:
            return await self._create_sync(payload, final_model)

    async def _create_sync(self, payload: dict[str, Any], model: str) -> ChatCompletion:
        response = await self._client.post(
            CHAT_COMPLETIONS_ENDPOINT,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return _parse_completion(data, model)

    async def _stream_response(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        payload["stream"] = True
        try:
            async with self._client.stream(
                "POST",
                CHAT_COMPLETIONS_ENDPOINT,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    done, chunk = _parse_sse_data_line(line)
                    if done:
                        break
                    if chunk is not None:
                        yield chunk
        except asyncio.CancelledError:
            raise


def _prepare_completion_request(
    default_model: str | None,
    model: str | None,
    messages: Sequence[dict[str, Any]] | None,
    temperature: float | None,
    max_tokens: int | None,
    tools: list[Tool] | None,
    **kwargs: Any,
) -> tuple[str, dict[str, Any]]:
    final_model = model or default_model
    if not final_model:
        raise ValueError("model is required")

    payload = _build_payload(
        model=final_model,
        messages=_normalize_messages(messages),
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        **kwargs,
    )
    return final_model, payload


def _normalize_messages(
    messages: Sequence[dict[str, Any]] | None,
) -> Sequence[dict[str, Any]]:
    if messages is None:
        return []
    return messages


def _parse_sse_data_line(raw_line: str) -> tuple[bool, dict[str, Any] | None]:
    line = raw_line.strip()
    if not line.startswith(SSE_DATA_PREFIX):
        return False, None

    data = line[len(SSE_DATA_PREFIX) :]
    if data == SSE_DONE_MARKER:
        return True, None

    try:
        return False, json.loads(data)
    except json.JSONDecodeError:
        return False, None


def _build_payload(
    model: str,
    messages: Sequence[dict[str, Any]],
    temperature: float | None,
    max_tokens: int | None,
    tools: list[Tool] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
    }

    if temperature is not None:
        payload["temperature"] = temperature

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if tools:
        payload["tools"] = [tool.to_openai_schema() for tool in tools]

    payload.update(kwargs)
    return payload


def _parse_completion(data: dict[str, Any], model: str) -> ChatCompletion:
    choices = _parse_choices(data.get("choices", []))
    return ChatCompletion(
        choices=choices,
        model=model,
        finish_reason=data.get("choices", [{}])[0].get("finish_reason")
        if choices
        else None,
    )


def _parse_choices(choices_data: list[dict[str, Any]]) -> list[Choice]:
    choices = []
    for i, choice_data in enumerate(choices_data):
        message_data = choice_data.get("message", {})
        message = Message(
            content=message_data.get("content") or "",
            role=message_data.get("role", "assistant"),
            reasoning_content=message_data.get("reasoning_content"),
        )

        tool_calls = _extract_tool_calls(choice_data, message_data)
        finish_reason = choice_data.get("finish_reason")

        choices.append(
            Choice(
                message=message,
                index=choice_data.get("index", i),
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )
        )

    return choices


def _extract_tool_calls(
    choice_data: dict[str, Any], message_data: dict[str, Any]
) -> list[dict[str, Any]]:
    tool_calls = choice_data.get("tool_calls")
    if tool_calls is not None:
        return tool_calls
    return message_data.get("tool_calls", [])
