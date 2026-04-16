"""Runtime-level tests for MCP tool-calling orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from llmlib.runtime import ChatMessage, ChatOrchestrator, ChatRuntime, LLMSettings


def test_mcp_registry_add_uses_name_and_url(monkeypatch) -> None:
    runtime = ChatRuntime()
    seen: dict[str, str] = {}

    class FakeHTTPMCPClient:
        def __init__(self, name: str, url: str):
            seen["name"] = name
            seen["url"] = url

    monkeypatch.setattr("llmlib.runtime.mcp_registry.HTTPMCPClient", FakeHTTPMCPClient)

    result = runtime.add_mcp_server("websearch", "https://mcp.exa.ai/mcp")

    assert "websearch" in result
    assert seen == {"name": "websearch", "url": "https://mcp.exa.ai/mcp"}
    assert "websearch" in runtime.list_mcp_servers()


def test_runtime_configure_mcp_clients_uses_name_and_url(monkeypatch) -> None:
    runtime = ChatRuntime()
    seen: list[tuple[str, str]] = []

    class FakeHTTPMCPClient:
        def __init__(self, name: str, url: str):
            seen.append((name, url))

    monkeypatch.setattr("llmlib.runtime.mcp_registry.HTTPMCPClient", FakeHTTPMCPClient)

    settings = LLMSettings(api_url="http://127.0.0.1:1234/v1", api_key="", model="qwen")
    runtime.configure(settings, {"exa": "https://mcp.exa.ai/mcp"})

    assert seen == [("exa", "https://mcp.exa.ai/mcp")]
    assert "exa" in runtime.list_mcp_servers()


@dataclass
class _FakeToolSchema:
    name: str
    description: str
    input_schema: dict


class _FakeToolResult:
    def __init__(self, content: list[str], is_error: bool = False) -> None:
        self.content = content
        self.is_error = is_error


class _FakeMCPClient:
    def __init__(self) -> None:
        self.is_connected = False
        self.calls: list[tuple[str, dict]] = []

    async def connect(self) -> bool:
        self.is_connected = True
        return True

    async def list_tools(self):
        return [
            _FakeToolSchema(
                name="get_time",
                description="Get current time.",
                input_schema={
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                },
            )
        ]

    async def call_tool(self, tool_name: str, arguments: dict):
        self.calls.append((tool_name, arguments))
        return _FakeToolResult(["2026-01-01T00:00:00+00:00"])


class _FakeCompletions:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs.get("stream"):

            async def _stream():
                yield {"choices": [{"delta": {"content": "Respuesta final."}}]}

            return _stream()

        return type(
            "_Completion",
            (),
            {
                "choices": [
                    type(
                        "_Choice",
                        (),
                        {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "exa__get_time",
                                        "arguments": '{"timezone": "UTC"}',
                                    },
                                }
                            ],
                            "message": type("_Msg", (), {"content": ""})(),
                        },
                    )()
                ]
            },
        )()


class _FakeCompletionsNoToolCalls:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs.get("stream"):

            async def _stream():
                yield {"choices": [{"delta": {"content": "Hecho con web."}}]}

            return _stream()

        return type(
            "_Completion",
            (),
            {
                "choices": [
                    type(
                        "_Choice",
                        (),
                        {
                            "tool_calls": [],
                            "message": type(
                                "_Msg",
                                (),
                                {
                                    "content": "No puedo usar herramientas directamente.",
                                },
                            )(),
                        },
                    )()
                ]
            },
        )()


class _FakeClientAsync:
    def __init__(self) -> None:
        self.chat_async = type("_ChatAsync", (), {"completions": _FakeCompletions()})()


class _FakeClientAsyncNoToolCalls:
    def __init__(self) -> None:
        self.chat_async = type(
            "_ChatAsync", (), {"completions": _FakeCompletionsNoToolCalls()}
        )()


def test_stream_responder_executes_mcp_tools_before_final_response() -> None:
    async def _run() -> None:
        fake_client = _FakeClientAsync()
        fake_mcp = _FakeMCPClient()
        responder = ChatOrchestrator(
            fake_client,
            {"exa": fake_mcp},
        )

        content_chunks = []
        async for event in responder.stream(
            [ChatMessage(role="user", content="dime hora")]
        ):
            if event.kind == "content":
                content_chunks.append(event.text)

        full_content = "".join(content_chunks)
        assert "Respuesta final" in full_content
        assert fake_mcp.calls == [("get_time", {"timezone": "UTC"})]

        first_request = fake_client.chat_async.completions.requests[0]
        assert "tools" in first_request
        user_messages = [m for m in first_request["messages"] if m["role"] == "user"]
        assert len(user_messages) == 1

    asyncio.run(_run())


def test_stream_responder_does_not_force_tool_when_model_emits_no_tool_calls() -> None:
    async def _run() -> None:
        fake_client = _FakeClientAsyncNoToolCalls()
        fake_mcp = _FakeMCPClient()
        responder = ChatOrchestrator(
            fake_client,
            {"exa": fake_mcp},
        )

        chunks = []
        async for event in responder.stream(
            [ChatMessage(role="user", content="busca noticias IA de hoy")]
        ):
            if event.kind == "content":
                chunks.append(event.text)

        assert "Hecho con web." in "".join(chunks)
        assert fake_mcp.calls == []
        assert len(fake_client.chat_async.completions.requests) >= 2

    asyncio.run(_run())
