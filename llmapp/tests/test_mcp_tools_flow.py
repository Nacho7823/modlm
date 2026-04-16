"""Tests for MCP tool wiring in llmapp."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from llmapp.services.config import ConfigService
from llmapp.services.mcp import MCPService
from llmapp.streaming import StreamResponder


class _ConfigStub:
    def __init__(self) -> None:
        self._servers: dict[str, str] = {}

    def add_mcp_server(self, name: str, url: str) -> None:
        self._servers[name] = url

    def get_mcp_servers(self) -> dict[str, str]:
        return dict(self._servers)


class _AppStub:
    def __init__(self) -> None:
        self.config_manager = _ConfigStub()
        self.mcp_clients: dict[str, object] = {}
        self.system_messages: list[str] = []

    def _add_system_message(self, msg: str) -> None:
        self.system_messages.append(msg)


def test_mcp_service_add_uses_name_and_url(monkeypatch):
    app = _AppStub()
    seen: dict[str, str] = {}

    class FakeHTTPMCPClient:
        def __init__(self, name: str, url: str):
            seen["name"] = name
            seen["url"] = url

    monkeypatch.setattr("llmlib.mcp.HTTPMCPClient", FakeHTTPMCPClient)

    service = MCPService(app)
    result = service.add("websearch", "https://mcp.exa.ai/mcp")

    assert "websearch" in result
    assert seen == {"name": "websearch", "url": "https://mcp.exa.ai/mcp"}
    assert "websearch" in app.mcp_clients


def test_config_service_init_mcp_clients_uses_name_and_url(monkeypatch):
    app = _AppStub()
    app.config_manager.add_mcp_server("exa", "https://mcp.exa.ai/mcp")
    seen: list[tuple[str, str]] = []

    class FakeHTTPMCPClient:
        def __init__(self, name: str, url: str):
            seen.append((name, url))

    monkeypatch.setattr("llmlib.mcp.HTTPMCPClient", FakeHTTPMCPClient)

    service = ConfigService(app)
    service._init_mcp_clients()

    assert seen == [("exa", "https://mcp.exa.ai/mcp")]
    assert "exa" in app.mcp_clients


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


def test_stream_responder_executes_mcp_tools_before_final_response():
    async def _run() -> None:
        fake_client = _FakeClientAsync()
        fake_mcp = _FakeMCPClient()
        responder = StreamResponder(
            fake_client,
            {"exa": fake_mcp},
        )

        content_chunks = []
        async for content, _thinking in responder.stream(
            [{"role": "user", "content": "dime hora"}]
        ):
            content_chunks.append(content)

        full_content = "".join(content_chunks)
        assert "Respuesta final" in full_content
        assert fake_mcp.calls == [("get_time", {"timezone": "UTC"})]

        first_request = fake_client.chat_async.completions.requests[0]
        assert "tools" in first_request
        user_messages = [m for m in first_request["messages"] if m["role"] == "user"]
        assert len(user_messages) == 1

    asyncio.run(_run())


def test_stream_responder_does_not_force_tool_when_model_emits_no_tool_calls():
    async def _run() -> None:
        fake_client = _FakeClientAsyncNoToolCalls()
        fake_mcp = _FakeMCPClient()
        responder = StreamResponder(
            fake_client,
            {"exa": fake_mcp},
        )

        chunks = []
        async for content, _thinking in responder.stream(
            [{"role": "user", "content": "busca noticias IA de hoy"}]
        ):
            chunks.append(content)

        assert "Hecho con web" in "".join(chunks)
        assert fake_mcp.calls == []

    asyncio.run(_run())
