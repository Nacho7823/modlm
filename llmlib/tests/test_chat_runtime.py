"""Unit tests for llmlib facade."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from llmlib.models import Message, LLMSettings, StreamEvent, MCPServerConfig
from llmlib import ChatRuntime, ChatOrchestrator
from llmlib.mcp.registry import MCPRegistry


@pytest.mark.asyncio
async def test_chat_runtime_configure_and_list_servers() -> None:
    runtime = ChatRuntime()
    cfg = MCPServerConfig(name="exa", server_type="remote", url="https://mcp.exa.ai/mcp")
    
    with patch("llmlib.mcp.registry.HTTPMCPClient") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.test_connection = AsyncMock(return_value={"status": "connected", "tools": []})
        
        warnings = await runtime.configure(
            LLMSettings(api_url="http://127.0.0.1:1234/v1", api_key="", model="qwen"),
            {"exa": cfg},
        )

        assert warnings == []
        servers = runtime.list_mcp_servers()
        assert "exa" in servers
        assert servers["exa"].url == "https://mcp.exa.ai/mcp"


@pytest.mark.asyncio
async def test_chat_runtime_add_and_remove_server() -> None:
    runtime = ChatRuntime()
    
    with patch("llmlib.mcp.registry.HTTPMCPClient") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.test_connection = AsyncMock(return_value={"status": "connected", "tools": []})
        
        await runtime.configure(
            LLMSettings(api_url="http://127.0.0.1:1234/v1", api_key="", model="qwen"),
            {},
        )

        cfg = MCPServerConfig(name="web", server_type="remote", url="https://mcp.exa.ai/mcp")
        result = await runtime.add_mcp_server(cfg)
        assert "web" in result
        assert "web" in runtime.list_mcp_servers()

        assert await runtime.remove_mcp_server("web") is True
        assert "web" not in runtime.list_mcp_servers()


def test_chat_runtime_stream_requires_configured_llm() -> None:
    runtime = ChatRuntime()

    async def run() -> None:
        iterator = runtime.stream([{"role": "user", "content": "hola"}])
        await iterator.__anext__()

    try:
        asyncio.run(run())
    except ValueError as error:
        assert "not configured" in str(error)
    else:
        raise AssertionError("Expected ValueError for unconfigured runtime")


    # Note: _append_assistant_message was removed in refactor.
    # We now test through normalized history directly or through tool resolution steps.
    pass



from llmlib.mcp.executor import ToolExecutor

def test_chat_orchestrator_truncates_tool_result() -> None:
    executor = ToolExecutor(mcp_clients={})
    text = "x" * 3000

    truncated = executor.truncate(text)

    assert len(truncated) < len(text)
    assert truncated.endswith("...[tool result truncated]")



@pytest.mark.asyncio
async def test_mcp_registry_disconnects_each_client_once() -> None:
    class _DummyClient:
        def __init__(self) -> None:
            self.calls = 0

        async def disconnect(self) -> None:
            self.calls += 1

    client = _DummyClient()
    reg = MCPRegistry()
    reg._clients = {"a": client, "b": client}
    
    await reg.shutdown_all()
    assert client.calls == 1


def test_chat_orchestrator_stream_returns_typed_events() -> None:
    class _FakeCompletions:
        async def create(self, **kwargs):
            if kwargs.get("stream"):

                async def _stream():
                    yield {"choices": [{"delta": {"content": "hola"}}]}

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
                                "message": type("_Msg", (), {"content": "hola", "thinking": "", "to_dict": lambda: {"role": "assistant", "content": "hola"}})(),
                            },
                        )()
                    ]
                },
            )()


    class _FakeClient:
        def __init__(self) -> None:
            self.chat_async = type(
                "_ChatAsync", (), {"completions": _FakeCompletions()}
            )()

    async def _run() -> list[StreamEvent]:
        orchestrator = ChatOrchestrator(client_async=_FakeClient(), mcp_clients={})
        events: list[StreamEvent] = []
        async for event in orchestrator.stream(
            [Message(role="user", content="saluda")],
            streaming_enabled=True,
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert events
    assert any(event.kind == "content" and event.text == "hola" for event in events)
    assert events[-1].kind == "done"


def test_chat_orchestrator_emits_empty_completion_fallback() -> None:
    class _FakeCompletions:
        async def create(self, **kwargs):
            if kwargs.get("stream"):

                async def _stream():
                    if False:
                        yield {}

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
                                "message": type("_Msg", (), {"content": "", "thinking": "", "to_dict": lambda: {"role": "assistant", "content": ""}})(),
                            },
                        )()
                    ]
                },
            )()


    class _FakeClient:
        def __init__(self) -> None:
            self.chat_async = type(
                "_ChatAsync", (), {"completions": _FakeCompletions()}
            )()

    async def _run() -> list[StreamEvent]:
        orchestrator = ChatOrchestrator(client_async=_FakeClient(), mcp_clients={})
        events: list[StreamEvent] = []
        async for event in orchestrator.stream(
            [Message(role="user", content="hola")],
            streaming_enabled=True,
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert any(
        event.kind == "content"
        and event.text == "No response returned by model (empty completion)."
        for event in events
    )


@pytest.mark.asyncio
async def test_chat_runtime_emits_error_event_when_orchestrator_fails(monkeypatch) -> None:
    class _BrokenOrchestrator:
        def __init__(self, client_async, mcp_clients):
            self._client_async = client_async
            self._mcp_clients = mcp_clients

        async def stream(self, messages, streaming_enabled=True):
            raise RuntimeError("boom")
            yield StreamEvent.done()

    monkeypatch.setattr(
        "llmlib.chat_runtime.ChatOrchestrator", _BrokenOrchestrator
    )

    runtime = ChatRuntime()
    # Mocking configure to avoid real connection attempt
    with patch("llmlib.mcp.registry.MCPRegistry._validate", AsyncMock(return_value=[])):
        await runtime.configure(
            LLMSettings(api_url="http://127.0.0.1:1234/v1", api_key="", model="qwen"),
            {},
        )

    events: list[StreamEvent] = []
    async for event in runtime.stream([{"role": "user", "content": "hola"}]):
        events.append(event)

    assert events[0].kind == "error"
    assert "boom" in events[0].text
    assert events[-1].kind == "done"
