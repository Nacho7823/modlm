"""Unit tests for llmlib runtime facade."""

from __future__ import annotations

import asyncio

from llmlib.runtime import ChatRuntime, LLMSettings
from llmlib.runtime.chat_orchestrator import ChatOrchestrator


def test_chat_runtime_configure_and_list_servers() -> None:
    runtime = ChatRuntime()
    warnings = runtime.configure(
        LLMSettings(api_url="http://127.0.0.1:1234/v1", api_key="", model="qwen"),
        {"exa": "https://mcp.exa.ai/mcp"},
    )

    assert warnings == []
    assert runtime.list_mcp_servers() == {"exa": "https://mcp.exa.ai/mcp"}


def test_chat_runtime_add_and_remove_server() -> None:
    runtime = ChatRuntime()
    runtime.configure(
        LLMSettings(api_url="http://127.0.0.1:1234/v1", api_key="", model="qwen"),
        {},
    )

    result = runtime.add_mcp_server("web", "https://mcp.exa.ai/mcp")
    assert "web" in result
    assert "web" in runtime.list_mcp_servers()

    assert runtime.remove_mcp_server("web") is True
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


def test_chat_orchestrator_skips_empty_assistant_tool_step_message() -> None:
    orchestrator = ChatOrchestrator(client_async=object())
    request_messages = [{"role": "user", "content": "hola"}]

    empty_choice = type(
        "_Choice",
        (),
        {
            "message": type("_Msg", (), {"content": "", "reasoning_content": ""})(),
            "tool_calls": [],
        },
    )()

    reasoning = orchestrator._append_assistant_message(request_messages, empty_choice)

    assert reasoning == ""
    assert request_messages == [{"role": "user", "content": "hola"}]


def test_chat_orchestrator_truncates_tool_result() -> None:
    orchestrator = ChatOrchestrator(client_async=object())
    text = "x" * 3000

    truncated = orchestrator._truncate_tool_result(text)

    assert len(truncated) < len(text)
    assert truncated.endswith("...[tool result truncated]")
