from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from llmlib.models import Tool, ToolCall, ChatCompletion, Message, StreamEvent
from llmlib.llm import OpenAI
from llmlib.mcp import ToolExecutor

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Handles LLM streaming with optional MCP tool execution."""

    def __init__(
        self,
        client_async: OpenAI,
        mcp_clients: dict[str, Any] | None = None,
    ):
        self._client = client_async
        self._tool_executor = ToolExecutor(mcp_clients or {})

    async def stream(
        self,
        messages: list[Message],
        streaming_enabled: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Yield typed stream events from the LLM runtime."""
        self._ensure_client()

        request_messages = self._prepare_request_messages(messages)
        try:
            tools, tool_targets, errors = await self._tool_executor.discover_tools()
            for error in errors:
                yield StreamEvent.content(f"\n[bold yellow]⚠ MCP Warning:[/bold yellow] {error}\n")

            tool_loop_content = ""
            async for event in self._resolve_tool_calls(
                request_messages,
                tools,
                tool_targets,
            ):
                if event.kind == "content":
                    tool_loop_content = event.text
                yield event

            async for chunk in self._emit_final_response(
                request_messages,
                tools,
                streaming_enabled,
                fallback_content=tool_loop_content,
            ):
                yield chunk
            yield StreamEvent.done()
        finally:
            # We no longer disconnect clients here because they are persistent!
            pass

    def _ensure_client(self) -> None:
        if not self._client:
            raise ValueError("LLM client not initialized")

    def _prepare_request_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for message in messages:
            if message.is_empty_assistant():
                continue
            prepared.append(message.to_dict())
        return prepared

    async def _resolve_tool_calls(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[Tool],
        tool_targets: dict[str, tuple[Any, str]],
    ) -> AsyncIterator[StreamEvent]:
        if not tools:
            return

        executed_calls: set[str] = set()
        while True:
            raw_step = await self._client.chat_async.completions.create(
                messages=request_messages,
                tools=tools,
                tool_choice="auto",
            )
            step = ChatCompletion.from_raw(raw_step)
            if not step.choices:
                return

            choice = step.choices[0]
            tool_calls = choice.tool_calls
            content = choice.message.content
            thinking = self._append_assistant_message(request_messages, choice)
            
            if thinking:
                yield StreamEvent.thinking(thinking)
            if content:
                yield StreamEvent.content(content)

            if not tool_calls:
                return

            call_keys = self._build_call_keys(tool_calls)

            tool_results = await self._tool_executor.execute_batch(tool_calls, tool_targets)
            for res in tool_results:
                tool_msg = f"\n[bold green]🛠 Executed Tool [cyan]{res['tool_call_id']}[/cyan]:[/bold green] {res['content']}\n"
                yield StreamEvent.content(tool_msg)
                
            request_messages.extend(tool_results)
            executed_calls.update(call_keys)

    def _append_assistant_message(
        self, request_messages: list[dict[str, Any]], choice: Any
    ) -> str:
        msg = choice.message
        content = msg.content
        thinking = msg.thinking
        
        if not content and not thinking and not choice.tool_calls:
            return ""

        assistant_message = msg.to_dict()

        if choice.tool_calls:
            assistant_message["tool_calls"] = ToolCall.wrap_raw_list(choice.tool_calls)

        request_messages.append(assistant_message)
        return thinking or ""

    def _build_call_keys(self, tool_calls: list[dict[str, Any]]) -> set[str]:
        return {
            f"{tool_call.name}:{json.dumps(tool_call.arguments, sort_keys=True)}"
            for tool_call in (ToolCall.from_dict(data) for data in tool_calls)
        }

    async def _emit_final_response(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[Tool],
        streaming_enabled: bool,
        fallback_content: str = "",
    ) -> AsyncIterator[StreamEvent]:
        final_tools = tools or None

        if streaming_enabled:
            emitted = False
            async for raw_chunk in await self._client.chat_async.completions.create(
                messages=request_messages,
                tools=final_tools,
                stream=True,
            ):
                chunk = ChatCompletion.from_raw(raw_chunk)
                if not chunk.choices:
                    continue
                    
                msg = chunk.choices[0].message
                content = msg.content
                thinking = msg.thinking
                
                if content or thinking:
                    emitted = True
                if content:
                    yield StreamEvent.content(content)
                if thinking:
                    yield StreamEvent.thinking(thinking)

            if emitted:
                return

        content, thinking = await self._final_non_stream_response(
            request_messages,
            final_tools,
        )
        if not content and not thinking and fallback_content:
            yield StreamEvent.content(fallback_content)
            return
        if not content and not thinking:
            yield StreamEvent.content(
                "No response returned by model (empty completion)."
            )
            return
        if content:
            yield StreamEvent.content(content)
        if thinking:
            yield StreamEvent.thinking(thinking)

    async def _final_non_stream_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None,
    ) -> tuple[str, str]:
        raw_completion = await self._client.chat_async.completions.create(
            messages=messages,
            tools=tools,
        )
        completion = ChatCompletion.from_raw(raw_completion)
        if completion.choices:
            msg = completion.choices[0].message
            if msg.content or msg.thinking:
                return msg.content, msg.thinking

        if tools:
            raw_retry = await self._client.chat_async.completions.create(
                messages=messages,
            )
            retry = ChatCompletion.from_raw(raw_retry)
            if retry.choices:
                msg = retry.choices[0].message
                if msg.content or msg.thinking:
                    return msg.content, msg.thinking

        return "", ""
