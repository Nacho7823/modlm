from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from llmlib.models import Tool, ToolCall, ChatCompletion, Message, StreamEvent
from llmlib.llm import OpenAI
from llmlib.mcp import ToolExecutor

logger = logging.getLogger(__name__)



class ChatOrchestrator:
    """Handles LLM interaction loop with MCP tool execution."""

    def __init__(self, client_async: OpenAI, mcp_clients: dict[str, Any] | None = None):
        self._client = client_async
        self._tool_executor = ToolExecutor(mcp_clients or {})

    async def stream(
        self, messages: list[Message], streaming_enabled: bool = True
    ) -> AsyncIterator[StreamEvent]:
        """Yield typed stream events from the LLM runtime."""
        history = [m.to_dict() for m in messages if not m.is_empty_assistant()]
        
        try:
            tools, tool_targets, errors = await self._tool_executor.discover_tools()
            for err in errors:
                yield StreamEvent.content(f"\n[bold yellow]⚠ MCP Warning:[/bold yellow] {err}\n")

            # 1. Resolve all tool calls first
            async for event in self._resolve_tools(history, tools, tool_targets):
                yield event

            # 2. Final response (streamed if enabled)
            async for event in self._emit_final(history, tools, streaming_enabled):
                yield event
                
            yield StreamEvent.done()
        finally:
            pass

    async def _resolve_tools(
        self, history: list[dict[str, Any]], tools: list[Tool], targets: dict
    ) -> AsyncIterator[StreamEvent]:
        if not tools: return

        while True:
            full_msg, tool_calls_map = Message(role="assistant"), {}
            async for chunk_raw in await self._client.chat_async.completions.create(
                messages=history, tools=tools, tool_choice="auto", stream=True
            ):
                chunk = ChatCompletion.from_raw(chunk_raw)
                if not chunk.choices: continue
                delta = chunk.choices[0].message
                
                if delta.thinking:
                    full_msg.thinking += delta.thinking
                    yield StreamEvent.thinking(delta.thinking)
                if delta.content:
                    full_msg.content += delta.content
                    yield StreamEvent.content(delta.content)
                
                # Accumulate tool calls from stream
                for tc_raw in chunk.tool_calls:
                    idx = tc_raw.get("index", 0)
                    if idx not in tool_calls_map: tool_calls_map[idx] = tc_raw
                    else:
                        # Merge function arguments
                        curr_func = tool_calls_map[idx].get("function", {})
                        new_func = tc_raw.get("function", {})
                        curr_func["arguments"] = curr_func.get("arguments", "") + new_func.get("arguments", "")

            # Convert accumulated map back to list
            tool_calls = sorted(tool_calls_map.values(), key=lambda x: x.get("index", 0))
            if not full_msg.content and not full_msg.thinking and not tool_calls: break

            history_entry = full_msg.to_dict()
            if tool_calls: history_entry["tool_calls"] = ToolCall.wrap_raw_list(tool_calls)
            history.append(history_entry)
            
            if not tool_calls: break

            # Execute tools and add results to history
            results = await self._tool_executor.execute_batch(tool_calls, targets)
            for res in results:
                yield StreamEvent.content(f"\n[bold green]🛠 Tool [cyan]{res['tool_call_id']}[/cyan]:[/bold green] {res['content']}\n")
            history.extend(results)


    async def _emit_final(
        self, history: list[dict[str, Any]], tools: list[Tool], stream: bool
    ) -> AsyncIterator[StreamEvent]:
        if stream:
            emitted = False
            async for chunk_raw in await self._client.chat_async.completions.create(
                messages=history, tools=tools or None, stream=True
            ):
                emitted = True
                chunk = ChatCompletion.from_raw(chunk_raw)
                if not chunk.choices: continue
                msg = chunk.choices[0].message
                if msg.thinking: yield StreamEvent.thinking(msg.thinking)
                if msg.content: yield StreamEvent.content(msg.content)
            
            if emitted: return


        # Fallback or non-streaming
        resp = await self._client.chat_async.completions.create(messages=history, tools=tools or None)
        if resp.choices:
            msg = resp.choices[0].message
            if msg.content: yield StreamEvent.content(msg.content)
            if msg.thinking: yield StreamEvent.thinking(msg.thinking)
            if not msg.content and not msg.thinking:
                yield StreamEvent.content("No response returned by model (empty completion).")
        elif not any(m["role"] == "assistant" for m in history):
            yield StreamEvent.content("No response returned by model.")


