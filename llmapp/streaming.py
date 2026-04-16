"""Streaming support for LLM responses."""

import json
from typing import Any, AsyncIterator

from llmlib.llm import OpenAI, Tool, ToolCall

MAX_TOOL_STEPS = 4


class StreamResponder:
    """Handles LLM streaming, yielding chunks to caller."""

    def __init__(
        self,
        client_async: OpenAI,
        mcp_clients: dict[str, Any] | None = None,
    ):
        self._client = client_async
        self._mcp_clients = mcp_clients or {}

    async def stream(
        self,
        messages: list[dict[str, Any]],
        streaming_enabled: bool = True,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield (content, reasoning_content) chunks from the LLM."""
        self._ensure_client()

        request_messages = self._prepare_request_messages(messages)
        tools, tool_targets = await self._build_tool_list()

        async for thinking in self._resolve_tool_calls(
            request_messages,
            tools,
            tool_targets,
        ):
            yield ("", thinking)

        async for chunk in self._emit_final_response(
            request_messages,
            tools,
            streaming_enabled,
        ):
            yield chunk

    def _ensure_client(self) -> None:
        if not self._client:
            raise ValueError("LLM client not initialized")

    def _prepare_request_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            tool_calls = message.get("tool_calls")

            if role == "assistant" and not content and not tool_calls:
                continue

            prepared_message = {"role": message["role"], "content": content}
            if tool_calls:
                prepared_message["tool_calls"] = tool_calls
            if "tool_call_id" in message:
                prepared_message["tool_call_id"] = message["tool_call_id"]
            if message.get("reasoning_content"):
                prepared_message["reasoning_content"] = message["reasoning_content"]
            prepared.append(prepared_message)

        return prepared

    async def _resolve_tool_calls(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[Tool],
        tool_targets: dict[str, tuple[Any, str]],
    ) -> AsyncIterator[str]:
        if not tools:
            return

        executed_calls: set[str] = set()
        for _ in range(MAX_TOOL_STEPS):
            step = await self._client.chat_async.completions.create(
                messages=request_messages,
                tools=tools,
                tool_choice="auto",
            )
            choice = step.choices[0] if step.choices else None
            if not choice:
                return

            tool_calls = choice.tool_calls or []
            reasoning_content = self._append_assistant_message(request_messages, choice)
            if reasoning_content:
                yield reasoning_content

            if not tool_calls:
                return

            call_keys = self._build_call_keys(tool_calls)
            if call_keys and call_keys.issubset(executed_calls):
                return

            request_messages.extend(
                await self._execute_mcp_tool_calls(tool_calls, tool_targets)
            )
            executed_calls.update(call_keys)

    def _append_assistant_message(
        self, request_messages: list[dict[str, Any]], choice: Any
    ) -> str:
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": choice.message.content or "",
        }

        reasoning_content = getattr(choice.message, "reasoning_content", "") or ""
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        if choice.tool_calls:
            assistant_message["tool_calls"] = choice.tool_calls

        request_messages.append(assistant_message)
        return reasoning_content

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
    ) -> AsyncIterator[tuple[str, str]]:
        final_tools = tools or None

        if streaming_enabled:
            emitted = False
            async for chunk in await self._client.chat_async.completions.create(
                messages=request_messages,
                tools=final_tools,
                stream=True,
            ):
                content, reasoning = self._extract_delta_text(chunk)
                if content or reasoning:
                    emitted = True
                yield (content, reasoning)

            if emitted:
                return

        content, reasoning = await self._final_non_stream_response(
            request_messages,
            final_tools,
        )
        yield (content, reasoning)

    def _extract_delta_text(self, chunk: dict[str, Any]) -> tuple[str, str]:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        return delta.get("content", ""), delta.get("reasoning_content", "")

    async def _final_non_stream_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None,
    ) -> tuple[str, str]:
        completion = await self._client.chat_async.completions.create(
            messages=messages,
            tools=tools,
        )
        content, reasoning = self._extract_choice_text(completion)
        if content or reasoning:
            return content, reasoning

        if tools:
            retry = await self._client.chat_async.completions.create(
                messages=messages,
            )
            retry_content, retry_reasoning = self._extract_choice_text(retry)
            if retry_content or retry_reasoning:
                return retry_content, retry_reasoning

        return "No pude obtener respuesta del modelo.", ""

    def _extract_choice_text(self, completion: Any) -> tuple[str, str]:
        if not completion or not getattr(completion, "choices", None):
            return "", ""
        message = completion.choices[0].message
        return message.content or "", getattr(message, "reasoning_content", "") or ""

    async def _build_tool_list(self) -> tuple[list[Tool], dict[str, tuple[Any, str]]]:
        tools: list[Tool] = []
        targets: dict[str, tuple[Any, str]] = {}

        for server_name, client in self._mcp_clients.items():
            try:
                if not client.is_connected:
                    await client.connect()

                server_tools = await client.list_tools()
                for mcp_tool in server_tools:
                    tool_name = f"{server_name}__{mcp_tool.name}"
                    tools.append(
                        Tool(
                            name=tool_name,
                            description=mcp_tool.description
                            or f"Tool {mcp_tool.name} from MCP server {server_name}.",
                            parameters=mcp_tool.input_schema or {"type": "object"},
                        )
                    )
                    targets[tool_name] = (client, mcp_tool.name)
            except Exception:
                continue

        return tools, targets

    async def _execute_mcp_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        tool_targets: dict[str, tuple[Any, str]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for call_data in tool_calls:
            tc = ToolCall.from_dict(call_data)
            target = tool_targets.get(tc.name)
            if not target:
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: Unknown MCP tool '{tc.name}'",
                    }
                )
                continue

            client, mcp_tool_name = target
            try:
                mcp_result = await client.call_tool(mcp_tool_name, tc.arguments)
                text_parts = []
                for item in mcp_result.content:
                    if hasattr(item, "text"):
                        text_parts.append(item.text)
                    elif isinstance(item, str):
                        text_parts.append(item)
                    else:
                        text_parts.append(str(item))
                content = "\n".join(part for part in text_parts if part)
                if not content:
                    content = "[empty tool result]"
            except Exception as e:
                content = f"Error: {e}"

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )

        return results
