"""Streaming support for LLM responses."""

import json
from typing import Any, AsyncIterator

from llmlib.llm import OpenAI, Tool, ToolCall


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
        max_tokens: int = 200,
        streaming_enabled: bool = True,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield (content, reasoning_content) chunks from the LLM."""
        if not self._client:
            raise ValueError("LLM client not initialized")

        request_messages: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            tool_calls = m.get("tool_calls")

            if role == "assistant" and not content and not tool_calls:
                continue

            msg = {"role": m["role"], "content": m.get("content", "")}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if "tool_call_id" in m:
                msg["tool_call_id"] = m["tool_call_id"]
            request_messages.append(msg)

        tools, tool_targets = await self._build_tool_list()

        if tools:
            executed_calls: set[str] = set()
            for _ in range(4):
                step = await self._client.chat_async.completions.create(
                    messages=request_messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                )

                choice = step.choices[0] if step.choices else None
                if not choice:
                    break

                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": choice.message.content or "",
                }
                if choice.tool_calls:
                    assistant_message["tool_calls"] = choice.tool_calls

                request_messages.append(assistant_message)

                if not choice.tool_calls:
                    break

                call_keys = {
                    f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True)}"
                    for tc in (ToolCall.from_dict(data) for data in choice.tool_calls)
                }
                if call_keys and call_keys.issubset(executed_calls):
                    break

                tool_results = await self._execute_mcp_tool_calls(
                    choice.tool_calls, tool_targets
                )
                for tool_result in tool_results:
                    request_messages.append(tool_result)

                executed_calls.update(call_keys)

        if streaming_enabled:
            emitted = False
            async for chunk in await self._client.chat_async.completions.create(
                messages=request_messages,
                tools=tools or None,
                stream=True,
                max_tokens=max_tokens,
            ):
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                reasoning = delta.get("reasoning_content", "")
                if content or reasoning:
                    emitted = True
                yield (content, reasoning)
            if emitted:
                return

            content, reasoning = await self._final_non_stream_response(
                request_messages,
                max_tokens,
                tools or None,
            )
            yield (content, reasoning)
            return

        content, reasoning = await self._final_non_stream_response(
            request_messages,
            max_tokens,
            tools or None,
        )
        yield (content, reasoning)

    async def _final_non_stream_response(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        tools: list[Tool] | None,
    ) -> tuple[str, str]:
        completion = await self._client.chat_async.completions.create(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )
        content, reasoning = self._extract_choice_text(completion)
        if content or reasoning:
            return content, reasoning

        if tools:
            retry = await self._client.chat_async.completions.create(
                messages=messages,
                max_tokens=max_tokens,
            )
            retry_content, retry_reasoning = self._extract_choice_text(retry)
            if retry_content or retry_reasoning:
                return retry_content, retry_reasoning

        return "No pude obtener respuesta del modelo.", ""

    def _extract_choice_text(self, completion: Any) -> tuple[str, str]:
        if not completion or not getattr(completion, "choices", None):
            return "", ""
        message = completion.choices[0].message
        return message.content or "", message.reasoning_content or ""

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
