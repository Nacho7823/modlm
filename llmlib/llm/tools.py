"""Tool support for LLM client."""

from __future__ import annotations

import json
from typing import Any, Callable

from llmlib.models import Tool, ToolCall, ToolResult


class ToolExecutor:
    """Executes tool calls using registered functions."""

    def __init__(self):
        self._functions: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        self._functions[name] = func

    def execute(self, tool_call: ToolCall) -> ToolResult:
        func = self._functions.get(tool_call.name)
        if not func:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: Unknown tool '{tool_call.name}'",
                is_error=True,
            )

        try:
            result = func(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=str(result) if result is not None else "",
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: {str(e)}",
                is_error=True,
            )

    def execute_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        return [self.execute(tc) for tc in tool_calls]
