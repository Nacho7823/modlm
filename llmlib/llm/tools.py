"""Tool support for LLM client."""

from __future__ import annotations

import json
from typing import Any, Callable


class Tool:
    """Represents a tool/function that can be called by the LLM."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


class ToolCall:
    """Represents a tool call from the LLM."""

    def __init__(
        self,
        id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
    ):
        self.id = id
        self.name = name
        self.arguments = arguments or {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        func = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            name=func.get("name", ""),
            arguments=json.loads(func.get("arguments", "{}"))
            if isinstance(func.get("arguments"), str)
            else func.get("arguments", {}),
        )

    def __repr__(self) -> str:
        return f"ToolCall(id={self.id!r}, name={self.name!r})"


class ToolResult:
    """Result from executing a tool."""

    def __init__(
        self,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
    ):
        self.tool_call_id = tool_call_id
        self.content = content
        self.is_error = is_error

    def to_message(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "role": "tool",
        }

    def __repr__(self) -> str:
        return (
            f"ToolResult(tool_call_id={self.tool_call_id!r}, content={self.content!r})"
        )


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
