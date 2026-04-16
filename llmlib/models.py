"""Unified domain models for llmlib."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Callable


class MCPError(Exception):
    """Base exception for MCP operations."""
    def __init__(self, message: str, details: Any = None):
        super().__init__(message)
        self.details = details


class MCPConnectionError(MCPError):
    """Raised when MCP connection fails."""
    pass


class MCPToolError(MCPError):
    """Raised when MCP tool call fails."""
    pass


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    server_type: str  # "remote" | "local"
    url: str = ""
    command: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Any) -> "MCPServerConfig":
        """Build from config dict. Accepts legacy str (url-only) values."""
        if isinstance(data, str):
            return cls(name=name, server_type="remote", url=data)
        return cls(
            name=name,
            server_type=data.get("type", "remote"),
            url=data.get("url", ""),
            command=data.get("command", []),
            headers=data.get("headers", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to config-file dict."""
        result: dict[str, Any] = {"type": self.server_type}
        if self.server_type == "remote":
            result["url"] = self.url
            if self.headers:
                result["headers"] = self.headers
        else:
            result["command"] = self.command
        return result


@dataclass(frozen=True)
class LLMSettings:
    """Configuration for LLM clients."""

    api_url: str
    api_key: str
    model: str


@dataclass
class Message:
    """Unified application-level chat message."""

    role: str
    content: str = ""
    thinking: str = ""  # Also known as reasoning_content
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Message":
        """Parse from application-level dictionary."""
        legacy_thinking = raw.get("reasoning_content", "")
        return cls(
            role=str(raw.get("role", "assistant")),
            content=str(raw.get("content", "")),
            thinking=str(raw.get("thinking", legacy_thinking)),
            tool_calls=list(raw.get("tool_calls", []) or []),
            tool_call_id=raw.get("tool_call_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible dictionary."""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.thinking:
            result["reasoning_content"] = self.thinking
        if self.tool_calls:
            result["tool_calls"] = ToolCall.wrap_raw_list(self.tool_calls)
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


    @property
    def reasoning_content(self) -> str:
        """Alias for thinking."""
        return self.thinking

    @reasoning_content.setter
    def reasoning_content(self, value: str) -> None:
        self.thinking = value

    @property
    def effective_content(self) -> str:
        """Returns the primary content (either content or thinking)."""
        return self.content or self.thinking

    def is_empty_assistant(self) -> bool:
        """Check if this is an empty assistant message (common in tool loops)."""
        return (
            self.role == "assistant"
            and not self.content
            and not self.tool_calls
            and not self.thinking
        )

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, content={self.content[:20]!r}, thinking={bool(self.thinking)})"


class Tool:
    """Unified representation of a tool/function."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.name = name
        self.description = description
        # Legacy support for input_schema keyword
        self.parameters = parameters or kwargs.get("input_schema") or {}

    @property
    def input_schema(self) -> dict[str, Any]:
        """Alias for MCP compatibility."""
        return self.parameters

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy/protocol dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }

    @classmethod
    def from_mcp_raw(cls, server_name: str, mcp_tool_raw: Any) -> "Tool":
        """Map from a raw MCP tool structure to unified Tool."""
        return cls(
            name=mcp_tool_raw.name,
            description=mcp_tool_raw.description or f"Tool from {server_name}",
            parameters=getattr(mcp_tool_raw, "input_schema", None) 
                    or getattr(mcp_tool_raw, "inputSchema", None) 
                    or {"type": "object"}
        )

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI API function definition."""
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
    def from_dict(cls, data: Any) -> "ToolCall":
        """Parse from raw OpenAI SDK object or dictionary."""
        if isinstance(data, dict):
            id_val = data.get("id", "")
            func = data.get("function", {})
            name = func.get("name", "")
            args_raw = func.get("arguments", "{}")
        else:
            id_val = getattr(data, "id", "")
            func = getattr(data, "function", None)
            name = getattr(func, "name", "") if func else ""
            args_raw = getattr(func, "arguments", "{}") if func else "{}"

        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                arguments = {"_raw_arguments": args_raw}
        elif isinstance(args_raw, dict):
            arguments = args_raw
        else:
            arguments = {}

        return cls(id=id_val, name=name, arguments=arguments)

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible dictionary."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments)
                if isinstance(self.arguments, dict)
                else self.arguments,
            },
        }

    @staticmethod
    def wrap_raw_list(raw_calls: list[Any]) -> list[dict[str, Any]]:
        """Utility for normalizing lists of tool calls."""
        if not raw_calls:
            return []
        
        results = []
        for tc in raw_calls:
            if hasattr(tc, "model_dump"):
                results.append(tc.model_dump())
            elif hasattr(tc, "to_dict"):
                results.append(tc.to_dict())
            elif isinstance(tc, dict):
                results.append(tc)
            else:
                func = getattr(tc, "function", None)
                results.append({
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(func, "name", None),
                        "arguments": getattr(func, "arguments", "{}"),
                    }
                })
        return results

    def __repr__(self) -> str:
        return f"ToolCall(id={self.id!r}, name={self.name!r})"


class ToolResult:
    """Unified result from tool execution."""

    def __init__(
        self,
        tool_call_id: str = "",
        content: str | list[Any] = "",
        is_error: bool = False,
    ):
        self.tool_call_id = tool_call_id
        self.content = content or []
        self.is_error = is_error

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy/protocol dictionary."""
        # Convert content to MCP-style JSON parts
        legacy_content = []
        if isinstance(self.content, list):
            for part in self.content:
                if hasattr(part, "text"):
                    text = getattr(part, "text", "")
                    legacy_content.append({"type": "text", "text": text})
                elif hasattr(part, "data"):
                    data = getattr(part, "data", {})
                    legacy_content.append({"type": "resource", "data": data})
                elif isinstance(part, str):
                    legacy_content.append({"type": "text", "text": part})
                else:
                    legacy_content.append({"type": "text", "text": str(part)})
        else:
            legacy_content.append({"type": "text", "text": str(self.content)})

        return {
            "content": legacy_content,
            "isError": self.is_error,
        }

    def to_message(self) -> dict[str, Any]:
        """Convert to OpenAI tool message."""
        # Multi-modal or structured content list support
        if isinstance(self.content, list):
            text_parts = []
            for item in self.content:
                if hasattr(item, "text"):
                    text_parts.append(item.text)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
                else:
                    text_parts.append(str(item))
            content_str = "\n".join(part for part in text_parts if part)
        else:
            content_str = str(self.content)

        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": content_str or "[empty tool result]",
        }

    def __repr__(self) -> str:
        return f"ToolResult(id={self.tool_call_id!r}, is_error={self.is_error})"


class Choice:
    """Unified choice from LLM completion."""

    def __init__(
        self,
        message: Message | None = None,
        index: int = 0,
        finish_reason: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ):
        self.message = message or Message(content="", role="assistant")
        self.index = index
        self.finish_reason = finish_reason
        # tool_calls is sometimes in the message, sometimes in the choice
        self.tool_calls = tool_calls or self.message.tool_calls or []

    @property
    def content(self) -> str:
        """Convenience access to message content."""
        return self.message.content

    @classmethod
    def from_raw(cls, data: Any) -> "Choice":
        """Robust parsing from varying API response structures."""
        if not data:
            return cls()

        msg_raw = getattr(data, "message", None) or (data.get("message") if isinstance(data, dict) else None)
        delta_raw = getattr(data, "delta", None) or (data.get("delta") if isinstance(data, dict) else None)
        leaf = msg_raw or delta_raw or data
        
        def _get(obj, key, default=""):
            val = getattr(obj, key, None)
            if val is None and isinstance(obj, dict):
                val = obj.get(key)
            return val if val is not None else default

        content = _get(leaf, "content") or _get(data, "content")
        thinking = _get(leaf, "thinking") or _get(leaf, "reasoning_content") or _get(data, "thinking") or _get(data, "reasoning_content")
        role = _get(leaf, "role") or _get(data, "role", "assistant")
        
        tool_calls_raw = _get(leaf, "tool_calls", None)
        if tool_calls_raw is None or (isinstance(tool_calls_raw, list) and not tool_calls_raw):
            tool_calls_raw = _get(data, "tool_calls", [])
        
        index = _get(data, "index", 0)
        finish_reason = _get(data, "finish_reason", None)

        return cls(
            message=Message(content=content, role=role, thinking=thinking),
            index=index,
            finish_reason=finish_reason,
            tool_calls=tool_calls_raw
        )

    def __repr__(self) -> str:
        return f"Choice(index={self.index}, message={self.message!r})"


class ChatCompletion:
    """Unified chat completion response."""

    def __init__(
        self,
        choices: list[Choice],
        model: str | None = None,
    ):
        self.choices = choices
        self.model = model

    @classmethod
    def from_raw(cls, data: Any) -> "ChatCompletion":
        """Map from raw OpenAI response or chunk."""
        if not data:
            return cls(choices=[])

        raw_choices = getattr(data, "choices", None) or (data.get("choices", []) if isinstance(data, dict) else [])
        choices = [Choice.from_raw(c) for c in raw_choices]
        model = getattr(data, "model", "") or (data.get("model", "") if isinstance(data, dict) else "")
        
        return cls(choices=choices, model=model)

    @property
    def content(self) -> str | None:
        """Content of the first choice."""
        if not self.choices:
            return None
        return self.choices[0].message.content

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Aggregated tool calls from all choices."""
        results = []
        for choice in self.choices:
            if choice.tool_calls:
                # Deduplicate or just collect
                results.extend(choice.tool_calls)
        return results

    @property
    def usage(self) -> dict[str, int]:
        """Compatibility property for token usage."""
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def __repr__(self) -> str:
        return f"ChatCompletion(model={self.model!r}, choices={len(self.choices)})"


@dataclass(frozen=True)
class StreamEvent:
    """Typed runtime events for the UI."""

    kind: Literal["content", "thinking", "error", "done"]
    text: str = ""

    @classmethod
    def content(cls, text: str) -> "StreamEvent":
        return cls(kind="content", text=text)

    @classmethod
    def thinking(cls, text: str) -> "StreamEvent":
        return cls(kind="thinking", text=text)

    @classmethod
    def error(cls, text: str) -> "StreamEvent":
        return cls(kind="error", text=text)

    @classmethod
    def done(cls) -> "StreamEvent":
        return cls(kind="done", text="")
