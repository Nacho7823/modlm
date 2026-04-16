"""Runtime type definitions for llmlib orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class LLMSettings:
    """Configuration for LLM clients."""

    api_url: str
    api_key: str
    model: str


@dataclass
class ChatMessage:
    """Application-level chat message exchanged with the runtime."""

    role: str
    content: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ChatMessage":
        legacy_thinking = raw.get("reasoning_content", "")
        return cls(
            role=str(raw.get("role", "assistant")),
            content=str(raw.get("content", "")),
            thinking=str(raw.get("thinking", legacy_thinking)),
            tool_calls=list(raw.get("tool_calls", []) or []),
            tool_call_id=raw.get("tool_call_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


StreamEventKind = Literal["content", "thinking", "error", "done"]


@dataclass(frozen=True)
class StreamEvent:
    """Typed runtime event emitted during chat streaming."""

    kind: StreamEventKind
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
