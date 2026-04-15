"""Data models for LLM client."""

from __future__ import annotations

from typing import Any


class Message:
    """Message object for chat completion."""

    def __init__(
        self,
        content: str,
        role: str = "assistant",
        reasoning_content: str | None = None,
    ):
        self.content = content
        self.role = role
        self.reasoning_content = reasoning_content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def __repr__(self) -> str:
        reason = self.reasoning_content[:20] if self.reasoning_content else ""
        return f"Message(role={self.role!r}, content={self.content!r}, reasoning={reason!r})"

    @property
    def effective_content(self) -> str:
        """Return content or reasoning_content if content is empty."""
        return self.content if self.content else self.reasoning_content or ""


class Choice:
    """Choice object from chat completion."""

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
        self.tool_calls = tool_calls or []

    def __repr__(self) -> str:
        return f"Choice(index={self.index}, message={self.message!r})"


class ChatCompletion:
    """Chat completion response object."""

    def __init__(
        self,
        choices: list[Choice],
        model: str | None = None,
        finish_reason: str | None = None,
    ):
        self.choices = choices
        self.model = model
        self.finish_reason = finish_reason

    @property
    def id(self) -> str:
        return "chatcmpl-local"

    @property
    def usage(self) -> dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @property
    def content(self) -> str | None:
        if self.choices:
            return self.choices[0].message.content
        return None

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        if self.choices:
            return self.choices[0].tool_calls
        return []

    def __repr__(self) -> str:
        return f"ChatCompletion(model={self.model!r}, choices={len(self.choices)})"
