from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict


class MessageDict(TypedDict):
    role: str
    content: str
    created_at: str
    tool_calls: list["ToolCallDict"] | None


class ToolCallDict(TypedDict):
    tool_name: str
    arguments: dict
    result: str | None


class ToolDict(TypedDict):
    name: str
    description: str
    function: str


@dataclass
class Message:
    role: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    tool_calls: list["ToolCall"] | None = None

    def to_dict(self) -> MessageDict:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "tool_calls": [tc.to_dict() for tc in self.tool_calls]
            if self.tool_calls
            else None,
        }

    @classmethod
    def from_dict(cls, data: MessageDict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            tool_calls=[ToolCall.from_dict(tc) for tc in data["tool_calls"]]
            if data.get("tool_calls")
            else None,
        )


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict
    result: str | None = None

    def to_dict(self) -> ToolCallDict:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: ToolCallDict) -> "ToolCall":
        return cls(
            tool_name=data["tool_name"],
            arguments=data["arguments"],
            result=data.get("result"),
        )


@dataclass
class Tool:
    name: str
    description: str
    function: str

    def to_dict(self) -> ToolDict:
        return {
            "name": self.name,
            "description": self.description,
            "function": self.function,
        }

    @classmethod
    def from_dict(cls, data: ToolDict) -> "Tool":
        return cls(
            name=data["name"],
            description=data["description"],
            function=data["function"],
        )
