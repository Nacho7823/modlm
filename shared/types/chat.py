from dataclasses import dataclass, field
from datetime import datetime

from .message import Message, MessageDict, ToolCallDict


class ChatDict(dict):
    id: str
    title: str
    messages: list[MessageDict]
    created_at: str
    updated_at: str


class ChatHistoryDict(dict):
    chats: list[ChatDict]
    active_chat_id: str | None


@dataclass
class Chat:
    id: str
    title: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> ChatDict:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: ChatDict) -> "Chat":
        return cls(
            id=data["id"],
            title=data["title"],
            messages=[Message.from_dict(m) for m in data["messages"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class ChatHistory:
    chats: list[Chat] = field(default_factory=list)
    active_chat_id: str | None = None

    def to_dict(self) -> ChatHistoryDict:
        return {
            "chats": [chat.to_dict() for chat in self.chats],
            "active_chat_id": self.active_chat_id,
        }

    @classmethod
    def from_dict(cls, data: ChatHistoryDict) -> "ChatHistory":
        return cls(
            chats=[Chat.from_dict(c) for c in data.get("chats", [])],
            active_chat_id=data.get("active_chat_id"),
        )
