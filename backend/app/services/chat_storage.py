import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.config import settings
from ..core.exceptions import NotFoundException, StorageException
from ..domain.schemas import ChatHistorySchema, ChatSchema, MessageSchema

logger = logging.getLogger(__name__)


class ChatStorage:
    def __init__(self):
        self._data_dir = Path(settings.data_dir)
        self._history_file = self._data_dir / settings.chat_history_file
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _load_history(self) -> dict[str, Any]:
        if not self._history_file.exists():
            return {"chats": [], "active_chat_id": None}
        try:
            with open(self._history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted history file, resetting: {e}")
            return {"chats": [], "active_chat_id": None}
        except Exception as e:
            raise StorageException(
                operation="load",
                reason=str(e),
            )

    def _save_history(self, data: dict[str, Any]) -> None:
        try:
            serializable_data = self._serialize_datetimes(data)
            with open(self._history_file, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise StorageException(
                operation="save",
                reason=str(e),
            )

    def _serialize_datetimes(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._serialize_datetimes(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_datetimes(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    def get_history(self) -> ChatHistorySchema:
        data = self._load_history()
        chats = [
            ChatSchema(
                id=chat["id"],
                title=chat["title"],
                messages=[
                    MessageSchema(
                        role=msg["role"],
                        content=msg["content"],
                        created_at=datetime.fromisoformat(msg["created_at"]),
                    )
                    for msg in chat.get("messages", [])
                ],
                created_at=datetime.fromisoformat(chat["created_at"]),
                updated_at=datetime.fromisoformat(chat["updated_at"]),
            )
            for chat in data.get("chats", [])
        ]
        return ChatHistorySchema(
            chats=chats,
            active_chat_id=data.get("active_chat_id"),
        )

    def get_chat(self, chat_id: str) -> ChatSchema:
        history = self.get_history()
        for chat in history.chats:
            if chat.id == chat_id:
                return chat
        raise NotFoundException(resource="Chat", resource_id=chat_id)

    def create_chat(self, title: str | None = None) -> ChatSchema:
        history = self.get_history()
        chat_id = str(uuid.uuid4())
        chat_title = title or f"Chat {len(history.chats) + 1}"
        now = datetime.now()

        chat = ChatSchema(
            id=chat_id,
            title=chat_title,
            messages=[],
            created_at=now,
            updated_at=now,
        )

        data = history.model_dump()
        data["chats"].append(chat.model_dump())
        data["active_chat_id"] = chat_id
        self._save_history(data)

        logger.info(f"Created chat: {chat_id}")
        return chat

    def add_message(self, chat_id: str, message: MessageSchema) -> ChatSchema:
        chat = self.get_chat(chat_id)
        chat.messages.append(message)
        chat.updated_at = datetime.now()

        data = self._load_history()
        for i, c in enumerate(data["chats"]):
            if c["id"] == chat_id:
                data["chats"][i] = chat.model_dump()
                break
        self._save_history(data)

        logger.info(f"Added message to chat: {chat_id}")
        return chat

    def update_chat(self, chat_id: str, updates: dict[str, Any]) -> ChatSchema:
        chat = self.get_chat(chat_id)
        for key, value in updates.items():
            if hasattr(chat, key):
                setattr(chat, key, value)
        chat.updated_at = datetime.now()

        data = self._load_history()
        for i, c in enumerate(data["chats"]):
            if c["id"] == chat_id:
                data["chats"][i] = chat.model_dump()
                break
        self._save_history(data)

        logger.info(f"Updated chat: {chat_id}")
        return chat

    def delete_chat(self, chat_id: str) -> None:
        data = self._load_history()
        original_len = len(data["chats"])
        data["chats"] = [c for c in data["chats"] if c["id"] != chat_id]

        if len(data["chats"]) == original_len:
            raise NotFoundException(resource="Chat", resource_id=chat_id)

        if data.get("active_chat_id") == chat_id:
            data["active_chat_id"] = data["chats"][-1]["id"] if data["chats"] else None

        self._save_history(data)
        logger.info(f"Deleted chat: {chat_id}")

    def clear_history(self) -> None:
        self._save_history({"chats": [], "active_chat_id": None})
        logger.info("Cleared chat history")

    def set_active_chat(self, chat_id: str | None) -> None:
        if chat_id:
            self.get_chat(chat_id)
        data = self._load_history()
        data["active_chat_id"] = chat_id
        self._save_history(data)
        logger.info(f"Active chat set to: {chat_id}")


chat_storage = ChatStorage()
