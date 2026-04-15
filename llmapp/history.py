"""Conversation history management for llmapp."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ConversationManager:
    """Manages conversation storage and retrieval."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        if storage_dir is None:
            storage_dir = Path(__file__).parent / "conversations"
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(exist_ok=True)

    def _get_filename(self, name: str) -> Path:
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
        return self.storage_dir / f"{safe_name}.json"

    def save(self, name: str, messages: list[dict[str, Any]]) -> str:
        filename = self._get_filename(name)
        data = {
            "name": name,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "messages": messages,
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        return str(filename)

    def load(self, name: str) -> list[dict[str, Any]] | None:
        filename = self._get_filename(name)
        if not filename.exists():
            return None
        with open(filename, "r") as f:
            data = json.load(f)
        return data.get("messages", [])

    def list(self) -> list[dict[str, Any]]:
        conversations = []
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    conversations.append(
                        {
                            "name": data.get("name", file.stem),
                            "created": data.get("created", ""),
                            "updated": data.get("updated", ""),
                            "message_count": len(data.get("messages", [])),
                        }
                    )
            except (json.JSONDecodeError, IOError):
                continue
        conversations.sort(key=lambda x: x.get("updated", ""), reverse=True)
        return conversations

    def delete(self, name: str) -> bool:
        filename = self._get_filename(name)
        if filename.exists():
            filename.unlink()
            return True
        return False

    def exists(self, name: str) -> bool:
        return self._get_filename(name).exists()
