"""Conversation history actions for ChatApp."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmapp.app import ChatApp


class HistoryController:
    """Handles session listing, loading, deleting, and reset actions."""

    def __init__(self, app: "ChatApp") -> None:
        self._app = app

    def list(self) -> list[dict[str, Any]]:
        return self._app.history_manager.list()

    def load(self, name: str) -> str:
        messages = self._app.history_manager.load(name)
        if messages is None:
            return f"Conversation '{name}' not found."

        self._app.messages = messages
        self._app.current_session = name
        container = self._app._get_chat_container()
        container.remove_children()
        for message in messages:
            container.add_message(message["role"], message["content"])
        return f"Loaded conversation '{name}'."

    def delete(self, name: str) -> str:
        if self._app.history_manager.delete(name):
            if self._app.current_session == name:
                self._app.current_session = None
            return f"Conversation '{name}' deleted."
        return f"Conversation '{name}' not found."

    def save_current(self) -> None:
        if not self._app.messages:
            return
        session_name = self._app.current_session or "default"
        self._app.history_manager.save(session_name, self._app.messages)

    def start_new(self) -> None:
        self._app.messages = []
        self._app.current_session = None
        container = self._app._get_chat_container()
        container.remove_children()
        self._app._add_message("system", "Started new conversation.")
