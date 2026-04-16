"""History service for llmapp."""

from typing import Any


class HistoryService:
    """Handles conversation history and session management."""

    def __init__(self, app: "Any") -> None:
        self._app = app

    def list(self) -> list[dict[str, Any]]:
        """List all saved sessions."""
        return self._app.history_manager.list()

    def load(self, name: str) -> str:
        """Load a session by name, returning status message."""
        messages = self._app.history_manager.load(name)
        if messages is None:
            return f"Conversation '{name}' not found."
        self._app.messages = messages
        self._app.current_session = name
        container = self._app._get_chat_container()
        container.remove_children()
        for msg in messages:
            container.add_message(msg["role"], msg["content"])
        return f"Loaded conversation '{name}'."

    def delete(self, name: str) -> str:
        """Delete a session by name, returning status message."""
        if self._app.history_manager.delete(name):
            if self._app.current_session == name:
                self._app.current_session = None
            return f"Conversation '{name}' deleted."
        return f"Conversation '{name}' not found."

    def save_current(self) -> None:
        """Save current conversation to history."""
        if self._app.messages:
            session = self._app.current_session or "default"
            self._app.history_manager.save(session, self._app.messages)

    def start_new(self) -> None:
        """Start a new conversation."""
        self._app.messages = []
        self._app.current_session = None
        container = self._app._get_chat_container()
        container.remove_children()
        self._app._add_system_message("Started new conversation.")
