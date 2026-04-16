"""Command dispatch actions for ChatApp."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmapp.app import ChatApp


class CommandController:
    """Handles parsed command execution side effects in ChatApp."""

    def __init__(self, app: "ChatApp") -> None:
        self._app = app

    def handle_input(self, text: str) -> bool:
        command = self._app.command_handler.parse(text)
        if not command:
            return False

        result = self._app.command_handler.execute(command)
        if result == "quit":
            self._app.exit()
            return True
        if result:
            self._app._add_message("system", result)
        return True

    def streaming_command(self, args: list[str]) -> str:
        current = bool(self._app.config_manager.get("streaming", True))

        if not args or args[0].lower() == "status":
            return f"Streaming is {'on' if current else 'off'}."

        action = args[0].lower()
        if action == "on":
            self.set_streaming(True)
            return "Streaming enabled."
        if action == "off":
            self.set_streaming(False)
            return "Streaming disabled."
        if action == "toggle":
            next_value = not current
            self.set_streaming(next_value)
            return f"Streaming {'enabled' if next_value else 'disabled'}."

        return "Usage: /streaming [on|off|toggle|status]"

    def set_streaming(self, enabled: bool) -> None:
        self._app.runtime_controller.set_streaming(enabled)

    def request_quit(self) -> None:
        self._app.history_controller.save_current()
