"""Main TUI chat application for llmapp."""

import asyncio
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Input

from llmlib.runtime import ChatRuntime

from .config import ConfigManager
from .history import ConversationManager
from .command import CommandHandler
from .widgets import ChatContainer, ChatInput
from .controllers import (
    CommandController,
    HistoryController,
    RuntimeController,
    StreamController,
)


class ChatApp(App):
    """Main chat application."""

    CSS = """
    Screen {
        background: $surface;
    }
    #main-container {
        height: 100%;
    }
    #chat-container {
        height: 100%;
        padding: 1;
    }
    #input-container {
        dock: bottom;
        height: auto;
        background: $primary-darken-1;
        padding: 1;
    }
    #chat-input {
        width: 100%;
    }
    MessageView {
        width: 100%;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
        ("escape", "cancel_stream", "Cancel"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config_manager = ConfigManager()
        self.history_manager = ConversationManager()
        self.messages: list[dict[str, Any]] = []
        self.current_session: str | None = None
        self.runtime = ChatRuntime()
        self.history_controller = HistoryController(self)
        self.runtime_controller = RuntimeController(self)
        self.stream_controller = StreamController(self)
        self.command_controller = CommandController(self)
        self.command_handler = CommandHandler()
        self._streaming_task: asyncio.Task | None = None
        self._streaming_active = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            ChatContainer(id="chat-container"),
            id="main-container",
        )
        yield Container(
            ChatInput(id="chat-input"),
            id="input-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.config_manager.load()
        self.runtime_controller.configure_runtime()
        self.query_one("#chat-input", ChatInput).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        input_widget = self.query_one("#chat-input", ChatInput)
        input_widget.value = ""

        if self._handle_command(text):
            return

        await self._send_message(text)

    async def _send_message(self, text: str) -> None:
        await self.stream_controller.send_message(text)

    def _handle_command(self, text: str) -> bool:
        return self.command_controller.handle_input(text)

    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        container = self._get_chat_container()
        container.add_message(role, content)

    def _get_chat_container(self) -> ChatContainer:
        return self.query_one("#chat-container", ChatContainer)

    def action_cancel_stream(self) -> None:
        self.stream_controller.cancel_active_stream()

    async def on_unmount(self) -> None:
        self.stream_controller.cancel_active_stream(notify=False)
        task = self._streaming_task
        if task:
            try:
                await task
            except asyncio.CancelledError:
                pass
