"""Main TUI chat application for llmapp."""

import asyncio
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Input

from llmlib.llm import OpenAI

from .config import ConfigManager
from .history import ConversationManager
from .command import CommandHandler
from .widgets import ChatContainer, ChatInput
from .streaming import StreamResponder
from .constants import THINKING_PLACEHOLDER
from .services import ConfigService, HistoryService, MCPService


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
        self.client: OpenAI | None = None
        self.client_async: OpenAI | None = None
        self.mcp_clients: dict[str, Any] = {}
        self.config_service = ConfigService(self)
        self.history_service = HistoryService(self)
        self.mcp_service = MCPService(self)
        self.command_handler = self._create_command_handler()
        self._streaming_task: asyncio.Task | None = None
        self._streaming_active = False

    def _create_command_handler(self) -> CommandHandler:
        return CommandHandler(
            config_handler=self._show_config,
            mcp_add_handler=self._add_mcp_server,
            mcp_list_handler=self._list_mcp_servers,
            mcp_remove_handler=self._remove_mcp_server,
            session_list_handler=self._list_sessions,
            session_load_handler=self._load_session,
            session_delete_handler=self._delete_session,
            new_handler=self._new_conversation,
            streaming_handler=self._streaming_command,
            quit_handler=self._request_quit,
        )

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
        self.config_service.init_clients()
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
        self._add_user_message(text)

        msg_idx = self._start_assistant_message()

        self._streaming_active = True

        self._streaming_task = asyncio.create_task(self._stream_response(msg_idx))

    def _handle_command(self, text: str) -> bool:
        command = self.command_handler.parse(text)
        if not command:
            return False

        result = self.command_handler.execute(command)
        if result == "quit":
            self.exit()
            return True
        if result:
            self._add_system_message(result)
        return True

    def _start_assistant_message(self) -> int:
        msg_idx = len(self.messages)
        self.messages.append({"role": "assistant", "content": ""})
        container = self._get_chat_container()
        container.add_message("assistant", "")
        self._update_streaming_thinking(THINKING_PLACEHOLDER)
        return msg_idx

    async def _stream_response(self, msg_idx: int) -> None:
        got_output = False
        try:
            responder = StreamResponder(
                self.client_async,
                self.mcp_clients,
            )
            streaming_enabled = bool(self.config_manager.get("streaming", True))
            async for content, thinking in responder.stream(
                self.messages,
                streaming_enabled=streaming_enabled,
            ):
                got_output = self._handle_stream_chunk(
                    msg_idx,
                    content,
                    thinking,
                    got_output,
                )
        except asyncio.CancelledError:
            self._handle_cancelled_stream(msg_idx)
            raise
        except Exception as e:
            self._update_streaming_response(f"\nError: {e}")
        finally:
            self._finalize_stream_state(msg_idx, got_output)
            self._streaming_active = False

    def _handle_stream_chunk(
        self,
        msg_idx: int,
        content: str,
        thinking: str,
        got_output: bool,
    ) -> bool:
        if content:
            got_output = True
            self.messages[msg_idx]["content"] += content
            self._update_streaming_response(content)

        if thinking:
            got_output = True
            existing = self.messages[msg_idx].get("reasoning_content", "")
            self.messages[msg_idx]["reasoning_content"] = existing + thinking
            self._update_streaming_thinking(thinking)

        return got_output

    def _handle_cancelled_stream(self, msg_idx: int) -> None:
        if self.messages[msg_idx].get("content", "") != "":
            return
        fallback = "Generation cancelled."
        self.messages[msg_idx]["content"] = fallback
        self._replace_last_assistant_view(fallback)

    def _finalize_stream_state(self, msg_idx: int, got_output: bool) -> None:
        if got_output or self.messages[msg_idx].get("content"):
            return
        fallback = "No response returned by model."
        self.messages[msg_idx]["content"] = fallback
        self._replace_last_assistant_view(fallback)

    def _replace_last_assistant_view(self, content: str) -> None:
        container = self._get_chat_container()
        view = container.last_assistant_view()
        if view:
            view.thinking = ""
            view.content = content

    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        container = self._get_chat_container()
        container.add_message(role, content)

    def _add_user_message(self, content: str) -> None:
        self._add_message("user", content)

    def _add_system_message(self, content: str) -> None:
        self._add_message("system", content)

    def _get_chat_container(self) -> ChatContainer:
        return self.query_one("#chat-container", ChatContainer)

    def _update_streaming_response(self, new_content: str) -> None:
        container = self._get_chat_container()
        view = container.last_assistant_view()
        if view:
            if view.thinking == THINKING_PLACEHOLDER:
                view.thinking = ""
            view.append_content(new_content)

    def _update_streaming_thinking(self, new_thinking: str) -> None:
        container = self._get_chat_container()
        view = container.last_assistant_view()
        if view:
            if view.thinking == THINKING_PLACEHOLDER:
                view.thinking = new_thinking
            else:
                view.append_thinking(new_thinking)

    def _show_config(self) -> None:
        self.config_service.show()

    def _list_mcp_servers(self) -> dict[str, str]:
        return self.mcp_service.list()

    def _add_mcp_server(self, name: str, url: str) -> str:
        return self.mcp_service.add(name, url)

    def _remove_mcp_server(self, name: str) -> str:
        return self.mcp_service.remove(name)

    def _list_sessions(self) -> list[dict[str, Any]]:
        return self.history_service.list()

    def _load_session(self, name: str) -> str:
        return self.history_service.load(name)

    def _delete_session(self, name: str) -> str:
        return self.history_service.delete(name)

    def _new_conversation(self) -> None:
        self.history_service.start_new()

    def _streaming_command(self, args: list[str]) -> str:
        current = bool(self.config_manager.get("streaming", True))

        if not args or args[0].lower() == "status":
            return f"Streaming is {'on' if current else 'off'}."

        action = args[0].lower()
        if action == "on":
            self._set_streaming(True)
            return "Streaming enabled."
        if action == "off":
            self._set_streaming(False)
            return "Streaming disabled."
        if action == "toggle":
            next_value = not current
            self._set_streaming(next_value)
            return f"Streaming {'enabled' if next_value else 'disabled'}."

        return "Usage: /streaming [on|off|toggle|status]"

    def _set_streaming(self, enabled: bool) -> None:
        self.config_manager.set("streaming", enabled)
        self.config_manager.save()

    def _request_quit(self) -> None:
        self.history_service.save_current()

    def action_cancel_stream(self) -> None:
        if self._streaming_task and not self._streaming_task.done():
            self._streaming_task.cancel()
            self._streaming_active = False
            self._add_system_message("Generation cancelled.")
        elif self._streaming_active:
            self._streaming_active = False
