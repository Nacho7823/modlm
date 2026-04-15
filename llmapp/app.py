"""Main TUI chat application for llmapp."""

import asyncio
from typing import Any, AsyncIterator

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Footer, Static, Input, Label

from llmlib.llm import OpenAI
from llmlib.mcp import HTTPMCPClient

from .config import ConfigManager
from .history import ConversationManager
from .command import CommandHandler


THINKING_PLACEHOLDER = "Thinking..."


class MessageView(Static):
    """Displays a single message in the chat."""

    def __init__(self, role: str, content: str, thinking: str = ""):
        self.role = role
        self._content = content
        self._thinking = thinking
        super().__init__(markup=True)
        self._update_display()

    def _update_display(self) -> None:
        prefix = "You" if self.role == "user" else "Assistant"
        text = f"[bold]{prefix}:[/bold]\n"
        if self._thinking:
            text += f"[i][dim]{self._thinking}[/dim][/i]\n"
        if self._content:
            text += self._content
        self.update(text)

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value
        self._update_display()

    @property
    def thinking(self) -> str:
        return self._thinking

    @thinking.setter
    def thinking(self, value: str) -> None:
        self._thinking = value
        self._update_display()

    def append_content(self, text: str) -> None:
        self._content += text
        self._update_display()

    def append_thinking(self, text: str) -> None:
        self._thinking += text
        self._update_display()


class ChatContainer(VerticalScroll):
    """Container for chat messages."""

    def add_message(self, role: str, content: str) -> None:
        self.mount(MessageView(role=role, content=content))
        self.scroll_end()


class ChatInput(Input):
    """Input field for chat messages."""

    def __init__(
        self, placeholder: str = "Type a message or /help...", **kwargs: Any
    ) -> None:
        super().__init__(placeholder=placeholder, **kwargs)
        self._app: ChatApp | None = None

    def on_mount(self) -> None:
        self._app = self.app


class ConfigModal(Static):
    """Modal for displaying and editing configuration."""

    def __init__(self, config: ConfigManager) -> None:
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        yield Static("[bold]Current Configuration[/bold]", markup=True)
        yield Static(f"API URL: {self.config.get('api_url', '')}")
        yield Static(
            f"API Key: {self.config.get('api_key', '')[:10]}..."
            if self.config.get("api_key")
            else "API Key: (not set)"
        )
        yield Static(f"Model: {self.config.get('model', '')}")
        mcp = self.config.get_mcp_servers()
        if mcp:
            yield Static("MCP Servers:")
            for name, url in mcp.items():
                yield Static(f"  - {name}: {url}")


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
        self.mcp_clients: dict[str, HTTPMCPClient] = {}
        self.command_handler = self._create_command_handler()
        self._quit_requested = False
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
        self._init_client()
        self.query_one("#chat-input", ChatInput).focus()

    def _init_client(self) -> None:
        api_url, api_key, model = self.config_manager.get_api_config()
        self.client = OpenAI(base_url=api_url, api_key=api_key, model=model)
        self.client_async = OpenAI(base_url=api_url, api_key=api_key, model=model)
        self._init_mcp_clients()

    def _init_mcp_clients(self) -> None:
        for name, url in self.config_manager.get_mcp_servers().items():
            try:
                client = HTTPMCPClient(base_url=url)
                self.mcp_clients[name] = client
            except Exception as e:
                self._add_system_message(
                    f"Failed to connect to MCP server '{name}': {e}"
                )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        input_widget = self.query_one("#chat-input", ChatInput)
        input_widget.value = ""

        cmd = self.command_handler.parse(text)
        if cmd:
            result = self.command_handler.execute(cmd)
            if result == "quit":
                self._quit_requested = True
                self.exit()
            elif result:
                self._add_system_message(result)
            return

        await self._send_message(text)

    async def _send_message(self, text: str) -> None:
        self._add_user_message(text)

        streaming_msg_idx = len(self.messages)
        self.messages.append({"role": "assistant", "content": ""})
        container = self.query_one("#chat-container", ChatContainer)
        container.add_message("assistant", THINKING_PLACEHOLDER)

        self._streaming_active = True

        self._streaming_task = asyncio.create_task(
            self._stream_response(text, streaming_msg_idx)
        )

    async def _stream_response(self, prompt: str, msg_idx: int) -> None:
        try:
            async for content, thinking in self._call_llm_stream(prompt):
                if content:
                    self.messages[msg_idx]["content"] += content
                    self._update_streaming_response(content)
                if thinking:
                    self._update_streaming_thinking(thinking)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._update_streaming_response(f"\nError: {e}")
        finally:
            self._streaming_active = False

    async def _call_llm(self, prompt: str) -> str:
        if not self.client:
            raise ValueError("LLM client not initialized")

        messages = [{"role": m["role"], "content": m["content"]} for m in self.messages]
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(messages=messages)
        return response.choices[0].message.content or ""

    async def _call_llm_stream(self, prompt: str) -> AsyncIterator[tuple[str, str]]:
        if not self.client_async:
            raise ValueError("LLM client not initialized")

        messages = [{"role": m["role"], "content": m["content"]} for m in self.messages]
        messages.append({"role": "user", "content": prompt})

        async for chunk in await self.client_async.chat_async.completions.create(
            messages=messages, stream=True
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")
            yield (content, reasoning)

    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        container = self.query_one("#chat-container", ChatContainer)
        container.add_message(role, content)

    def _add_user_message(self, content: str) -> None:
        self._add_message("user", content)

    def _add_assistant_message(self, content: str) -> None:
        self._add_message("assistant", content)

    def _add_system_message(self, content: str) -> None:
        self._add_message("system", content)

    def _remove_thinking(self) -> None:
        container = self.query_one("#chat-container", ChatContainer)
        thinking_widget = next(
            (
                child
                for child in container.children
                if isinstance(child, MessageView)
                and child.role == "assistant"
                and child.content == THINKING_PLACEHOLDER
            ),
            None,
        )
        if thinking_widget:
            thinking_widget.remove()

    def _update_streaming_response(self, new_content: str) -> None:
        container = self.query_one("#chat-container", ChatContainer)
        for child in reversed(container.children):
            if isinstance(child, MessageView) and child.role == "assistant":
                child.append_content(new_content)
                break

    def _update_streaming_thinking(self, new_thinking: str) -> None:
        container = self.query_one("#chat-container", ChatContainer)
        for child in reversed(container.children):
            if isinstance(child, MessageView) and child.role == "assistant":
                child.append_thinking(new_thinking)
                break

    def _show_config(self) -> None:
        api_url, api_key, model = self.config_manager.get_api_config()
        mcp_servers = self.config_manager.get_mcp_servers()
        lines = [
            "[bold]Current Configuration[/bold]",
            f"API URL: {api_url}",
            f"API Key: {'(set)' if api_key else '(not set)'}",
            f"Model: {model}",
        ]
        if mcp_servers:
            lines.append("MCP Servers:")
            for name, url in mcp_servers.items():
                lines.append(f"  - {name}: {url}")
        self._add_system_message("\n".join(lines))

    def _add_mcp_server(self, name: str, url: str) -> str:
        self.config_manager.add_mcp_server(name, url)
        try:
            client = HTTPMCPClient(base_url=url)
            self.mcp_clients[name] = client
            return f"MCP server '{name}' added and connected."
        except Exception as e:
            return f"MCP server '{name}' added but failed to connect: {e}"

    def _list_mcp_servers(self) -> dict[str, str]:
        return self.config_manager.get_mcp_servers()

    def _remove_mcp_server(self, name: str) -> str:
        if self.config_manager.remove_mcp_server(name):
            if name in self.mcp_clients:
                del self.mcp_clients[name]
            return f"MCP server '{name}' removed."
        return f"MCP server '{name}' not found."

    def _list_sessions(self) -> list[dict[str, Any]]:
        return self.history_manager.list()

    def _load_session(self, name: str) -> str:
        messages = self.history_manager.load(name)
        if messages is None:
            return f"Conversation '{name}' not found."
        self.messages = messages
        self.current_session = name
        container = self.query_one("#chat-container", ChatContainer)
        container.remove_children()
        for msg in messages:
            container.add_message(msg["role"], msg["content"])
        return f"Loaded conversation '{name}'."

    def _delete_session(self, name: str) -> str:
        if self.history_manager.delete(name):
            if self.current_session == name:
                self.current_session = None
            return f"Conversation '{name}' deleted."
        return f"Conversation '{name}' not found."

    def _new_conversation(self) -> None:
        self.messages = []
        self.current_session = None
        container = self.query_one("#chat-container", ChatContainer)
        container.remove_children()
        self._add_system_message("Started new conversation.")

    def _request_quit(self) -> None:
        self._save_current_conversation()
        self._quit_requested = True

    def action_cancel_stream(self) -> None:
        if self._streaming_task and not self._streaming_task.done():
            self._streaming_task.cancel()
            self._streaming_active = False
            self._add_system_message("Generation cancelled.")
        elif self._streaming_active:
            self._streaming_active = False
