"""Main TUI chat application for llmapp."""

import asyncio
from typing import Any, AsyncIterator

from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Footer, Static, Input, Label

from llmlib.llm import OpenAI
from llmlib.mcp import HTTPMCPClient

from .config import ConfigManager
from .history import ConversationManager
from .command import CommandHandler


class MessageView(Static):
    """Displays a single message in the chat."""

    def __init__(self, role: str, content: str) -> None:
        super().__init__()
        self.role = role
        self._content = content

    def compose(self) -> ComposeResult:
        prefix = "You" if self.role == "user" else "Assistant"
        yield Label(f"[bold]{prefix}:[/bold] {self._content}", markup=True)

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value

    def append_content(self, text: str) -> None:
        self._content += text
        prefix = "You" if self.role == "user" else "Assistant"
        self.update(f"[bold]{prefix}:[/bold] {self._content}")


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
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config_manager = ConfigManager()
        self.history_manager = ConversationManager()
        self.messages: list[dict[str, Any]] = []
        self.current_session: str | None = None
        self.client: OpenAI | None = None
        self.mcp_clients: dict[str, HTTPMCPClient] = {}
        self.command_handler = self._create_command_handler()
        self._quit_requested = False

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
        container.add_message("assistant", "")

        try:
            async for chunk in self._call_llm_stream(text):
                if chunk:
                    self.messages[streaming_msg_idx]["content"] += chunk
                    self._update_streaming_response(chunk)
        except Exception as e:
            self._update_streaming_response(f"\nError: {e}")

    async def _call_llm(self, prompt: str) -> str:
        if not self.client:
            raise ValueError("LLM client not initialized")

        messages = [{"role": m["role"], "content": m["content"]} for m in self.messages]
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(messages=messages)
        return response.choices[0].message.content or ""

    async def _call_llm_stream(self, prompt: str) -> AsyncIterator[str]:
        if not self.client:
            raise ValueError("LLM client not initialized")

        messages = [{"role": m["role"], "content": m["content"]} for m in self.messages]
        messages.append({"role": "user", "content": prompt})

        for chunk in self.client.chat.stream_completions.create(messages=messages):
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                yield content

    def _add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        container = self.query_one("#chat-container", ChatContainer)
        container.add_message("user", content)

    def _add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        container = self.query_one("#chat-container", ChatContainer)
        container.add_message("assistant", content)

    def _add_system_message(self, content: str) -> None:
        container = self.query_one("#chat-container", ChatContainer)
        container.add_message("system", content)

    def _remove_thinking(self) -> None:
        container = self.query_one("#chat-container", ChatContainer)
        for child in container.children:
            if isinstance(child, MessageView) and child.content == "Thinking...":
                child.remove()
                break

    def _update_streaming_response(self, new_content: str) -> None:
        container = self.query_one("#chat-container", ChatContainer)
        for child in reversed(container.children):
            if isinstance(child, MessageView) and child.role == "assistant":
                child.append_content(new_content)
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
        if self.messages and self.current_session:
            self.history_manager.save(self.current_session, self.messages)
        elif self.messages:
            self.history_manager.save("default", self.messages)
        self._quit_requested = True
