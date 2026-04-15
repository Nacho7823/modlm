"""Reusable TUI widgets for llmapp."""

from typing import Any

from textual.widgets import Static, Input
from textual.containers import VerticalScroll

from .constants import THINKING_PLACEHOLDER


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

    def last_assistant_view(self) -> MessageView | None:
        for child in reversed(self.children):
            if isinstance(child, MessageView) and child.role == "assistant":
                return child
        return None


class ChatInput(Input):
    """Input field for chat messages."""

    def __init__(
        self, placeholder: str = "Type a message or /help...", **kwargs: Any
    ) -> None:
        super().__init__(placeholder=placeholder, **kwargs)


class ConfigModal(Static):
    """Modal for displaying and editing configuration."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    def compose(self) -> Any:
        from textual.app import ComposeResult

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
