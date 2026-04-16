"""Reusable TUI widgets and modal screens for llmapp."""

from __future__ import annotations

from typing import Any
from shlex import split as shlex_split

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.screen import ModalScreen, Screen
from llmlib.models import Message, StreamEvent
from textual.widgets import Static, Input, Label, RadioButton, RadioSet, Button, TextArea

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
        if self.role == "user":
            prefix = "You"
        elif self.role == "system":
            prefix = "System"
        else:
            prefix = "Assistant"
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
        self.refresh()

    def append_thinking(self, text: str) -> None:
        self._thinking += text
        self._update_display()
        self.refresh()



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
            for name, cfg in mcp.items():
                target = cfg.url if cfg.server_type == "remote" else " ".join(cfg.command)
                yield Static(f"  - {name} ({cfg.server_type}): {target}")


class AddMCPServerModal(ModalScreen[dict[str, Any] | None]):
    """Modal form for adding an MCP server."""

    DEFAULT_CSS = """
    AddMCPServerModal {
        align: center middle;
    }

    #modal-container {
        width: 60;
        height: auto;
        max-height: 80vh;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #modal-title {
        text-align: center;
        margin-bottom: 1;
        color: $primary;
        text-style: bold;
    }

    #form-scroll {
        height: auto;
        max-height: 20;
        padding: 0 1;
    }

    .field-label {
        margin-top: 1;
        text-style: bold;
    }

    #remote-fields, #local-fields {
        display: none;
        height: auto;
    }

    #remote-fields.visible, #local-fields.visible {
        display: block;
    }

    #button-row {
        margin-top: 1;
        align: right middle;
        height: auto;
    }

    #btn-cancel {
        margin-right: 1;
    }

    TextArea {
        height: 5;
    }
    """

    BINDINGS = [
        ("escape", "dismiss(None)", "Cancel"),
        ("ctrl+q", "dismiss(None)", "Cancel"),
    ]

    def on_mount(self) -> None:
        self.query_one("#input-name").focus()

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label("Add MCP Server", id="modal-title")

            with VerticalScroll(id="form-scroll"):
                yield Label("Name:", classes="field-label")
                yield Input(placeholder="my-server", id="input-name")

                yield Label("Type:", classes="field-label")
                with RadioSet(id="type-radio"):
                    yield RadioButton("Remote (HTTP/SSE)", value=True, id="radio-remote")
                    yield RadioButton("Local (stdio/command)", id="radio-local")

                with Vertical(id="remote-fields", classes="visible"):
                    yield Label("URL:", classes="field-label")
                    yield Input(placeholder="http://localhost:8080/sse", id="input-url")
                    yield Label("Headers (key: value, one per line):", classes="field-label")
                    yield TextArea(id="input-headers")

                with Vertical(id="local-fields"):
                    yield Label("Command:", classes="field-label")
                    yield Input(
                        placeholder="npx -y @modelcontextprotocol/server-brave-search",
                        id="input-command"
                    )

            with Horizontal(id="button-row"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Add Server", variant="primary", id="btn-add")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        is_remote = event.index == 0
        self.query_one("#remote-fields").set_class(is_remote, "visible")
        self.query_one("#local-fields").set_class(not is_remote, "visible")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-add":
            result = self._build_result()
            if result:
                self.dismiss(result)

    def _build_result(self) -> dict[str, Any] | None:
        name = self.query_one("#input-name", Input).value.strip()
        if not name:
            self.app.notify("Name is required.", severity="error")
            return None

        radio_set = self.query_one("#type-radio", RadioSet)
        is_remote = radio_set.pressed_index == 0

        if is_remote:
            return self._build_remote_result(name)
        return self._build_local_result(name)

    def _build_remote_result(self, name: str) -> dict[str, Any] | None:
        url = self.query_one("#input-url", Input).value.strip()
        if not url:
            self.app.notify("URL is required for remote servers.", severity="error")
            return None
        headers = self._parse_headers()
        return {"name": name, "type": "remote", "url": url, "headers": headers}

    def _build_local_result(self, name: str) -> dict[str, Any] | None:
        raw_command = self.query_one("#input-command", Input).value.strip()
        if not raw_command:
            self.app.notify("Command is required for local servers.", severity="error")
            return None
        try:
            command = shlex_split(raw_command)
        except ValueError as error:
            self.app.notify(f"Invalid command: {error}", severity="error")
            return None
        return {"name": name, "type": "local", "command": command}

    def _parse_headers(self) -> dict[str, str]:
        raw = self.query_one("#input-headers", TextArea).text
        headers: dict[str, str] = {}
        for line in raw.splitlines():
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip()] = value.strip()
        return headers
