"""Command parser for llmapp."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Command:
    """Represents a parsed command."""

    name: str
    args: list[str]
    raw: str


class CommandHandler:
    """Parses and dispatches commands."""

    COMMANDS = {
        "config": "Show/edit configuration",
        "mcp": "MCP server management (add, list, remove)",
        "session": "Session management (list, load, delete)",
        "new": "Start new conversation",
        "quit": "Exit application",
        "q": "Exit application (shortcut)",
        "help": "Show available commands",
    }

    def __init__(
        self,
        config_handler: Callable[[], None] | None = None,
        mcp_add_handler: Callable[[str, str], str] | None = None,
        mcp_list_handler: Callable[[], list] | None = None,
        mcp_remove_handler: Callable[[str], str] | None = None,
        session_list_handler: Callable[[], list] | None = None,
        session_load_handler: Callable[[str], str] | None = None,
        session_delete_handler: Callable[[str], str] | None = None,
        new_handler: Callable[[], None] | None = None,
        quit_handler: Callable[[], None] | None = None,
    ) -> None:
        self.config_handler = config_handler
        self.mcp_add_handler = mcp_add_handler
        self.mcp_list_handler = mcp_list_handler
        self.mcp_remove_handler = mcp_remove_handler
        self.session_list_handler = session_list_handler
        self.session_load_handler = session_load_handler
        self.session_delete_handler = session_delete_handler
        self.new_handler = new_handler
        self.quit_handler = quit_handler

    def parse(self, text: str) -> Command | None:
        text = text.strip()
        if not text.startswith("/"):
            return None
        parts = text.split()
        name = parts[0][1:].lower()
        args = parts[1:]
        return Command(name=name, args=args, raw=text)

    def execute(self, command: Command) -> str:
        if command.name == "help":
            return self._help()
        if command.name == "config":
            return self._config()
        if command.name == "mcp":
            return self._mcp(command.args)
        if command.name == "session":
            return self._session(command.args)
        if command.name == "new":
            return self._new()
        if command.name in ("quit", "q"):
            return self._quit()
        return f"Unknown command: /{command.name}"

    def _help(self) -> str:
        lines = ["Available commands:"]
        for name, desc in self.COMMANDS.items():
            lines.append(f"  /{name} - {desc}")
        return "\n".join(lines)

    def _config(self) -> str:
        if self.config_handler:
            self.config_handler()
        return ""

    def _mcp(self, args: list[str]) -> str:
        if not args:
            return "Usage: /mcp add <name> <url> | list | remove <name>"
        subcmd = args[0].lower()
        if subcmd == "add":
            if len(args) < 3:
                return "Usage: /mcp add <name> <url>"
            name, url = args[1], args[2]
            if self.mcp_add_handler:
                return self.mcp_add_handler(name, url)
            return f"MCP server '{name}' would be added with URL: {url}"
        if subcmd == "list":
            if self.mcp_list_handler:
                servers = self.mcp_list_handler()
                if not servers:
                    return "No MCP servers configured."
                lines = ["Configured MCP servers:"]
                for name, url in servers.items():
                    lines.append(f"  {name}: {url}")
                return "\n".join(lines)
            return "No MCP servers configured."
        if subcmd == "remove":
            if len(args) < 2:
                return "Usage: /mcp remove <name>"
            name = args[1]
            if self.mcp_remove_handler:
                return self.mcp_remove_handler(name)
            return f"MCP server '{name}' would be removed."
        return f"Unknown MCP subcommand: {subcmd}"

    def _session(self, args: list[str]) -> str:
        if not args:
            if self.session_list_handler:
                sessions = self.session_list_handler()
                if not sessions:
                    return "No saved conversations."
                lines = ["Saved conversations:"]
                for s in sessions:
                    lines.append(f"  {s['name']} ({s['message_count']} messages)")
                return "\n".join(lines)
            return "No saved conversations."
        subcmd = args[0].lower()
        if subcmd == "load":
            if len(args) < 2:
                return "Usage: /session load <name>"
            name = args[1]
            if self.session_load_handler:
                return self.session_load_handler(name)
            return f"Conversation '{name}' would be loaded."
        if subcmd == "delete":
            if len(args) < 2:
                return "Usage: /session delete <name>"
            name = args[1]
            if self.session_delete_handler:
                return self.session_delete_handler(name)
            return f"Conversation '{name}' would be deleted."
        if subcmd == "list":
            if self.session_list_handler:
                sessions = self.session_list_handler()
                if not sessions:
                    return "No saved conversations."
                lines = ["Saved conversations:"]
                for s in sessions:
                    lines.append(f"  {s['name']} ({s['message_count']} messages)")
                return "\n".join(lines)
            return "No saved conversations."
        return f"Unknown session subcommand: {subcmd}"

    def _new(self) -> str:
        if self.new_handler:
            self.new_handler()
        return "Started new conversation."

    def _quit(self) -> str:
        if self.quit_handler:
            self.quit_handler()
        return "quit"
