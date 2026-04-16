"""Command parser for llmapp."""

from dataclasses import dataclass


@dataclass
class Command:
    """Represents a parsed command."""

    name: str
    args: list[str]
    raw: str


class CommandHandler:
    """Parses chat input into slash commands."""

    COMMANDS = {
        "config": "Show/edit configuration",
        "mcp": "MCP server management (add, list, remove)",
        "session": "Session management (list, load, delete)",
        "new": "Start new conversation",
        "streaming": "Streaming mode (on|off|toggle|status)",
        "quit": "Exit application",
        "q": "Exit application (shortcut)",
        "help": "Show available commands",
    }

    def parse(self, text: str) -> Command | None:
        text = text.strip()
        if not text.startswith("/"):
            return None
        parts = text.split()
        name = parts[0][1:].lower()
        args = parts[1:]
        return Command(name=name, args=args, raw=text)

    @classmethod
    def help_text(cls) -> str:
        lines = ["Available commands:"]
        for name, desc in cls.COMMANDS.items():
            lines.append(f"  /{name} - {desc}")
        return "\n".join(lines)
