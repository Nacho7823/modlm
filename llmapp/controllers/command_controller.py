import asyncio
import inspect
from typing import TYPE_CHECKING, Callable, Any

from llmapp.command import Command, CommandHandler

if TYPE_CHECKING:
    from llmapp.app import ChatApp


class CommandController:
    """Handles parsed command execution side effects in ChatApp."""

    def __init__(self, app: "ChatApp") -> None:
        self._app = app

    async def handle_input(self, text: str) -> bool:
        command = self._app.command_handler.parse(text)
        if not command:
            return False

        result = await self.execute(command)
        if result == "quit":
            self._app.exit()
            return True
        if result:
            self._app._add_message("system", result)
        return True

    async def execute(self, command: Command) -> str:
        handlers: dict[str, Any] = {
            "help": self._help,
            "config": self._config,
            "mcp": lambda: self._mcp(command.args),
            "session": lambda: self._session(command.args),
            "new": self._new,
            "streaming": lambda: self.streaming_command(command.args),
            "quit": self._quit,
            "q": self._quit,
        }
        handler = handlers.get(command.name)
        if handler:
            if inspect.iscoroutinefunction(handler):
                return await handler()
            # Handle lambdas or sync functions
            res = handler()
            if asyncio.iscoroutine(res):
                return await res
            return res
        return f"Unknown command: /{command.name}"

    def _help(self) -> str:
        return CommandHandler.help_text()

    def _config(self) -> str:
        self._app.runtime_controller.show_config()
        return ""

    async def _mcp(self, args: list[str]) -> str:
        if not args:
            return self._format_mcp_servers(self._app.runtime_controller.list_mcp_servers())
        subcommand = args[0].lower()
        if subcommand == "add":
            self._app.open_add_mcp_modal()
            return ""
        if subcommand == "list":
            return self._format_mcp_servers(self._app.runtime_controller.list_mcp_servers())
        if subcommand == "remove":
            if len(args) < 2:
                return "Usage: /mcp remove <name>"
            return await self._app.runtime_controller.remove_mcp_server(args[1])
        return f"Unknown MCP subcommand: {subcommand}"


    def _session(self, args: list[str]) -> str:
        if not args or args[0].lower() == "list":
            return self._list_sessions()

        subcommand = args[0].lower()
        if subcommand == "load":
            if len(args) < 2:
                return "Usage: /session load <name>"
            return self._app.history_controller.load(args[1])
        if subcommand == "delete":
            if len(args) < 2:
                return "Usage: /session delete <name>"
            return self._app.history_controller.delete(args[1])
        return f"Unknown session subcommand: {subcommand}"

    def _list_sessions(self) -> str:
        sessions = self._app.history_controller.list()
        if not sessions:
            return "No saved conversations."

        lines = ["Saved conversations:"]
        for session in sessions:
            lines.append(f"  {session['name']} ({session['message_count']} messages)")
        return "\n".join(lines)

    def _new(self) -> str:
        self._app.history_controller.start_new()
        return "Started new conversation."

    def _quit(self) -> str:
        self.request_quit()
        return "quit"

    def _format_mcp_servers(self, servers: dict) -> str:
        if not servers:
            return "No MCP servers configured."
        lines = ["Configured MCP servers:"]
        for name, cfg in servers.items():
            target = cfg.url if cfg.server_type == "remote" else " ".join(cfg.command)
            lines.append(f"  {name} ({cfg.server_type}): {target}")
        return "\n".join(lines)

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
