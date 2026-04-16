"""Specialized executor for MCP tool calls."""

from __future__ import annotations

import logging
from typing import Any
from llmlib.models import ToolCall, ToolResult

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 2000
TOOL_RESULT_TRUNCATED_SUFFIX = "\n...[tool result truncated]"

class ToolExecutor:
    """Handles execution of tool calls and result formatting."""

    def __init__(self, mcp_clients: dict[str, Any]):
        self._mcp_clients = mcp_clients

    async def discover_tools(self) -> tuple[list[Tool], dict[str, tuple[Any, str]], list[str]]:
        """Discover tools from all configured MCP clients."""
        tools, targets, errors = [], {}, []
        for name, client in self._mcp_clients.items():
            try:
                if not client.is_connected: await client.connect()
                server_tools = await client.list_tools()
                for t in server_tools:
                    targets[t.name] = (client, t.name)
                    tools.append(t)
            except Exception as e:
                logger.error(f"Discovery failed for '{name}': {e}")
                errors.append(f"Could not load tools from '{name}': {e}")
        return tools, targets, errors

    async def execute_batch(
        self, tool_calls: list[dict[str, Any]], targets: dict[str, tuple[Any, str]]
    ) -> list[dict[str, Any]]:
        """Execute a batch of tool calls and return message-formatted results."""
        results = []
        for data in tool_calls:
            tc = ToolCall.from_dict(data)
            target = targets.get(tc.name)
            if not target:
                results.append({"role": "tool", "tool_call_id": tc.id, "content": f"Error: Unknown tool '{tc.name}'"})
                continue
            
            client, mcp_name = target
            try:
                res = await client.call_tool(mcp_name, tc.arguments)
                res.tool_call_id = tc.id
                msg = res.to_message()
                msg["content"] = self.truncate(str(msg.get("content", "")))
                results.append(msg)
            except Exception as e:
                logger.error(f"Tool execution failed for '{tc.name}': {e}")
                results.append({"role": "tool", "tool_call_id": tc.id, "content": f"Error: {e}"})
        return results

    def truncate(self, text: str) -> str:
        """Truncate long tool results to avoid context window overflow."""
        if len(text) <= MAX_TOOL_RESULT_CHARS: return text
        return text[:MAX_TOOL_RESULT_CHARS].rstrip() + TOOL_RESULT_TRUNCATED_SUFFIX

