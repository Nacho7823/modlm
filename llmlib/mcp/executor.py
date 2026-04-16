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
        tools: list[Tool] = []
        targets: dict[str, tuple[Any, str]] = {}
        errors: list[str] = []

        for server_name, client in self._mcp_clients.items():
            try:
                if not client.is_connected:
                    await client.connect()

                server_tools = await client.list_tools()
                for tool in server_tools:
                    targets[tool.name] = (client, tool.name)
                    tools.append(tool)
            except Exception as e:
                logger.error(f"Discovery failed for '{server_name}': {e}")
                errors.append(f"Could not load tools from '{server_name}': {e}")

        return tools, targets, errors

    async def execute_batch(
        self,
        tool_calls: list[dict[str, Any]],
        tool_targets: dict[str, tuple[Any, str]],
    ) -> list[dict[str, Any]]:
        """Execute a batch of tool calls and return message-formatted results."""
        results: list[dict[str, Any]] = []

        for call_data in tool_calls:
            tc = ToolCall.from_dict(call_data)
            target = tool_targets.get(tc.name)
            
            if not target:
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: Unknown MCP tool '{tc.name}'",
                    }
                )
                continue

            client, mcp_tool_name = target
            try:
                mcp_result = await client.call_tool(mcp_tool_name, tc.arguments)
                mcp_result.tool_call_id = tc.id  # Ensure ID is set
                
                # Use unified serialization but maybe truncate
                res_msg = mcp_result.to_message()
                res_msg["content"] = self.truncate_result(str(res_msg.get("content", "")))
                results.append(res_msg)
            except Exception as error:
                logger.error(f"Tool execution failed for '{tc.name}': {error}")
                results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: {error}"
                })

        return results

    def truncate_result(self, content: str) -> str:
        """Truncate long tool results to avoid context window overflow."""
        if not isinstance(content, str):
            content = str(content)
            
        if len(content) <= MAX_TOOL_RESULT_CHARS:
            return content
        return content[:MAX_TOOL_RESULT_CHARS].rstrip() + TOOL_RESULT_TRUNCATED_SUFFIX
